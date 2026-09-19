"""Tests for the HDL and Vivado exports.

No Verilog simulator is assumed, so these do not compile the generated RTL.
What they do check is the part that could actually be wrong: the *arithmetic*
the RTL encodes. :func:`transposed_fir_model` is a bit-exact Python model of
the datapath the Verilog and VHDL generators emit -- same transposed
structure, same round-to-nearest, same saturation -- and it is held against
the filter it is supposed to implement.

The coefficient files are checked by decoding them back to integers.
"""

from __future__ import annotations

import numpy as np
import pytest

from filter_engine.codegen import hdl
from filter_engine.core.design import design
from filter_engine.core.quantize import quantize
from filter_engine.core.spec import (
    FilterFamily,
    FirMethod,
    IirMethod,
    Response,
    FilterSpec,
)


# --------------------------------------------------------------------------
# A bit-exact model of the emitted datapath
# --------------------------------------------------------------------------
def transposed_fir_model(
    int_taps: list[int],
    x_int: list[int],
    coef_frac: int,
    data_w: int,
) -> list[int]:
    """Mirror the generated transposed-form FIR, exactly.

    Python's ``>>`` on a negative integer floors, which is what Verilog's
    ``>>>`` and VHDL's ``shift_right`` on a signed value do, so the rounding
    here matches the hardware rather than approximating it.
    """
    n = len(int_taps)
    acc = [0] * n
    round_add = (1 << (coef_frac - 1)) if coef_frac > 0 else 0
    max_v = (1 << (data_w - 1)) - 1
    min_v = -(1 << (data_w - 1))

    out: list[int] = []
    for sample in x_int:
        # Every stage updates from the *previous* cycle's accumulators, which
        # is what the non-blocking assignments in the generated always block
        # do. Computing in place would quietly collapse the pipeline.
        nxt = [0] * n
        nxt[n - 1] = sample * int_taps[n - 1]
        for k in range(n - 2, -1, -1):
            nxt[k] = acc[k + 1] + sample * int_taps[k]
        acc = nxt

        rounded = (acc[0] + round_add) >> coef_frac
        out.append(max(min_v, min(max_v, rounded)))
    return out


def test_the_model_matches_a_direct_convolution():
    """The transposed pipeline must compute the same sum as the obvious loop."""
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, auto_order=False, num_taps=9))
    q = quantize(fd, 16)
    taps = [int(v) for v in q.int_taps]
    n = len(taps)

    rng = np.random.default_rng(0)
    x = [int(v) for v in rng.integers(-2000, 2000, size=64)]

    produced = transposed_fir_model(taps, x, q.fmt.frac_bits, 16)

    # Latency is n-1 samples: output index i corresponds to input index i.
    for i in range(n - 1, len(x)):
        window = x[i - n + 1 : i + 1][::-1]
        expected_acc = sum(t * s for t, s in zip(taps, window))
        expected = (expected_acc + (1 << (q.fmt.frac_bits - 1))) >> q.fmt.frac_bits
        assert produced[i] == expected, f"mismatch at sample {i}"


def test_the_fixed_point_datapath_tracks_the_floating_point_filter():
    """Hardware output must follow the design it came from, to within an LSB."""
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, auto_order=False, num_taps=31))
    q = quantize(fd, 18)
    taps = [int(v) for v in q.int_taps]

    rng = np.random.default_rng(1)
    x = [int(v) for v in rng.integers(-8000, 8000, size=400)]

    produced = np.array(
        transposed_fir_model(taps, x, q.fmt.frac_bits, 16), dtype=float
    )
    reference = np.convolve(np.asarray(x, dtype=float), q.taps)[: len(x)]

    # Compare only where the pipeline is full.
    start = len(taps)
    error = np.abs(produced[start:] - reference[start:])
    assert np.max(error) <= 1.0, f"worst error {np.max(error)} LSB"


def test_saturation_clips_rather_than_wrapping():
    """A wrapped overflow inverts the sample; clipping merely flattens it."""
    taps = [1 << 15]  # unity in Q1.15
    data_w = 16
    # Drive well past full scale.
    out = transposed_fir_model(taps, [100_000, -100_000, 0], 15, data_w)
    assert out[0] == (1 << (data_w - 1)) - 1
    assert out[1] == -(1 << (data_w - 1))
    assert out[2] == 0


def test_rounding_is_to_nearest_not_truncating():
    """Truncation biases every sample downward, which shows up as a DC offset."""
    taps = [1 << 14]  # 0.5 in Q1.15
    out = transposed_fir_model(taps, [1, 3, 5, -1, -3], 15, 16)
    # 0.5, 1.5, 2.5, -0.5, -1.5 -> round half up
    assert out == [1, 2, 3, 0, -1]


# --------------------------------------------------------------------------
# Coefficient files
# --------------------------------------------------------------------------
def _decode_twos_complement(text: str, width: int) -> int:
    value = int(text, 2)
    return value - (1 << width) if value >= (1 << (width - 1)) else value


@pytest.mark.parametrize("bits", [12, 16, 18, 24])
def test_coe_decimal_values_are_the_integer_taps(bits):
    fd = design(FilterSpec(name="coe_check", fir_method=FirMethod.REMEZ))
    q = quantize(fd, bits)
    text = hdl.generate_coe(q, radix=10)

    body = text.split("coefdata =", 1)[1]
    values = [int(v) for v in body.replace(";", "").split(",") if v.strip()]
    assert values == [int(v) for v in q.int_taps]


@pytest.mark.parametrize("radix", [2, 16])
def test_coe_alternate_radices_decode_back_to_the_same_taps(radix):
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ))
    q = quantize(fd, 16)
    width = q.fmt.total_bits
    body = hdl.generate_coe(q, radix=radix).split("coefdata =", 1)[1]
    tokens = [t.strip() for t in body.replace(";", "").split(",") if t.strip()]

    decoded = []
    for token in tokens:
        raw = int(token, radix)
        decoded.append(raw - (1 << width) if raw >= (1 << (width - 1)) else raw)
    assert decoded == [int(v) for v in q.int_taps]


def test_mif_has_one_word_per_line_and_decodes_back():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ))
    q = quantize(fd, 16)
    lines = hdl.generate_mif(q).strip().splitlines()
    assert len(lines) == q.int_taps.size
    assert all(len(line) == 16 for line in lines)
    decoded = [_decode_twos_complement(line, 16) for line in lines]
    assert decoded == [int(v) for v in q.int_taps]


def test_coe_and_mif_refuse_an_iir_design():
    fd = design(
        FilterSpec(family=FilterFamily.IIR, iir_method=IirMethod.ELLIP)
    )
    q = quantize(fd, 16)
    for generator in (hdl.generate_coe, hdl.generate_mif):
        with pytest.raises(ValueError, match="IIR"):
            generator(q)


# --------------------------------------------------------------------------
# Emitted source, structurally
# --------------------------------------------------------------------------
def test_verilog_declares_every_tap_and_the_right_widths():
    fd = design(
        FilterSpec(name="my filter!", fir_method=FirMethod.REMEZ,
                   auto_order=False, num_taps=21)
    )
    q = quantize(fd, 16)
    code = hdl.generate_verilog(q, data_bits=16)

    assert "module my_filter #(" in code  # name sanitised into an identifier
    assert code.count("coeff[") >= 21
    assert "parameter integer NTAPS   = 21," in code
    assert "parameter integer COEF_W  = 16," in code
    assert f"parameter integer COEF_FRAC = {q.fmt.frac_bits}" in code
    assert code.count("module ") == 1 and code.count("endmodule") == 1
    assert "`default_nettype none" in code


def test_verilog_accumulator_is_wide_enough_to_never_overflow():
    """Worst case is every tap at full scale against a full-scale input."""
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, auto_order=False, num_taps=101))
    q = quantize(fd, 16)
    data_bits = 16
    code = hdl.generate_verilog(q, data_bits=data_bits)

    acc_w = int(
        [line for line in code.splitlines() if "ACC_W   =" in line][0]
        .split("=")[1].strip().rstrip(",")
    )
    worst = sum(abs(int(t)) for t in q.int_taps) * (1 << (data_bits - 1))
    assert worst < (1 << (acc_w - 1)), "accumulator can overflow"


def test_vhdl_emits_a_package_and_an_entity_for_fir():
    fd = design(FilterSpec(name="vhdl_fir", fir_method=FirMethod.REMEZ))
    q = quantize(fd, 16)
    code = hdl.generate_vhdl(q)
    assert "package vhdl_fir_pkg is" in code
    assert "entity vhdl_fir is" in code
    assert "architecture rtl of vhdl_fir is" in code
    assert code.count("constant COEFFS") == 1


def test_iir_hdl_exports_coefficients_without_pretending_to_emit_a_filter():
    """A fixed-point IIR body needs scaling decided against real signal levels."""
    fd = design(
        FilterSpec(name="iir_hdl", family=FilterFamily.IIR,
                   response=Response.HIGHPASS, iir_method=IirMethod.ELLIP)
    )
    q = quantize(fd, 18)

    verilog = hdl.generate_verilog(q)
    assert "endmodule" not in verilog
    assert "localparam" in verilog
    assert "limit cycle" in verilog  # the warning explaining why

    vhdl = hdl.generate_vhdl(q)
    assert "package iir_hdl_pkg is" in vhdl
    assert "entity" not in vhdl.split("package")[1].split("end package")[0]
