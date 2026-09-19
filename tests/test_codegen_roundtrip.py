"""The generated Python must reproduce the coefficients it ships with.

This is the property that matters most for the whole tool: if the emitted
``design_taps()`` disagrees with the emitted ``TAPS``, one of them is wrong
and the user has no way to tell which.  Each case is executed as a real
subprocess, exactly as a user would run the file.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from filter_engine.codegen import python_scipy
from filter_engine.core.design import design
from filter_engine.core.spec import (
    FilterFamily,
    FirMethod,
    IirMethod,
    Response,
    FilterSpec,
)

# One case per design path that has a scipy call behind it.
CASES: dict[str, FilterSpec] = {
    "lp_remez": FilterSpec(name="lp_remez", fir_method=FirMethod.REMEZ),
    "lp_window": FilterSpec(name="lp_window", fir_method=FirMethod.WINDOW),
    "lp_kaiser": FilterSpec(
        name="lp_kaiser", fir_method=FirMethod.WINDOW, window="kaiser"
    ),
    "lp_blackman": FilterSpec(
        name="lp_blackman", fir_method=FirMethod.WINDOW, window="blackman"
    ),
    "hp_firls": FilterSpec(
        name="hp_firls", response=Response.HIGHPASS, fir_method=FirMethod.FIRLS
    ),
    "bp_remez": FilterSpec(
        name="bp_remez", response=Response.BANDPASS, fir_method=FirMethod.REMEZ
    ),
    "bs_window": FilterSpec(name="bs_window", response=Response.BANDSTOP),
    "lp_gain": FilterSpec(name="lp_gain", fir_method=FirMethod.REMEZ, gain=4.0),
    "rrc": FilterSpec(name="rrc", response=Response.RRC),
    "rc": FilterSpec(name="rc", response=Response.RC),
    "gaussian": FilterSpec(name="gaussian", response=Response.GAUSSIAN),
    "hilbert": FilterSpec(
        name="hilbert", response=Response.HILBERT, f_low=50e3, f_high=450e3
    ),
    "differentiator": FilterSpec(
        name="differentiator", response=Response.DIFFERENTIATOR, f_high=400e3
    ),
}

# Every IIR prototype, in both an auto-order and an explicit-order form, and
# for band types that double the prototype order.
for _method in IirMethod:
    for _resp in (Response.LOWPASS, Response.HIGHPASS, Response.BANDPASS, Response.BANDSTOP):
        CASES[f"iir_{_method.value}_{_resp.value}_auto"] = FilterSpec(
            name=f"iir_{_method.value}_{_resp.value}_auto",
            family=FilterFamily.IIR,
            response=_resp,
            iir_method=_method,
        )
        CASES[f"iir_{_method.value}_{_resp.value}_fixed"] = FilterSpec(
            name=f"iir_{_method.value}_{_resp.value}_fixed",
            family=FilterFamily.IIR,
            response=_resp,
            iir_method=_method,
            auto_order=False,
            order=5,
        )
CASES["iir_gain"] = FilterSpec(
    name="iir_gain",
    family=FilterFamily.IIR,
    iir_method=IirMethod.ELLIP,
    gain=0.5,
)


def _lfm(name: str, **changes) -> FilterSpec:
    return FilterSpec(
        name=name,
        response=Response.MATCHED_LFM,
        sample_rate=40e6,
        pulse_width_s=10e-6,
        chirp_bandwidth_hz=10e6,
        pri_s=1e-3,
    ).copy(**changes)


# Radar. The complex-tap path has its own failure modes -- a taper parameter
# that is derived rather than stored, quadrature coefficients that a careless
# dtype throws away -- so every weighting gets a case.
for _window in ("boxcar", "hann", "hamming", "blackman", "chebwin", "kaiser"):
    CASES[f"lfm_{_window}"] = _lfm(f"lfm_{_window}", window=_window)

for _sll in (25.0, 35.0, 50.0):
    # Derived nbar: the generated code has to emit the value the design
    # actually used, not the untouched field.
    CASES[f"lfm_taylor_{int(_sll)}"] = _lfm(
        f"lfm_taylor_{int(_sll)}", window="taylor", taylor_sll_db=_sll
    )

CASES["lfm_taylor_fixed_nbar"] = _lfm(
    "lfm_taylor_fixed_nbar",
    window="taylor",
    taylor_sll_db=50.0,
    auto_taylor_nbar=False,
    taylor_nbar=4,
)
CASES["lfm_down_chirp"] = _lfm("lfm_down_chirp", down_chirp=True)
CASES["lfm_gain"] = _lfm("lfm_gain", gain=3.0)

for _pulses in (2, 3, 4, 5):
    CASES[f"mti_{_pulses}"] = FilterSpec(
        name=f"mti_{_pulses}",
        response=Response.MTI_CANCELLER,
        sample_rate=1000.0,
        pri_s=1e-3,
        mti_pulses=_pulses,
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_generated_module_reproduces_its_coefficients(case, tmp_path):
    fd = design(CASES[case])
    source = python_scipy.generate(fd)

    path = tmp_path / f"{case}.py"
    path.write_text(source, encoding="utf-8")

    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"generated module failed to run:\n{result.stderr[-2000:]}"
    )
    assert "WARNING" not in result.stdout, (
        f"coefficients did not round-trip:\n{result.stdout}"
    )
    assert "OK:" in result.stdout, f"unexpected output:\n{result.stdout}"


@pytest.mark.parametrize("case", sorted(CASES))
def test_generated_module_is_valid_syntax(case):
    """Cheap check that covers every case even if execution is skipped."""
    fd = design(CASES[case])
    compile(python_scipy.generate(fd), f"<{case}>", "exec")


def _import_generated(fd, tmp_path, module_name: str):
    """Write the generated module, import it, and hand back the module."""
    import importlib

    path = tmp_path / f"{module_name}.py"
    path.write_text(python_scipy.generate(fd), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(tmp_path))


def test_generated_filter_matches_in_process_filtering(tmp_path):
    """The emitted class must filter identically to the design it came from."""
    from filter_engine.core.signals import apply_filter

    fd = design(FilterSpec(name="check", fir_method=FirMethod.REMEZ))
    try:
        module = _import_generated(fd, tmp_path, "check")
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4096)
        expected = apply_filter(fd, x)
        actual = module.Check().filter_block(x)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    finally:
        sys.modules.pop("check", None)


def test_generated_complex_filter_keeps_its_quadrature_half(tmp_path):
    """A dtype slip in the emitted class would silently discard Q.

    That is the worst kind of code-generation bug: the module imports, runs,
    and produces a plausible-looking but wrong result.
    """
    from filter_engine.core.signals import apply_filter

    fd = design(_lfm("cplxcheck", window="taylor"))
    assert fd.is_complex
    try:
        module = _import_generated(fd, tmp_path, "cplxcheck")
        filt = module.Cplxcheck()
        assert np.iscomplexobj(filt.taps), "generated taps lost their imaginary part"
        np.testing.assert_allclose(filt.taps, fd.b, rtol=0, atol=1e-12)

        rng = np.random.default_rng(0)
        x = rng.standard_normal(2048) + 1j * rng.standard_normal(2048)
        np.testing.assert_allclose(
            filt.filter_block(x), apply_filter(fd, x), rtol=0, atol=1e-12
        )
    finally:
        sys.modules.pop("cplxcheck", None)
