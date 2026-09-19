"""Fixed-point coefficient quantization and FPGA resource estimation.

An FIR that is perfect in double precision can be useless once its taps are
rounded to 16 bits: the stopband fills in, and for an IIR the poles move and
the filter can go unstable outright.  This module makes that visible before
any HDL is written.

Conventions follow Xilinx's: a signed ``Qm.n`` word has ``m`` integer bits
(including the sign) and ``n`` fractional bits, ``m + n = total_bits``.  The
Artix-7 numbers in :func:`estimate_fpga` are for the DSP48E1 slice, which is
what that family carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .analysis import db20, frequency_response
from .design import FilterDesign

__all__ = [
    "FixedPointFormat",
    "QuantizedFilter",
    "FpgaEstimate",
    "quantize",
    "estimate_fpga",
]

#: The DSP48E1 slice on 7-series parts is a signed 25x18 multiplier: 18 bits
#: for the coefficient port, 25 for the data port.
DSP48E1_COEFF_BITS = 18
DSP48E1_DATA_BITS = 25
DSP48E1_ACC_BITS = 48


@dataclass(frozen=True)
class FixedPointFormat:
    """A signed or unsigned fixed-point word format."""

    total_bits: int
    frac_bits: int
    signed: bool = True

    def __post_init__(self) -> None:
        if self.total_bits < 2:
            raise ValueError("total_bits must be at least 2")
        if not 0 <= self.frac_bits <= self.total_bits:
            raise ValueError("frac_bits must be between 0 and total_bits")

    @property
    def int_bits(self) -> int:
        """Integer bits, sign bit included."""
        return self.total_bits - self.frac_bits

    @property
    def scale(self) -> int:
        """Multiplier that maps a real value onto the integer grid."""
        return 1 << self.frac_bits

    @property
    def max_int(self) -> int:
        if self.signed:
            return (1 << (self.total_bits - 1)) - 1
        return (1 << self.total_bits) - 1

    @property
    def min_int(self) -> int:
        return -(1 << (self.total_bits - 1)) if self.signed else 0

    @property
    def max_value(self) -> float:
        return self.max_int / self.scale

    @property
    def min_value(self) -> float:
        return self.min_int / self.scale

    @property
    def resolution(self) -> float:
        """Value of one least-significant bit."""
        return 1.0 / self.scale

    @property
    def name(self) -> str:
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.int_bits}.{self.frac_bits}"

    def __str__(self) -> str:
        return f"{self.name} ({self.total_bits}-bit {'signed' if self.signed else 'unsigned'})"

    @classmethod
    def best_for(
        cls, values: np.ndarray, total_bits: int, signed: bool = True
    ) -> "FixedPointFormat":
        """Most precise format of ``total_bits`` that represents ``values`` without clipping.

        Precision is maximised by giving every bit not needed for the integer
        range to the fraction.
        """
        values = np.asarray(values)
        if np.iscomplexobj(values):
            # Real and imaginary parts are stored in separate words, so what
            # has to fit is the larger of the two -- not the magnitude, which
            # would waste up to half a bit on every complex filter.
            peak = float(
                max(np.max(np.abs(values.real)), np.max(np.abs(values.imag)))
            )
        else:
            peak = float(np.max(np.abs(values.astype(float))))
        if peak == 0.0 or not math.isfinite(peak):
            return cls(total_bits, total_bits - 1, signed)

        # Integer bits needed for the magnitude, plus one for the sign.
        mag_bits = max(0, math.ceil(math.log2(peak)))
        int_bits = mag_bits + (1 if signed else 0)
        # A value exactly on a power of two needs one more bit of headroom.
        if peak >= (1 << mag_bits) and mag_bits > 0:
            pass
        int_bits = max(int_bits, 1 if signed else 0)
        frac_bits = min(total_bits - int_bits, total_bits)
        frac_bits = max(frac_bits, 0)
        fmt = cls(total_bits, frac_bits, signed)

        # Rounding can push the largest tap one LSB past full scale; back the
        # fraction off by a bit if that happens.
        if frac_bits > 0 and round(peak * fmt.scale) > fmt.max_int:
            fmt = cls(total_bits, frac_bits - 1, signed)
        return fmt

    def quantize(self, values: np.ndarray) -> tuple[np.ndarray, bool]:
        """Round ``values`` onto the grid.  Returns ``(integers, clipped)``.

        Complex input is rounded component-wise and returned as a complex
        array with integral parts, which is what a fixed-point I/Q filter
        actually holds: two integer words per tap.
        """
        values = np.asarray(values)
        if np.iscomplexobj(values):
            real, clipped_r = self.quantize(values.real)
            imag, clipped_i = self.quantize(values.imag)
            return real + 1j * imag, (clipped_r or clipped_i)

        raw = np.round(values.astype(float) * self.scale)
        clipped = bool(np.any(raw > self.max_int) or np.any(raw < self.min_int))
        return np.clip(raw, self.min_int, self.max_int).astype(np.int64), clipped

    def dequantize(self, integers: np.ndarray) -> np.ndarray:
        integers = np.asarray(integers)
        if np.iscomplexobj(integers):
            return integers / self.scale
        return integers.astype(float) / self.scale


@dataclass
class QuantizedFilter:
    """A design rounded onto a fixed-point grid, plus what that cost."""

    design: FilterDesign
    fmt: FixedPointFormat
    #: Integer coefficients -- what goes into a .coe file or an HDL constant.
    int_taps: np.ndarray
    #: The same coefficients back in floating point: what the hardware sees.
    taps: np.ndarray
    #: Integer second-order sections, for an IIR.  ``None`` for FIR.
    int_sos: np.ndarray | None = None
    clipped: bool = False
    notes: list[str] = field(default_factory=list)

    # --- measured effects of quantization ------------------------------------
    #: Peak deviation from the ideal response, in dB relative to the passband.
    #: This is the error floor rounding puts under the filter, so it is
    #: negative and more negative is better: -70 dB means quantization noise
    #: sits 70 dB below the passband, and no stopband deeper than that is
    #: reachable at this word length.
    #:
    #: It is measured as a *linear* magnitude difference on purpose. Comparing
    #: dB magnitudes would be dominated by the ideal filter's stopband nulls,
    #: where the ideal response is -300 dB and any finite number looks like a
    #: catastrophic error.
    error_floor_db: float = 0.0
    #: Worst-case magnitude error inside the passband, in dB.
    passband_error_db: float = 0.0
    #: Stopband attenuation actually achieved after quantization, in dB.
    quantized_stopband_db: float | None = None
    #: Coefficient signal-to-quantization-noise ratio, in dB.
    coefficient_snr_db: float = 0.0
    #: For IIR: whether the filter is still stable once rounded.
    stable: bool = True
    #: True when the taps are symmetric, which halves the multipliers needed.
    symmetric: bool = False
    #: True when rounding destroyed the filter outright -- a section's leading
    #: numerator coefficient went to zero. The measured errors are meaningless
    #: in that case and are reported as NaN.
    degenerate: bool = False

    @property
    def usable(self) -> bool:
        """Whether this word length yields a filter worth implementing."""
        return self.stable and not self.degenerate and not self.clipped

    @property
    def is_complex(self) -> bool:
        return bool(np.iscomplexobj(self.int_taps))

    @property
    def int_taps_i(self) -> np.ndarray:
        """In-phase integer coefficients."""
        taps = np.asarray(self.int_taps)
        return (taps.real if np.iscomplexobj(taps) else taps).astype(np.int64)

    @property
    def int_taps_q(self) -> np.ndarray:
        """Quadrature integer coefficients; all zero for a real filter."""
        taps = np.asarray(self.int_taps)
        if not np.iscomplexobj(taps):
            return np.zeros(taps.size, dtype=np.int64)
        return taps.imag.astype(np.int64)

    @property
    def quantized_design(self) -> FilterDesign:
        """A :class:`FilterDesign` carrying the rounded coefficients.

        Feed this to the analysis and simulation code to see exactly what the
        hardware will do.
        """
        if self.int_sos is not None:
            sos_q = self.fmt.dequantize(self.int_sos)
            b, a = _sos_to_tf_quiet(sos_q)
            return FilterDesign(
                spec=self.design.spec, b=b, a=a, sos=sos_q, notes=list(self.notes)
            )
        return FilterDesign(
            spec=self.design.spec,
            b=self.taps,
            a=np.array([1.0]),
            notes=list(self.notes),
        )

    def summary_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Format", str(self.fmt)),
            ("LSB value", f"{self.fmt.resolution:.6g}"),
            ("Coefficient SNR", f"{self.coefficient_snr_db:.1f} dB"),
            ("Quantization error floor", f"{self.error_floor_db:.1f} dB"),
            ("Passband error", f"{self.passband_error_db:.4f} dB"),
        ]
        if self.quantized_stopband_db is not None:
            rows.append(
                ("Stopband after rounding", f"{self.quantized_stopband_db:.1f} dB")
            )
        rows.append(("Symmetric taps", "yes" if self.symmetric else "no"))
        if self.int_sos is not None:
            rows.append(("Stable after rounding", "yes" if self.stable else "NO"))
        if self.clipped:
            rows.append(("Clipping", "YES - coefficients hit full scale"))
        return rows


def _sos_to_tf_quiet(sos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy import signal

    from .design import _quiet_bad_coefficients

    # A section whose leading numerator coefficient has rounded to zero makes
    # scipy normalise by zero.  That is reported via the ``degenerate`` flag;
    # here we only need the division not to spew.
    with _quiet_bad_coefficients():
        b, a = signal.sos2tf(sos)
    return np.asarray(b), np.asarray(a)


# --------------------------------------------------------------------------
def quantize(
    fd: FilterDesign,
    total_bits: int = 16,
    frac_bits: int | None = None,
    num_points: int = 4096,
) -> QuantizedFilter:
    """Round ``fd``'s coefficients to ``total_bits`` and measure the damage.

    ``frac_bits`` defaults to the most precise split that avoids clipping.
    """
    if fd.sos is not None:
        return _quantize_iir(fd, total_bits, frac_bits, num_points)
    return _quantize_fir(fd, total_bits, frac_bits, num_points)


def _make_format(
    values: np.ndarray, total_bits: int, frac_bits: int | None
) -> FixedPointFormat:
    if frac_bits is None:
        return FixedPointFormat.best_for(values, total_bits)
    return FixedPointFormat(total_bits, frac_bits)


def _quantize_fir(
    fd: FilterDesign, total_bits: int, frac_bits: int | None, num_points: int
) -> QuantizedFilter:
    fmt = _make_format(fd.b, total_bits, frac_bits)
    int_taps, clipped = fmt.quantize(fd.b)
    taps = fmt.dequantize(int_taps)
    notes: list[str] = []

    if clipped:
        notes.append(
            "Coefficients clipped at full scale. Use more integer bits, or "
            "reduce the filter gain."
        )

    # A complex matched filter is conjugate-symmetric at best, which the
    # folded real structure cannot exploit, so only real taps count here.
    symmetric = not np.iscomplexobj(int_taps) and bool(
        np.array_equal(int_taps, int_taps[::-1])
    )
    if symmetric:
        notes.append(
            "Taps are symmetric: a folded FIR structure needs only "
            f"{math.ceil(int_taps.size / 2)} multipliers instead of {int_taps.size}."
        )

    q = QuantizedFilter(
        design=fd,
        fmt=fmt,
        int_taps=int_taps,
        taps=taps,
        clipped=clipped,
        notes=notes,
        symmetric=symmetric,
    )
    _measure_quantization(fd, q, num_points)
    return q


def _quantize_iir(
    fd: FilterDesign, total_bits: int, frac_bits: int | None, num_points: int
) -> QuantizedFilter:
    assert fd.sos is not None
    sos = fd.sos
    notes: list[str] = []

    # The a0 column is always 1.0, so it must fit; that sets a floor on the
    # integer bits regardless of how small the other coefficients are.
    fmt = _make_format(sos, total_bits, frac_bits)
    int_sos, clipped = fmt.quantize(sos)
    if clipped:
        notes.append(
            "Section coefficients clipped at full scale. Use more integer "
            "bits, or split the cascade differently."
        )

    q = QuantizedFilter(
        design=fd,
        fmt=fmt,
        int_taps=int_sos.reshape(-1),
        taps=fmt.dequantize(int_sos).reshape(-1),
        int_sos=int_sos,
        clipped=clipped,
        notes=notes,
    )
    # A zero leading numerator coefficient is not a small error -- the section
    # stops being second order and every derived quantity is nonsense.
    if np.any(int_sos[:, 0] == 0):
        q.degenerate = True
        notes.append(
            f"DEGENERATE at {total_bits} bits: at least one section's leading "
            "numerator coefficient rounded to zero. The word length is far too "
            "short for this filter."
        )

    q.stable = q.quantized_design.is_stable
    if not q.stable:
        notes.append(
            f"UNSTABLE at {total_bits} bits: rounding pushed at least one pole "
            "onto or outside the unit circle. Increase the word length or "
            "lower the order."
        )
    else:
        poles = np.abs(q.quantized_design.poles())
        if poles.size:
            notes.append(
                f"Largest pole radius after rounding: {np.max(poles):.6f} "
                "(must stay below 1)."
            )
    _measure_quantization(fd, q, num_points)
    return q


def _measure_quantization(
    fd: FilterDesign, q: QuantizedFilter, num_points: int
) -> None:
    """Fill in the measured error fields of ``q``."""
    ideal = np.asarray(fd.sos if fd.sos is not None else fd.b)
    actual = q.fmt.dequantize(
        q.int_sos if q.int_sos is not None else q.int_taps
    ).reshape(ideal.shape)

    error = actual - ideal
    signal_power = float(np.sum(np.abs(ideal) ** 2))
    noise_power = float(np.sum(np.abs(error) ** 2))
    if noise_power <= 0.0:
        q.coefficient_snr_db = float("inf")
    elif signal_power <= 0.0:
        q.coefficient_snr_db = 0.0
    else:
        q.coefficient_snr_db = 10.0 * math.log10(signal_power / noise_power)

    if q.degenerate:
        q.error_floor_db = float("nan")
        q.passband_error_db = float("nan")
        return

    # Compare the two responses on the same grid.
    fr_ideal = frequency_response(fd, num_points=num_points)
    try:
        fr_q = frequency_response(q.quantized_design, num_points=num_points)
    except Exception:  # an unstable quantized IIR may not evaluate cleanly
        q.error_floor_db = float("nan")
        q.passband_error_db = float("nan")
        return

    ref_linear = float(np.max(fr_ideal.mag_linear))
    if ref_linear <= 0.0:
        ref_linear = 1.0
    peak_error = float(np.max(np.abs(fr_q.h - fr_ideal.h)))
    q.error_floor_db = float(db20(peak_error / ref_linear))

    from .analysis import _band_mask, _mask_bands

    bands = _mask_bands(fd)
    if bands is None:
        # No mask (pulse shaping, Hilbert): the passband is the whole band.
        q.passband_error_db = float(np.max(np.abs(fr_q.mag_db - fr_ideal.mag_db)))
        return

    pass_bands, stop_bands = bands
    pass_mask = _band_mask(fr_q.freqs, pass_bands)
    stop_mask = _band_mask(fr_q.freqs, stop_bands)
    if pass_mask.any():
        q.passband_error_db = float(
            np.max(np.abs(fr_q.mag_db[pass_mask] - fr_ideal.mag_db[pass_mask]))
        )
        if stop_mask.any():
            ref = float(np.max(fr_q.mag_db[pass_mask]))
            q.quantized_stopband_db = ref - float(np.max(fr_q.mag_db[stop_mask]))


# --------------------------------------------------------------------------
# FPGA resource estimation
# --------------------------------------------------------------------------
@dataclass
class FpgaEstimate:
    """A first-order resource estimate for a 7-series (Artix-7) target.

    These are the numbers you would sanity-check a design against before
    opening Vivado -- not a substitute for synthesis, but enough to tell you
    whether a 200-tap filter at 40 MS/s is plausible on the part you have.
    """

    num_taps: int
    #: Effective multipliers after exploiting coefficient symmetry.
    effective_multipliers: int
    #: Clock cycles available per input sample.
    cycles_per_sample: int
    #: DSP48E1 slices needed once time-multiplexing is taken into account.
    dsp_slices: int
    #: Full-precision accumulator width before any rounding.
    accumulator_bits: int
    clock_hz: float
    sample_rate: float
    coeff_bits: int
    data_bits: int
    notes: list[str] = field(default_factory=list)
    feasible: bool = True

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Clock", f"{self.clock_hz / 1e6:,.6g} MHz"),
            ("Sample rate", f"{self.sample_rate / 1e6:,.6g} MS/s"),
            ("Cycles per sample", f"{self.cycles_per_sample}"),
            ("Taps", f"{self.num_taps}"),
            ("Multipliers (after folding)", f"{self.effective_multipliers}"),
            ("DSP48E1 slices", f"{self.dsp_slices}"),
            ("Accumulator width", f"{self.accumulator_bits} bits"),
            ("Coefficient width", f"{self.coeff_bits} bits"),
            ("Data width", f"{self.data_bits} bits"),
            ("Fits the timing budget", "yes" if self.feasible else "no"),
        ]


def estimate_fpga(
    q: QuantizedFilter,
    clock_hz: float = 100e6,
    sample_rate: float | None = None,
    data_bits: int = 16,
) -> FpgaEstimate:
    """Estimate DSP usage for the quantized filter on an Artix-7.

    The model is the standard one: a systolic FIR uses one multiplier per tap,
    symmetry folds that roughly in half, and any clock faster than the sample
    rate lets you reuse each multiplier that many times per sample.
    """
    fd = q.design
    fs = sample_rate if sample_rate is not None else fd.sample_rate
    notes: list[str] = []

    if fs <= 0:
        raise ValueError("sample rate must be > 0")
    if clock_hz < fs:
        notes.append(
            f"The {clock_hz / 1e6:.6g} MHz clock is slower than the "
            f"{fs / 1e6:.6g} MS/s sample rate; a single-clock implementation "
            "cannot keep up."
        )
        cycles = 1
        feasible = False
    else:
        cycles = int(clock_hz // fs)
        feasible = True

    if q.int_sos is not None:
        # An IIR cascade needs 5 multipliers per biquad (b0,b1,b2,a1,a2).
        num_taps = int(q.int_sos.shape[0]) * 5
        effective = num_taps
        notes.append(
            f"{q.int_sos.shape[0]} biquads x 5 multipliers. An IIR cannot be "
            "folded the way a symmetric FIR can, and its feedback path caps "
            "the achievable clock rate."
        )
    else:
        num_taps = int(q.int_taps.size)
        effective = math.ceil(num_taps / 2) if q.symmetric else num_taps
        if q.is_complex:
            # Complex coefficients against a complex data stream: four real
            # multiplies per tap (or three with the Karatsuba trick, at the
            # cost of extra adds).
            effective *= 4
            notes.append(
                "Complex (I/Q) coefficients: each tap costs four real "
                "multiplies against a complex input stream. Three is possible "
                "with the Karatsuba identity if DSP slices are tight."
            )
        if q.symmetric:
            notes.append(
                "Symmetric taps folded: pre-add the mirrored sample pairs and "
                "use half the multipliers."
            )

    dsp = max(1, math.ceil(effective / cycles))

    # Past a few hundred taps a direct-form FIR stops being the sensible
    # implementation. Pulse compression filters routinely run to thousands of
    # taps, and those are built as fast convolution -- FFT, multiply, inverse
    # FFT -- whose cost grows as N log N instead of N. Reporting a DSP count
    # in the thousands without saying so would be technically true and
    # practically useless.
    if num_taps > 256:
        fft_size = 1 << max(int(num_taps - 1).bit_length() + 1, 8)
        notes.append(
            f"At {num_taps} taps a direct FIR is the wrong structure. Use fast "
            f"convolution (overlap-save with a {fft_size}-point FFT): the work "
            f"drops from {num_taps} multiplies per sample to roughly "
            f"{int(math.log2(fft_size)) * 2}, and Vivado's FFT core handles "
            "the hard part."
        )

    growth = math.ceil(math.log2(max(num_taps, 2)))
    acc_bits = data_bits + q.fmt.total_bits + growth

    if q.fmt.total_bits > DSP48E1_COEFF_BITS:
        notes.append(
            f"A {q.fmt.total_bits}-bit coefficient exceeds the DSP48E1's "
            f"{DSP48E1_COEFF_BITS}-bit port; each multiply will be built from "
            "two DSP slices."
        )
        dsp *= 2
    if data_bits > DSP48E1_DATA_BITS:
        notes.append(
            f"A {data_bits}-bit data word exceeds the DSP48E1's "
            f"{DSP48E1_DATA_BITS}-bit port; each multiply will be split across "
            "slices."
        )
    if acc_bits > DSP48E1_ACC_BITS:
        notes.append(
            f"The {acc_bits}-bit full-precision accumulator exceeds the "
            f"{DSP48E1_ACC_BITS}-bit DSP48E1 accumulator; round or truncate "
            "intermediate results."
        )

    return FpgaEstimate(
        num_taps=num_taps,
        effective_multipliers=effective,
        cycles_per_sample=cycles,
        dsp_slices=dsp,
        accumulator_bits=acc_bits,
        clock_hz=clock_hz,
        sample_rate=fs,
        coeff_bits=q.fmt.total_bits,
        data_bits=data_bits,
        notes=notes,
        feasible=feasible,
    )
