"""Synthetic test signals and the filter's response to them.

Plots of H(f) tell you what a filter does to a steady sinusoid.  They do not
tell you what it does to a *pulse* -- how much the edges ring, how far the
envelope smears, how a chirp comes out the other side.  This module generates
excitations with the parameters an SDR or radar user actually thinks in
(width, PRF, carrier offset, SNR) and runs them through a design.

Signals may be real or complex.  Complex baseband is the default whenever a
carrier offset is requested, because that is how an SDR front end presents
the world: a B2xx delivers I/Q, and a real-only view of a 5 MHz offset would
fold the image on top of the signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal as sig

from .design import FilterDesign
from .spec import SpecError

__all__ = [
    "PulseKind",
    "PulseSpec",
    "GeneratedSignal",
    "generate",
    "apply_filter",
    "spectrum",
]


class PulseKind(str, Enum):
    """Available excitation shapes."""

    IMPULSE = "impulse"
    STEP = "step"
    RECT = "rectangular"
    GAUSSIAN = "gaussian"
    SINC = "sinc"
    RAISED_COSINE = "raised_cosine"
    TONE_BURST = "tone_burst"
    CHIRP = "chirp"
    BARKER13 = "barker13"
    PRBS_BPSK = "prbs_bpsk"
    TWO_TONE = "two_tone"
    AWGN = "awgn"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


@dataclass
class PulseSpec:
    """Parameters for a synthetic excitation.

    Times are in seconds and frequencies in Hz; ``sample_rate`` converts.
    Not every field applies to every ``kind`` -- the irrelevant ones are
    simply ignored, which keeps the GUI free to show one parameter panel.
    """

    kind: PulseKind = PulseKind.RECT
    sample_rate: float = 1_000_000.0
    duration_s: float = 1e-3
    amplitude: float = 1.0

    #: Pulse width (RECT, GAUSSIAN, SINC, RAISED_COSINE, TONE_BURST, CHIRP).
    width_s: float = 5e-5
    #: Delay from the start of the record to the pulse centre.
    delay_s: float = 1e-4

    #: Carrier offset. Non-zero forces a complex-baseband result.
    carrier_hz: float = 0.0
    #: Force complex output even at zero carrier offset.
    force_complex: bool = False

    #: Pulse repetition frequency. 0 emits a single pulse.
    prf_hz: float = 0.0

    # --- chirp ---------------------------------------------------------------
    chirp_f0_hz: float = -100_000.0
    chirp_f1_hz: float = 100_000.0
    chirp_method: str = "linear"  # linear | quadratic | logarithmic | hyperbolic

    # --- two tone ------------------------------------------------------------
    tone1_hz: float = 50_000.0
    tone2_hz: float = 60_000.0

    # --- digital -------------------------------------------------------------
    symbol_rate: float = 100_000.0
    seed: int = 0

    # --- impairment ----------------------------------------------------------
    #: Additive white Gaussian noise level relative to the signal, in dB.
    #: ``None`` adds no noise.  For :attr:`PulseKind.AWGN` the noise *is* the
    #: signal and this is ignored.
    snr_db: float | None = None

    @property
    def num_samples(self) -> int:
        n = int(round(self.duration_s * self.sample_rate))
        return max(n, 8)

    @property
    def is_complex(self) -> bool:
        return self.force_complex or self.carrier_hz != 0.0

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise SpecError("Sample rate must be greater than 0 Hz.")
        if self.duration_s <= 0:
            raise SpecError("Record duration must be greater than 0 s.")
        if self.num_samples > 4_000_000:
            raise SpecError(
                f"A {self.duration_s:.6g} s record at "
                f"{self.sample_rate:,.6g} Hz is {self.num_samples:,} samples. "
                "Shorten the record or lower the sample rate."
            )
        nyq = self.sample_rate / 2.0
        if abs(self.carrier_hz) > nyq:
            raise SpecError(
                f"Carrier offset ({self.carrier_hz:,.6g} Hz) is beyond Nyquist "
                f"({nyq:,.6g} Hz)."
            )
        needs_width = self.kind in (
            PulseKind.RECT,
            PulseKind.GAUSSIAN,
            PulseKind.SINC,
            PulseKind.RAISED_COSINE,
            PulseKind.TONE_BURST,
            PulseKind.CHIRP,
        )
        if needs_width and self.width_s <= 0:
            raise SpecError("Pulse width must be greater than 0 s.")
        if needs_width and self.width_s * self.sample_rate < 2:
            raise SpecError(
                f"A {self.width_s:.6g} s pulse is only "
                f"{self.width_s * self.sample_rate:.2f} samples wide. Widen the "
                "pulse or raise the sample rate."
            )
        if self.prf_hz < 0:
            raise SpecError("PRF cannot be negative.")
        if self.prf_hz > 0 and self.prf_hz * self.width_s > 1.0:
            raise SpecError(
                "The pulse is wider than the repetition interval; lower the "
                "PRF or narrow the pulse."
            )
        if self.kind is PulseKind.PRBS_BPSK:
            if self.symbol_rate <= 0:
                raise SpecError("Symbol rate must be greater than 0 Hz.")
            if self.sample_rate / self.symbol_rate < 2:
                raise SpecError(
                    "Sample rate must be at least twice the symbol rate."
                )
        if self.kind is PulseKind.TWO_TONE:
            for name, f in (("Tone 1", self.tone1_hz), ("Tone 2", self.tone2_hz)):
                if abs(f) > nyq:
                    raise SpecError(
                        f"{name} ({f:,.6g} Hz) is beyond Nyquist "
                        f"({nyq:,.6g} Hz)."
                    )


@dataclass
class GeneratedSignal:
    """A generated excitation."""

    t: np.ndarray
    x: np.ndarray
    sample_rate: float
    spec: PulseSpec
    notes: list[str] = field(default_factory=list)

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.x)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generate(ps: PulseSpec) -> GeneratedSignal:
    """Build the excitation described by ``ps``."""
    ps.validate()
    n = ps.num_samples
    fs = ps.sample_rate
    t = np.arange(n) / fs
    notes: list[str] = []

    envelope = _build_envelope(ps, t, notes)

    # Repeat the pulse at the requested PRF.  Shape-only kinds (noise, PRBS,
    # two-tone) fill the record already and are not repeated.
    if ps.prf_hz > 0 and ps.kind not in (
        PulseKind.AWGN,
        PulseKind.PRBS_BPSK,
        PulseKind.TWO_TONE,
        PulseKind.STEP,
    ):
        envelope = _repeat(envelope, ps, notes)

    x = envelope * ps.amplitude

    # Mix up to the carrier offset.  A real signal at a non-zero offset would
    # alias its own image, so this is where the result turns complex.
    if ps.carrier_hz != 0.0:
        x = x.astype(complex) * np.exp(2j * np.pi * ps.carrier_hz * t)
        notes.append(
            f"Mixed to a {ps.carrier_hz:,.6g} Hz offset; the result is complex "
            "baseband (I/Q)."
        )
    elif ps.force_complex and not np.iscomplexobj(x):
        x = x.astype(complex)

    if ps.snr_db is not None and ps.kind is not PulseKind.AWGN:
        x = _add_noise(x, ps.snr_db, ps.seed, notes)

    return GeneratedSignal(t=t, x=x, sample_rate=fs, spec=ps, notes=notes)


def _build_envelope(ps: PulseSpec, t: np.ndarray, notes: list[str]) -> np.ndarray:
    fs = ps.sample_rate
    n = t.size
    kind = ps.kind
    centre = ps.delay_s
    half = ps.width_s / 2.0

    if kind is PulseKind.IMPULSE:
        x = np.zeros(n)
        idx = int(np.clip(round(centre * fs), 0, n - 1))
        x[idx] = 1.0
        notes.append(
            f"Unit impulse at sample {idx}. The filter's output is its impulse "
            "response, so this is the time-domain equivalent of the tap plot."
        )
        return x

    if kind is PulseKind.STEP:
        idx = int(np.clip(round(centre * fs), 0, n - 1))
        x = np.zeros(n)
        x[idx:] = 1.0
        notes.append(f"Unit step at sample {idx}.")
        return x

    if kind is PulseKind.RECT:
        x = ((t >= centre - half) & (t < centre + half)).astype(float)
        notes.append(
            f"Rectangular pulse, {ps.width_s * fs:.1f} samples wide. Its "
            f"spectrum is a sinc with nulls every {1.0 / ps.width_s:,.6g} Hz."
        )
        return x

    if kind is PulseKind.GAUSSIAN:
        # Treat width as the full width at half maximum.
        sigma = ps.width_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        x = np.exp(-0.5 * ((t - centre) / sigma) ** 2)
        notes.append(
            f"Gaussian pulse, {ps.width_s * fs:.1f} samples FWHM. It has no "
            "spectral sidelobes, so any ringing you see is the filter's."
        )
        return x

    if kind is PulseKind.SINC:
        x = np.sinc((t - centre) / half)
        notes.append(
            f"Sinc pulse; its spectrum is a rectangle "
            f"{1.0 / half:,.6g} Hz wide."
        )
        return x

    if kind is PulseKind.RAISED_COSINE:
        # A single raised-cosine (Hann) shaped pulse, not the Nyquist filter.
        x = np.zeros(n)
        inside = (t >= centre - half) & (t <= centre + half)
        x[inside] = 0.5 * (1.0 + np.cos(np.pi * (t[inside] - centre) / half))
        notes.append("Raised-cosine (Hann) shaped pulse: smooth edges, low sidelobes.")
        return x

    if kind is PulseKind.TONE_BURST:
        gate = ((t >= centre - half) & (t < centre + half)).astype(float)
        # The tone itself sits at the carrier offset, applied later; inside
        # the burst the envelope is flat.
        notes.append(
            f"Tone burst gated over {ps.width_s * fs:.1f} samples. Set a "
            "carrier offset to place the tone away from DC."
        )
        return gate

    if kind is PulseKind.CHIRP:
        gate = (t >= centre - half) & (t < centre + half)
        x = np.zeros(n)
        tb = t[gate] - (centre - half)
        if tb.size:
            phase = sig.chirp(
                tb,
                f0=ps.chirp_f0_hz,
                t1=ps.width_s,
                f1=ps.chirp_f1_hz,
                method=ps.chirp_method,
                phi=-90,
            )
            x[gate] = phase
        bw = abs(ps.chirp_f1_hz - ps.chirp_f0_hz)
        notes.append(
            f"{ps.chirp_method} chirp sweeping {ps.chirp_f0_hz:,.6g} to "
            f"{ps.chirp_f1_hz:,.6g} Hz in {ps.width_s:.6g} s "
            f"(time-bandwidth product {bw * ps.width_s:.1f}). A chirp shows "
            "the filter's whole passband in one shot."
        )
        return x

    if kind is PulseKind.BARKER13:
        code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
        chip_samples = max(int(round(ps.width_s * fs / code.size)), 1)
        burst = np.repeat(code, chip_samples)
        x = np.zeros(n)
        start = int(np.clip(round((centre - half) * fs), 0, max(n - 1, 0)))
        end = min(start + burst.size, n)
        x[start:end] = burst[: end - start]
        if end - start < burst.size:
            notes.append(
                "The record is too short to hold the whole Barker sequence; it "
                "has been truncated."
            )
        notes.append(
            f"13-chip Barker code, {chip_samples} samples per chip. Its "
            "autocorrelation has 22.3 dB peak-to-sidelobe ratio."
        )
        return x

    if kind is PulseKind.PRBS_BPSK:
        sps = fs / ps.symbol_rate
        num_symbols = max(int(n / sps), 1)
        rng = np.random.default_rng(ps.seed)
        symbols = rng.choice([-1.0, 1.0], size=num_symbols)
        x = np.repeat(symbols, int(round(sps)))[:n]
        if x.size < n:
            x = np.pad(x, (0, n - x.size))
        notes.append(
            f"{num_symbols} random BPSK symbols at {sps:.3g} samples/symbol "
            f"(seed {ps.seed}). Unshaped, so its spectrum is a sinc-squared "
            "roll-off -- filter it to see pulse shaping work."
        )
        return x

    if kind is PulseKind.TWO_TONE:
        x = np.cos(2 * np.pi * ps.tone1_hz * t) + np.cos(2 * np.pi * ps.tone2_hz * t)
        x = x / 2.0
        notes.append(
            f"Two tones at {ps.tone1_hz:,.6g} and {ps.tone2_hz:,.6g} Hz "
            f"({abs(ps.tone2_hz - ps.tone1_hz):,.6g} Hz apart)."
        )
        return x

    if kind is PulseKind.AWGN:
        rng = np.random.default_rng(ps.seed)
        if ps.is_complex:
            x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2.0)
        else:
            x = rng.standard_normal(n)
        notes.append(
            f"White Gaussian noise (seed {ps.seed}). Filtering it traces out "
            "the magnitude response directly."
        )
        return x

    raise SpecError(f"unsupported pulse kind {kind!r}")


def _repeat(envelope: np.ndarray, ps: PulseSpec, notes: list[str]) -> np.ndarray:
    """Tile a single pulse at the requested PRF."""
    period = int(round(ps.sample_rate / ps.prf_hz))
    if period < 1:
        return envelope
    n = envelope.size
    out = np.zeros_like(envelope)
    count = 0
    for start in range(0, n, period):
        seg = envelope[: n - start]
        out[start : start + seg.size] += seg
        count += 1
    notes.append(
        f"Repeated at {ps.prf_hz:,.6g} Hz PRF ({count} pulses, "
        f"{ps.prf_hz * ps.width_s * 100:.2f}% duty cycle)."
    )
    return out


def _add_noise(
    x: np.ndarray, snr_db: float, seed: int, notes: list[str]
) -> np.ndarray:
    """Add AWGN at ``snr_db`` relative to the signal's mean power."""
    rng = np.random.default_rng(seed + 1)
    signal_power = float(np.mean(np.abs(x) ** 2))
    if signal_power <= 0.0:
        notes.append("Signal power is zero; no noise was added.")
        return x
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    if np.iscomplexobj(x):
        noise = math.sqrt(noise_power / 2.0) * (
            rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
        )
    else:
        noise = math.sqrt(noise_power) * rng.standard_normal(x.size)
    notes.append(f"Added white Gaussian noise at {snr_db:.1f} dB SNR.")
    return x + noise


# --------------------------------------------------------------------------
# Filtering and spectra
# --------------------------------------------------------------------------
def apply_filter(fd: FilterDesign, x: np.ndarray) -> np.ndarray:
    """Run ``x`` through ``fd``.

    Uses the second-order sections for an IIR -- the same code path the
    generated Python uses, so what you see here is what that code produces.
    Complex input is filtered directly; the coefficients are real, so this is
    equivalent to filtering I and Q separately.
    """
    x = np.asarray(x)
    if fd.sos is not None:
        return sig.sosfilt(fd.sos, x)
    return sig.lfilter(fd.b, fd.a, x)


def spectrum(
    x: np.ndarray,
    sample_rate: float,
    window: str = "hann",
    db: bool = True,
    ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of ``x`` as ``(freqs_hz, magnitude)``.

    A real input returns a one-sided spectrum from DC to Nyquist.  A complex
    input returns the full two-sided spectrum centred on DC, which is what an
    I/Q capture actually occupies.

    ``ref`` normalises the dB scale; it defaults to the peak, so the plot
    reads as dB relative to the strongest component.
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return np.zeros(0), np.zeros(0)

    win = sig.get_window(window, n, fftbins=True)
    # Coherent gain correction keeps amplitudes honest after windowing.
    win = win / np.mean(win)
    xw = x * win

    if np.iscomplexobj(x):
        mag = np.abs(np.fft.fftshift(np.fft.fft(xw))) / n
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
    else:
        mag = np.abs(np.fft.rfft(xw)) / n
        # Fold the negative-frequency energy onto the positive axis, leaving
        # DC and Nyquist alone since they have no mirror partner.
        if mag.size > 2:
            mag[1:-1] *= 2.0
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    if not db:
        return freqs, mag

    peak = ref if ref is not None else float(np.max(mag))
    if peak <= 0.0:
        peak = 1.0
    return freqs, 20.0 * np.log10(np.maximum(mag / peak, 1e-12))
