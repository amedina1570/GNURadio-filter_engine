"""Response computation and performance measurement.

Two jobs live here.  The first is turning a :class:`FilterDesign` into the
curves the GUI plots -- magnitude, phase, group delay, impulse, step.  The
second is *measuring* the result: a design is only useful if you can see
whether it actually met the mask you asked for, so :func:`measure` re-derives
the achieved ripple, attenuation and cutoffs from the realised coefficients
rather than from the request.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .design import FilterDesign, _quiet_bad_coefficients
from .spec import Response

__all__ = [
    "FrequencyResponse",
    "Measurements",
    "RadarMetrics",
    "frequency_response",
    "impulse_response",
    "step_response",
    "measure",
    "compressed_response",
    "radar_metrics",
    "mti_velocity_response",
    "ambiguity_function",
    "db20",
]

#: Magnitudes below this are clamped before taking a log, so a perfect null
#: plots as a deep notch instead of -inf.
_FLOOR = 1e-12


def db20(x: np.ndarray | float) -> np.ndarray:
    """Amplitude ratio to dB, with a floor so nulls stay plottable."""
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(x, dtype=complex)), _FLOOR))


@dataclass
class FrequencyResponse:
    """Sampled frequency response of a design."""

    #: Frequency axis in Hz, spanning 0 to Nyquist (or the requested range).
    freqs: np.ndarray
    #: Complex response H(f).
    h: np.ndarray
    sample_rate: float

    @property
    def mag_db(self) -> np.ndarray:
        return db20(self.h)

    @property
    def mag_linear(self) -> np.ndarray:
        return np.abs(self.h)

    @property
    def phase_rad(self) -> np.ndarray:
        """Wrapped phase in radians."""
        return np.angle(self.h)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.h))

    @property
    def phase_unwrapped_deg(self) -> np.ndarray:
        return np.degrees(np.unwrap(np.angle(self.h)))

    @property
    def phase_delay_samples(self) -> np.ndarray:
        """Phase delay ``-phase / omega``, in samples.

        Undefined at DC, where omega is zero; that point is returned as NaN so
        plotting simply leaves a gap.
        """
        omega = 2.0 * np.pi * self.freqs / self.sample_rate
        with np.errstate(divide="ignore", invalid="ignore"):
            pd = -np.unwrap(np.angle(self.h)) / omega
        pd[omega == 0] = np.nan
        return pd


@dataclass
class Measurements:
    """What the realised filter actually achieves.

    Every field is measured from the coefficients.  ``None`` means the
    quantity does not apply to this response type (a Hilbert transformer has
    no stopband attenuation in the usual sense, for instance).
    """

    passband_ripple_db: float | None = None
    stopband_atten_db: float | None = None
    #: Frequencies where the response first falls 3 dB / 6 dB below the
    #: passband reference, in Hz.
    cutoff_3db_hz: list[float] | None = None
    cutoff_6db_hz: list[float] | None = None
    #: Mean group delay across the passband, in samples.
    passband_group_delay: float | None = None
    #: DC gain in dB, and peak gain in dB.
    dc_gain_db: float | None = None
    peak_gain_db: float | None = None
    #: True when the realised response satisfies the requested mask.
    meets_spec: bool | None = None
    notes: list[str] | None = None

    def as_rows(self) -> list[tuple[str, str]]:
        """Label/value pairs for display in a table."""
        rows: list[tuple[str, str]] = []

        def add(label: str, value: object, fmt: str = "{:.4g}", unit: str = "") -> None:
            if value is None:
                return
            if isinstance(value, list):
                if not value:
                    return
                text = ", ".join(fmt.format(v) for v in value)
            elif isinstance(value, bool):
                text = "yes" if value else "no"
            else:
                text = fmt.format(value)
            rows.append((label, f"{text}{unit}"))

        add("Passband ripple", self.passband_ripple_db, "{:.4g}", " dB")
        add("Stopband attenuation", self.stopband_atten_db, "{:.4g}", " dB")
        add("-3 dB cutoff", self.cutoff_3db_hz, "{:,.6g}", " Hz")
        add("-6 dB cutoff", self.cutoff_6db_hz, "{:,.6g}", " Hz")
        add("Passband group delay", self.passband_group_delay, "{:.4g}", " samples")
        add("DC gain", self.dc_gain_db, "{:.4g}", " dB")
        add("Peak gain", self.peak_gain_db, "{:.4g}", " dB")
        add("Meets spec", self.meets_spec)
        return rows


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------
def frequency_response(
    fd: FilterDesign,
    num_points: int = 4096,
    f_min: float = 0.0,
    f_max: float | None = None,
    log_spacing: bool = False,
) -> FrequencyResponse:
    """Evaluate H(f) on ``num_points`` frequencies between ``f_min`` and ``f_max``.

    ``f_max`` defaults to Nyquist.  Log spacing starts at ``f_min`` or, if that
    is zero, at Nyquist/1e5 -- you cannot put DC on a log axis.
    """
    fs = fd.sample_rate
    nyq = fs / 2.0
    if f_max is None:
        f_max = nyq
    f_max = min(f_max, nyq)

    if log_spacing:
        lo = f_min if f_min > 0 else nyq / 1e5
        freqs = np.logspace(math.log10(lo), math.log10(f_max), num_points)
    else:
        freqs = np.linspace(f_min, f_max, num_points)

    with _quiet_bad_coefficients():
        if fd.sos is not None:
            _w, h = signal.sosfreqz(fd.sos, worN=freqs, fs=fs)
        else:
            _w, h = signal.freqz(fd.b, fd.a, worN=freqs, fs=fs)
    return FrequencyResponse(freqs=np.asarray(freqs), h=np.asarray(h), sample_rate=fs)


def group_delay(fd: FilterDesign, freqs: np.ndarray) -> np.ndarray:
    """Group delay in samples at ``freqs`` (Hz).

    For an IIR cascade the delay is summed section by section.  Running
    :func:`scipy.signal.group_delay` on the expanded ``(b, a)`` of a high-order
    filter gives garbage, whereas each second-order section is trivially well
    conditioned and group delays add.
    """
    fs = fd.sample_rate
    freqs = np.asarray(freqs, dtype=float)
    with _quiet_bad_coefficients(), warnings.catch_warnings():
        # A high-order filter's denominator gets very small near its band
        # edges and scipy warns about the possible singularity.  The delay it
        # returns there is still the best available estimate and the plot
        # shows it; a page of warning text on every replot is not useful.
        warnings.filterwarnings(
            "ignore", message=".*denominator is extremely small.*"
        )
        if fd.sos is not None:
            total = np.zeros_like(freqs)
            for section in fd.sos:
                _w, gd = signal.group_delay(
                    (section[:3], section[3:]), w=freqs, fs=fs
                )
                total += gd
            return total
        _w, gd = signal.group_delay((fd.b, fd.a), w=freqs, fs=fs)
    return np.asarray(gd)


def impulse_response(fd: FilterDesign, num_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Impulse response as ``(sample_index, amplitude)``.

    For FIR this is exactly the tap vector.  For IIR the response is run out
    far enough to decay, capped so a very high-Q filter cannot hang the GUI.
    """
    if fd.is_fir:
        n = fd.b.size if num_samples is None else int(num_samples)
        x = np.zeros(n)
        x[0] = 1.0
        h = signal.lfilter(fd.b, fd.a, x)
        return np.arange(n), h

    n = int(num_samples) if num_samples is not None else _iir_settling_length(fd)
    x = np.zeros(n)
    x[0] = 1.0
    h = signal.sosfilt(fd.sos, x)
    return np.arange(n), h


def step_response(fd: FilterDesign, num_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Step response as ``(sample_index, amplitude)``."""
    if fd.is_fir:
        n = fd.b.size * 2 if num_samples is None else int(num_samples)
        x = np.ones(n)
        y = signal.lfilter(fd.b, fd.a, x)
    else:
        n = int(num_samples) if num_samples is not None else _iir_settling_length(fd)
        x = np.ones(n)
        y = signal.sosfilt(fd.sos, x)
    return np.arange(n), y


def _iir_settling_length(fd: FilterDesign, decay_db: float = 120.0) -> int:
    """How many samples it takes the slowest pole to decay by ``decay_db``."""
    poles = fd.poles()
    if poles.size == 0:
        return 256
    r = float(np.max(np.abs(poles)))
    if r >= 1.0:  # unstable: show a short window rather than diverge forever
        return 512
    r = max(r, 1e-6)
    n = int(math.ceil(decay_db / (-20.0 * math.log10(r))))
    return int(np.clip(n, 64, 16_384))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def measure(fd: FilterDesign, num_points: int = 8192) -> Measurements:
    """Measure the realised response against the requested mask."""
    spec = fd.spec
    fr = frequency_response(fd, num_points=num_points)
    mag_db = fr.mag_db
    freqs = fr.freqs
    notes: list[str] = []

    m = Measurements(notes=notes)
    m.dc_gain_db = float(mag_db[0])
    m.peak_gain_db = float(np.max(mag_db))

    bands = _mask_bands(fd)
    if bands is None:
        notes.append(
            f"No pass/stop mask is defined for a {spec.response.value} response, "
            "so ripple and attenuation are not measured."
        )
        return m

    pass_bands, stop_bands = bands

    # Reference level: the mean of the passband, which is what "0 dB relative"
    # means for a filter whose gain is not 1.
    pass_mask = _band_mask(freqs, pass_bands)
    if not pass_mask.any():
        notes.append("The passband is too narrow to resolve at this frequency grid.")
        return m
    pass_db = mag_db[pass_mask]
    ref_db = float(np.max(pass_db))

    m.passband_ripple_db = float(np.max(pass_db) - np.min(pass_db))

    stop_mask = _band_mask(freqs, stop_bands)
    if stop_mask.any():
        m.stopband_atten_db = float(ref_db - np.max(mag_db[stop_mask]))

    m.cutoff_3db_hz = _crossings(freqs, mag_db, ref_db - 3.0)
    m.cutoff_6db_hz = _crossings(freqs, mag_db, ref_db - 6.0)

    gd = group_delay(fd, freqs[pass_mask])
    finite = gd[np.isfinite(gd)]
    if finite.size:
        m.passband_group_delay = float(np.mean(finite))

    if m.stopband_atten_db is not None:
        m.meets_spec = (
            m.passband_ripple_db <= spec.passband_ripple_db * 1.05
            and m.stopband_atten_db >= spec.stopband_atten_db * 0.95
        )
        if not m.meets_spec:
            notes.append(
                "The realised response misses the requested mask. Increase the "
                "order/tap count, widen the transition, or relax the tolerances."
            )
    return m


def _mask_bands(fd: FilterDesign) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """Passband and stopband intervals implied by the spec, in Hz.

    Returns ``None`` for responses that have no simple pass/stop mask.
    """
    spec = fd.spec
    nyq = spec.nyquist
    half = spec.transition_width / 2.0

    if spec.response is Response.LOWPASS:
        return [(0.0, max(spec.f_low - half, 0.0))], [(min(spec.f_low + half, nyq), nyq)]
    if spec.response is Response.HIGHPASS:
        return [(min(spec.f_low + half, nyq), nyq)], [(0.0, max(spec.f_low - half, 0.0))]
    if spec.response is Response.BANDPASS:
        return (
            [(spec.f_low + half, spec.f_high - half)],
            [(0.0, max(spec.f_low - half, 0.0)), (min(spec.f_high + half, nyq), nyq)],
        )
    if spec.response is Response.BANDSTOP:
        return (
            [(0.0, max(spec.f_low - half, 0.0)), (min(spec.f_high + half, nyq), nyq)],
            [(spec.f_low + half, spec.f_high - half)],
        )
    return None


def _band_mask(freqs: np.ndarray, bands: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in bands:
        if hi > lo:
            mask |= (freqs >= lo) & (freqs <= hi)
    return mask


def _crossings(freqs: np.ndarray, mag_db: np.ndarray, level_db: float) -> list[float]:
    """Frequencies where ``mag_db`` crosses ``level_db``, linearly interpolated."""
    above = mag_db >= level_db
    idx = np.flatnonzero(np.diff(above.astype(np.int8)) != 0)
    out: list[float] = []
    for i in idx:
        y0, y1 = mag_db[i], mag_db[i + 1]
        if y1 == y0:
            out.append(float(freqs[i]))
            continue
        t = (level_db - y0) / (y1 - y0)
        out.append(float(freqs[i] + t * (freqs[i + 1] - freqs[i])))
    return out


# --------------------------------------------------------------------------
# Radar figures of merit
#
# A pulse compression filter is not judged by passband ripple. It is judged by
# what a strong target does to its neighbours: how high the worst range
# sidelobe sits (PSLR), how much total energy leaks out of the mainlobe
# (ISLR), and how much resolution the weighting cost to get there.
# --------------------------------------------------------------------------
@dataclass
class RadarMetrics:
    """Range-sidelobe and resolution performance of a compression filter."""

    #: Peak sidelobe ratio: the worst sidelobe, dB below the compressed peak.
    #: The number that decides whether a strong target masks a weak one.
    pslr_db: float = float("nan")
    #: Integrated sidelobe ratio: total sidelobe energy over mainlobe energy,
    #: in dB. What matters against distributed clutter rather than a point
    #: target.
    islr_db: float = float("nan")
    #: -3 dB width of the compressed pulse, in seconds and in metres.
    mainlobe_3db_s: float = float("nan")
    mainlobe_3db_m: float = float("nan")
    #: Null-to-null width of the compressed mainlobe, in seconds.
    mainlobe_null_s: float = float("nan")
    #: Resolution implied by the chirp bandwidth alone, for comparison.
    range_resolution_m: float = float("nan")
    #: How much the weighting broadened the mainlobe, against no weighting.
    broadening: float = float("nan")
    #: Pulse compression ratio and the gain it buys.
    time_bandwidth_product: float = float("nan")
    processing_gain_db: float = float("nan")
    #: SNR given up by weighting the filter away from a true matched filter.
    #: Always a loss: the matched filter is optimal for SNR by definition, and
    #: every taper that buys sidelobe suppression pays for it here.
    weighting_loss_db: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def as_rows(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for label, value, spec, unit in (
            ("Peak sidelobe (PSLR)", self.pslr_db, ".1f", " dB"),
            ("Integrated sidelobe (ISLR)", self.islr_db, ".1f", " dB"),
            ("Compressed width (-3 dB)", self.mainlobe_3db_s * 1e6, ".3f", " us"),
            ("Range resolution achieved", self.mainlobe_3db_m, ",.4g", " m"),
            ("Range resolution from bandwidth", self.range_resolution_m, ",.4g", " m"),
            ("Mainlobe broadening", self.broadening, ".2f", "x"),
            ("Compression ratio", self.time_bandwidth_product, ",.0f", ""),
            ("Processing gain", self.processing_gain_db, ".1f", " dB"),
            ("Weighting SNR loss", self.weighting_loss_db, ".2f", " dB"),
        ):
            if value is None or not np.isfinite(value):
                continue
            rows.append((label, f"{value:{spec}}{unit}"))
        return rows


def compressed_response(fd: FilterDesign) -> tuple[np.ndarray, np.ndarray]:
    """The compressed pulse: ``(lag_seconds, complex_amplitude)``.

    For an LFM matched filter this is the filter's response to the real,
    *unweighted* transmit pulse. That distinction matters: autocorrelating the
    filter instead applies the taper twice and reports range sidelobes more
    than 10 dB better than the hardware will ever produce.

    For any other FIR the autocorrelation is the right analogue, since such a
    filter is its own matched reference.
    """
    from .radar import lfm_transmit_pulse

    taps = np.asarray(fd.b)
    spec = fd.spec

    if spec.response is Response.MATCHED_LFM:
        reference = lfm_transmit_pulse(
            spec.sample_rate,
            spec.pulse_width_s,
            spec.chirp_bandwidth_hz,
            spec.down_chirp,
        )
        out = np.convolve(taps, reference)
    else:
        out = np.convolve(taps, np.conj(taps[::-1]))

    peak = int(np.argmax(np.abs(out)))
    lags = (np.arange(out.size) - peak) / spec.sample_rate
    return lags, out


def _mainlobe_bounds(mag: np.ndarray, peak: int) -> tuple[int, int]:
    """Indices of the first null either side of ``peak``."""
    right = peak
    while right + 1 < mag.size and mag[right + 1] < mag[right]:
        right += 1
    left = peak
    while left - 1 >= 0 and mag[left - 1] < mag[left]:
        left -= 1
    return left, right


def _width_at(mag: np.ndarray, peak: int, fraction: float) -> float:
    """Width in samples where ``mag`` falls to ``fraction`` of its peak.

    Both crossings are interpolated between the bracketing samples. At a
    realistic sample rate a compressed pulse is only a couple of samples wide,
    so taking the nearest sample instead would quantise the answer into
    uselessness.
    """
    target = mag[peak] * fraction
    right = peak
    while right + 1 < mag.size and mag[right + 1] > target:
        right += 1
    left = peak
    while left - 1 >= 0 and mag[left - 1] > target:
        left -= 1

    # The crossing sits between `right` and `right+1`, and between `left-1`
    # and `left`. The fraction is measured from the inner sample outwards, so
    # it is *added* going right and *subtracted* going left.
    hi = float(right)
    if right + 1 < mag.size:
        drop = mag[right] - mag[right + 1]
        if drop > 0:
            hi = right + (mag[right] - target) / drop

    lo = float(left)
    if left - 1 >= 0:
        drop = mag[left] - mag[left - 1]
        if drop > 0:
            lo = left - (mag[left] - target) / drop

    return max(hi - lo, 0.0)


def radar_metrics(fd: FilterDesign) -> RadarMetrics:
    """Measure range sidelobes and resolution from the compressed pulse."""
    from .radar import C_LIGHT, lfm_matched_filter, lfm_transmit_pulse

    spec = fd.spec
    metrics = RadarMetrics()
    notes = metrics.notes

    _lags, compressed = compressed_response(fd)
    mag = np.abs(compressed)
    if mag.size < 5 or np.max(mag) <= 0:
        notes.append("The compressed pulse is too short to measure.")
        return metrics

    peak = int(np.argmax(mag))
    mag = mag / mag[peak]
    left, right = _mainlobe_bounds(mag, peak)

    sidelobes = np.concatenate([mag[:left], mag[right + 1 :]])
    mainlobe = mag[left : right + 1]

    if sidelobes.size:
        metrics.pslr_db = float(db20(np.max(sidelobes)))
        side_energy = float(np.sum(sidelobes**2))
        main_energy = float(np.sum(mainlobe**2))
        if main_energy > 0:
            metrics.islr_db = 10.0 * math.log10(max(side_energy / main_energy, 1e-30))

    fs = spec.sample_rate
    width_3db = _width_at(mag, peak, 10 ** (-3.0 / 20.0))
    metrics.mainlobe_3db_s = width_3db / fs
    metrics.mainlobe_3db_m = width_3db / fs * C_LIGHT / 2.0
    metrics.mainlobe_null_s = (right - left) / fs

    if spec.response is not Response.MATCHED_LFM:
        return metrics

    # --- pulse-compression-only figures ----------------------------------
    metrics.range_resolution_m = spec.range_resolution_m
    metrics.time_bandwidth_product = spec.time_bandwidth_product
    metrics.processing_gain_db = 10.0 * math.log10(
        max(spec.time_bandwidth_product, 1e-12)
    )

    # Broadening is measured against the unweighted filter, which is the true
    # matched filter and therefore both the resolution and the SNR optimum.
    unweighted = lfm_matched_filter(
        sample_rate=spec.sample_rate,
        pulse_width_s=spec.pulse_width_s,
        bandwidth_hz=spec.chirp_bandwidth_hz,
        window="boxcar",
        down_chirp=spec.down_chirp,
        normalisation="energy",
    )
    tx = lfm_transmit_pulse(
        spec.sample_rate, spec.pulse_width_s, spec.chirp_bandwidth_hz, spec.down_chirp
    )
    plain = np.abs(np.convolve(unweighted, tx))
    plain_peak = int(np.argmax(plain))
    plain = plain / plain[plain_peak]
    plain_width = _width_at(plain, plain_peak, 10 ** (-3.0 / 20.0))
    if plain_width > 0:
        metrics.broadening = width_3db / plain_width

    # Mismatch loss of the taper: (sum w)^2 / (N * sum w^2), which is 0 dB for
    # a rectangular window and a loss for every other.
    weights = np.abs(np.asarray(fd.b))
    peak_w = float(np.max(weights))
    if peak_w > 0:
        weights = weights / peak_w
        n = weights.size
        denominator = n * float(np.sum(weights**2))
        if denominator > 0:
            efficiency = float(np.sum(weights)) ** 2 / denominator
            metrics.weighting_loss_db = -10.0 * math.log10(max(efficiency, 1e-30))

    if spec.time_bandwidth_product < 50:
        notes.append(
            f"A time-bandwidth product of {spec.time_bandwidth_product:,.0f} is "
            "low. Below roughly 50 the chirp spectrum is too ragged at its "
            "edges for the weighting to reach its nominal sidelobe level."
        )
    return metrics


# --------------------------------------------------------------------------
# MTI / Doppler
# --------------------------------------------------------------------------
def mti_velocity_response(
    fd: FilterDesign, num_points: int = 2048, max_velocity_ms: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Canceller response against radial velocity, as ``(m/s, dB)``.

    An MTI canceller's frequency axis *is* Doppler, so the useful view is
    velocity: it shows the clutter notch at zero and the blind speeds where
    legitimate targets vanish along with the clutter.
    """
    from .radar import doppler_hz, velocity_ms

    spec = fd.spec
    prf = spec.sample_rate
    if max_velocity_ms is None:
        # Far enough to show the first blind speed and back down again.
        max_velocity_ms = abs(velocity_ms(prf * 1.25, spec.radar_carrier_hz))

    velocities = np.linspace(0.0, max_velocity_ms, num_points)
    taps = np.asarray(fd.b, dtype=float)[::-1]
    # Evaluating past Nyquist is deliberate here: that wrap-around is exactly
    # what produces blind speeds, so it must show on the plot.
    z = np.exp(
        -2j
        * np.pi
        * np.array([doppler_hz(v, spec.radar_carrier_hz) for v in velocities])
        / prf
    )
    h = np.polyval(taps, z)
    return velocities, db20(h)


def ambiguity_function(
    fd: FilterDesign,
    num_doppler: int = 121,
    max_doppler_hz: float | None = None,
    delay_decimation: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range-Doppler ambiguity surface, as ``(delay_s, doppler_hz, dB)``.

    Shows what compression does to a target that is *not* at zero Doppler. An
    LFM chirp's characteristic diagonal ridge is range-Doppler coupling: a
    moving target still compresses cleanly, but at the wrong range, and this
    plot says how far wrong.
    """
    from .radar import lfm_transmit_pulse

    spec = fd.spec
    taps = np.asarray(fd.b)

    if spec.response is Response.MATCHED_LFM:
        reference = lfm_transmit_pulse(
            spec.sample_rate,
            spec.pulse_width_s,
            spec.chirp_bandwidth_hz,
            spec.down_chirp,
        )
    else:
        reference = np.conj(taps[::-1])

    if max_doppler_hz is None:
        max_doppler_hz = (
            spec.chirp_bandwidth_hz / 4.0
            if spec.chirp_bandwidth_hz > 0
            else spec.sample_rate / 8.0
        )

    dopplers = np.linspace(-max_doppler_hz, max_doppler_hz, num_doppler)
    t = np.arange(reference.size) / spec.sample_rate

    rows = []
    for shift in dopplers:
        rows.append(
            np.abs(np.convolve(taps, reference * np.exp(2j * np.pi * shift * t)))[
                ::delay_decimation
            ]
        )

    grid = np.asarray(rows)
    peak = float(np.max(grid))
    if peak > 0:
        grid = grid / peak

    centre = int(np.argmax(grid[num_doppler // 2]))
    delays = (np.arange(grid.shape[1]) - centre) * delay_decimation / spec.sample_rate
    return delays, dopplers, db20(grid)
