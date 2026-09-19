"""Radar waveforms, weighting functions and the units that go with them.

Radar filtering asks different questions from communications filtering. Nobody
designing a pulse compression filter cares about passband ripple; they care
about *range sidelobes* -- how much a strong target smears across neighbouring
range cells and buries a weak one. The figures of merit are peak sidelobe
ratio (PSLR) and integrated sidelobe ratio (ISLR), and the knob that sets them
is the weighting window.

That is what the Taylor window is for, and why it is the one you see
everywhere in radar: it lets you name the sidelobe level you want and how many
sidelobes should sit at that level, and it gets there with less mainlobe
broadening (less lost resolution) than a Hamming or Blackman taper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal as sig

__all__ = [
    "C_LIGHT",
    "range_resolution_m",
    "unambiguous_range_m",
    "unambiguous_velocity_ms",
    "blind_speed_ms",
    "wavelength_m",
    "doppler_hz",
    "velocity_ms",
    "lfm_matched_filter",
    "lfm_transmit_pulse",
    "mti_canceller",
    "WEIGHTINGS",
    "weighting_window",
    "minimum_taylor_nbar",
    "lfm_length",
]

#: Speed of light in vacuum, m/s.
C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------
# Unit conversions
#
# These exist so the GUI can show a user what their numbers mean in the units
# they actually think in. "10 MHz of chirp bandwidth" is abstract; "15 m range
# resolution" is not.
# --------------------------------------------------------------------------
def range_resolution_m(bandwidth_hz: float) -> float:
    """Range resolution from swept bandwidth.

    Two targets closer than this merge into one return. It depends only on
    bandwidth -- not on pulse length, and not on transmit power.
    """
    if bandwidth_hz <= 0:
        return float("inf")
    return C_LIGHT / (2.0 * bandwidth_hz)


def unambiguous_range_m(pri_s: float) -> float:
    """Furthest range whose echo returns before the next pulse goes out.

    An echo from beyond this arrives after the next transmission and is
    reported at a short range instead -- a "second-time-around" target.
    """
    return C_LIGHT * pri_s / 2.0


def wavelength_m(carrier_hz: float) -> float:
    if carrier_hz <= 0:
        return float("inf")
    return C_LIGHT / carrier_hz


def doppler_hz(velocity_ms: float, carrier_hz: float) -> float:
    """Two-way Doppler shift for a closing velocity."""
    return 2.0 * velocity_ms / wavelength_m(carrier_hz)


def velocity_ms(doppler_shift_hz: float, carrier_hz: float) -> float:
    """Radial velocity implied by a Doppler shift."""
    return doppler_shift_hz * wavelength_m(carrier_hz) / 2.0


def unambiguous_velocity_ms(pri_s: float, carrier_hz: float) -> float:
    """Velocity span the PRF can measure without folding, +/- half of this."""
    if pri_s <= 0:
        return float("inf")
    return wavelength_m(carrier_hz) / (2.0 * pri_s)


def blind_speed_ms(pri_s: float, carrier_hz: float, n: int = 1) -> float:
    """The n-th blind speed of an MTI canceller.

    A target moving at exactly this speed advances a whole number of Doppler
    cycles between pulses, so it looks stationary and the canceller removes it
    along with the clutter. Staggering the PRI is the usual way out.
    """
    if pri_s <= 0:
        return float("inf")
    return n * wavelength_m(carrier_hz) / (2.0 * pri_s)


# --------------------------------------------------------------------------
# Weighting windows
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Weighting:
    """A weighting (taper) function and what it costs.

    ``pslr_db`` and ``broadening`` are the two numbers that decide a radar
    taper: how far down the sidelobes go, and how much mainlobe -- how much
    resolution -- you give up to get there.
    """

    name: str
    label: str
    #: Typical peak sidelobe ratio, dB below the mainlobe peak. ``None`` when
    #: the window is parameterised and you choose it.
    pslr_db: float | None
    #: Mainlobe width relative to an unweighted (rectangular) aperture.
    broadening: float
    description: str


#: Weightings worth offering for pulse compression and aperture tapering,
#: cheapest sidelobes first. The broadening figures are for the -3 dB
#: mainlobe width against a rectangular taper.
WEIGHTINGS: tuple[Weighting, ...] = (
    Weighting(
        "boxcar", "None (rectangular)", -13.3, 1.00,
        "No weighting. Best resolution and no SNR loss, but -13 dB sidelobes "
        "mean a strong target hides weak ones several cells away.",
    ),
    Weighting(
        "hann", "Hann", -31.5, 1.65,
        "Sidelobes fall away quickly with distance, which helps against "
        "distributed clutter.",
    ),
    Weighting(
        "hamming", "Hamming", -42.7, 1.47,
        "The classic compromise: much better than Hann near the mainlobe for "
        "slightly less broadening.",
    ),
    Weighting(
        "blackman", "Blackman", -58.1, 1.90,
        "Deep sidelobes at a real cost in resolution.",
    ),
    Weighting(
        "blackmanharris", "Blackman-Harris", -92.0, 2.24,
        "Very deep sidelobes; broad mainlobe.",
    ),
    Weighting(
        "taylor", "Taylor", None, 1.0,
        "The radar standard. You name the sidelobe level and how many "
        "sidelobes sit at it, and it gets there with less mainlobe "
        "broadening than any fixed window achieving the same level.",
    ),
    Weighting(
        "chebwin", "Dolph-Chebyshev", None, 1.0,
        "Every sidelobe at exactly the level you name -- the narrowest "
        "possible mainlobe for that level. The far sidelobes never decay, "
        "which can integrate badly against extended clutter.",
    ),
    Weighting(
        "kaiser", "Kaiser", None, 1.0,
        "A close, cheaper approximation to Dolph-Chebyshev, tuned by a "
        "single beta parameter.",
    ),
)

_WEIGHTING_BY_NAME = {w.name: w for w in WEIGHTINGS}


def get_weighting(name: str) -> Weighting | None:
    return _WEIGHTING_BY_NAME.get(name)


def minimum_taylor_nbar(sll_db: float) -> int:
    """Smallest ``nbar`` that can actually deliver ``sll_db`` of suppression.

    Taylor's own constraint: the number of flat-topped sidelobes has to grow
    with the depth you ask for, or the design degenerates and the realised
    sidelobes sit well above the nominal level. The standard rule is

        nbar >= 2*A^2 + 0.5,   A = arccosh(10^(SLL/20)) / pi

    Asking for 50 dB with ``nbar=4`` quietly gets you 45 -- one of the most
    common ways a Taylor taper disappoints, and the reason this tool derives
    ``nbar`` by default rather than leaving it at a fixed value.
    """
    sll = abs(float(sll_db))
    if sll <= 0:
        return 1
    a = math.acosh(10.0 ** (sll / 20.0)) / math.pi
    return max(1, int(math.ceil(2.0 * a * a + 0.5)))


def weighting_window(
    name: str,
    length: int,
    taylor_nbar: int = 4,
    taylor_sll_db: float = 35.0,
    cheb_atten_db: float = 60.0,
    kaiser_beta: float = 8.6,
) -> np.ndarray:
    """Build a weighting window of ``length`` samples.

    The parameterised windows each take their own knob; the rest ignore all
    of them. Keeping one entry point means callers never have to know which
    is which.
    """
    if length < 1:
        raise ValueError("window length must be at least 1")

    if name == "taylor":
        # scipy takes sll as a positive number of dB below the peak.
        return sig.windows.taylor(
            length, nbar=int(taylor_nbar), sll=abs(taylor_sll_db), norm=True, sym=True
        )
    if name == "chebwin":
        if cheb_atten_db < 45.0:
            # scipy warns below 45 dB because the equivalent noise bandwidth
            # stops growing monotonically; the window is still well defined.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return sig.windows.chebwin(length, at=abs(cheb_atten_db), sym=True)
        return sig.windows.chebwin(length, at=abs(cheb_atten_db), sym=True)
    if name == "kaiser":
        return sig.windows.kaiser(length, beta=kaiser_beta, sym=True)
    return sig.get_window(name, length, fftbins=False)


# --------------------------------------------------------------------------
# Waveforms
# --------------------------------------------------------------------------
def lfm_transmit_pulse(
    sample_rate: float,
    pulse_width_s: float,
    bandwidth_hz: float,
    down_chirp: bool = False,
) -> np.ndarray:
    """The transmitted LFM pulse itself, unweighted.

    Needed to measure range sidelobes honestly. The compressed pulse is this
    waveform run through the matched filter -- *not* the filter's own
    autocorrelation. Autocorrelating the filter applies the taper twice, which
    flatters a Taylor-weighted design by more than 10 dB and would report
    sidelobes the hardware will never deliver.
    """
    ntaps = lfm_length(sample_rate, pulse_width_s)
    t = (np.arange(ntaps) - (ntaps - 1) / 2.0) / sample_rate
    chirp_rate = bandwidth_hz / pulse_width_s
    if down_chirp:
        chirp_rate = -chirp_rate
    return np.exp(1j * np.pi * chirp_rate * t**2)


def lfm_length(sample_rate: float, pulse_width_s: float) -> int:
    """Tap count for an LFM matched filter, forced odd.

    Public because the GUI reports it before the design runs, and a figure
    that disagreed with the real one by a tap would be its own small bug.
    """
    ntaps = int(round(pulse_width_s * sample_rate))
    if ntaps % 2 == 0:
        ntaps += 1
    return max(ntaps, 3)


def lfm_matched_filter(
    sample_rate: float,
    pulse_width_s: float,
    bandwidth_hz: float,
    window: str = "taylor",
    down_chirp: bool = False,
    taylor_nbar: int = 4,
    taylor_sll_db: float = 35.0,
    cheb_atten_db: float = 60.0,
    kaiser_beta: float = 8.6,
    normalisation: str = "energy",
) -> np.ndarray:
    """Matched filter for a linear-FM (chirp) pulse.

    The transmitted pulse sweeps ``bandwidth_hz`` over ``pulse_width_s``::

        s(t) = exp(j*pi*K*t^2),    K = B / tau,   -tau/2 <= t <= tau/2

    The matched filter is its time-reversed conjugate. Because ``t^2`` is
    even, the reversal is a no-op and the filter is simply ``conj(s(t))`` --
    the same chirp sweeping the other way.

    **The taps are complex.** A matched filter for a complex baseband chirp
    has to be; a real-tap filter cannot distinguish the positive-frequency
    sweep from its mirror image, which would fold the compressed pulse on top
    of itself.

    Returns the weighted, normalised tap vector.
    """
    if sample_rate <= 0:
        raise ValueError("sample rate must be > 0")
    if pulse_width_s <= 0:
        raise ValueError("pulse width must be > 0")
    if bandwidth_hz <= 0:
        raise ValueError("chirp bandwidth must be > 0")

    ntaps = lfm_length(sample_rate, pulse_width_s)
    # The matched filter is the conjugate of the transmitted pulse. Because
    # t^2 is even, time-reversing it changes nothing, so this is the same
    # chirp sweeping the other way.
    taps = np.conj(
        lfm_transmit_pulse(sample_rate, pulse_width_s, bandwidth_hz, down_chirp)
    )

    taps = taps * weighting_window(
        window, ntaps, taylor_nbar, taylor_sll_db, cheb_atten_db, kaiser_beta
    )
    return _normalise_complex(taps, normalisation)


def _normalise_complex(taps: np.ndarray, mode: str) -> np.ndarray:
    """Scale complex taps. ``energy`` is the matched-filter convention."""
    if mode == "energy":
        scale = math.sqrt(float(np.sum(np.abs(taps) ** 2)))
    elif mode == "peak":
        scale = float(np.max(np.abs(taps)))
    elif mode == "sum":
        scale = float(np.abs(np.sum(taps)))
    else:  # none
        scale = 1.0
    if not np.isfinite(scale) or scale < 1e-20:
        scale = 1.0
    return taps / scale


#: Binomial MTI canceller coefficients, indexed by pulse count.
#:
#: An N-pulse canceller is (N-1) cascaded single cancellers, so its
#: coefficients are the binomial row with alternating signs. Every one of them
#: has an (N-1)-order null at zero Doppler, which is exactly what you want
#: against stationary clutter.
_MTI_TAPS: dict[int, list[float]] = {
    2: [1.0, -1.0],
    3: [1.0, -2.0, 1.0],
    4: [1.0, -3.0, 3.0, -1.0],
    5: [1.0, -4.0, 6.0, -4.0, 1.0],
}


def mti_canceller(pulses: int, normalise: bool = True) -> np.ndarray:
    """Coefficients for an ``pulses``-pulse MTI canceller.

    This filter runs in *slow time* -- across pulses at the same range, one
    sample per PRI -- not along the received pulse. Its sample rate is the
    PRF, so every frequency on its response plot is a Doppler frequency.

    ``normalise`` scales for unit gain at the Nyquist Doppler (the velocity
    the canceller passes best), so cancellers of different length can be
    compared on one axis.
    """
    try:
        taps = np.array(_MTI_TAPS[int(pulses)], dtype=float)
    except KeyError:
        raise ValueError(
            f"{pulses}-pulse canceller is not supported; choose "
            f"{sorted(_MTI_TAPS)}"
        ) from None
    if normalise:
        # Gain at fd = PRF/2 is the sum of absolute values for these taps.
        taps = taps / np.sum(np.abs(taps))
    return taps


def mti_improvement_factor_db(pulses: int, clutter_spread_ratio: float) -> float:
    """Approximate MTI improvement factor against Gaussian clutter.

    ``clutter_spread_ratio`` is the clutter spectral standard deviation
    divided by the PRF. The result says how much clutter power the canceller
    removes relative to how much target power it keeps -- the number that
    decides whether an extra pulse is worth the extra blind-speed trouble.
    """
    if clutter_spread_ratio <= 0:
        return float("inf")
    n = int(pulses) - 1  # canceller order
    if n < 1:
        return 0.0
    # I_n = (2 (2 pi sigma/PRF)^2)^-n * (2n)! / (n! 2^n)  -- standard result.
    x = 2.0 * math.pi * clutter_spread_ratio
    factor = math.factorial(2 * n) / (math.factorial(n) * (2.0**n))
    value = factor / (x ** (2 * n))
    return 10.0 * math.log10(value) if value > 0 else 0.0
