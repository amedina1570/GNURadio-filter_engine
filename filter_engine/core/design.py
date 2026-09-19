"""Turn a :class:`~filter_engine.core.spec.FilterSpec` into coefficients.

Everything funnels through :func:`design`, which dispatches on family and
response and returns a :class:`FilterDesign`.  IIR results always carry
second-order sections as the authoritative form -- transposed direct form II
cascades are what you actually want to implement, and ``(b, a)`` above order
~8 is numerically hopeless.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.signal import BadCoefficients

from . import firdes
from .spec import (
    FilterFamily,
    FirMethod,
    IirMethod,
    Response,
    FilterSpec,
    SpecError,
)

__all__ = ["FilterDesign", "DesignError", "design"]


class DesignError(RuntimeError):
    """Raised when a valid spec still cannot be realised by the algorithm."""


@contextlib.contextmanager
def _quiet_bad_coefficients():
    """Silence the numeric complaints scipy makes while converting forms.

    Converting second-order sections to another form makes scipy re-normalise
    each section by its leading numerator coefficient, which produces two
    kinds of noise we handle ourselves:

    * ``BadCoefficients`` whenever that coefficient is merely small, which is
      routine for a narrow lowpass and says nothing about the cascade we
      actually implement.  Where it *is* meaningful -- expanding a whole
      cascade into one ``(b, a)`` -- it is caught explicitly and reported as a
      design note.
    * a divide-by-zero once the coefficient is exactly zero, which only
      happens to coefficients we have deliberately rounded into the ground in
      :mod:`filter_engine.core.quantize`.  That module flags the filter as
      degenerate rather than trusting any number derived from it.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", BadCoefficients)
        yield


def _drop_origin_pole_zero_pairs(
    z: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove pole/zero pairs sitting exactly at the origin.

    An odd-order filter in second-order-section form carries one section
    padded to second order, which shows up as a coincident pole and zero at
    ``z = 0``.  They cancel exactly, so they are an artefact of the storage
    format rather than part of the filter: left in, they clutter the
    pole-zero plot and inflate the reported order by one.
    """
    z, p = np.asarray(z), np.asarray(p)
    z_at_origin = np.flatnonzero(z == 0)
    p_at_origin = np.flatnonzero(p == 0)
    n_pairs = min(z_at_origin.size, p_at_origin.size)
    if n_pairs == 0:
        return z, p
    return (
        np.delete(z, z_at_origin[:n_pairs]),
        np.delete(p, p_at_origin[:n_pairs]),
    )


@dataclass
class FilterDesign:
    """A designed filter: coefficients plus how they came to be."""

    spec: FilterSpec
    b: np.ndarray
    a: np.ndarray
    #: Second-order sections.  Always present for IIR, ``None`` for FIR.
    sos: np.ndarray | None = None
    #: Human-readable notes about estimation and any adjustments made.
    notes: list[str] = field(default_factory=list)

    #: IIR only: the *prototype* order handed to :func:`scipy.signal.iirfilter`.
    #: This is not :attr:`order` -- a bandpass or bandstop doubles the
    #: prototype, so an order-6 prototype yields an order-12 filter. Code
    #: generation must pass the prototype back or it designs a different
    #: filter.
    prototype_order: int | None = None
    #: IIR only: the critical frequencies actually used, in Hz. The ``*ord``
    #: helpers return a natural frequency that is generally *not* the
    #: requested passband edge (Butterworth places it between the passband and
    #: stopband edges), so the realised value is recorded rather than
    #: re-derived.
    critical_freqs: float | list[float] | None = None

    @property
    def is_fir(self) -> bool:
        return self.sos is None

    @property
    def is_complex(self) -> bool:
        """True when the taps are complex (I/Q).

        A matched filter for a complex-baseband chirp has to be complex: real
        taps cannot tell an up-sweep from its mirror image, and the compressed
        pulse would fold on top of itself.
        """
        return bool(np.iscomplexobj(self.b))

    @property
    def sample_rate(self) -> float:
        return self.spec.sample_rate

    @property
    def num_taps(self) -> int:
        """Tap count for FIR; meaningless for IIR (use :attr:`order`)."""
        return int(self.b.size)

    @property
    def order(self) -> int:
        if self.is_fir:
            return int(self.b.size) - 1
        return int(self.poles().size)

    @property
    def num_sections(self) -> int:
        return 0 if self.sos is None else int(self.sos.shape[0])

    @property
    def is_stable(self) -> bool:
        """True when every pole is strictly inside the unit circle."""
        if self.is_fir:
            return True
        poles = self.poles()
        return bool(poles.size == 0 or np.max(np.abs(poles)) < 1.0)

    def zeros(self) -> np.ndarray:
        return self._zpk()[0]

    def poles(self) -> np.ndarray:
        return self._zpk()[1]

    def _zpk(self) -> tuple[np.ndarray, np.ndarray, float]:
        with _quiet_bad_coefficients():
            if self.sos is not None:
                z, p, k = signal.sos2zpk(self.sos)
                z, p = _drop_origin_pole_zero_pairs(z, p)
            else:
                z, p, k = signal.tf2zpk(self.b, self.a)
        return np.asarray(z), np.asarray(p), float(k)

    def summary(self) -> str:
        """One-paragraph description, shown in the GUI status area."""
        s = self.spec
        lines = [
            f"{s.response.value} / {s.family.value.upper()} / {s.method}",
            f"sample rate: {s.sample_rate:,.6g} Hz  (Nyquist {s.nyquist:,.6g} Hz)",
        ]
        if self.is_fir:
            lines.append(f"taps: {self.num_taps}  (order {self.order})")
            lines.append(f"group delay: {(self.num_taps - 1) / 2.0:.1f} samples")
        else:
            lines.append(
                f"order: {self.order}  ({self.num_sections} second-order sections)"
            )
            lines.append(f"stable: {'yes' if self.is_stable else 'NO'}")
        lines.extend(self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def design(spec: FilterSpec) -> FilterDesign:
    """Design the filter described by ``spec``.

    Raises :class:`~filter_engine.core.spec.SpecError` for an unrealisable
    spec and :class:`DesignError` when the algorithm itself fails.
    """
    spec.validate()

    if spec.response.is_radar:
        return _design_radar(spec)
    if spec.response.is_pulse_shaping:
        return _design_pulse_shaping(spec)
    if spec.family is FilterFamily.FIR:
        return _design_fir(spec)
    return _design_iir(spec)


# --------------------------------------------------------------------------
# Radar
# --------------------------------------------------------------------------
def _design_radar(spec: FilterSpec) -> FilterDesign:
    from . import radar

    if spec.response is Response.MTI_CANCELLER:
        taps = radar.mti_canceller(spec.mti_pulses) * spec.gain
        notes = [
            f"{spec.mti_pulses}-pulse binomial canceller "
            f"(order {spec.mti_pulses - 1}).",
            "This filter runs in slow time -- one sample per PRI, across "
            "pulses at the same range -- so its sample rate is the "
            f"{spec.prf_hz:,.6g} Hz PRF and every frequency on the response "
            "plot is a Doppler frequency.",
            f"Blind speeds every {spec.blind_speed_ms:,.4g} m/s: a target "
            "moving at one of those advances a whole Doppler cycle between "
            "pulses and is cancelled along with the clutter.",
        ]
        return FilterDesign(
            spec=spec, b=np.asarray(taps, dtype=float), a=np.array([1.0]), notes=notes
        )

    # Matched filter for a linear-FM pulse.
    taps = radar.lfm_matched_filter(
        sample_rate=spec.sample_rate,
        pulse_width_s=spec.pulse_width_s,
        bandwidth_hz=spec.chirp_bandwidth_hz,
        window=spec.window,
        down_chirp=spec.down_chirp,
        taylor_nbar=spec.effective_taylor_nbar,
        taylor_sll_db=spec.taylor_sll_db,
        cheb_atten_db=spec.cheb_atten_db,
        kaiser_beta=spec.window_param,
        normalisation="energy",
    )
    taps = taps * spec.gain

    tbp = spec.time_bandwidth_product
    notes = [
        f"{'Down' if spec.down_chirp else 'Up'}-chirp matched filter: "
        f"{spec.chirp_bandwidth_hz / 1e6:,.6g} MHz swept over "
        f"{spec.pulse_width_s * 1e6:,.6g} us.",
        f"Time-bandwidth product {tbp:,.0f} -- the pulse compresses by that "
        f"factor, for {10 * math.log10(tbp):.1f} dB of processing gain.",
        f"Range resolution {spec.range_resolution_m:,.4g} m, set by the "
        "bandwidth alone.",
        f"Weighting: {spec.window}. Taps are complex (I/Q).",
    ]
    if spec.window == "taylor":
        nbar = spec.effective_taylor_nbar
        minimum = radar.minimum_taylor_nbar(spec.taylor_sll_db)
        notes.append(
            f"Taylor weighting: {spec.taylor_sll_db:g} dB design sidelobe "
            f"level with nbar={nbar}"
            + (" (derived from the sidelobe level)." if spec.auto_taylor_nbar
               else ".")
        )
        if nbar < minimum:
            notes.append(
                f"WARNING: nbar={nbar} is too small for a "
                f"{spec.taylor_sll_db:g} dB design. Taylor needs nbar of at "
                f"least {minimum} to hold that level; below it the realised "
                "sidelobes sit several dB higher than asked for."
            )
    if spec.window == "boxcar":
        notes.append(
            "Unweighted: range sidelobes will sit near -13 dB, so a strong "
            "target will mask weaker ones a few cells away. Apply a Taylor "
            "weighting to push them down."
        )
    return FilterDesign(
        spec=spec, b=np.asarray(taps), a=np.array([1.0]), notes=notes
    )


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------
def _design_pulse_shaping(spec: FilterSpec) -> FilterDesign:
    sps = spec.samples_per_symbol
    kwargs = dict(
        sps=sps,
        span_symbols=spec.span_symbols,
        gain=spec.gain,
        normalisation=spec.normalisation,
    )
    if spec.response is Response.RRC:
        taps = firdes.root_raised_cosine(rolloff=spec.rolloff, **kwargs)
    elif spec.response is Response.RC:
        taps = firdes.raised_cosine(rolloff=spec.rolloff, **kwargs)
    else:
        taps = firdes.gaussian(bt=spec.bt, **kwargs)

    notes = [
        f"{sps:.4g} samples/symbol, {spec.span_symbols} symbol span "
        f"-> {taps.size} taps",
        f"normalisation: {spec.normalisation}",
    ]
    return FilterDesign(spec=spec, b=taps, a=np.array([1.0]), notes=notes)


# --------------------------------------------------------------------------
# FIR
# --------------------------------------------------------------------------
def _design_fir(spec: FilterSpec) -> FilterDesign:
    notes: list[str] = []
    ntaps = _fir_tap_count(spec, notes)

    if spec.response is Response.HILBERT:
        taps = _fir_hilbert(spec, ntaps, notes)
    elif spec.response is Response.DIFFERENTIATOR:
        taps = _fir_differentiator(spec, ntaps, notes)
    elif spec.fir_method is FirMethod.WINDOW:
        taps = _fir_window(spec, ntaps, notes)
    elif spec.fir_method is FirMethod.REMEZ:
        taps = _fir_remez(spec, ntaps, notes)
    elif spec.fir_method is FirMethod.FIRLS:
        taps = _fir_firls(spec, ntaps, notes)
    elif spec.fir_method is FirMethod.FREQ_SAMPLING:
        taps = _fir_freq_sampling(spec, ntaps, notes)
    else:
        raise DesignError(f"unsupported FIR method {spec.fir_method!r}")

    taps = np.asarray(taps, dtype=float)
    if not np.all(np.isfinite(taps)):
        raise DesignError(
            "The design produced non-finite taps. This usually means the tap "
            "count is far too high for the requested transition width."
        )
    return FilterDesign(spec=spec, b=taps, a=np.array([1.0]), notes=notes)


def _fir_tap_count(spec: FilterSpec, notes: list[str]) -> int:
    """Resolve the tap count, estimating it when ``auto_order`` is set."""
    # Types II and IV have a forced zero at Nyquist, so any response that must
    # pass Nyquist needs an odd tap count (Type I / III).
    needs_odd = spec.response in (
        Response.HIGHPASS,
        Response.BANDSTOP,
        Response.HILBERT,
        Response.DIFFERENTIATOR,
    ) or spec.fir_method is FirMethod.FIRLS

    if spec.auto_order:
        cap = 32_001
        if spec.response is Response.HILBERT:
            # No transition width is specified for a Hilbert transformer; base
            # the length on the width of the band it has to cover.
            width = max(spec.f_low, spec.nyquist - spec.f_high)
            width = max(width, spec.sample_rate * 1e-4)
        elif spec.response is Response.DIFFERENTIATOR:
            # The only transition a differentiator has is the guard between
            # its band edge and Nyquist.  They also go ill-conditioned fast --
            # the Remez exchange stops converging well short of a hundred
            # taps -- so the length is capped hard.
            width = max(spec.nyquist - spec.f_high, spec.sample_rate * 1e-3)
            cap = 63
        else:
            width = spec.transition_width
        ntaps, note = firdes.estimate_fir_taps(
            spec.fir_method.value,
            spec.sample_rate,
            width,
            spec.passband_ripple_db,
            spec.stopband_atten_db,
            window=spec.window,
            force_odd=needs_odd,
        )
        if ntaps > cap:
            ntaps = cap
            note += f", capped at {cap} taps to keep the design well conditioned"
        notes.append(f"tap count {ntaps} estimated: {note}")
    else:
        ntaps = int(spec.num_taps)
        if needs_odd and ntaps % 2 == 0:
            ntaps += 1
            notes.append(
                f"tap count raised to {ntaps}: this response needs an odd "
                "number of taps to avoid a forced null at Nyquist"
            )
    return ntaps


def _fir_cutoffs(spec: FilterSpec) -> list[float]:
    """The -6 dB cutoff(s) for a windowed design, in Hz."""
    if spec.response.is_multiband:
        return [spec.f_low, spec.f_high]
    return [spec.f_low]


def _fir_window(spec: FilterSpec, ntaps: int, notes: list[str]) -> np.ndarray:
    if spec.window == "kaiser":
        if spec.auto_order:
            # Pair the estimated length with the beta the same rule implies.
            beta = firdes.kaiser_beta_for_atten(spec.stopband_atten_db)
            notes.append(f"Kaiser beta {beta:.3f} from the stopband spec")
        else:
            beta = spec.window_param
        win: object = ("kaiser", beta)
    else:
        win = spec.window
        limit = firdes.window_atten_limit(spec.window)
        if limit is not None and spec.stopband_atten_db > limit:
            notes.append(
                f"WARNING: a {spec.window} window tops out near {limit:.0f} dB "
                f"of stopband attenuation, short of the {spec.stopband_atten_db:.0f} dB "
                "requested. Adding taps narrows the transition but will not "
                "deepen the stopband. Switch to a "
                f"{firdes.suggest_window(spec.stopband_atten_db)} window, or use "
                "the remez method, to reach it."
            )

    pass_zero: object
    if spec.response is Response.LOWPASS:
        pass_zero = "lowpass"
    elif spec.response is Response.HIGHPASS:
        pass_zero = "highpass"
    elif spec.response is Response.BANDPASS:
        pass_zero = "bandpass"
    else:
        pass_zero = "bandstop"

    notes.append(f"firwin with a {spec.window} window; cutoffs are the -6 dB points")
    return signal.firwin(
        ntaps,
        _fir_cutoffs(spec),
        window=win,
        pass_zero=pass_zero,
        scale=True,
        fs=spec.sample_rate,
    ) * spec.gain


def _band_edges(spec: FilterSpec) -> tuple[list[float], list[float], list[float]]:
    """Build ``(bands, desired, weights)`` for remez/firls.

    ``bands`` is a flat list of band edges in Hz; ``desired`` has one entry per
    band; ``weights`` has one entry per band and is inversely proportional to
    the allowed deviation, which is how you tell Parks-McClellan that 0.1 dB
    of passband ripple matters far less than 60 dB of stopband rejection.
    """
    nyq = spec.nyquist
    half = spec.transition_width / 2.0
    dp, ds = firdes.ripple_to_delta(spec.passband_ripple_db, spec.stopband_atten_db)
    w_pass, w_stop = 1.0 / dp, 1.0 / ds
    g = spec.gain

    if spec.response is Response.LOWPASS:
        bands = [0.0, spec.f_low - half, spec.f_low + half, nyq]
        desired = [g, 0.0]
        weights = [w_pass, w_stop]
    elif spec.response is Response.HIGHPASS:
        bands = [0.0, spec.f_low - half, spec.f_low + half, nyq]
        desired = [0.0, g]
        weights = [w_stop, w_pass]
    elif spec.response is Response.BANDPASS:
        bands = [
            0.0,
            spec.f_low - half,
            spec.f_low + half,
            spec.f_high - half,
            spec.f_high + half,
            nyq,
        ]
        desired = [0.0, g, 0.0]
        weights = [w_stop, w_pass, w_stop]
    elif spec.response is Response.BANDSTOP:
        bands = [
            0.0,
            spec.f_low - half,
            spec.f_low + half,
            spec.f_high - half,
            spec.f_high + half,
            nyq,
        ]
        desired = [g, 0.0, g]
        weights = [w_pass, w_stop, w_pass]
    else:
        raise DesignError(
            f"{spec.response.value} is not a multi-band mask design"
        )
    return bands, desired, weights


def _fir_remez(spec: FilterSpec, ntaps: int, notes: list[str]) -> np.ndarray:
    bands, desired, weights = _band_edges(spec)
    notes.append("Parks-McClellan (remez): equiripple in every band")
    try:
        return signal.remez(
            ntaps, bands, desired, weight=weights, fs=spec.sample_rate
        )
    except Exception as exc:  # scipy raises a bare ValueError on non-convergence
        raise DesignError(
            "The Parks-McClellan exchange failed to converge. Try more taps, a "
            "wider transition, or a less aggressive stopband.\n"
            f"(scipy said: {exc})"
        ) from exc


def _fir_firls(spec: FilterSpec, ntaps: int, notes: list[str]) -> np.ndarray:
    bands, desired, weights = _band_edges(spec)
    # firls wants a desired value at each band *edge*, not one per band.
    desired_edges: list[float] = []
    for value in desired:
        desired_edges.extend([value, value])
    if ntaps % 2 == 0:
        ntaps += 1
        notes.append(f"tap count raised to {ntaps}: firls requires an odd length")
    notes.append("Weighted least squares (firls): minimises total squared error")
    try:
        return signal.firls(
            ntaps, bands, desired_edges, weight=weights, fs=spec.sample_rate
        )
    except Exception as exc:
        raise DesignError(f"firls failed: {exc}") from exc


def _fir_freq_sampling(spec: FilterSpec, ntaps: int, notes: list[str]) -> np.ndarray:
    bands, desired, _weights = _band_edges(spec)
    freqs: list[float] = []
    gains: list[float] = []
    for i, value in enumerate(desired):
        lo, hi = bands[2 * i], bands[2 * i + 1]
        freqs.extend([lo, hi])
        gains.extend([value, value])
    # firwin2 requires a strictly increasing frequency vector starting at 0
    # and ending at Nyquist, with no repeated interior points.
    freqs[0] = 0.0
    freqs[-1] = spec.nyquist
    for i in range(1, len(freqs)):
        if freqs[i] <= freqs[i - 1]:
            freqs[i] = freqs[i - 1] + spec.sample_rate * 1e-9

    if gains[-1] != 0.0 and ntaps % 2 == 0:
        ntaps += 1
        notes.append(
            f"tap count raised to {ntaps}: a non-zero gain at Nyquist needs an "
            "odd length"
        )
    notes.append("Frequency sampling (firwin2): interpolates the requested mask")
    try:
        return signal.firwin2(ntaps, freqs, gains, fs=spec.sample_rate)
    except Exception as exc:
        raise DesignError(f"firwin2 failed: {exc}") from exc


def _fir_hilbert(spec: FilterSpec, ntaps: int, notes: list[str]) -> np.ndarray:
    if ntaps % 2 == 0:
        ntaps += 1
    nyq = spec.nyquist
    lo = max(spec.f_low, nyq * 1e-4)
    hi = min(spec.f_high, nyq * (1.0 - 1e-4))
    if hi <= lo:
        raise DesignError("Hilbert band edges collapse; widen the passband.")
    notes.append(
        f"Type III Hilbert transformer over {lo:.6g}-{hi:.6g} Hz "
        "(90 deg phase shift, zero response at DC and Nyquist)"
    )
    try:
        taps = signal.remez(
            ntaps, [lo, hi], [spec.gain], type="hilbert", fs=spec.sample_rate
        )
    except Exception as exc:
        raise DesignError(f"Hilbert design failed: {exc}") from exc
    return taps


def _fir_differentiator(spec: FilterSpec, ntaps: int, notes: list[str]) -> np.ndarray:
    if ntaps % 2 == 0:
        ntaps += 1
    nyq = spec.nyquist
    hi = min(spec.f_high, nyq * (1.0 - 1e-4))
    notes.append(
        f"Type III differentiator up to {hi:.6g} Hz "
        "(magnitude proportional to frequency)"
    )
    try:
        # remez' differentiator type wants the desired slope per unit
        # frequency; scaling by 1/fs gives a unit-gain-per-radian response.
        taps = signal.remez(
            ntaps,
            [0.0, hi],
            [spec.gain / spec.sample_rate],
            type="differentiator",
            fs=spec.sample_rate,
        )
    except Exception as exc:
        raise DesignError(f"Differentiator design failed: {exc}") from exc
    return taps


# --------------------------------------------------------------------------
# IIR
# --------------------------------------------------------------------------
_ORD_FUNCS = {
    IirMethod.BUTTER: signal.buttord,
    IirMethod.CHEBY1: signal.cheb1ord,
    IirMethod.CHEBY2: signal.cheb2ord,
    IirMethod.ELLIP: signal.ellipord,
}


def _design_iir(spec: FilterSpec) -> FilterDesign:
    notes: list[str] = []
    btype = {
        Response.LOWPASS: "lowpass",
        Response.HIGHPASS: "highpass",
        Response.BANDPASS: "bandpass",
        Response.BANDSTOP: "bandstop",
    }.get(spec.response)
    if btype is None:
        raise SpecError(
            f"{spec.response.value} has no IIR form in this tool; use FIR for "
            "Hilbert, differentiator and pulse-shaping responses."
        )

    order, wn = _iir_order_and_edges(spec, btype, notes)

    kwargs: dict[str, float] = {}
    if spec.iir_method in (IirMethod.CHEBY1, IirMethod.ELLIP):
        kwargs["rp"] = spec.passband_ripple_db
    if spec.iir_method in (IirMethod.CHEBY2, IirMethod.ELLIP):
        kwargs["rs"] = spec.stopband_atten_db

    try:
        sos = signal.iirfilter(
            order,
            wn,
            btype=btype,
            ftype=spec.iir_method.value,
            output="sos",
            fs=spec.sample_rate,
            **kwargs,
        )
    except Exception as exc:
        raise DesignError(f"IIR design failed: {exc}") from exc

    sos = np.atleast_2d(np.asarray(sos, dtype=float))
    # Apply the requested gain to the first section's numerator so the
    # cascade's overall gain scales without disturbing the pole locations.
    if spec.gain != 1.0:
        sos = sos.copy()
        sos[0, 0:3] *= spec.gain

    # (b, a) is provided for display and for the generated code's reference,
    # but above roughly order 8 the expanded polynomial is numerically
    # worthless -- scipy says so via BadCoefficients.  Catch that and turn it
    # into a note instead of letting a warning leak out of the library.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b, a = signal.sos2tf(sos)
    if any(issubclass(w.category, BadCoefficients) for w in caught):
        notes.append(
            f"order {order} is too high for a single direct-form (b, a) "
            "section; use the second-order sections for any implementation."
        )

    design_obj = FilterDesign(
        spec=spec,
        b=np.asarray(b),
        a=np.asarray(a),
        sos=sos,
        notes=notes,
        prototype_order=int(order),
        critical_freqs=(
            [float(v) for v in np.atleast_1d(wn)]
            if np.ndim(wn) > 0
            else float(wn)
        ),
    )
    if not design_obj.is_stable:
        notes.append(
            "WARNING: at least one pole is on or outside the unit circle. "
            "Reduce the order or widen the transition."
        )
    if spec.iir_method is IirMethod.BESSEL:
        notes.append(
            "Bessel: maximally flat group delay, but the digital "
            "(bilinear-transformed) version only approximates that near DC."
        )
    return design_obj


def _iir_order_and_edges(
    spec: FilterSpec, btype: str, notes: list[str]
) -> tuple[int, float | list[float]]:
    """Return ``(order, critical_frequencies)`` for :func:`scipy.signal.iirfilter`."""
    half = spec.transition_width / 2.0

    if spec.response is Response.LOWPASS:
        wp: float | list[float] = spec.f_low - half
        ws: float | list[float] = spec.f_low + half
    elif spec.response is Response.HIGHPASS:
        wp = spec.f_low + half
        ws = spec.f_low - half
    elif spec.response is Response.BANDPASS:
        wp = [spec.f_low + half, spec.f_high - half]
        ws = [spec.f_low - half, spec.f_high + half]
    else:  # bandstop
        wp = [spec.f_low - half, spec.f_high + half]
        ws = [spec.f_low + half, spec.f_high - half]

    if spec.auto_order:
        ord_func = _ORD_FUNCS.get(spec.iir_method)
        if ord_func is None:
            # Bessel has no order-selection helper: its stopband is so gentle
            # that a tolerance-based estimate is meaningless.
            notes.append(
                f"Bessel has no order estimator; using the explicit order "
                f"{spec.order}."
            )
            return spec.order, wp
        try:
            order, wn = ord_func(
                wp,
                ws,
                spec.passband_ripple_db,
                spec.stopband_atten_db,
                fs=spec.sample_rate,
            )
        except Exception as exc:
            raise DesignError(
                "Order estimation failed. Check that the passband and stopband "
                f"edges make sense for a {spec.response.value}.\n(scipy said: {exc})"
            ) from exc
        order = int(order)
        if order < 1:
            order = 1
        notes.append(
            f"order {order} estimated from {spec.passband_ripple_db:.3g} dB "
            f"ripple / {spec.stopband_atten_db:.1f} dB stopband"
        )
        # cheby2 places its critical frequency in the stopband, so the natural
        # frequencies returned by the *ord helper are the ones to use.
        return order, wn

    notes.append(f"explicit order {spec.order}")
    return spec.order, wp
