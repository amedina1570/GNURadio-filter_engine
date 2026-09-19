"""Plots of a design, drawn onto a matplotlib figure.

Deliberately free of Qt: every function takes a :class:`~matplotlib.figure.Figure`
and a design, so the same code serves the GUI, a script, and the tests.

Two conventions run through all of it.  Frequency axes auto-scale to Hz/kHz/MHz
rather than printing 61440000. And wherever a design has been quantized, the
fixed-point curve is drawn over the ideal one in a second colour -- the gap
between them is the whole point of looking.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .core import analysis
from .core.design import FilterDesign
from .core.quantize import QuantizedFilter
from .core.spec import Response

__all__ = [
    "freq_scale",
    "plot_magnitude",
    "plot_phase",
    "plot_group_delay",
    "plot_impulse",
    "plot_step",
    "plot_pole_zero",
    "plot_taps",
    "plot_overview",
    "plot_pulse_time",
    "plot_pulse_spectrum",
    "plot_compressed_pulse",
    "plot_weighting",
    "plot_ambiguity",
    "plot_mti_velocity",
]

IDEAL = dict(color="tab:blue", linewidth=1.4)
QUANT = dict(color="tab:orange", linewidth=1.1)
MASK = dict(color="tab:red", alpha=0.10)
MASK_EDGE = dict(color="tab:red", alpha=0.5, linewidth=0.8, linestyle="--")
GRID = dict(alpha=0.25, linewidth=0.6)


def freq_scale(max_hz: float) -> tuple[float, str]:
    """Divisor and unit label for a frequency axis topping out at ``max_hz``."""
    if max_hz >= 1e9:
        return 1e9, "GHz"
    if max_hz >= 1e6:
        return 1e6, "MHz"
    if max_hz >= 1e3:
        return 1e3, "kHz"
    return 1.0, "Hz"


def _time_scale(max_s: float) -> tuple[float, str]:
    if max_s >= 1.0:
        return 1.0, "s"
    if max_s >= 1e-3:
        return 1e-3, "ms"
    if max_s >= 1e-6:
        return 1e-6, "us"
    return 1e-9, "ns"


def _grid(ax) -> None:
    ax.grid(True, which="both", **GRID)


# --------------------------------------------------------------------------
# Frequency-domain plots
# --------------------------------------------------------------------------
def plot_magnitude(
    fig: Figure,
    fd: FilterDesign,
    quantized: QuantizedFilter | None = None,
    log_freq: bool = False,
    db: bool = True,
    show_mask: bool = True,
    num_points: int = 4096,
    f_max: float | None = None,
) -> None:
    """Magnitude response, with the requested mask shaded behind it."""
    ax = fig.add_subplot(111)
    fr = analysis.frequency_response(
        fd, num_points=num_points, log_spacing=log_freq, f_max=f_max
    )
    div, unit = freq_scale(float(fr.freqs[-1]))
    x = fr.freqs / div

    if show_mask:
        _draw_mask(ax, fd, div, db)

    y = fr.mag_db if db else fr.mag_linear
    ax.plot(x, y, label="ideal", **IDEAL)

    if quantized is not None:
        try:
            frq = analysis.frequency_response(
                quantized.quantized_design,
                num_points=num_points,
                log_spacing=log_freq,
                f_max=f_max,
            )
            yq = frq.mag_db if db else frq.mag_linear
            ax.plot(
                frq.freqs / div,
                yq,
                label=f"{quantized.fmt.name} fixed point",
                **QUANT,
            )
            ax.legend(loc="best", fontsize=8)
        except Exception:
            # An unusable quantization (unstable, degenerate) still has a
            # useful ideal curve; show that rather than nothing.
            pass

    if log_freq:
        ax.set_xscale("log")
    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude")
    if db:
        floor = max(-200.0, float(np.min(y)) - 5.0)
        top = float(np.max(y)) + 5.0
        ax.set_ylim(max(floor, top - 160.0), top)
    ax.set_title(f"Magnitude response - {fd.spec.name}")
    _grid(ax)


def _draw_mask(ax, fd: FilterDesign, div: float, db: bool) -> None:
    """Shade the passband and stopband the spec asked for."""
    bands = analysis._mask_bands(fd)
    if bands is None:
        return
    spec = fd.spec
    pass_bands, stop_bands = bands

    if db:
        # Passband tolerance sits around the design's own gain, not around 0 dB.
        ref = 20.0 * np.log10(max(spec.gain, 1e-12))
        pass_lo, pass_hi = ref - spec.passband_ripple_db, ref + spec.passband_ripple_db
        stop_hi = ref - spec.stopband_atten_db
        stop_lo = stop_hi - 200.0
    else:
        ref = spec.gain
        tol = ref * (10.0 ** (spec.passband_ripple_db / 20.0) - 1.0)
        pass_lo, pass_hi = ref - tol, ref + tol
        stop_hi = ref * 10.0 ** (-spec.stopband_atten_db / 20.0)
        stop_lo = 0.0

    # fill_between, not axhspan: axhspan takes its x limits as *axes
    # fractions*, so feeding it frequencies silently smears the passband
    # across the whole plot and pushes the stopband off the right-hand edge.
    for lo, hi in pass_bands:
        if hi > lo:
            ax.fill_between([lo / div, hi / div], pass_lo, pass_hi, **MASK)
    for lo, hi in stop_bands:
        if hi > lo:
            ax.fill_between([lo / div, hi / div], stop_lo, stop_hi, **MASK)

    for lo, hi in pass_bands + stop_bands:
        for edge in (lo, hi):
            if 0 < edge < spec.nyquist:
                ax.axvline(edge / div, **MASK_EDGE)


def plot_phase(
    fig: Figure, fd: FilterDesign, unwrap: bool = True, num_points: int = 4096
) -> None:
    """Phase response, plus phase delay on a twin axis."""
    ax = fig.add_subplot(111)
    fr = analysis.frequency_response(fd, num_points=num_points)
    div, unit = freq_scale(float(fr.freqs[-1]))

    phase = fr.phase_unwrapped_deg if unwrap else fr.phase_deg
    ax.plot(fr.freqs / div, phase, **IDEAL)
    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel("Phase (degrees)")
    kind = "unwrapped" if unwrap else "wrapped"
    ax.set_title(f"Phase response ({kind}) - {fd.spec.name}")
    _grid(ax)

    if fd.is_fir and fd.spec.response is not Response.HILBERT:
        ax.text(
            0.02,
            0.04,
            "Linear phase: constant group delay, no phase distortion.",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.75,
        )


def plot_group_delay(
    fig: Figure, fd: FilterDesign, num_points: int = 2048
) -> None:
    """Group delay in samples, with the passband marked."""
    ax = fig.add_subplot(111)
    freqs = np.linspace(0.0, fd.spec.nyquist, num_points)
    gd = analysis.group_delay(fd, freqs)
    div, unit = freq_scale(float(freqs[-1]))

    ax.plot(freqs / div, gd, **IDEAL)

    bands = analysis._mask_bands(fd)
    if bands is not None:
        for lo, hi in bands[0]:
            if hi > lo:
                ax.axvspan(lo / div, hi / div, color="tab:green", alpha=0.08)

    finite = gd[np.isfinite(gd)]
    if finite.size:
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        pad = max((hi - lo) * 0.2, 0.5)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel("Group delay (samples)")
    ax.set_title(f"Group delay - {fd.spec.name}")
    _grid(ax)


# --------------------------------------------------------------------------
# Time-domain plots
# --------------------------------------------------------------------------
def plot_impulse(
    fig: Figure, fd: FilterDesign, quantized: QuantizedFilter | None = None
) -> None:
    ax = fig.add_subplot(111)
    n, h = analysis.impulse_response(fd)

    if np.iscomplexobj(h):
        # matplotlib would quietly plot the real part alone, which for a
        # chirp looks like a plain modulated pulse and hides the quadrature
        # half entirely.
        ax.plot(n, np.real(h), linewidth=0.9, alpha=0.75, label="I")
        ax.plot(n, np.imag(h), linewidth=0.9, alpha=0.75, label="Q")
        ax.plot(n, np.abs(h), linewidth=1.5, color="k", alpha=0.7,
                label="envelope")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Impulse response - {fd.spec.name}")
        _grid(ax)
        return

    ax.stem(n, h, linefmt="tab:blue", markerfmt="o", basefmt=" ")
    for line in ax.get_lines():
        line.set_markersize(2.5)
        line.set_linewidth(0.9)

    if quantized is not None and quantized.usable:
        nq, hq = analysis.impulse_response(quantized.quantized_design)
        ax.plot(nq, hq, ".", color="tab:orange", markersize=3, label=quantized.fmt.name)
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Impulse response - {fd.spec.name}")
    _grid(ax)


def plot_step(fig: Figure, fd: FilterDesign) -> None:
    ax = fig.add_subplot(111)
    n, y = analysis.step_response(fd)

    if np.iscomplexobj(y):
        # Overshoot against a complex settling value is not a meaningful
        # number, so show the envelope and its components instead of
        # inventing one.
        ax.plot(n, np.real(y), linewidth=0.9, alpha=0.75, label="I")
        ax.plot(n, np.imag(y), linewidth=0.9, alpha=0.75, label="Q")
        ax.plot(n, np.abs(y), linewidth=1.4, color="k", alpha=0.7,
                label="envelope")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"Step response - {fd.spec.name}")
    else:
        ax.plot(n, y, **IDEAL)
        final = float(y[-1])
        ax.axhline(final, color="tab:grey", linewidth=0.8, linestyle="--")
        overshoot = (
            (float(np.max(y)) - final) / abs(final) * 100 if final else 0.0
        )
        ax.set_title(
            f"Step response - {fd.spec.name}  (overshoot {overshoot:.1f}%)"
        )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    _grid(ax)


def plot_pole_zero(fig: Figure, fd: FilterDesign) -> None:
    """Poles and zeros against the unit circle."""
    ax = fig.add_subplot(111)
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), color="tab:grey", linewidth=0.9)
    ax.axhline(0, color="tab:grey", linewidth=0.5, alpha=0.6)
    ax.axvline(0, color="tab:grey", linewidth=0.5, alpha=0.6)

    z, p = fd.zeros(), fd.poles()
    if z.size:
        ax.plot(z.real, z.imag, "o", markerfacecolor="none",
                markeredgecolor="tab:blue", markersize=7, label=f"{z.size} zeros")
    if p.size:
        ax.plot(p.real, p.imag, "x", color="tab:red", markersize=7,
                label=f"{p.size} poles")

    if fd.is_fir:
        note = "FIR: all poles sit at the origin, so it is always stable."
    else:
        radius = float(np.max(np.abs(p))) if p.size else 0.0
        note = (
            f"Largest pole radius {radius:.4f} - "
            + ("stable." if radius < 1.0 else "UNSTABLE: outside the unit circle.")
        )
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8, alpha=0.8)

    lim = 1.35
    if p.size or z.size:
        allpts = np.concatenate([np.abs(z), np.abs(p)]) if z.size and p.size else (
            np.abs(z) if z.size else np.abs(p)
        )
        lim = max(1.35, float(np.max(allpts)) * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(f"Pole-zero plot - {fd.spec.name}")
    if z.size or p.size:
        ax.legend(loc="upper right", fontsize=8)
    _grid(ax)


def plot_taps(
    fig: Figure, fd: FilterDesign, quantized: QuantizedFilter | None = None
) -> None:
    """Coefficient values, ideal against quantized."""
    if not fd.is_fir:
        ax = fig.add_subplot(111)
        assert fd.sos is not None
        im = ax.imshow(fd.sos, aspect="auto", cmap="RdBu_r",
                       vmin=-np.max(np.abs(fd.sos)), vmax=np.max(np.abs(fd.sos)))
        ax.set_xticks(range(6))
        ax.set_xticklabels(["b0", "b1", "b2", "a0", "a1", "a2"])
        ax.set_yticks(range(fd.num_sections))
        ax.set_ylabel("Section")
        ax.set_title(f"Second-order sections - {fd.spec.name}")
        fig.colorbar(im, ax=ax, label="Coefficient value")
        return

    if fd.is_complex:
        # Showing only the real part would look like an unmodulated pulse and
        # hide the chirp entirely; the envelope is where the weighting shows.
        ax = fig.add_subplot(111)
        ax.plot(np.real(fd.b), linewidth=0.9, alpha=0.75, label="I")
        ax.plot(np.imag(fd.b), linewidth=0.9, alpha=0.75, label="Q")
        ax.plot(
            np.abs(fd.b),
            linewidth=1.6,
            color="k",
            alpha=0.75,
            label="envelope (the weighting)",
        )
        ax.set_xlabel("Tap index")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"{fd.num_taps} complex taps - {fd.spec.name} "
            f"({fd.spec.window} weighting)"
        )
        _grid(ax)
        return

    if quantized is None or not quantized.usable:
        ax = fig.add_subplot(111)
        ax.plot(fd.b, ".-", markersize=3, **IDEAL)
        ax.set_xlabel("Tap index")
        ax.set_ylabel("Value")
        ax.set_title(f"{fd.num_taps} taps - {fd.spec.name}")
        _grid(ax)
        return

    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    top.plot(fd.b, ".-", markersize=3, label="ideal", **IDEAL)
    top.plot(quantized.taps, ".", markersize=3,
             label=quantized.fmt.name, color="tab:orange")
    top.set_ylabel("Value")
    top.legend(loc="best", fontsize=8)
    top.set_title(f"{fd.num_taps} taps - {fd.spec.name}")
    _grid(top)

    err = quantized.taps - fd.b
    bottom.stem(err, linefmt="tab:red", markerfmt=" ", basefmt=" ")
    bottom.axhline(quantized.fmt.resolution / 2, color="tab:grey",
                   linestyle="--", linewidth=0.8)
    bottom.axhline(-quantized.fmt.resolution / 2, color="tab:grey",
                   linestyle="--", linewidth=0.8)
    bottom.set_xlabel("Tap index")
    bottom.set_ylabel("Error")
    bottom.set_title("Rounding error (dashed lines: +/- half an LSB)", fontsize=9)
    _grid(bottom)


def plot_overview(
    fig: Figure, fd: FilterDesign, quantized: QuantizedFilter | None = None
) -> None:
    """Magnitude, phase, impulse and pole-zero in one figure."""
    axes = fig.subplots(2, 2)

    # Magnitude
    fr = analysis.frequency_response(fd, num_points=2048)
    div, unit = freq_scale(float(fr.freqs[-1]))
    axes[0, 0].plot(fr.freqs / div, fr.mag_db, **IDEAL)
    if quantized is not None and quantized.usable:
        frq = analysis.frequency_response(quantized.quantized_design, num_points=2048)
        axes[0, 0].plot(frq.freqs / div, frq.mag_db, **QUANT)
    axes[0, 0].set_title("Magnitude (dB)", fontsize=9)
    axes[0, 0].set_xlabel(unit, fontsize=8)
    axes[0, 0].set_ylim(max(-160.0, float(np.min(fr.mag_db)) - 5), float(np.max(fr.mag_db)) + 5)
    _grid(axes[0, 0])

    # Phase
    axes[0, 1].plot(fr.freqs / div, fr.phase_unwrapped_deg, **IDEAL)
    axes[0, 1].set_title("Phase (deg, unwrapped)", fontsize=9)
    axes[0, 1].set_xlabel(unit, fontsize=8)
    _grid(axes[0, 1])

    # Impulse. Complex taps are shown as their envelope: at overview size
    # there is no room for I and Q, and the envelope is the informative half.
    n, h = analysis.impulse_response(fd)
    if np.iscomplexobj(h):
        axes[1, 0].plot(n, np.abs(h), linewidth=1.0, color="tab:blue")
        axes[1, 0].set_title("Impulse response (envelope)", fontsize=9)
    else:
        axes[1, 0].plot(n, h, linewidth=1.0, color="tab:blue")
        axes[1, 0].set_title("Impulse response", fontsize=9)
    axes[1, 0].set_xlabel("sample", fontsize=8)
    _grid(axes[1, 0])

    # Pole-zero
    theta = np.linspace(0, 2 * np.pi, 256)
    axes[1, 1].plot(np.cos(theta), np.sin(theta), color="tab:grey", linewidth=0.8)
    z, p = fd.zeros(), fd.poles()
    if z.size:
        axes[1, 1].plot(z.real, z.imag, "o", markerfacecolor="none",
                        markeredgecolor="tab:blue", markersize=4)
    if p.size:
        axes[1, 1].plot(p.real, p.imag, "x", color="tab:red", markersize=4)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("Poles and zeros", fontsize=9)
    _grid(axes[1, 1])

    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    fig.suptitle(fd.spec.name, fontsize=11)


# --------------------------------------------------------------------------
# Pulse response
# --------------------------------------------------------------------------
def plot_pulse_time(
    fig: Figure,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    align_delay: float = 0.0,
    db: bool = False,
    dynamic_range_db: float = 80.0,
) -> None:
    """Input and filtered output against time.

    ``align_delay`` shifts the output back by the filter's group delay so the
    two line up; without it every comparison looks like a timing error.

    ``db`` switches to a logarithmic envelope view. That is not a cosmetic
    preference for radar: a target 40 dB below its neighbour is one hundredth
    of the height on a linear axis and simply cannot be seen, which is the
    entire question a compression filter exists to answer.
    """
    if db:
        _plot_pulse_time_db(fig, t, x, y, align_delay, dynamic_range_db)
        return

    div, unit = _time_scale(float(t[-1]) if t.size else 1.0)
    complex_signal = np.iscomplexobj(x) or np.iscomplexobj(y)

    if complex_signal:
        axes = fig.subplots(2, 1, sharex=True)
        axes[0].plot(t / div, np.real(x), linewidth=0.9, alpha=0.6, label="input I")
        axes[0].plot(t / div, np.imag(x), linewidth=0.9, alpha=0.6, label="input Q")
        axes[0].set_ylabel("Input")
        axes[0].legend(loc="upper right", fontsize=7)
        _grid(axes[0])

        ty = (t - align_delay) / div
        axes[1].plot(ty, np.real(y), linewidth=1.0, label="output I")
        axes[1].plot(ty, np.imag(y), linewidth=1.0, label="output Q")
        axes[1].plot(ty, np.abs(y), linewidth=1.2, color="k", alpha=0.5,
                     label="envelope")
        axes[1].set_ylabel("Output")
        axes[1].set_xlabel(f"Time ({unit})")
        axes[1].legend(loc="upper right", fontsize=7)
        _grid(axes[1])
    else:
        ax = fig.add_subplot(111)
        ax.plot(t / div, np.real(x), linewidth=0.9, alpha=0.55, label="input")
        ax.plot((t - align_delay) / div, np.real(y), linewidth=1.2, label="output")
        ax.set_xlabel(f"Time ({unit})")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        _grid(ax)

    title = "Pulse response"
    if align_delay:
        title += f"  (output shifted back {align_delay * 1e6:.3g} us to align)"
    fig.suptitle(title, fontsize=11)


def _plot_pulse_time_db(
    fig: Figure,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    align_delay: float,
    dynamic_range_db: float,
) -> None:
    """Envelope view in dB: the radar range profile."""
    div, unit = _time_scale(float(t[-1]) if t.size else 1.0)
    ax = fig.add_subplot(111)

    out = np.abs(np.asarray(y))
    peak = float(np.max(out)) if out.size else 0.0
    if peak <= 0:
        ax.text(0.5, 0.5, "No output", ha="center", va="center")
        ax.set_axis_off()
        return

    inp = np.abs(np.asarray(x))
    ax.plot(
        t / div,
        analysis.db20(inp / peak),
        linewidth=0.8,
        alpha=0.45,
        label="input envelope",
    )
    ax.plot(
        (t - align_delay) / div,
        analysis.db20(out / peak),
        linewidth=1.2,
        label="output envelope",
    )

    ax.set_ylim(-dynamic_range_db, 5.0)
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Envelope (dB, relative to the output peak)")
    ax.legend(loc="upper right", fontsize=8)
    title = "Pulse response"
    if align_delay:
        title += f"  (output shifted back {align_delay * 1e6:.3g} us to align)"
    ax.set_title(title, fontsize=11)
    _grid(ax)


def plot_pulse_spectrum(
    fig: Figure,
    freqs_in: np.ndarray,
    mag_in: np.ndarray,
    freqs_out: np.ndarray,
    mag_out: np.ndarray,
    fd: FilterDesign | None = None,
) -> None:
    """Input and output spectra, with the filter's own response overlaid."""
    ax = fig.add_subplot(111)
    span = float(np.max(np.abs(freqs_in))) if freqs_in.size else 1.0
    div, unit = freq_scale(span)

    ax.plot(freqs_in / div, mag_in, linewidth=0.9, alpha=0.55, label="input")
    ax.plot(freqs_out / div, mag_out, linewidth=1.1, label="output")

    if fd is not None:
        fr = analysis.frequency_response(fd, num_points=2048)
        f = fr.freqs
        m = fr.mag_db - float(np.max(fr.mag_db))
        if freqs_in.size and float(np.min(freqs_in)) < 0:
            # Mirror onto the negative axis for a two-sided (complex) view.
            f = np.concatenate([-f[::-1], f])
            m = np.concatenate([m[::-1], m])
        ax.plot(f / div, m, "--", color="tab:grey", linewidth=1.0,
                label="filter response")

    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel("Magnitude (dB, relative to peak)")
    ax.set_ylim(-140, 5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Pulse spectrum")
    _grid(ax)


# --------------------------------------------------------------------------
# Radar plots
# --------------------------------------------------------------------------
def plot_compressed_pulse(
    fig: Figure,
    fd: FilterDesign,
    span_cells: float = 40.0,
    show_metrics: bool = True,
) -> None:
    """The compressed pulse, in dB, against relative range.

    This is the plot a pulse compression design lives or dies by. The peak is
    the target; everything either side of it is range sidelobe, and anything
    weaker than the worst sidelobe is invisible no matter how long you
    integrate.
    """
    from .core import analysis
    from .core.radar import C_LIGHT

    ax = fig.add_subplot(111)
    # Interpolate to match what radar_metrics measures, or the annotated peak
    # sidelobe line would sit above a curve that never appears to reach it.
    lags, compressed = analysis.compressed_response(
        fd, oversample=analysis._measurement_oversample(fd.spec)
    )
    mag = np.abs(compressed)
    peak = float(np.max(mag))
    if peak <= 0:
        ax.text(0.5, 0.5, "No compressed response", ha="center", va="center")
        ax.set_axis_off()
        return
    mag_db = analysis.db20(mag / peak)

    # Decide the visible window first, then pick the unit from *that*. Scaling
    # to the full correlation instead would label a 300 m view in kilometres.
    spec = fd.spec
    half_m = None
    if spec.chirp_bandwidth_hz > 0:
        half_m = span_cells * C_LIGHT / (2.0 * spec.chirp_bandwidth_hz)

    ranges_m = lags * C_LIGHT / 2.0
    span = half_m if half_m else float(np.max(np.abs(ranges_m)) or 1.0)
    divisor, unit = (1000.0, "km") if span >= 1000.0 else (1.0, "m")

    ax.plot(ranges_m / divisor, mag_db, **IDEAL)
    if half_m:
        ax.set_xlim(-half_m / divisor, half_m / divisor)

    metrics = analysis.radar_metrics(fd)
    if show_metrics and np.isfinite(metrics.pslr_db):
        ax.axhline(
            metrics.pslr_db,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"peak sidelobe {metrics.pslr_db:.1f} dB",
        )
        ax.axhline(-3.0, color="tab:grey", linestyle=":", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=8)

    floor = -90.0
    if np.isfinite(metrics.pslr_db):
        floor = min(-90.0, metrics.pslr_db - 25.0)
    ax.set_ylim(max(floor, -140.0), 5.0)
    ax.set_xlabel(f"Range relative to the target ({unit})")
    ax.set_ylabel("Compressed amplitude (dB)")

    title = f"Compressed pulse - {spec.name}"
    if np.isfinite(metrics.mainlobe_3db_m):
        title += f"   resolution {metrics.mainlobe_3db_m:,.3g} m"
    ax.set_title(title)
    _grid(ax)


def plot_weighting(fig: Figure, fd: FilterDesign) -> None:
    """The weighting window itself, and what it does to the spectrum."""
    from .core import analysis
    from .core.radar import weighting_window

    spec = fd.spec
    n = fd.num_taps
    top, bottom = fig.subplots(2, 1)

    window = weighting_window(
        spec.window,
        n,
        spec.taylor_nbar,
        spec.taylor_sll_db,
        spec.cheb_atten_db,
        spec.window_param,
    )
    top.plot(window, **IDEAL)
    top.set_ylabel("Weight")
    top.set_xlabel("Tap index")
    top.set_title(f"{spec.window} weighting across {n} taps", fontsize=10)
    _grid(top)

    # The transform of the taper is what the compressed sidelobes look like.
    pad = 32
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(window, n * pad)))
    spectrum = analysis.db20(spectrum / np.max(spectrum))
    bins = (np.arange(spectrum.size) - spectrum.size // 2) / pad
    bottom.plot(bins, spectrum, **IDEAL)
    bottom.set_xlim(-20, 20)
    bottom.set_ylim(-120, 5)
    bottom.set_xlabel("Offset from the peak (resolution cells)")
    bottom.set_ylabel("Response (dB)")
    bottom.set_title("Weighting transform: the sidelobe pattern it produces", fontsize=10)

    if spec.window == "taylor":
        bottom.axhline(
            -abs(spec.taylor_sll_db),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"design level {-abs(spec.taylor_sll_db):.0f} dB",
        )
        bottom.legend(loc="upper right", fontsize=8)
    elif spec.window == "chebwin":
        bottom.axhline(
            -abs(spec.cheb_atten_db),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"design level {-abs(spec.cheb_atten_db):.0f} dB",
        )
        bottom.legend(loc="upper right", fontsize=8)
    _grid(bottom)


def plot_ambiguity(
    fig: Figure, fd: FilterDesign, num_doppler: int = 101, dynamic_range_db: float = 60.0
) -> None:
    """Range-Doppler ambiguity surface.

    The diagonal ridge of an LFM waveform is range-Doppler coupling: a moving
    target still compresses to a sharp peak, but at the wrong range. Reading
    the slope off this plot tells you how much range error a given closing
    speed produces.
    """
    from .core import analysis
    from .core.radar import C_LIGHT

    spec = fd.spec
    # Keep the surface cheap enough to redraw interactively.
    decimation = max(1, fd.num_taps // 400)
    delays, dopplers, grid = analysis.ambiguity_function(
        fd, num_doppler=num_doppler, delay_decimation=decimation
    )

    ranges = delays * C_LIGHT / 2.0
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(
        ranges,
        dopplers / 1e3,
        grid,
        cmap="viridis",
        vmin=-dynamic_range_db,
        vmax=0.0,
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label="Response (dB)")

    if spec.chirp_bandwidth_hz > 0:
        cell = C_LIGHT / (2.0 * spec.chirp_bandwidth_hz)
        ax.set_xlim(-40 * cell, 40 * cell)

    ax.set_xlabel("Range offset (m)")
    ax.set_ylabel("Doppler (kHz)")

    title = f"Ambiguity surface - {spec.name}"
    if spec.pulse_width_s > 0 and spec.chirp_bandwidth_hz > 0:
        # Range-Doppler coupling for an LFM: dR = -c * fd * tau / (2 * B).
        coupling = C_LIGHT * spec.pulse_width_s / (2.0 * spec.chirp_bandwidth_hz)
        per_ms = coupling * 2.0 / (C_LIGHT / spec.radar_carrier_hz)
        title += f"   coupling {per_ms:,.3g} m per m/s"
    ax.set_title(title, fontsize=10)


def plot_mti_velocity(fig: Figure, fd: FilterDesign) -> None:
    """MTI canceller response against radial velocity.

    The notch at zero is the clutter rejection you wanted. The notches at the
    blind speeds are the price: a target at one of those is removed just as
    thoroughly as the clutter.
    """
    from .core import analysis
    from .core.radar import blind_speed_ms

    spec = fd.spec
    ax = fig.add_subplot(111)
    velocities, response = analysis.mti_velocity_response(fd)
    peak = float(np.max(response))
    ax.plot(velocities, response - peak, **IDEAL)

    blind = blind_speed_ms(spec.pri_s, spec.radar_carrier_hz)
    n = 1
    while blind * n <= velocities[-1] if blind > 0 else False:
        ax.axvline(
            blind * n,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label="blind speed" if n == 1 else None,
        )
        n += 1
        if n > 8:
            break

    ax.axvline(0.0, color="tab:green", linestyle=":", linewidth=1.2)
    ax.text(
        0.01,
        0.05,
        f"Clutter notch at 0 m/s. Blind speeds every {blind:,.4g} m/s.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Response (dB)")
    ax.set_title(
        f"{spec.mti_pulses}-pulse MTI response - "
        f"{spec.prf_hz:,.6g} Hz PRF at {spec.radar_carrier_hz / 1e9:,.4g} GHz",
        fontsize=10,
    )
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)
    _grid(ax)
