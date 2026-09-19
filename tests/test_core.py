"""DSP correctness tests.

These check properties that must hold regardless of implementation -- a
root-raised-cosine pair has no ISI at symbol instants, a linear-phase FIR has
constant group delay, more coefficient bits cannot make a filter worse -- so
they keep their value if the internals are ever rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest

from filter_engine.core import analysis, firdes, signals
from filter_engine.core.design import DesignError, design
from filter_engine.core.quantize import FixedPointFormat, estimate_fpga, quantize
from filter_engine.core.spec import (
    FilterFamily,
    FirMethod,
    IirMethod,
    Response,
    FilterSpec,
    SpecError,
)
from filter_engine.presets import FILTER_PRESETS, FPGA_PLATFORMS, SDR_PLATFORMS


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------
def test_spec_json_round_trip():
    spec = FilterSpec(
        name="round_trip",
        family=FilterFamily.IIR,
        response=Response.BANDPASS,
        iir_method=IirMethod.ELLIP,
        sample_rate=61.44e6,
        f_low=1e6,
        f_high=4e6,
    )
    assert FilterSpec.from_json(spec.to_json()) == spec


def test_spec_rejects_unknown_fields():
    data = FilterSpec().to_dict()
    data["nonsense"] = 1
    with pytest.raises(SpecError, match="unknown spec fields"):
        FilterSpec.from_dict(data)


@pytest.mark.parametrize(
    "spec,message",
    [
        (FilterSpec(sample_rate=0), "Sample rate"),
        (FilterSpec(f_low=600e3), "Nyquist"),
        (FilterSpec(response=Response.BANDPASS, f_low=300e3, f_high=200e3), "above"),
        (FilterSpec(transition_width=0), "Transition width"),
        (FilterSpec(f_low=1e3, transition_width=50e3), "does not fit"),
        (FilterSpec(stopband_atten_db=0.05), "must exceed"),
        (FilterSpec(response=Response.RRC, symbol_rate=900e3), "[Ss]amples per symbol"),
        (FilterSpec(response=Response.RRC, rolloff=1.5), "Roll-off"),
        (FilterSpec(auto_order=False, num_taps=0), "at least 1"),
    ],
)
def test_invalid_specs_are_rejected_with_a_useful_message(spec, message):
    with pytest.raises(SpecError, match=message):
        spec.validate()


def test_bandpass_transition_bands_may_not_overlap():
    spec = FilterSpec(
        response=Response.BANDPASS, f_low=100e3, f_high=120e3, transition_width=40e3
    )
    with pytest.raises(SpecError, match="overlap"):
        spec.validate()


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
@pytest.mark.parametrize("response", [Response.LOWPASS, Response.HIGHPASS,
                                      Response.BANDPASS, Response.BANDSTOP])
@pytest.mark.parametrize("method", list(FirMethod))
def test_fir_designs_produce_finite_taps(response, method):
    fd = design(FilterSpec(response=response, fir_method=method))
    assert fd.is_fir
    assert fd.num_taps >= 3
    assert np.all(np.isfinite(fd.b))


@pytest.mark.parametrize("response", [Response.LOWPASS, Response.HIGHPASS,
                                      Response.BANDPASS, Response.BANDSTOP])
@pytest.mark.parametrize("method", list(IirMethod))
def test_iir_designs_are_stable(response, method):
    fd = design(
        FilterSpec(family=FilterFamily.IIR, response=response, iir_method=method)
    )
    assert not fd.is_fir
    assert fd.is_stable, f"{method.value} {response.value} came out unstable"
    assert fd.sos is not None and fd.sos.shape[1] == 6


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 8, 9])
def test_iir_order_is_reported_exactly(order):
    """An odd order pads its SOS array; the reported order must not drift."""
    fd = design(
        FilterSpec(family=FilterFamily.IIR, auto_order=False, order=order)
    )
    assert fd.order == order
    assert fd.poles().size == order


def test_equiripple_lowpass_meets_its_mask():
    fd = design(
        FilterSpec(fir_method=FirMethod.REMEZ, stopband_atten_db=70, num_taps=0,
                   transition_width=40e3)
    )
    measured = analysis.measure(fd)
    assert measured.meets_spec, measured.notes
    assert measured.stopband_atten_db >= 70 * 0.95


def test_kaiser_window_reaches_the_requested_stopband():
    for atten in (40, 60, 80, 100):
        fd = design(
            FilterSpec(fir_method=FirMethod.WINDOW, window="kaiser",
                       stopband_atten_db=atten)
        )
        measured = analysis.measure(fd)
        assert measured.stopband_atten_db >= atten - 2.0, (
            f"kaiser fell short at {atten} dB: {measured.stopband_atten_db:.1f}"
        )


@pytest.mark.parametrize("window", sorted(firdes.WINDOW_MAX_ATTEN_DB))
def test_each_window_reaches_its_documented_ceiling(window):
    """The per-window length rule must actually deliver the ceiling we publish.

    A single Kaiser-derived length is wrong here: a Blackman window's main
    lobe is far wider than a Hamming's, so the same tap count leaves its
    transition band unfinished and it measures *worse* despite better
    sidelobes.
    """
    ceiling = firdes.WINDOW_MAX_ATTEN_DB[window]
    fd = design(
        FilterSpec(fir_method=FirMethod.WINDOW, window=window,
                   stopband_atten_db=ceiling)
    )
    measured = analysis.measure(fd)
    assert measured.stopband_atten_db >= ceiling - 2.0, (
        f"{window} reached only {measured.stopband_atten_db:.1f} dB of its "
        f"documented {ceiling} dB"
    )


def test_a_window_that_cannot_reach_the_spec_says_so():
    fd = design(
        FilterSpec(fir_method=FirMethod.WINDOW, window="hamming",
                   stopband_atten_db=90)
    )
    assert any("tops out" in note for note in fd.notes)


def test_an_impossible_remez_design_fails_loudly_or_reports_missing_the_mask():
    """Remez either fails to converge or returns a filter that misses the mask.

    Both are acceptable; silently returning something that looks fine is not.
    Five taps cannot give 120 dB over a 1 kHz transition by any route.
    """
    spec = FilterSpec(
        fir_method=FirMethod.REMEZ,
        auto_order=False,
        num_taps=5,
        transition_width=1e3,
        stopband_atten_db=120,
    )
    try:
        fd = design(spec)
    except DesignError:
        return
    measured = analysis.measure(fd)
    assert measured.meets_spec is False


def test_pulse_shaping_responses_reject_an_iir_family():
    with pytest.raises(SpecError, match="no IIR form"):
        design(FilterSpec(family=FilterFamily.IIR, response=Response.HILBERT))


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------
def test_root_raised_cosine_pair_has_no_isi():
    """Two cascaded RRCs form a raised cosine, which is zero at every other
    symbol instant. That is the entire reason RRC is used."""
    sps, alpha = 8, 0.35
    h = firdes.root_raised_cosine(sps, alpha, 21, normalisation="energy")
    combined = np.convolve(h, h)
    centre = (combined.size - 1) // 2
    peak = combined[centre]
    # Sample at symbol instants either side of the peak, away from the tails
    # where truncation dominates.
    neighbours = combined[centre + sps : centre + 6 * sps : sps]
    assert np.all(np.abs(neighbours / peak) < 5e-3)


def test_raised_cosine_is_exactly_zero_at_symbol_instants():
    sps, alpha = 8, 0.35
    h = firdes.raised_cosine(sps, alpha, 21, normalisation="peak")
    centre = (h.size - 1) // 2
    assert h[centre] == pytest.approx(1.0)
    assert np.allclose(h[centre + sps :: sps], 0.0, atol=1e-12)


def test_zero_rolloff_degenerates_to_a_sinc():
    a = firdes.root_raised_cosine(4, 0.0, 11, normalisation="peak")
    b = firdes.raised_cosine(4, 0.0, 11, normalisation="peak")
    assert np.allclose(a, b)


@pytest.mark.parametrize("mode", ["sum", "energy", "peak"])
def test_normalisation_modes_do_what_they_say(mode):
    h = firdes.root_raised_cosine(4, 0.35, 11, gain=2.0, normalisation=mode)
    if mode == "sum":
        assert h.sum() == pytest.approx(2.0)
    elif mode == "energy":
        assert np.sqrt(np.dot(h, h)) == pytest.approx(2.0)
    else:
        assert np.max(np.abs(h)) == pytest.approx(2.0)


def test_gaussian_taps_are_symmetric_and_positive():
    h = firdes.gaussian(8, 0.3, 5)
    assert np.allclose(h, h[::-1])
    assert np.all(h > 0)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def test_linear_phase_fir_has_constant_group_delay():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ))
    freqs = np.linspace(0, fd.spec.f_low * 0.8, 256)
    gd = analysis.group_delay(fd, freqs)
    expected = (fd.num_taps - 1) / 2.0
    assert np.allclose(gd, expected, atol=1e-6)


def test_fir_impulse_response_is_the_tap_vector():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ))
    _n, h = analysis.impulse_response(fd)
    assert np.allclose(h, fd.b)


def test_lowpass_dc_gain_matches_the_requested_gain():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, gain=2.0))
    fr = analysis.frequency_response(fd, num_points=512)
    assert np.abs(fr.h[0]) == pytest.approx(2.0, rel=0.02)


def test_measured_cutoff_lands_near_the_requested_edge():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, f_low=100e3))
    measured = analysis.measure(fd)
    assert measured.cutoff_6db_hz
    assert measured.cutoff_6db_hz[0] == pytest.approx(100e3, rel=0.05)


def test_log_frequency_axis_skips_dc():
    fd = design(FilterSpec())
    fr = analysis.frequency_response(fd, num_points=128, log_spacing=True)
    assert fr.freqs[0] > 0
    assert np.all(np.diff(fr.freqs) > 0)


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------
def test_more_bits_never_makes_the_error_floor_worse():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, stopband_atten_db=90))
    floors = [quantize(fd, bits).error_floor_db for bits in (8, 12, 16, 20, 24)]
    assert floors == sorted(floors, reverse=True), floors


def test_error_floor_improves_by_roughly_six_db_per_bit():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, stopband_atten_db=100))
    a = quantize(fd, 12).error_floor_db
    b = quantize(fd, 20).error_floor_db
    per_bit = (a - b) / 8.0
    assert 5.0 < per_bit < 7.0, f"{per_bit:.2f} dB per bit"


def test_symmetric_taps_are_detected_and_halve_the_multipliers():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ))
    q = quantize(fd, 16)
    assert q.symmetric
    estimate = estimate_fpga(q, clock_hz=1e9, sample_rate=fd.sample_rate)
    assert estimate.effective_multipliers == (q.int_taps.size + 1) // 2


def test_quantization_at_too_few_bits_is_flagged_not_silently_wrong():
    fd = design(
        FilterSpec(family=FilterFamily.IIR, iir_method=IirMethod.ELLIP,
                   auto_order=False, order=8)
    )
    q = quantize(fd, 8)
    assert not q.usable
    assert np.isnan(q.error_floor_db)
    assert any("DEGENERATE" in note for note in q.notes)


def test_fixed_point_format_names_follow_the_q_convention():
    fmt = FixedPointFormat(16, 15)
    assert fmt.name == "Q1.15"
    assert fmt.max_int == 32767
    assert fmt.resolution == pytest.approx(1 / 32768)


def test_quantize_round_trips_integers_through_the_format():
    fmt = FixedPointFormat(16, 15)
    values = np.array([0.0, 0.5, -0.5, 0.999, -1.0])
    ints, _clipped = fmt.quantize(values)
    assert np.allclose(fmt.dequantize(ints), values, atol=fmt.resolution)


def test_a_slow_clock_is_reported_as_infeasible():
    fd = design(FilterSpec(fir_method=FirMethod.REMEZ, sample_rate=61.44e6))
    q = quantize(fd, 16)
    estimate = estimate_fpga(q, clock_hz=10e6, sample_rate=61.44e6)
    assert not estimate.feasible
    assert any("cannot keep up" in note for note in estimate.notes)


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", list(signals.PulseKind))
def test_every_excitation_generates_and_filters(kind):
    ps = signals.PulseSpec(
        kind=kind, sample_rate=1e6, duration_s=2e-3, width_s=5e-5, delay_s=3e-4
    )
    generated = signals.generate(ps)
    assert generated.x.size == ps.num_samples
    assert np.all(np.isfinite(generated.x))

    fd = design(FilterSpec(fir_method=FirMethod.REMEZ))
    y = signals.apply_filter(fd, generated.x)
    assert y.size == generated.x.size
    assert np.all(np.isfinite(y))


def test_a_carrier_offset_produces_complex_baseband_at_that_frequency():
    ps = signals.PulseSpec(
        kind=signals.PulseKind.TONE_BURST,
        sample_rate=1e6,
        duration_s=4e-3,
        width_s=3e-3,
        delay_s=2e-3,
        carrier_hz=150e3,
    )
    generated = signals.generate(ps)
    assert generated.is_complex
    freqs, mag = signals.spectrum(generated.x, ps.sample_rate, db=False)
    assert freqs[int(np.argmax(mag))] == pytest.approx(150e3, abs=2e3)


def test_real_signals_get_a_one_sided_spectrum_and_complex_a_two_sided_one():
    ps = signals.PulseSpec(kind=signals.PulseKind.RECT, sample_rate=1e6,
                           duration_s=1e-3)
    real = signals.generate(ps)
    freqs, _mag = signals.spectrum(real.x, ps.sample_rate)
    assert freqs[0] == 0.0

    complex_signal = signals.generate(ps.__class__(**{**ps.__dict__,
                                                     "force_complex": True}))
    freqs2, _mag2 = signals.spectrum(complex_signal.x, ps.sample_rate)
    assert freqs2[0] < 0 < freqs2[-1]


def test_noise_is_added_at_the_requested_snr():
    ps = signals.PulseSpec(kind=signals.PulseKind.TWO_TONE, sample_rate=1e6,
                           duration_s=20e-3, snr_db=20.0)
    clean = signals.generate(ps.__class__(**{**ps.__dict__, "snr_db": None}))
    noisy = signals.generate(ps)
    noise = noisy.x - clean.x
    measured = 10 * np.log10(np.mean(clean.x**2) / np.mean(noise**2))
    assert measured == pytest.approx(20.0, abs=0.5)


def test_an_over_long_record_is_refused_rather_than_exhausting_memory():
    ps = signals.PulseSpec(sample_rate=61.44e6, duration_s=10.0)
    with pytest.raises(SpecError, match="samples"):
        ps.validate()


# --------------------------------------------------------------------------
# Presets and platforms
# --------------------------------------------------------------------------
@pytest.mark.parametrize("preset", FILTER_PRESETS, ids=lambda p: p.name)
def test_every_preset_designs(preset):
    fd = design(preset.spec)
    assert fd.num_taps > 0 if fd.is_fir else fd.is_stable


def test_platform_limits_are_enforced():
    b210 = SDR_PLATFORMS["USRP B210"]
    assert b210.check_sample_rate(10e6) is None
    assert "exceeds" in b210.check_sample_rate(100e6)

    artix = FPGA_PLATFORMS["Artix-7 XC7A35T"]
    assert artix.check_usage(10) is None
    assert "exceeds" in artix.check_usage(500)
