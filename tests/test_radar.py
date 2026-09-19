"""Radar correctness tests.

The numbers here are checked against published values rather than against
whatever this code happens to produce. A pulse compression tool that reports
sidelobes better than reality is worse than no tool at all, so the tests that
matter most are the ones pinning PSLR, mainlobe broadening and weighting loss
to the textbook figures.
"""

from __future__ import annotations

import numpy as np
import pytest

from filter_engine.core import analysis, radar, signals
from filter_engine.core.design import design
from filter_engine.core.explain import PARAM_HELP, RESPONSE_INFO, describe
from filter_engine.core.quantize import quantize
from filter_engine.core.spec import FilterSpec, Response, SpecError

# A realistic X-band chirp: 20 MHz over 20 us, sampled at 2x the bandwidth.
FS, TAU, BW = 40e6, 20e-6, 20e6


def lfm(window: str = "taylor", **kwargs) -> FilterSpec:
    return FilterSpec(
        name=f"lfm_{window}",
        response=Response.MATCHED_LFM,
        sample_rate=FS,
        pulse_width_s=TAU,
        chirp_bandwidth_hz=BW,
        pri_s=1e-3,
        window=window,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
def test_range_resolution_is_150_metres_per_megahertz():
    """The identity every radar engineer checks a tool against first."""
    assert radar.range_resolution_m(1e6) == pytest.approx(149.9, abs=0.2)
    assert radar.range_resolution_m(10e6) == pytest.approx(14.99, abs=0.02)
    assert radar.range_resolution_m(150e6) == pytest.approx(0.9993, abs=0.002)


def test_unambiguous_range_is_150_km_per_millisecond():
    assert radar.unambiguous_range_m(1e-3) == pytest.approx(149_896, rel=1e-3)


def test_blind_speed_matches_the_textbook_x_band_figure():
    """At X-band with a 1 kHz PRF the first blind speed is about 15 m/s."""
    assert radar.blind_speed_ms(1e-3, 10e9) == pytest.approx(14.99, abs=0.05)


def test_doppler_and_velocity_are_inverses():
    for velocity in (-300.0, 0.0, 12.5, 250.0):
        shift = radar.doppler_hz(velocity, 10e9)
        assert radar.velocity_ms(shift, 10e9) == pytest.approx(velocity)


# --------------------------------------------------------------------------
# Weighting windows
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sll", [20.0, 30.0, 35.0, 40.0, 50.0])
def test_taylor_window_hits_the_sidelobe_level_it_is_asked_for(sll):
    """The defining property of a Taylor taper: you name the level, you get it.

    Given an adequate ``nbar``. The parameters are coupled -- a deeper level
    needs more flat sidelobes -- which is why the designer derives nbar by
    default instead of leaving it fixed.
    """
    nbar = radar.minimum_taylor_nbar(sll)
    window = radar.weighting_window("taylor", 257, taylor_nbar=nbar, taylor_sll_db=sll)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(window, window.size * 64)))
    spectrum = spectrum / spectrum.max()

    peak = int(np.argmax(spectrum))
    i = peak
    while i + 1 < spectrum.size and spectrum[i + 1] < spectrum[i]:
        i += 1
    measured = 20 * np.log10(spectrum[i:].max())
    assert measured == pytest.approx(-sll, abs=1.5), (
        f"asked for {-sll} dB, measured {measured:.2f} dB"
    )


@pytest.mark.parametrize("atten", [40.0, 60.0, 80.0])
def test_chebyshev_sidelobes_are_exactly_equal_ripple(atten):
    window = radar.weighting_window("chebwin", 257, cheb_atten_db=atten)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(window, window.size * 64)))
    spectrum = spectrum / spectrum.max()
    peak = int(np.argmax(spectrum))
    i = peak
    while i + 1 < spectrum.size and spectrum[i + 1] < spectrum[i]:
        i += 1
    assert 20 * np.log10(spectrum[i:].max()) == pytest.approx(-atten, abs=0.5)


def test_low_attenuation_chebyshev_does_not_leak_a_scipy_warning():
    """scipy warns below 45 dB; the window is still well defined."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        window = radar.weighting_window("chebwin", 64, cheb_atten_db=30.0)
    assert np.all(np.isfinite(window))


# --------------------------------------------------------------------------
# Pulse compression
# --------------------------------------------------------------------------
def test_matched_filter_taps_are_complex_and_unit_energy():
    fd = design(lfm())
    assert fd.is_complex
    assert np.sum(np.abs(fd.b) ** 2) == pytest.approx(1.0)


def test_matched_filter_length_is_the_pulse_length():
    fd = design(lfm())
    assert fd.num_taps == pytest.approx(TAU * FS, abs=1)


#: Published figures for an LFM compressed with each taper: peak sidelobe,
#: mainlobe broadening against no weighting, and mismatch loss.
WEIGHTING_REFERENCE = {
    "boxcar": (-13.3, 1.00, 0.00),
    "hann": (-31.5, 1.65, 1.76),
    "hamming": (-42.7, 1.47, 1.34),
    "blackman": (-58.1, 1.90, 2.37),
}


@pytest.mark.parametrize("window", sorted(WEIGHTING_REFERENCE))
def test_compressed_sidelobes_match_published_figures(window):
    """Published figures are asymptotic, so they need a large enough chirp.

    At a time-bandwidth product of 1000 the ripple at the edges of the chirp
    spectrum has stopped dominating; Blackman measures -53 dB at TBP 400 and
    only reaches its nominal -58 dB beyond about 1000.
    """
    expected_pslr, expected_broadening, expected_loss = WEIGHTING_REFERENCE[window]
    spec = lfm(window).copy(chirp_bandwidth_hz=100e6, sample_rate=200e6)
    assert spec.time_bandwidth_product >= 1000
    metrics = analysis.radar_metrics(design(spec))

    assert metrics.pslr_db == pytest.approx(expected_pslr, abs=2.0)
    assert metrics.broadening == pytest.approx(expected_broadening, abs=0.08)
    assert metrics.weighting_loss_db == pytest.approx(expected_loss, abs=0.15)


def test_a_low_time_bandwidth_product_limits_the_achievable_sidelobes():
    """No taper can beat the chirp's own spectral ripple."""
    low = analysis.radar_metrics(
        design(lfm("blackman").copy(chirp_bandwidth_hz=5e6, sample_rate=20e6))
    )
    high = analysis.radar_metrics(
        design(lfm("blackman").copy(chirp_bandwidth_hz=100e6, sample_rate=200e6))
    )
    assert low.pslr_db > high.pslr_db + 8.0
    assert high.pslr_db == pytest.approx(-58.1, abs=2.0)


# --------------------------------------------------------------------------
# Taylor's nbar / sidelobe-level coupling
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sll,expected_minimum", [(20.0, 3), (30.0, 4), (40.0, 7), (50.0, 9)]
)
def test_recommended_nbar_follows_the_standard_rule(sll, expected_minimum):
    assert radar.minimum_taylor_nbar(sll) == expected_minimum


def test_nbar_is_derived_from_the_sidelobe_level_by_default():
    spec = lfm("taylor", taylor_sll_db=50.0)
    assert spec.auto_taylor_nbar
    assert spec.effective_taylor_nbar == radar.minimum_taylor_nbar(50.0)


def test_deriving_nbar_is_what_makes_a_deep_taylor_design_work():
    """The failure this default exists to prevent."""
    wide = lfm("taylor", taylor_sll_db=50.0).copy(
        chirp_bandwidth_hz=100e6, sample_rate=200e6
    )
    derived = analysis.radar_metrics(design(wide))
    fixed = analysis.radar_metrics(
        design(wide.copy(auto_taylor_nbar=False, taylor_nbar=4))
    )

    assert derived.pslr_db == pytest.approx(-50.0, abs=2.0)
    assert fixed.pslr_db > derived.pslr_db + 3.0, (
        "too small an nbar must visibly miss the requested level"
    )
    assert any("nbar" in note for note in fixed.notes), (
        "and the tool must say why"
    )


@pytest.mark.parametrize("sll", [25.0, 35.0, 45.0])
def test_taylor_weighted_compression_reaches_its_design_level(sll):
    """The nominal sidelobe level must survive the whole compression chain."""
    metrics = analysis.radar_metrics(design(lfm("taylor", taylor_sll_db=sll)))
    assert metrics.pslr_db == pytest.approx(-sll, abs=3.0)


def test_lower_sidelobes_always_cost_resolution_and_snr():
    """The central trade. Neither can improve without the other getting worse."""
    results = [
        analysis.radar_metrics(design(lfm("taylor", taylor_sll_db=sll)))
        for sll in (25.0, 35.0, 45.0, 55.0)
    ]
    pslr = [m.pslr_db for m in results]
    width = [m.mainlobe_3db_m for m in results]
    loss = [m.weighting_loss_db for m in results]

    assert pslr == sorted(pslr, reverse=True), "sidelobes must fall"
    assert width == sorted(width), "mainlobe must widen"
    assert loss == sorted(loss), "SNR loss must grow"


def test_unweighted_matched_filter_has_the_best_resolution_and_no_loss():
    plain = analysis.radar_metrics(design(lfm("boxcar")))
    taylored = analysis.radar_metrics(design(lfm("taylor")))
    assert plain.mainlobe_3db_m < taylored.mainlobe_3db_m
    assert plain.weighting_loss_db == pytest.approx(0.0, abs=0.01)


def test_processing_gain_is_the_time_bandwidth_product():
    metrics = analysis.radar_metrics(design(lfm()))
    assert metrics.time_bandwidth_product == pytest.approx(TAU * BW)
    assert metrics.processing_gain_db == pytest.approx(
        10 * np.log10(TAU * BW), abs=0.01
    )


@pytest.mark.parametrize("oversample_ratio", [1.25, 2.0, 5.0, 10.0])
def test_measured_sidelobes_do_not_depend_on_the_sample_rate(oversample_ratio):
    """Radar samples barely above the chirp bandwidth.

    On that grid the compressed mainlobe is about one sample wide, so a naive
    measurement steps over the sidelobe peaks and reports a figure 10 dB too
    good. The measurement interpolates to avoid exactly that.
    """
    spec = lfm("boxcar").copy(sample_rate=BW * oversample_ratio)
    metrics = analysis.radar_metrics(design(spec))
    assert metrics.pslr_db == pytest.approx(-13.3, abs=0.6)


def test_compression_is_measured_against_the_transmit_pulse_not_the_filter():
    """Autocorrelating the filter applies the taper twice and flatters it."""
    fd = design(lfm("taylor", taylor_sll_db=35.0))
    honest = analysis.radar_metrics(fd).pslr_db

    # What the wrong method would have reported.
    taps = fd.b
    doubled = np.convolve(taps, np.conj(taps[::-1]))
    mag = np.abs(doubled) / np.max(np.abs(doubled))
    peak = int(np.argmax(mag))
    i = peak
    while i + 1 < mag.size and mag[i + 1] < mag[i]:
        i += 1
    flattered = 20 * np.log10(mag[i:].max())

    assert honest == pytest.approx(-35.0, abs=3.0)
    assert flattered < honest - 5.0, (
        "the double-weighted figure should be visibly better, which is why it "
        "must not be the one reported"
    )


def test_down_chirp_mirrors_the_up_chirp():
    up = design(lfm("taylor"))
    down = design(lfm("taylor").copy(down_chirp=True))
    assert np.allclose(up.b, np.conj(down.b))


def test_ambiguity_surface_peaks_at_the_origin():
    delays, dopplers, grid = analysis.ambiguity_function(
        design(lfm()), num_doppler=21, delay_decimation=4
    )
    row, col = np.unravel_index(int(np.argmax(grid)), grid.shape)
    assert dopplers[row] == pytest.approx(0.0, abs=abs(dopplers[1] - dopplers[0]))
    assert abs(delays[col]) <= 4 / FS * 2


# --------------------------------------------------------------------------
# MTI
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pulses,expected",
    [(2, [1, -1]), (3, [1, -2, 1]), (4, [1, -3, 3, -1]), (5, [1, -4, 6, -4, 1])],
)
def test_mti_coefficients_are_the_binomial_row(pulses, expected):
    assert np.allclose(radar.mti_canceller(pulses, normalise=False), expected)


@pytest.mark.parametrize("pulses", [2, 3, 4, 5])
def test_mti_rejects_stationary_clutter_completely(pulses):
    """Zero Doppler must be a null: that is the entire purpose."""
    taps = radar.mti_canceller(pulses)
    assert abs(np.sum(taps)) < 1e-12


@pytest.mark.parametrize("pulses", [2, 3, 4])
def test_more_pulses_deepen_the_clutter_notch(pulses):
    """An N-pulse canceller has an (N-1)-order null, so it falls off faster."""
    spec = FilterSpec(
        name="mti",
        response=Response.MTI_CANCELLER,
        sample_rate=1000.0,
        pri_s=1e-3,
        mti_pulses=pulses,
    )
    fd = design(spec)
    # Response a little off zero Doppler, normalised to the peak.
    fr = analysis.frequency_response(fd, num_points=2048)
    normalised = fr.mag_db - np.max(fr.mag_db)
    near_dc = normalised[np.argmin(np.abs(fr.freqs - 10.0))]
    # Order n-1 means 20*(n-1) dB per decade of Doppler.
    assert near_dc < -14.0 * (pulses - 1)


def test_mti_velocity_response_shows_a_blind_speed():
    spec = FilterSpec(
        name="mti",
        response=Response.MTI_CANCELLER,
        sample_rate=1000.0,
        pri_s=1e-3,
        mti_pulses=2,
        radar_carrier_hz=10e9,
    )
    fd = design(spec)
    velocities, response = analysis.mti_velocity_response(fd)
    blind = radar.blind_speed_ms(spec.pri_s, spec.radar_carrier_hz)

    index = int(np.argmin(np.abs(velocities - blind)))
    assert response[index] - np.max(response) < -40.0, (
        "a target at the blind speed must be cancelled like clutter"
    )


def test_mti_sample_rate_is_the_prf():
    spec = FilterSpec(
        response=Response.MTI_CANCELLER, sample_rate=2000.0, pri_s=5e-4
    )
    assert spec.prf_hz == pytest.approx(2000.0)


# --------------------------------------------------------------------------
# Two-target scenario
# --------------------------------------------------------------------------
def test_weighting_decides_whether_a_weak_target_is_visible():
    """The reason Taylor weighting exists, as an executable statement.

    A target 40 dB below a nearby strong one is recoverable only if the
    compression sidelobes at that range sit well below it.
    """
    ps = signals.PulseSpec(
        kind=signals.PulseKind.TWO_TARGETS,
        sample_rate=FS,
        duration_s=60e-6,
        width_s=TAU,
        delay_s=15e-6,
        lfm_bandwidth_hz=BW,
        target_separation_s=1.5e-6,
        target2_relative_db=-40.0,
    )
    both = signals.generate(ps)
    alone = signals.generate(ps.__class__(**{**ps.__dict__,
                                             "kind": signals.PulseKind.LFM_PULSE}))
    offset = int(round(ps.target_separation_s * FS))

    def level(x, fd):
        y = np.abs(signals.apply_filter(fd, x))
        peak = int(np.argmax(y))
        return 20 * np.log10(max(y[peak + offset] / y[peak], 1e-12))

    plain = design(lfm("boxcar"))
    weighted = design(lfm("taylor", taylor_sll_db=45.0))

    # Unweighted: the sidelobe at that range is comparable to the target, so
    # the reading is meaningless.
    assert level(alone.x, plain) > -46.0

    # Weighted: the sidelobe is far below the target, so it reads its true
    # strength.
    assert level(alone.x, weighted) < -50.0
    assert level(both.x, weighted) == pytest.approx(-40.0, abs=2.0)


# --------------------------------------------------------------------------
# Specification and validation
# --------------------------------------------------------------------------
def test_chirp_wider_than_the_sample_rate_is_refused():
    with pytest.raises(SpecError, match="does not fit"):
        lfm().copy(chirp_bandwidth_hz=FS * 2).validate()


def test_pulse_longer_than_the_pri_is_refused():
    with pytest.raises(SpecError, match="never switch off"):
        lfm().copy(pri_s=TAU / 2).validate()


def test_a_pointless_time_bandwidth_product_is_refused():
    with pytest.raises(SpecError, match="time-bandwidth"):
        lfm().copy(pulse_width_s=1e-7, chirp_bandwidth_hz=1e6).validate()


def test_too_many_mti_pulses_is_refused():
    spec = FilterSpec(response=Response.MTI_CANCELLER, mti_pulses=9)
    with pytest.raises(SpecError, match="5 pulses|not offered"):
        spec.validate()


def test_radar_spec_round_trips_through_json():
    spec = lfm("taylor", taylor_sll_db=42.0, taylor_nbar=6)
    assert FilterSpec.from_json(spec.to_json()) == spec


# --------------------------------------------------------------------------
# Fixed point and code generation
# --------------------------------------------------------------------------
def test_complex_taps_quantize_into_separate_i_and_q_words():
    q = quantize(design(lfm()), 16)
    assert q.is_complex
    assert q.int_taps_i.size == q.int_taps_q.size == design(lfm()).num_taps
    assert q.int_taps_i.dtype.kind == "i"
    assert np.max(np.abs(q.int_taps_i)) <= q.fmt.max_int


def test_complex_quantization_improves_with_word_length():
    floors = [quantize(design(lfm()), bits).error_floor_db for bits in (8, 12, 16, 20)]
    assert floors == sorted(floors, reverse=True)


def test_complex_filter_costs_four_multiplies_per_tap():
    from filter_engine.core.quantize import estimate_fpga

    fd = design(lfm())
    q = quantize(fd, 16)
    estimate = estimate_fpga(q, clock_hz=1e9, sample_rate=FS)
    assert estimate.effective_multipliers == 4 * fd.num_taps
    assert any("four real multiplies" in note for note in estimate.notes)


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def test_every_response_has_an_explanation():
    for response in Response:
        info = describe(response)
        assert info.label and info.summary, f"{response.value} has no explanation"
        assert response in RESPONSE_INFO, f"{response.value} missing from RESPONSE_INFO"


def test_every_editable_spec_field_has_help_text():
    """A parameter the user can change but not understand is a bug."""
    skip = {"metadata", "family", "response", "fir_method", "iir_method"}
    missing = [
        field
        for field in FilterSpec.__dataclass_fields__
        if field not in skip and field not in PARAM_HELP
    ]
    assert not missing, f"no help text for: {sorted(missing)}"


def test_radar_responses_are_categorised_as_radar():
    assert RESPONSE_INFO[Response.MATCHED_LFM].category == "Radar"
    assert RESPONSE_INFO[Response.MTI_CANCELLER].category == "Radar"
