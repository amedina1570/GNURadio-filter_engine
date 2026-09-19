"""Target hardware profiles and ready-made filter specifications.

The point of these is to stop you guessing at the numbers that constrain a
design.  A USRP B2xx cannot stream 100 MS/s no matter what you type into a
sample-rate box, and an XC7A35T has 90 DSP slices, not 900 -- so the tool
carries those limits and checks designs against them.

Part numbers and limits below come from the Ettus B200/B210 and Xilinx
7-series datasheets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.spec import (
    FilterFamily,
    FirMethod,
    IirMethod,
    Response,
    FilterSpec,
)

__all__ = [
    "SdrPlatform",
    "FpgaPlatform",
    "SDR_PLATFORMS",
    "FPGA_PLATFORMS",
    "FILTER_PRESETS",
    "get_sdr",
    "get_fpga",
    "get_preset",
]


@dataclass(frozen=True)
class SdrPlatform:
    """Streaming and RF limits of an SDR front end."""

    name: str
    #: Highest host sample rate, in Hz, that the transport can sustain.
    max_sample_rate: float
    #: A sensible starting sample rate.
    default_sample_rate: float
    #: Analog RF bandwidth, in Hz.
    max_rf_bandwidth: float
    #: Tuning range, in Hz.
    freq_min: float
    freq_max: float
    #: Converter resolution, in bits.
    adc_bits: int
    num_channels: int
    notes: str = ""

    def check_sample_rate(self, fs: float) -> str | None:
        """Return a warning if ``fs`` is outside what this radio can do."""
        if fs > self.max_sample_rate:
            return (
                f"{fs / 1e6:,.6g} MS/s exceeds the {self.name}'s "
                f"{self.max_sample_rate / 1e6:,.6g} MS/s ceiling."
            )
        if fs > self.max_rf_bandwidth:
            return (
                f"{fs / 1e6:,.6g} MS/s is wider than the {self.name}'s "
                f"{self.max_rf_bandwidth / 1e6:,.6g} MHz analog bandwidth; the "
                "front end will roll off before Nyquist."
            )
        return None


@dataclass(frozen=True)
class FpgaPlatform:
    """Capacity of an FPGA target, for resource estimates."""

    name: str
    family: str
    #: DSP48E1 slices available on the part.
    dsp_slices: int
    #: 36 kbit block RAMs.
    block_rams: int
    logic_cells: int
    #: A realistic fabric clock for DSP logic, in Hz.
    typical_clock_hz: float
    notes: str = ""

    def check_usage(self, dsp_used: int) -> str | None:
        if dsp_used > self.dsp_slices:
            return (
                f"{dsp_used} DSP slices exceeds the {self.name}'s "
                f"{self.dsp_slices}. Fold the filter harder, raise the clock, "
                "or move to a larger part."
            )
        if dsp_used > 0.8 * self.dsp_slices:
            return (
                f"{dsp_used} of {self.dsp_slices} DSP slices ("
                f"{100 * dsp_used / self.dsp_slices:.0f}%) leaves little room "
                "for the rest of the design."
            )
        return None


# --------------------------------------------------------------------------
# SDR front ends
# --------------------------------------------------------------------------
#: The B2xx family all use an Analog Devices AD9361/AD9364 transceiver over
#: USB 3.0.  The sample-rate ceilings are what the USB link sustains, which is
#: lower than what the transceiver itself can clock.
SDR_PLATFORMS: dict[str, SdrPlatform] = {
    "USRP B200": SdrPlatform(
        name="USRP B200",
        max_sample_rate=61.44e6,
        default_sample_rate=10e6,
        max_rf_bandwidth=56e6,
        freq_min=70e6,
        freq_max=6e9,
        adc_bits=12,
        num_channels=1,
        notes="AD9364, 1x1, USB 3.0. 61.44 MS/s is the practical single-channel ceiling.",
    ),
    "USRP B210": SdrPlatform(
        name="USRP B210",
        max_sample_rate=61.44e6,
        default_sample_rate=10e6,
        max_rf_bandwidth=56e6,
        freq_min=70e6,
        freq_max=6e9,
        adc_bits=12,
        num_channels=2,
        notes=(
            "AD9361, 2x2, USB 3.0. 61.44 MS/s applies to one channel; running "
            "both halves it to about 30.72 MS/s."
        ),
    ),
    "USRP B205mini": SdrPlatform(
        name="USRP B205mini",
        max_sample_rate=56e6,
        default_sample_rate=10e6,
        max_rf_bandwidth=56e6,
        freq_min=70e6,
        freq_max=6e9,
        adc_bits=12,
        num_channels=1,
        notes="AD9364, 1x1, USB 3.0, Spartan-6 fabric.",
    ),
    "USRP B200mini": SdrPlatform(
        name="USRP B200mini",
        max_sample_rate=56e6,
        default_sample_rate=10e6,
        max_rf_bandwidth=56e6,
        freq_min=70e6,
        freq_max=6e9,
        adc_bits=12,
        num_channels=1,
        notes="AD9364, 1x1, USB 3.0.",
    ),
    "Generic 1 MS/s": SdrPlatform(
        name="Generic 1 MS/s",
        max_sample_rate=1e6,
        default_sample_rate=1e6,
        max_rf_bandwidth=1e6,
        freq_min=0.0,
        freq_max=6e9,
        adc_bits=16,
        num_channels=1,
        notes="A neutral starting point with no hardware limits worth enforcing.",
    ),
}


# --------------------------------------------------------------------------
# FPGA targets
# --------------------------------------------------------------------------
#: Artix-7 parts carry DSP48E1 slices (signed 25x18 multiply-accumulate).
FPGA_PLATFORMS: dict[str, FpgaPlatform] = {
    "Artix-7 XC7A35T": FpgaPlatform(
        name="Artix-7 XC7A35T",
        family="Artix-7",
        dsp_slices=90,
        block_rams=50,
        logic_cells=33_280,
        typical_clock_hz=100e6,
        notes="The part on an Arty A7-35T or Basys 3.",
    ),
    "Artix-7 XC7A50T": FpgaPlatform(
        name="Artix-7 XC7A50T",
        family="Artix-7",
        dsp_slices=120,
        block_rams=75,
        logic_cells=52_160,
        typical_clock_hz=100e6,
    ),
    "Artix-7 XC7A75T": FpgaPlatform(
        name="Artix-7 XC7A75T",
        family="Artix-7",
        dsp_slices=180,
        block_rams=105,
        logic_cells=75_520,
        typical_clock_hz=100e6,
    ),
    "Artix-7 XC7A100T": FpgaPlatform(
        name="Artix-7 XC7A100T",
        family="Artix-7",
        dsp_slices=240,
        block_rams=135,
        logic_cells=101_440,
        typical_clock_hz=100e6,
        notes="The part on an Arty A7-100T or Nexys 4 DDR.",
    ),
    "Artix-7 XC7A200T": FpgaPlatform(
        name="Artix-7 XC7A200T",
        family="Artix-7",
        dsp_slices=740,
        block_rams=365,
        logic_cells=215_360,
        typical_clock_hz=100e6,
        notes="The largest Artix-7; the USRP X310 uses a Kintex-7 instead.",
    ),
    "Spartan-6 XC6SLX75": FpgaPlatform(
        name="Spartan-6 XC6SLX75",
        family="Spartan-6",
        dsp_slices=132,
        block_rams=172,
        logic_cells=74_637,
        typical_clock_hz=100e6,
        notes="DSP48A1 slices (18x18) -- the fabric inside a USRP B200/B210.",
    ),
}


# --------------------------------------------------------------------------
# Starting-point filter specifications
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    """A named starting point, with a note on why it is shaped that way."""

    name: str
    description: str
    spec: FilterSpec
    tags: tuple[str, ...] = field(default_factory=tuple)


FILTER_PRESETS: tuple[Preset, ...] = (
    Preset(
        name="Narrowband channel filter (B210, 10 MS/s)",
        description=(
            "200 kHz channel out of a 10 MS/s I/Q stream. Equiripple so the "
            "stopband is uniformly deep, which is what adjacent-channel "
            "rejection needs."
        ),
        spec=FilterSpec(
            name="channel_filter",
            family=FilterFamily.FIR,
            response=Response.LOWPASS,
            fir_method=FirMethod.REMEZ,
            sample_rate=10e6,
            f_low=100e3,
            transition_width=50e3,
            passband_ripple_db=0.1,
            stopband_atten_db=80.0,
        ),
        tags=("sdr", "usrp"),
    ),
    Preset(
        name="RRC matched filter (1 Msym QPSK, 4 sps)",
        description=(
            "Root-raised-cosine pulse shaping at 0.35 roll-off. Energy "
            "normalised so a matched transmit/receive pair has unit gain."
        ),
        spec=FilterSpec(
            name="rrc_matched",
            family=FilterFamily.FIR,
            response=Response.RRC,
            sample_rate=4e6,
            symbol_rate=1e6,
            rolloff=0.35,
            span_symbols=11,
            normalisation="energy",
        ),
        tags=("sdr", "modem"),
    ),
    Preset(
        name="GMSK Gaussian shaping (BT 0.3)",
        description=(
            "The Gaussian pre-modulation filter used by GSM and most GMSK "
            "links, at a bandwidth-time product of 0.3."
        ),
        spec=FilterSpec(
            name="gmsk_gaussian",
            family=FilterFamily.FIR,
            response=Response.GAUSSIAN,
            sample_rate=1.083333e6,
            symbol_rate=270.833e3,
            bt=0.3,
            span_symbols=4,
        ),
        tags=("sdr", "modem"),
    ),
    Preset(
        name="Decimation anti-alias (16x, 61.44 MS/s)",
        description=(
            "Anti-aliasing filter ahead of a 16x decimator at the B2xx's top "
            "rate. The transition has to close before the new Nyquist at "
            "1.92 MHz."
        ),
        spec=FilterSpec(
            name="decim_antialias",
            family=FilterFamily.FIR,
            response=Response.LOWPASS,
            fir_method=FirMethod.REMEZ,
            sample_rate=61.44e6,
            f_low=1.6e6,
            transition_width=600e3,
            passband_ripple_db=0.05,
            stopband_atten_db=80.0,
        ),
        tags=("sdr", "usrp", "multirate"),
    ),
    Preset(
        name="FPGA-friendly lowpass (16-bit, Artix-7)",
        description=(
            "Sized so a folded symmetric FIR fits comfortably in an XC7A35T "
            "at 100 MHz. Keeps the stopband inside what 16-bit coefficients "
            "can actually deliver."
        ),
        spec=FilterSpec(
            name="fpga_lowpass",
            family=FilterFamily.FIR,
            response=Response.LOWPASS,
            fir_method=FirMethod.REMEZ,
            sample_rate=10e6,
            f_low=1e6,
            transition_width=400e3,
            passband_ripple_db=0.1,
            stopband_atten_db=65.0,
        ),
        tags=("fpga", "artix7"),
    ),
    Preset(
        name="DC-block highpass (IIR elliptic)",
        description=(
            "Removes the LO leakage and DC offset that every direct-conversion "
            "receiver produces. Elliptic keeps the order -- and so the DSP "
            "cost -- as low as possible."
        ),
        spec=FilterSpec(
            name="dc_block",
            family=FilterFamily.IIR,
            response=Response.HIGHPASS,
            iir_method=IirMethod.ELLIP,
            sample_rate=1e6,
            f_low=2e3,
            transition_width=3e3,
            passband_ripple_db=0.1,
            stopband_atten_db=60.0,
        ),
        tags=("sdr", "iir"),
    ),
    Preset(
        name="Audio de-emphasis band (300 Hz - 3.4 kHz)",
        description=(
            "Voice-band bandpass for the audio side of an FM or SSB receiver, "
            "at a 48 kHz audio rate."
        ),
        spec=FilterSpec(
            name="voice_band",
            family=FilterFamily.FIR,
            response=Response.BANDPASS,
            fir_method=FirMethod.REMEZ,
            sample_rate=48e3,
            f_low=300.0,
            f_high=3400.0,
            transition_width=200.0,
            passband_ripple_db=0.5,
            stopband_atten_db=60.0,
        ),
        tags=("audio",),
    ),
    # --- radar ---------------------------------------------------------------
    Preset(
        name="Pulse compression, Taylor 35 dB (X-band)",
        description=(
            "A 20 us chirp swept over 20 MHz, compressed 400:1 for 26 dB of "
            "processing gain and 7.5 m range resolution. Taylor weighting at "
            "35 dB is the standard radar starting point: sidelobes low enough "
            "that a strong target does not mask its neighbours, for about "
            "0.9 dB of SNR and a third more mainlobe width."
        ),
        spec=FilterSpec(
            name="pulse_compression",
            response=Response.MATCHED_LFM,
            sample_rate=25e6,
            pulse_width_s=20e-6,
            chirp_bandwidth_hz=20e6,
            pri_s=1e-3,
            radar_carrier_hz=10e9,
            window="taylor",
            taylor_sll_db=35.0,
            taylor_nbar=4,
        ),
        tags=("radar", "pulse_compression"),
    ),
    Preset(
        name="Pulse compression, unweighted (best resolution)",
        description=(
            "The same chirp with no weighting. The matched filter is optimal "
            "for SNR and gives the narrowest mainlobe, but -13 dB range "
            "sidelobes mean anything more than about 13 dB weaker than a "
            "nearby target is invisible. Compare its compressed pulse against "
            "the Taylor preset."
        ),
        spec=FilterSpec(
            name="pulse_compression_plain",
            response=Response.MATCHED_LFM,
            sample_rate=25e6,
            pulse_width_s=20e-6,
            chirp_bandwidth_hz=20e6,
            pri_s=1e-3,
            radar_carrier_hz=10e9,
            window="boxcar",
        ),
        tags=("radar", "pulse_compression"),
    ),
    Preset(
        name="High-resolution compression (1 m, Taylor 45 dB)",
        description=(
            "150 MHz of chirp for 1 m range resolution, weighted hard at "
            "45 dB. This is imaging-radar territory -- note how far the "
            "sample rate and the tap count have to climb."
        ),
        spec=FilterSpec(
            name="hires_compression",
            response=Response.MATCHED_LFM,
            sample_rate=200e6,
            pulse_width_s=10e-6,
            chirp_bandwidth_hz=150e6,
            pri_s=200e-6,
            radar_carrier_hz=10e9,
            window="taylor",
            taylor_sll_db=45.0,
            taylor_nbar=6,
        ),
        tags=("radar", "pulse_compression", "sar"),
    ),
    Preset(
        name="Two-pulse MTI canceller (X-band, 1 kHz PRF)",
        description=(
            "The simplest clutter canceller: subtract each pulse from the "
            "last. Deep null at zero Doppler, and blind speeds every 15 m/s "
            "at these settings. Look at the velocity response to see what it "
            "costs you."
        ),
        spec=FilterSpec(
            name="mti_2pulse",
            response=Response.MTI_CANCELLER,
            sample_rate=1000.0,
            pri_s=1e-3,
            mti_pulses=2,
            radar_carrier_hz=10e9,
        ),
        tags=("radar", "mti", "doppler"),
    ),
    Preset(
        name="Three-pulse MTI canceller (deeper notch)",
        description=(
            "Two cancellers in series. The clutter notch is far deeper and "
            "flatter, but it is also wider, so slow-moving targets go with "
            "the clutter. The blind speeds are unchanged -- only the PRF "
            "moves those."
        ),
        spec=FilterSpec(
            name="mti_3pulse",
            response=Response.MTI_CANCELLER,
            sample_rate=1000.0,
            pri_s=1e-3,
            mti_pulses=3,
            radar_carrier_hz=10e9,
        ),
        tags=("radar", "mti", "doppler"),
    ),
    Preset(
        name="Radar IF anti-alias (Taylor-weighted)",
        description=(
            "A conventional lowpass ahead of the ADC, but tapered with a "
            "Taylor window so its stopband behaviour matches the rest of the "
            "radar chain."
        ),
        spec=FilterSpec(
            name="radar_antialias",
            family=FilterFamily.FIR,
            response=Response.LOWPASS,
            fir_method=FirMethod.WINDOW,
            window="taylor",
            taylor_sll_db=60.0,
            taylor_nbar=5,
            sample_rate=100e6,
            f_low=20e6,
            transition_width=5e6,
            stopband_atten_db=60.0,
        ),
        tags=("radar", "sdr"),
    ),
    Preset(
        name="Hilbert transformer (SSB generation)",
        description=(
            "90-degree phase shifter for single-sideband modulation and for "
            "turning a real signal into an analytic one."
        ),
        spec=FilterSpec(
            name="hilbert_ssb",
            family=FilterFamily.FIR,
            response=Response.HILBERT,
            sample_rate=48e3,
            f_low=300.0,
            f_high=23_700.0,
            auto_order=False,
            num_taps=65,
        ),
        tags=("sdr", "audio"),
    ),
)


def get_sdr(name: str) -> SdrPlatform:
    try:
        return SDR_PLATFORMS[name]
    except KeyError:
        raise KeyError(
            f"unknown SDR platform {name!r}; known: {sorted(SDR_PLATFORMS)}"
        ) from None


def get_fpga(name: str) -> FpgaPlatform:
    try:
        return FPGA_PLATFORMS[name]
    except KeyError:
        raise KeyError(
            f"unknown FPGA platform {name!r}; known: {sorted(FPGA_PLATFORMS)}"
        ) from None


def get_preset(name: str) -> Preset:
    for preset in FILTER_PRESETS:
        if preset.name == name:
            return preset
    raise KeyError(
        f"unknown preset {name!r}; known: {[p.name for p in FILTER_PRESETS]}"
    )
