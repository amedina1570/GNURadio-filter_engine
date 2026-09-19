"""Plain-language explanations of every response and every parameter.

Kept out of the GUI on purpose. A tooltip buried in a widget constructor is
invisible to anyone reading the code and impossible to test; here the help
text is data, sits next to the thing it describes, and a test can check that
every field the user can edit actually has an explanation.

The house style for these: say what the number *does*, then what happens when
you change it. "Transition width: how fast the filter cuts off. Halving it
doubles the tap count." beats "Sets the transition bandwidth."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Response

__all__ = ["ResponseInfo", "RESPONSE_INFO", "PARAM_HELP", "describe", "help_for"]


@dataclass(frozen=True)
class ResponseInfo:
    """What a response type is for, in the words a user would use."""

    label: str
    #: One line, shown as a banner above the parameters.
    summary: str
    #: Two or three sentences on when you would reach for it.
    detail: str
    #: The parameters that actually matter for this response, in the order
    #: they should be read.
    key_params: tuple[str, ...] = field(default_factory=tuple)
    #: Radar responses get their own group in the GUI.
    category: str = "General"


RESPONSE_INFO: dict[Response, ResponseInfo] = {
    Response.LOWPASS: ResponseInfo(
        "Low pass",
        "Keeps everything below the cutoff, removes everything above it.",
        "The workhorse. Used ahead of a decimator to stop out-of-band signals "
        "folding into the band you care about, and to pull one channel out of "
        "a wide capture.",
        ("f_low", "transition_width", "stopband_atten_db"),
    ),
    Response.HIGHPASS: ResponseInfo(
        "High pass",
        "Removes everything below the cutoff, keeps everything above it.",
        "Most often used to strip DC offset and LO leakage out of a "
        "direct-conversion receiver, where a large spike sits at zero "
        "frequency and swamps the converter.",
        ("f_low", "transition_width", "stopband_atten_db"),
    ),
    Response.BANDPASS: ResponseInfo(
        "Band pass",
        "Keeps a band between two edges, removes everything outside it.",
        "Selects one channel and rejects its neighbours. The stopband "
        "attenuation is what sets adjacent-channel rejection.",
        ("f_low", "f_high", "transition_width", "stopband_atten_db"),
    ),
    Response.BANDSTOP: ResponseInfo(
        "Band stop (notch)",
        "Removes a band between two edges, keeps everything outside it.",
        "Used to excise a known interferer -- a carrier, a spur, a jammer -- "
        "without disturbing the rest of the band.",
        ("f_low", "f_high", "transition_width", "stopband_atten_db"),
    ),
    Response.HILBERT: ResponseInfo(
        "Hilbert transformer",
        "Shifts every frequency by 90 degrees without changing its amplitude.",
        "Turns a real signal into an analytic one, which is how single-sideband "
        "modulation is generated and how an envelope is recovered. It has no "
        "response at DC or Nyquist, so it always needs an odd tap count.",
        ("f_low", "f_high", "num_taps"),
    ),
    Response.DIFFERENTIATOR: ResponseInfo(
        "Differentiator",
        "Gain rises in proportion to frequency: it differentiates the signal.",
        "Used in FM demodulation and in discriminators. Keep the band edge "
        "well below Nyquist -- a full-band differentiator is badly conditioned "
        "and the design will not converge.",
        ("f_high", "num_taps"),
    ),
    Response.RRC: ResponseInfo(
        "Root raised cosine",
        "Half of a Nyquist filter, to be used at both ends of a link.",
        "Put one in the transmitter and its twin in the receiver and the pair "
        "multiply out to a raised cosine, which has zero inter-symbol "
        "interference at the symbol instants. Use 'energy' normalisation so "
        "the matched pair has unit gain.",
        ("symbol_rate", "rolloff", "span_symbols", "normalisation"),
    ),
    Response.RC: ResponseInfo(
        "Raised cosine",
        "A complete Nyquist filter: zero inter-symbol interference on its own.",
        "Use this when all the shaping happens at one end. If both ends shape, "
        "you want the root raised cosine instead.",
        ("symbol_rate", "rolloff", "span_symbols"),
    ),
    Response.GAUSSIAN: ResponseInfo(
        "Gaussian",
        "A smooth pulse with no spectral sidelobes at all.",
        "The pre-modulation filter for GMSK and GFSK, including GSM and "
        "Bluetooth. A lower bandwidth-time product means a tighter spectrum "
        "but more inter-symbol interference.",
        ("symbol_rate", "bt", "span_symbols"),
    ),
    Response.MATCHED_LFM: ResponseInfo(
        "Pulse compression (LFM matched filter)",
        "Compresses a long chirped pulse into a short one, for range "
        "resolution without needing a short pulse.",
        "A radar wants a long pulse to put energy on target and a short one "
        "to resolve range. A chirp gets both: transmit a long pulse swept "
        "across a wide band, then compress it in the receiver by this matched "
        "filter. Resolution comes from the bandwidth, energy from the pulse "
        "length. The weighting sets how far range sidelobes fall -- the price "
        "is a slightly wider mainlobe and a fraction of a dB of SNR.",
        (
            "pulse_width_s",
            "chirp_bandwidth_hz",
            "window",
            "taylor_sll_db",
            "taylor_nbar",
        ),
        category="Radar",
    ),
    Response.MTI_CANCELLER: ResponseInfo(
        "MTI canceller",
        "Removes stationary clutter by subtracting one pulse from the next.",
        "Ground, sea and weather return is stationary or nearly so, and sits "
        "at zero Doppler. Subtracting successive pulses puts a deep null "
        "there and leaves moving targets behind. The cost is blind speeds: "
        "a target that happens to move a whole Doppler cycle between pulses "
        "looks stationary too, and is cancelled with the clutter.",
        ("pri_s", "mti_pulses", "radar_carrier_hz"),
        category="Radar",
    ),
}


#: Field name -> plain-language explanation. Used for tooltips and for the
#: parameter reference in the GUI's help panel.
PARAM_HELP: dict[str, str] = {
    # --- rates and band edges ---
    "sample_rate": (
        "How many samples per second the filter runs at. Everything else is "
        "measured against this: the highest frequency the filter can act on "
        "is half of it (the Nyquist frequency)."
    ),
    "f_low": (
        "For a low or high pass, the cutoff -- where the filter changes from "
        "passing to blocking. For a band filter, the lower of the two edges."
    ),
    "f_high": "The upper band edge, for band pass and band stop filters.",
    "transition_width": (
        "How sharply the filter cuts off: the gap between the end of the "
        "passband and the start of the stopband. This is the expensive knob "
        "-- halving it roughly doubles the number of taps."
    ),
    "passband_ripple_db": (
        "How much the gain is allowed to wobble across the passband. Smaller "
        "is flatter but costs taps. 0.1 dB is a common choice; below about "
        "0.01 dB you are usually paying for nothing."
    ),
    "stopband_atten_db": (
        "How far down unwanted signals are pushed. 60 dB means an interferer "
        "comes out a thousand times smaller in amplitude. This is what sets "
        "adjacent-channel rejection."
    ),
    "gain": (
        "Overall passband gain, as a plain multiplier rather than dB. Leave "
        "at 1 unless you need the filter to scale as well as filter."
    ),
    # --- order ---
    "auto_order": (
        "Let the tool pick the smallest filter that meets the ripple and "
        "attenuation you asked for. Turn this off to set the size yourself "
        "and see what performance that buys."
    ),
    "num_taps": (
        "How many coefficients the FIR has. More taps means a sharper "
        "transition and more delay -- the filter delays the signal by half "
        "the tap count -- and more multipliers in hardware."
    ),
    "order": (
        "How many poles the IIR has. An IIR reaches a given sharpness with "
        "far fewer coefficients than an FIR, but distorts phase and can go "
        "unstable once the coefficients are rounded."
    ),
    # --- window ---
    "window": (
        "The taper applied across the coefficients. It trades sidelobe level "
        "against how wide the transition is: a gentler taper cuts off faster "
        "but leaks more."
    ),
    "window_param": (
        "Kaiser beta. Higher pushes the stopband deeper and widens the "
        "transition. With order estimation on, it is derived from the "
        "stopband attenuation you asked for."
    ),
    # --- pulse shaping ---
    "symbol_rate": (
        "How many symbols per second the link carries. The sample rate "
        "divided by this gives samples per symbol, which must be at least 2."
    ),
    "rolloff": (
        "Excess bandwidth, from 0 to 1. Zero would be a perfect brick wall "
        "that cannot be built; 0.35 is the usual compromise. Lower is more "
        "spectrally efficient but needs tighter timing."
    ),
    "bt": (
        "Bandwidth-time product for the Gaussian pulse. Lower means a "
        "narrower spectrum and more inter-symbol interference. GSM uses 0.3, "
        "Bluetooth 0.5."
    ),
    "span_symbols": (
        "How many symbol periods the filter covers. Longer is a more faithful "
        "pulse shape at the cost of taps and delay; 8 to 12 is typical."
    ),
    "normalisation": (
        "How the taps are scaled. 'energy' is right for a matched "
        "transmit/receive pair; 'sum' gives unity gain at DC and matches GNU "
        "Radio's firdes."
    ),
    # --- radar weighting ---
    "taylor_sll_db": (
        "The sidelobe level you are designing for, in dB below the peak. "
        "35 dB is a common radar choice. Lower sidelobes cost mainlobe width "
        "-- that is, range resolution -- and a fraction of a dB of SNR."
    ),
    "taylor_nbar": (
        "How many sidelobes either side of the peak are held flat at the "
        "design level before they start falling away. 4 to 6 is the usual "
        "range: too few and the near sidelobes rise above the design level, "
        "too many and the taper develops spikes at its ends and loses "
        "efficiency."
    ),
    "cheb_atten_db": (
        "Dolph-Chebyshev sidelobe level. Every sidelobe sits at exactly this "
        "level, which gives the narrowest possible mainlobe for it -- but the "
        "far sidelobes never decay, so extended clutter integrates badly."
    ),
    # --- radar: pulse compression ---
    "pulse_width_s": (
        "How long the transmitted pulse lasts. Longer puts more energy on "
        "target, so it sets detection range. It does not set resolution -- "
        "that is what compression is for."
    ),
    "chirp_bandwidth_hz": (
        "How far the chirp sweeps in frequency across the pulse. This alone "
        "sets range resolution: 150 m per MHz, so 10 MHz gives 15 m. The "
        "sample rate must be at least this wide."
    ),
    "down_chirp": (
        "Sweep downwards in frequency instead of upwards. It flips which way "
        "a moving target's range error goes, which is how some radars resolve "
        "range-Doppler coupling by alternating pulses."
    ),
    # --- radar: timing and Doppler ---
    "pri_s": (
        "Pulse repetition interval: the time from one transmitted pulse to "
        "the next. It sets how far you can see unambiguously -- an echo "
        "arriving after the next pulse goes out is reported at the wrong "
        "range -- and how fast a target can go before its Doppler folds."
    ),
    "mti_pulses": (
        "How many pulses the canceller combines. Two is a single canceller "
        "with a narrow clutter notch; three and four dig deeper but widen the "
        "notch, so slow targets are lost along with the clutter."
    ),
    "radar_carrier_hz": (
        "Transmit carrier frequency. Used only to convert Doppler into "
        "velocity -- 10 GHz is X-band, 3 GHz is S-band, 1.3 GHz is L-band."
    ),
    # --- bookkeeping ---
    "name": "Used for the generated class, module and HDL entity names.",
    "notes": "Free text carried into the generated code's header comment.",
}


def describe(response: Response) -> ResponseInfo:
    """Explanation for ``response``, with a usable fallback."""
    return RESPONSE_INFO.get(
        response,
        ResponseInfo(response.value.replace("_", " ").title(), "", ""),
    )


def help_for(field_name: str) -> str:
    """Plain-language help for a spec field, or an empty string."""
    return PARAM_HELP.get(field_name, "")
