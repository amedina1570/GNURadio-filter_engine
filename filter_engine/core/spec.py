"""Filter specification model.

A :class:`FilterSpec` is a pure-data description of *what* filter the user
wants.  It carries no design results -- :mod:`filter_engine.core.design`
turns a spec into coefficients.  Keeping the two apart means a spec can be
serialised, diffed, saved in a project file and round-tripped through the
generated code without dragging numpy arrays around.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


class FilterFamily(str, Enum):
    """Top-level implementation family."""

    FIR = "fir"
    IIR = "iir"


class Response(str, Enum):
    """Requested magnitude response shape."""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    HILBERT = "hilbert"
    DIFFERENTIATOR = "differentiator"
    RRC = "root_raised_cosine"
    RC = "raised_cosine"
    GAUSSIAN = "gaussian"
    #: Pulse compression: the matched filter for a linear-FM (chirp) pulse.
    MATCHED_LFM = "matched_lfm"
    #: Moving target indication: an N-pulse canceller running in slow time.
    MTI_CANCELLER = "mti_canceller"

    @property
    def is_pulse_shaping(self) -> bool:
        return self in (Response.RRC, Response.RC, Response.GAUSSIAN)

    @property
    def is_radar(self) -> bool:
        """True for the responses defined by radar parameters, not band edges."""
        return self in (Response.MATCHED_LFM, Response.MTI_CANCELLER)

    @property
    def is_complex(self) -> bool:
        """True when the design has complex (I/Q) coefficients."""
        return self is Response.MATCHED_LFM

    @property
    def is_multiband(self) -> bool:
        """True when the response needs both a low and a high band edge."""
        return self in (Response.BANDPASS, Response.BANDSTOP, Response.HILBERT)


class FirMethod(str, Enum):
    """FIR design algorithms, all backed by :mod:`scipy.signal`."""

    WINDOW = "window"                 # firwin  -- windowed sinc
    REMEZ = "remez"                   # remez   -- Parks-McClellan equiripple
    FIRLS = "firls"                   # firls   -- weighted least squares
    FREQ_SAMPLING = "freq_sampling"   # firwin2 -- frequency sampling


class IirMethod(str, Enum):
    """IIR prototype families, all backed by :mod:`scipy.signal`."""

    BUTTER = "butter"
    CHEBY1 = "cheby1"
    CHEBY2 = "cheby2"
    ELLIP = "ellip"
    BESSEL = "bessel"


#: Windows accepted by the FIR ``window`` method.  ``kaiser`` is parameterised
#: by :attr:`FilterSpec.window_param` (beta); the rest take no parameter.
WINDOWS: tuple[str, ...] = (
    "hamming",
    "hann",
    "blackman",
    "blackmanharris",
    "bartlett",
    "boxcar",
    "nuttall",
    "flattop",
    "kaiser",
    # Radar weightings. Both are parameterised: you name the sidelobe level
    # rather than inheriting whatever the window happens to give.
    "taylor",
    "chebwin",
)

#: Windows whose sidelobe level is set by their own parameters rather than
#: fixed by their shape.  These can reach any attenuation you ask for.
PARAMETERISED_WINDOWS: frozenset[str] = frozenset({"kaiser", "taylor", "chebwin"})

#: Normalisation applied to pulse-shaping taps.
NORMALISATIONS: tuple[str, ...] = ("sum", "energy", "peak", "none")


class SpecError(ValueError):
    """Raised when a spec cannot describe a realisable filter."""


@dataclass
class FilterSpec:
    """Everything needed to design one filter.

    Frequencies are in Hz and are always interpreted against
    :attr:`sample_rate`; nothing in the public API uses normalised frequency,
    because SDR and FPGA users think in Hz.
    """

    # --- what kind of filter -------------------------------------------------
    family: FilterFamily = FilterFamily.FIR
    response: Response = Response.LOWPASS
    fir_method: FirMethod = FirMethod.WINDOW
    iir_method: IirMethod = IirMethod.BUTTER

    # --- rates and band edges ------------------------------------------------
    sample_rate: float = 1_000_000.0
    #: Lower passband edge.  For lowpass/highpass this is *the* cutoff.
    f_low: float = 100_000.0
    #: Upper passband edge.  Ignored for lowpass/highpass.
    f_high: float = 200_000.0
    #: Width of the transition band, in Hz.
    transition_width: float = 25_000.0

    # --- tolerances ----------------------------------------------------------
    passband_ripple_db: float = 0.1
    stopband_atten_db: float = 60.0

    # --- order control -------------------------------------------------------
    #: When True the order / tap count is estimated from the tolerances.
    auto_order: bool = True
    #: FIR tap count, used when ``auto_order`` is False.
    num_taps: int = 65
    #: IIR order, used when ``auto_order`` is False.
    order: int = 4

    # --- FIR window options --------------------------------------------------
    #: Kaiser by default: it is the only fixed window whose stopband is not
    #: capped by its own sidelobes, so the default design actually meets the
    #: default tolerances. A Hamming window tops out near 53 dB and would
    #: miss the 60 dB below however many taps it were given.
    window: str = "kaiser"
    #: Kaiser beta.  Only read when :attr:`window` is ``"kaiser"`` and
    #: :attr:`auto_order` is off; otherwise it is derived from the stopband.
    window_param: float = 8.6

    # --- radar weighting parameters ------------------------------------------
    #: Taylor: how many sidelobes either side of the mainlobe are held at the
    #: design level before they start falling away.  More gives a flatter,
    #: better-controlled near-in sidelobe region; too many and the taper
    #: develops end spikes and loses efficiency.  4 to 6 is the usual choice.
    taylor_nbar: int = 4
    #: Derive :attr:`taylor_nbar` from :attr:`taylor_sll_db` instead of using
    #: the value above.  On by default: nbar has to grow with the sidelobe
    #: depth you ask for, and getting it wrong silently costs several dB.
    auto_taylor_nbar: bool = True
    #: Taylor: the design sidelobe level, in dB below the mainlobe peak.
    taylor_sll_db: float = 35.0
    #: Dolph-Chebyshev: every sidelobe sits exactly this far below the peak.
    cheb_atten_db: float = 60.0

    # --- radar: pulse compression --------------------------------------------
    #: Transmitted pulse width (before compression), in seconds.
    pulse_width_s: float = 10e-6
    #: Bandwidth the chirp sweeps across the pulse, in Hz.  This alone sets
    #: range resolution.
    chirp_bandwidth_hz: float = 10e6
    #: Sweep downwards in frequency instead of upwards.
    down_chirp: bool = False

    # --- radar: pulse timing and Doppler -------------------------------------
    #: Pulse repetition interval: the time between transmitted pulses.
    #: For an MTI canceller this *is* the sample interval, because the filter
    #: runs across pulses rather than along one.
    pri_s: float = 1e-3
    #: Number of pulses the MTI canceller combines (2 = single canceller).
    mti_pulses: int = 2
    #: Transmit carrier frequency, used to turn Doppler into velocity.
    radar_carrier_hz: float = 10e9

    # --- pulse shaping -------------------------------------------------------
    symbol_rate: float = 100_000.0
    #: Excess bandwidth / roll-off factor for RRC and RC.
    rolloff: float = 0.35
    #: Bandwidth-time product for the Gaussian response.
    bt: float = 0.3
    #: Pulse-shaping filter length, in symbol periods.
    span_symbols: int = 11
    normalisation: str = "sum"

    # --- output scaling ------------------------------------------------------
    #: Linear passband gain applied to the designed taps.
    gain: float = 1.0

    # --- bookkeeping ---------------------------------------------------------
    name: str = "my_filter"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ utils
    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    @property
    def samples_per_symbol(self) -> float:
        if self.symbol_rate <= 0:
            raise SpecError("symbol_rate must be > 0 for a pulse-shaping filter")
        return self.sample_rate / self.symbol_rate

    # ------------------------------------------------------- radar quantities
    @property
    def effective_taylor_nbar(self) -> int:
        """The nbar the design will actually use."""
        if not self.auto_taylor_nbar:
            return self.taylor_nbar
        from .radar import minimum_taylor_nbar

        return minimum_taylor_nbar(self.taylor_sll_db)

    @property
    def prf_hz(self) -> float:
        """Pulse repetition frequency, the reciprocal of the PRI."""
        return 1.0 / self.pri_s if self.pri_s > 0 else float("inf")

    @property
    def time_bandwidth_product(self) -> float:
        """Pulse compression ratio: how much the pulse shortens on compression.

        Also the processing gain in power terms, so 10*log10 of it is the SNR
        improvement compression buys.
        """
        return self.chirp_bandwidth_hz * self.pulse_width_s

    @property
    def duty_cycle(self) -> float:
        """Fraction of time the transmitter is on."""
        return self.pulse_width_s / self.pri_s if self.pri_s > 0 else float("inf")

    @property
    def range_resolution_m(self) -> float:
        from .radar import range_resolution_m

        return range_resolution_m(self.chirp_bandwidth_hz)

    @property
    def unambiguous_range_m(self) -> float:
        from .radar import unambiguous_range_m

        return unambiguous_range_m(self.pri_s)

    @property
    def unambiguous_velocity_ms(self) -> float:
        from .radar import unambiguous_velocity_ms

        return unambiguous_velocity_ms(self.pri_s, self.radar_carrier_hz)

    @property
    def blind_speed_ms(self) -> float:
        from .radar import blind_speed_ms

        return blind_speed_ms(self.pri_s, self.radar_carrier_hz)

    @property
    def method(self) -> str:
        """The active design method for this spec's family."""
        return (
            self.fir_method.value
            if self.family is FilterFamily.FIR
            else self.iir_method.value
        )

    def copy(self, **changes: Any) -> "FilterSpec":
        """Return a modified copy.  Thin wrapper over :func:`dataclasses.replace`."""
        return replace(self, **changes)

    # ------------------------------------------------------- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Enums serialise as their string values so the JSON stays readable.
        for key in ("family", "response", "fir_method", "iir_method"):
            d[key] = getattr(self, key).value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilterSpec":
        data = dict(data)
        enum_fields = {
            "family": FilterFamily,
            "response": Response,
            "fir_method": FirMethod,
            "iir_method": IirMethod,
        }
        for key, enum_cls in enum_fields.items():
            if key in data and not isinstance(data[key], enum_cls):
                data[key] = enum_cls(data[key])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise SpecError(f"unknown spec fields: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "FilterSpec":
        return cls.from_dict(json.loads(text))

    # -------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise :class:`SpecError` if the spec is not realisable.

        This is intentionally strict: the GUI calls it on every edit, so a
        clear message here is what the user sees instead of a scipy traceback.
        """
        if self.sample_rate <= 0:
            raise SpecError("Sample rate must be greater than 0 Hz.")
        nyq = self.nyquist

        if self.response.is_radar:
            self._validate_radar()
            return

        if self.response.is_pulse_shaping:
            self._validate_pulse_shaping()
            self._validate_order()
            return

        if self.response is Response.DIFFERENTIATOR:
            if not 0 < self.f_high < nyq:
                raise SpecError(
                    "Differentiator band edge must be between 0 and Nyquist "
                    f"({nyq:.6g} Hz)."
                )
            self._validate_order()
            return

        # Ordinary frequency-selective responses.
        if not 0 < self.f_low < nyq:
            raise SpecError(
                f"Lower band edge ({self.f_low:.6g} Hz) must be between 0 and "
                f"Nyquist ({nyq:.6g} Hz)."
            )
        if self.response.is_multiband:
            if not 0 < self.f_high < nyq:
                raise SpecError(
                    f"Upper band edge ({self.f_high:.6g} Hz) must be between 0 "
                    f"and Nyquist ({nyq:.6g} Hz)."
                )
            if self.f_high <= self.f_low:
                raise SpecError("Upper band edge must be above the lower band edge.")

        if self.family is FilterFamily.FIR and self.response is not Response.HILBERT:
            if self.transition_width <= 0:
                raise SpecError("Transition width must be greater than 0 Hz.")
            self._validate_transition(nyq)

        if self.passband_ripple_db <= 0:
            raise SpecError("Passband ripple must be greater than 0 dB.")
        if self.stopband_atten_db <= 0:
            raise SpecError("Stopband attenuation must be greater than 0 dB.")
        if self.stopband_atten_db <= self.passband_ripple_db:
            raise SpecError("Stopband attenuation must exceed the passband ripple.")
        self._validate_order()

    def _validate_radar(self) -> None:
        """Check the radar responses, in radar terms."""
        if self.pri_s <= 0:
            raise SpecError("Pulse repetition interval must be greater than 0 s.")
        if self.radar_carrier_hz <= 0:
            raise SpecError("Carrier frequency must be greater than 0 Hz.")

        if self.response is Response.MTI_CANCELLER:
            if self.mti_pulses < 2:
                raise SpecError("An MTI canceller needs at least 2 pulses.")
            if self.mti_pulses > 5:
                raise SpecError(
                    "Cancellers beyond 5 pulses are not offered: each extra "
                    "pulse widens the clutter notch and eats more of the "
                    "usable Doppler band for very little extra rejection."
                )
            return

        # Matched filter for an LFM pulse.
        if self.pulse_width_s <= 0:
            raise SpecError("Pulse width must be greater than 0 s.")
        if self.chirp_bandwidth_hz <= 0:
            raise SpecError("Chirp bandwidth must be greater than 0 Hz.")
        if self.pulse_width_s >= self.pri_s:
            raise SpecError(
                f"The pulse ({self.pulse_width_s * 1e6:.4g} us) is longer than "
                f"the PRI ({self.pri_s * 1e6:.4g} us); the transmitter would "
                "never switch off. Shorten the pulse or lengthen the PRI."
            )
        # The chirp occupies +/-B/2 at complex baseband, so the sample rate has
        # to cover the whole sweep or the ends of it alias inward.
        if self.chirp_bandwidth_hz > self.sample_rate:
            raise SpecError(
                f"A {self.chirp_bandwidth_hz / 1e6:.4g} MHz chirp does not fit "
                f"in a {self.sample_rate / 1e6:.4g} MS/s complex sample rate. "
                "Raise the sample rate to at least the chirp bandwidth."
            )
        if self.time_bandwidth_product < 10:
            raise SpecError(
                f"The time-bandwidth product is only "
                f"{self.time_bandwidth_product:.1f}. Below about 10 there is "
                "nothing to compress and the sidelobe behaviour is erratic; "
                "lengthen the pulse or widen the chirp."
            )
        if self.window not in WINDOWS:
            raise SpecError(
                f"Unknown weighting {self.window!r}; expected one of "
                f"{list(WINDOWS)}."
            )

    def _validate_pulse_shaping(self) -> None:
        if self.symbol_rate <= 0:
            raise SpecError("Symbol rate must be greater than 0 Hz.")
        if self.samples_per_symbol < 2:
            raise SpecError(
                f"Samples per symbol is {self.samples_per_symbol:.3f}; it must "
                "be at least 2 (sample rate must be at least twice the symbol "
                "rate)."
            )
        if self.span_symbols < 1:
            raise SpecError("Filter span must be at least 1 symbol.")
        if self.response in (Response.RRC, Response.RC) and not (
            0.0 <= self.rolloff <= 1.0
        ):
            raise SpecError("Roll-off factor must be between 0 and 1.")
        if self.response is Response.GAUSSIAN and self.bt <= 0:
            raise SpecError("Bandwidth-time product must be greater than 0.")
        if self.normalisation not in NORMALISATIONS:
            raise SpecError(
                f"Unknown normalisation {self.normalisation!r}; expected one of "
                f"{list(NORMALISATIONS)}."
            )

    def _validate_transition(self, nyq: float) -> None:
        """Check that every transition band fits inside 0..Nyquist."""
        half = self.transition_width / 2.0
        if self.response.is_multiband:
            edges = [
                (self.f_low - half, self.f_low + half),
                (self.f_high - half, self.f_high + half),
            ]
        else:  # lowpass / highpass
            edges = [(self.f_low - half, self.f_low + half)]

        for lo, hi in edges:
            if lo <= 0 or hi >= nyq:
                raise SpecError(
                    f"A transition width of {self.transition_width:.6g} Hz does "
                    f"not fit between 0 and Nyquist ({nyq:.6g} Hz) around the "
                    "chosen band edges. Narrow the transition or move the edges."
                )
        if len(edges) == 2 and edges[0][1] >= edges[1][0]:
            raise SpecError(
                "The two transition bands overlap. Widen the gap between the "
                "band edges or narrow the transition width."
            )

    def _validate_order(self) -> None:
        if self.auto_order:
            return
        if self.family is FilterFamily.FIR:
            if self.num_taps < 1:
                raise SpecError("Tap count must be at least 1.")
            if self.num_taps > 100_000:
                raise SpecError("Tap count above 100000 is not practical.")
        else:
            if self.order < 1:
                raise SpecError("IIR order must be at least 1.")
            if self.order > 50:
                raise SpecError(
                    "IIR order above 50 is numerically unusable; use a cascade "
                    "of lower-order sections instead."
                )
