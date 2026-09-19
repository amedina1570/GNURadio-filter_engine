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

    @property
    def is_pulse_shaping(self) -> bool:
        return self in (Response.RRC, Response.RC, Response.GAUSSIAN)

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
)

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
    window: str = "hamming"
    #: Kaiser beta.  Only read when :attr:`window` is ``"kaiser"``.
    window_param: float = 8.6

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
