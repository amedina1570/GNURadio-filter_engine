"""The filter specification form.

Emits :attr:`DesignPanel.specChanged` whenever the user edits anything.  The
panel owns no design results -- it produces a :class:`FilterSpec` and the main
window decides what to do with it.

Fields appear and disappear with the response type.  A Hilbert transformer has
no transition width and a root-raised-cosine has no stopband attenuation, so
showing those boxes greyed out would only invite the user to set a value that
gets ignored.
"""

from __future__ import annotations

from ...core.spec import (
    NORMALISATIONS,
    WINDOWS,
    FilterFamily,
    FirMethod,
    IirMethod,
    Response,
    FilterSpec,
    SpecError,
)
from ...presets import FILTER_PRESETS, SDR_PLATFORMS
from ..qt import QtWidgets, Signal
from ..widgets.fields import FrequencyEdit, LabelledRow, combo_enum, select_data

__all__ = ["DesignPanel"]

#: Friendly names for the response types, in the order they appear.
_RESPONSE_LABELS = [
    (Response.LOWPASS, "Low pass"),
    (Response.HIGHPASS, "High pass"),
    (Response.BANDPASS, "Band pass"),
    (Response.BANDSTOP, "Band stop"),
    (Response.HILBERT, "Hilbert transformer"),
    (Response.DIFFERENTIATOR, "Differentiator"),
    (Response.RRC, "Root raised cosine"),
    (Response.RC, "Raised cosine"),
    (Response.GAUSSIAN, "Gaussian"),
]

_FIR_METHOD_LABELS = [
    (FirMethod.WINDOW, "Windowed sinc (firwin)"),
    (FirMethod.REMEZ, "Equiripple / Parks-McClellan (remez)"),
    (FirMethod.FIRLS, "Least squares (firls)"),
    (FirMethod.FREQ_SAMPLING, "Frequency sampling (firwin2)"),
]

_IIR_METHOD_LABELS = [
    (IirMethod.BUTTER, "Butterworth (maximally flat)"),
    (IirMethod.CHEBY1, "Chebyshev I (passband ripple)"),
    (IirMethod.CHEBY2, "Chebyshev II (stopband ripple)"),
    (IirMethod.ELLIP, "Elliptic (lowest order)"),
    (IirMethod.BESSEL, "Bessel (flat group delay)"),
]

#: Responses that only exist as FIR.
_FIR_ONLY = {
    Response.HILBERT,
    Response.DIFFERENTIATOR,
    Response.RRC,
    Response.RC,
    Response.GAUSSIAN,
}


class DesignPanel(QtWidgets.QWidget):
    """Form for building a :class:`FilterSpec`."""

    specChanged = Signal(object)  # FilterSpec

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loading = False
        self._spec = FilterSpec()
        self._build()
        self._load_spec(self._spec)
        self._update_visibility()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 6, 0)

        layout.addWidget(self._build_preset_group())
        layout.addWidget(self._build_type_group())
        layout.addWidget(self._build_frequency_group())
        layout.addWidget(self._build_tolerance_group())
        layout.addWidget(self._build_order_group())
        layout.addWidget(self._build_pulse_group())
        layout.addStretch(1)

    def _build_preset_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Start from")
        form = QtWidgets.QFormLayout(box)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("(custom)", None)
        for preset in FILTER_PRESETS:
            self.preset_combo.addItem(preset.name, preset)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        form.addRow("Preset", self.preset_combo)

        self.platform_combo = QtWidgets.QComboBox()
        self.platform_combo.addItem("(no hardware limits)", None)
        for name in SDR_PLATFORMS:
            self.platform_combo.addItem(name, name)
        self.platform_combo.currentIndexChanged.connect(self._on_platform)
        form.addRow("Radio", self.platform_combo)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setToolTip(
            "Used for the generated class, module and HDL entity names."
        )
        self.name_edit.textEdited.connect(self._emit)
        form.addRow("Name", self.name_edit)
        return box

    def _build_type_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Filter type")
        form = QtWidgets.QFormLayout(box)

        self.response_combo = QtWidgets.QComboBox()
        for response, label in _RESPONSE_LABELS:
            self.response_combo.addItem(label, response)
        self.response_combo.currentIndexChanged.connect(self._on_response)
        form.addRow("Response", self.response_combo)

        self.family_combo = QtWidgets.QComboBox()
        self.family_combo.addItem("FIR (finite impulse response)", FilterFamily.FIR)
        self.family_combo.addItem("IIR (infinite impulse response)", FilterFamily.IIR)
        self.family_combo.currentIndexChanged.connect(self._on_family)
        form.addRow("Family", self.family_combo)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.currentIndexChanged.connect(self._emit)
        form.addRow("Method", self.method_combo)

        self.window_combo = QtWidgets.QComboBox()
        for name in WINDOWS:
            self.window_combo.addItem(name, name)
        self.window_combo.currentIndexChanged.connect(self._on_window)
        self.window_row = LabelledRow(form, "Window", self.window_combo)

        self.beta_spin = QtWidgets.QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 30.0)
        self.beta_spin.setDecimals(3)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setToolTip(
            "Kaiser beta: higher means deeper stopband and a wider transition."
        )
        self.beta_spin.valueChanged.connect(self._emit)
        self.beta_row = LabelledRow(form, "Kaiser beta", self.beta_spin)
        return box

    def _build_frequency_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Frequencies")
        form = QtWidgets.QFormLayout(box)

        self.fs_edit = FrequencyEdit(1e6)
        self.fs_edit.valueChanged.connect(self._on_sample_rate)
        form.addRow("Sample rate", self.fs_edit)

        self.nyquist_label = QtWidgets.QLabel()
        self.nyquist_label.setStyleSheet("font-style: italic;")
        form.addRow("", self.nyquist_label)

        self.flow_edit = FrequencyEdit(100e3)
        self.flow_edit.valueChanged.connect(self._emit)
        self.flow_row = LabelledRow(form, "Cutoff", self.flow_edit)

        self.fhigh_edit = FrequencyEdit(200e3)
        self.fhigh_edit.valueChanged.connect(self._emit)
        self.fhigh_row = LabelledRow(form, "Upper edge", self.fhigh_edit)

        self.tw_edit = FrequencyEdit(25e3)
        self.tw_edit.valueChanged.connect(self._emit)
        self.tw_edit.setToolTip(
            "Width of the transition band. Halving this roughly doubles the "
            "tap count."
        )
        self.tw_row = LabelledRow(form, "Transition width", self.tw_edit)

        self.platform_warning = QtWidgets.QLabel()
        self.platform_warning.setWordWrap(True)
        self.platform_warning.setStyleSheet("color: #c0392b;")
        self.platform_warning.setVisible(False)
        form.addRow("", self.platform_warning)
        return box

    def _build_tolerance_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Tolerances")
        form = QtWidgets.QFormLayout(box)

        self.ripple_spin = QtWidgets.QDoubleSpinBox()
        self.ripple_spin.setRange(0.0001, 20.0)
        self.ripple_spin.setDecimals(4)
        self.ripple_spin.setSingleStep(0.05)
        self.ripple_spin.setSuffix(" dB")
        self.ripple_spin.valueChanged.connect(self._emit)
        self.ripple_row = LabelledRow(form, "Passband ripple", self.ripple_spin)

        self.atten_spin = QtWidgets.QDoubleSpinBox()
        self.atten_spin.setRange(1.0, 200.0)
        self.atten_spin.setDecimals(1)
        self.atten_spin.setSingleStep(5.0)
        self.atten_spin.setSuffix(" dB")
        # Not just _emit: a derived Kaiser beta is a function of this value,
        # so the displayed beta has to follow it.
        self.atten_spin.valueChanged.connect(self._on_stopband)
        self.atten_row = LabelledRow(form, "Stopband attenuation", self.atten_spin)

        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(-1e6, 1e6)
        self.gain_spin.setDecimals(6)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self._emit)
        form.addRow("Passband gain", self.gain_spin)
        return box

    def _build_order_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Order")
        form = QtWidgets.QFormLayout(box)

        self.auto_check = QtWidgets.QCheckBox("Estimate from the tolerances")
        self.auto_check.setToolTip(
            "Pick the smallest order that meets the ripple and attenuation "
            "you asked for."
        )
        self.auto_check.toggled.connect(self._on_auto)
        form.addRow("", self.auto_check)

        self.taps_spin = QtWidgets.QSpinBox()
        self.taps_spin.setRange(1, 100_000)
        self.taps_spin.valueChanged.connect(self._emit)
        self.taps_row = LabelledRow(form, "Taps", self.taps_spin)

        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 50)
        self.order_spin.valueChanged.connect(self._emit)
        self.order_row = LabelledRow(form, "Order", self.order_spin)
        return box

    def _build_pulse_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Pulse shaping")
        self.pulse_group = box
        form = QtWidgets.QFormLayout(box)

        self.symrate_edit = FrequencyEdit(100e3)
        self.symrate_edit.valueChanged.connect(self._on_sample_rate)
        form.addRow("Symbol rate", self.symrate_edit)

        self.sps_label = QtWidgets.QLabel()
        self.sps_label.setStyleSheet("font-style: italic;")
        form.addRow("", self.sps_label)

        self.rolloff_spin = QtWidgets.QDoubleSpinBox()
        self.rolloff_spin.setRange(0.0, 1.0)
        self.rolloff_spin.setDecimals(3)
        self.rolloff_spin.setSingleStep(0.05)
        self.rolloff_spin.setToolTip(
            "Excess bandwidth. 0 is a brick wall you cannot build; 0.35 is the "
            "usual compromise."
        )
        self.rolloff_spin.valueChanged.connect(self._emit)
        self.rolloff_row = LabelledRow(form, "Roll-off", self.rolloff_spin)

        self.bt_spin = QtWidgets.QDoubleSpinBox()
        self.bt_spin.setRange(0.01, 5.0)
        self.bt_spin.setDecimals(3)
        self.bt_spin.setSingleStep(0.05)
        self.bt_spin.valueChanged.connect(self._emit)
        self.bt_row = LabelledRow(form, "BT product", self.bt_spin)

        self.span_spin = QtWidgets.QSpinBox()
        self.span_spin.setRange(1, 201)
        self.span_spin.setSuffix(" symbols")
        self.span_spin.valueChanged.connect(self._emit)
        form.addRow("Span", self.span_spin)

        self.norm_combo = QtWidgets.QComboBox()
        for mode in NORMALISATIONS:
            self.norm_combo.addItem(mode, mode)
        self.norm_combo.setToolTip(
            "'energy' is the right choice for a matched transmit/receive pair; "
            "'sum' matches GNU Radio's firdes."
        )
        self.norm_combo.currentIndexChanged.connect(self._emit)
        form.addRow("Normalisation", self.norm_combo)
        return box

    # ------------------------------------------------------------- public API
    def spec(self) -> FilterSpec:
        """Build a spec from the current widget values."""
        return FilterSpec(
            name=self.name_edit.text().strip() or "my_filter",
            family=self._family(),
            response=self._response(),
            fir_method=self._current_fir_method(),
            iir_method=self._current_iir_method(),
            sample_rate=self.fs_edit.value(),
            f_low=self.flow_edit.value(),
            f_high=self.fhigh_edit.value(),
            transition_width=self.tw_edit.value(),
            passband_ripple_db=self.ripple_spin.value(),
            stopband_atten_db=self.atten_spin.value(),
            auto_order=self.auto_check.isChecked(),
            num_taps=self.taps_spin.value(),
            order=self.order_spin.value(),
            window=str(self.window_combo.currentData()),
            window_param=self.beta_spin.value(),
            symbol_rate=self.symrate_edit.value(),
            rolloff=self.rolloff_spin.value(),
            bt=self.bt_spin.value(),
            span_symbols=self.span_spin.value(),
            normalisation=str(self.norm_combo.currentData()),
            gain=self.gain_spin.value(),
        )

    def setSpec(self, spec: FilterSpec) -> None:
        """Load ``spec`` into the form without emitting a change."""
        self._load_spec(spec)
        self._update_visibility()
        self._emit()

    # ------------------------------------------------------------- internals
    # Every enum-valued combo is read through combo_enum: Qt returns our
    # str-based enums as plain strings, which breaks identity and set tests.
    def _response(self) -> Response:
        return combo_enum(self.response_combo, Response, self._spec.response)

    def _family(self) -> FilterFamily:
        return combo_enum(self.family_combo, FilterFamily, self._spec.family)

    def _current_fir_method(self) -> FirMethod:
        if self._family() is FilterFamily.FIR:
            return combo_enum(self.method_combo, FirMethod, self._spec.fir_method)
        return self._spec.fir_method

    def _current_iir_method(self) -> IirMethod:
        if self._family() is FilterFamily.IIR:
            return combo_enum(self.method_combo, IirMethod, self._spec.iir_method)
        return self._spec.iir_method

    def _load_spec(self, spec: FilterSpec) -> None:
        self._loading = True
        try:
            self._spec = spec
            self.name_edit.setText(spec.name)
            select_data(self.response_combo, spec.response)
            select_data(self.family_combo, spec.family)
            self._rebuild_methods(spec)
            self.fs_edit.setValue(spec.sample_rate)
            self.flow_edit.setValue(spec.f_low)
            self.fhigh_edit.setValue(spec.f_high)
            self.tw_edit.setValue(spec.transition_width)
            self.ripple_spin.setValue(spec.passband_ripple_db)
            self.atten_spin.setValue(spec.stopband_atten_db)
            self.auto_check.setChecked(spec.auto_order)
            self.taps_spin.setValue(spec.num_taps)
            self.order_spin.setValue(spec.order)
            select_data(self.window_combo, spec.window)
            self.beta_spin.setValue(spec.window_param)
            self.symrate_edit.setValue(spec.symbol_rate)
            self.rolloff_spin.setValue(spec.rolloff)
            self.bt_spin.setValue(spec.bt)
            self.span_spin.setValue(spec.span_symbols)
            select_data(self.norm_combo, spec.normalisation)
            self.gain_spin.setValue(spec.gain)
        finally:
            self._loading = False
        self._update_labels()

    def _rebuild_methods(self, spec: FilterSpec) -> None:
        """Repopulate the method list for the current family."""
        blocked = self.method_combo.blockSignals(True)
        self.method_combo.clear()
        if spec.family is FilterFamily.FIR:
            for method, label in _FIR_METHOD_LABELS:
                self.method_combo.addItem(label, method)
            select_data(self.method_combo, spec.fir_method)
        else:
            for method, label in _IIR_METHOD_LABELS:
                self.method_combo.addItem(label, method)
            select_data(self.method_combo, spec.iir_method)
        self.method_combo.blockSignals(blocked)

    # --------------------------------------------------------------- handlers
    def _on_preset(self) -> None:
        preset = self.preset_combo.currentData()
        if preset is None or self._loading:
            return
        self.setSpec(preset.spec.copy())

    def _on_platform(self) -> None:
        name = self.platform_combo.currentData()
        if name is None:
            self.platform_warning.setVisible(False)
            return
        platform = SDR_PLATFORMS[name]
        if not self._loading:
            self.fs_edit.setValue(platform.default_sample_rate)
        self._update_labels()
        self._emit()

    def _on_sample_rate(self, _value: float = 0.0) -> None:
        self._update_labels()
        self._emit()

    def _on_response(self) -> None:
        if self._loading:
            return
        response = self._response()
        # Force FIR for responses that have no IIR form, rather than letting
        # the user pick a combination that will only fail on design.
        if response in _FIR_ONLY:
            blocked = self.family_combo.blockSignals(True)
            select_data(self.family_combo, FilterFamily.FIR)
            self.family_combo.blockSignals(blocked)
            self._rebuild_methods(self.spec())
        self.family_combo.setEnabled(response not in _FIR_ONLY)
        self._update_visibility()
        self._emit()

    def _on_family(self) -> None:
        if self._loading:
            return
        self._rebuild_methods(self.spec())
        self._update_visibility()
        self._emit()

    def _on_window(self) -> None:
        self._update_visibility()
        self._emit()

    def _on_auto(self) -> None:
        self._update_visibility()
        self._emit()

    def _on_stopband(self, _value: float = 0.0) -> None:
        self._update_visibility()
        self._emit()

    # ----------------------------------------------------------- presentation
    def _update_labels(self) -> None:
        from ..widgets.fields import format_frequency

        fs = self.fs_edit.value()
        self.nyquist_label.setText(f"Nyquist: {format_frequency(fs / 2)}Hz")

        symbol_rate = self.symrate_edit.value()
        if symbol_rate > 0:
            sps = fs / symbol_rate
            warning = "" if sps >= 2 else "  (must be at least 2)"
            self.sps_label.setText(f"{sps:.4g} samples/symbol{warning}")
        else:
            self.sps_label.setText("")

        name = self.platform_combo.currentData()
        if name is None:
            self.platform_warning.setVisible(False)
        else:
            message = SDR_PLATFORMS[name].check_sample_rate(fs)
            self.platform_warning.setText(message or "")
            self.platform_warning.setVisible(bool(message))

    def _sync_kaiser_beta(self, auto: bool) -> None:
        """Show the beta the design will really use, and lock it when derived."""
        from ...core.firdes import kaiser_beta_for_atten

        self.beta_spin.setEnabled(not auto)
        if auto:
            derived = kaiser_beta_for_atten(self.atten_spin.value())
            blocked = self.beta_spin.blockSignals(True)
            self.beta_spin.setValue(derived)
            self.beta_spin.blockSignals(blocked)
            self.beta_row.label.setText("Kaiser beta (derived)")
            self.beta_spin.setToolTip(
                f"Derived from the {self.atten_spin.value():.1f} dB stopband "
                "requirement. Turn off order estimation to set it yourself."
            )
        else:
            self.beta_row.label.setText("Kaiser beta")
            self.beta_spin.setToolTip(
                "Higher beta means a deeper stopband and a wider transition."
            )

    def _update_visibility(self) -> None:
        response = self._response()
        family = self._family()
        is_pulse = response.is_pulse_shaping
        is_fir = family is FilterFamily.FIR
        auto = self.auto_check.isChecked()

        self.pulse_group.setVisible(is_pulse)

        # Band edges
        self.flow_row.setVisible(
            not is_pulse and response is not Response.DIFFERENTIATOR
        )
        self.fhigh_row.setVisible(
            not is_pulse
            and (response.is_multiband or response is Response.DIFFERENTIATOR)
        )
        if response is Response.DIFFERENTIATOR:
            self.fhigh_row.label.setText("Band edge")
        elif response.is_multiband:
            self.fhigh_row.label.setText("Upper edge")
        self.flow_row.label.setText(
            "Lower edge" if response.is_multiband else "Cutoff"
        )

        # A Hilbert transformer has no transition band, and pulse shapes are
        # defined by roll-off rather than by a transition width.
        needs_transition = not is_pulse and response not in (
            Response.HILBERT,
            Response.DIFFERENTIATOR,
        )
        self.tw_row.setVisible(needs_transition)
        self.ripple_row.setVisible(needs_transition)
        self.atten_row.setVisible(needs_transition)

        # Window options only apply to the windowed FIR method.
        windowed = (
            is_fir
            and not is_pulse
            and self._current_fir_method() is FirMethod.WINDOW
            and response is not Response.DIFFERENTIATOR
        )
        self.window_row.setVisible(windowed or response is Response.HILBERT)
        is_kaiser = str(self.window_combo.currentData()) == "kaiser"
        self.beta_row.setVisible(
            (windowed or response is Response.HILBERT) and is_kaiser
        )
        # With auto order on, beta is derived from the stopband requirement and
        # whatever is in this box is ignored. Showing an editable 8.6 that has
        # no effect is worse than showing nothing, so display the value that is
        # actually used and lock the field.
        if is_kaiser and windowed:
            self._sync_kaiser_beta(auto)
        self.method_combo.setEnabled(not is_pulse and response is not Response.HILBERT)

        # Order
        self.auto_check.setVisible(not is_pulse)
        self.taps_row.setVisible(not is_pulse and is_fir and not auto)
        self.order_row.setVisible(not is_pulse and not is_fir and not auto)

        # Roll-off versus BT product
        self.rolloff_row.setVisible(response in (Response.RRC, Response.RC))
        self.bt_row.setVisible(response is Response.GAUSSIAN)

    # --------------------------------------------------------------- emitting
    def _emit(self, *_args) -> None:
        if self._loading:
            return
        self._update_labels()
        try:
            spec = self.spec()
        except (SpecError, TypeError):
            return
        self._spec = spec
        self.specChanged.emit(spec)
