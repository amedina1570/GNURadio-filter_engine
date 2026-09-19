"""Inject a synthetic pulse and look at what comes out.

The frequency response says what the filter does to a steady sinusoid. This
panel answers the other question: what it does to a pulse -- how far the
edges ring, how much the envelope smears, whether a chirp survives.

The output is drawn shifted back by the filter's group delay, so input and
output line up and you are looking at distortion rather than latency. The
raw latency is reported separately.
"""

from __future__ import annotations

import numpy as np

from ...core import signals
from ...core.design import FilterDesign
from ...core.signals import PulseKind, PulseSpec
from ...core.spec import SpecError
from ... import plotting
from ..qt import Qt, QtWidgets
from ..widgets.fields import FrequencyEdit, combo_enum
from ..widgets.mpl_canvas import PlotCanvas

__all__ = ["PulsePanel"]

#: Which parameter rows each excitation actually uses.
_USES_WIDTH = {
    PulseKind.RECT,
    PulseKind.GAUSSIAN,
    PulseKind.SINC,
    PulseKind.RAISED_COSINE,
    PulseKind.TONE_BURST,
    PulseKind.CHIRP,
    PulseKind.BARKER13,
}
_USES_DELAY = _USES_WIDTH | {PulseKind.IMPULSE, PulseKind.STEP}


class PulsePanel(QtWidgets.QWidget):
    """Pulse generator controls beside time and spectrum plots."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._design: FilterDesign | None = None
        self._generated: signals.GeneratedSignal | None = None
        self._output: np.ndarray | None = None
        self._loading = False
        self._build()
        self._update_visibility()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_controls())

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.time_canvas = PlotCanvas(self._draw_time)
        self.spec_canvas = PlotCanvas(self._draw_spectrum)
        self.tabs.addTab(self.time_canvas, "Time domain")
        self.tabs.addTab(self.spec_canvas, "Spectrum")
        right_layout.addWidget(self.tabs)

        self.notes_label = QtWidgets.QLabel()
        self.notes_label.setWordWrap(True)
        self.notes_label.setStyleSheet("font-style: italic; font-size: 11px;")
        right_layout.addWidget(self.notes_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 900])

    def _build_controls(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(320)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        column = QtWidgets.QVBoxLayout(inner)

        box = QtWidgets.QGroupBox("Excitation")
        form = QtWidgets.QFormLayout(box)

        self.kind_combo = QtWidgets.QComboBox()
        for kind in PulseKind:
            self.kind_combo.addItem(kind.label, kind)
        self.kind_combo.currentIndexChanged.connect(self._on_kind)
        form.addRow("Signal", self.kind_combo)

        self.duration_spin = _spin(1.0, 0.001, 10_000.0, 3, " ms")
        self.duration_spin.valueChanged.connect(self._regenerate)
        form.addRow("Record length", self.duration_spin)

        self.amplitude_spin = _spin(1.0, -1e6, 1e6, 4)
        self.amplitude_spin.valueChanged.connect(self._regenerate)
        form.addRow("Amplitude", self.amplitude_spin)

        self.width_spin = _spin(50.0, 0.001, 1e6, 4, " us")
        self.width_spin.valueChanged.connect(self._regenerate)
        self.width_label = QtWidgets.QLabel("Pulse width")
        form.addRow(self.width_label, self.width_spin)

        self.delay_spin = _spin(100.0, 0.0, 1e9, 4, " us")
        self.delay_spin.valueChanged.connect(self._regenerate)
        self.delay_label = QtWidgets.QLabel("Centre at")
        form.addRow(self.delay_label, self.delay_spin)

        # PRI, not PRF: radar timing is reasoned about as an interval, and
        # the PRI is directly the listening window that sets unambiguous range.
        self.pri_spin = _spin(0.0, 0.0, 1e9, 4, " us")
        self.pri_spin.setToolTip(
            "Time from one pulse to the next. 0 emits a single pulse."
        )
        self.pri_spin.valueChanged.connect(self._regenerate)
        self.pri_label = QtWidgets.QLabel("PRI")
        form.addRow(self.pri_label, self.pri_spin)

        self.pri_derived = QtWidgets.QLabel()
        self.pri_derived.setStyleSheet("font-style: italic;")
        form.addRow("", self.pri_derived)

        self.carrier_edit = FrequencyEdit(0.0)
        self.carrier_edit.setToolTip(
            "Mix the pulse to this offset. Anything non-zero makes the signal "
            "complex baseband, which is what an SDR actually delivers."
        )
        self.carrier_edit.valueChanged.connect(self._regenerate)
        form.addRow("Carrier offset", self.carrier_edit)

        self.complex_check = QtWidgets.QCheckBox("Force complex (I/Q)")
        self.complex_check.toggled.connect(self._regenerate)
        form.addRow("", self.complex_check)
        column.addWidget(box)

        # --- chirp ---
        self.chirp_box = QtWidgets.QGroupBox("Chirp")
        chirp_form = QtWidgets.QFormLayout(self.chirp_box)
        self.chirp_f0 = FrequencyEdit(-100e3)
        self.chirp_f0.valueChanged.connect(self._regenerate)
        chirp_form.addRow("Start", self.chirp_f0)
        self.chirp_f1 = FrequencyEdit(100e3)
        self.chirp_f1.valueChanged.connect(self._regenerate)
        chirp_form.addRow("Stop", self.chirp_f1)
        self.chirp_method = QtWidgets.QComboBox()
        for method in ("linear", "quadratic", "logarithmic", "hyperbolic"):
            self.chirp_method.addItem(method, method)
        self.chirp_method.currentIndexChanged.connect(self._regenerate)
        chirp_form.addRow("Sweep", self.chirp_method)
        column.addWidget(self.chirp_box)

        # --- two tone ---
        self.tone_box = QtWidgets.QGroupBox("Two tone")
        tone_form = QtWidgets.QFormLayout(self.tone_box)
        self.tone1 = FrequencyEdit(50e3)
        self.tone1.valueChanged.connect(self._regenerate)
        tone_form.addRow("Tone 1", self.tone1)
        self.tone2 = FrequencyEdit(60e3)
        self.tone2.valueChanged.connect(self._regenerate)
        tone_form.addRow("Tone 2", self.tone2)
        column.addWidget(self.tone_box)

        # --- digital ---
        self.digital_box = QtWidgets.QGroupBox("Symbols")
        dig_form = QtWidgets.QFormLayout(self.digital_box)
        self.symrate_edit = FrequencyEdit(100e3)
        self.symrate_edit.valueChanged.connect(self._regenerate)
        dig_form.addRow("Symbol rate", self.symrate_edit)
        column.addWidget(self.digital_box)

        # --- radar ---
        self.radar_box = QtWidgets.QGroupBox("Radar echo")
        radar_form = QtWidgets.QFormLayout(self.radar_box)
        self.lfm_bw_edit = FrequencyEdit(10e6)
        self.lfm_bw_edit.setToolTip(
            "Chirp bandwidth of the echo. Match it to the filter or the pulse "
            "will not compress."
        )
        self.lfm_bw_edit.valueChanged.connect(self._regenerate)
        radar_form.addRow("Echo bandwidth", self.lfm_bw_edit)

        self.match_button = QtWidgets.QPushButton("Match the filter")
        self.match_button.setToolTip(
            "Copy the pulse width and chirp bandwidth from the current design."
        )
        self.match_button.clicked.connect(self._match_design)
        radar_form.addRow("", self.match_button)

        self.sep_spin = _spin(2.0, 0.001, 1e6, 4, " us")
        self.sep_spin.setToolTip(
            "How far behind the first target the second one sits."
        )
        self.sep_spin.valueChanged.connect(self._regenerate)
        self.sep_row_label = QtWidgets.QLabel("Target separation")
        radar_form.addRow(self.sep_row_label, self.sep_spin)

        self.sep_derived = QtWidgets.QLabel()
        self.sep_derived.setStyleSheet("font-style: italic;")
        radar_form.addRow("", self.sep_derived)

        self.target2_spin = _spin(-40.0, -120.0, 0.0, 1, " dB")
        self.target2_spin.setToolTip(
            "How much weaker the second target is. If this is above the "
            "filter's peak sidelobe level it will be buried."
        )
        self.target2_spin.valueChanged.connect(self._regenerate)
        self.target2_label = QtWidgets.QLabel("Second target")
        radar_form.addRow(self.target2_label, self.target2_spin)
        column.addWidget(self.radar_box)

        # --- impairments ---
        noise_box = QtWidgets.QGroupBox("Impairments")
        noise_form = QtWidgets.QFormLayout(noise_box)
        self.noise_check = QtWidgets.QCheckBox("Add noise")
        self.noise_check.toggled.connect(self._regenerate)
        noise_form.addRow("", self.noise_check)
        self.snr_spin = _spin(20.0, -40.0, 120.0, 1, " dB")
        self.snr_spin.valueChanged.connect(self._regenerate)
        noise_form.addRow("SNR", self.snr_spin)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setToolTip("Same seed gives the same noise every time.")
        self.seed_spin.valueChanged.connect(self._regenerate)
        noise_form.addRow("Seed", self.seed_spin)
        column.addWidget(noise_box)

        # --- view ---
        view_box = QtWidgets.QGroupBox("View")
        view_form = QtWidgets.QFormLayout(view_box)
        self.align_check = QtWidgets.QCheckBox("Compensate group delay")
        self.align_check.setChecked(True)
        self.align_check.setToolTip(
            "Shift the output back by the filter's group delay so input and "
            "output line up."
        )
        self.align_check.toggled.connect(self._redraw)
        view_form.addRow("", self.align_check)
        self.window_combo = QtWidgets.QComboBox()
        for name in ("hann", "hamming", "blackmanharris", "flattop", "boxcar"):
            self.window_combo.addItem(name, name)
        self.window_combo.currentIndexChanged.connect(self._redraw)
        view_form.addRow("FFT window", self.window_combo)
        column.addWidget(view_box)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        column.addWidget(self.status_label)
        column.addStretch(1)
        return scroll

    # ------------------------------------------------------------- public API
    def setDesign(self, design: FilterDesign | None) -> None:
        """Point the panel at a new design and recompute."""
        self._design = design
        if design is not None:
            self._sync_rates(design)
        self._regenerate()

    def _sync_rates(self, design: FilterDesign) -> None:
        """Keep pulse defaults sensible when the filter's sample rate changes."""
        from ...core.spec import Response

        fs = design.sample_rate
        self._loading = True
        try:
            if (
                abs(self.symrate_edit.value()) < 1e-9
                or self.symrate_edit.value() > fs / 2
            ):
                self.symrate_edit.setValue(fs / 10.0)
            # An echo wider than the sample rate cannot be generated at all,
            # so follow the design rather than leaving an invalid default.
            if self.lfm_bw_edit.value() > fs:
                self.lfm_bw_edit.setValue(
                    design.spec.chirp_bandwidth_hz
                    if 0 < design.spec.chirp_bandwidth_hz <= fs
                    else fs / 2.0
                )
        finally:
            self._loading = False
        self.match_button.setEnabled(
            design.spec.response is Response.MATCHED_LFM
        )

    # --------------------------------------------------------------- handlers
    def _kind(self) -> PulseKind:
        """Current excitation, coerced back from Qt's flattened string."""
        return combo_enum(self.kind_combo, PulseKind, PulseKind.RECT)

    def _on_kind(self) -> None:
        self._update_visibility()
        self._regenerate()

    def _update_visibility(self) -> None:
        kind = self._kind()
        self.chirp_box.setVisible(kind is PulseKind.CHIRP)
        self.tone_box.setVisible(kind is PulseKind.TWO_TONE)
        self.digital_box.setVisible(kind is PulseKind.PRBS_BPSK)
        self.radar_box.setVisible(kind.is_radar)
        two_targets = kind is PulseKind.TWO_TARGETS
        self.sep_spin.setVisible(two_targets)
        self.sep_row_label.setVisible(two_targets)
        self.sep_derived.setVisible(two_targets)
        self.target2_spin.setVisible(two_targets)
        self.target2_label.setVisible(two_targets)

        uses_width = kind in _USES_WIDTH
        self.width_spin.setVisible(uses_width)
        self.width_label.setVisible(uses_width)
        uses_delay = kind in _USES_DELAY
        self.delay_spin.setVisible(uses_delay)
        self.delay_label.setVisible(uses_delay)
        self.pri_spin.setVisible(uses_width)
        self.pri_label.setVisible(uses_width)
        self.pri_derived.setVisible(uses_width)

        self.noise_check.setEnabled(kind is not PulseKind.AWGN)
        self.snr_spin.setEnabled(
            self.noise_check.isChecked() and kind is not PulseKind.AWGN
        )

    def _pulse_spec(self) -> PulseSpec:
        fs = self._design.sample_rate if self._design else 1e6
        return PulseSpec(
            kind=self._kind(),
            sample_rate=fs,
            duration_s=self.duration_spin.value() * 1e-3,
            amplitude=self.amplitude_spin.value(),
            width_s=self.width_spin.value() * 1e-6,
            delay_s=self.delay_spin.value() * 1e-6,
            carrier_hz=self.carrier_edit.value(),
            force_complex=self.complex_check.isChecked(),
            pri_s=self.pri_spin.value() * 1e-6,
            chirp_f0_hz=self.chirp_f0.value(),
            chirp_f1_hz=self.chirp_f1.value(),
            chirp_method=str(self.chirp_method.currentData()),
            tone1_hz=self.tone1.value(),
            tone2_hz=self.tone2.value(),
            symbol_rate=self.symrate_edit.value(),
            seed=self.seed_spin.value(),
            lfm_bandwidth_hz=self.lfm_bw_edit.value(),
            target_separation_s=self.sep_spin.value() * 1e-6,
            target2_relative_db=self.target2_spin.value(),
            snr_db=self.snr_spin.value() if self.noise_check.isChecked() else None,
        )

    def _match_design(self) -> None:
        """Copy the pulse parameters from the filter being designed.

        A matched filter only compresses the waveform it was built for, so
        this removes the most common way to get a confusing result: an echo
        whose chirp does not match the filter's.
        """
        from ...core.spec import Response

        if self._design is None:
            return
        spec = self._design.spec
        if spec.response is not Response.MATCHED_LFM:
            return
        self._loading = True
        try:
            self.width_spin.setValue(spec.pulse_width_s * 1e6)
            self.lfm_bw_edit.setValue(spec.chirp_bandwidth_hz)
            if spec.pri_s > 0:
                self.duration_spin.setValue(min(spec.pri_s, 5e-3) * 1e3)
                self.delay_spin.setValue(spec.pulse_width_s * 1e6)
        finally:
            self._loading = False
        self._regenerate()

    def _update_derived(self) -> None:
        """Show the radar consequences of the timing choices."""
        from ...core.radar import C_LIGHT, unambiguous_range_m

        pri = self.pri_spin.value() * 1e-6
        if pri > 0:
            width = self.width_spin.value() * 1e-6
            self.pri_derived.setText(
                f"= {1.0 / pri:,.6g} Hz PRF, "
                f"{unambiguous_range_m(pri) / 1e3:,.4g} km unambiguous, "
                f"{100 * width / pri:.2f}% duty"
            )
        else:
            self.pri_derived.setText("single pulse")

        separation = self.sep_spin.value() * 1e-6
        bandwidth = self.lfm_bw_edit.value()
        if separation > 0 and bandwidth > 0:
            self.sep_derived.setText(
                f"= {separation * C_LIGHT / 2:,.4g} m apart, "
                f"{separation * bandwidth:.1f} resolution cells"
            )
        else:
            self.sep_derived.setText("")

    def _regenerate(self, *_args) -> None:
        if self._loading:
            return
        self._update_visibility()
        self._update_derived()
        if self._design is None:
            return
        try:
            generated = signals.generate(self._pulse_spec())
        except SpecError as exc:
            self._generated = None
            self._output = None
            self.status_label.setText(str(exc))
            self.status_label.setStyleSheet("color: #c0392b;")
            self._redraw()
            return

        self._generated = generated
        self._output = signals.apply_filter(self._design, generated.x)
        self.status_label.setText(
            f"{generated.x.size:,} samples, "
            f"{'complex' if generated.is_complex else 'real'}"
        )
        self.status_label.setStyleSheet("font-style: italic;")
        self.notes_label.setText("  ".join(f"- {n}" for n in generated.notes))
        self._redraw()

    def _redraw(self, *_args) -> None:
        self.time_canvas.invalidate()
        self.spec_canvas.invalidate()

    # ----------------------------------------------------------------- drawing
    def _group_delay_seconds(self) -> float:
        if not self.align_check.isChecked() or self._design is None:
            return 0.0
        if self._design.is_fir:
            return (self._design.num_taps - 1) / 2.0 / self._design.sample_rate
        # An IIR has no single group delay; use the passband average as the
        # best available alignment.
        from ...core.analysis import measure

        measured = measure(self._design, num_points=1024)
        if measured.passband_group_delay is None:
            return 0.0
        return measured.passband_group_delay / self._design.sample_rate

    def _draw_time(self, figure) -> None:
        if self._generated is None or self._output is None:
            _placeholder(figure, "Design a filter and choose an excitation.")
            return
        plotting.plot_pulse_time(
            figure,
            self._generated.t,
            self._generated.x,
            self._output,
            align_delay=self._group_delay_seconds(),
        )

    def _draw_spectrum(self, figure) -> None:
        if self._generated is None or self._output is None:
            _placeholder(figure, "Design a filter and choose an excitation.")
            return
        fs = self._generated.sample_rate
        window = str(self.window_combo.currentData())
        # Both spectra share the input's peak as the reference, so the plot
        # shows how far the filter pushed each component down rather than
        # renormalising the output back to the top of the axis.
        _f, mag_in_lin = signals.spectrum(self._generated.x, fs, window, db=False)
        ref = float(np.max(mag_in_lin)) or 1.0
        f_in, m_in = signals.spectrum(self._generated.x, fs, window, db=True, ref=ref)
        f_out, m_out = signals.spectrum(self._output, fs, window, db=True, ref=ref)
        plotting.plot_pulse_spectrum(figure, f_in, m_in, f_out, m_out, self._design)


def _spin(
    value: float, low: float, high: float, decimals: int, suffix: str = ""
) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _placeholder(figure, message: str) -> None:
    ax = figure.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", alpha=0.6)
    ax.set_axis_off()
