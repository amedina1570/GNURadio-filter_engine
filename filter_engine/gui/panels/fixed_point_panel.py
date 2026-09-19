"""Fixed-point quantization and FPGA resource estimation.

This is where a design meets the hardware.  A filter that is flawless in
double precision can lose 30 dB of stopband at 12 bits, or -- for an IIR --
go unstable outright, and there is no way to see that from the floating-point
response.  Pick a word length here and the number that matters appears
immediately: the stopband you can actually have.
"""

from __future__ import annotations

from ...core.design import FilterDesign
from ...core.quantize import QuantizedFilter, estimate_fpga, quantize
from ...presets import FPGA_PLATFORMS
from ..qt import Qt, QtWidgets, Signal

__all__ = ["FixedPointPanel"]

#: Word lengths worth offering, with why you would pick each.
_COMMON_WIDTHS = [
    (8, "8 - minimal, audio-grade at best"),
    (12, "12 - matches a B2xx converter"),
    (16, "16 - the usual default"),
    (18, "18 - fills the DSP48E1 coefficient port"),
    (24, "24 - near-transparent"),
    (32, "32 - effectively floating point"),
]


class FixedPointPanel(QtWidgets.QWidget):
    """Word-length controls, quantization metrics and an FPGA estimate."""

    quantizedChanged = Signal(object)  # QuantizedFilter | None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._design: FilterDesign | None = None
        self._quantized: QuantizedFilter | None = None
        self._loading = False
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 700])

    def _build_controls(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(360)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        column = QtWidgets.QVBoxLayout(inner)

        coeff_box = QtWidgets.QGroupBox("Coefficient format")
        form = QtWidgets.QFormLayout(coeff_box)

        self.width_combo = QtWidgets.QComboBox()
        for bits, label in _COMMON_WIDTHS:
            self.width_combo.addItem(label, bits)
        self.width_combo.setCurrentIndex(2)  # 16 bits
        self.width_combo.currentIndexChanged.connect(self._recompute)
        form.addRow("Word length", self.width_combo)

        self.auto_frac_check = QtWidgets.QCheckBox("Choose the binary point")
        self.auto_frac_check.setChecked(True)
        self.auto_frac_check.setToolTip(
            "Give every bit not needed for the integer range to the fraction."
        )
        self.auto_frac_check.toggled.connect(self._on_auto_frac)
        form.addRow("", self.auto_frac_check)

        self.frac_spin = QtWidgets.QSpinBox()
        self.frac_spin.setRange(0, 32)
        self.frac_spin.setValue(15)
        self.frac_spin.setEnabled(False)
        self.frac_spin.valueChanged.connect(self._recompute)
        form.addRow("Fractional bits", self.frac_spin)
        column.addWidget(coeff_box)

        fpga_box = QtWidgets.QGroupBox("FPGA target")
        fpga_form = QtWidgets.QFormLayout(fpga_box)

        self.fpga_combo = QtWidgets.QComboBox()
        for name in FPGA_PLATFORMS:
            self.fpga_combo.addItem(name, name)
        self.fpga_combo.currentIndexChanged.connect(self._on_fpga)
        fpga_form.addRow("Device", self.fpga_combo)

        self.clock_spin = QtWidgets.QDoubleSpinBox()
        self.clock_spin.setRange(0.1, 1000.0)
        self.clock_spin.setDecimals(3)
        self.clock_spin.setValue(100.0)
        self.clock_spin.setSuffix(" MHz")
        self.clock_spin.valueChanged.connect(self._recompute)
        fpga_form.addRow("Fabric clock", self.clock_spin)

        self.data_bits_spin = QtWidgets.QSpinBox()
        self.data_bits_spin.setRange(4, 48)
        self.data_bits_spin.setValue(16)
        self.data_bits_spin.setToolTip("Sample width entering the filter.")
        self.data_bits_spin.valueChanged.connect(self._recompute)
        fpga_form.addRow("Data width", self.data_bits_spin)
        column.addWidget(fpga_box)

        self.sweep_button = QtWidgets.QPushButton("Compare word lengths")
        self.sweep_button.setToolTip(
            "Show the stopband each word length can actually deliver."
        )
        self.sweep_button.clicked.connect(self._sweep)
        column.addWidget(self.sweep_button)

        column.addStretch(1)
        return scroll

    def _build_results(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.headline = QtWidgets.QLabel()
        self.headline.setWordWrap(True)
        self.headline.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.headline)

        tables = QtWidgets.QHBoxLayout()
        self.quant_table = _make_table("Quantization")
        self.fpga_table = _make_table("FPGA resources")
        tables.addWidget(self.quant_table)
        tables.addWidget(self.fpga_table)
        layout.addLayout(tables)

        self.notes = QtWidgets.QTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(200)
        layout.addWidget(self.notes)
        return panel

    # ------------------------------------------------------------- public API
    def setDesign(self, design: FilterDesign | None) -> None:
        self._design = design
        self._recompute()

    def quantized(self) -> QuantizedFilter | None:
        return self._quantized

    # --------------------------------------------------------------- handlers
    def _on_auto_frac(self, checked: bool) -> None:
        self.frac_spin.setEnabled(not checked)
        if checked and self._quantized is not None:
            self.frac_spin.setValue(self._quantized.fmt.frac_bits)
        self._recompute()

    def _on_fpga(self) -> None:
        name = self.fpga_combo.currentData()
        if name:
            self._loading = True
            self.clock_spin.setValue(FPGA_PLATFORMS[name].typical_clock_hz / 1e6)
            self._loading = False
        self._recompute()

    def _recompute(self, *_args) -> None:
        if self._loading:
            return
        if self._design is None:
            self._quantized = None
            self.headline.setText("No design yet.")
            self.quantizedChanged.emit(None)
            return

        bits = self.width_combo.currentData()
        frac = None if self.auto_frac_check.isChecked() else self.frac_spin.value()
        if frac is not None and frac > bits:
            frac = bits
            self.frac_spin.setValue(bits)

        try:
            quantized = quantize(self._design, total_bits=bits, frac_bits=frac)
        except Exception as exc:
            self._quantized = None
            self.headline.setText(f"<b style='color:#c0392b'>{exc}</b>")
            self.quantizedChanged.emit(None)
            return

        self._quantized = quantized
        if self.auto_frac_check.isChecked():
            blocked = self.frac_spin.blockSignals(True)
            self.frac_spin.setValue(quantized.fmt.frac_bits)
            self.frac_spin.blockSignals(blocked)

        self._show(quantized)
        self.quantizedChanged.emit(quantized)

    def _show(self, q: QuantizedFilter) -> None:
        _fill_table(self.quant_table, q.summary_rows())

        estimate = estimate_fpga(
            q,
            clock_hz=self.clock_spin.value() * 1e6,
            sample_rate=self._design.sample_rate if self._design else None,
            data_bits=self.data_bits_spin.value(),
        )
        _fill_table(self.fpga_table, estimate.as_rows())

        notes = list(q.notes) + list(estimate.notes)
        device_name = self.fpga_combo.currentData()
        if device_name:
            device = FPGA_PLATFORMS[device_name]
            usage = device.check_usage(estimate.dsp_slices)
            if usage:
                notes.append(usage)
            else:
                notes.append(
                    f"{estimate.dsp_slices} of the {device.name}'s "
                    f"{device.dsp_slices} DSP slices "
                    f"({100 * estimate.dsp_slices / device.dsp_slices:.1f}%)."
                )
            if device.notes:
                notes.append(device.notes)
        self.notes.setPlainText("\n\n".join(f"- {n}" for n in notes))

        self.headline.setText(self._headline(q))

    def _headline(self, q: QuantizedFilter) -> str:
        if q.degenerate:
            return (
                "<b style='color:#c0392b'>Unusable at this word length.</b> "
                "Rounding destroyed the filter; use more bits."
            )
        if not q.stable:
            return (
                "<b style='color:#c0392b'>Unstable at this word length.</b> "
                "Rounding pushed a pole outside the unit circle."
            )
        requested = self._design.spec.stopband_atten_db if self._design else 0.0
        if q.quantized_stopband_db is None:
            return (
                f"<b>{q.fmt}</b> - quantization noise floor "
                f"{q.error_floor_db:.1f} dB."
            )
        achieved = q.quantized_stopband_db
        # Three states, not two: a word length that lands a decibel short is
        # a different situation from one that meets the mask, and calling it
        # "meets" because it is within a tolerance would be a lie the user
        # only discovers in hardware.
        if achieved >= requested:
            colour, verdict = "#1e8449", "meets the"
        elif achieved >= requested - 3.0:
            colour, verdict = "#b9770e", "falls just short of the"
        else:
            colour, verdict = "#c0392b", "misses the"
        return (
            f"<b>{q.fmt}</b> gives <b style='color:{colour}'>"
            f"{achieved:.1f} dB</b> of stopband attenuation, which {verdict} "
            f"{requested:.0f} dB requested. "
            f"Quantization noise floor: {q.error_floor_db:.1f} dB."
        )

    # ------------------------------------------------------------------ sweep
    def _sweep(self) -> None:
        """Quantize at every offered word length and tabulate the result."""
        if self._design is None:
            return
        rows = []
        for bits, _label in _COMMON_WIDTHS:
            try:
                q = quantize(self._design, total_bits=bits)
            except Exception as exc:
                rows.append((f"{bits} bits", f"failed: {exc}"))
                continue
            if q.degenerate:
                verdict = "unusable (coefficients round to zero)"
            elif not q.stable:
                verdict = "UNSTABLE"
            elif q.quantized_stopband_db is not None:
                verdict = (
                    f"{q.quantized_stopband_db:.1f} dB stopband, "
                    f"floor {q.error_floor_db:.1f} dB, {q.fmt.name}"
                )
            else:
                verdict = f"floor {q.error_floor_db:.1f} dB, {q.fmt.name}"
            rows.append((f"{bits} bits", verdict))

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Word length comparison")
        dialog.resize(620, 300)
        layout = QtWidgets.QVBoxLayout(dialog)
        intro = QtWidgets.QLabel(
            f"Stopband actually achievable for <b>{self._design.spec.name}</b> "
            f"(requested: {self._design.spec.stopband_atten_db:.0f} dB)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        table = _make_table("")
        _fill_table(table, rows)
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()


def _make_table(title: str) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, 2)
    table.setHorizontalHeaderLabels([title or "", ""])
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(
        0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
    )
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(
        QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    )
    table.setAlternatingRowColors(True)
    return table


def _fill_table(table: QtWidgets.QTableWidget, rows: list[tuple[str, str]]) -> None:
    table.setRowCount(len(rows))
    for i, (label, value) in enumerate(rows):
        table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(label)))
        item = QtWidgets.QTableWidgetItem(str(value))
        upper = str(value).upper()
        if "NO" == upper or "UNSTABLE" in upper or "YES -" in upper:
            item.setForeground(Qt.GlobalColor.red)
        table.setItem(i, 1, item)
