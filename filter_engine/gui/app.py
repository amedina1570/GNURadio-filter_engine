"""The application window.

Wiring: the design panel emits a spec, the window designs it, and everything
downstream -- plots, fixed point, pulse response, code -- is refreshed from
the result.

Two things keep it responsive on a 700-tap filter. Redesigns are debounced,
so holding a spin box's arrow key does not queue a hundred designs. And plots
redraw lazily, so only the tab you are looking at costs anything.
"""

from __future__ import annotations

import traceback

from ..core.design import DesignError, FilterDesign, design
from ..core.analysis import measure
from ..core.spec import FilterSpec, SpecError
from .. import plotting
from .panels.code_panel import CodePanel
from .panels.design_panel import DesignPanel
from .panels.fixed_point_panel import FixedPointPanel
from .panels.pulse_panel import PulsePanel
from .qt import QT_API, Qt, QtCore, QtGui, QtWidgets
from .widgets.mpl_canvas import PlotCanvas

__all__ = ["MainWindow", "main"]

#: How long to wait after the last edit before redesigning, in milliseconds.
REDESIGN_DELAY_MS = 120


class MainWindow(QtWidgets.QMainWindow):
    """Filter design, analysis, fixed-point checking and code generation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Digital Filter Engine")
        self.resize(1440, 900)

        self._design: FilterDesign | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDESIGN_DELAY_MS)
        self._timer.timeout.connect(self._redesign)
        self._pending: FilterSpec | None = None

        self._build()
        self._connect()
        # Design the default spec so the window is never empty on startup.
        self._on_spec_changed(self.design_panel.spec())

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.design_panel = DesignPanel()
        self.design_panel.setMinimumWidth(320)
        self.design_panel.setMaximumWidth(460)
        splitter.addWidget(self.design_panel)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_response_tab(), "Response")
        self.pulse_panel = PulsePanel()
        self.tabs.addTab(self.pulse_panel, "Pulse response")
        self.fixed_point_panel = FixedPointPanel()
        self.tabs.addTab(self.fixed_point_panel, "Fixed point && FPGA")
        self.code_panel = CodePanel()
        self.tabs.addTab(self.code_panel, "Generated code")
        right_layout.addWidget(self.tabs)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1060])

        self._build_status_bar()
        self._build_menus()

    def _build_response_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        options = QtWidgets.QHBoxLayout()
        self.db_check = QtWidgets.QCheckBox("dB scale")
        self.db_check.setChecked(True)
        self.log_check = QtWidgets.QCheckBox("Log frequency")
        self.mask_check = QtWidgets.QCheckBox("Show spec mask")
        self.mask_check.setChecked(True)
        self.overlay_check = QtWidgets.QCheckBox("Overlay fixed point")
        self.overlay_check.setChecked(True)
        self.overlay_check.setToolTip(
            "Draw the quantized response from the Fixed point tab on top of "
            "the ideal one."
        )
        for box in (self.db_check, self.log_check, self.mask_check, self.overlay_check):
            box.toggled.connect(self._invalidate_plots)
            options.addWidget(box)
        options.addStretch(1)
        layout.addLayout(options)

        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_tabs.setDocumentMode(True)
        self.canvases: dict[str, PlotCanvas] = {}
        for label, draw in (
            ("Magnitude", self._draw_magnitude),
            ("Phase", self._draw_phase),
            ("Group delay", self._draw_group_delay),
            ("Impulse", self._draw_impulse),
            ("Step", self._draw_step),
            ("Poles && zeros", self._draw_pole_zero),
            ("Coefficients", self._draw_taps),
            ("Overview", self._draw_overview),
        ):
            canvas = PlotCanvas(draw)
            self.canvases[label] = canvas
            self.plot_tabs.addTab(canvas, label)
        layout.addWidget(self.plot_tabs)
        return panel

    def _build_status_bar(self) -> None:
        self.status = self.statusBar()
        self.summary_label = QtWidgets.QLabel()
        self.status.addWidget(self.summary_label, 1)
        self.measure_label = QtWidgets.QLabel()
        self.status.addPermanentWidget(self.measure_label)

        self.detail_dock = QtWidgets.QDockWidget("Design notes", self)
        self.detail_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.detail_text = QtWidgets.QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_dock.setWidget(self.detail_text)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.detail_dock)
        self.detail_dock.hide()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QtGui.QAction("&Open specification...", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_spec)
        file_menu.addAction(open_action)

        save_action = QtGui.QAction("&Save specification...", self)
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_spec)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        export_plot = QtGui.QAction("Export current &plot...", self)
        export_plot.triggered.connect(self._export_plot)
        file_menu.addAction(export_plot)

        export_taps = QtGui.QAction("Export &coefficients as CSV...", self)
        export_taps.triggered.connect(self._export_coefficients)
        file_menu.addAction(export_taps)

        file_menu.addSeparator()
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.detail_dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("&Help")
        about = QtGui.QAction("&About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _connect(self) -> None:
        self.design_panel.specChanged.connect(self._on_spec_changed)
        self.fixed_point_panel.quantizedChanged.connect(self._on_quantized)

    # --------------------------------------------------------------- redesign
    def _on_spec_changed(self, spec: FilterSpec) -> None:
        # Coalesce bursts of edits; only the last spec is designed.
        self._pending = spec
        self._timer.start()

    def _redesign(self) -> None:
        spec = self._pending
        if spec is None:
            return
        try:
            result = design(spec)
        except (SpecError, DesignError) as exc:
            self._design = None
            self._set_error(str(exc))
            self._propagate()
            return
        except Exception as exc:  # unexpected: show the traceback, keep running
            self._design = None
            self._set_error(f"Unexpected error: {exc}")
            self.detail_text.setPlainText(traceback.format_exc())
            self._propagate()
            return

        self._design = result
        self._show_summary(result)
        self._propagate()

    def _propagate(self) -> None:
        """Hand the new design to every panel and invalidate the plots."""
        self.fixed_point_panel.setDesign(self._design)
        self.pulse_panel.setDesign(self._design)
        self.code_panel.setDesign(self._design)
        self._invalidate_plots()

    def _on_quantized(self, _quantized) -> None:
        self.code_panel.setQuantized(self.fixed_point_panel.quantized())
        if self.overlay_check.isChecked():
            self._invalidate_plots()

    def _invalidate_plots(self, *_args) -> None:
        for canvas in self.canvases.values():
            canvas.invalidate()

    # ------------------------------------------------------------ presentation
    def _set_error(self, message: str) -> None:
        self.summary_label.setText(message)
        self.summary_label.setStyleSheet("color: #c0392b;")
        self.measure_label.setText("")

    def _show_summary(self, fd: FilterDesign) -> None:
        self.summary_label.setStyleSheet("")
        if fd.is_fir:
            headline = f"{fd.num_taps} taps"
        else:
            headline = f"order {fd.order} ({fd.num_sections} biquads)"
            if not fd.is_stable:
                headline += "  UNSTABLE"
        spec = fd.spec
        self.summary_label.setText(
            f"{spec.response.value} / {spec.family.value.upper()} / "
            f"{spec.method} - {headline}"
        )

        try:
            measured = measure(fd, num_points=4096)
        except Exception:
            self.measure_label.setText("")
            self.detail_text.setPlainText(fd.summary())
            return

        parts = []
        if measured.passband_ripple_db is not None:
            parts.append(f"ripple {measured.passband_ripple_db:.3g} dB")
        if measured.stopband_atten_db is not None:
            parts.append(f"stopband {measured.stopband_atten_db:.1f} dB")
        if measured.passband_group_delay is not None:
            parts.append(f"delay {measured.passband_group_delay:.4g} samples")
        if measured.meets_spec is False:
            parts.append("MISSES SPEC")
        self.measure_label.setText("   ".join(parts))
        self.measure_label.setStyleSheet(
            "color: #c0392b;" if measured.meets_spec is False else ""
        )

        detail = [fd.summary(), ""]
        detail.append("Measured:")
        detail.extend(f"  {k}: {v}" for k, v in measured.as_rows())
        if measured.notes:
            detail.append("")
            detail.extend(f"  {n}" for n in measured.notes)
        self.detail_text.setPlainText("\n".join(detail))

    # ------------------------------------------------------------------ plots
    def _quantized_for_plot(self):
        if not self.overlay_check.isChecked():
            return None
        return self.fixed_point_panel.quantized()

    def _draw_magnitude(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_magnitude(
            figure,
            self._design,
            quantized=self._quantized_for_plot(),
            log_freq=self.log_check.isChecked(),
            db=self.db_check.isChecked(),
            show_mask=self.mask_check.isChecked(),
        )

    def _draw_phase(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_phase(figure, self._design)

    def _draw_group_delay(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_group_delay(figure, self._design)

    def _draw_impulse(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_impulse(figure, self._design, self._quantized_for_plot())

    def _draw_step(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_step(figure, self._design)

    def _draw_pole_zero(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_pole_zero(figure, self._design)

    def _draw_taps(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_taps(figure, self._design, self._quantized_for_plot())

    def _draw_overview(self, figure) -> None:
        if self._design is None:
            return _no_design(figure)
        plotting.plot_overview(figure, self._design, self._quantized_for_plot())

    # ------------------------------------------------------------------ files
    def _save_spec(self) -> None:
        if self._design is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save specification", f"{self._design.spec.name}.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._design.spec.to_json())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.status.showMessage(f"Saved {path}", 4000)

    def _open_spec(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open specification", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                spec = FilterSpec.from_json(handle.read())
            spec.validate()
        except (OSError, SpecError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Could not open", str(exc))
            return
        self.design_panel.setSpec(spec)
        self.status.showMessage(f"Loaded {path}", 4000)

    def _export_plot(self) -> None:
        canvas = self.plot_tabs.currentWidget()
        if not isinstance(canvas, PlotCanvas):
            return
        name = self._design.spec.name if self._design else "plot"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plot",
            f"{name}_{self.plot_tabs.tabText(self.plot_tabs.currentIndex()).lower()}.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            canvas.save(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not export", str(exc))
            return
        self.status.showMessage(f"Exported {path}", 4000)

    def _export_coefficients(self) -> None:
        if self._design is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export coefficients", f"{self._design.spec.name}.csv", "CSV (*.csv)"
        )
        if not path:
            return
        fd = self._design
        quantized = self.fixed_point_panel.quantized()
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                if fd.is_fir:
                    if quantized is not None:
                        handle.write("index,ideal,quantized,integer\n")
                        for i, value in enumerate(fd.b):
                            handle.write(
                                f"{i},{value!r},{quantized.taps[i]!r},"
                                f"{int(quantized.int_taps[i])}\n"
                            )
                    else:
                        handle.write("index,tap\n")
                        for i, value in enumerate(fd.b):
                            handle.write(f"{i},{value!r}\n")
                else:
                    handle.write("section,b0,b1,b2,a0,a1,a2\n")
                    for i, row in enumerate(fd.sos):
                        handle.write(
                            f"{i}," + ",".join(repr(float(v)) for v in row) + "\n"
                        )
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not export", str(exc))
            return
        self.status.showMessage(f"Exported {path}", 4000)

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About",
            "<h3>Digital Filter Engine</h3>"
            "<p>Design, inspect and export digital filters for software-defined "
            "radio and FPGA targets.</p>"
            "<p>Built on NumPy and SciPy. Generates standalone Python, GNU Radio "
            "blocks, Xilinx coefficient files, Verilog and VHDL.</p>"
            f"<p style='color:grey'>Qt binding: {QT_API}</p>",
        )


def _no_design(figure) -> None:
    ax = figure.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        "The current specification does not describe a valid filter.\n"
        "See the message in the status bar.",
        ha="center",
        va="center",
        alpha=0.6,
    )
    ax.set_axis_off()


def main(argv: list[str] | None = None) -> int:
    """Run the application."""
    import sys

    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("Digital Filter Engine")
    window = MainWindow()
    window.show()
    return app.exec()
