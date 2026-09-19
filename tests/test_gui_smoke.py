"""Headless exercise of the GUI.

Runs under Qt's ``offscreen`` platform, so it needs no display and works in
CI.  The point is not to check pixels but to prove the wiring holds: a spec
change reaches every panel, every plot draws, every preset designs, and the
code panel produces something for each target.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="a Qt binding is required for the GUI tests")

from filter_engine.gui.app import MainWindow  # noqa: E402
from filter_engine.gui.qt import QtWidgets  # noqa: E402
from filter_engine.gui.widgets.fields import (  # noqa: E402
    FrequencyParseError,
    format_frequency,
    parse_frequency,
)
from filter_engine.presets import FILTER_PRESETS  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


@pytest.fixture
def window(app):
    win = MainWindow()
    # The window debounces redesigns behind a timer; fire it directly so the
    # test does not have to spin an event loop.
    win._redesign()
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def test_window_starts_with_a_valid_design(window):
    assert window._design is not None
    assert window._design.num_taps > 0
    assert "taps" in window.summary_label.text()


def test_every_plot_draws(window):
    for label, canvas in window.canvases.items():
        canvas.refresh(force=True)
        assert canvas.figure.axes, f"{label} produced no axes"
        # The canvas swallows draw errors into a red message; make sure none
        # of the plots took that path.
        texts = [
            t.get_text()
            for ax in canvas.figure.axes
            for t in ax.texts
        ]
        assert not any("Could not plot" in t for t in texts), f"{label} failed"


@pytest.mark.parametrize("preset", FILTER_PRESETS, ids=lambda p: p.name)
def test_every_preset_designs_through_the_gui(window, preset):
    window.design_panel.setSpec(preset.spec.copy())
    window._redesign()
    assert window._design is not None, preset.name
    for canvas in window.canvases.values():
        canvas.refresh(force=True)


def test_code_panel_produces_output_for_every_target(window):
    panel = window.code_panel
    for row in range(panel.target_combo.count()):
        target = panel.target_combo.itemData(row)
        panel.target_combo.setCurrentIndex(row)
        panel.refresh()
        code = panel.currentCode()
        assert code.strip(), f"{target.label} produced nothing"
        if target.supports(window._design):
            assert "Generation failed" not in code, f"{target.label}: {code[:200]}"


def test_fixed_point_panel_reports_a_usable_format(window):
    panel = window.fixed_point_panel
    quantized = panel.quantized()
    assert quantized is not None
    assert quantized.fmt.total_bits == 16
    assert panel.quant_table.rowCount() > 0
    assert panel.fpga_table.rowCount() > 0
    assert "dB" in panel.headline.text()


def test_pulse_panel_filters_every_excitation(window):
    from filter_engine.core.signals import PulseKind

    panel = window.pulse_panel
    assert panel.kind_combo.count() == len(PulseKind)
    for row in range(panel.kind_combo.count()):
        panel.kind_combo.setCurrentIndex(row)
        # Qt returns the str-based enum as a plain string; the panel coerces
        # it back, and that coercion is what the rest of the code relies on.
        kind = panel._kind()
        assert isinstance(kind, PulseKind)
        assert kind.value == panel.kind_combo.itemData(row)
        assert panel._generated is not None, f"{kind.value} generated nothing"
        assert panel._output is not None, f"{kind.value} was not filtered"
        assert panel._output.size == panel._generated.x.size
        panel.time_canvas.refresh(force=True)
        panel.spec_canvas.refresh(force=True)


def test_invalid_spec_reports_an_error_rather_than_crashing(window):
    spec = window._design.spec.copy(f_low=10e6)  # far beyond Nyquist
    window.design_panel.setSpec(spec)
    window._redesign()
    assert window._design is None
    assert "Nyquist" in window.summary_label.text()
    # The plots must still render their "no design" placeholder.
    for canvas in window.canvases.values():
        canvas.refresh(force=True)


def test_iir_design_disables_fir_only_targets(window):
    from filter_engine.core.spec import FilterFamily, IirMethod

    spec = window._design.spec.copy(
        family=FilterFamily.IIR, iir_method=IirMethod.ELLIP
    )
    window.design_panel.setSpec(spec)
    window._redesign()
    assert window._design is not None
    assert not window._design.is_fir

    panel = window.code_panel
    model = panel.target_combo.model()
    for row in range(panel.target_combo.count()):
        target = panel.target_combo.itemData(row)
        enabled = model.item(row).isEnabled()
        assert enabled == target.supports(window._design), target.label


# --------------------------------------------------------------------------
# Frequency field
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("1000", 1000.0),
        ("48k", 48_000.0),
        ("10M", 10e6),
        ("10 MHz", 10e6),
        ("61.44M", 61.44e6),
        ("2.4G", 2.4e9),
        ("1e6", 1e6),
        ("-100k", -100e3),
        ("1,000,000", 1e6),
        ("  250 kHz  ", 250e3),
    ],
)
def test_frequency_parsing(text, expected):
    assert parse_frequency(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "10 XHz", "--5", "k"])
def test_frequency_parsing_rejects_nonsense(text):
    with pytest.raises(FrequencyParseError):
        parse_frequency(text)


@pytest.mark.parametrize(
    "value,expected",
    [(0, "0"), (1000, "1k"), (61.44e6, "61.44M"), (2.4e9, "2.4G"), (250, "250")],
)
def test_frequency_formatting_round_trips(value, expected):
    text = format_frequency(value)
    assert text == expected
    assert parse_frequency(text) == pytest.approx(value)
