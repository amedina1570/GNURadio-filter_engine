"""Input widgets tuned for DSP parameters.

The one that matters is :class:`FrequencyEdit`.  SDR sample rates are things
like 61440000, and typing that into a spin box is miserable and error-prone,
so this accepts engineering notation -- ``61.44M``, ``48k``, ``2.4 GHz`` --
and redisplays whatever you type in a readable form.
"""

from __future__ import annotations

import re

from ..qt import QtGui, QtWidgets, Signal

__all__ = [
    "FrequencyEdit",
    "parse_frequency",
    "format_frequency",
    "combo_enum",
    "select_data",
]


def combo_enum(combo, enum_cls, fallback=None):
    """Read a combo box's current data back as ``enum_cls``.

    Qt stores item data as a QVariant, and our enums subclass ``str`` so that
    they serialise readably -- which means Qt hands them back as plain
    strings.  That matters more than it looks: ``Enum.__hash__`` is derived
    from the member *name*, so a returned ``"lowpass"`` is not equal to
    ``Response.LOWPASS`` for set membership, and any ``in`` test against a set
    of members quietly returns False.  Every read of an enum-valued combo goes
    through here.
    """
    data = combo.currentData()
    if isinstance(data, enum_cls):
        return data
    try:
        return enum_cls(data)
    except (ValueError, TypeError):
        return fallback


def select_data(combo, value) -> None:
    """Select the item whose data matches ``value``, enum or plain."""
    raw = getattr(value, "value", value)
    index = combo.findData(raw)
    if index < 0:
        index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)

_SUFFIXES = {
    "": 1.0,
    "k": 1e3,
    "m": 1e6,
    "g": 1e9,
    "t": 1e12,
}

_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([kKmMgGtT]?)\s*(?:hz)?\s*$",
    re.IGNORECASE,
)


class FrequencyParseError(ValueError):
    """Raised when text cannot be read as a frequency."""


def parse_frequency(text: str) -> float:
    """Read ``text`` as a frequency in Hz.

    Accepts a bare number, a k/M/G/T suffix, an optional ``Hz``, and
    scientific notation.  ``10M``, ``10 MHz``, ``10e6`` and ``10000000`` are
    all the same value.
    """
    match = _PATTERN.match(text.replace(",", "").replace("_", ""))
    if not match:
        raise FrequencyParseError(
            f"{text!r} is not a frequency. Try 1000, 48k, 10M or 2.4G."
        )
    mantissa, suffix = match.groups()
    return float(mantissa) * _SUFFIXES[suffix.lower()]


def format_frequency(value: float, decimals: int = 6) -> str:
    """Render ``value`` Hz compactly, with a k/M/G suffix where it helps."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    for threshold, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if magnitude >= threshold:
            scaled = value / threshold
            text = f"{scaled:.{decimals}g}"
            return f"{text}{suffix}"
    return f"{value:.{decimals}g}"


class FrequencyEdit(QtWidgets.QLineEdit):
    """A line edit holding a frequency in Hz.

    Emits :attr:`valueChanged` only when the text parses, so a half-typed
    ``10M`` never triggers a redesign at 1 Hz.
    """

    valueChanged = Signal(float)

    def __init__(self, value: float = 0.0, parent=None) -> None:
        super().__init__(parent)
        self._value = float(value)
        self.setText(format_frequency(value))
        self.setPlaceholderText("e.g. 10M")
        self.setClearButtonEnabled(True)
        self.textEdited.connect(self._on_edited)
        self.editingFinished.connect(self._on_finished)
        self._normal_palette = self.palette()

    # ------------------------------------------------------------------
    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        """Set the value without emitting a change signal."""
        self._value = float(value)
        blocked = self.blockSignals(True)
        self.setText(format_frequency(value))
        self.blockSignals(blocked)
        self._mark_valid(True)

    # ------------------------------------------------------------------
    def _on_edited(self, text: str) -> None:
        try:
            value = parse_frequency(text)
        except FrequencyParseError:
            self._mark_valid(False)
            return
        self._mark_valid(True)
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)

    def _on_finished(self) -> None:
        """Rewrite the field in canonical form once the user moves on."""
        try:
            value = parse_frequency(self.text())
        except FrequencyParseError:
            # Put back the last good value rather than leaving nonsense.
            self.setValue(self._value)
            return
        self.setValue(value)

    def _mark_valid(self, valid: bool) -> None:
        if valid:
            self.setPalette(self._normal_palette)
            self.setToolTip("")
            return
        palette = QtGui.QPalette(self._normal_palette)
        palette.setColor(
            QtGui.QPalette.ColorRole.Text, QtGui.QColor("#c0392b")
        )
        self.setPalette(palette)
        self.setToolTip("Not a frequency. Try 1000, 48k, 10M or 2.4G.")


class LabelledRow:
    """Helper for building form rows that can be shown and hidden together."""

    def __init__(
        self, form: QtWidgets.QFormLayout, label: str, widget: QtWidgets.QWidget
    ) -> None:
        self.label = QtWidgets.QLabel(label)
        self.widget = widget
        form.addRow(self.label, widget)

    def setVisible(self, visible: bool) -> None:
        self.label.setVisible(visible)
        self.widget.setVisible(visible)
