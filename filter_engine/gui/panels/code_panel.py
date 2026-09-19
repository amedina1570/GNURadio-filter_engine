"""Generated code: pick a target, read it, copy it, save it.

Includes a lightweight syntax highlighter.  A wall of undifferentiated
monospace is hard to scan, and the point of this panel is that the user
actually reads what they are about to paste into a flowgraph.
"""

from __future__ import annotations

import re

from ... import codegen
from ...core.design import FilterDesign
from ...core.quantize import QuantizedFilter
from ..qt import Qt, QtGui, QtWidgets

__all__ = ["CodePanel"]

_KEYWORDS = {
    "python": (
        "and as assert async await break class continue def del elif else except "
        "finally for from global if import in is lambda nonlocal not or pass raise "
        "return try while with yield True False None self"
    ).split(),
    "verilog": (
        "module endmodule input output wire reg signed always assign parameter "
        "localparam integer begin end if else for initial posedge negedge "
        "default_nettype"
    ).split(),
    "vhdl": (
        "library use entity architecture port is begin end process signal constant "
        "variable type array of downto to in out if elsif else then loop for "
        "rising_edge package others"
    ).split(),
    "text": [],
}


class _Highlighter(QtGui.QSyntaxHighlighter):
    """Keyword, string, number and comment colouring for the shown language."""

    def __init__(self, document, language: str = "python") -> None:
        super().__init__(document)
        self.set_language(language)

    def set_language(self, language: str) -> None:
        self._language = language
        # Two palettes: the light-theme colours are too dark to read on a dark
        # background and vice versa, so pick from the widget's own background.
        base = QtGui.QGuiApplication.palette().color(
            QtGui.QPalette.ColorRole.Base
        )
        dark = base.lightness() < 128
        if dark:
            keyword, string, number, comment = (
                "#569cd6", "#ce9178", "#b5cea8", "#6a9955",
            )
        else:
            keyword, string, number, comment = (
                "#0b6cbf", "#a03030", "#7d5fb2", "#4b7a3a",
            )

        keyword_format = QtGui.QTextCharFormat()
        keyword_format.setForeground(QtGui.QColor(keyword))
        keyword_format.setFontWeight(QtGui.QFont.Weight.Bold)

        string_format = QtGui.QTextCharFormat()
        string_format.setForeground(QtGui.QColor(string))

        number_format = QtGui.QTextCharFormat()
        number_format.setForeground(QtGui.QColor(number))

        comment_format = QtGui.QTextCharFormat()
        comment_format.setForeground(QtGui.QColor(comment))
        comment_format.setFontItalic(True)

        rules: list[tuple[re.Pattern, QtGui.QTextCharFormat]] = []
        words = _KEYWORDS.get(language, [])
        if words:
            pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
            rules.append((re.compile(pattern), keyword_format))
        rules.append((re.compile(r"\b\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?\b"), number_format))
        rules.append((re.compile(r"'[^'\n]*'|\"[^\"\n]*\""), string_format))

        if language == "python":
            rules.append((re.compile(r"#[^\n]*"), comment_format))
        elif language in ("verilog",):
            rules.append((re.compile(r"//[^\n]*"), comment_format))
        elif language == "vhdl":
            rules.append((re.compile(r"--[^\n]*"), comment_format))
        else:
            rules.append((re.compile(r"[;#][^\n]*"), comment_format))

        self._rules = rules
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt naming)
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class CodePanel(QtWidgets.QWidget):
    """Target selector plus the generated source."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._design: FilterDesign | None = None
        self._quantized: QuantizedFilter | None = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Target"))

        self.target_combo = QtWidgets.QComboBox()
        for target in codegen.TARGETS:
            self.target_combo.addItem(target.label, target)
        self.target_combo.currentIndexChanged.connect(self.refresh)
        self.target_combo.setMinimumWidth(240)
        bar.addWidget(self.target_combo)

        self.description = QtWidgets.QLabel()
        self.description.setStyleSheet("font-style: italic;")
        self.description.setWordWrap(True)
        bar.addWidget(self.description, 1)

        self.copy_button = QtWidgets.QPushButton("Copy")
        self.copy_button.clicked.connect(self._copy)
        bar.addWidget(self.copy_button)

        self.save_button = QtWidgets.QPushButton("Save as...")
        self.save_button.clicked.connect(self._save)
        bar.addWidget(self.save_button)

        self.save_all_button = QtWidgets.QPushButton("Export all...")
        self.save_all_button.setToolTip(
            "Write every applicable target into one folder."
        )
        self.save_all_button.clicked.connect(self._save_all)
        bar.addWidget(self.save_all_button)
        layout.addLayout(bar)

        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )
        font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        font.setPointSize(max(font.pointSize(), 10))
        self.editor.setFont(font)
        self.editor.setTabStopDistance(
            4 * QtGui.QFontMetricsF(font).horizontalAdvance(" ")
        )
        self._highlighter = _Highlighter(self.editor.document(), "python")
        layout.addWidget(self.editor)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("font-style: italic;")
        layout.addWidget(self.status)

    # ------------------------------------------------------------- public API
    def setDesign(self, design: FilterDesign | None) -> None:
        self._design = design
        self._update_target_availability()
        self.refresh()

    def setQuantized(self, quantized: QuantizedFilter | None) -> None:
        self._quantized = quantized
        target = self.target_combo.currentData()
        # Only the fixed-point targets change when the quantization changes.
        if target is not None and target.needs_quantized:
            self.refresh()

    def currentCode(self) -> str:
        return self.editor.toPlainText()

    # --------------------------------------------------------------- internals
    def _update_target_availability(self) -> None:
        """Grey out targets that cannot describe the current design."""
        model = self.target_combo.model()
        for row in range(self.target_combo.count()):
            target = self.target_combo.itemData(row)
            usable = self._design is None or target.supports(self._design)
            item = model.item(row)
            if item is not None:
                item.setEnabled(usable)
                item.setToolTip(
                    ""
                    if usable
                    else f"{target.label} only describes FIR filters."
                )
        current = self.target_combo.currentData()
        if (
            self._design is not None
            and current is not None
            and not current.supports(self._design)
        ):
            self.target_combo.setCurrentIndex(0)

    def refresh(self, *_args) -> None:
        target = self.target_combo.currentData()
        if target is None:
            return
        self.description.setText(target.description)
        self._highlighter.set_language(target.language)

        if self._design is None:
            self.editor.setPlainText("# Design a filter first.")
            self.status.setText("")
            return
        if not target.supports(self._design):
            self.editor.setPlainText(
                f"# {target.label} only describes FIR filters.\n"
                "# This design is IIR; use the Verilog or VHDL target for its\n"
                "# biquad coefficients."
            )
            self.status.setText("")
            return
        if target.needs_quantized and self._quantized is None:
            self.editor.setPlainText(
                "# This target needs fixed-point coefficients.\n"
                "# Open the Fixed point tab and choose a word length."
            )
            self.status.setText("")
            return

        try:
            code = codegen.generate(target.key, self._design, self._quantized)
        except Exception as exc:
            self.editor.setPlainText(f"# Generation failed:\n# {exc}")
            self.status.setText("")
            return

        self.editor.setPlainText(code)
        self.status.setText(
            f"{len(code.splitlines()):,} lines, {len(code):,} characters - "
            f"suggested extension {target.extension}"
        )

    def _copy(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.editor.toPlainText())
        self.status.setText("Copied to the clipboard.")

    def _suggested_name(self, target) -> str:
        base = self._design.spec.name if self._design else "filter"
        if target.key == "gnuradio":
            base = f"{base}_gr"
        return f"{base}{target.extension}"

    def _save(self) -> None:
        target = self.target_combo.currentData()
        if target is None or self._design is None:
            return
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save generated code", self._suggested_name(target)
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.editor.toPlainText())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.status.setText(f"Saved to {path}")

    def _save_all(self) -> None:
        """Write every applicable target into a chosen folder."""
        if self._design is None:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export every target into a folder"
        )
        if not folder:
            return

        import os

        written: list[str] = []
        skipped: list[str] = []
        for target in codegen.TARGETS:
            if not target.supports(self._design):
                skipped.append(f"{target.label}: FIR only")
                continue
            if target.needs_quantized and self._quantized is None:
                skipped.append(f"{target.label}: needs a word length")
                continue
            try:
                code = codegen.generate(target.key, self._design, self._quantized)
                path = os.path.join(folder, self._suggested_name(target))
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(code)
                written.append(os.path.basename(path))
            except Exception as exc:
                skipped.append(f"{target.label}: {exc}")

        message = f"Wrote {len(written)} file(s) to {folder}:\n  " + "\n  ".join(written)
        if skipped:
            message += "\n\nSkipped:\n  " + "\n  ".join(skipped)
        QtWidgets.QMessageBox.information(self, "Export complete", message)
        self.status.setText(f"Exported {len(written)} file(s) to {folder}")
