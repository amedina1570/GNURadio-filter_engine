"""A matplotlib canvas with a toolbar, and lazy redrawing.

A 700-tap filter takes real time to plot, and the design panel emits a change
on every keystroke.  Two things keep the UI responsive: a canvas only redraws
when it is actually visible (:meth:`PlotCanvas.invalidate` marks it dirty
instead), and the main window coalesces rapid edits behind a short timer.
"""

from __future__ import annotations

from typing import Callable

from matplotlib.figure import Figure

from ..qt import FigureCanvas, NavigationToolbar, QtWidgets

__all__ = ["PlotCanvas"]

#: Shared styling so every plot in the app reads as one set.
GRID_STYLE = dict(alpha=0.25, linewidth=0.6)
SPEC_MASK_STYLE = dict(color="tab:red", alpha=0.12)
IDEAL_STYLE = dict(color="tab:blue", linewidth=1.4)
QUANTIZED_STYLE = dict(color="tab:orange", linewidth=1.1, alpha=0.9)


class PlotCanvas(QtWidgets.QWidget):
    """One figure, its toolbar, and a draw callback.

    The owner supplies ``draw_fn(figure)``; the canvas decides when to call
    it.
    """

    def __init__(
        self,
        draw_fn: Callable[[Figure], None],
        parent: QtWidgets.QWidget | None = None,
        toolbar: bool = True,
    ) -> None:
        super().__init__(parent)
        self._draw_fn = draw_fn
        self._dirty = True

        self.figure = Figure(figsize=(7, 5), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(320, 240)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None
        layout.addWidget(self.canvas)

    # ------------------------------------------------------------------
    def invalidate(self) -> None:
        """Mark the plot stale; it redraws when next shown."""
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def refresh(self, force: bool = False) -> None:
        """Redraw if stale (or if ``force``)."""
        if not (self._dirty or force):
            return
        self.figure.clear()
        try:
            self._draw_fn(self.figure)
        except Exception as exc:  # a bad design must not take the window down
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Could not plot:\n{exc}",
                ha="center",
                va="center",
                wrap=True,
                fontsize=9,
                color="tab:red",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        self._dirty = False
        self.canvas.draw_idle()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        # Redraw on becoming visible, which is what makes lazy drawing work.
        self.refresh()

    def save(self, path: str, dpi: int = 150) -> None:
        self.refresh()
        self.figure.savefig(path, dpi=dpi)
