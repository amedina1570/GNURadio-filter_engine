"""Qt binding shim.

The application runs on PySide6 or PyQt6.  Importing everything through this
module means the rest of the GUI never names a binding, and matplotlib is
told which one to use before it picks for itself -- get that order wrong and
it can load a second, different binding into the same process.

Set ``FILTER_ENGINE_QT_API`` to force a choice.
"""

from __future__ import annotations

import os

__all__ = [
    "QT_API",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Qt",
    "Signal",
    "Slot",
    "FigureCanvas",
    "NavigationToolbar",
]


def _load():
    """Import a Qt binding and return the pieces the GUI needs."""
    forced = os.environ.get("FILTER_ENGINE_QT_API", "").lower()
    candidates = ["pyside6", "pyqt6"]
    if forced:
        if forced not in candidates:
            raise ImportError(
                f"FILTER_ENGINE_QT_API={forced!r} is not supported; "
                f"choose one of {candidates}"
            )
        candidates = [forced]

    errors: list[str] = []
    for api in candidates:
        try:
            if api == "pyside6":
                from PySide6 import QtCore, QtGui, QtWidgets

                return api, QtCore, QtGui, QtWidgets, QtCore.Signal, QtCore.Slot
            from PyQt6 import QtCore, QtGui, QtWidgets

            return api, QtCore, QtGui, QtWidgets, QtCore.pyqtSignal, QtCore.pyqtSlot
        except ImportError as exc:
            errors.append(f"  {api}: {exc}")

    raise ImportError(
        "No usable Qt binding was found. Install one with:\n"
        "    pip install PySide6\n"
        "Tried:\n" + "\n".join(errors)
    )


QT_API, QtCore, QtGui, QtWidgets, Signal, Slot = _load()
Qt = QtCore.Qt

# matplotlib consults QT_API when it chooses a Qt backend; setting it here,
# after we know which binding loaded, keeps the two in step.
os.environ["QT_API"] = QT_API

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
