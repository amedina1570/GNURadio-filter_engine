"""Qt binding shim.

The application runs on PySide6 or PyQt6.  Importing everything through this
module means the rest of the GUI never names a binding, and matplotlib is
told which one to use before it picks for itself -- get that order wrong and
it can load a second, different binding into the same process.

Set ``FILTER_ENGINE_QT_API`` to force a choice. Otherwise an existing Qt
binding or matplotlib's ``QT_API`` setting is used.
"""

from __future__ import annotations

import os
import sys

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
    requested = forced or os.environ.get("QT_API", "").lower()
    candidates = ["pyside6", "pyqt6"]
    loaded = [
        api
        for api, module in (("pyside6", "PySide6.QtCore"), ("pyqt6", "PyQt6.QtCore"))
        if module in sys.modules
    ]
    if len(loaded) > 1:
        raise ImportError("Both PySide6 and PyQt6 are already loaded; use one Qt binding.")
    if requested:
        if requested not in candidates:
            raise ImportError(
                f"Qt binding {requested!r} is not supported; "
                f"choose one of {candidates}"
            )
        if loaded and loaded[0] != requested:
            raise ImportError(
                f"{loaded[0]} is already loaded, but {requested} was requested. "
                "Use the same Qt binding throughout the process."
            )
        candidates = [requested]
    elif loaded:
        candidates = loaded

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
