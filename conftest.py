"""Make the package importable when running pytest from the repo root."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Plots must never try to open a window during tests.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
