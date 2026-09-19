#!/usr/bin/env python3
"""Launch the Digital Filter Engine GUI."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from filter_engine.gui.app import main as run
    except ImportError as exc:
        print(f"Could not start the GUI: {exc}\n", file=sys.stderr)
        print("Install the dependencies with:", file=sys.stderr)
        print("    pip install -r requirements.txt", file=sys.stderr)
        return 1
    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
