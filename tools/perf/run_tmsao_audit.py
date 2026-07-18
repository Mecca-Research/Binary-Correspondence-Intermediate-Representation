#!/usr/bin/env python3
"""Repository wrapper for the bounded BCIR TMSAO audit."""

from __future__ import annotations

import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bcir.performance_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
