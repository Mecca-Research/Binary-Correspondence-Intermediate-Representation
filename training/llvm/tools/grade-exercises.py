#!/usr/bin/env python3
"""Deterministic, dependency-free grader for training/llvm exercise artifacts.

The grading itself lives in `training/tools/grading.py`, which is shared by
every subject; this script is the LLVM half -- the answer kinds, the tools, and
the `lli` harness, all of them in `llvm_profile.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from llvm_profile import PROFILE  # noqa: E402
import grading  # noqa: E402

if __name__ == "__main__":
    grading.run(PROFILE, __doc__ or "")
