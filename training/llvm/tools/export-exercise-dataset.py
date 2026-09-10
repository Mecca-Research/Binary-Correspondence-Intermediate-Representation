#!/usr/bin/env python3
"""Export graded exercises as a deterministic JSON Lines evaluation dataset.

The export itself lives in `training/tools/dataset_export.py` and is shared by
every subject. What is LLVM's here is the version assumption each record
carries: which LLVM the prompts were written against, and whether this
particular exercise also assumes MLIR.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
SHARED_TOOLS = TOOLS_DIR.parent.parent / "tools"
for _path in (str(TOOLS_DIR), str(SHARED_TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import dataset_export  # noqa: E402
from llvm_profile import PROFILE  # noqa: E402

EXPORTER = "training/llvm/tools/export-exercise-dataset.py"
LLVM_ASSUMPTION = "LLVM >= 15 with opaque pointers"
MLIR_ASSUMPTION = "MLIR >= 15"


def version_assumptions(manifest: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """An exercise assumes MLIR if it reviews it, runs its tools, or names its files."""
    required_tools = list(manifest["required_tools"])
    uses_mlir = (
        manifest["answer_kind"] == "mlir_review"
        or "mlir-opt" in required_tools
        or any(path.endswith(".mlir") for path in paths)
    )
    return {
        "llvm": LLVM_ASSUMPTION,
        "mlir": MLIR_ASSUMPTION if uses_mlir else None,
        "tools": manifest["minimum_tool_versions"],
    }


if __name__ == "__main__":
    raise SystemExit(
        dataset_export.main(
            PROFILE,
            __doc__ or "",
            version_assumptions=version_assumptions,
            exporter=EXPORTER,
        )
    )
