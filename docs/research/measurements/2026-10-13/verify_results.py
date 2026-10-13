"""Serialize the available local gates and record unsupported-toolchain failures.

This research PR does not change the production sources. A failing existing
optional-provider assertion is reported, not silently skipped or relaxed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = next(p for p in Path(__file__).resolve().parents if (p / "bcir").is_dir())
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        OPENBLAS_THREAD_TIMEOUT="4", CMAKE_BUILD_PARALLEL_LEVEL="2",
    )  # fmt: skip
    commands = [
        ("focused", [sys.executable, "-m", "pytest", "-q", "bcir/tests/test_native_decoder.py",
                     "bcir/tests/test_hosted_training_pipeline.py", "bcir/tests/test_train_c_kernels.py"]),
        ("quick", [sys.executable, "-m", "bcir.tests.run_all", "--tier", "quick", "-j", "2"]),
        ("latest-status", [sys.executable, "tools/local/latest_toolchain.py", "status"]),
        ("latest-confirm", ["bash", "tools/local/check_latest.sh", "--legs", "status,confirm",
                            "--out", str(args.output / "latest")]),
        ("status", [sys.executable, "tools/docs/gen_status.py", "--check"]),
        ("links", [sys.executable, "tools/docs/check_links.py"]),
        ("retired-paths", [sys.executable, "tools/docs/check_retired_paths.py"]),
        ("claims", [sys.executable, "tools/docs/check_claims.py"]),
        ("law-range", [sys.executable, "tools/docs/check_law_range.py"]),
        ("imports", [sys.executable, "tools/perf/import_graph.py", "--check"]),
        ("inventory", [sys.executable, "tools/perf/oracle_inventory.py", "--check"]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "--force-exclude", "."]),
        ("lint", [sys.executable, "-m", "ruff", "check", "--force-exclude", "."]),
        ("diff", ["git", "diff", "--check"]),
    ]  # fmt: skip
    records = []
    for name, command in commands:
        started = time.perf_counter()
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=log)
        record = {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "seconds": time.perf_counter() - started,
            "tail": (args.output / f"{name}.log").read_text()[-2500:],
        }
        records.append(record)
        (args.output / "verification.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
