"""Three alternating before/after repetitions of the unchanged structure harness.

Use an output directory separate from the source trees. Run no model/test gates
concurrently. Each subprocess prepares its own fixture outside its call timers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stages = ("gem", "asn1", "crc", "frontier", "kernels", "attention", "its")
    records = []
    for rep in range(1, 4):
        order = ("before", "after") if rep % 2 else ("after", "before")
        for side in order:
            repo = getattr(args, side).resolve()
            source = repo / "docs/research/measurements/2026-10-10/run_performance_tests.py"
            for stage in stages:
                directory = args.output / "structures" / f"r{rep}" / side / stage
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(source), stage, "--output", str(directory)]
                started = time.perf_counter()
                with (directory / "stdout.log").open("w") as log:
                    result = subprocess.run(command, cwd=repo, stdout=log, stderr=log)
                records.append(
                    {
                        "repetition": rep,
                        "revision": side,
                        "stage": stage,
                        "command": command,
                        "exit_code": result.returncode,
                        "seconds": time.perf_counter() - started,
                    }
                )
                (args.output / "structure-progress.json").write_text(
                    json.dumps(records, indent=2, sort_keys=True) + "\n"
                )
                print(json.dumps(records[-1]), flush=True)
                if result.returncode:
                    raise RuntimeError(f"{side} {stage} failed")


if __name__ == "__main__":
    main()
