"""Serialize alternating, matched-thread decoder timings for PR816.

No other benchmark or build/test gate should run concurrently with this driver.
Each observation loads the same initial weights and 32-update schedule prefix.
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--eight-thread-screen", action="store_true")
    args = parser.parse_args()
    configs = [
        ("before-native-v1-t1", args.before, "bcir", "v1", 1),
        ("after-native-v1-t1", args.after, "bcir", "v1", 1),
        ("after-native-kblock-exp-t1", args.after, "bcir", "kblock+exp", 1),
        ("after-native-v1-t4", args.after, "bcir", "v1", 4),
        ("after-native-kblock-exp-t4", args.after, "bcir", "kblock+exp", 4),
        ("after-openblas-kblock-exp-t4", args.after, "bcir-openblas", "kblock+exp", 4),
        ("torch-eager-t1", args.after, "torch-eager", "v1", 1),
        ("torch-eager-t4", args.after, "torch-eager", "v1", 4),
        ("torch-compiled-t1", args.after, "torch-compiled", "v1", 1),
        ("torch-compiled-t4", args.after, "torch-compiled", "v1", 4),
    ]
    if args.eight_thread_screen:
        configs = [
            ("after-native-kblock-exp-t8", args.after, "bcir", "kblock+exp", 8),
            ("torch-eager-t8", args.after, "torch-eager", "v1", 8),
            ("torch-compiled-t8", args.after, "torch-compiled", "v1", 8),
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    observations = []
    for rep in range(args.repetitions):
        order = configs if rep % 2 == 0 else list(reversed(configs))
        for name, repo, backend, contract, threads in order:
            directory = args.output / f"r{rep + 1}" / name
            directory.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).with_name("rerun_training.py")),
                "--repo", str(repo), "--data", str(args.data),
                "--output", str(directory), "--backend", backend,
                "--contract", contract, "--threads", str(threads),
                "--steps", str(args.steps), "--eval-every", str(args.steps),
                "--evaluation-windows", "8", "--lr-scale", "0.25",
            ]  # fmt: skip
            start = time.perf_counter()
            with (directory / "stdout.log").open("w") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            record = {
                "name": name,
                "repetition": rep + 1,
                "command": command,
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - start,
            }
            if result.returncode == 0:
                files = list((directory / "runs").glob("*.json"))
                assert len(files) == 1
                record["record"] = str(files[0])
                data = json.loads(files[0].read_text())
                record["median_step_ms"] = data["median_step_ms"]
                record["state_sha256"] = data["state_sha256"]
            else:
                record["failure_tail"] = (directory / "stdout.log").read_text()[-3000:]
            observations.append(record)
            (args.output / "progress.json").write_text(
                json.dumps(observations, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
