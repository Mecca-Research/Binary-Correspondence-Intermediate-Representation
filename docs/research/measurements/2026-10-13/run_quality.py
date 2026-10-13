"""Full matched-thread LM and copy runs, then native identity and phase breakdown.

Run only after run_matrix.py finishes. No heavy gate runs concurrently.
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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    adapter = Path(__file__).with_name("rerun_training.py")
    source = args.repo / "docs/research/measurements/2026-10-11/train_comparison.py"
    observations = []

    def run(command, name):
        directory = args.output / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        # Emit validation progress while retaining the complete subprocess output.
        with (directory / f"{name}.log").open("w") as log:
            child = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = child.wait()
        observations.append(
            {
                "name": name,
                "command": command,
                "exit_code": code,
                "seconds": time.perf_counter() - started,
            }
        )
        (args.output / "quality-progress.json").write_text(
            json.dumps(observations, indent=2, sort_keys=True) + "\n"
        )
        if code:
            raise RuntimeError(f"{name} exited {code}")

    for task in ("lm", "copy"):
        # Reverse backend order on the second task to expose drift rather than hide it.
        backends = ("bcir", "torch-compiled")
        for backend in backends if task == "lm" else reversed(backends):
            name = f"{task}-{backend}"
            command = [
                sys.executable, str(adapter), "--repo", str(args.repo),
                "--data", str(args.data), "--output", str(args.output / "full" / name),
                "--backend", backend, "--contract", "kblock+exp" if backend == "bcir" else "v1",
                "--threads", str(args.threads), "--steps", "1000", "--eval-every", "100",
                "--task", task, "--lr-scale", "0.25",
            ]  # fmt: skip
            command.append("--save-checkpoint")
            run(command, name)
    run(
        [
            sys.executable,
            str(source),
            "identity",
            "--data",
            str(args.data),
            "--steps",
            "8",
            "--cc",
            "gcc",
            "--threads",
            "1,2,4,8",
            "--contract",
            "v1,kblock+exp",
            "--variants",
            "widest,avx2",
        ],  # fmt: skip
        "identity",
    )
    breakdowns = args.output / "breakdowns"
    breakdowns.mkdir(exist_ok=True)
    for contract, threads in (("v1", 1), ("kblock+exp", 1), ("kblock+exp", 4), ("kblock+exp", 8)):
        name = f"breakdown-{contract}-t{threads}"
        run(
            [
                sys.executable,
                str(source),
                "bcir-breakdown",
                "--data",
                str(args.data),
                "--rounds",
                "5",
                "--cc",
                "gcc",
                "--threads",
                str(threads),
                "--contract",
                contract,
            ],  # fmt: skip
            name,
        )
        emitted = args.data / f"breakdown-bcir-{contract}-t{threads}.json"
        (breakdowns / f"{name}.json").write_bytes(emitted.read_bytes())


if __name__ == "__main__":
    main()
