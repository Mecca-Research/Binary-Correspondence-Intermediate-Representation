"""The 2026-10-10 kernels stage's portable rows, against the builds the trainer actually shipped.

That stage compiled the base commit's `bcir_tensor.c` with `-O3 -march=native -fno-math-errno
-ffp-contract=off`. The trainer never was: `NativeDecoder.build` used `-O3 -ffp-contract=off` and
the gates their default `-O2`. This builds the base kernel all three ways and times the stage's
four shapes through `bcir_tensor_mm`, one thread, the median of eleven calls each, and requires
the three builds' outputs to be identical (the flags change code, never a result). Needs GCC and
NumPy; a measurement, not a gate.

  python march_check.py --output DIR [--base 75d50d4] [--cc gcc]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import time
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "bcir").is_dir())
BUILDS = {
    "2026-10-10 harness": ("-O3", "-march=native", "-fno-math-errno", "-ffp-contract=off"),
    "NativeDecoder.build": ("-O3", "-ffp-contract=off"),
    "gates (default -O2)": ("-O2", "-ffp-contract=off"),
}
SHAPES = ((64, 64, 48, 0), (256, 384, 384, 0), (256, 1024, 384, 1), (384, 384, 256, 0))


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="75d50d4f6e280e35a25032024edffd92f25c52f2")
    parser.add_argument("--cc", default="gcc")
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ("bcir_tensor.c", "bcir_tensor.h"):
        (a.output / name).write_bytes(
            subprocess.run(
                ["git", "-C", str(REPO), "show", f"{a.base}:runtime/c/{name}"],
                check=True, capture_output=True,
            ).stdout
        )  # fmt: skip
    fp = ctypes.POINTER(ctypes.c_float)
    libs = {}
    for i, (label, flags) in enumerate(BUILDS.items()):
        target = a.output / f"tensor{i}.so"
        subprocess.run(
            [a.cc, "-std=c11", *flags, "-shared", "-fPIC", str(a.output / "bcir_tensor.c"),
             "-o", str(target), "-lm"],
            check=True, capture_output=True, timeout=300,
        )  # fmt: skip
        lib = ctypes.CDLL(str(target))
        lib.bcir_tensor_mm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_float, fp, fp, ctypes.c_float, fp]  # fmt: skip
        libs[label] = lib
    rng = np.random.default_rng(713)  # the 2026-10-10 stage's seed and scaling
    rows = []
    for m, n, k, tb in SHAPES:
        a_ = rng.standard_normal((m, k), dtype=np.float32) / 8
        b_ = rng.standard_normal((n, k) if tb else (k, n), dtype=np.float32) / 8
        row = {"m": m, "n": n, "k": k, "transpose_b": tb, "median_ms": {}}
        outputs = set()
        for label, lib in libs.items():
            c = np.zeros((m, n), np.float32)
            args = (None, 0, tb, m, n, k, 1.0, a_.ctypes.data_as(fp), b_.ctypes.data_as(fp),
                    0.0, c.ctypes.data_as(fp))  # fmt: skip
            lib.bcir_tensor_mm(*args)
            times = []
            for _ in range(11):
                t0 = time.perf_counter()
                lib.bcir_tensor_mm(*args)
                times.append(1000 * (time.perf_counter() - t0))
            row["median_ms"][label] = statistics.median(times)
            outputs.add(c.tobytes())
        row["outputs_identical"] = len(outputs) == 1
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"base": a.base, "compiler": subprocess.run([a.cc, "--version"], capture_output=True,
              text=True).stdout.splitlines()[0], "builds": {k: list(v) for k, v in BUILDS.items()},
              "rows": rows}  # fmt: skip
    (a.output / "march.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
