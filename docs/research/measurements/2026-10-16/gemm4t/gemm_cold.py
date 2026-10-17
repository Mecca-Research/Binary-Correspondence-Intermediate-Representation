"""NDT-GEMM-4T: the 9.84M decoder step's GEMM shapes at THREADS threads with warm operands (one
operand set reused) and cold ones (each call on the next of SETS sets, which together exceed the
33 MiB L3), on 2 MB pages, alternated call by call; every result checked bit for bit against the
serial kernel. With a second build B, both builds run cold, A and B on disjoint sets, alternated
call by call, and B's results are checked against A's. The step's GEMM time on each side (every
call of the step at its shape's median), as JSON.

    python3 gemm_cold.py A.so [B.so] THREADS SETS REPS OUT.json [ta]

`ta` keeps the transposed-A shapes only (the weight gradients `X^T dY`).
"""

import ctypes as ct
import json
import mmap
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models.native_program import training_step_program  # noqa: E402

FP = ct.POINTER(ct.c_float)
SIG = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_float]
SIG += [FP, FP, ct.c_float, FP]
HUGE = 2 << 20
KEEP = []


class Build:
    def __init__(self, path, threads):
        self.lib = ct.CDLL(path)
        self.pooled, self.serial = (
            self.lib.bcir_tensor_mm_kblock_pool,
            self.lib.bcir_tensor_mm_kblock,
        )
        for f in (self.pooled, self.serial):
            f.argtypes, f.restype = SIG, None
        self.lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
        self.lib.bcir_tensor_pool_size.restype = ct.c_size_t
        self.lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
        self.lib.bcir_tensor_pool_init.restype = ct.c_void_p
        size = self.lib.bcir_tensor_pool_size(threads)
        self.mem = (ct.c_char * size)()
        self.pool = self.lib.bcir_tensor_pool_init(self.mem, size, threads)


def placed(values):
    m = mmap.mmap(-1, values.nbytes + HUGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m.madvise(mmap.MADV_HUGEPAGE)
    KEEP.append(m)
    raw = np.frombuffer(m, dtype=np.uint8)
    lead = -raw.ctypes.data % HUGE
    out = raw[lead : lead + values.nbytes].view(np.float32).reshape(values.shape)
    out[...] = values
    return out


def ptr(x):
    return x.ctypes.data_as(FP)


def main(argv) -> int:
    args = argv[1:]
    ta_only = args[-1] == "ta"
    if ta_only:
        args = args[:-1]
    two = len(args) == 6
    paths = args[:2] if two else args[:1]
    threads, sets, reps, out = int(args[-4]), int(args[-3]), int(args[-2]), Path(args[-1])
    builds = [Build(p, threads) for p in paths]
    sides = ("A", "B") if two else ("warm", "cold")
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4)
    calls = Counter(g for step in training_step_program(spec, 8, 128).steps for g in step.gemms)
    rng = np.random.default_rng(3)
    rows, mismatch = [], 0
    for (ta, tb, m, n, k), count in calls.items():
        if ta_only and not ta:
            continue
        group = [(placed((rng.standard_normal((k, m) if ta else (m, k)) * 0.5).astype(np.float32)),
                  placed((rng.standard_normal((n, k) if tb else (k, n)) * 0.5).astype(np.float32)),
                  placed(np.zeros((m, n), np.float32))) for _ in range(2 * sets)]  # fmt: skip
        times = {side: [] for side in sides}
        nxt = {sides[0]: 0, sides[1]: sets}
        for r in range(reps):
            for side in sides if r % 2 == 0 else sides[::-1]:
                build = builds[sides.index(side)] if two else builds[0]
                if not two and side == "warm":
                    x, y, c = group[0]
                else:
                    base = 0 if side == sides[0] else sets
                    nxt[side] = base + (nxt[side] - base + 1) % sets
                    x, y, c = group[nxt[side]]
                t0 = time.perf_counter()
                build.pooled(build.pool, ta, tb, m, n, k, 1.0, ptr(x), ptr(y), 0.0, ptr(c))
                times[side].append(time.perf_counter() - t0)
        x, y, _c = group[1]
        want = np.zeros((m, n), np.float32)
        builds[0].serial(None, ta, tb, m, n, k, 1.0, ptr(x), ptr(y), 0.0, ptr(want))
        for build in builds:
            got = np.zeros((m, n), np.float32)
            build.pooled(build.pool, ta, tb, m, n, k, 1.0, ptr(x), ptr(y), 0.0, ptr(got))
            mismatch += not np.array_equal(got, want)
        med = {side: statistics.median(v) * 1e3 for side, v in times.items()}
        row = {"ta": ta, "tb": tb, "m": m, "n": n, "k": k, "calls": count,
               **{f"{side}.ms": round(v, 4) for side, v in med.items()},
               **{f"{side}.gflops": round(2 * m * n * k / v / 1e6, 1) for side, v in med.items()}}  # fmt: skip
        rows.append(row)
        print(row, flush=True)
        del group
        KEEP.clear()
    step = {side: round(sum(r[f"{side}.ms"] * r["calls"] for r in rows), 1) for side in sides}
    record = {"schema": "bcir.ndt_gemm_cold.v1", "threads": threads, "sets": sets, "reps": reps,
              "builds": paths, "transposed_a_only": ta_only, "mismatch": mismatch,
              "step_gemm_ms": step, "rows": rows}  # fmt: skip
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(step, "mismatch", mismatch)
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
