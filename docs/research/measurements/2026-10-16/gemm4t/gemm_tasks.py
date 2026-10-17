"""NDT-GEMM-4T: the 9.84M decoder step's GEMM shapes on the timed build (tasks_instrument.py) at
one and at THREADS threads, operands on 2 MB pages and warm (one set reused): per shape the wall
time, the busy fraction (every thread's summed task time over threads x wall) and the task
inflation (the median over tasks of a task's time at THREADS threads over the same task's at one),
and the step's GEMM time at each count (every call of the step at its shape's median), as JSON.

    python3 gemm_tasks.py LIB.so THREADS REPS OUT.json
"""

import ctypes as ct
import json
import mmap
import statistics
import sys
import time
from collections import Counter, defaultdict
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
    lib_path, threads, reps, out = argv[1], int(argv[2]), int(argv[3]), Path(argv[4])
    lib = ct.CDLL(lib_path)
    gemm = lib.bcir_tensor_mm_kblock_pool
    gemm.argtypes, gemm.restype = SIG, None
    lib.bcir_tensor_pool_size.argtypes, lib.bcir_tensor_pool_size.restype = [ct.c_uint], ct.c_size_t
    lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
    lib.bcir_tensor_pool_init.restype = ct.c_void_p
    lib.bcir_prof_count.restype = ct.c_size_t
    lib.bcir_prof_get.argtypes = [ct.c_size_t, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_double)]
    pools = {}
    for t in sorted({1, threads}):
        size = lib.bcir_tensor_pool_size(t)
        mem = (ct.c_char * size)()
        pools[t] = (mem, lib.bcir_tensor_pool_init(mem, size, t))
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4)
    program = training_step_program(spec, 8, 128)
    calls = Counter(g for step in program.steps for g in step.gemms)
    rng = np.random.default_rng(1)
    rows = []
    for (ta, tb, m, n, k), count in calls.items():
        x = placed((rng.standard_normal((k, m) if ta else (m, k)) * 0.5).astype(np.float32))
        y = placed((rng.standard_normal((n, k) if tb else (k, n)) * 0.5).astype(np.float32))
        c = placed(np.zeros((m, n), np.float32))
        res = {}
        for t in sorted({1, threads}):
            walls, busy, per_task = [], [], defaultdict(list)
            for r in range(reps + 1):
                lib.bcir_prof_reset()
                t0 = time.perf_counter()
                gemm(pools[t][1], ta, tb, m, n, k, 1.0, ptr(x), ptr(y), 0.0, ptr(c))
                wall = time.perf_counter() - t0
                if r == 0:
                    continue  # the warm-up call
                walls.append(wall)
                v, tt, by = (ct.c_uint * 4)(), (ct.c_double * 2)(), defaultdict(float)
                for q in range(lib.bcir_prof_count()):
                    lib.bcir_prof_get(q, v, tt)
                    per_task[v[1]].append(tt[1] - tt[0])
                    by[v[2]] += tt[1] - tt[0]
                busy.append(sum(by.values()) / 1e9)
            res[t] = {"wall_ms": statistics.median(walls) * 1e3,
                      "busy_ms": statistics.median(busy) * 1e3,
                      "task_us": {q: statistics.median(v) / 1e3 for q, v in per_task.items()}}  # fmt: skip
        one, many = res[1], res[threads]
        inflation = statistics.median(
            many["task_us"][q] / one["task_us"][q] for q in one["task_us"]
        )
        row = {"ta": ta, "tb": tb, "m": m, "n": n, "k": k, "calls": count,
               "tasks": len(many["task_us"]), "t1_wall_ms": round(one["wall_ms"], 3),
               f"t{threads}_wall_ms": round(many["wall_ms"], 3),
               "speedup": round(one["wall_ms"] / many["wall_ms"], 2),
               "busy_fraction": round(many["busy_ms"] / threads / many["wall_ms"], 3),
               "task_inflation": round(inflation, 3),
               "gflops_t1": round(2 * m * n * k / one["wall_ms"] / 1e6, 1),
               f"gflops_t{threads}": round(2 * m * n * k / many["wall_ms"] / 1e6, 1)}  # fmt: skip
        rows.append(row)
        print(row, flush=True)
        KEEP.clear()
    step = {"t1": sum(r["t1_wall_ms"] * r["calls"] for r in rows),
            f"t{threads}": sum(r[f"t{threads}_wall_ms"] * r["calls"] for r in rows)}  # fmt: skip
    record = {"schema": "bcir.ndt_gemm_tasks.v1", "threads": threads, "reps": reps,
              "step_gemm_ms": {key: round(v, 1) for key, v in step.items()}, "rows": rows}  # fmt: skip
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(record["step_gemm_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
