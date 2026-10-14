"""NDT-PAGES' mechanism record: the 9.84M decoder step's GEMM shapes on operands placed on 4 KB
pages and on 2 MB transparent huge pages, alternated call by call at one and at four threads --
the median and best of R calls each, every result checked bit for bit against the serial kernel
on the other placement -- and the step's GEMM time on each placement (every call of the step at
its shape's median), written as JSON.

    python3 gemm_pages.py LIB.so OUT.json [reps]

LIB is the trainer's shared library (`NativeDecoder.build`, CC=gcc for the records). The 4 KB
operands are numpy arrays with numpy's own huge-page advice switched off; the 2 MB ones sit on a
private anonymous mapping advised MADV_HUGEPAGE before it is touched, as the trainer's state does
(bcir.hosted.models.native._state_buffers). Linux only."""

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


def placed(values, huge):
    """A copy of `values` on 4 KB pages, or on a mapping advised for 2 MB pages."""
    if not huge:
        return values.copy()
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


def main(argv):
    lib_path, out = argv[1], argv[2]
    reps = int(argv[3]) if len(argv) > 3 else 9
    np._core.multiarray._set_madvise_hugepage(False)  # the 4 KB arm stays on 4 KB pages
    lib = ct.CDLL(lib_path)
    serial, pooled = lib.bcir_tensor_mm_kblock, lib.bcir_tensor_mm_kblock_pool
    serial.argtypes = pooled.argtypes = SIG
    serial.restype = pooled.restype = None
    lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
    lib.bcir_tensor_pool_size.restype = ct.c_size_t
    lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
    lib.bcir_tensor_pool_init.restype = ct.c_void_p
    pools = {}
    for t in (1, 4):
        size = lib.bcir_tensor_pool_size(t)
        mem = (ct.c_char * size)()
        pools[t] = (mem, lib.bcir_tensor_pool_init(mem, size, t))
        assert pools[t][1]
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4)
    program = training_step_program(spec, 8, 128)
    calls = Counter(g for step in program.steps for g in step.gemms)  # every call of the step
    rng = np.random.default_rng(2026)
    rows, mismatch = [], 0
    step = {f"t{t}.{p}": 0.0 for t in (1, 4) for p in ("4k", "2m")}
    for ta, tb, m, n, k in calls:
        x = (rng.standard_normal((k, m) if ta else (m, k)) * 0.5).astype(np.float32)
        y = (rng.standard_normal((n, k) if tb else (k, n)) * 0.5).astype(np.float32)
        arms = {p: {"x": placed(x, p == "2m"), "y": placed(y, p == "2m"),
                    "c": placed(np.zeros((m, n), np.float32), p == "2m")} for p in ("4k", "2m")}  # fmt: skip
        want = np.zeros((m, n), np.float32)
        serial(None, ta, tb, m, n, k, 1.0, ptr(x), ptr(y), 0.0, ptr(want))
        row = {"ta": ta, "tb": tb, "m": m, "n": n, "k": k, "calls": calls[(ta, tb, m, n, k)]}
        for t in (1, 4):
            ts = {"4k": [], "2m": []}
            for r in range(reps):
                for p in ("4k", "2m") if r % 2 == 0 else ("2m", "4k"):
                    a = arms[p]
                    t0 = time.perf_counter()
                    pooled(pools[t][1], ta, tb, m, n, k, 1.0, ptr(a["x"]), ptr(a["y"]), 0.0,
                           ptr(a["c"]))  # fmt: skip
                    ts[p].append(time.perf_counter() - t0)
                    mismatch += not np.array_equal(a["c"], want)
            for p, v in ts.items():
                med = statistics.median(v)
                row[f"t{t}.{p}.gflops.median"] = round(2.0 * m * n * k / med / 1e9, 1)
                row[f"t{t}.{p}.gflops.best"] = round(2.0 * m * n * k / min(v) / 1e9, 1)
                step[f"t{t}.{p}"] += 1000 * med * row["calls"]
        rows.append(row)
        print(row, flush=True)
    huge = [line.strip() for line in open("/proc/meminfo") if line.startswith("AnonHugePages")]
    record = {
        "schema": "bcir.ndt_pages_gemm.v1",
        "reps": reps,
        "mismatch": mismatch,
        "step_gemm_ms": {key: round(v, 1) for key, v in step.items()},
        "anon_huge_pages_after": huge,
        "rows": rows,
    }
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    print({key: record[key] for key in ("mismatch", "step_gemm_ms")})
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
