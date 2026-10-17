"""The step's 15 GEMM shapes at four threads on two builds of the trainer's units (A, B), alternated
call by call: the median and best GFLOP/s of R calls each, every pooled result checked bit for bit
against A's serial kernel, written as JSON.

    python3 gemm_rates.py A.so B.so OUT.json [threads] [reps]

A and B are shared builds of runtime/c/bcir_tensor.c and bcir_decoder_train.c (gcc -std=c11 -O3
-ffp-contract=off -fno-math-errno -pthread -DBCIR_DT_BUILD_SHARED -shared -fPIC); the shapes are the
9.84M decoder's step program's (bcir.hosted.models.native_program), read from this checkout."""

import ctypes as ct
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from bcir.frontends.models.decode import DecoderSpec  # noqa: E402
from bcir.hosted.models.native_program import training_step_program  # noqa: E402

FP = ct.POINTER(ct.c_float)
SIG = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_float]
SIG += [FP, FP, ct.c_float, FP]


class Lib:
    def __init__(self, path, threads):
        self.lib = ct.CDLL(path)
        self.serial = self.lib.bcir_tensor_mm_kblock
        self.serial.argtypes, self.serial.restype = SIG, None
        self.pooled = self.lib.bcir_tensor_mm_kblock_pool
        self.pooled.argtypes, self.pooled.restype = SIG, None
        self.lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
        self.lib.bcir_tensor_pool_size.restype = ct.c_size_t
        self.lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
        self.lib.bcir_tensor_pool_init.restype = ct.c_void_p
        size = self.lib.bcir_tensor_pool_size(threads)
        self.mem = (ct.c_char * size)()
        self.pool = self.lib.bcir_tensor_pool_init(self.mem, size, threads)
        assert self.pool


def ptr(x):
    return x.ctypes.data_as(FP)


def run(fn, pool, ta, tb, x, y, beta, c0):
    m, n = c0.shape
    k = x.shape[0] if ta else x.shape[1]
    c = c0.copy()
    t0 = time.perf_counter()
    fn(pool, ta, tb, m, n, k, 1.0, ptr(x), ptr(y), beta, ptr(c))
    return c, time.perf_counter() - t0


def main(argv):
    a_path, b_path, out = argv[1], argv[2], argv[3]
    threads = int(argv[4]) if len(argv) > 4 else 4
    reps = int(argv[5]) if len(argv) > 5 else 15
    libs = {"A": Lib(a_path, threads), "B": Lib(b_path, threads)}
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4)
    shapes = list(dict.fromkeys(training_step_program(spec, 8, 128).gemm_shapes()))
    rng = np.random.default_rng(2026)
    rows, mism = [], {"A": 0, "B": 0}
    for ta, tb, m, n, k in shapes:
        x = (rng.standard_normal((k, m) if ta else (m, k)) * 0.5).astype(np.float32)
        y = (rng.standard_normal((n, k) if tb else (k, n)) * 0.5).astype(np.float32)
        c0 = rng.standard_normal((m, n)).astype(np.float32)
        want, _ = run(libs["A"].serial, None, ta, tb, x, y, 1.0, c0)
        for name, lib in libs.items():
            got, _ = run(lib.pooled, lib.pool, ta, tb, x, y, 1.0, c0)
            mism[name] += not np.array_equal(got, want)
        ts = {"A": [], "B": []}
        for _ in range(reps):
            for name, lib in libs.items():
                ts[name].append(run(lib.pooled, lib.pool, ta, tb, x, y, 0.0, c0)[1])
        flop = 2.0 * m * n * k
        r = {"ta": ta, "tb": tb, "m": m, "n": n, "k": k, "flop": flop}
        for name in ("A", "B"):
            r[name + ".median"] = round(flop / statistics.median(ts[name]) / 1e9, 1)
            r[name + ".best"] = round(flop / min(ts[name]) / 1e9, 1)
        rows.append(r)
        print(ta, tb, m, n, k, {kk: v for kk, v in r.items() if "." in kk}, flush=True)
    work = sum(r["flop"] for r in rows)
    keys = ("A.median", "B.median", "A.best", "B.best")
    weighted = {kk: round(work / sum(r["flop"] / r[kk] for r in rows), 1) for kk in keys}
    record = {"threads": threads, "reps": reps, "a": a_path, "b": b_path, "mismatches": mism}
    record.update({"rows": rows, "work_weighted": weighted})
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    print("mismatches", mism)
    print("work-weighted", weighted)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
