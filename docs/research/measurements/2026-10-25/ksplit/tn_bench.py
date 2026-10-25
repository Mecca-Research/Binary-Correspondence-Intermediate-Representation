"""NEXT-3: the k-blocked GEMM's weight-adjoint form (ta=1: C = A^T B, A stored k x m) and its forward form (nn) at the
GEO step's shapes against torch.mm (MKL), one thread and four: each arm the median of ROUNDS rounds of REPS calls,
rotated; BCIR on its own pool (threads 1: the serial path). Every BCIR output checked against a double reference
first (within float rounding: the forms' values are the contract's, not MKL's). usage: tn_bench.py LIB.so OUT.json"""

import ctypes as ct
import json
import statistics
import sys
import time

import numpy as np
import torch

LIB, OUT = sys.argv[1], sys.argv[2]
lib = ct.CDLL(LIB)
F = ct.POINTER(ct.c_float)
lib.bcir_tensor_mm_kblock_pool.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_float,
    F,
    F,
    ct.c_float,
    F,
]
lib.bcir_tensor_pool_size.restype = ct.c_size_t
lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
lib.bcir_tensor_pool_init.restype = ct.c_void_p
lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
lib.bcir_tensor_pool_stop.argtypes = [ct.c_void_p]
ROUNDS, REPS = 9, 5
pools = {}
for th in (1, 4):
    if th == 1:
        pools[th] = None
        continue
    size = lib.bcir_tensor_pool_size(th)
    mem = (ct.c_ubyte * (size + 64))()
    base = (ct.addressof(mem) + 63) // 64 * 64
    pools[th] = (lib.bcir_tensor_pool_init(base, size, th), mem)
rows = []
rng = np.random.default_rng(7)
for bt in (1024, 4096):
    for m, n, form in (
        (128, 384, "tn"),
        (128, 128, "tn"),
        (128, 256, "tn"),
        (256, 128, "tn"),
        (260, 128, "tn"),
        (bt, 384, "nn"),
        (bt, 256, "nn"),
    ):
        k = bt if form == "tn" else 128
        if form == "tn":
            a = rng.standard_normal((k, m), dtype=np.float32)  # stored k x m: C = a^T b
            b = rng.standard_normal((k, n), dtype=np.float32)
            ref = a.astype(np.float64).T @ b.astype(np.float64)
            ta = 1
        else:
            a = rng.standard_normal((m, k), dtype=np.float32)
            b = rng.standard_normal((k, n), dtype=np.float32)
            ref = a.astype(np.float64) @ b.astype(np.float64)
            ta = 0
        c = np.zeros((m, n), dtype=np.float32)
        fp = lambda x: x.ctypes.data_as(F)  # noqa: E731
        ta_t, b_t = torch.from_numpy(a), torch.from_numpy(b)
        at = ta_t.t() if form == "tn" else ta_t
        for th in (1, 4):
            pool = pools[th][0] if pools[th] else None
            lib.bcir_tensor_mm_kblock_pool(pool, ta, 0, m, n, k, 1.0, fp(a), fp(b), 0.0, fp(c))
            err = float(np.max(np.abs(c - ref)) / (np.max(np.abs(ref)) + 1e-30))
            assert err < 1e-5, (m, n, k, form, err)
            torch.set_num_threads(th)
            times = {"bcir": [], "mkl": []}
            for r in range(ROUNDS):
                order = ("bcir", "mkl") if r % 2 == 0 else ("mkl", "bcir")
                for arm in order:
                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        if arm == "bcir":
                            lib.bcir_tensor_mm_kblock_pool(
                                pool, ta, 0, m, n, k, 1.0, fp(a), fp(b), 0.0, fp(c)
                            )
                        else:
                            torch.mm(at, b_t)
                    times[arm].append((time.perf_counter() - t0) / REPS)
            flops = 2.0 * m * n * k
            med = {arm: statistics.median(v) for arm, v in times.items()}
            row = {
                "form": form,
                "m": m,
                "n": n,
                "k": k,
                "threads": th,
                "bcir_gflops": flops / med["bcir"] / 1e9,
                "mkl_gflops": flops / med["mkl"] / 1e9,
                "bcir_over_mkl_time": med["bcir"] / med["mkl"],
            }
            rows.append(row)
            print(
                f"{form} {m}x{n}x{k} {th}t: BCIR {row['bcir_gflops']:6.1f} GFLOP/s, MKL {row['mkl_gflops']:6.1f}, "
                f"BCIR/MKL time x{row['bcir_over_mkl_time']:.3f}",
                flush=True,
            )
for th, p in pools.items():
    if p:
        lib.bcir_tensor_pool_stop(p[0])
json.dump({"rows": rows}, open(OUT, "w"), indent=1)
