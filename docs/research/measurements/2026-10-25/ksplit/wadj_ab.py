"""KSPLIT: the blocked contract's weight adjoint -- C = A^T B, A stored k x m, B k x n, alpha 1, beta 0, the
step's dW = dY^T X -- on one build of the tensor unit, through bcir_tensor_mm_contract_variant_pool lent a room for
its partials (KSPLIT's split: a task a K block, then the fold) and lent none (the nest, as before), alternating in
one process. Every output of the split is held to the nest's byte for byte first; then ROUNDS rounds of REPS calls
of each arm, the arms' order alternating, and the geometric mean of the rounds' split/nest ratios with its 95%
interval, beside each arm's median. With --torch, torch.mm(A.T, B) at the same shape on the same thread count
(MKL), each round with the others: the reference rate, not a value the contract holds BCIR to.
usage: wadj_ab.py LIB THREADS SHAPES [--torch]    SHAPES: m,n,k;m,n,k;...    prints one JSON line per shape"""

import ctypes as ct
import json
import math
import statistics
import sys
import time

import numpy as np

LIB, TH = sys.argv[1], int(sys.argv[2])
SHAPES = [tuple(int(v) for v in s.split(",")) for s in sys.argv[3].split(";")]
TORCH = "--torch" in sys.argv[4:]
ROUNDS, REPS = 16, 4
T95 = {15: 2.131}  # Student's t, 15 degrees of freedom (16 rounds)
F = ct.POINTER(ct.c_float)

lib = ct.CDLL(LIB)
lib.bcir_tensor_mm_contract_variant_pool.restype = ct.c_int
lib.bcir_tensor_mm_contract_variant_pool.argtypes = [
    ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_float, F, F,
    ct.c_float, F, ct.c_void_p, ct.c_size_t,
]  # fmt: skip
lib.bcir_tensor_kblock_split.restype = ct.c_size_t
lib.bcir_tensor_kblock_split.argtypes = [ct.c_size_t, ct.c_void_p, ct.c_size_t]
lib.bcir_tensor_pool_size.restype = ct.c_size_t
lib.bcir_tensor_pool_size.argtypes = [ct.c_uint]
lib.bcir_tensor_pool_init.restype = ct.c_void_p
lib.bcir_tensor_pool_init.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_uint]
lib.bcir_tensor_pool_stop.argtypes = [ct.c_void_p]
pool, pool_mem = None, None
if TH > 1:
    size = lib.bcir_tensor_pool_size(TH)
    pool_mem = (ct.c_ubyte * (size + 64))()
    pool = lib.bcir_tensor_pool_init((ct.addressof(pool_mem) + 63) // 64 * 64, size, TH)
    if not pool:
        sys.exit("no pool")
if TORCH:
    import torch

    torch.set_num_threads(TH)


def ptr(x):
    return x.ctypes.data_as(F)


def call(arm, m, n, k, a, b, c, room, need):
    """One weight adjoint: lent the room (split) or none (nest)."""
    ws, nb = (room, need) if arm == "split" else (None, 0)
    if lib.bcir_tensor_mm_contract_variant_pool(pool, 2, 0, 1, 0, m, n, k, 1.0, ptr(a), ptr(b), 0.0,
                                                ptr(c[arm]), ws, nb) == 0:  # fmt: skip
        sys.exit("refused")


rng = np.random.default_rng(25)
for m, n, k in SHAPES:
    a = rng.standard_normal((k, m), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    blocks = lib.bcir_tensor_kblock_split(k, None, 0)
    need = (blocks * m * n * 4 + 63) // 64 * 64
    room_mem = (ct.c_ubyte * (need + 64))()
    room = (ct.addressof(room_mem) + 63) // 64 * 64
    c = {arm: np.zeros((m, n), dtype=np.float32) for arm in ("split", "nest")}
    args = (m, n, k, a, b, c, room, need)
    call("split", *args)
    call("nest", *args)
    if c["split"].tobytes() != c["nest"].tobytes():
        sys.exit(f"{m},{n},{k}: the split's bytes are not the nest's")
    if TORCH:
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        torch.mm(ta.T, tb)
    times = {"split": [], "nest": [], "torch": []}
    ratios = []
    for r in range(ROUNDS):
        order = ("split", "nest") if r % 2 == 0 else ("nest", "split")
        for arm in order:
            t0 = time.perf_counter()
            for _ in range(REPS):
                call(arm, *args)
            times[arm].append((time.perf_counter() - t0) / REPS)
        if TORCH:
            t0 = time.perf_counter()
            for _ in range(REPS):
                torch.mm(ta.T, tb)
            times["torch"].append((time.perf_counter() - t0) / REPS)
        ratios.append(times["split"][-1] / times["nest"][-1])
    logs = [math.log(x) for x in ratios]
    mean, half = (
        statistics.fmean(logs),
        T95[ROUNDS - 1] * statistics.stdev(logs) / math.sqrt(ROUNDS),
    )
    row = {
        "m": m, "n": n, "k": k, "k_blocks": blocks, "room_bytes": need, "threads": TH, "bytes_equal": True,
        "split_us": round(statistics.median(times["split"]) * 1e6, 1),
        "nest_us": round(statistics.median(times["nest"]) * 1e6, 1),
        "split_over_nest": [round(math.exp(mean), 3), round(math.exp(mean - half), 3), round(math.exp(mean + half), 3)],
        "split_gflops": round(2 * m * n * k / statistics.median(times["split"]) / 1e9, 1),
        "nest_gflops": round(2 * m * n * k / statistics.median(times["nest"]) / 1e9, 1),
    }  # fmt: skip
    if TORCH:
        row["torch_us"] = round(statistics.median(times["torch"]) * 1e6, 1)
        row["torch_gflops"] = round(2 * m * n * k / statistics.median(times["torch"]) / 1e9, 1)
    print(json.dumps(row), flush=True)
if pool:
    lib.bcir_tensor_pool_stop(pool)
