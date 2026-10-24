#!/usr/bin/env python3
"""OT-05's SwiGLU block taken apart at one thread: each of the block's three GEMM forms -- nn
(the forward's x W), nt (the input adjoint's dU W^T), tn (the weight adjoint's x^T dU) -- under
BCIR's three GEMM contracts (v1, the fused v2, the blocked v3) and torch.mm, and the block's two
elementwise passes (the SiLU product, the SwiGLU VJP) under v1's libm SiLU, the BCIR exponential's
(v2) and PyTorch's, at OT-01's nine shapes; each part's median over rotated rounds, every arm on one
thread (BCIR's serial path, PyTorch at torch.set_num_threads(1)).

It answers where OT-05's one-thread gap lies: a GEMM form slower than MKL's, or an elementwise
pass. PyTorch's VJP is the one its autograd computes for silu(u) * v, written out (sigmoid, its
derivative, two products); timings are this host's, indicative.

usage: TORCH_PY block_parts.py LIB.so OUT.json   (LIB: the plain tensor library,
       bcir.tests.native_decoder_fixtures.build_library)
"""

import ctypes as ct
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

#: OT-01's (and OT-05's) nine shapes, tools/perf/ot_run.py's OT01_SHAPES
SHAPES = tuple((m, k, 1024) for m in (16, 128, 512) for k in (256, 768)) + tuple(
    (m, 257, 1025) for m in (16, 128, 512)
)
GEMMS = {"v1": "bcir_tensor_mm_pool", "fused": "bcir_tensor_mm_fused_pool",
         "kblock": "bcir_tensor_mm_kblock_pool"}  # fmt: skip
SILU_MUL = {"v1": "bcir_tensor_silu_mul_pool", "v2": "bcir_tensor_silu_mul_v2_pool"}
VJP = {"v1": "bcir_tensor_swiglu_backward_pool", "v2": "bcir_tensor_swiglu_backward_v2_pool"}
ROUNDS, REPS = 9, 3


def bind(lib):
    fp, sz = ct.POINTER(ct.c_float), ct.c_size_t
    for name in GEMMS.values():
        f = getattr(lib, name)
        f.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, sz, sz, sz, ct.c_float, fp, fp, ct.c_float,
                      fp]  # fmt: skip
        f.restype = None
    for name in SILU_MUL.values():
        getattr(lib, name).argtypes = [ct.c_void_p, sz, fp, fp, fp]
        getattr(lib, name).restype = None
    for name in VJP.values():
        getattr(lib, name).argtypes = [ct.c_void_p, sz, fp, fp, fp, fp, fp]
        getattr(lib, name).restype = None
    return lib


def arms_for(lib, m, k, n, rng):
    fp = ct.POINTER(ct.c_float)
    x = rng.standard_normal((m, k), dtype=np.float32)
    w = rng.standard_normal((k, n), dtype=np.float32) * np.float32(0.05)
    du = rng.standard_normal((m, n), dtype=np.float32)
    c, dx, dw = (np.zeros(s, np.float32) for s in ((m, n), (m, k), (k, n)))
    u, v, y, dg, dv = (rng.standard_normal((m, n), dtype=np.float32) for _ in range(5))
    # each operand's address taken once, as OT-05's arms take theirs (ot_torch._block_native)
    px, pw, pdu, pc, pdx, pdw, pu, pv, py, pdg, pdv = (
        a.ctypes.data_as(fp) for a in (x, w, du, c, dx, dw, u, v, y, dg, dv)
    )
    X, W, DU, U, V = (torch.from_numpy(a) for a in (x, w, du, u, v))
    arms = {}
    for tag, name in GEMMS.items():
        f = getattr(lib, name)
        arms[f"nn/{tag}"] = lambda f=f: f(None, 0, 0, m, n, k, 1.0, px, pw, 0.0, pc)
        arms[f"nt/{tag}"] = lambda f=f: f(None, 0, 1, m, k, n, 1.0, pdu, pw, 0.0, pdx)
        arms[f"tn/{tag}"] = lambda f=f: f(None, 1, 0, k, n, m, 1.0, px, pdu, 0.0, pdw)
    arms["nn/torch"] = lambda: torch.mm(X, W)
    arms["nt/torch"] = lambda: torch.mm(DU, W.t())
    arms["tn/torch"] = lambda: torch.mm(X.t(), DU)
    e = m * n
    for tag, name in SILU_MUL.items():
        f = getattr(lib, name)
        arms[f"silu_mul/{tag}"] = lambda f=f: f(None, e, pu, pv, py)
    arms["silu_mul/torch"] = lambda: torch.nn.functional.silu(U) * V
    for tag, name in VJP.items():
        f = getattr(lib, name)
        arms[f"swiglu_vjp/{tag}"] = lambda f=f: f(None, e, pu, pv, pdu, pdg, pdv)

    def torch_vjp():
        s = torch.sigmoid(U)
        silu = U * s
        return DU * V * (s * (1 + U * (1 - s))), DU * silu

    arms["swiglu_vjp/torch"] = torch_vjp  # (a data_as pointer keeps its array alive)
    return arms


def rotated(arms):
    names = list(arms)
    for f in arms.values():
        f()
    ms = {a: [] for a in names}
    for r in range(ROUNDS):
        for j in range(len(names)):
            a = names[(r + j) % len(names)]
            t0 = time.perf_counter()
            for _ in range(REPS):
                arms[a]()
            ms[a].append(1000 * (time.perf_counter() - t0) / REPS)
    return ms


def main(argv):
    lib_path, out_path = Path(argv[1]), Path(argv[2])
    torch.set_num_threads(1)
    lib = bind(ct.CDLL(str(lib_path)))
    rng = np.random.default_rng(5)
    root = Path(__file__).resolve().parents[5]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()  # fmt: skip
    rows = []
    for m, k, n in SHAPES:
        ms = rotated(arms_for(lib, m, k, n, rng))
        med = {a: statistics.median(v) for a, v in ms.items()}
        flop = 2 * m * k * n
        rows.append({"shape": {"m": m, "k": k, "n": n}, "ms": med,
                     "gflops": {a: flop / (t / 1000) / 1e9 for a, t in med.items()
                                if a.split("/")[0] in ("nn", "nt", "tn")},
                     "raw_ms": {a: [round(t, 4) for t in v] for a, v in ms.items()}})  # fmt: skip
        print(f"{m}x{k}x{n} " + " ".join(f"{a}={t:.3f}" for a, t in med.items()), flush=True)
    out = {"what": "OT-05's SwiGLU block taken apart at one thread (block_parts.py)",
           "commit": commit, "python": platform.python_version(), "numpy": np.__version__,
           "torch": str(torch.__version__), "torch_threads": torch.get_num_threads(),
           "rounds": ROUNDS, "calls_per_round": REPS, "rows": rows}  # fmt: skip
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
