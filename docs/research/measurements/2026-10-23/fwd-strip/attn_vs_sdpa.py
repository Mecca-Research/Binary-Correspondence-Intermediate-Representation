"""The attention alone, as each side trains it: BCIR's binary32 attention (the tensor unit's workspace entries,
the forward storing P, the backward reading it) against PyTorch's scaled_dot_product_attention, causal, on
the same inputs -- its default CPU kernel (flash: no stored probabilities, the backward recomputing them) and
its math kernel (the probabilities stored) -- arms in one process in rotated rounds after a warm one, BCIR's
pool made and stopped outside the clock, an A/A arm (BCIR again). Each ratio is an arm's time over BCIR's in
the same round; the geometric mean and its 95% interval over the rounds (Student t).
usage: python -I attn_vs_sdpa.py LIBTENSOR.so OUT.json [--b 4 --t 2048 --h 4 --d 32 --threads 4 --rounds 24
       --contract 1]"""

import argparse
import ctypes as C
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")
ap = argparse.ArgumentParser()
ap.add_argument("lib")
ap.add_argument("out")
for k, v in (
    ("b", 4),
    ("t", 2048),
    ("h", 4),
    ("d", 32),
    ("threads", 4),
    ("rounds", 24),
    ("contract", 1),
):
    ap.add_argument("--" + k, type=int, default=v)
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

torch.set_num_threads(a.threads)
L = C.CDLL(a.lib)
sz, vp, ui, ci = C.c_size_t, C.c_void_p, C.c_uint, C.c_int
L.bcir_tensor_pool_size.restype, L.bcir_tensor_pool_size.argtypes = sz, [ui]
L.bcir_tensor_pool_init.restype, L.bcir_tensor_pool_init.argtypes = vp, [vp, sz, ui]
L.bcir_tensor_pool_stop.restype, L.bcir_tensor_pool_stop.argtypes = None, [vp]
L.bcir_tensor_attention_pool_workspace.restype = sz
L.bcir_tensor_attention_pool_workspace.argtypes = [sz] * 5 + [ui]
fwd = L.bcir_tensor_attention_contract_variant_scan_ws_pool
fwd.restype, fwd.argtypes = ci, [vp, ci, ui] + [sz] * 5 + [vp] * 5 + [C.POINTER(ci), vp, sz]
bwd = L.bcir_tensor_attention_backward_contract_variant_ws_pool
bwd.restype, bwd.argtypes = ci, [vp, ci, ui] + [sz] * 5 + [vp] * 8 + [vp, sz]


def aligned(nbytes):
    raw = np.empty(nbytes + 64, dtype=np.uint8)
    off = (-raw.ctypes.data) % 64
    return raw, raw.ctypes.data + off


b, t, h, d = a.b, a.t, a.h, a.d
g = torch.Generator().manual_seed(7)
q0, k0, v0, dy0 = (
    torch.randn(b, t, h, d, generator=g) * 0.5 for _ in range(4)
)  # [B, T, H, D], contiguous
p_keep, p_ptr = aligned(b * h * t * t * 4)
y = torch.empty(b, t, h, d)
dq, dk, dv = torch.empty(b, t, h, d), torch.empty(b, t, h, d), torch.empty(b, t, h, d)
need = L.bcir_tensor_attention_pool_workspace(b, t, h, h, d, a.threads)
ws_keep, ws_ptr = aligned(max(need, 64))
bad = ci(0)


def bcir():
    pool, keep = None, None
    if a.threads > 1:
        n = L.bcir_tensor_pool_size(a.threads)
        keep, mem = aligned(n)
        pool = L.bcir_tensor_pool_init(mem, n, a.threads)
    t0 = time.perf_counter()
    r1 = fwd(
        pool,
        0,
        a.contract,
        b,
        t,
        h,
        h,
        d,
        q0.data_ptr(),
        k0.data_ptr(),
        v0.data_ptr(),
        p_ptr,
        y.data_ptr(),
        C.byref(bad),
        ws_ptr,
        need,
    )
    t1 = time.perf_counter()
    r2 = bwd(
        pool,
        0,
        a.contract,
        b,
        t,
        h,
        h,
        d,
        q0.data_ptr(),
        k0.data_ptr(),
        v0.data_ptr(),
        p_ptr,
        dy0.data_ptr(),
        dq.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        ws_ptr,
        need,
    )
    t2 = time.perf_counter()
    if pool:
        L.bcir_tensor_pool_stop(pool)
    del keep
    if not r1 or not r2:
        raise SystemExit(f"BCIR's entries refused: {r1} {r2}")
    return t1 - t0, t2 - t1


def torch_arm(backend):
    def run():
        ql, kl, vl = (x.clone().requires_grad_() for x in (q0, k0, v0))
        with sdpa_kernel(backend):
            t0 = time.perf_counter()
            o = F.scaled_dot_product_attention(
                ql.transpose(1, 2), kl.transpose(1, 2), vl.transpose(1, 2), is_causal=True
            )
            t1 = time.perf_counter()
            o.backward(dy0.transpose(1, 2))
            t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    return run


ARMS = {
    "bcir": bcir,
    "flash": torch_arm(SDPBackend.FLASH_ATTENTION),
    "math": torch_arm(SDPBackend.MATH),
    "bcir-AA": bcir,
}
names = list(ARMS)
ms = {n: {"forward": [], "backward": [], "total": []} for n in names}
for r in range(-1, a.rounds):
    order = names if r < 0 else names[r % len(names) :] + names[: r % len(names)]
    for n in order:
        f, bk = ARMS[n]()
        if r >= 0:
            ms[n]["forward"].append(1000 * f)
            ms[n]["backward"].append(1000 * bk)
            ms[n]["total"].append(1000 * (f + bk))
T95 = (
    0,
    0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
)


def interval(ratios):
    logs = [math.log(x) for x in ratios]
    n = len(logs)
    tq = T95[n] if n <= 24 else 1.96 + 2.4 / (n - 1)
    m, hw = statistics.fmean(logs), tq * statistics.stdev(logs) / math.sqrt(n)
    return {"geometric_mean": math.exp(m), "ci95": [math.exp(m - hw), math.exp(m + hw)]}


out = {
    "shape": [b, t, h, d],
    "threads": a.threads,
    "rounds": a.rounds,
    "contract": a.contract,
    "workspace_bytes": need,
    "torch": torch.__version__,
    "arms": {},
}
for n in names:
    out["arms"][n] = {
        ph: {"median_ms": statistics.median(v), "raw_ms": v} for ph, v in ms[n].items()
    }
    if n != "bcir":
        for ph in ("forward", "backward", "total"):
            out["arms"][n][ph]["over_bcir"] = interval(
                [x / y for x, y in zip(ms[n][ph], ms["bcir"][ph])]
            )
with open(a.out, "w") as fh:
    json.dump(out, fh, indent=1)
for n in names:
    row = out["arms"][n]
    extra = (
        ""
        if n == "bcir"
        else "  over BCIR: "
        + ", ".join(
            f"{ph} {row[ph]['over_bcir']['geometric_mean']:.3f} ({row[ph]['over_bcir']['ci95'][0]:.3f}-"
            f"{row[ph]['over_bcir']['ci95'][1]:.3f})"
            for ph in ("forward", "backward", "total")
        )
    )
    print(
        f"{n:8s} fwd {row['forward']['median_ms']:8.2f} bwd {row['backward']['median_ms']:8.2f} total "
        f"{row['total']['median_ms']:8.2f} ms{extra}"
    )
