#!/usr/bin/env python3
"""Every GEMM call of the real step timed in place, with its pool's idle (NDT-GEMM-4T's instrument).

A copy of the two C units (`--src`, any tree's runtime/c) with a monotonic clock around each of
the step's 43 forward linears and 43 backward pairs (`linear`, `linear_backward`), and every GEMM
task timed by `tasks_instrument.py`'s timers; built with the trainer's flags and run on the
comparison's 9.84M decoder, weights and first batch. Per shape: the wall time per step, the rate,
and the pool's idle inside the call -- threads x the call's wall less the tasks' time.

    python3 gemm_calls.py --src runtime/c --data DATA --threads 4 --steps 8 --out OUT.json
"""

from __future__ import annotations

import argparse
import bisect
import ctypes as ct
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
FLAGS = ["-std=c11", "-O3", "-ffp-contract=off", "-fno-math-errno", "-Wall", "-Wextra",
         "-DBCIR_DT_BUILD_SHARED", "-shared", "-fPIC", "-pthread"]  # fmt: skip

PROF = r"""
#include <time.h>
static double gc_ns[512], gc_t0[512]; static unsigned gc_meta[512][4]; static unsigned gc_n;
static double gc_now(void) {
  struct timespec q; clock_gettime(CLOCK_MONOTONIC,&q); return (double)q.tv_sec*1e9+(double)q.tv_nsec;
}
static void gc_add(unsigned kind,size_t m,size_t n,size_t k,double t0) {
  if (gc_n<512) { gc_t0[gc_n]=t0; gc_ns[gc_n]=gc_now()-t0; gc_meta[gc_n][0]=kind;
    gc_meta[gc_n][1]=(unsigned)m; gc_meta[gc_n][2]=(unsigned)n; gc_meta[gc_n][3]=(unsigned)k; gc_n++; }
}
BCIR_DT_API void bcir_gc_reset(void) { gc_n=0; }
BCIR_DT_API unsigned bcir_gc_count(void) { return gc_n; }
BCIR_DT_API void bcir_gc_get(unsigned i,unsigned *meta,double *t) {
  unsigned j; for (j=0;j<4;j++) meta[j]=gc_meta[i][j];
  t[0]=gc_t0[i]; t[1]=gc_ns[i];
}
"""
EDITS = (
    ("#include <float.h>\n", "#include <float.h>\n" + PROF),
    (
        "    const float *w,float *y) { gemm(s,0,1,r,out,in,1.0f,x,w,0.0f,y); }",
        "    const float *w,float *y) { double t0=gc_now(); gemm(s,0,1,r,out,in,1.0f,x,w,0.0f,y);"
        " gc_add(0,r,out,in,t0); }",
    ),
    (
        "  bcir_tensor_mm_pair_pool(s->pool,gemm_contract(s),&cx,&cw);\n",
        "  { double t0=gc_now(); bcir_tensor_mm_pair_pool(s->pool,gemm_contract(s),&cx,&cw);"
        " gc_add(1,r,in,out,t0); }\n",
    ),
)
#: (kind, rows, n, k) -> the step's name for the call and its count per step
NAMES = {
    (0, 1024, 384, 384): ("forward q, o (x W^T, 384 x 384)", 12),
    (0, 1024, 192, 384): ("forward k, v (x W^T, 192 x 384)", 12),
    (0, 1024, 1024, 384): ("forward gate, up (x W^T, 1024 x 384)", 12),
    (0, 1024, 384, 1024): ("forward down (x W^T, 384 x 1024)", 6),
    (0, 1024, 260, 384): ("forward head (x E^T, 260 x 384)", 1),
    (1, 1024, 384, 384): ("backward q, o (dY W and dY^T X)", 12),
    (1, 1024, 384, 192): ("backward k, v (dY W and dY^T X)", 12),
    (1, 1024, 384, 1024): ("backward gate, up (dY W and dY^T X)", 12),
    (1, 1024, 1024, 384): ("backward down (dY W and dY^T X)", 6),
    (1, 1024, 384, 260): ("backward head (dY E and dY^T X)", 1),
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(src: Path, work: Path, cc: str) -> Path:
    tins = load("tasks_instrument", HERE / "tasks_instrument.py")
    for name in ("bcir_tensor.h", "bcir_decoder_train.h"):
        (work / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    tensor = (src / "bcir_tensor.c").read_text(encoding="utf-8")
    for old, new in tins.EDITS[:3]:  # the task timers, not the one-thread routing
        if tensor.count(old) != 1:
            raise SystemExit("gemm_calls: the tensor unit moved; update tasks_instrument.py")
        tensor = tensor.replace(old, new)
    (work / "bcir_tensor.c").write_text("#define _GNU_SOURCE 1\n" + tensor, encoding="utf-8")
    train = (src / "bcir_decoder_train.c").read_text(encoding="utf-8")
    for old, new in EDITS:
        if train.count(old) != 1:
            raise SystemExit(f"gemm_calls: {old[:50]!r} not found once; update the instrument")
        train = train.replace(old, new)
    (work / "bcir_decoder_train.c").write_text(
        "#define _POSIX_C_SOURCE 199309L\n" + train, encoding="utf-8"
    )
    lib = work / "bcir_decoder_train.so"
    subprocess.run(
        [
            cc,
            *FLAGS,
            "-I",
            str(work),
            str(work / "bcir_tensor.c"),
            str(work / "bcir_decoder_train.c"),
            "-o",
            str(lib),
            "-lm",
        ],
        check=True,
    )
    return lib


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True, help="a tree's runtime/c")
    p.add_argument("--data", type=Path, required=True, help="the comparison's data directory")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cc", default="gcc")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    tc = load(
        "train_comparison", ROOT / "docs/research/measurements/2026-10-11/train_comparison.py"
    )
    from bcir.hosted.models.native import NativeAdamW

    with tempfile.TemporaryDirectory() as tmp:
        lib = build(a.src, Path(tmp), a.cc)
        nd, _ = tc._native(a.data, lib, threads=a.threads, contract="kblock+exp+reduce")
        L = nd.lib
        L.bcir_gc_count.restype = ct.c_uint
        L.bcir_gc_get.argtypes = [ct.c_uint, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_double)]
        L.bcir_prof_count.restype = ct.c_size_t
        L.bcir_prof_get.argtypes = [ct.c_size_t, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_double)]
        o = tc.OPT
        opt = NativeAdamW(
            lr=o["peak"] * 0.25,
            beta1=o["beta1"],
            beta2=o["beta2"],
            epsilon=o["eps"],
            weight_decay=o["decay"],
            grad_clip=o["clip"],
        )
        stream, offsets = (
            tc._load(a.data, "lm-train.u16", "H"),
            tc._load(a.data, "lm-schedule.u32", "I"),
        )
        xs, ys = tc._batch(stream, offsets, 0, "lm")
        for _ in range(2):
            nd.run(xs, ys, steps=1, optimizer=opt)
        wall, idle = defaultdict(list), defaultdict(list)
        for _ in range(a.steps):
            L.bcir_gc_reset()
            L.bcir_prof_reset()
            nd.run(xs, ys, steps=1, optimizer=opt)
            meta, t = (ct.c_uint * 4)(), (ct.c_double * 2)()
            calls = []
            for q in range(L.bcir_gc_count()):
                L.bcir_gc_get(q, meta, t)
                calls.append((t[0], t[0] + t[1], tuple(meta)))
            calls.sort()
            starts = [c[0] for c in calls]
            busy = defaultdict(float)
            v, tt = (ct.c_uint * 4)(), (ct.c_double * 2)()
            for q in range(L.bcir_prof_count()):
                L.bcir_prof_get(q, v, tt)
                k = bisect.bisect_right(starts, tt[0]) - 1
                if k >= 0 and tt[1] <= calls[k][1]:
                    busy[k] += tt[1] - tt[0]
            wsum, isum = defaultdict(float), defaultdict(float)
            for k, (c0, c1, key) in enumerate(calls):
                wsum[key] += c1 - c0
                isum[key] += a.threads * (c1 - c0) - busy[k]
            for key in wsum:
                wall[key].append(wsum[key] / 1e6)
                idle[key].append(isum[key] / 1e6 / a.threads)
    rows = []
    for key in sorted(wall, key=lambda k: -statistics.median(wall[k])):
        kind, m, n, k = key
        name, count = NAMES.get(key, (str(key), 1))
        ms = statistics.median(wall[key])
        flop = 2.0 * m * n * k * (2 if kind else 1) * count
        rows.append(
            {
                "call": name,
                "count": count,
                "m_n_k": [m, n, k],
                "ms_per_step": round(ms, 3),
                "gflops": round(flop / (ms * 1e-3) / 1e9, 1),
                "idle_ms_per_step": round(statistics.median(idle[key]), 3),
            }
        )
    rec = {
        "schema": "bcir.ndt_gemm_calls.v1",
        "threads": a.threads,
        "steps": a.steps,
        "compiler": a.cc,
        "contract": "kblock+exp+reduce",
        "rows": rows,
        "gemm_ms_per_step": round(sum(r["ms_per_step"] for r in rows), 2),
    }
    a.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(
            f"{r['call']:40s} {r['ms_per_step']:8.2f} ms  {r['gflops']:7.1f} GFLOP/s  idle {r['idle_ms_per_step']:6.2f} ms"
        )
    print("gemm per step", rec["gemm_ms_per_step"], "ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
