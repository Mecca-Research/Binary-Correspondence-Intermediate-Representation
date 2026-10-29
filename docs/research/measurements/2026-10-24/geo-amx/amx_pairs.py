"""GEO-AMX: the PR818 study's models trained under binary32 and under the bf16 GEMM contract on this AMX host.

The bundle's own paired timing (benchmark_update.py, 'Numerics/BF16/{query_full,attention}/B4T256') ran on an AMD
EPYC 9V74 without AMX, so its bf16 GEMMs took the AVX-512F instance, which emulates the tile unit's arithmetic
(NDT-BF16: every instance computes the same bits), and read x0.877 and x0.960 of binary32's rate. Here the same
pairs run where the AMX instance runs, against main's tensor unit of this branch (build_amx.py), with an A/A arm:
  FP32      contract 14 (exp + kblock + reduce), the study's primary
  FP32-AA   the same again, its own instance (the host's noise)
  BF16      contract 26 (exp + reduce + bf16: the GEMMs on bfloat16 operands), the study's bf16 arm
  BF16-ATTN contract 90 (26 + attn-bf16: the attention's products on bfloat16 too; the attention model only)
One observation is `steps` updates (bcir_decoder_run_accum) from the same initial state, restored outside the
clock; the arms' order rotates every round. The first update's loss under each arm is recorded (bf16 is another
arithmetic: its losses differ from binary32's by design), and the bf16 instances the host admits.
usage: python3 amx_pairs.py WORK OUT.json [--rounds 24] [--steps 3] [--cases attention:256,...]
"""

import argparse
import ctypes as ct
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
p = argparse.ArgumentParser()
p.add_argument("work", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--rounds", type=int, default=24)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=4)
p.add_argument(
    "--cases",
    default="attention:256,attention:512,attention:1024,attention:2048,global:256,global:512,"
    "query_full:256,query_full:512",
)
a = p.parse_args()
sys.path.insert(0, str(a.work.resolve()))
import numpy as np  # noqa: E402
import experiment as ex  # noqa: E402

LIB = {"global": "guard_compact", "query_full": "query_guard_compact", "attention": "attention"}
ARMS = (("FP32", 14), ("FP32-AA", 14), ("BF16", 26), ("BF16-ATTN", 90))
T95 = {7: 2.365, 11: 2.201, 15: 2.131, 23: 2.069, 31: 2.040, 47: 2.012}


def recipe(model):
    if model == "attention":
        return ex.Recipe(kind="transformer", ff=256, rope=True)
    return ex.Recipe(
        kind="global", ff=257, query_aware=model.startswith("query"), precision_guard=True
    )


def interval(ratios):
    logs = [math.log(x) for x in ratios]
    n = len(logs)
    t = T95.get(n - 1, 1.96 + 2.4 / (n - 1))
    m, h = statistics.fmean(logs), t * statistics.stdev(logs) / math.sqrt(n)
    return {"geometric_mean": math.exp(m), "ci95": [math.exp(m - h), math.exp(m + h)], "rounds": n}


def case(model, t):
    r = replace(recipe(model), context=t, batch=4)
    rng = np.random.default_rng(81820261018 + t + r.ff)
    xs = rng.integers(0, r.vocab, (a.steps, r.batch, r.context), dtype=np.uint32)
    ys = np.roll(xs, -1, axis=2).copy()
    rates = np.full(a.steps, 1e-4, np.float64)
    use = [(name, c) for name, c in ARMS if model == "attention" or name != "BF16-ATTN"]
    arms = [
        (
            name,
            ex.Native(replace(r, contract=c), seed=881, threads=a.threads, library_name=LIB[model]),
        )
        for name, c in use
    ]
    try:
        lib = arms[0][1].lib
        lib.bcir_tensor_mm_bf16_variants.restype = ct.c_int
        mask = int(lib.bcir_tensor_mm_bf16_variants())
        init = {name: n.snapshot() for name, n in arms}
        first = {}
        for name, n in arms:  # a warm observation, and the first update's loss on each arm
            n.restore(init[name])
            _, events = n.train(xs, ys, rates)
            first[name] = events[0]["loss"]
        ms = {name: [] for name, _ in arms}
        digests = {name: set() for name, _ in arms}
        for q in range(a.rounds):
            order = arms[q % len(arms) :] + arms[: q % len(arms)]
            for name, n in order:
                n.restore(init[name])
                seconds, _ = n.train(xs, ys, rates)
                ms[name].append(1000 * seconds / a.steps)
                digests[name].add(n.digest())
        base = ms["FP32"]
        out = {
            "model": model,
            "batch": 4,
            "time": t,
            "recipe": asdict(r),
            "parameters": int(arms[0][1].plan.parameters),
            "threads": a.threads,
            "rounds": a.rounds,
            "updates_per_observation": a.steps,
            "bf16_variants_mask": mask,
            "amx": bool(mask & 8),
            "contracts": dict(use),
            "workspace_bytes": {
                name: 0 if n.pack is None else int(n.pack.nbytes) for name, n in arms
            },
            "first_update_loss": first,
            "deterministic": {name: len(d) == 1 for name, d in digests.items()},
            "fp32_equals_aa": digests["FP32"] == digests["FP32-AA"],
            "ms_per_update_median": {k: statistics.median(v) for k, v in ms.items()},
            "over_fp32": {
                k: interval([x / b for b, x in zip(base, v)]) for k, v in ms.items() if k != "FP32"
            },
            "raw_ms": ms,
        }
    finally:
        for _, n in arms:
            n.close()
    m = out["ms_per_update_median"]
    print(
        json.dumps(
            {
                "case": f"{model}/B4T{t}",
                "amx": out["amx"],
                "ms": {k: round(v, 3) for k, v in m.items()},
                "over_fp32": {
                    k: [round(v["geometric_mean"], 3), *(round(x, 3) for x in v["ci95"])]
                    for k, v in out["over_fp32"].items()
                },
                "first_loss": {k: round(v, 5) for k, v in first.items()},
            }
        ),
        flush=True,
    )
    return out


cases = []
for spec in a.cases.split(","):
    model, t = spec.split(":")
    cases.append(case(model, int(t)))
    a.out.write_text(json.dumps({"what": __doc__, "cases": cases}, indent=1) + "\n")
