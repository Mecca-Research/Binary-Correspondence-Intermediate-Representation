#!/usr/bin/env python3
"""ATTN-RATE's timings: one bench binary built from the tree before (BEFORE) and after (AFTER) --
elem_bench.c (an elementwise kernel), kern_bench.c (the attention forward), fwd_bench.c (the decoder's
forward) or step_bench.c (its training step) -- each invocation timing K calls after a warm one, in
rotated rounds of BEFORE, AFTER and AFTER again (the A/A control); one digest of the outputs across
every invocation of a case first, so the two builds are timed on the same bits.
usage: python3 -I bench_rounds.py BEFORE_BIN AFTER_BIN OUT.json --cases SET [--only TEXT]
       [--rounds 12] [--k 5]   (SET: elementwise, attention, forward or step)"""

import argparse
import json
import math
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument("before")
p.add_argument("after")
p.add_argument("out")
p.add_argument("--cases", choices=("elementwise", "attention", "forward", "step"), required=True)
p.add_argument("--only", default="")
p.add_argument("--rounds", type=int, default=12)
p.add_argument("--k", type=int, default=5)
a = p.parse_args()

DECODER = (
    "vocab",
    "width",
    "heads",
    "kvheads",
    "layers",
    "ff",
    "batch",
    "time",
    "contract",
    "threads",
)
#: set -> (the spec's fields, the digest's key, [(name, *args)]); K follows the args on the command line
SETS = {
    "elementwise": (
        ("kernel", "n"),
        "digest_fnv1a",
        [
            ("exp, 16K elements", "exp", 16384),
            ("exp, 1M elements", "exp", 1048576),
            ("silu_mul_v2, 16K elements", "silu_mul_v2", 16384),
            ("silu_mul_v2, 1M elements", "silu_mul_v2", 1048576),
            ("swiglu_vjp_v2, 16K elements", "swiglu_vjp_v2", 16384),
            ("swiglu_vjp_v2, 1M elements", "swiglu_vjp_v2", 1048576),
        ],
    ),
    "attention": (
        ("batch", "time", "heads", "kvheads", "dim", "threads", "contract", "scratch"),
        "y_fnv1a",
        [
            ("geo 4x1024x4x4x32 scratch", 4, 1024, 4, 4, 32, 4, 1, 1),
            ("geo 4x2048x4x4x32 scratch", 4, 2048, 4, 4, 32, 4, 1, 1),
            ("geo 4x2048x4x4x32 scratch, one thread", 4, 2048, 4, 4, 32, 1, 1, 1),
            ("geo 4x2048x4x4x32 stored", 4, 2048, 4, 4, 32, 4, 1, 0),
            ("geo 1x8192x4x4x32 scratch", 1, 8192, 4, 4, 32, 4, 1, 1),
            ("decoder 8x128x8x4x48 stored", 8, 128, 8, 4, 48, 4, 3, 0),
            ("decoder 8x128x8x4x48 scratch", 8, 128, 8, 4, 48, 4, 3, 1),
            ("1x4096x8x8x64 scratch", 1, 4096, 8, 8, 64, 4, 3, 1),
        ],
    ),
    "forward": (
        DECODER,
        "logits_fnv1a",
        [
            ("decoder-9.84M kblock+exp+reduce+attn", 260, 384, 8, 4, 6, 1024, 8, 128, 46, 4),
            ("decoder-9.84M bf16+exp+reduce+attnbf16", 260, 384, 8, 4, 6, 1024, 8, 128, 90, 4),
            ("geo-dims T1024 kblock+exp+reduce", 260, 128, 4, 4, 3, 256, 4, 1024, 14, 4),
            ("geo-dims T2048 kblock+exp+reduce", 260, 128, 4, 4, 3, 256, 4, 2048, 14, 4),
            ("geo-dims T2048 bf16+exp+reduce+attnbf16", 260, 128, 4, 4, 3, 256, 4, 2048, 90, 4),
            (
                "geo-dims T2048 kblock+exp+reduce, one thread",
                260,
                128,
                4,
                4,
                3,
                256,
                4,
                2048,
                14,
                1,
            ),
        ],
    ),
    "step": (
        DECODER,
        "weights_fnv1a",
        [
            ("decoder-9.84M kblock+exp+reduce+attn, step", 260, 384, 8, 4, 6, 1024, 8, 128, 46, 4),
            (
                "decoder-9.84M kblock+exp+reduce+attn, step, one thread",
                260,
                384,
                8,
                4,
                6,
                1024,
                8,
                128,
                46,
                1,
            ),
            ("geo-dims T512 kblock+exp+reduce, step", 260, 128, 4, 4, 3, 256, 4, 512, 14, 4),
        ],
    ),
}
T95 = {7: 2.365, 11: 2.201, 15: 2.131, 23: 2.069}


def interval(ratios):
    logs = [math.log(r) for r in ratios]
    m = statistics.fmean(logs)
    h = T95.get(len(logs) - 1, 2.0) * statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "geometric_mean": math.exp(m),
        "ci95": [math.exp(m - h), math.exp(m + h)],
        "rounds": len(logs),
    }


def run(binary, case):
    args = [binary, *(str(v) for v in case[1:]), str(a.k)]
    out = subprocess.run(args, capture_output=True, text=True, timeout=900)
    if out.returncode:
        raise SystemExit(f"{binary} {case[0]}: exit {out.returncode}: {out.stderr.strip()}")
    return json.loads(out.stdout)


fields, key, cases = SETS[a.cases]
cases = [c for c in cases if a.only in c[0]]
if not cases:
    raise SystemExit(f"no {a.cases} case matches {a.only!r}")
doc = {"set": a.cases, "k_calls_per_invocation": a.k, "rounds": a.rounds, "cases": []}
for case in cases:
    bins = {"BEFORE": a.before, "AFTER": a.after, "AFTER-AA": a.after}
    ms, digests, last = {k: [] for k in bins}, set(), {}
    names = list(bins)
    for r in range(a.rounds):
        for j in range(len(names)):
            arm = names[(r + j) % len(names)]
            got = run(bins[arm], case)
            ms[arm].append(got["median_ms"])
            digests.add(got[key])
            last[arm] = got
    assert len(digests) == 1, (case[0], digests)  # one set of bits, before and after
    extra = {k: v for k, v in last["AFTER"].items() if k not in ("median_ms", key)}
    row = {"case": case[0], "spec": dict(zip(fields, case[1:])), key: digests.pop(), "after_reports": extra,
           "ms_per_call": {k: statistics.median(v) for k, v in ms.items()},
           "after_over_before": interval([x / y for x, y in zip(ms["AFTER"], ms["BEFORE"])]),
           "aa": interval([x / y for x, y in zip(ms["AFTER-AA"], ms["AFTER"])]),
           "raw_ms": ms}  # fmt: skip
    doc["cases"].append(row)
    print(
        json.dumps({k: row[k] for k in ("case", "ms_per_call", "after_over_before", "aa")}),
        flush=True,
    )
with open(a.out, "w", encoding="utf-8") as f:
    f.write(json.dumps(doc, indent=1) + "\n")
print("wrote", a.out)
