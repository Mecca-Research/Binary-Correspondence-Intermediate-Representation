#!/usr/bin/env python3
"""INFER-ATTN end to end: bcir_decoder_forward built from the tree before (BEFORE) and after (AFTER),
each invocation K timed forwards after a warm one (fwd_bench.c), in rotated rounds of BEFORE, AFTER
and AFTER again (the A/A control), the logits' digest held equal across every invocation first.
usage: python3 -I fwd_rounds.py BEFORE_BIN AFTER_BIN OUT.json [--rounds 12] [--k 5]"""

import argparse
import json
import math
import statistics
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("before")
p.add_argument("after")
p.add_argument("out")
p.add_argument("--rounds", type=int, default=12)
p.add_argument("--k", type=int, default=5)
a = p.parse_args()
#: (name, vocab, width, heads, kvheads, layers, ff, batch, time, contract, threads)
CASES = [
    ("decoder-9.84M kblock+exp+reduce+attn", 260, 384, 8, 4, 6, 1024, 8, 128, 46, 4),
    ("decoder-9.84M bf16+exp+reduce+attnbf16", 260, 384, 8, 4, 6, 1024, 8, 128, 90, 4),
    ("geo-dims T1024 kblock+exp+reduce", 260, 128, 4, 4, 3, 256, 4, 1024, 14, 4),
    ("geo-dims T2048 kblock+exp+reduce", 260, 128, 4, 4, 3, 256, 4, 2048, 14, 4),
    ("geo-dims T2048 bf16+exp+reduce+attnbf16", 260, 128, 4, 4, 3, 256, 4, 2048, 90, 4),
    ("geo-dims T2048 kblock+exp+reduce, one thread", 260, 128, 4, 4, 3, 256, 4, 2048, 14, 1),
]
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


doc = {"k_forwards_per_invocation": a.k, "rounds": a.rounds, "cases": []}
for case in CASES:
    bins = {"BEFORE": a.before, "AFTER": a.after, "AFTER-AA": a.after}
    ms, digests, floats = {k: [] for k in bins}, set(), {}
    names = list(bins)
    for r in range(a.rounds):
        for j in range(len(names)):
            arm = names[(r + j) % len(names)]
            got = run(bins[arm], case)
            ms[arm].append(got["median_ms"])
            digests.add(got["logits_fnv1a"])
            floats[arm] = got["activations_floats"]
    assert len(digests) == 1, (case[0], digests)  # one set of logits, before and after
    row = {"case": case[0], "spec": dict(zip(("vocab", "width", "heads", "kvheads", "layers", "ff",
                                             "batch", "time", "contract", "threads"), case[1:])),
           "logits_fnv1a": digests.pop(), "activations_floats": floats["AFTER"],
           "ms_per_forward": {k: statistics.median(v) for k, v in ms.items()},
           "after_over_before": interval([x / y for x, y in zip(ms["AFTER"], ms["BEFORE"])]),
           "aa": interval([x / y for x, y in zip(ms["AFTER-AA"], ms["AFTER"])]),
           "raw_ms": ms}  # fmt: skip
    doc["cases"].append(row)
    print(
        json.dumps({k: row[k] for k in ("case", "ms_per_forward", "after_over_before", "aa")}),
        flush=True,
    )
with open(a.out, "w", encoding="utf-8") as f:
    f.write(json.dumps(doc, indent=1) + "\n")
print("wrote", a.out)
