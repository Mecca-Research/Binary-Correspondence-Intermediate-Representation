"""Median per binary of bwd_bench4 lines tagged 'NAME {json}': TSC cycles a probability (x threads) for a few phases.
usage: summ.py FILE"""

import json
import statistics
import sys

rows = {}
for line in open(sys.argv[1], encoding="utf-8"):
    name, _, js = line.partition(" ")
    x = json.loads(js)
    b, t, h, kh, d = x["shape"]
    probs = b * h * t * (t + 1) / 2
    r = {k: x[k] / probs for k in ("dv", "dk", "dq", "dp", "dot", "ds")}
    r["wall_ms"] = x["wall_cycles_per_call"] / 2.3e6
    r["check"] = (x["dq0"], x["dk0"], x["dv0"])
    rows.setdefault(name, []).append(r)
base = None
for name, rs in rows.items():
    med = {
        k: statistics.median(r[k] for r in rs)
        for k in ("dv", "dk", "dq", "dp", "dot", "ds", "wall_ms")
    }
    checks = {r["check"] for r in rs}
    print(
        f"{name:14s} n={len(rs)} wall {med['wall_ms']:6.2f} ms  dv {med['dv']:.3f} dk {med['dk']:.3f} dq {med['dq']:.3f} "
        f"dp {med['dp']:.3f} dot {med['dot']:.3f} ds {med['ds']:.3f}  checks {len(checks)} {sorted(checks)[0]}"
    )
