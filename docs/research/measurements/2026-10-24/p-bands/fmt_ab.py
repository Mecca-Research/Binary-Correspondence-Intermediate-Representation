"""ab_bands lines: each arm's ratio over base per phase, and the verdict. usage: fmt_ab.py FILE"""

import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    if not line.startswith("{"):
        continue
    x = json.loads(line)
    arms = x["arms"]
    parts = []
    for ph in ("forward", "backward", "total"):
        pr, aa = arms["proto"][ph], arms["A/A"][ph]
        lo, hi = pr["interval"]
        v = (
            "not resolved: holds 1"
            if lo <= 1 <= hi
            else "resolved"
            if hi < aa["interval"][0] or lo > aa["interval"][1]
            else "not resolved: meets A/A"
        )
        parts.append(
            f"{ph} {arms['base'][ph]['median_ms']:.2f} ms, proto {pr['ratio']:.3f} ({lo:.3f}-{hi:.3f}), A/A {aa['ratio']:.3f} "
            f"({aa['interval'][0]:.3f}-{aa['interval'][1]:.3f}): {v}"
        )
    print(
        f"B{x['shape'][0]} T{x['shape'][1]} {x['shape'][2]}x{x['shape'][3]} heads of {x['shape'][4]}, {x['threads']} thread(s), contract {x['contract']}, {x['rounds']} rounds, outputs equal {x.get('outputs_equal')}"
    )
    for p in parts:
        print("  ", p)
