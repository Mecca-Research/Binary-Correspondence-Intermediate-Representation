"""Read bwd_bound_bench's JSON lines: each phase in TSC cycles a probability (summed over threads), and the
products' share of v1's measured rate (fma_peak.c: 102 GFLOP/s a core in multiply-then-add, i.e. a 32-term
product of v1 in 32*2/102e9 s = 1.45 TSC cycles at 2.3 GHz; a d-term product d/32 of that).
usage: read_phases.py FILE.jsonl [V1_GFLOPS_PER_CORE]"""

import json
import sys

TSC = 2.3e9
v1 = float(sys.argv[2]) if len(sys.argv) > 2 else 102.0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    b, t, h, kh, d = r["shape"]
    probs = b * h * t * (t + 1) // 2
    bound = (2.0 * d / (v1 * 1e9)) * TSC  # TSC cycles for one d-term v1 product
    th = r["threads"]
    wall = r["wall_cycles_per_call"]
    print(
        f"shape {r['shape']} threads {th} contract {r['contract']} ws {r['workspace_bytes']} rounds {r['rounds']}"
    )
    print(
        f"  wall {wall / TSC * 1e3:.2f} ms a call; {wall * th / probs:.2f} TSC cycles a probability (x threads)"
    )
    tot = 0.0
    for k in ("dv", "dp", "dot", "ds", "dq", "dk", "dp_again"):
        c = r[k] / probs
        tot += c
        share = (
            f"  {bound / c * 100:5.1f}% of v1's rate" if k in ("dv", "dp", "dq", "dk") and c else ""
        )
        print(f"  {k:9s} {c:6.2f}{share}")
    ct = r["contig_task"] / probs
    print(
        f"  heads    {tot:6.2f}; contiguous tasks whole {ct:6.2f} (copies and the rest {ct - tot:6.2f})"
        if ct
        else f"  heads    {tot:6.2f}"
    )
    print(f"  v1 bound of the four products: {4 * bound:.2f}")
