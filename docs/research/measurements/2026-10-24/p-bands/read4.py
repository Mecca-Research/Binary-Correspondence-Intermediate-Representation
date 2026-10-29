"""bwd_bench4's lines: TSC cycles a probability (x threads) a phase, and the v1 rate of each product.
usage: read4.py LINES.jsonl"""

import json
import sys

V1 = (
    1.443 / 32
)  # TSC cycles a multiply-add at v1's measured rate on this host (bwd-vt's micro.json, d=32)
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.startswith("{"):
        continue
    x = json.loads(line)
    b, t, h, kh, d = x["shape"]
    probs = b * h * t * (t + 1) / 2
    c = {
        k: v / probs
        for k, v in x.items()
        if isinstance(v, (int, float))
        and k not in ("shape", "threads", "contract", "evict", "workspace_bytes", "rounds")
        and k not in ("dq0", "dk0", "dv0")
    }
    wall = x["wall_cycles_per_call"] * x["threads"] / probs
    inner = sum(
        c[k] for k in ("dv", "dp", "dot", "ds", "dq", "dk", "dp_again", "tile_zero", "tile_panel")
    )
    copies = sum(c[k] for k in ("kv_in", "qdy_in", "dq_out", "dkdv_out"))
    print(
        f"shape {x['shape']} threads {x['threads']} contract {x['contract']} evict {x['evict']}: wall "
        f"{x['wall_cycles_per_call'] / 2.3e6:.2f} ms, {wall:.2f} TSC cycles a probability (x threads)"
    )
    for k in ("dv", "dp", "tile_zero", "tile_panel", "dot", "ds", "dq", "dk", "dp_again"):
        rate = (
            f"  {100 * V1 * d / c[k]:5.1f}% of v1's rate" if k in ("dv", "dp", "dq", "dk") else ""
        )
        print(f"  {k:10s} {c[k]:6.3f}{rate}")
    for k in ("kv_in", "qdy_in", "dq_out", "dkdv_out"):
        print(f"  {k:10s} {c[k]:6.3f}")
    rest = c["contig_task"] - inner - copies
    print(
        f"  heads {inner:.2f}, copies {copies:.2f}, task whole {c['contig_task']:.2f} (unplaced {rest:.2f}); "
        f"wall less tasks {wall - c['contig_task']:.2f}; v1 bound of the four products {4 * V1 * d:.2f}"
    )
