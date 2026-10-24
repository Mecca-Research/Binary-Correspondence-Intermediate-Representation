"""COMPILE-LIMIT's audit: every P-IND (torch.compile) case this branch's records hold, classified by how it compiled.

Each record below is one process (its `confirm`, where it has one, another): geo_compare.py, train_compare.py and
infer_compare.py compile one P-IND module a case, all through oracle.torch_model_next's one forward code object, so
the k-th case of a process (from 0) is Dynamo's k-th compile of that code object -- witness.py shows the mechanism:
  k = 0      static: the case's own fixed shapes
  1 <= k < 8 dynamic: automatic dynamic shapes, every dimension an earlier case's differed in made dynamic (the
             tokens' time; in geo-compare.json past its first model, parameters' too)
  k >= 8     eager: past Dynamo's recompile limit (8) the call compiles nothing and runs the module eagerly
usage: python3 -I audit.py REPO OUT.json
"""

import json
import sys
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
M = REPO / "docs/research/measurements"
LIMIT = 8
# record -> the processes it holds: (the json path of a case list, the key of that list), in run order
RECORDS = {
    "2026-10-22/geo2/geo-compare.json": [[("train", "bcir_over"), ("infer", "bcir_over")]],
    "2026-10-22/bwd-width/geo-long.json": [[("train", "bcir_over"), ("infer", "bcir_over")]],
    "2026-10-22/infer-attn/geo-infer.json": [[("cases", "after_over")]],
    "2026-10-22/attn-rate/geo-infer.json": [[("cases", "after_over")]],
    "2026-10-22/bwd-width/geo-train.json": [[("cases", "after_over")]],
    "2026-10-22/bwd-width/geo-train-second.json": [[("cases", "after_over")]],
    "2026-10-22/bwd-long/geo-train.json": [[("cases", "after_over")]],
    "2026-10-23/att-contig/geo-train.json": [[("cases", "after_over")]],
    "2026-10-23/att-contig/prototype.json": [
        [("layout_and_copies/geo_prototype/cases", "after_over")]
    ],
    "2026-10-23/fwd-strip/geo-train.json": [
        [("cases", "after_over")],
        [("confirm/cases", "after_over")],
    ],
    "2026-10-23/p-lower/geo-train.json": [[("cases", "after_over")]],
    "2026-10-23/bwd-keep/geo-train.json": [[("cases", "after_over")]],
    "2026-10-23/att-bound/inference.json": [[("cases", "after_over")]],
    "2026-10-23/mm-rows/geo-train.json": [[("cases", "after_over")]],
    "2026-10-23/mm-rows/inference.json": [[("cases", "after_over")]],
    "2026-10-23/scores-t/inference.json": [[("cases", "after_over")]],
    "2026-10-23/softmax-t/geo-train.json": [[("cases", "after_over")]],
    "2026-10-23/softmax-t/inference.json": [
        [("cases", "after_over")],
        [("confirm/cases", "after_over")],
    ],
}

# records whose P-IND compiles every shape from a fresh Dynamo state already
CLEAN = {
    "2026-10-22/ot/ot.json": "tools/perf/ot_torch.py: OT-05's block compiles each shape after torch._dynamo.reset() "
    "with dynamic=False, and OT-02/OT-10/OT-11 run each compiled route in its own process",
}


def at(doc, path):
    for part in path.split("/"):
        doc = doc[part]
    return doc


rows, seen = [], set()
for rel, processes in RECORDS.items():
    doc = json.loads((M / rel).read_text(encoding="utf-8"))
    for pi, parts in enumerate(processes):
        k = 0
        for path, ratio_key in parts:
            for c in at(doc, path):
                ms = c.get("ms_per_update") or c.get("ms_per_forward")
                if "P-IND" not in ms:
                    continue
                state = "static" if k == 0 else "dynamic" if k < LIMIT else "eager"
                compile_s = c.get("compile_first_update_s", c.get("compile_first_call_s"))
                rows.append(
                    {
                        "record": rel,
                        "process": pi,
                        "part": path,
                        "model": c.get("model", "attention"),
                        "time": c["time"],
                        "mode": "infer" if "ms_per_forward" in c else "train",
                        "compile_index": k,
                        "state": state,
                        "compile_s": round(compile_s, 3),
                        "p_eager_ms": round(ms["P-EAGER"], 2),
                        "p_ind_ms": round(ms["P-IND"], 2),
                        "b_over_p_ind": round(c[ratio_key]["P-IND"]["geometric_mean"], 3),
                        "b_over_p_eager": round(c[ratio_key]["P-EAGER"]["geometric_mean"], 3),
                    }
                )
                k += 1
    seen.add(rel)
# every record of the branch's measurements that holds a P-IND arm is in the table above
holding = sorted(
    str(p.relative_to(M))
    for p in M.glob("2026-10-2*/**/*.json")
    if p.name != "README.json" and '"P-IND"' in p.read_text(encoding="utf-8")
)
missing = [h for h in holding if h not in seen and h not in CLEAN]
assert not missing, f"records with P-IND the audit does not classify: {missing}"
counts = {s: sum(r["state"] == s for r in rows) for s in ("static", "dynamic", "eager")}
doc = {"what": __doc__, "counts": counts, "clean": CLEAN, "rows": rows}
OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
print(counts)
for r in rows:
    print(
        f"{r['record']:44s} p{r['process']} {r['mode']:5s} {r['model']:10s} T{r['time']:<5d} k{r['compile_index']} "
        f"{r['state']:7s} compile {r['compile_s']:7.3f} s  P-IND {r['p_ind_ms']:7.2f}  "
        f"P-EAGER {r['p_eager_ms']:7.2f}  B/P-IND {r['b_over_p_ind']:.3f}"
    )
