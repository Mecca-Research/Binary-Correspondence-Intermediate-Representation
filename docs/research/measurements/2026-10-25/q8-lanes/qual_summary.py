"""Q8-LANES' qualification rows: the C latency section, the logits' agreement, the simulated cache rows
(callgrind's model: one run of each rail; under valgrind the AVX2 instance runs) and every check's verdict of two
run_qualification.py reports on one model (BEFORE: the tree before Q8-LANES; AFTER: the tree after), each tool run
from its own tree with --require-callgrind; prints one JSON object (no host paths).
usage: qual_summary.py BEFORE.json AFTER.json"""

import json
import sys

rows = {}
for tag, path in (("before", sys.argv[1]), ("after", sys.argv[2])):
    d = json.load(open(path))
    c = d["latency"]["c"]
    rows[tag] = {
        "commit": d.get("commit"),
        "status": d.get("status"),
        "monolithic_median_ms": round(c["monolithic_median_ms"], 3),
        "gem_median_ms": round(c["gem_median_ms"], 3),
        "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4),
        "trust_boundary_ms": round(c["trust_boundary_ms"], 3),
        "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3),
        "per_token_ms_median": round(c["per_token_ms"]["median"], 4),
        "tokens_per_second": round(c["tokens_per_second"], 1),
        "per_operation_ms": {k: round(v, 3) for k, v in c["per_operation_ms"].items()},
        "c_logits_sha256": d["correctness"]["logits"]["c_gem_sha256"],
        "c_gem_equals_monolithic": d["correctness"]["logits"]["c_bitwise_identical"],
        "max_abs_c_vs_oracle": d["correctness"]["logits"]["max_abs_c_vs_python"],
        "simulated": {
            rail: {
                k: d["traffic"]["simulated"][rail][k]
                for k in ("Ir", "data_refs", "D1mr", "D1mw", "ll_data_misses")
            }
            for rail in ("gem", "monolithic")
        },
        "checks": {c["check"]: c["ok"] for c in d["checks"]},
    }
print(json.dumps(rows, indent=1, sort_keys=True))
