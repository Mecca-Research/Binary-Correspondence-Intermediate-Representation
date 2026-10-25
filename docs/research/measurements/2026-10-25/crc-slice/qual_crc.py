"""CRC-SLICE's qualification rows: of two run_qualification.py reports on one model (BEFORE: the tree before CRC-SLICE;
AFTER: the tree after), each tool run from its own tree with --require-callgrind -- the C rail's load, its DER ->
native conversion (it writes the native pack's CRC), the trust boundary, GEM against monolithic, the first token, the
logits' agreement, the simulated cache rows (callgrind's model, one run of each rail) and every check's verdict.
Prints one JSON object (no host paths).  usage: qual_crc.py BEFORE.json AFTER.json"""

import json
import sys

rows = {}
for tag, path in (("before", sys.argv[1]), ("after", sys.argv[2])):
    d = json.load(open(path))
    c = d["latency"]["c"]
    rows[tag] = {
        "commit": d.get("commit"),
        "status": d.get("status"),
        "load_ms": round(c["load_ms"], 3),
        "der_to_native_ms": round(c["der_to_native_ms"], 3),
        "trust_boundary_ms": round(c["trust_boundary_ms"], 3),
        "monolithic_median_ms": round(c["monolithic_median_ms"], 3),
        "gem_median_ms": round(c["gem_median_ms"], 3),
        "gem_overhead_ms": round(c["gem_overhead_ms"], 3),
        "gem_over_monolithic_median": round(c["gem_over_monolithic_median"], 4),
        "time_to_first_token_ms": round(c["time_to_first_token_ms"], 3),
        "per_token_ms_median": round(c["per_token_ms"]["median"], 4),
        "c_logits_sha256": d["correctness"]["logits"]["c_gem_sha256"],
        "c_gem_equals_monolithic": d["correctness"]["logits"]["c_bitwise_identical"],
        "max_abs_c_vs_oracle": d["correctness"]["logits"]["max_abs_c_vs_python"],
        "pack_native_sha256": d["artifacts"]["pack"]["native"]["sha256"],
        "simulated": {
            rail: {
                k: d["traffic"]["simulated"][rail][k] for k in ("Ir", "data_refs", "D1mr", "D1mw")
            }
            for rail in ("gem", "monolithic")
        },
        "checks": {ch["check"]: ch["ok"] for ch in d["checks"]},
    }
print(json.dumps(rows, indent=1, sort_keys=True))
