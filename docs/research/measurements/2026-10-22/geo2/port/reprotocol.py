"""GEO-2b level 2: the frozen protocol carried to the port.

Every entry of Codex's protocol.json keeps its hash, except the sources the port changes by
construction: the eight overlays build_main.py grafts from current main (build/<name>.c) and the
three drivers host_abi5.py edits for ABI 5 (native_abi.py, experiment.py, policy.py). Those
entries take the ported files' hashes, and the protocol records which entries changed and why, so
freeze() keeps holding every other source -- the drivers, the foundation overlays, the vendored
pins, the data -- to the bytes Codex froze.
usage: python3 -I reprotocol.py WORK_DIR
"""

import hashlib, json, sys
from pathlib import Path

work = Path(sys.argv[1])
p = work / "protocol.json"
proto = json.loads(p.read_text())
grafted = [
    "build/" + n + ".c"
    for n in (
        "guard_compact",
        "guard_frame_ff_compact",
        "guard_layer_frame_ff",
        "query_guard_compact",
        "query_guard_frame_ff_compact",
        "query_guard_layer_frame_ff",
        "guard_hybrid_compact",
        "attention",
    )
]
edited = ["native_abi.py", "experiment.py", "policy.py"]
changed = {}
for name in grafted + edited:
    old = proto["source_sha256"][name]
    new = hashlib.sha256((work / name).read_bytes()).hexdigest()
    proto["source_sha256"][name] = new
    changed[name] = {"frozen": old, "ported": new}
proto["geo2_level2"] = {
    "note": "GEO-2b level 2: the study rerun on current main. These entries are the port, by "
    "construction; every other entry is the protocol Codex froze.",
    "grafted_from_main": "build_main.py: the blocks build_upgrade.py grafts, taken from origin/main",
    "edited_for_abi5": "host_abi5.py: the trailing ABI 5 recompute word (0) and ABI 5 admitted",
    "changed": changed,
}
p.write_text(json.dumps(proto, indent=2))
print(
    "protocol: %d entries re-hashed, %d kept"
    % (len(changed), len(proto["source_sha256"]) - len(changed))
)
