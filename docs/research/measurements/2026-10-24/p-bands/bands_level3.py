#!/usr/bin/env python3
"""P-BANDS on the GEO study's attention model (level 3): ../../2026-10-23/p-lower/lower_level3.py leaves the level-3
overlay's binary32 attention calling the tensor unit's workspace entries, its stored forward under
BCIR_TENSOR_ATTENTION_LOWER; this makes two exact edits to that attention.c, so its stored forward and its backward run
under BCIR_TENSOR_ATTENTION_BANDS (32) as this branch's trainer's now do -- the overlay's backward is the tensor unit's
binary32 backward and nothing else of the overlay reads the probabilities -- and rebuilds the overlay and the shared
object with build_level3's own two commands. A tree before P-BANDS ignores the bit, so both arms take the same edits
and differ in the tensor unit alone.

usage: python3 -I bands_level3.py OUT_WORK
  OUT_WORK as lower_level3.py left it; rewrites OUT_WORK/build/attention.c, attention.o and attention.so, and writes
  OUT_WORK/build/attention-build-bands.json
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
build = OUT / "build"
EDITS = (
    (
        "        contract|(scratch ? 0u : BCIR_TENSOR_ATTENTION_LOWER),b,t,h,kh,d,q,k,v,p,y,bad,w,n);\n",
        "        contract|(scratch ? 0u : BCIR_TENSOR_ATTENTION_LOWER|BCIR_TENSOR_ATTENTION_BANDS),b,t,h,kh,d,q,k,v,p,y,bad,w,\n"
        "        n);\n",
    ),
    (
        "    bcir_tensor_attention_backward_contract_variant_ws_pool(s->pool,0,attention_contract(s),g.b,g.t,\n",
        "    bcir_tensor_attention_backward_contract_variant_ws_pool(s->pool,0,\n"
        "        attention_contract(s)|BCIR_TENSOR_ATTENTION_BANDS,g.b,g.t,\n",
    ),
)
GUARD_AT = "static int attention(bcir_decoder_state *s,int scratch,"
GUARD = (
    "#ifndef BCIR_TENSOR_ATTENTION_BANDS\n"
    "#define BCIR_TENSOR_ATTENTION_BANDS 32u  /* P-BANDS' bit (the after tree's bcir_tensor.h) */\n"
    "#endif\n"
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if not (build / "attention-build-lower.json").is_file():
    raise SystemExit("the level-3 build must be lower_level3.py's")
record = json.loads((build / "attention-build-level3.json").read_text(encoding="utf-8"))
source = (build / "attention.c").read_text(encoding="utf-8")
before = sha(build / "attention.c")
for anchor in (*(old for old, _ in EDITS), GUARD_AT):
    if source.count(anchor) != 1:
        raise SystemExit(f"edit anchor found {source.count(anchor)} times: {anchor[:70]!r}")
for old, new in EDITS:
    source = source.replace(old, new)
source = source.replace(GUARD_AT, GUARD + GUARD_AT)
(build / "attention.c").write_text(source, encoding="utf-8")
warnings = []
# the overlay's object and the shared object, as build_level3 built them
for cmd in record["commands"][1:]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)}\n{r.stderr}")
    warnings.append(r.stderr)
out = {
    "lower_record": "attention-build-lower.json",
    "commands": record["commands"][1:],
    "compiler_stderr": warnings,
    "edits": 3,
    "overlay_before_sha256": before,
    "overlay_sha256": sha(build / "attention.c"),
    "native_sha256": sha(build / "attention.so"),
}
(build / "attention-build-bands.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print("bands", build / "attention.so", "3 edits")
