#!/usr/bin/env python3
"""P-LOWER on the GEO study's attention model (level 3): ../att-contig/contig_level3.py leaves the level-3
overlay's binary32 attention calling the tensor unit's workspace entries; this makes one exact edit to that
attention.c, so its stored forward runs under BCIR_TENSOR_ATTENTION_LOWER (16) as main's trainer's now does --
the overlay's backward is the tensor unit's binary32 backward, which reads no probability past a row's
diagonal, and nothing else of the overlay reads the probabilities (the study reads the logits alone) -- and
rebuilds the overlay and the shared object with build_level3's own two commands. A tree before P-LOWER ignores
the bit (its forward fills every future column), so both arms take the same edit and differ in the tensor unit
alone. Nothing else of the overlay changes.

usage: python3 -I lower_level3.py OUT_WORK
  OUT_WORK as contig_level3.py left it; rewrites OUT_WORK/build/attention.c, attention.o and attention.so, and
  writes OUT_WORK/build/attention-build-lower.json
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
build = OUT / "build"
OLD = (
    "    bcir_tensor_attention_contract_variant_scan_ws_pool(s->pool,0,contract,b,t,h,kh,d,q,k,v,p,y,\n"
    "        bad,w,n);\n"
)
NEW = (
    "    bcir_tensor_attention_contract_variant_scan_ws_pool(s->pool,0,\n"
    "        contract|(scratch ? 0u : BCIR_TENSOR_ATTENTION_LOWER),b,t,h,kh,d,q,k,v,p,y,bad,w,n);\n"
)
GUARD_AT = "static int attention(bcir_decoder_state *s,int scratch,"
GUARD = (
    "#ifndef BCIR_TENSOR_ATTENTION_LOWER\n"
    "#define BCIR_TENSOR_ATTENTION_LOWER 16u  /* P-LOWER's bit (the after tree's bcir_tensor.h) */\n"
    "#endif\n"
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if not (build / "attention-build-contig.json").is_file():
    raise SystemExit("the level-3 build must be contig_level3.py's")
record = json.loads((build / "attention-build-level3.json").read_text(encoding="utf-8"))
source = (build / "attention.c").read_text(encoding="utf-8")
before = sha(build / "attention.c")
for anchor in (OLD, GUARD_AT):
    if source.count(anchor) != 1:
        raise SystemExit(f"edit anchor found {source.count(anchor)} times: {anchor[:70]!r}")
source = source.replace(OLD, NEW).replace(GUARD_AT, GUARD + GUARD_AT)
(build / "attention.c").write_text(source, encoding="utf-8")
warnings = []
# the overlay's object and the shared object, as build_level3 built them
for cmd in record["commands"][1:]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)}\n{r.stderr}")
    warnings.append(r.stderr)
out = {
    "contig_record": "attention-build-contig.json",
    "commands": record["commands"][1:],
    "compiler_stderr": warnings,
    "edits": 2,
    "overlay_before_sha256": before,
    "overlay_sha256": sha(build / "attention.c"),
    "native_sha256": sha(build / "attention.so"),
}
(build / "attention-build-lower.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print("lower", build / "attention.so", "2 edits")
