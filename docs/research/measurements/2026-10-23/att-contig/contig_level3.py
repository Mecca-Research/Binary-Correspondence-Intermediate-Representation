#!/usr/bin/env python3
"""ATT-CONTIG on the GEO study's attention model (level 3): ../../2026-10-22/infer-attn/build_level3.py
builds the level-2 overlay (GEO-2b) against a tensor unit; this takes that build and makes three exact
edits to its attention.c, so its binary32 attention runs through the tensor unit's workspace entries
as main's trainer runs it: the forward and the backward call them, each lent the regions the pool
needs, from a buffer the overlay holds (the study's FP32 trainer lends no workspace; main's trainer
lends its state's), grown to the largest need it has met and kept. The entries' declarations are
copied from the tensor unit's own header -- main's, which the overlay builds against, has none of
them -- so their prototypes are the unit's; then the overlay and the shared object are rebuilt with
build_level3's own two commands. Nothing else of the overlay changes.

usage: python3 -I contig_level3.py OUT_WORK TENSOR_SRC
  OUT_WORK as build_level3.py left it (with --scratch); rewrites OUT_WORK/build/attention.c,
  attention.o and attention.so, and writes OUT_WORK/build/attention-build-contig.json
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

OUT, TENSOR = (Path(a) for a in sys.argv[1:3])
ENTRIES = (
    "bcir_tensor_attention_pool_workspace",
    "bcir_tensor_attention_contract_variant_scan_ws_pool",
    "bcir_tensor_attention_backward_contract_variant_ws_pool",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def declarations(header):
    """Each entry's declaration as the header spells it, whole (from BCIR_DT_API to its `;`)."""
    text = Path(header).read_text(encoding="utf-8")
    out = []
    for name in ENTRIES:
        found = re.findall(r"BCIR_DT_API[^;{]*?\b" + name + r"\s*\([^;]*?\)\s*;", text, re.S)
        if len(found) != 1:
            raise SystemExit(f"{name}: {len(found)} declarations in {header}")
        out.append(found[0])
    return out


build = OUT / "build"
record = json.loads((build / "attention-build-level3.json").read_text(encoding="utf-8"))
if not record.get("scratch"):
    raise SystemExit("the level-3 build must be build_level3.py's --scratch build")
decls = declarations(TENSOR / "bcir_tensor.h")
HELPER = (
    "/* ATT-CONTIG (level 3, contig_level3.py): the tensor unit's workspace entries, declared as its\n"
    " * header declares them, and the regions they copy a head's rows into -- held here, since the\n"
    " * study's FP32 trainer lends no workspace: grown to the most the pool has needed, then kept. */\n"
    "#include <stdlib.h>\n" + "\n".join(decls) + "\n"
    "static void *contig_ws;\n"
    "static size_t contig_bytes;\n"
    "static void *contig_lend(const bcir_decoder_state *s,size_t b,size_t t,size_t h,size_t kh,\n"
    "    size_t d,size_t *bytes) {\n"
    "  size_t need=bcir_tensor_attention_pool_workspace(b,t,h,kh,d,bcir_tensor_pool_threads(s->pool));\n"
    "  if (need>contig_bytes) {\n"
    "    free(contig_ws); contig_ws=malloc(need); contig_bytes=contig_ws ? need : 0;\n"
    "  }\n"
    "  *bytes=contig_bytes;\n"
    "  return contig_ws;\n"
    "}\n"
)
EDITS = (
    (
        "static int attention(bcir_decoder_state *s,int scratch,size_t b,size_t t,size_t h,size_t kh,"
        "size_t d,\n",
        HELPER
        + "static int attention(bcir_decoder_state *s,int scratch,size_t b,size_t t,size_t h,size_t kh,"
        "size_t d,\n",
    ),
    (
        "  bcir_tensor_attention_contract_scan_pool(s->pool,contract,b,t,h,kh,d,q,k,v,p,y,bad);\n",
        "  {\n"
        "    size_t n; void *w=contig_lend(s,b,t,h,kh,d,&n);\n"
        "    bcir_tensor_attention_contract_variant_scan_ws_pool(s->pool,0,contract,b,t,h,kh,d,q,k,v,p,y,\n"
        "        bad,w,n);\n"
        "  }\n",
    ),
    (
        "  bcir_tensor_attention_backward_contract_pool(s->pool,attention_contract(s),g.b,g.t,g.h,g.kh,\n"
        "      g.hd,q,k,v,p,dy,dq,dk,dv);\n",
        "  {\n"
        "    size_t n; void *w=contig_lend(s,g.b,g.t,g.h,g.kh,g.hd,&n);\n"
        "    bcir_tensor_attention_backward_contract_variant_ws_pool(s->pool,0,attention_contract(s),g.b,g.t,\n"
        "        g.h,g.kh,g.hd,q,k,v,p,dy,dq,dk,dv,w,n);\n"
        "  }\n",
    ),
)
source = (build / "attention.c").read_text(encoding="utf-8")
before = sha(build / "attention.c")
for old, new in EDITS:
    if source.count(old) != 1:
        raise SystemExit(f"edit anchor found {source.count(old)} times: {old[:70]!r}")
    source = source.replace(old, new)
(build / "attention.c").write_text(source, encoding="utf-8")
warnings = []
# the overlay's object and the shared object, as build_level3 built them
for cmd in record["commands"][1:]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)}\n{r.stderr}")
    warnings.append(r.stderr)
out = {
    "level3_record": "attention-build-level3.json",
    "commands": record["commands"][1:],
    "compiler_stderr": warnings,
    "entries_declared": decls,
    "edits": len(EDITS),
    "overlay_before_sha256": before,
    "overlay_sha256": sha(build / "attention.c"),
    "tensor_h_sha256": sha(TENSOR / "bcir_tensor.h"),
    "native_sha256": sha(build / "attention.so"),
}
(build / "attention-build-contig.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print("contig", build / "attention.so", len(EDITS), "edits,", len(decls), "entries declared")
