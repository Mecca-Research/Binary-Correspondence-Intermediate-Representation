#!/usr/bin/env python3
"""INFER-ATTN on the GEO study's attention model (level 3): GEO-2b's level-2 overlay -- Codex's
research trainer with origin/main's blocks grafted in (port/build_main.py) -- compiled against
main's headers as level 2 was, and linked with a tensor unit of this branch: the tree before
INFER-ATTN for BEFORE, the tree after it for AFTER. Every tensor entry the overlay calls has the
same prototype in main's header and the branch's (checked here, or nothing is built), so the
overlay's code is level 2's in both builds and only the tensor unit differs -- plus, with
--scratch, the overlay's inference forward passing BCIR_TENSOR_ATTENTION_SCRATCH, as INFER-ATTN's
bcir_decoder_forward does (five exact edits; the backward's forward keeps its probabilities).

usage: python3 -I build_level3.py LEVEL2_WORK OUT_WORK TENSOR_SRC MAIN_SRC [--scratch]
  writes OUT_WORK/build/attention.c, attention.so and attention-build-level3.json
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

LEVEL2, OUT, TENSOR, MAIN = (Path(a) for a in sys.argv[1:5])
SCRATCH = "--scratch" in sys.argv[5:]
EDITS = (
    (
        "static int attention(bcir_decoder_state *s,size_t b,size_t t,size_t h,size_t kh,size_t d,\n"
        "    const float *q,const float *k,const float *v,float *p,float *y,int *bad) {\n"
        "  if (s->contract&BCIR_DT_CONTRACT_ATTNBF16)\n"
        "    return bcir_tensor_attention_bf16_scan_pool(s->pool,0,attention_contract(s),b,t,h,kh,d,q,k,v,p,\n"
        "        y,bad,s->workspace,s->workspace_bytes)!=0;\n"
        "  bcir_tensor_attention_contract_scan_pool(s->pool,attention_contract(s),b,t,h,kh,d,q,k,v,p,y,bad);\n",
        "#ifndef BCIR_TENSOR_ATTENTION_SCRATCH\n"
        "#define BCIR_TENSOR_ATTENTION_SCRATCH 8u  /* INFER-ATTN's bit (the after tree's bcir_tensor.h) */\n"
        "#endif\n"
        "static int attention(bcir_decoder_state *s,int scratch,size_t b,size_t t,size_t h,size_t kh,size_t d,\n"
        "    const float *q,const float *k,const float *v,float *p,float *y,int *bad) {\n"
        "  unsigned contract=attention_contract(s)|(scratch ? BCIR_TENSOR_ATTENTION_SCRATCH : 0u);\n"
        "  if (s->contract&BCIR_DT_CONTRACT_ATTNBF16)\n"
        "    return bcir_tensor_attention_bf16_scan_pool(s->pool,0,contract,b,t,h,kh,d,q,k,v,p,\n"
        "        y,bad,s->workspace,s->workspace_bytes)!=0;\n"
        "  bcir_tensor_attention_contract_scan_pool(s->pool,contract,b,t,h,kh,d,q,k,v,p,y,bad);\n",
    ),
    (
        "static int forward(bcir_decoder_state *s,const uint32_t *tokens,geometry g) {",
        "static int forward(bcir_decoder_state *s,const uint32_t *tokens,geometry g,int scratch) {",
    ),
    (
        "    attention(s,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,a.ctx,&bad);",
        "    attention(s,scratch,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,a.ctx,&bad);",
    ),
    (
        "  return forward(s,tokens,g) ? BCIR_DT_OK : BCIR_DT_NUMERIC;",
        "  return forward(s,tokens,g,1) ? BCIR_DT_OK : BCIR_DT_NUMERIC;",
    ),
    (
        "  if (!forward(s,tokens,g)) return BCIR_DT_NUMERIC;",
        "  if (!forward(s,tokens,g,0)) return BCIR_DT_NUMERIC;",
    ),
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protos(header):
    text = Path(header).read_text(encoding="utf-8")
    pat = r"BCIR_DT_API\s+([^;{]*?)\b(bcir_tensor_[a-z0-9_]+)\s*\(([^;]*?)\)\s*;"
    return {
        m.group(2): (" ".join(m.group(1).split()), " ".join(m.group(3).split()))
        for m in re.finditer(pat, text, re.S)
    }


source = (LEVEL2 / "build" / "attention.c").read_text(encoding="utf-8")
called = sorted(set(re.findall(r"\b(bcir_tensor_[a-z0-9_]+)\s*\(", source)))
main, branch = protos(MAIN / "bcir_tensor.h"), protos(TENSOR / "bcir_tensor.h")
declared = [f for f in called if f in main]
differ = [f for f in declared if branch.get(f) != main[f]]
if differ or not declared:
    raise SystemExit(f"the overlay's tensor entries differ between the headers: {differ}")
if SCRATCH:
    for old, new in EDITS:
        if source.count(old) != 1:
            raise SystemExit(f"edit anchor found {source.count(old)} times: {old[:60]!r}")
        source = source.replace(old, new)
build = OUT / "build"
build.mkdir(parents=True, exist_ok=True)
(build / "attention.c").write_text(source, encoding="utf-8")
level2 = json.loads((LEVEL2 / "build" / "attention-build-main.json").read_text(encoding="utf-8"))
cc = level2["command"][0]
defines = [a for a in level2["command"] if a.startswith("-D")]
flags = ["-std=c11", "-O3", "-fPIC", "-ffp-contract=off", "-fno-math-errno", "-pthread"]
cmds = [
    [
        cc,
        *flags,
        "-c",
        "-I",
        str(TENSOR),
        str(TENSOR / "bcir_tensor.c"),
        "-o",
        str(build / "bcir_tensor.o"),
    ],
    [
        cc,
        *flags,
        "-c",
        "-I",
        str(MAIN),
        *defines,
        str(build / "attention.c"),
        "-o",
        str(build / "attention.o"),
    ],
    [
        cc,
        "-shared",
        "-pthread",
        str(build / "bcir_tensor.o"),
        str(build / "attention.o"),
        "-lm",
        "-o",
        str(build / "attention.so"),
    ],
]
for cmd in cmds:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)}\n{r.stderr}")
record = {
    "commands": cmds,
    "scratch": SCRATCH,
    "entries_checked": declared,
    "overlay_sha256": sha(build / "attention.c"),
    "level2_overlay_sha256": sha(LEVEL2 / "build" / "attention.c"),
    "tensor_c_sha256": sha(TENSOR / "bcir_tensor.c"),
    "native_sha256": sha(build / "attention.so"),
}
(build / "attention-build-level3.json").write_text(
    json.dumps(record, indent=1) + "\n", encoding="utf-8"
)
print(
    "built",
    build / "attention.so",
    "scratch" if SCRATCH else "stored",
    len(declared),
    "entries checked",
)
