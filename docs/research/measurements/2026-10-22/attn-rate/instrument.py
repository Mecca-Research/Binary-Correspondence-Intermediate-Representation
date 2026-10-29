"""A scratch copy of bcir_tensor.c with each scratch strip's phases timed (TSC cycles, summed per
phase over the call): the K panels and scores, the softmax, the context. usage: instrument.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
hdr = "#include <x86intrin.h>\nunsigned long long bcir_ar_cycles[4];\n"
anchor = "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {\n"
assert src.count(anchor) == 1
src = src.replace(anchor, hdr + anchor, 1)
edits = [
    (
        "    if (i0<v0) (a->fused ? att512_scores_from_fma : att512_scores_from)(i0,v0,d,q,ldq,k,ldk,s,t);\n",
        "    { unsigned long long c0=__rdtsc();\n    if (i0<v0) (a->fused ? att512_scores_from_fma : att512_scores_from)(i0,v0,d,q,ldq,k,ldk,s,t);\n    __atomic_fetch_add(&bcir_ar_cycles[0],__rdtsc()-c0,__ATOMIC_RELAXED); }\n",
    ),
    (
        "  att_softmax_block(a,s,t,i0,i1-i0);\n#ifdef BCIR_MM_DISPATCH\n  if (i0<v0) (a->fused ? att512_context_from_fma : att512_context_from)(i0,v0,d,s,t,v,ldk,y,ldq);\n#endif\n",
        "  { unsigned long long c1=__rdtsc(); att_softmax_block(a,s,t,i0,i1-i0); __atomic_fetch_add(&bcir_ar_cycles[1],__rdtsc()-c1,__ATOMIC_RELAXED); }\n#ifdef BCIR_MM_DISPATCH\n  { unsigned long long c2=__rdtsc();\n  if (i0<v0) (a->fused ? att512_context_from_fma : att512_context_from)(i0,v0,d,s,t,v,ldk,y,ldq);\n  __atomic_fetch_add(&bcir_ar_cycles[2],__rdtsc()-c2,__ATOMIC_RELAXED); }\n#endif\n",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new, 1)
out.write_text(src)
print("instrumented", out)
