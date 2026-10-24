"""A copy of bcir_tensor.c with the banded stored forward (att_head_bands512) timed a phase at a time: TSC cycles
summed per phase over the call and its threads, thread-local counters flushed once a head: [0] the queries'
transpose (att512_tq), [1] the scores, [2] the softmax (max, exp, sums), [3] the context (with its division),
[4] the band's zero fill past the group (without LOWER), [5] the head whole. usage: instrument_fwd.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
hdr = (
    "#include <x86intrin.h>\nunsigned long long bcir_fw_cycles[16];\n"
    "static __thread unsigned long long fw_local[16];\n"
    "#define FW_ADD(k,c) (fw_local[k]+=__rdtsc()-(c))\n"
    "static void fw_flush(void) {\n  int k;\n  for (k=0;k<16;k++) if (fw_local[k]) {\n"
    "    __atomic_fetch_add(&bcir_fw_cycles[k],fw_local[k],__ATOMIC_RELAXED); fw_local[k]=0;\n  }\n}\n"
)
anchor = "static void att_head_bands512(const att_job *a, size_t r, size_t h) {\n"
assert src.count(anchor) == 1
old = (
    "  for (R=0;R<t;R+=16) {\n"
    "    float *s=ph+R*t;\n"
    "    if (!a->lower && R+16<t) memset(s+(R+16)*16,0,(t-R-16)*16*sizeof(float));\n"
    "    att512_tq(d,q+R*ldq,ldq,qt);\n"
    "    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);\n"
    "    if (a->exp2) att512_tsoftmax(R,scale,s,sums);\n"
    "    else att_tsoftmax_libm(R,scale,s,sums);\n"
    "    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,sums,v,ldk,y,ldq);\n"
    "  }\n"
    "}\n"
)
new = (
    "  unsigned long long c5=__rdtsc();\n"
    "  for (R=0;R<t;R+=16) {\n"
    "    float *s=ph+R*t; unsigned long long c=__rdtsc();\n"
    "    if (!a->lower && R+16<t) memset(s+(R+16)*16,0,(t-R-16)*16*sizeof(float));\n"
    "    FW_ADD(4,c); c=__rdtsc();\n"
    "    att512_tq(d,q+R*ldq,ldq,qt);\n"
    "    FW_ADD(0,c); c=__rdtsc();\n"
    "    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);\n"
    "    FW_ADD(1,c); c=__rdtsc();\n"
    "    if (a->exp2) att512_tsoftmax(R,scale,s,sums);\n"
    "    else att_tsoftmax_libm(R,scale,s,sums);\n"
    "    FW_ADD(2,c); c=__rdtsc();\n"
    "    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,sums,v,ldk,y,ldq);\n"
    "    FW_ADD(3,c);\n"
    "  }\n"
    "  FW_ADD(5,c5); fw_flush();\n"
    "}\n"
)
i = src.index(anchor)
assert src.count(old) == 1, "the banded forward's loop"
assert src.index(old) > i
src = src.replace(anchor, hdr + anchor, 1).replace(old, new, 1)
out.write_text(src)
print("instrumented", out)
