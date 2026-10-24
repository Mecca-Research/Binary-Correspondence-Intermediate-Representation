"""A copy of bcir_tensor.c with each head's backward timed a phase at a time (TSC cycles, summed per
phase over the call): [0] dV = P^T dY, [1] pass 0's dP tiles, [2] pass 0's row dots, [3] pass 1's
recomputed dP tiles, [4] pass 1's score adjoints, [5] dQ = dS K, [6] dK = dS^T Q.
usage: instrument_bwd.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
hdr = "#include <x86intrin.h>\nunsigned long long bcir_bw_cycles[8];\n#define BW_ADD(k,c) __atomic_fetch_add(&bcir_bw_cycles[k],__rdtsc()-(c),__ATOMIC_RELAXED)\n"
anchor = "static void att_head_backward(const att_job *a, size_t r, size_t h, const float *vt) {\n"
assert src.count(anchor) == 1
src = src.replace(anchor, hdr + anchor, 1)
edits = [
    (  # dV
        "    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,h%(nh/nk)==0);\n",
        "    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,h%(nh/nk)==0); BW_ADD(0,c); }\n",
    ),
    (  # the tile's products: phase 1 (pass 0) or 3 (pass 1)
        "        if (!pass || jt>=BCIR_ATT_KEEP) {\n          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));\n",
        "        unsigned long long ct=__rdtsc();\n"
        "        if (!pass || jt>=BCIR_ATT_KEEP) {\n          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));\n",
    ),
    (
        "        i=0;\n#ifdef BCIR_MM_DISPATCH\n        if (!pass && variant==4) for (;i+16<=rows;i+=16)\n",
        "        BW_ADD(pass ? 3 : 1,ct); ct=__rdtsc();\n"
        "        i=0;\n#ifdef BCIR_MM_DISPATCH\n        if (!pass && variant==4) for (;i+16<=rows;i+=16)\n",
    ),
    (  # the row's dots (pass 0) or adjoints (pass 1): phase 2 or 4
        "            else pi[j]=pi[j]*(dp-dot[i])*scale;\n          }\n        }\n",
        "            else pi[j]=pi[j]*(dp-dot[i])*scale;\n          }\n        }\n        BW_ADD(pass ? 4 : 2,ct);\n",
    ),
    (  # dQ and dK
        "    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1);\n"
        "    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,h%(nh/nk)==0);\n",
        "    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1); BW_ADD(5,c); }\n"
        "    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;\n"
        "    { unsigned long long c=__rdtsc(); att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,h%(nh/nk)==0); BW_ADD(6,c); }\n",
    ),
]
for old, new in edits:
    assert src.count(old) == 1, old[:70]
    src = src.replace(old, new, 1)
out.write_text(src)
print("instrumented", out)
