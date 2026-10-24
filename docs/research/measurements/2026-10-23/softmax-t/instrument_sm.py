"""SOFTMAX-T's phase instrumentation: a copy of a tensor unit (the tree's lanes path, or SOFTMAX-T's) with each 16-row
group's phases timed in TSC cycles, summed over the call and its threads -- bcir_ar_cycles [0] the queries' transpose
and the scores, [1] the softmax, [2] the context (SOFTMAX-T's divides inside it), [4] the softmax's maximum, [5] its
exponential and sum, [6] its division (the tree's form; SOFTMAX-T's has none here). ../scores-t/phase_bench_t.c reads
them. usage: instrument_sm.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])


def once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new, 1)


hdr = (
    "#include <x86intrin.h>\nunsigned long long bcir_ar_cycles[8];\n"
    "#define AR_ADD(k,c) __atomic_fetch_add(&bcir_ar_cycles[k],__rdtsc()-(c),__ATOMIC_RELAXED)\n"
)
src = once(src, "#define BCIR_ATT_TQ 256\n", hdr + "#define BCIR_ATT_TQ 256\n")
fused = "att512_tsoftmax(R,scale,s,sums);" in src
soft = "    att512_tsoftmax(R,scale,s,sums);\n" if fused else "    att512_tsoftmax(R,scale,s);\n"
ctx = (
    "    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,sums,v,ldk,y,ldq);\n"
    if fused
    else "    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,v,ldk,y,ldq);\n"
)
src = once(
    src,
    "    att512_tq(d,q+R*ldq,ldq,qt);\n    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);\n"
    + soft
    + ctx,
    "    unsigned long long c=__rdtsc();\n"
    "    att512_tq(d,q+R*ldq,ldq,qt);\n    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);\n"
    "    AR_ADD(0,c); c=__rdtsc();\n"
    + soft
    + "    AR_ADD(1,c); c=__rdtsc();\n"
    + ctx
    + "    AR_ADD(2,c);\n",
)
src = once(
    src,
    "  __m512 mx=_mm512_set1_ps(-INFINITY),s=_mm512_setzero_ps(),sc=_mm512_set1_ps(scale); size_t jb,c;\n",
    "  __m512 mx=_mm512_set1_ps(-INFINITY),s=_mm512_setzero_ps(),sc=_mm512_set1_ps(scale); size_t jb,c;\n"
    "  unsigned long long c4=__rdtsc();\n",
)
src = once(
    src,
    "  mx=_mm512_mul_ps(mx,sc);\n  for (jb=0;jb<=R;jb+=16)",
    "  mx=_mm512_mul_ps(mx,sc);\n  AR_ADD(4,c4); c4=__rdtsc();\n  for (jb=0;jb<=R;jb+=16)",
)
if fused:
    src = once(
        src,
        "  _mm512_store_ps(sums,s);  /* the division:",
        "  AR_ADD(5,c4);\n  _mm512_store_ps(sums,s);  /* the division:",
    )
else:
    src = once(
        src,
        "  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {\n"
        "    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);\n"
        "    __m512 x=_mm512_loadu_ps(sg+jb*16+c*16);\n",
        "  AR_ADD(5,c4); c4=__rdtsc();\n"
        "  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {\n"
        "    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);\n"
        "    __m512 x=_mm512_loadu_ps(sg+jb*16+c*16);\n",
    )
    src = once(
        src,
        "    _mm512_storeu_ps(sg+jb*16+c*16,_mm512_mask_div_ps(x,k,x,s));\n  }\n}\n",
        "    _mm512_storeu_ps(sg+jb*16+c*16,_mm512_mask_div_ps(x,k,x,s));\n  }\n  AR_ADD(6,c4);\n}\n",
    )
out.write_text(src)
print("instrumented", "SOFTMAX-T's form" if fused else "the tree's form", out)
