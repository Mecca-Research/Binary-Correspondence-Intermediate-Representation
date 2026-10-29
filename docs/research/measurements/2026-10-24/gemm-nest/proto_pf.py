"""GEMM-PF-0's prototype: the k-blocked AVX-512 kernel's untransposed path asks for the next 8-row block's lines of A
a depth chunk (16 terms: a line of each row) ahead, and for its own block's lines of C, with write intent, before its
chain -- hints only, every fused chain and every store as it was. usage: proto_pf.py SRC OUT [a|c|ac]"""

import sys
from pathlib import Path

t = Path(sys.argv[1]).read_text()
mode = sys.argv[3] if len(sys.argv) > 3 else "ac"


def once(old, new):
    global t
    assert t.count(old) == 1, (old[:70], t.count(old))
    t = t.replace(old, new)


pre_c = (
    """    for (r=0;r<mr;r++) {  /* GEMM-PF: this block's lines of C, with write intent, before its chain */
      const float *cr=x->c+(i+r)*x->ldc+jb;
      __builtin_prefetch(cr,1,3);
      if (nv>1) __builtin_prefetch(cr+16,1,3);
      if (nv>2) __builtin_prefetch(cr+32,1,3);
    }
"""
    if "c" in mode
    else ""
)
pre_a = (
    """      if ((l&15)==0) for (r=0;r<mr;r++) _mm_prefetch((const char *)(ap+(mr+r)*lda+l),_MM_HINT_T0);  /* GEMM-PF */
"""
    if "a" in mode
    else ""
)
once(
    """  } else {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
    for (l=0;l<nl;l++) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();""",
    """  } else {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
"""
    + pre_c
    + """    for (l=0;l<nl;l++) {
"""
    + pre_a
    + """      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();""",
)
Path(sys.argv[2]).write_text(t)
print("GEMM-PF prototype (" + mode + "):", sys.argv[2])
