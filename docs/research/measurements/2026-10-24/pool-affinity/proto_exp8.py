"""EXP8-0: a copy of the tree's unit whose banded group softmax (att512_tsoftmax) evaluates EXPW (default 8) of a
block's columns at once through exp1_512n -- each lane's operations exp1's in its order, n vectors interleaved
(EXP-LANES), the sums by lane in column order as before -- where the tree takes four. Every value is the tree's.
usage: proto_exp8.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
old_sig = (
    '__attribute__((target("avx512f"))) BCIR_MM_INLINE void exp1_512n(__m512 *v, int n) {\n'
    "  __m512 xc[4],kf[4],r[4],p[4]; int l;\n"
)
assert src.count(old_sig) == 1
src = src.replace(
    old_sig,
    '__attribute__((target("avx512f"))) BCIR_MM_INLINE void exp1_512n(__m512 *v, int n) {\n'
    "  __m512 xc[8],kf[8],r[8],p[8]; int l;\n",
)
old_sm = """  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c+=4) {  /* scaled, less the maximum, exp1, masked, summed: one pass */
    float *b=sg+jb*16+c*16; __m512 v[4]; int l;
    for (l=0;l<4;l++) v[l]=_mm512_sub_ps(_mm512_mul_ps(_mm512_loadu_ps(b+l*16),sc),mx);
    exp1_512n(v,4);
    for (l=0;l<4;l++) {  /* the columns in order */"""
new_sm = """  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c+=EXPW) {  /* scaled, less the maximum, exp1, masked, summed: one pass */
    float *b=sg+jb*16+c*16; __m512 v[EXPW]; int l;
    for (l=0;l<EXPW;l++) v[l]=_mm512_sub_ps(_mm512_mul_ps(_mm512_loadu_ps(b+l*16),sc),mx);
    exp1_512n(v,EXPW);
    for (l=0;l<EXPW;l++) {  /* the columns in order */"""
assert src.count(old_sm) == 1
src = src.replace(old_sm, new_sm)
Path(sys.argv[2]).write_text("#ifndef EXPW\n#define EXPW 8\n#endif\n" + src)
print("wrote", sys.argv[2])
