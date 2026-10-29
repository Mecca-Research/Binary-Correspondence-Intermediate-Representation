"""Prototypes of the banded stored forward's band writes (FWD-BANDW-0), each a copy of the tree's unit:
  pfw -- the context's passes prefetch the next group's band lines for writing (PREFETCHW), one a key;
  nt  -- the group computed in a thread's scratch slot and each final line streamed to the band
         (non-temporal stores) in the context's last pass over it, sfence at the head's end.
Every value is the tree's: the same kernels in the same order; only where the group's lines sit.
usage: proto_fwd.py SRC OUT_PFW OUT_NT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()

# a copy of att512_tctx_block / att512_tcontext_any with a prefetch base and a stream destination
blk_start = src.index(
    '__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tctx_block(int f, int nv,'
)
ctx_any = src.index(
    '__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tcontext_any(int f,'
)
ctx_end = src.index('__attribute__((target("avx512f"))) static void att512_tcontext(size_t R,')
blk = src[blk_start:ctx_any]
anyf = src[ctx_any:ctx_end]

blk2 = blk.replace(
    "att512_tctx_block(int f, int nv, size_t R, size_t rb,\n    size_t c0, float *sg, size_t dlo, __m512 sm, const float *v, size_t ldv, float *y, size_t ldy,\n    __mmask16 m0, __mmask16 m1, __mmask16 m2) {",
    "att512_tctx_block_x(int f, int nv, size_t R, size_t rb,\n    size_t c0, float *sg, size_t dlo, __m512 sm, const float *v, size_t ldv, float *y, size_t ldy,\n    __mmask16 m0, __mmask16 m1, __mmask16 m2, const float *pf, size_t pf0, float *out) {",
)
assert blk2 != blk, "block signature"
old_j = "    if (j>=dlo) att512_tdiv(R,j,sg,sm);\n    __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();\n"
assert blk2.count(old_j) == 1
blk2 = blk2.replace(
    old_j, old_j + "    if (pf) _mm_prefetch((const char *)(pf+(pf0+j)*16),_MM_HINT_ET0);\n"
)
old_r = "    for (r=0;r<8;r++) {\n      __m512 pv=_mm512_set1_ps(pj[r]);\n      a0[r]=ATT_MADD(f,a0[r],pv,b0);\n      if (nv>1) a1[r]=ATT_MADD(f,a1[r],pv,b1);\n      if (nv>2) a2[r]=ATT_MADD(f,a2[r],pv,b2);\n    }\n  }\n"
assert blk2.count(old_r) == 1
blk2 = blk2.replace(
    old_r, old_r[:-4] + "    if (out) _mm512_stream_ps(out+j*16,_mm512_load_ps(sg+j*16));\n  }\n"
)
old_d = "    if (i0+dj>=dlo) att512_tdiv(R,i0+dj,sg,sm);\n"
assert blk2.count(old_d) == 1
blk2 = blk2.replace(
    old_d, old_d + "    if (pf) _mm_prefetch((const char *)(pf+(pf0+i0+dj)*16),_MM_HINT_ET0);\n"
)
old_dr = "      if (nv>2) a2[r]=ATT_MADD_MASK(f,a2[r],take,pv,b2);\n    }\n  }\n"
assert blk2.count(old_dr) == 1
blk2 = blk2.replace(
    old_dr,
    "      if (nv>2) a2[r]=ATT_MADD_MASK(f,a2[r],take,pv,b2);\n    }\n    if (out) _mm512_stream_ps(out+(i0+dj)*16,_mm512_load_ps(sg+(i0+dj)*16));\n  }\n",
)

any2 = anyf.replace(
    "att512_tcontext_any(int f, size_t R, size_t d,\n    float *sg, const float *sums, const float *v, size_t ldv, float *y, size_t ldy) {",
    "att512_tcontext_x(int f, size_t R, size_t d,\n    float *sg, const float *sums, const float *v, size_t ldv, float *y, size_t ldy, const float *pf, float *out) {",
)
assert any2 != anyf, "context signature"
for nv in ("3", "2", "1"):
    o = "att512_tctx_block(f,%s,R,rb,c0,sg,dlo,sm,v,ldv,y,ldy,m0,m1,m2);" % nv
    assert any2.count(o) == 1, o
    any2 = any2.replace(
        o,
        "att512_tctx_block_x(f,%s,R,rb,c0,sg,dlo,sm,v,ldv,y,ldy,m0,m1,m2,c0==0 ? pf : NULL,rb ? R+8 : 0,c0+48>=d && rb ? out : NULL);"
        % nv,
    )
# the rb==8 pass of the last c0 streams; the first pass prefetches keys j (next slot's lines 0..R+7), the second
# keys R+8+j (lines R+8..R+31 of the next group's R+32)
wrappers = (
    '__attribute__((target("avx512f"))) static void att512_tcontext_xv(size_t R, size_t d, float *sg, const float *sums,\n'
    "    const float *v, size_t ldv, float *y, size_t ldy, const float *pf, float *out) {\n"
    "  att512_tcontext_x(0,R,d,sg,sums,v,ldv,y,ldy,pf,out);\n}\n"
    '__attribute__((target("avx512f"))) static void att512_tcontext_xf(size_t R, size_t d, float *sg, const float *sums,\n'
    "    const float *v, size_t ldv, float *y, size_t ldy, const float *pf, float *out) {\n"
    "  att512_tcontext_x(1,R,d,sg,sums,v,ldv,y,ldy,pf,out);\n}\n"
)
extra = blk2 + any2 + wrappers

head_old = """    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);
    if (a->exp2) att512_tsoftmax(R,scale,s,sums);
    else att_tsoftmax_libm(R,scale,s,sums);
    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,sums,v,ldk,y,ldq);
  }
}
"""
assert src.count(head_old) == 1
pfw_head = """    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);
    if (a->exp2) att512_tsoftmax(R,scale,s,sums);
    else att_tsoftmax_libm(R,scale,s,sums);
    (a->fused ? att512_tcontext_xf : att512_tcontext_xv)(R,d,s,sums,v,ldk,y,ldq,R+16<t ? s+16*t : NULL,NULL);
  }
}
"""
nt_head = """    float *g=bandw_scr;
    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,g);
    if (a->exp2) att512_tsoftmax(R,scale,g,sums);
    else att_tsoftmax_libm(R,scale,g,sums);
    (a->fused ? att512_tcontext_xf : att512_tcontext_xv)(R,d,g,sums,v,ldk,y,ldq,NULL,s);
  }
  _mm_sfence();
}
"""
hdr = "static void att_head_bands512(const att_job *a, size_t r, size_t h) {\n"
assert src.count(hdr) == 1
pfw = src.replace(head_old, pfw_head).replace(hdr, extra + hdr)
nt = src.replace(head_old, nt_head).replace(
    hdr, extra + "static __thread _Alignas(64) float bandw_scr[4096*16];\n" + hdr
)
Path(sys.argv[2]).write_text(pfw)
Path(sys.argv[3]).write_text(nt)
print("wrote", sys.argv[2], sys.argv[3])
