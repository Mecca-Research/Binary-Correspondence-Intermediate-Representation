"""SCORES-T prototype: a copy of bcir_tensor.c whose strip forward (INFER-ATTN's scratch strips; FWD-STRIP's stored
strips where the task has a region) runs each 16-row group with its rows in the vector lanes on the AVX-512 instance
under the exp contract: the group's queries transposed once, its scores as 16 x 16 blocks (lane = row, one accumulator
a key, each score its channels in order from +0.0), the softmax's maximum, exponential-and-sum and division on the
blocks with no register transpose, the context reading the probabilities from the blocks (a key's eight rows side by
side), and for a stored forward one transpose out to p's rows -- the future columns +0.0 where the row path stores
them. Every value the row path's. usage: proto_scorest.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])


def once(text, old, new):
    assert text.count(old) == 1, (old[:90], text.count(old))
    return text.replace(old, new, 1)


# 1. the job's room for a stored forward's group (the region's keep room)
src = once(
    src,
    "  float *keep;\n} att_job;\n",
    "  float *keep;\n"
    "  /* SCORES-T: a stored forward task's room for a 16-row group's transposed scores (its region's keep room),\n"
    "   * or NULL */\n"
    "  float *ts;\n} att_job;\n",
)
src = once(
    src,
    "  a.lower=lower && !scratch;  /* P-LOWER */\n  a.keep=NULL;\n",
    "  a.lower=lower && !scratch;  /* P-LOWER */\n  a.keep=NULL; a.ts=NULL;\n",
)
src = once(
    src,
    "  a.strips=0; a.lower=0; a.keep=NULL;\n",
    "  a.strips=0; a.lower=0; a.keep=NULL; a.ts=NULL;\n",
)
src = once(
    src,
    "  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;\n",
    "  s.b=1; s.nh=1; s.nk=1; s.group=1; s.q=qc; s.k=kc; s.v=vc; s.y=yc; s.bad=NULL; s.ws=NULL;\n"
    "  s.ts=qc+7*f;  /* SCORES-T: the region's keep room, the backward's */\n",
)

# 2. the kernels, after the backward's dP blocks
kernels = r"""
/* SCORES-T: a 16-row group of a strip (rows R..R+15, R a multiple of 16) with its rows in the lanes. The group's
 * scores are 16 x 16 blocks, block jb (keys jb..jb+15, jb <= R) at sg + jb*16, its column c (key jb+c) a vector
 * whose lane u is row R+u -- so p(R+u, j) is sg[j*16 + u], sg aligned or not -- each score its channels' products
 * in order from +0.0
 * (one rounded product and one rounded sum a term, or one fmaf under ATTN-FMA), as att512_scores_any's. */
#define BCIR_ATT_TQ 4096
__attribute__((target("avx512f"))) static void att512_tq(size_t d, const float *q, size_t ldq, float *qt) {
  size_t c,u,dw=d/16*16;
  for (c=0;c<dw;c+=16) kb512_transpose16(q+c,ldq,qt+c*16,16);
  for (c=dw;c<d;c++) for (u=0;u<16;u++) qt[c*16+u]=q[u*ldq+c];
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tscores_any(int f, size_t R, size_t d,
    const float *qt, const float *k, size_t ldk, float *sg) {
  size_t jb,c,kk;
  for (jb=0;jb<=R;jb+=16) {
    __m512 acc[16]; const float *kb=k+jb*ldk; float *blk=sg+jb*16;
    for (kk=0;kk<16;kk++) acc[kk]=_mm512_setzero_ps();
    for (c=0;c<d;c++) {
      __m512 qv=_mm512_load_ps(qt+c*16);
      for (kk=0;kk<16;kk++) acc[kk]=ATT_MADD(f,acc[kk],qv,_mm512_set1_ps(kb[kk*ldk+c]));
    }
    for (kk=0;kk<16;kk++) _mm512_storeu_ps(blk+kk*16,acc[kk]);
  }
}
__attribute__((target("avx512f"))) static void att512_tscores(size_t R, size_t d, const float *qt,
    const float *k, size_t ldk, float *sg) {
  att512_tscores_any(0,R,d,qt,k,ldk,sg);
}
__attribute__((target("avx512f"))) static void att512_tscores_fma(size_t R, size_t d, const float *qt,
    const float *k, size_t ldk, float *sg) {
  att512_tscores_any(1,R,d,qt,k,ldk,sg);
}
/* The group's softmax under the exp contract on its blocks, att_softmax_block's 16-row arithmetic without its
 * transposes: lane u takes column c of block jb where jb+c <= R+u; the raw scores' maxima, scaled; each element
 * exp1(x*scale - max), the lanes past a row's diagonal +0.0, summed in column order; then each element divided
 * by its row's sum. */
__attribute__((target("avx512f"))) static void att512_tsoftmax(size_t R, float scale, float *sg) {
  __m512 mx=_mm512_set1_ps(-INFINITY),s=_mm512_setzero_ps(),sc=_mm512_set1_ps(scale); size_t jb,c;
  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {
    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);
    mx=_mm512_mask_max_ps(mx,k,_mm512_loadu_ps(sg+jb*16+c*16),mx);
  }
  mx=_mm512_mul_ps(mx,sc);
  for (jb=0;jb<=R;jb+=16) {
    float *blk=sg+jb*16;
    for (c=0;c<16;c++) _mm512_storeu_ps(blk+c*16,_mm512_sub_ps(_mm512_mul_ps(_mm512_loadu_ps(blk+c*16),sc),mx));
    exp_inplace_avx512(256,blk);
    for (c=0;c<16;c++) {
      __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_maskz_mov_ps(k,_mm512_loadu_ps(blk+c*16));
      s=_mm512_mask_add_ps(s,k,s,x);
      _mm512_storeu_ps(blk+c*16,x);
    }
  }
  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {
    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);
    __m512 x=_mm512_loadu_ps(sg+jb*16+c*16);
    _mm512_storeu_ps(sg+jb*16+c*16,_mm512_mask_div_ps(x,k,x,s));
  }
}
/* One 8-row block of the group's context (rows R+rb.., rb 0 or 8) from its blocks: att512_ctx_block's terms in
 * its order from +0.0, the eight rows' probabilities of key j side by side at sg + j*16 + rb. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tctx_block(int f, int nv, size_t R, size_t rb,
    size_t c0, const float *sg, const float *v, size_t ldv, float *y, size_t ldy, __mmask16 m0, __mmask16 m1,
    __mmask16 m2) {
  __m512 a0[8],a1[8],a2[8]; size_t r,j,dj,i0=R+rb;
  for (r=0;r<8;r++) { a0[r]=_mm512_setzero_ps(); a1[r]=_mm512_setzero_ps(); a2[r]=_mm512_setzero_ps(); }
  for (j=0;j<=i0;j++) {  /* every row of the block takes key j */
    const float *vj=v+j*ldv+c0,*pj=sg+j*16+rb;
    __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
    if (nv>1) b1=_mm512_maskz_loadu_ps(m1,vj+16);
    if (nv>2) b2=_mm512_maskz_loadu_ps(m2,vj+32);
    for (r=0;r<8;r++) {
      __m512 pv=_mm512_set1_ps(pj[r]);
      a0[r]=ATT_MADD(f,a0[r],pv,b0);
      if (nv>1) a1[r]=ATT_MADD(f,a1[r],pv,b1);
      if (nv>2) a2[r]=ATT_MADD(f,a2[r],pv,b2);
    }
  }
  for (dj=1;dj<8;dj++) {  /* key i0+dj: the rows r >= dj, the others masked out */
    const float *vj=v+(i0+dj)*ldv+c0,*pj=sg+(i0+dj)*16+rb;
    __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
    if (nv>1) b1=_mm512_maskz_loadu_ps(m1,vj+16);
    if (nv>2) b2=_mm512_maskz_loadu_ps(m2,vj+32);
    for (r=0;r<8;r++) {
      __mmask16 take=(__mmask16)(r>=dj ? 0xffffu : 0u);
      __m512 pv=_mm512_set1_ps(pj[r]);
      a0[r]=ATT_MADD_MASK(f,a0[r],take,pv,b0);
      if (nv>1) a1[r]=ATT_MADD_MASK(f,a1[r],take,pv,b1);
      if (nv>2) a2[r]=ATT_MADD_MASK(f,a2[r],take,pv,b2);
    }
  }
  for (r=0;r<8;r++) {
    float *yr=y+(i0+r)*ldy+c0;
    _mm512_mask_storeu_ps(yr,m0,a0[r]);
    if (nv>1) _mm512_mask_storeu_ps(yr+16,m1,a1[r]);
    if (nv>2) _mm512_mask_storeu_ps(yr+32,m2,a2[r]);
  }
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tcontext_any(int f, size_t R, size_t d,
    const float *sg, const float *v, size_t ldv, float *y, size_t ldy) {
  size_t c0,rb;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (rb=0;rb<16;rb+=8) {
      if (nc>32) att512_tctx_block(f,3,R,rb,c0,sg,v,ldv,y,ldy,m0,m1,m2);
      else if (nc>16) att512_tctx_block(f,2,R,rb,c0,sg,v,ldv,y,ldy,m0,m1,m2);
      else att512_tctx_block(f,1,R,rb,c0,sg,v,ldv,y,ldy,m0,m1,m2);
    }
  }
}
__attribute__((target("avx512f"))) static void att512_tcontext(size_t R, size_t d, const float *sg,
    const float *v, size_t ldv, float *y, size_t ldy) {
  att512_tcontext_any(0,R,d,sg,v,ldv,y,ldy);
}
__attribute__((target("avx512f"))) static void att512_tcontext_fma(size_t R, size_t d, const float *sg,
    const float *v, size_t ldv, float *y, size_t ldy) {
  att512_tcontext_any(1,R,d,sg,v,ldv,y,ldy);
}
/* A stored forward's group out to p's rows R..R+15 (p at row R, ldp apart): each block to the diagonal one
 * transposed (the diagonal's future lanes the +0.0 the softmax left), and in the first half of a 32-key block
 * the next 16 keys' columns, to the strip's end, +0.0 -- the columns the row path's score blocks store. */
__attribute__((target("avx512f"))) static void att512_tout(size_t R, size_t end, const float *sg, float *p,
    size_t ldp) {
  size_t jb,u;
  for (jb=0;jb<=R;jb+=16) kb512_transpose16(sg+jb*16,16,p+jb,ldp);
  if (R%32==0 && R+16<end) {
    __mmask16 m=att512_lanes(end-(R+16));
    for (u=0;u<16;u++) _mm512_mask_storeu_ps(p+u*ldp+R+16,m,_mm512_setzero_ps());
  }
}
#endif
enum { ATT_SCORES, ATT_CONTEXT, ATT_TRANSPOSED };
"""
src = once(src, "#endif\nenum { ATT_SCORES, ATT_CONTEXT, ATT_TRANSPOSED };\n", kernels)

# 3. the strip: its groups rows-in-lanes where the instance, the contract and the room allow
strip_t = r"""/* SCORES-T: a strip's rows [i0, i1) as 16-row groups with their rows in the lanes -- the AVX-512 instance under
 * the exp contract, i1 a multiple of 16 (the strip's rows whole groups, none the v1 kernel's), sg the group's room
 * (a scratch forward's slot; a stored forward's region): per group its queries transposed, its scores, its
 * softmax and its context, and a stored forward's probabilities out to p's rows. Every value the row path's. */
static void att_strip_t(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s, float *sg) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),ldq=nh*d,ldk=nk*d,R;
  const float *q=a->q+(r*t*nh+h)*d,*k=a->k+(r*t*nk+kh)*d,*v=a->v+(r*t*nk+kh)*d;
  float *y=a->y+(r*t*nh+h)*d,scale=1.0f/sqrtf((float)d);
  _Alignas(64) float qt[BCIR_ATT_TQ];
  for (R=i0;R<i1;R+=16) {
    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,sg);
    att512_tsoftmax(R,scale,sg);
    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,sg,v,ldk,y,ldq);
    if (!a->scratch) att512_tout(R,i1,sg,s+(R-i0)*t,t);
  }
}
#endif
static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {
"""
src = once(
    src,
    "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {\n",
    "#ifdef BCIR_MM_DISPATCH\n" + strip_t,
)
src = once(
    src,
    "  float *y=a->y+(r*t*nh+h)*d; int family=a->fused ? MM_FUSED : MM_V1;\n  mm_args x; mm_range all;\n",
    "  float *y=a->y+(r*t*nh+h)*d; int family=a->fused ? MM_FUSED : MM_V1;\n  mm_args x; mm_range all;\n"
    "#ifdef BCIR_MM_DISPATCH\n"
    "  if (a->variant==4 && a->exp2 && d*16<=BCIR_ATT_TQ && i1%16==0 && (a->scratch || a->ts)) {  /* SCORES-T */\n"
    "    att_strip_t(a,r,h,i0,i1,s,a->scratch ? s : a->ts);\n"
    "    return;\n"
    "  }\n"
    "#endif\n",
)
out.write_text(src)
print("prototype", out)
