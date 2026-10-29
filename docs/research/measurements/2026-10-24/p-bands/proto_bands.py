"""P-BANDS prototype over bcir_tensor.c: under contract bit 32 (BCIR_TENSOR_ATTENTION_BANDS) a stored forward
writes each head's probabilities in 16-row bands with the rows in the lanes (band R at p + R*t, key j's sixteen
values at + j*16, row R+u in lane u) and the backward reads them so -- AVX-512, heads of at most BCIR_ATT_TQ
channels, t a multiple of 16 (the prototype's reach; elsewhere the bit is ignored). Every value is the row
path's: each product's terms in its order. usage: proto_bands.py SRC OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])


def edit(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:90])
    src = src.replace(old, new)


# the bit
edit(
    "#define BCIR_ATT_DC 128\n",
    "#ifndef BCIR_TENSOR_ATTENTION_BANDS\n#define BCIR_TENSOR_ATTENTION_BANDS 32u\n#endif\n#define BCIR_ATT_DC 128\n",
)
edit(
    "  int lower;  /* P-LOWER: its strips leave their rows' future columns to the blocks that reach them */\n",
    "  int lower;  /* P-LOWER: its strips leave their rows' future columns to the blocks that reach them */\n"
    "  int bands;  /* P-BANDS: a stored head's probabilities in 16-row bands, the rows in the lanes */\n",
)

FWD = r"""
#ifdef BCIR_MM_DISPATCH
/* P-BANDS: the group's raw scores' maxima by lane (att512_tsoftmax's masked VMAXPS, the scalar
 * `x > max ? x : max`), scaled */
__attribute__((target("avx512f"))) static void att512_tmax(size_t R, float scale, const float *sg, float *m) {
  __m512 mx=_mm512_set1_ps(-INFINITY); size_t jb,c;
  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {
    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);
    mx=_mm512_mask_max_ps(mx,k,_mm512_loadu_ps(sg+jb*16+c*16),mx);
  }
  _mm512_storeu_ps(m,_mm512_mul_ps(mx,_mm512_set1_ps(scale)));
}
/* P-BANDS: the group's softmax under libm's expf -- att_softmax_block's 16-row v1 arithmetic on the blocks as
 * they stand, each lane's chain in key order: each element expf(x*scale - max) where its lane takes the key and
 * +0.0 elsewhere, the sums by lane in key order; the division is the context's (att512_tdiv). Plain C: libm's
 * calls keep no vector live. */
static void att_tsoftmax_libm(size_t R, float scale, float *sg, float *sums) {
  float m[16]; size_t j,u;
  att512_tmax(R,scale,sg,m);
  for (u=0;u<16;u++) sums[u]=0.0f;
  for (j=0;j<R;j++) {
    float *b=sg+j*16;
    for (u=0;u<16;u++) { float x=expf(b[u]*scale-m[u]); b[u]=x; sums[u]+=x; }
  }
  for (j=R;j<R+16;j++) {
    float *b=sg+j*16;
    for (u=0;u<16;u++) if (j-R<=u) { float x=expf(b[u]*scale-m[u]); b[u]=x; sums[u]+=x; } else b[u]=0.0f;
  }
}
/* P-BANDS: a stored forward's head a 16-row group at a time in its own band of p -- SCORES-T's scores,
 * softmax and context (att_strip_lanes) with the group's slot the band itself, so the probabilities stay
 * where the groups computed them; without LOWER the band's keys past its last row are +0.0 first. */
static void att_head_bands(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),ldq=nh*d,ldk=nk*d,R;
  const float *q=a->q+(r*t*nh+h)*d,*k=a->k+(r*t*nk+kh)*d,*v=a->v+(r*t*nk+kh)*d;
  float *y=a->y+(r*t*nh+h)*d,*ph=a->p+(r*nh+h)*t*t,scale=1.0f/sqrtf((float)d);
  _Alignas(64) float qt[BCIR_ATT_TQ*16],sums[16];
  for (R=0;R<t;R+=16) {
    float *s=ph+R*t;
    if (!a->lower && R+16<t) memset(s+(R+16)*16,0,(t-R-16)*16*sizeof(float));
    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);
    if (a->exp2) att512_tsoftmax(R,scale,s,sums); else att_tsoftmax_libm(R,scale,s,sums);
    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,sums,v,ldk,y,ldq);
  }
  if (a->bad) att_verdict(a,r,h);
}
#endif
"""
edit(
    "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {\n",
    FWD
    + "static void att_strip(const att_job *a, size_t r, size_t h, size_t i0, size_t i1, float *s) {\n",
)

BWD = r"""
#ifdef BCIR_MM_DISPATCH
/* P-BANDS: att512_tr_block's arithmetic over banded probabilities -- term i's values for rows j0..j0+7 at
 * pb + (i/16)*16*t + (j0+r)*16 + i%16, every other operand and every term's order as there */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_btr_block(int f, int nv, size_t j0, size_t ib,
    size_t ie, size_t c0, size_t t, const float *pb, const float *x, size_t ldx, float *out, size_t ldo,
    int zero, __mmask16 m0, __mmask16 m1, __mmask16 m2) {
  __m512 a0[8],a1[8],a2[8]; size_t r,i,di; int diag=j0>=ib;
  for (r=0;r<8;r++) {
    const float *o=out+(j0+r)*ldo+c0;
    a0[r]=_mm512_maskz_loadu_ps(zero ? 0 : m0,o);
    if (nv>1) a1[r]=_mm512_maskz_loadu_ps(zero ? 0 : m1,o+16);
    if (nv>2) a2[r]=_mm512_maskz_loadu_ps(zero ? 0 : m2,o+32);
  }
  for (di=0;diag && di<7;di++) {  /* term j0+di: the rows r <= di */
    size_t ii=j0+di; const float *xi=x+ii*ldx+c0,*mi=pb+(ii>>4)*16*t+(ii&15)+j0*16;
    __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
    if (nv>1) b1=_mm512_maskz_loadu_ps(m1,xi+16);
    if (nv>2) b2=_mm512_maskz_loadu_ps(m2,xi+32);
    for (r=0;r<8;r++) {
      __mmask16 take=(__mmask16)(r<=di ? 0xffffu : 0u);
      __m512 mv=_mm512_set1_ps(mi[r*16]);
      a0[r]=ATT_MADD_MASK(f,a0[r],take,mv,b0);
      if (nv>1) a1[r]=ATT_MADD_MASK(f,a1[r],take,mv,b1);
      if (nv>2) a2[r]=ATT_MADD_MASK(f,a2[r],take,mv,b2);
    }
  }
  for (i=diag ? j0+7 : ib;i<ie;i++) {  /* every row of the block takes term i */
    const float *xi=x+i*ldx+c0,*mi=pb+(i>>4)*16*t+(i&15)+j0*16;
    __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
    if (nv>1) b1=_mm512_maskz_loadu_ps(m1,xi+16);
    if (nv>2) b2=_mm512_maskz_loadu_ps(m2,xi+32);
    for (r=0;r<8;r++) {
      __m512 mv=_mm512_set1_ps(mi[r*16]);
      a0[r]=ATT_MADD(f,a0[r],mv,b0);
      if (nv>1) a1[r]=ATT_MADD(f,a1[r],mv,b1);
      if (nv>2) a2[r]=ATT_MADD(f,a2[r],mv,b2);
    }
  }
  for (r=0;r<8;r++) {
    float *o=out+(j0+r)*ldo+c0;
    _mm512_mask_storeu_ps(o,m0,a0[r]);
    if (nv>1) _mm512_mask_storeu_ps(o+16,m1,a1[r]);
    if (nv>2) _mm512_mask_storeu_ps(o+32,m2,a2[r]);
  }
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_btransposed_any(int f, size_t t, size_t d,
    const float *pb, const float *x, size_t ldx, float *out, size_t ldo, int first) {
  size_t c0,j0,ib,band=t<BCIR_ATT_BAND_FROM ? t : BCIR_ATT_BAND;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (ib=0;ib<t;ib+=band) {
      size_t ie=t-ib<band ? t : ib+band;
      for (j0=0;j0<ie;j0+=8) {
        int zero=first && j0>=ib;
        if (nc>32) att512_btr_block(f,3,j0,ib,ie,c0,t,pb,x,ldx,out,ldo,zero,m0,m1,m2);
        else if (nc>16) att512_btr_block(f,2,j0,ib,ie,c0,t,pb,x,ldx,out,ldo,zero,m0,m1,m2);
        else att512_btr_block(f,1,j0,ib,ie,c0,t,pb,x,ldx,out,ldo,zero,m0,m1,m2);
      }
    }
  }
}
__attribute__((target("avx512f"))) static void att512_btransposed(size_t t, size_t d, const float *pb,
    const float *x, size_t ldx, float *out, size_t ldo, int first) {
  att512_btransposed_any(0,t,d,pb,x,ldx,out,ldo,first);
}
__attribute__((target("avx512f"))) static void att512_btransposed_fma(size_t t, size_t d, const float *pb,
    const float *x, size_t ldx, float *out, size_t ldo, int first) {
  att512_btransposed_any(1,t,d,pb,x,ldx,out,ldo,first);
}
/* P-BANDS: one 16-key block of a group's probability adjoints with the rows in the lanes, dp_uj = dy_u . v_j
 * from +0.0 over the channels in order (att512_dp_any's terms), the group's dy transposed in dyt */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_bdp_block(int f, size_t d, const float *dyt,
    const float *vb, size_t ldv, __m512 *acc) {
  size_t c,kk;
  for (kk=0;kk<16;kk++) acc[kk]=_mm512_setzero_ps();
  for (c=0;c<d;c++) {
    __m512 dv=_mm512_load_ps(dyt+c*16);
    for (kk=0;kk<16;kk++) acc[kk]=ATT_MADD(f,acc[kk],dv,_mm512_set1_ps(vb[kk*ldv+c]));
  }
}
/* P-BANDS: a 16-row group of the backward in its band -- pass 0 the adjoints' blocks (kept to kb, keys below
 * `kept`) and the rows' dots sum_j p_uj dp_uj in key order on the way (att512_dot_any's terms); pass 1 the
 * score adjoints p (dp - dot) scale in place where the lane takes the key (a block past `kept` evaluated
 * again: the same bits); then dQ of the group's rows from the band (att512_tctx_block's terms, no division) */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_bgroup_any(int f, size_t R, size_t d, float scale,
    const float *dy, size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept,
    float *dq, size_t ldq) {
  _Alignas(64) float dyt[BCIR_ATT_TQ*16]; __m512 dot=_mm512_setzero_ps(),sc=_mm512_set1_ps(scale);
  size_t jb,kk,c0,rb;
  att512_tq(d,dy+R*ldy,ldy,dyt);
  for (jb=0;jb<=R;jb+=16) {
    __m512 acc[16];
    att512_bdp_block(f,d,dyt,v+jb*ldk,ldk,acc);
    for (kk=0;kk<16;kk++) {
      __mmask16 m=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<kk)&0xffffu);
      if (jb<kept) _mm512_storeu_ps(kb+(jb+kk)*16,acc[kk]);
      dot=ATT_MADD_MASK(f,dot,m,_mm512_loadu_ps(band+(jb+kk)*16),acc[kk]);
    }
  }
  for (jb=0;jb<=R;jb+=16) {
    __m512 acc[16];
    if (jb>=kept) att512_bdp_block(f,d,dyt,v+jb*ldk,ldk,acc);
    for (kk=0;kk<16;kk++) {
      size_t j=jb+kk; __mmask16 m=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<kk)&0xffffu);
      __m512 dp=jb<kept ? _mm512_loadu_ps(kb+j*16) : acc[kk],p=_mm512_loadu_ps(band+j*16);
      _mm512_mask_storeu_ps(band+j*16,m,_mm512_mul_ps(_mm512_mul_ps(p,_mm512_sub_ps(dp,dot)),sc));
    }
  }
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (rb=0;rb<16;rb+=8) {
      if (nc>32) att512_tctx_block(f,3,R,rb,c0,band,(size_t)-1,sc,k,ldk,dq,ldq,m0,m1,m2);
      else if (nc>16) att512_tctx_block(f,2,R,rb,c0,band,(size_t)-1,sc,k,ldk,dq,ldq,m0,m1,m2);
      else att512_tctx_block(f,1,R,rb,c0,band,(size_t)-1,sc,k,ldk,dq,ldq,m0,m1,m2);
    }
  }
}
__attribute__((target("avx512f"))) static void att512_bgroup(size_t R, size_t d, float scale, const float *dy,
    size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept, float *dq, size_t ldq) {
  att512_bgroup_any(0,R,d,scale,dy,ldy,v,k,ldk,band,kb,kept,dq,ldq);
}
__attribute__((target("avx512f"))) static void att512_bgroup_fma(size_t R, size_t d, float scale, const float *dy,
    size_t ldy, const float *v, const float *k, size_t ldk, float *band, float *kb, size_t kept, float *dq, size_t ldq) {
  att512_bgroup_any(1,R,d,scale,dy,ldy,v,k,ldk,band,kb,kept,dq,ldq);
}
/* P-BANDS: one (batch, query head) of the backward over banded probabilities: dV = P^T dY, then a 16-row
 * group at a time its adjoints, dots, score adjoints and dQ, then dK = dS^T Q -- every accumulator the row
 * path's terms in the row path's order. kb: the task's room for a group's adjoint blocks (BWD-KEEP's region,
 * every key) or the caller's stack (its first `stack` floats' keys). */
static void att_head_backward_bands(const att_job *a, size_t r, size_t h, float *stack, size_t room) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),ldh=nh*d,ldk=nk*d,R,kept;
  float scale=1.0f/sqrtf((float)d),*ph=a->p+(r*nh+h)*t*t,*kb;
  const float *qh=a->q+(r*t*nh+h)*d,*dyh=a->dy+(r*t*nh+h)*d,*kk=a->k+(r*t*nk+kh)*d,*vk=a->v+(r*t*nk+kh)*d;
  float *dqh=a->dq+(r*t*nh+h)*d,*dkh=a->dk+(r*t*nk+kh)*d,*dvh=a->dv+(r*t*nk+kh)*d;
  int first=h%(nh/nk)==0 && !a->acc;
  if (a->keep) { kb=a->keep; kept=t; } else { kb=stack; kept=room/16/16*16; }
  (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,dyh,ldh,dvh,ldk,first);
  for (R=0;R<t;R+=16)
    (a->fused ? att512_bgroup_fma : att512_bgroup)(R,d,scale,dyh,ldh,vk,kk,ldk,ph+R*t,kb,kept,dqh,ldh);
  (a->fused ? att512_btransposed_fma : att512_btransposed)(t,d,ph,qh,ldh,dkh,ldk,first);
}
#endif
"""
edit(
    "static void att_head_backward(const att_job *a, size_t r, size_t h, const float *vt, float *tp) {\n",
    BWD
    + "static void att_head_backward(const att_job *a, size_t r, size_t h, const float *vt, float *tp) {\n",
)
edit(
    "  (void)vt; (void)tp;  /* the AVX-512 instance's panels; NULL elsewhere */\n",
    "  (void)vt; (void)tp;  /* the AVX-512 instance's panels; NULL elsewhere */\n"
    "#ifdef BCIR_MM_DISPATCH\n  if (a->bands) { att_head_backward_bands(a,r,h,tp,BCIR_ATT_PANEL); return; }\n#endif\n",
)
# the in-place task lends its panel as the stack's room under P-BANDS
edit(
    "    else if (a->variant==4 && d*BCIR_ATT_COLS<=BCIR_ATT_PANEL) tp=panel;\n#endif\n"
    "    for (h=first;h<first+a->group;h++) att_head_backward(a,r,h,vt,tp);\n",
    "    else if (a->variant==4 && d*BCIR_ATT_COLS<=BCIR_ATT_PANEL) tp=panel;\n"
    "    if (a->bands) { vt=NULL; tp=panel; }\n#endif\n"
    "    for (h=first;h<first+a->group;h++) att_head_backward(a,r,h,vt,tp);\n",
)
# forward dispatch
edit(
    "  if (a->scratch) att_head_scratch(a,task/a->nh,task%a->nh);\n"
    "  else if (a->strips) att_head_strips(a,task/a->nh,task%a->nh);  /* FWD-STRIP */\n",
    "  if (a->scratch) att_head_scratch(a,task/a->nh,task%a->nh);\n"
    "#ifdef BCIR_MM_DISPATCH\n  else if (a->bands) att_head_bands(a,task/a->nh,task%a->nh);  /* P-BANDS */\n#endif\n"
    "  else if (a->strips) att_head_strips(a,task/a->nh,task%a->nh);  /* FWD-STRIP */\n",
)
edit(
    "  if (a->scratch) att_head_scratch(&s,0,0);\n  else if (a->strips) att_head_strips(&s,0,0);  /* FWD-STRIP */\n",
    "  if (a->scratch) att_head_scratch(&s,0,0);\n"
    "#ifdef BCIR_MM_DISPATCH\n  else if (a->bands) att_head_bands(&s,0,0);  /* P-BANDS */\n#endif\n"
    "  else if (a->strips) att_head_strips(&s,0,0);  /* FWD-STRIP */\n",
)
# the entries: the bit where the prototype reaches
edit(
    "static void attention_any(bcir_tensor_pool *pool, int exp2, int fused, int scratch, int lower,\n"
    "    int variant, size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q, const float *k,\n"
    "    const float *v, float *p, float *y, int *bad, void *workspace, size_t room) {\n",
    "static int attention_bands_next;  /* the prototype's bit for the next attention_any (one caller) */\n"
    "static void attention_any(bcir_tensor_pool *pool, int exp2, int fused, int scratch, int lower,\n"
    "    int variant, size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q, const float *k,\n"
    "    const float *v, float *p, float *y, int *bad, void *workspace, size_t room) {\n",
)
edit(
    "  a.lower=lower && !scratch;  /* P-LOWER */\n  a.keep=NULL;\n",
    "  a.lower=lower && !scratch;  /* P-LOWER */\n  a.keep=NULL;\n"
    "  a.bands=attention_bands_next && !scratch && variant==4 && d<=BCIR_ATT_TQ && t%16==0;\n"
    "  attention_bands_next=0;\n",
)
edit(
    "  if (!iv) return 0;\n  attention_any(pool,(contract&BCIR_TENSOR_ATTENTION_EXP)!=0,fused,\n",
    "  if (!iv) return 0;\n  attention_bands_next=(contract&BCIR_TENSOR_ATTENTION_BANDS)!=0;\n"
    "  attention_any(pool,(contract&BCIR_TENSOR_ATTENTION_EXP)!=0,fused,\n",
)
edit(
    "  a.strips=0; a.lower=0; a.keep=NULL;\n",
    "  a.strips=0; a.lower=0; a.keep=NULL;\n"
    "  a.bands=(contract&BCIR_TENSOR_ATTENTION_BANDS)!=0 && iv==4 && d<=BCIR_ATT_TQ && t%16==0;\n",
)
out.write_text(src)
print("wrote", out)
