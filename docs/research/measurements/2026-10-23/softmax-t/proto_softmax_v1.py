"""SOFTMAX-T-0's prototypes: the tree's bcir_tensor.c edited three ways, cumulatively.

level 1 (EXP-SCALEF): the AVX-512 instance's exponential and v2 sigmoid as 16-lane intrinsics,
    2^k applied by one vscalefps (exp1's two power-of-two scalings round once, as vscalefps
    rounds once), every other operation exp1's / sigmoid2's in their order
level 2: + the lanes softmax's scale, exponential, mask and sum in one pass over each block
level 3: + the normalising division inside the context's first pass (each key's quotients
    computed once, before the context's first read of them, with the division's own masks)

usage: proto_softmax.py SRC_bcir_tensor.c OUT.c LEVEL"""

import sys
from pathlib import Path

src, out, level = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
t = src.read_text()


def once(old, new):
    global t
    assert t.count(old) == 1, (old[:70], t.count(old))
    t = t.replace(old, new)


# ---- level 1: the vector exponential and sigmoid -------------------------------------------
once(
    '__attribute__((target("avx512f"))) static void exp_avx512(size_t n, const float *x, float *y) { exp_body(64,n,x,y); }\n',
    """/* EXP-SCALEF: exp1 on AVX-512, sixteen lanes, 2^k applied by one vscalefps of p by kf. exp1's two
 * power-of-two scalings round once -- the first is exact, |k1| <= 76 keeping p * 2^k1 normal -- and
 * vscalefps rounds p * 2^kf once, so each lane is exp1's; the clamp, the reduction and the polynomial
 * are exp1's operations in its order, a multiply then an add each, the special cases selected last. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512 exp1_512(__m512 x) {
  __m512 xc=_mm512_min_ps(_mm512_max_ps(x,_mm512_set1_ps(-104.0f)),_mm512_set1_ps(89.0f));
  __m512 kf=_mm512_roundscale_ps(_mm512_mul_ps(xc,_mm512_set1_ps(BCIR_EXP_LOG2E)),
      _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  __m512 r=_mm512_sub_ps(_mm512_sub_ps(xc,_mm512_mul_ps(kf,_mm512_set1_ps(BCIR_EXP_LN2_HI))),
      _mm512_mul_ps(kf,_mm512_set1_ps(BCIR_EXP_LN2_LO)));
  __m512 p=_mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(1.0f/5040.0f),r),_mm512_set1_ps(1.0f/720.0f)),y;
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f/120.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f/24.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f/6.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(0.5f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f));
  y=_mm512_scalef_ps(p,kf);
  y=_mm512_mask_mov_ps(y,_mm512_cmp_ps_mask(x,_mm512_set1_ps(88.72283172607421875f),_CMP_GT_OQ),
      _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000)));
  y=_mm512_mask_mov_ps(y,_mm512_cmp_ps_mask(x,_mm512_set1_ps(-103.972084045410156f),_CMP_LT_OQ),
      _mm512_setzero_ps());
  return _mm512_mask_mov_ps(x,_mm512_cmp_ps_mask(x,x,_CMP_EQ_OQ),y);
}
/* sigmoid2 on AVX-512: exp1_512 of (x < 0 ? x : -x), then sigmoid2's operations */
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512 sigmoid2_512(__m512 x) {
  __m512 z=_mm512_setzero_ps(),one=_mm512_set1_ps(1.0f),e,s;
  __m512 nx=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x),_mm512_set1_epi32(INT32_MIN)));
  e=exp1_512(_mm512_mask_mov_ps(nx,_mm512_cmp_ps_mask(x,z,_CMP_LT_OQ),x));
  s=_mm512_div_ps(one,_mm512_add_ps(one,e));
  return _mm512_mask_mov_ps(_mm512_mul_ps(e,s),_mm512_cmp_ps_mask(x,z,_CMP_GE_OQ),s);
}
BCIR_MM_INLINE __mmask16 ew_tail(size_t left) { return (__mmask16)((1u<<left)-1u); }
/* an output its input or apart from it: the element loop's values whatever the order */
BCIR_MM_INLINE int ew_safe(const float *out, const float *in, size_t n) { return out==in || ew_apart(out,in,n); }
__attribute__((target("avx512f"))) static void exp_avx512(size_t n, const float *x, float *y) {
  size_t i=0;
  if (ew_safe(y,x,n)) {
    for (;i+16<=n;i+=16) _mm512_storeu_ps(y+i,exp1_512(_mm512_loadu_ps(x+i)));
    if (i<n) { __mmask16 m=ew_tail(n-i); _mm512_mask_storeu_ps(y+i,m,exp1_512(_mm512_maskz_loadu_ps(m,x+i))); i=n; }
  }
  for (;i<n;i++) y[i]=exp1(x[i]);
}
""",
)
once(
    '__attribute__((target("avx512f"))) static void exp_inplace_avx512(size_t n, float *x) { exp_inplace_body(64,n,x); }\n',
    """__attribute__((target("avx512f"))) static void exp_inplace_avx512(size_t n, float *x) {
  size_t i=0;
  for (;i+16<=n;i+=16) _mm512_storeu_ps(x+i,exp1_512(_mm512_loadu_ps(x+i)));
  if (i<n) { __mmask16 m=ew_tail(n-i); _mm512_mask_storeu_ps(x+i,m,exp1_512(_mm512_maskz_loadu_ps(m,x+i))); }
}
""",
)
once(
    '__attribute__((target("avx512f"))) static void silu2_avx512(size_t n, const float *x, float *y) { silu2_body(n,x,y); }\n',
    """__attribute__((target("avx512f"))) static void silu2_avx512(size_t n, const float *x, float *y) {
  size_t i=0;
  if (ew_safe(y,x,n)) {
    for (;i+16<=n;i+=16) { __m512 a=_mm512_loadu_ps(x+i); _mm512_storeu_ps(y+i,_mm512_mul_ps(a,sigmoid2_512(a))); }
    if (i<n) {
      __mmask16 m=ew_tail(n-i); __m512 a=_mm512_maskz_loadu_ps(m,x+i);
      _mm512_mask_storeu_ps(y+i,m,_mm512_mul_ps(a,sigmoid2_512(a))); i=n;
    }
  }
  for (;i<n;i++) y[i]=x[i]*sigmoid2(x[i]);
}
""",
)
once(
    """__attribute__((target("avx512f"))) static void silu2mul_avx512(size_t n, const float *x, const float *u,
    float *y) { silu2mul_body(n,x,u,y); }
""",
    """__attribute__((target("avx512f"))) static void silu2mul_avx512(size_t n, const float *x, const float *u,
    float *y) {
  size_t i=0;
  if (ew_safe(y,x,n) && ew_safe(y,u,n)) {
    for (;i+16<=n;i+=16) {
      __m512 a=_mm512_loadu_ps(x+i);
      _mm512_storeu_ps(y+i,_mm512_mul_ps(_mm512_mul_ps(a,sigmoid2_512(a)),_mm512_loadu_ps(u+i)));
    }
    if (i<n) {
      __mmask16 m=ew_tail(n-i); __m512 a=_mm512_maskz_loadu_ps(m,x+i);
      _mm512_mask_storeu_ps(y+i,m,_mm512_mul_ps(_mm512_mul_ps(a,sigmoid2_512(a)),_mm512_maskz_loadu_ps(m,u+i))); i=n;
    }
  }
  for (;i<n;i++) { float s=x[i]*sigmoid2(x[i]); y[i]=s*u[i]; }
}
""",
)
once(
    """__attribute__((target("avx512f"))) static void swiglu2_avx512(size_t n, const float *g, const float *u,
    const float *dy, float *dg, float *du) { swiglu2_body(n,g,u,dy,dg,du); }
""",
    """__attribute__((target("avx512f"))) BCIR_MM_INLINE void swiglu2_512(__m512 g, __m512 u, __m512 dy, __m512 *dg,
    __m512 *du) {  /* dg = dy*u*(s+g*s*(1-s)), du = dy*g*s, in swiglu2_body's order */
  __m512 s=sigmoid2_512(g),one=_mm512_set1_ps(1.0f);
  *dg=_mm512_mul_ps(_mm512_mul_ps(dy,u),_mm512_add_ps(s,_mm512_mul_ps(_mm512_mul_ps(g,s),_mm512_sub_ps(one,s))));
  *du=_mm512_mul_ps(_mm512_mul_ps(dy,g),s);
}
__attribute__((target("avx512f"))) static void swiglu2_avx512(size_t n, const float *g, const float *u,
    const float *dy, float *dg, float *du) {
  size_t i=0;
  if (ew_safe(dg,g,n) && ew_safe(dg,u,n) && ew_safe(dg,dy,n) && ew_safe(du,g,n) && ew_safe(du,u,n) &&
      ew_safe(du,dy,n) && ew_safe(dg,du,n)) {
    for (;i+16<=n;i+=16) {
      __m512 a,b;
      swiglu2_512(_mm512_loadu_ps(g+i),_mm512_loadu_ps(u+i),_mm512_loadu_ps(dy+i),&a,&b);
      _mm512_storeu_ps(dg+i,a); _mm512_storeu_ps(du+i,b);
    }
    if (i<n) {
      __mmask16 m=ew_tail(n-i); __m512 a,b;
      swiglu2_512(_mm512_maskz_loadu_ps(m,g+i),_mm512_maskz_loadu_ps(m,u+i),_mm512_maskz_loadu_ps(m,dy+i),&a,&b);
      _mm512_mask_storeu_ps(dg+i,m,a); _mm512_mask_storeu_ps(du+i,m,b); i=n;
    }
  }
  for (;i<n;i++) {
    float s=sigmoid2(g[i]);
    dg[i]=dy[i]*u[i]*(s+g[i]*s*(1.0f-s)); du[i]=dy[i]*g[i]*s;
  }
}
""",
)

# ---- level 2: the lanes softmax's exponential pass in one -----------------------------------
if level >= 2:
    once(
        """  for (jb=0;jb<=R;jb+=16) {
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
""",
        """  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {  /* scaled, less the maximum, exp1, masked, summed: one pass */
    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);
    float *b=sg+jb*16+c*16;
    __m512 x=_mm512_maskz_mov_ps(k,exp1_512(_mm512_sub_ps(_mm512_mul_ps(_mm512_loadu_ps(b),sc),mx)));
    s=_mm512_mask_add_ps(s,k,s,x);
    _mm512_storeu_ps(b,x);
  }
""",
    )
    once(  # the stored forward's transposed blocks (att512_expsum): the same one pass
        """    kb512_transpose16(rows+jb,ld,blk,16);
    for (c=0;c<16;c++) _mm512_store_ps(blk+c*16,_mm512_sub_ps(_mm512_mul_ps(_mm512_load_ps(blk+c*16),sc),mx));
    exp_inplace_avx512(256,blk);
    for (c=0;c<16;c++) {
      __mmask16 k=(__mmask16)(jb<i0 ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_maskz_mov_ps(k,_mm512_load_ps(blk+c*16));
      s=_mm512_mask_add_ps(s,k,s,x);
      _mm512_store_ps(blk+c*16,x);
    }
    kb512_transpose16(blk,16,rows+jb,ld);
""",
        """    kb512_transpose16(rows+jb,ld,blk,16);
    for (c=0;c<16;c++) {  /* scaled, less the maximum, exp1, masked, summed: one pass */
      __mmask16 k=(__mmask16)(jb<i0 ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_maskz_mov_ps(k,exp1_512(_mm512_sub_ps(_mm512_mul_ps(_mm512_load_ps(blk+c*16),sc),mx)));
      s=_mm512_mask_add_ps(s,k,s,x);
      _mm512_store_ps(blk+c*16,x);
    }
    kb512_transpose16(blk,16,rows+jb,ld);
""",
    )

# ---- level 3: the division inside the context -----------------------------------------------
if level >= 3:
    once(
        '__attribute__((target("avx512f"))) static void att512_tsoftmax(size_t R, float scale, float *sg) {\n',
        '__attribute__((target("avx512f"))) static void att512_tsoftmax(size_t R, float scale, float *sg, float *sums) {\n',
    )
    once(
        """  for (jb=0;jb<=R;jb+=16) for (c=0;c<16;c++) {
    __mmask16 k=(__mmask16)(jb<R ? 0xffffu : (0xffffu<<c)&0xffffu);
    __m512 x=_mm512_loadu_ps(sg+jb*16+c*16);
    _mm512_storeu_ps(sg+jb*16+c*16,_mm512_mask_div_ps(x,k,x,s));
  }
}
""",
        """  _mm512_store_ps(sums,s);  /* the division: inside the context's first pass (att512_tdiv) */
}
/* CTX-DIV: key j's sixteen probabilities divided by their rows' sums, the lanes the softmax's division
 * takes (lane u where j <= R+u), the others as they stand -- once, before the context's first read */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tdiv(size_t R, size_t j, float *sg, __m512 sm) {
  __mmask16 k=(__mmask16)(j<R ? 0xffffu : (0xffffu<<(j-R))&0xffffu);
  __m512 x=_mm512_loadu_ps(sg+j*16);
  _mm512_storeu_ps(sg+j*16,_mm512_mask_div_ps(x,k,x,sm));
}
""",
    )
    once(
        """__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tctx_block(int f, int nv, size_t R, size_t rb,
    size_t c0, const float *sg, const float *v, size_t ldv, float *y, size_t ldy, __mmask16 m0,
    __mmask16 m1, __mmask16 m2) {
  __m512 a0[8],a1[8],a2[8]; size_t r,j,dj,i0=R+rb;
  for (r=0;r<8;r++) { a0[r]=_mm512_setzero_ps(); a1[r]=_mm512_setzero_ps(); a2[r]=_mm512_setzero_ps(); }
  for (j=0;j<=i0;j++) {  /* every row of the block takes key j */
    const float *vj=v+j*ldv+c0,*pj=sg+j*16+rb;
""",
        """__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tctx_block(int f, int nv, size_t R, size_t rb,
    size_t c0, float *sg, size_t dlo, __m512 sm, const float *v, size_t ldv, float *y, size_t ldy,
    __mmask16 m0, __mmask16 m1, __mmask16 m2) {
  __m512 a0[8],a1[8],a2[8]; size_t r,j,dj,i0=R+rb;
  for (r=0;r<8;r++) { a0[r]=_mm512_setzero_ps(); a1[r]=_mm512_setzero_ps(); a2[r]=_mm512_setzero_ps(); }
  for (j=0;j<=i0;j++) {  /* every row of the block takes key j */
    const float *vj=v+j*ldv+c0,*pj=sg+j*16+rb;
    if (j>=dlo) att512_tdiv(R,j,sg,sm);
""",
    )
    once(
        """  for (dj=1;dj<8;dj++) {  /* key i0+dj: the rows r >= dj, the others masked out */
    const float *vj=v+(i0+dj)*ldv+c0,*pj=sg+(i0+dj)*16+rb;
""",
        """  for (dj=1;dj<8;dj++) {  /* key i0+dj: the rows r >= dj, the others masked out */
    const float *vj=v+(i0+dj)*ldv+c0,*pj=sg+(i0+dj)*16+rb;
    if (i0+dj>=dlo) att512_tdiv(R,i0+dj,sg,sm);
""",
    )
    once(
        """__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tcontext_any(int f, size_t R, size_t d,
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
""",
        """__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tcontext_any(int f, size_t R, size_t d,
    float *sg, const float *sums, const float *v, size_t ldv, float *y, size_t ldy) {
  size_t c0,rb; __m512 sm=_mm512_load_ps(sums);
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (rb=0;rb<16;rb+=8) {
      /* the first pass divides: rows 0-7 reach keys to R+7, rows 8-15 the eight after */
      size_t dlo=c0 ? (size_t)-1 : rb ? R+8 : 0;
      if (nc>32) att512_tctx_block(f,3,R,rb,c0,sg,dlo,sm,v,ldv,y,ldy,m0,m1,m2);
      else if (nc>16) att512_tctx_block(f,2,R,rb,c0,sg,dlo,sm,v,ldv,y,ldy,m0,m1,m2);
      else att512_tctx_block(f,1,R,rb,c0,sg,dlo,sm,v,ldv,y,ldy,m0,m1,m2);
    }
  }
}
__attribute__((target("avx512f"))) static void att512_tcontext(size_t R, size_t d, float *sg, const float *sums,
    const float *v, size_t ldv, float *y, size_t ldy) {
  att512_tcontext_any(0,R,d,sg,sums,v,ldv,y,ldy);
}
__attribute__((target("avx512f"))) static void att512_tcontext_fma(size_t R, size_t d, float *sg, const float *sums,
    const float *v, size_t ldv, float *y, size_t ldy) {
  att512_tcontext_any(1,R,d,sg,sums,v,ldv,y,ldy);
}
""",
    )
    once(
        """  _Alignas(64) float qt[BCIR_ATT_TQ*16];
  for (R=i0;R<i1;R+=16) {
    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);
    att512_tsoftmax(R,scale,s);
    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,v,ldk,y,ldq);
  }
""",
        """  _Alignas(64) float qt[BCIR_ATT_TQ*16],sums[16];
  for (R=i0;R<i1;R+=16) {
    att512_tq(d,q+R*ldq,ldq,qt);
    (a->fused ? att512_tscores_fma : att512_tscores)(R,d,qt,k,ldk,s);
    att512_tsoftmax(R,scale,s,sums);
    (a->fused ? att512_tcontext_fma : att512_tcontext)(R,d,s,sums,v,ldk,y,ldq);
  }
""",
    )
out.write_text(t)
print(f"level {level}: {out}")
