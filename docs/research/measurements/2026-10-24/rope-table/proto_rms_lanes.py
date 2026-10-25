"""RMS-LANES-0: the tensor unit's RMSNorm jobs with their per-row passes in AVX-512 -- the forward's scale
and scan, the VJP's dx, the gamma adjoint's columns -- every operation the scalar loop's, in its order.
usage: proto_rms_lanes.py SRC/bcir_tensor.c DST/bcir_tensor.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


helpers = r"""
/* RMS-LANES-0: a 16-row chunk's forward -- statistics in the lanes, each element (x*inv)*g in AVX-512,
 * the scan's verdict folded in -- and the VJP's dx, each element ((dy*g)*inv)-(x*coeff) (and dx plus that
 * under ADD): the scalar loops' operations, in their order */
__attribute__((target("avx512f"))) static int rms16_forward(size_t w, float eps, const float *x,
    const float *g, float *y) {
  float sum[16]; size_t r,c; __mmask16 m=0; const __m512 lim=_mm512_set1_ps(FLT_MAX);
  rms16_stats(w,x,NULL,NULL,sum,NULL);
  for (r=0;r<16;r++) {
    const float *xr=x+r*w; float *yr=y+r*w; __m512 iv=_mm512_set1_ps(1.0f/sqrtf(sum[r]/(float)w+eps));
    for (c=0;c<w;c+=16) {
      __mmask16 k=(__mmask16)(w-c>=16 ? 0xffffu : (1u<<(w-c))-1u);
      __m512 v=_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(k,xr+c),iv),_mm512_maskz_loadu_ps(k,g+c));
      _mm512_mask_storeu_ps(yr+c,k,v);
      m|=_mm512_mask_cmp_ps_mask(k,_mm512_abs_ps(v),lim,_CMP_NLE_UQ);
    }
  }
  return m==0;
}
__attribute__((target("avx512f"))) static void rms16_dx(size_t w, float inv, float coeff, int add,
    const float *x, const float *g, const float *dy, float *dx) {
  __m512 iv=_mm512_set1_ps(inv),cv=_mm512_set1_ps(coeff); size_t c;
  for (c=0;c<w;c+=16) {
    __mmask16 k=(__mmask16)(w-c>=16 ? 0xffffu : (1u<<(w-c))-1u);
    __m512 t=_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(k,dy+c),_mm512_maskz_loadu_ps(k,g+c)),iv),
        _mm512_mul_ps(_mm512_maskz_loadu_ps(k,x+c),cv));
    if (add) t=_mm512_add_ps(_mm512_maskz_loadu_ps(k,dx+c),t);
    _mm512_mask_storeu_ps(dx+c,k,t);
  }
}
/* the gamma adjoint's columns c0..c1, each a 16-lane vector's chain over the rows in order */
__attribute__((target("avx512f"))) static void rms16_cols(size_t rows, size_t w, size_t c0, size_t c1,
    int first, const float *x, const float *dy, const float *inv, float *dg) {
  size_t r,c;
  for (c=c0;c<c1;c+=16) {
    __mmask16 k=(__mmask16)(c1-c>=16 ? 0xffffu : (1u<<(c1-c))-1u);
    __m512 acc=first ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(k,dg+c);
    for (r=0;r<rows;r++)
      acc=_mm512_add_ps(acc,_mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(k,dy+r*w+c),
          _mm512_maskz_loadu_ps(k,x+r*w+c)),_mm512_set1_ps(inv[r])));
    _mm512_mask_storeu_ps(dg+c,k,acc);
  }
}
#endif
typedef struct rms_job {"""
src = once(src, "#endif\ntypedef struct rms_job {", helpers)

old_fwd = """  if (j->variant==4 && r1-r0==16) {
    float sum[16]; size_t r,c,w=j->width;
    rms16_stats(w,j->x+r0*w,NULL,NULL,sum,NULL);
    for (r=0;r<16;r++) {
      const float *xr=j->x+(r0+r)*w; float *yr=j->y+(r0+r)*w,inv=1.0f/sqrtf(sum[r]/(float)w+j->eps);
      for (c=0;c<w;c++) yr[c]=xr[c]*inv*j->g[c];
    }
  } else
#endif"""
new_fwd = """  if (j->variant==4 && r1-r0==16) {
    int ok=rms16_forward(j->width,j->eps,j->x+r0*j->width,j->g,j->y+r0*j->width);
    if (j->bad) j->bad[task]=(unsigned char)!ok;
    return;
  }
#endif"""
src = once(src, old_fwd, new_fwd)

old_rows = """      float inv=1.0f/sqrtf(sum[r]/(float)w+j->eps),coeff=dot[r]*inv*inv*inv/(float)w;
      if (j->how&BCIR_TENSOR_RMSB_ADD) for (c=0;c<w;c++) dxr[c]=dxr[c]+(dyr[c]*j->g[c]*inv-xr[c]*coeff);
      else for (c=0;c<w;c++) dxr[c]=dyr[c]*j->g[c]*inv-xr[c]*coeff;
      j->inv[r0+r]=inv;
    }
    return;"""
new_rows = """      float inv=1.0f/sqrtf(sum[r]/(float)w+j->eps),coeff=dot[r]*inv*inv*inv/(float)w;
      rms16_dx(w,inv,coeff,(j->how&BCIR_TENSOR_RMSB_ADD)!=0,xr,j->g,dyr,dxr);
      j->inv[r0+r]=inv;
    }
    (void)c;
    return;"""
src = once(src, old_rows, new_rows)

old_cols = """  size_t c0=task*BCIR_COL_CHUNK,c1=c0+BCIR_COL_CHUNK<w ? c0+BCIR_COL_CHUNK : w,n=c1>c0 ? c1-c0 : 0;
  if (j->how&BCIR_TENSOR_RMSB_FIRST) for (c=0;c<n;c++) acc[c]=0.0f;"""
new_cols = """  size_t c0=task*BCIR_COL_CHUNK,c1=c0+BCIR_COL_CHUNK<w ? c0+BCIR_COL_CHUNK : w,n=c1>c0 ? c1-c0 : 0;
#ifdef BCIR_MM_DISPATCH
  if (j->variant==4) {
    rms16_cols(j->rows,w,c0,c1,(j->how&BCIR_TENSOR_RMSB_FIRST)!=0,j->x,j->dy,j->inv,j->dg);
    return;
  }
#endif
  if (j->how&BCIR_TENSOR_RMSB_FIRST) for (c=0;c<n;c++) acc[c]=0.0f;"""
src = once(src, old_cols, new_cols)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
