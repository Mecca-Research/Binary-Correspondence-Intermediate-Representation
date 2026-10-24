/* GP-LANES: the geometric heads' pair and query kernels sixteen rows at a time, a row a vector lane.
 * Each kernel below is the scalar kernel of the same name less _lanes, statement for statement:
 * every per-row float a 16-lane vector, every per-row double two 8-lane halves, and every scalar
 * operation the one intrinsic that performs it (a rounded product, sum, difference, quotient or
 * square root, a widening or a rounding conversion), in the scalar order -- so each lane runs its
 * row's IEEE operations in the scalar kernel's order and every value is that kernel's bit for bit.
 * No product is fused (each multiply and add its own instruction). A row the scalar kernel sends
 * to its FP64 fallback (wedge_ill_conditioned) is flagged and recomputed by the scalar kernel. */
#include <immintrin.h>
#define GPL 16
#define GPL_TARGET __attribute__((target("avx512f")))
typedef struct gpd { __m512d lo,hi; } gpd;  /* sixteen doubles: lanes 0-7, 8-15 */
GPL_TARGET static inline gpd gpd_of(__m512 f) {  /* (double)f, lane by lane */
  gpd r; r.lo=_mm512_cvtps_pd(_mm512_castps512_ps256(f));
  r.hi=_mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(f),1))); return r;
}
GPL_TARGET static inline __m512 gpd_float(gpd d) {  /* (float)d, lane by lane */
  __m256 lo=_mm512_cvtpd_ps(d.lo),hi=_mm512_cvtpd_ps(d.hi);
  return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(lo)),_mm256_castps_pd(hi),1));
}
GPL_TARGET static inline gpd gpd_set(double x) { gpd r; r.lo=_mm512_set1_pd(x); r.hi=r.lo; return r; }
#define GPD2(name,op) GPL_TARGET static inline gpd name(gpd a,gpd b) { gpd r; r.lo=op(a.lo,b.lo); r.hi=op(a.hi,b.hi); return r; }
GPD2(gpd_add,_mm512_add_pd) GPD2(gpd_sub,_mm512_sub_pd) GPD2(gpd_mul,_mm512_mul_pd) GPD2(gpd_div,_mm512_div_pd)
GPL_TARGET static inline gpd gpd_sqrt(gpd a) { gpd r; r.lo=_mm512_sqrt_pd(a.lo); r.hi=_mm512_sqrt_pd(a.hi); return r; }
/* lane l of the result is (cond_l ? a_l : b_l); cond a pair of 8-lane masks */
typedef struct gpm { __mmask8 lo,hi; } gpm;
GPL_TARGET static inline gpd gpd_pick(gpm m,gpd a,gpd b) { gpd r; r.lo=_mm512_mask_blend_pd(m.lo,b.lo,a.lo); r.hi=_mm512_mask_blend_pd(m.hi,b.hi,a.hi); return r; }
#define GPM(name,pred) GPL_TARGET static inline gpm name(gpd a,gpd b) { gpm m; m.lo=_mm512_cmp_pd_mask(a.lo,b.lo,pred); m.hi=_mm512_cmp_pd_mask(a.hi,b.hi,pred); return m; }
GPM(gpm_lt,_CMP_LT_OQ) GPM(gpm_le,_CMP_LE_OQ) GPM(gpm_gt,_CMP_GT_OQ) GPM(gpm_eq,_CMP_EQ_OQ)
#define LD(p) _mm512_load_ps(p)
#define ST(p,x) _mm512_store_ps(p,x)
/* 16 x 16 floats: dst[q*ldd + r] = src[r*ld + q] for r < rows and q < cols, the tensor unit's
 * kb512_transpose16 with the source rows past `rows` zero, each source row read through a mask of
 * its first `cols` floats and each destination row written through a mask of its first `rows`
 * (a masked lane is neither read nor written) */
GPL_TARGET static inline void gpl_t16(const float *src,size_t ld,size_t rows,size_t cols,float *dst,size_t ldd) {
  __m512 u[16],v[16]; int q; __mmask16 rm=(__mmask16)((1u<<cols)-1u),wm=(__mmask16)((1u<<rows)-1u);
  for (q=0;q<16;q++) u[q]=(size_t)q<rows ? _mm512_maskz_loadu_ps(rm,src+(size_t)q*ld) : _mm512_setzero_ps();
  for (q=0;q<8;q++) { v[2*q]=_mm512_unpacklo_ps(u[2*q],u[2*q+1]); v[2*q+1]=_mm512_unpackhi_ps(u[2*q],u[2*q+1]); }
  for (q=0;q<4;q++) {
    u[4*q]=_mm512_shuffle_ps(v[4*q],v[4*q+2],0x44); u[4*q+1]=_mm512_shuffle_ps(v[4*q],v[4*q+2],0xEE);
    u[4*q+2]=_mm512_shuffle_ps(v[4*q+1],v[4*q+3],0x44); u[4*q+3]=_mm512_shuffle_ps(v[4*q+1],v[4*q+3],0xEE);
  }
  for (q=0;q<4;q++) {
    v[q]=_mm512_shuffle_f32x4(u[q],u[q+4],0x88); v[q+4]=_mm512_shuffle_f32x4(u[q],u[q+4],0xDD);
    v[q+8]=_mm512_shuffle_f32x4(u[q+8],u[q+12],0x88); v[q+12]=_mm512_shuffle_f32x4(u[q+8],u[q+12],0xDD);
  }
  for (q=0;q<8;q++) { u[q]=_mm512_shuffle_f32x4(v[q],v[q+8],0x88); u[q+8]=_mm512_shuffle_f32x4(v[q],v[q+8],0xDD); }
  for (q=0;q<16 && (size_t)q<cols;q++) _mm512_mask_storeu_ps(dst+(size_t)q*ldd,wm,u[q]);
}
/* sixteen rows of `width` floats a row (row-major) into the lanes, and back; a row skipped on the
 * way out is left as it was */
GPL_TARGET static void gpl_in(const float *src,size_t width,float (*dst)[GPL]) {
  size_t c0;
  for (c0=0;c0<width;c0+=16) gpl_t16(src+c0,width,GPL,width-c0<16 ? width-c0 : 16,dst[c0],GPL);
}
GPL_TARGET static void gpl_out(const float (*src)[GPL],size_t width,float *dst,const unsigned char *skip) {
  size_t c0,i,l;
  if (skip) { for (l=0;l<GPL;l++) if (skip[l]) break; }
  if (!skip || l==GPL) {
    for (c0=0;c0<width;c0+=16) gpl_t16(src[c0],GPL,width-c0<16 ? width-c0 : 16,GPL,dst+c0,width);
    return;
  }
  for (l=0;l<GPL;l++) if (!skip[l]) for (i=0;i<width;i++) dst[l*width+i]=src[i][l];
}
/* wedge_ill_conditioned, a row a lane: ill[l] the scalar verdict for row l */
GPL_TARGET static void gpl_ill(const float (*U)[GPL],const float (*V)[GPL],unsigned char *ill) {
  size_t i,l; gpd uu=gpd_set(0),vv=gpd_set(0),uv=gpd_set(0),prod,det; gpm z,s;
  for(i=0;i<GP_RANK;i++){
    gpd u=gpd_of(LD(U[i])),v=gpd_of(LD(V[i]));
    uu=gpd_add(uu,gpd_mul(u,u));vv=gpd_add(vv,gpd_mul(v,v));uv=gpd_add(uv,gpd_mul(u,v));}
  prod=gpd_mul(uu,vv);det=gpd_sub(prod,gpd_mul(uv,uv));
  z=gpm_eq(prod,gpd_set(0));s=gpm_le(det,gpd_mul(gpd_set(1e-4),prod));
  for(l=0;l<8;l++){ill[l]=(unsigned char)(((z.lo|s.lo)>>l)&1u);ill[8+l]=(unsigned char)(((z.hi|s.hi)>>l)&1u);}
}
GPL_TARGET static void gp_pair_forward_lanes(const float (*U)[GPL],const float (*V)[GPL],float (*O)[GPL],
    unsigned char *ill) {
  size_t i,k,c; gpd norm=gpd_set(0),inv;
  gpl_ill(U,V,ill);
  for(i=0,c=0;i<GP_RANK;i++) {
    __m512 ui=LD(U[i]),vi=LD(V[i]);
    for(k=i+1;k<GP_RANK;k++,c++) {
      __m512 o=_mm512_sub_ps(_mm512_mul_ps(ui,LD(V[k])),_mm512_mul_ps(LD(U[k]),vi)); gpd od;
      ST(O[c],o); od=gpd_of(o); norm=gpd_add(norm,gpd_mul(od,od));
    }
  }
  norm=gpd_sqrt(norm);inv=gpd_div(gpd_set(1.0),gpd_pick(gpm_lt(norm,gpd_set(1e-6)),gpd_set(1e-6),norm));
  for(c=0;c<GP_P;c++)ST(O[c],gpd_float(gpd_mul(gpd_of(LD(O[c])),inv)));
}
GPL_TARGET static void gp_pair_backward_lanes(const float (*U)[GPL],const float (*V)[GPL],const float (*DP)[GPL],
    float (*DU)[GPL],float (*DV)[GPL],unsigned char *ill) {
  size_t i,k,c; _Alignas(64) float P[GP_P][GPL]; gpd norm=gpd_set(0),dot=gpd_set(0),inv,scale,correction; gpm clamped;
  gpl_ill(U,V,ill);
  for(i=0;i<GP_RANK;i++){ST(DU[i],_mm512_setzero_ps());ST(DV[i],_mm512_setzero_ps());}
  for(i=0,c=0;i<GP_RANK;i++) {
    __m512 ui=LD(U[i]),vi=LD(V[i]);
    for(k=i+1;k<GP_RANK;k++,c++) {
      __m512 p=_mm512_sub_ps(_mm512_mul_ps(ui,LD(V[k])),_mm512_mul_ps(LD(U[k]),vi)); gpd pd;
      ST(P[c],p); pd=gpd_of(p); norm=gpd_add(norm,gpd_mul(pd,pd)); dot=gpd_add(dot,gpd_mul(pd,gpd_of(LD(DP[c]))));
    }
  }
  norm=gpd_sqrt(norm);clamped=gpm_le(norm,gpd_set(1e-6));inv=gpd_div(gpd_set(1.0),gpd_pick(clamped,gpd_set(1e-6),norm));
  scale=inv;correction=gpd_pick(clamped,gpd_set(0),gpd_mul(gpd_mul(dot,inv),inv));
  for(i=0,c=0;i<GP_RANK;i++) {
    __m512 ui=LD(U[i]),vi=LD(V[i]);
    for(k=i+1;k<GP_RANK;k++,c++) {
      __m512 q=gpd_float(gpd_mul(gpd_sub(gpd_of(LD(DP[c])),gpd_mul(gpd_of(LD(P[c])),correction)),scale));
      ST(DU[i],_mm512_add_ps(LD(DU[i]),_mm512_mul_ps(q,LD(V[k]))));ST(DV[k],_mm512_add_ps(LD(DV[k]),_mm512_mul_ps(q,ui)));
      ST(DU[k],_mm512_sub_ps(LD(DU[k]),_mm512_mul_ps(q,vi)));ST(DV[i],_mm512_sub_ps(LD(DV[i]),_mm512_mul_ps(q,LD(U[k]))));
    }
  }
}
static int gpl_have(void) { return __builtin_cpu_supports("avx512f"); }  /* the lanes, where the CPU has them */
/* the next group's lines, to be read (r) or written (w): a hint to the caches, no value changes */
GPL_TARGET static inline void gpl_pf_r(const void *p,size_t bytes) {
  const char *c=(const char *)p; size_t i; for (i=0;i<bytes;i+=64) __builtin_prefetch(c+i,0,3);
}
GPL_TARGET static inline void gpl_pf_w(const void *p,size_t bytes) {
  const char *c=(const char *)p; size_t i; for (i=0;i<bytes;i+=64) __builtin_prefetch(c+i,1,3);
}
