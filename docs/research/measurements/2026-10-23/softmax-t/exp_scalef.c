/* EXP-SCALEF-0: the exp contract's AVX-512 instance with 2^k applied by one vscalefps instead of two
 * exponent-field scalings. exp1 (the contract, copied from runtime/c/bcir_tensor.c) applies 2^k as
 * (p * 2^k1) * 2^k2 with k1 = k/2, k2 = k - k1: the first product is exact (|k1| <= 76 keeps it
 * normal), so the pair rounds once, as vscalefps(p, k) rounds p * 2^k once. This file checks that
 * claim over every binary32 input and times the two block forms the tree uses (a 256-float block in
 * place, the softmax's; and a row, exp_inplace_body's) plus the SiLU's sigmoid.
 * usage: exp_scalef check [STRIDE]   -- every one of the 2^32 inputs (or every STRIDE-th block), bit for bit
 *        exp_scalef time ROUNDS      -- TSC cycles an element, current vs scalef, softmax-like inputs */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#define BCIR_MM_INLINE static inline __attribute__((always_inline))
#define BCIR_EXP_LOG2E 1.44269504088896341f
#define BCIR_EXP_LN2_HI 0.693145751953125f
#define BCIR_EXP_LN2_LO 1.428606765330187e-06f
BCIR_MM_INLINE float exp_bits(uint32_t b) { float f; memcpy(&f,&b,4); return f; }
BCIR_MM_INLINE float exp1(float x) {
  float lo=x>-104.0f ? x : -104.0f,xc=lo<89.0f ? lo : 89.0f,kf,r,p,s1,s2,y; int32_t k,k1,k2;
  kf=rintf(xc*BCIR_EXP_LOG2E); k=(int32_t)kf;
  r=(xc-kf*BCIR_EXP_LN2_HI)-kf*BCIR_EXP_LN2_LO;
  p=1.0f/5040.0f; p=p*r+1.0f/720.0f; p=p*r+1.0f/120.0f; p=p*r+1.0f/24.0f;
  p=p*r+1.0f/6.0f; p=p*r+0.5f; p=p*r+1.0f; p=p*r+1.0f;
  k1=k/2; k2=k-k1;
  s1=exp_bits((uint32_t)(k1+127)<<23); s2=exp_bits((uint32_t)(k2+127)<<23);
  y=(p*s1)*s2;
  y=x>88.72283172607421875f ? exp_bits(0x7f800000u) : y;
  y=x<-103.972084045410156f ? 0.0f : y;
  return x==x ? y : x;
}
BCIR_MM_INLINE float sigmoid2(float x) {
  float e=exp1(x<0.0f ? x : -x),s=1.0f/(1.0f+e);
  return x>=0.0f ? s : e*s;
}
/* --- the tree's AVX-512 forms, as bcir_tensor.c builds them ------------------------------------ */
#define BCIR_EXP_BLOCK 64
BCIR_MM_INLINE void exp1_block(size_t nb, const float *xs, float *y) {
  float xc[BCIR_EXP_BLOCK],kf[BCIR_EXP_BLOCK],r[BCIR_EXP_BLOCK],p[BCIR_EXP_BLOCK]; int32_t k[BCIR_EXP_BLOCK];
  size_t l;
  for (l=0;l<nb;l++) { float lo=xs[l]>-104.0f ? xs[l] : -104.0f; xc[l]=lo<89.0f ? lo : 89.0f; }
  for (l=0;l<nb;l++) { kf[l]=rintf(xc[l]*BCIR_EXP_LOG2E); k[l]=(int32_t)kf[l]; }
  for (l=0;l<nb;l++) r[l]=(xc[l]-kf[l]*BCIR_EXP_LN2_HI)-kf[l]*BCIR_EXP_LN2_LO;
  for (l=0;l<nb;l++) p[l]=1.0f/5040.0f*r[l]+1.0f/720.0f;
  for (l=0;l<nb;l++) p[l]=p[l]*r[l]+1.0f/120.0f;
  for (l=0;l<nb;l++) p[l]=p[l]*r[l]+1.0f/24.0f;
  for (l=0;l<nb;l++) p[l]=p[l]*r[l]+1.0f/6.0f;
  for (l=0;l<nb;l++) p[l]=p[l]*r[l]+0.5f;
  for (l=0;l<nb;l++) p[l]=p[l]*r[l]+1.0f;
  for (l=0;l<nb;l++) p[l]=p[l]*r[l]+1.0f;
  for (l=0;l<nb;l++) {
    int32_t k1=k[l]/2,k2=k[l]-k1;
    float v=(p[l]*exp_bits((uint32_t)(k1+127)<<23))*exp_bits((uint32_t)(k2+127)<<23);
    v=xs[l]>88.72283172607421875f ? exp_bits(0x7f800000u) : v;
    v=xs[l]<-103.972084045410156f ? 0.0f : v;
    y[l]=xs[l]==xs[l] ? v : xs[l];
  }
}
BCIR_MM_INLINE void exp_inplace_body(size_t nb, size_t n, float *x) {
  float xs[BCIR_EXP_BLOCK]; size_t i=0,l;
  if (nb) for (;i+nb<=n;i+=nb) {
    for (l=0;l<nb;l++) xs[l]=x[i+l];
    exp1_block(nb,xs,x+i);
  }
  for (;i<n;i++) x[i]=exp1(x[i]);
}
__attribute__((target("avx512f"),noinline)) static void exp_inplace_cur(size_t n, float *x) { exp_inplace_body(64,n,x); }
#if defined(__clang__)
#define BCIR_EXP_LANES _Pragma("clang loop interleave_count(4)")
#else
#define BCIR_EXP_LANES
#endif
__attribute__((target("avx512f"),noinline)) static void silu_cur(size_t n, const float *x, float *y) {
  size_t i; BCIR_EXP_LANES for (i=0;i<n;i++) y[i]=x[i]*sigmoid2(x[i]);
}
/* --- the scalef form ------------------------------------------------------------------------- */
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512 exp1_v(__m512 x) {
  const __m512 lo=_mm512_set1_ps(-104.0f),hi=_mm512_set1_ps(89.0f);
  __m512 xc=_mm512_min_ps(_mm512_max_ps(x,lo),hi);  /* max: x > lo ? x : lo; min: xc < hi ? xc : hi */
  __m512 kf=_mm512_roundscale_ps(_mm512_mul_ps(xc,_mm512_set1_ps(BCIR_EXP_LOG2E)),
      _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  __m512 r=_mm512_sub_ps(_mm512_sub_ps(xc,_mm512_mul_ps(kf,_mm512_set1_ps(BCIR_EXP_LN2_HI))),
      _mm512_mul_ps(kf,_mm512_set1_ps(BCIR_EXP_LN2_LO)));
  __m512 p=_mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(1.0f/5040.0f),r),_mm512_set1_ps(1.0f/720.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f/120.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f/24.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f/6.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(0.5f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f));
  p=_mm512_add_ps(_mm512_mul_ps(p,r),_mm512_set1_ps(1.0f));
  {
    __m512 y=_mm512_scalef_ps(p,kf);
    y=_mm512_mask_mov_ps(y,_mm512_cmp_ps_mask(x,_mm512_set1_ps(88.72283172607421875f),_CMP_GT_OQ),
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000)));
    y=_mm512_mask_mov_ps(y,_mm512_cmp_ps_mask(x,_mm512_set1_ps(-103.972084045410156f),_CMP_LT_OQ),
        _mm512_setzero_ps());
    return _mm512_mask_mov_ps(x,_mm512_cmp_ps_mask(x,x,_CMP_EQ_OQ),y);
  }
}
/* four vectors at once, each step over the four before the next: the chains interleaved in the instruction
 * stream whichever compiler schedules them (EXP-LANES' order) */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void exp1_v4(__m512 *v) {
  __m512 xc[4],kf[4],r[4],p[4]; int l;
  for (l=0;l<4;l++) xc[l]=_mm512_min_ps(_mm512_max_ps(v[l],_mm512_set1_ps(-104.0f)),_mm512_set1_ps(89.0f));
  for (l=0;l<4;l++) kf[l]=_mm512_roundscale_ps(_mm512_mul_ps(xc[l],_mm512_set1_ps(BCIR_EXP_LOG2E)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  for (l=0;l<4;l++) r[l]=_mm512_sub_ps(xc[l],_mm512_mul_ps(kf[l],_mm512_set1_ps(BCIR_EXP_LN2_HI)));
  for (l=0;l<4;l++) r[l]=_mm512_sub_ps(r[l],_mm512_mul_ps(kf[l],_mm512_set1_ps(BCIR_EXP_LN2_LO)));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(1.0f/5040.0f),r[l]),_mm512_set1_ps(1.0f/720.0f));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(p[l],r[l]),_mm512_set1_ps(1.0f/120.0f));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(p[l],r[l]),_mm512_set1_ps(1.0f/24.0f));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(p[l],r[l]),_mm512_set1_ps(1.0f/6.0f));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(p[l],r[l]),_mm512_set1_ps(0.5f));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(p[l],r[l]),_mm512_set1_ps(1.0f));
  for (l=0;l<4;l++) p[l]=_mm512_add_ps(_mm512_mul_ps(p[l],r[l]),_mm512_set1_ps(1.0f));
  for (l=0;l<4;l++) {
    __m512 y=_mm512_scalef_ps(p[l],kf[l]),x=v[l];
    y=_mm512_mask_mov_ps(y,_mm512_cmp_ps_mask(x,_mm512_set1_ps(88.72283172607421875f),_CMP_GT_OQ),_mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000)));
    y=_mm512_mask_mov_ps(y,_mm512_cmp_ps_mask(x,_mm512_set1_ps(-103.972084045410156f),_CMP_LT_OQ),_mm512_setzero_ps());
    v[l]=_mm512_mask_mov_ps(x,_mm512_cmp_ps_mask(x,x,_CMP_EQ_OQ),y);
  }
}
__attribute__((target("avx512f"),noinline)) static void exp_inplace_new4(size_t n, float *x) {
  size_t i=0; int l;
  for (;i+64<=n;i+=64) {
    __m512 v[4];
    for (l=0;l<4;l++) v[l]=_mm512_loadu_ps(x+i+16*l);
    exp1_v4(v);
    for (l=0;l<4;l++) _mm512_storeu_ps(x+i+16*l,v[l]);
  }
  for (;i+16<=n;i+=16) _mm512_storeu_ps(x+i,exp1_v(_mm512_loadu_ps(x+i)));
  if (i<n) { __mmask16 m=(__mmask16)((1u<<(n-i))-1u); _mm512_mask_storeu_ps(x+i,m,exp1_v(_mm512_maskz_loadu_ps(m,x+i))); }
}
__attribute__((target("avx512f"),noinline)) static void silu_new4(size_t n, const float *x, float *y) {
  size_t i=0; int l;
  for (;i+64<=n;i+=64) {
    __m512 a[4],e[4],one=_mm512_set1_ps(1.0f),z=_mm512_setzero_ps();
    for (l=0;l<4;l++) {
      a[l]=_mm512_loadu_ps(x+i+16*l);
      e[l]=_mm512_mask_mov_ps(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a[l]),_mm512_set1_epi32((int)0x80000000u))),
          _mm512_cmp_ps_mask(a[l],z,_CMP_LT_OQ),a[l]);
    }
    exp1_v4(e);
    for (l=0;l<4;l++) {
      __m512 s=_mm512_div_ps(one,_mm512_add_ps(one,e[l]));
      s=_mm512_mask_mov_ps(_mm512_mul_ps(e[l],s),_mm512_cmp_ps_mask(a[l],z,_CMP_GE_OQ),s);
      _mm512_storeu_ps(y+i+16*l,_mm512_mul_ps(a[l],s));
    }
  }
  for (;i<n;i++) y[i]=x[i]*sigmoid2(x[i]);
}
__attribute__((target("avx512f"),noinline)) static void exp_inplace_new(size_t n, float *x) {
  size_t i=0;
  for (;i+64<=n;i+=64) {  /* four vectors' chains in flight */
    __m512 a=_mm512_loadu_ps(x+i),b=_mm512_loadu_ps(x+i+16),c=_mm512_loadu_ps(x+i+32),d=_mm512_loadu_ps(x+i+48);
    _mm512_storeu_ps(x+i,exp1_v(a)); _mm512_storeu_ps(x+i+16,exp1_v(b));
    _mm512_storeu_ps(x+i+32,exp1_v(c)); _mm512_storeu_ps(x+i+48,exp1_v(d));
  }
  for (;i+16<=n;i+=16) _mm512_storeu_ps(x+i,exp1_v(_mm512_loadu_ps(x+i)));
  if (i<n) {
    __mmask16 m=(__mmask16)((1u<<(n-i))-1u);
    _mm512_mask_storeu_ps(x+i,m,exp1_v(_mm512_maskz_loadu_ps(m,x+i)));
  }
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512 sigmoid2_v(__m512 x) {
  /* e = exp1(x < 0 ? x : -x); s = 1/(1+e); x >= 0 ? s : e*s */
  __m512 nx=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x),_mm512_set1_epi32((int)0x80000000u)));
  __m512 a=_mm512_mask_mov_ps(nx,_mm512_cmp_ps_mask(x,_mm512_setzero_ps(),_CMP_LT_OQ),x);
  __m512 e=exp1_v(a),one=_mm512_set1_ps(1.0f),s=_mm512_div_ps(one,_mm512_add_ps(one,e));
  return _mm512_mask_mov_ps(_mm512_mul_ps(e,s),_mm512_cmp_ps_mask(x,_mm512_setzero_ps(),_CMP_GE_OQ),s);
}
__attribute__((target("avx512f"),noinline)) static void silu_new(size_t n, const float *x, float *y) {
  size_t i=0;
  for (;i+32<=n;i+=32) {
    __m512 a=_mm512_loadu_ps(x+i),b=_mm512_loadu_ps(x+i+16);
    _mm512_storeu_ps(y+i,_mm512_mul_ps(a,sigmoid2_v(a))); _mm512_storeu_ps(y+i+16,_mm512_mul_ps(b,sigmoid2_v(b)));
  }
  for (;i<n;i++) y[i]=x[i]*sigmoid2(x[i]);
}
/* --- the checks ------------------------------------------------------------------------------- */
static int check(uint64_t stride) {  /* stride 1: every input; more: every stride-th block of 2^16 (a smoke test) */
  enum { N=1<<16 };
  static float xs[N],a[N],b[N],sa[N],sb[N]; uint64_t base,bad=0,sbad=0,cbad=0;
  for (base=0;base<(1ull<<32);base+=(uint64_t)N*stride) {
    size_t i;
    for (i=0;i<N;i++) { uint32_t u=(uint32_t)(base+i); memcpy(&xs[i],&u,4); }
    for (i=0;i<N;i++) a[i]=exp1(xs[i]);
    memcpy(b,xs,sizeof b); exp_inplace_new(N,b);
    for (i=0;i<N;i++) if (memcmp(&a[i],&b[i],4)) { if (bad<5) printf("exp mismatch x=%a scalar %a scalef %a\n",xs[i],a[i],b[i]); bad++; }
    memcpy(b,xs,sizeof b); exp_inplace_new4(N,b);
    for (i=0;i<N;i++) if (memcmp(&a[i],&b[i],4)) { if (bad<5) printf("exp4 mismatch x=%a\n",xs[i]); bad++; }
    memcpy(b,xs,sizeof b); exp_inplace_cur(N,b);
    for (i=0;i<N;i++) if (memcmp(&a[i],&b[i],4)) cbad++;
    for (i=0;i<N;i++) sa[i]=xs[i]*sigmoid2(xs[i]);
    silu_new4(N,xs,sb);
    for (i=0;i<N;i++) if (memcmp(&sa[i],&sb[i],4) && !(xs[i]!=xs[i] && sb[i]!=sb[i])) { if (sbad<5) printf("silu4 mismatch x=%a\n",xs[i]); sbad++; }
    silu_new(N,xs,sb);
    for (i=0;i<N;i++) if (memcmp(&sa[i],&sb[i],4)) { if (sbad<5) printf("silu mismatch x=%a scalar %a scalef %a\n",xs[i],sa[i],sb[i]); sbad++; }
  }
  printf("every binary32 input: exp scalef %llu mismatches, the tree's block form %llu, silu scalef %llu\n",
      (unsigned long long)bad,(unsigned long long)cbad,(unsigned long long)sbad);
  return bad||sbad||cbad;
}
static double now_tsc(void) { return (double)__rdtsc(); }
static void time_it(int rounds) {
  enum { BLK=256, NB=4096 };  /* 4096 blocks of 256: 4 MiB, the softmax's 16 x 16 blocks one after another */
  float *src=aligned_alloc(64,(size_t)BLK*NB*sizeof(float)),*w=aligned_alloc(64,(size_t)BLK*NB*sizeof(float));
  float *y=aligned_alloc(64,(size_t)BLK*NB*sizeof(float)); size_t i,blk; int rd,arm;
  double best[6]={1e30,1e30,1e30,1e30,1e30,1e30};
  unsigned s=12345u;
  for (i=0;i<(size_t)BLK*NB;i++) { s=s*1664525u+1013904223u; src[i]=-(float)(s>>8)/(float)(1u<<24)*12.0f; }  /* [-12, 0] */
  for (rd=0;rd<rounds;rd++) for (arm=0;arm<6;arm++) {
    double t0,t1;
    memcpy(w,src,(size_t)BLK*NB*sizeof(float));
    t0=now_tsc();
    if (arm==0) for (blk=0;blk<NB;blk++) exp_inplace_cur(BLK,w+blk*BLK);
    else if (arm==1) for (blk=0;blk<NB;blk++) exp_inplace_new(BLK,w+blk*BLK);
    else if (arm==2) silu_cur((size_t)BLK*NB,src,y);
    else if (arm==3) silu_new((size_t)BLK*NB,src,y);
    else if (arm==4) for (blk=0;blk<NB;blk++) exp_inplace_new4(BLK,w+blk*BLK);
    else silu_new4((size_t)BLK*NB,src,y);
    t1=now_tsc();
    if (t1-t0<best[arm]) best[arm]=t1-t0;
  }
  printf("{\"exp_block_cur\":%.3f,\"exp_block_scalef\":%.3f,\"exp_block_scalef_x4\":%.3f,\"silu_cur\":%.3f,\"silu_scalef\":%.3f,\"silu_scalef_x4\":%.3f}  (TSC cycles an element, best of %d)\n",
      best[0]/(BLK*NB),best[1]/(BLK*NB),best[4]/(BLK*NB),best[2]/(BLK*NB),best[3]/(BLK*NB),best[5]/(BLK*NB),rounds);
  free(src); free(w); free(y);
}
int main(int argc, char **argv) {
  if (argc>1 && !strcmp(argv[1],"check")) return check(argc>2 ? strtoull(argv[2],NULL,10) : 1);
  if (argc>2 && !strcmp(argv[1],"time")) { time_it(atoi(argv[2])); return 0; }
  fprintf(stderr,"usage: exp_scalef check | time ROUNDS\n"); return 2;
}
