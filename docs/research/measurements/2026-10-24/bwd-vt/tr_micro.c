/* micro: att512_tr_block (copied from the tree): a band of 32 terms over every 8-row block of 1024 output rows,
 * d=32 (nv=2), m (P) at row stride LDM floats */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#define BCIR_MM_INLINE static inline __attribute__((always_inline))
#define ATT_MADD(f,acc,a,b) ((f) ? _mm512_fmadd_ps((a),(b),(acc)) : _mm512_add_ps((acc),_mm512_mul_ps((a),(b))))
#define ATT_MADD_MASK(f,acc,take,a,b) ((f) ? _mm512_mask3_fmadd_ps((a),(b),(acc),(take)) : \
    _mm512_mask_add_ps((acc),(take),(acc),_mm512_mul_ps((a),(b))))
__attribute__((target("avx512f"))) BCIR_MM_INLINE __mmask16 att512_lanes(size_t count) {
  return (__mmask16)(count>=16 ? 0xffffu : (1u<<count)-1u);
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_tr_block(int f, int nv, size_t j0, size_t ib,
    size_t ie, size_t c0, const float *m, size_t ldm, const float *x, size_t ldx, float *out, size_t ldo,
    int zero, __mmask16 m0, __mmask16 m1, __mmask16 m2) {
  __m512 a0[8],a1[8],a2[8]; size_t r,i,di; int diag=j0>=ib;  /* the band holds the block's diagonal */
  for (r=0;r<8;r++) {
    const float *o=out+(j0+r)*ldo+c0;
    a0[r]=_mm512_maskz_loadu_ps(zero ? 0 : m0,o);
    if (nv>1) a1[r]=_mm512_maskz_loadu_ps(zero ? 0 : m1,o+16);
    if (nv>2) a2[r]=_mm512_maskz_loadu_ps(zero ? 0 : m2,o+32);
  }
  for (di=0;diag && di<7;di++) {  /* term j0+di: the rows r <= di */
    const float *xi=x+(j0+di)*ldx+c0,*mi=m+(j0+di)*ldm+j0;
    __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
    if (nv>1) b1=_mm512_maskz_loadu_ps(m1,xi+16);
    if (nv>2) b2=_mm512_maskz_loadu_ps(m2,xi+32);
    for (r=0;r<8;r++) {  /* the rows r <= di, the others masked out */
      __mmask16 take=(__mmask16)(r<=di ? 0xffffu : 0u);
      __m512 mv=_mm512_set1_ps(mi[r]);
      a0[r]=ATT_MADD_MASK(f,a0[r],take,mv,b0);
      if (nv>1) a1[r]=ATT_MADD_MASK(f,a1[r],take,mv,b1);
      if (nv>2) a2[r]=ATT_MADD_MASK(f,a2[r],take,mv,b2);
    }
  }
  for (i=diag ? j0+7 : ib;i<ie;i++) {  /* every row of the block takes term i */
    const float *xi=x+i*ldx+c0,*mi=m+i*ldm+j0;
    __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_setzero_ps(),b2=_mm512_setzero_ps();
    if (nv>1) b1=_mm512_maskz_loadu_ps(m1,xi+16);
    if (nv>2) b2=_mm512_maskz_loadu_ps(m2,xi+32);
    for (r=0;r<8;r++) {
      __m512 mv=_mm512_set1_ps(mi[r]);
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

__attribute__((target("avx512f"),noinline)) static void run(size_t reps, size_t ldm, const float *m, const float *x, float *out) {
  size_t r,j0; __mmask16 m0=att512_lanes(32),m1=att512_lanes(16),m2=0;
  for (r=0;r<reps;r++) for (j0=0;j0<1024;j0+=8) att512_tr_block(0,2,j0,1024,1056,0,m,ldm,x,32,out,32,0,m0,m1,m2);
}
int main(int argc, char **argv) {
  size_t ldm=argc>1 ? strtoul(argv[1],0,10) : 2048, reps=200, i;
  float *m=aligned_alloc(64,(1056+8)*ldm*4),*x=aligned_alloc(64,1100*32*4),*out=aligned_alloc(64,1100*32*4);
  for (i=0;i<(1056+8)*ldm;i++) m[i]=(float)(i%11)*0.01f;
  for (i=0;i<1100*32;i++) { x[i]=(float)(i%5)*0.1f; out[i]=0.0f; }
  run(5,ldm,m,x,out);
  unsigned long long t0=__rdtsc(); run(reps,ldm,m,x,out); unsigned long long t1=__rdtsc();
  double probs=(double)reps*1024*32, cyc=(double)(t1-t0)/probs, bound=1.45;
  printf("ldm=%zu: %.3f TSC cycles a probability, %.1f%% of v1's rate (%g)\n",ldm,cyc,100*bound/cyc,out[7]);
  return 0;
}
