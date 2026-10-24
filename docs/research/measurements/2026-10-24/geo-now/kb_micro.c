/* KBK-0: the k-blocked AVX-512 kernel's 8 x 48 block (kb512_block's untransposed path, copied from the tree) on
 * L1-resident operands: A 8 rows x K (row stride LDA floats), the packed tile K x 48, C 8 x 48; REPS calls a
 * timing, the best of TRIES. usage: kb_micro K LDA REPS TRIES */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define INL static inline __attribute__((always_inline))
typedef struct { const float *a; size_t lda; float *c; size_t ldc; float alpha, beta; int fold; } args;
__attribute__((target("avx512f"))) INL __m512 kb512_c(const float *ci, __mmask16 load, __mmask16 scale, __m512 beta) {
  __m512 c=_mm512_maskz_loadu_ps(load,ci);
  return _mm512_mask_mul_ps(c,scale,beta,c);
}
__attribute__((target("avx512f"))) INL void kb512_block(size_t mr, const args *x, const float *bt, size_t i,
    size_t jb, size_t lb, size_t nl) {
  __m512 acc0[8],acc1[8],acc2[8],alpha,beta; size_t r,l; __mmask16 keep,scale;
  for (r=0;r<mr;r++) { acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }
  {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
    for (l=0;l<nl;l++) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r*lda+l]);
        acc0[r]=_mm512_fmadd_ps(av,b0,acc0[r]);
        acc1[r]=_mm512_fmadd_ps(av,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(av,b2,acc2[r]);
      }
    }
  }
  alpha=_mm512_set1_ps(x->alpha); beta=_mm512_set1_ps(x->beta);
  keep=(__mmask16)(x->fold && lb==0 && x->beta==0.0f ? 0u : 0xffffu);
  scale=(__mmask16)(x->fold && lb==0 && x->beta!=0.0f ? 0xffffu : 0u);
  for (r=0;r<mr;r++) {
    float *ci=x->c+(i+r)*x->ldc+jb;
    _mm512_storeu_ps(ci,_mm512_add_ps(kb512_c(ci,keep,scale,beta),_mm512_mul_ps(alpha,acc0[r])));
    _mm512_storeu_ps(ci+16,_mm512_add_ps(kb512_c(ci+16,keep,scale,beta),_mm512_mul_ps(alpha,acc1[r])));
    _mm512_storeu_ps(ci+32,_mm512_add_ps(kb512_c(ci+32,keep,scale,beta),_mm512_mul_ps(alpha,acc2[r])));
  }
}
__attribute__((target("avx512f"), noinline)) static void run(const args *x, const float *bt, size_t k, long reps) {
  long q;
  for (q=0;q<reps;q++) kb512_block(8,x,bt,0,0,0,k);
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
  size_t k=argc>1 ? strtoul(argv[1],0,10) : 128, lda=argc>2 ? strtoul(argv[2],0,10) : 128, i;
  long reps=argc>3 ? atol(argv[3]) : 200000; int tries=argc>4 ? atoi(argv[4]) : 5, t; double best=1e30;
  float *a=aligned_alloc(64,(8*lda+64)*4), *bt=aligned_alloc(64,k*48*4), *c=aligned_alloc(64,8*48*4);
  args x;
  for (i=0;i<8*lda;i++) a[i]=(float)(i%7)*0.25f-0.5f;
  for (i=0;i<k*48;i++) bt[i]=(float)(i%5)*0.125f;
  for (i=0;i<8*48;i++) c[i]=0.0f;
  x.a=a; x.lda=lda; x.c=c; x.ldc=48; x.alpha=1.0f; x.beta=1.0f; x.fold=1;
  for (t=0;t<tries;t++) { double t0=now(); run(&x,bt,k,reps); t0=now()-t0; if (t0<best) best=t0; }
  printf("K=%zu lda=%zu: %.1f GFLOP/s (%.3f ns a block; checksum %g)\n",k,lda,2.0*8*48*k*reps/best*1e-9,best/reps*1e9,c[17]);
  return 0;
}
