/* micro: att512_dp_any (copied from the tree) on cache-resident operands: rows=32, cols=64, d=32 */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#define BCIR_MM_INLINE static inline __attribute__((always_inline))
#define ATT_MADD(f,acc,a,b) ((f) ? _mm512_fmadd_ps((a),(b),(acc)) : _mm512_add_ps((acc),_mm512_mul_ps((a),(b))))
#define ATT_MADD1(f,acc,a,b) ((f) ? fmaf((a),(b),(acc)) : (acc)+(a)*(b))
__attribute__((target("avx512f"))) BCIR_MM_INLINE __mmask16 att512_lanes(size_t count) {
  return (__mmask16)(count>=16 ? 0xffffu : (1u<<count)-1u);
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_dp_any(int f, size_t rows, size_t i0,
    size_t jt, size_t cols, size_t d, const float *dy, size_t ldy, const float *vt, size_t ldv,
    float *tile, size_t ldt) {
  size_t jb,ig,r,c;
  for (jb=0;jb<cols;jb+=32) {
    size_t nj=cols-jb<32 ? cols-jb : 32;
    __mmask16 m0=att512_lanes(nj),m1=att512_lanes(nj>16 ? nj-16 : 0);
    for (ig=0;ig+8<=rows;ig+=8) {
      __m512 a0[8],a1[8];
      if (jt+jb>i0+ig+7) continue;
      for (r=0;r<8;r++) {
        const float *tr=tile+(ig+r)*ldt+jb;
        a0[r]=_mm512_maskz_loadu_ps(m0,tr); a1[r]=_mm512_maskz_loadu_ps(m1,tr+16);
      }
      for (c=0;c<d;c++) {
        const float *vc=vt+c*ldv+jb;
        __m512 b0=_mm512_maskz_loadu_ps(m0,vc),b1=_mm512_maskz_loadu_ps(m1,vc+16);
        for (r=0;r<8;r++) {
          __m512 qv=_mm512_set1_ps(dy[(i0+ig+r)*ldy+c]);
          a0[r]=ATT_MADD(f,a0[r],qv,b0); a1[r]=ATT_MADD(f,a1[r],qv,b1);
        }
      }
      for (r=0;r<8;r++) {
        float *tr=tile+(ig+r)*ldt+jb;
        _mm512_mask_storeu_ps(tr,m0,a0[r]); _mm512_mask_storeu_ps(tr+16,m1,a1[r]);
      }
    }
    for (;ig<rows;ig++) {
      size_t j;
      for (j=0;j<nj;j++) {
        float acc=tile[ig*ldt+jb+j];
        for (c=0;c<d;c++) acc=ATT_MADD1(f,acc,dy[(i0+ig)*ldy+c],vt[c*ldv+jb+j]);
        tile[ig*ldt+jb+j]=acc;
      }
    }
  }
}

__attribute__((target("avx512f"),noinline)) static void run(size_t reps, size_t ldv, size_t d, const float *dy, const float *vt, float *tile) {
  size_t r;
  for (r=0;r<reps;r++) { memset(tile,0,32*64*sizeof(float)); att512_dp_any(0,32,4096,0,64,d,dy,d,vt,ldv,tile,64); }
}
int main(int argc, char **argv) {
  size_t d=argc>1 ? strtoul(argv[1],0,10) : 32, ldv=argc>2 ? strtoul(argv[2],0,10) : 64, reps=20000, i;
  float *dy=aligned_alloc(64,8192*4*64),*vt=aligned_alloc(64,ldv*d*4+256),*tile=aligned_alloc(64,32*64*4);
  for (i=0;i<8192*64;i++) dy[i]=(float)(i%13)*0.01f;
  for (i=0;i<ldv*d;i++) vt[i]=(float)(i%7)*0.02f;
  run(100,ldv,d,dy,vt,tile);
  unsigned long long t0=__rdtsc(); run(reps,ldv,d,dy,vt,tile); unsigned long long t1=__rdtsc();
  double probs=(double)reps*32*64, cyc=(double)(t1-t0)/probs, bound=1.45*(double)d/32;
  printf("d=%zu ldv=%zu: %.3f TSC cycles a probability, %.1f%% of v1's rate (checksum %g)\n",d,ldv,cyc,100*bound/cyc,tile[5]);
  return 0;
}
