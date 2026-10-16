#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#if defined(_WIN32)
#define BCIR_POOL_WIN32 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#define BCIR_POOL_PTHREADS 1
#include <pthread.h>
#include <sched.h>
#endif

void bcir_tensor_cblas_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  bcir_cblas_sgemm_fn fn=ctx ? *(bcir_cblas_sgemm_fn *)ctx : NULL;
  if (!fn || m>INT_MAX || n>INT_MAX || k>INT_MAX) {
    bcir_tensor_mm(NULL,ta,tb,m,n,k,alpha,a,b,beta,c); return;
  }
  fn(101,ta ? 112 : 111,tb ? 112 : 111,(int)m,(int)n,(int)k,alpha,a,
      (int)(ta ? m : k),b,(int)(tb ? k : n),beta,c,(int)n);
}
/* The GEMM contract every path keeps: c[i][j] starts at its beta-scaled value and then adds
 * (alpha*A(i,l))*B(l,j) for l ascending -- one rounded multiply and one rounded add per term,
 * since contraction is off by the build contract. The kernel below changes only the loop nest:
 * a KCxNR tile of B is packed once per (column block, depth block) and reused by every row, and
 * an MRxNR block of C stays in registers across the tile's depth instead of making a round trip
 * through memory per term (the old nest was bound by that store-to-load latency, and its B
 * (tb=0) nest also by alias checks around a 32-iteration loop). A ragged right edge is padded
 * with zero columns that are computed and never stored. Each element sees the same terms in the
 * same order, so every variant is bit-identical to the ascending-k loop on every input, finite
 * or not; test_decoder_train.c holds each variant the host can run to that loop bit for bit. */
#define BCIR_MM_KC 64
#define BCIR_MM_MRMAX 4
#define BCIR_MM_NRMAX 64
#if defined(__GNUC__) || defined(__clang__)
#define BCIR_MM_INLINE static inline __attribute__((always_inline))
#else
#define BCIR_MM_INLINE static inline
#endif
/* The causal products attention issues reuse the kernel with a mask over the matrices' own
 * indices: MM_LOWER stores only C[i][j] with j <= i; MM_KLE adds term l to row i only when
 * l <= i; MM_KGE only when l >= i. A masked term is skipped, never added as a zero, so an
 * operand outside the mask -- the zeros above a causal diagonal, or a non-finite value -- cannot
 * reach a result. Strides are explicit (lda, ldb, ldc) so a head's rows can be read in place. */
enum { MM_FULL, MM_LOWER, MM_KLE, MM_KGE };
typedef struct mm_args {
  int ta,tb; size_t m,n,k; float alpha;
  const float *a; size_t lda; const float *b; size_t ldb; float *c; size_t ldc; int mode;
  float beta; int fold;  /* fold: the blocked kernels scale c by beta in their first K block */
} mm_args;
/* The contract an instance keeps, a compile-time constant of the instance: v1 adds one rounded
 * product per term, v2 (NDT-FMA) one fmaf. */
enum { MM_V1=0, MM_FUSED=1, MM_KBLOCK=2 };
/* Rows [i, i+mr) of C over columns [jb, jb+nj) and depths [lb, lb+nl); bt is the packed tile. */
BCIR_MM_INLINE void mm_block(size_t mr, size_t nr, int fused, const mm_args *x, const float *bt,
    size_t i, size_t jb, size_t nj, size_t lb, size_t nl, int mode) {
  float acc[BCIR_MM_MRMAX][BCIR_MM_NRMAX];
  size_t r,j,l;
#ifdef __clang_analyzer__
  /* Every caller passes nj = min(n - jb, nr), so the stores below never read past the columns
   * the block loaded. clang --analyze sees this block on its own and cannot know that, so the
   * analyzer alone is told: GCC does not fold the same line away, and the kernel's machine code
   * stays exactly the code that was measured. */
  if (nj>nr) nj=nr;
#endif
  for (r=0;r<mr;r++) {
    const float *ci=x->c+(i+r)*x->ldc+jb;
    if (nj==nr) { for (j=0;j<nr;j++) acc[r][j]=ci[j]; }
    else for (j=0;j<nr;j++) acc[r][j]=j<nj ? ci[j] : 0.0f;
  }
  for (l=0;l<nl;l++) {
    const float *bl=bt+l*nr; float av[BCIR_MM_MRMAX];
    for (r=0;r<mr;r++) av[r]=x->alpha*x->a[x->ta ? (lb+l)*x->lda+i+r : (i+r)*x->lda+lb+l];
    for (r=0;r<mr;r++) {
      if ((mode==MM_KLE && lb+l>i+r) || (mode==MM_KGE && lb+l<i+r)) continue;
      if (fused) { for (j=0;j<nr;j++) acc[r][j]=fmaf(av[r],bl[j],acc[r][j]); }
      else for (j=0;j<nr;j++) acc[r][j]+=av[r]*bl[j];
    }
  }
  for (r=0;r<mr;r++) {
    float *ci=x->c+(i+r)*x->ldc+jb; size_t end=nj;
    if (mode==MM_LOWER) end=i+r<jb ? 0 : i+r-jb+1<nj ? i+r-jb+1 : nj;
    /* every index into acc is a constant once the loops unroll, so clang keeps the block in
     * registers; a bound read from the masks would leave it on the stack across the depth */
    if (end==nr) { for (j=0;j<nr;j++) ci[j]=acc[r][j]; }
    else for (j=0;j<nr;j++) if (j<end) ci[j]=acc[r][j];
  }
}
/* Whether rows [i, i+mr) of this (column, depth) block are outside the mask's reach entirely. */
BCIR_MM_INLINE int mm_unmasked(int mode, size_t i, size_t mr, size_t jb, size_t nj, size_t lb,
    size_t nl) {
  return mode==MM_FULL || (mode==MM_LOWER && i>=jb+nj-1) || (mode==MM_KLE && i>=lb+nl-1) ||
    (mode==MM_KGE && i+mr-1<=lb);
}
/* The nest over column blocks [jb0, jb1) and rows [r0, r1): the serial kernel passes the whole
 * matrix, a pool task its own column block and row range. Rows are grouped mr at a time from r0
 * and the rest singly; each row's accumulator sees the same terms in the same order whatever the
 * grouping, so any row partition is bit-identical to the whole. */
BCIR_MM_INLINE void mm_kernel(size_t mr, size_t nr, int fused, const mm_args *x, size_t jb0,
    size_t jb1, size_t r0, size_t r1) {
  /* Aligned to a cache line: every row of the tile is then read in whole vectors that never
   * straddle two lines. GCC realigns the stack for the AVX-512 instance on its own; clang keeps
   * the ABI's 16 bytes, which left the tile's offset to the caller's stack depth and to ASLR. */
  _Alignas(64) float bt[BCIR_MM_KC*BCIR_MM_NRMAX];
  size_t i,j,l,jb,lb;
  for (jb=jb0;jb<jb1;jb+=nr) {
    size_t nj=x->n-jb<nr ? x->n-jb : nr;
    for (lb=0;lb<x->k;lb+=BCIR_MM_KC) {
      size_t nl=x->k-lb<BCIR_MM_KC ? x->k-lb : BCIR_MM_KC,i0=r0,i1=r1;
      /* the rows this block reaches at all: under MM_LOWER none above jb stores here, under
       * MM_KLE none above lb takes a term, under MM_KGE none past lb+nl-1 does */
      if (x->mode==MM_LOWER && jb>i0) i0=jb;
      if (x->mode==MM_KLE && lb>i0) i0=lb;
      if (x->mode==MM_KGE && lb+nl<i1) i1=lb+nl;
      if (i0>=i1) continue;
      /* pack the tile, reading each source row contiguously; the padding columns are zero */
      if (x->tb) {
        for (j=0;j<nj;j++) {
          const float *src=x->b+(jb+j)*x->ldb+lb;
          for (l=0;l<nl;l++) bt[l*nr+j]=src[l];
        }
      } else {
        for (l=0;l<nl;l++) {
          const float *src=x->b+(lb+l)*x->ldb+jb;
          for (j=0;j<nj;j++) bt[l*nr+j]=src[j];
        }
      }
      if (nj<nr) for (l=0;l<nl;l++) for (j=nj;j<nr;j++) bt[l*nr+j]=0.0f;
      for (i=i0;i+mr<=i1;i+=mr) {
        if (mm_unmasked(x->mode,i,mr,jb,nj,lb,nl)) mm_block(mr,nr,fused,x,bt,i,jb,nj,lb,nl,MM_FULL);
        else mm_block(mr,nr,fused,x,bt,i,jb,nj,lb,nl,x->mode);
      }
      for (;i<i1;i++) {
        if (mm_unmasked(x->mode,i,1,jb,nj,lb,nl)) mm_block(1,nr,fused,x,bt,i,jb,nj,lb,nl,MM_FULL);
        else mm_block(1,nr,fused,x,bt,i,jb,nj,lb,nl,x->mode);
      }
    }
  }
}
/* --- the blocked fused contract, v3 (NDT-KBLOCK) ------------------------------------------- */
/* The next block's length for `rem` remaining terms: the one rule, bcir_tensor_kblock_split's. */
static size_t kblock_next(size_t rem) {
  const size_t q=BCIR_MM_KBLOCK,u=BCIR_MM_KBLOCK_UNROLL;
  return rem>=2*q ? q : rem>q ? (rem/2+u-1)/u*u : rem;
}
size_t bcir_tensor_kblock_split(size_t k, size_t *lengths, size_t capacity) {
  size_t ls=0,count=0;
  while (ls<k) {
    size_t len=kblock_next(k-ls);
    if (lengths && count<capacity) lengths[count]=len;
    count++; ls+=len;
  }
  return count;
}
/* c's beta scaling, one element: zero under beta 0, else the rounded product -- the caller's
 * pass, or a blocked kernel's first K block (NDT-BETA). */
BCIR_MM_INLINE float mm_scaled(float beta, float c) { return beta==0.0f ? 0.0f : beta*c; }
/* Rows [i, i+mr) over columns [jb, jb+nj) for one K block: P from zero over the block's nl
 * packed depths (one fmaf per term, l ascending), then c = c + alpha*P, c beta-scaled first --
 * by the caller, or here in the first K block when x->fold says so (NDT-BETA: the same rounded
 * operations in the same order, without a pass over c of its own). The block stays in registers
 * across the whole K block, so c is read and written once per block, not once per depth chunk. */
BCIR_MM_INLINE void mm_block_kb(size_t mr, size_t nr, const mm_args *x, const float *bt,
    size_t i, size_t jb, size_t nj, size_t lb, size_t nl) {
  float acc[BCIR_MM_MRMAX][BCIR_MM_NRMAX];
  size_t r,j,l; int first;
#ifdef __clang_analyzer__
  if (nj>nr) nj=nr;
#endif
  for (r=0;r<mr;r++) for (j=0;j<nr;j++) acc[r][j]=0.0f;
  for (l=0;l<nl;l++) {
    const float *tile=bt+l*nr; float ar[BCIR_MM_MRMAX];
    for (r=0;r<mr;r++) ar[r]=x->ta ? x->a[(lb+l)*x->lda+i+r] : x->a[(i+r)*x->lda+lb+l];
    for (r=0;r<mr;r++) for (j=0;j<nr;j++) acc[r][j]=fmaf(ar[r],tile[j],acc[r][j]);
  }
  first=x->fold && lb==0;
  for (r=0;r<mr;r++) {
    float *ci=x->c+(i+r)*x->ldc+jb;
    if (first) {
      if (nj==nr) { for (j=0;j<nr;j++) ci[j]=mm_scaled(x->beta,ci[j])+x->alpha*acc[r][j]; }
      else for (j=0;j<nr;j++) if (j<nj) ci[j]=mm_scaled(x->beta,ci[j])+x->alpha*acc[r][j];
    } else if (nj==nr) { for (j=0;j<nr;j++) ci[j]=ci[j]+x->alpha*acc[r][j]; }
    else for (j=0;j<nr;j++) if (j<nj) ci[j]=ci[j]+x->alpha*acc[r][j];
  }
}
/* The nest: per column block, per K block of the rule, the block's tile packed once (every depth
 * of the block, padding columns zero) and every row group run over it. MM_FULL only: the masked
 * products are attention's, which keeps its own contracts. */
BCIR_MM_INLINE void mm_kernel_kb(size_t mr, size_t nr, const mm_args *x, size_t jb0, size_t jb1,
    size_t r0, size_t r1) {
  _Alignas(64) float bt[BCIR_MM_KBLOCK*BCIR_MM_NRMAX];
  size_t i,j,l,jb,lb,nl;
  for (jb=jb0;jb<jb1;jb+=nr) {
    size_t nj=x->n-jb<nr ? x->n-jb : nr;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      if (x->tb) {
        for (j=0;j<nj;j++) {
          const float *column=x->b+(jb+j)*x->ldb+lb;
          for (l=0;l<nl;l++) bt[l*nr+j]=column[l];
        }
      } else {
        for (l=0;l<nl;l++) {
          const float *row=x->b+(lb+l)*x->ldb+jb;
          for (j=0;j<nj;j++) bt[l*nr+j]=row[j];
        }
      }
      for (j=nj;j<nr;j++) for (l=0;l<nl;l++) bt[l*nr+j]=0.0f;  /* the padding columns */
      for (i=r0;i+mr<=r1;i+=mr) mm_block_kb(mr,nr,x,bt,i,jb,nj,lb,nl);
      for (;i<r1;i++) mm_block_kb(1,nr,x,bt,i,jb,nj,lb,nl);
    }
  }
}
/* A task's range: whole column blocks of the instance's width (mm_nr) and a row range. */
typedef struct mm_range { size_t jb0,jb1,r0,r1; } mm_range;
static void mm_generic(const mm_args *x, const mm_range *r) { mm_kernel(2,32,MM_V1,x,r->jb0,r->jb1,r->r0,r->r1); }
static void mm_generic_fused(const mm_args *x, const mm_range *r) { mm_kernel(2,32,MM_FUSED,x,r->jb0,r->jb1,r->r0,r->r1); }
static void mm_generic_kb(const mm_args *x, const mm_range *r) { mm_kernel_kb(2,32,x,r->jb0,r->jb1,r->r0,r->r1); }
/* x86-64 Linux with GCC or Clang also carries AVX2 and AVX-512 variants of the same kernel,
 * chosen per call from what the CPU and OS support; anything else runs mm_generic.
 * BCIR_TENSOR_MAX_VARIANT caps the choice (1, 2 or 4) so a build can run a narrower one. */
#ifndef BCIR_TENSOR_MAX_VARIANT
#define BCIR_TENSOR_MAX_VARIANT 4
#endif
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(BCIR_TENSOR_NO_DISPATCH) && BCIR_TENSOR_MAX_VARIANT>1
#define BCIR_MM_DISPATCH 1
#include <immintrin.h>
__attribute__((target("avx2"))) static void mm_avx2(const mm_args *x, const mm_range *r) {
  mm_kernel(2,32,MM_V1,x,r->jb0,r->jb1,r->r0,r->r1);
}
/* the AVX2 lane mask with the first `count` (0..8) lanes set */
__attribute__((target("avx2"))) BCIR_MM_INLINE __m256i kb256_mask(size_t count) {
  return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)count),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
}
__attribute__((target("avx512f"))) static void mm_avx512(const mm_args *x, const mm_range *r) {
  mm_kernel(4,64,MM_V1,x,r->jb0,r->jb1,r->r0,r->r1);
}
/* the fused instances: AVX2 with the FMA extension (fmaf fuses in registers), AVX-512F (which
 * carries FMA); the compiler folds the fmaf call into the vector fused multiply-add */
__attribute__((target("avx2,fma"))) static void mm_avx2_fused(const mm_args *x, const mm_range *r) {
  mm_kernel(2,32,MM_FUSED,x,r->jb0,r->jb1,r->r0,r->r1);
}
__attribute__((target("avx512f"))) static void mm_avx512_fused(const mm_args *x, const mm_range *r) {
  mm_kernel(4,64,MM_FUSED,x,r->jb0,r->jb1,r->r0,r->r1);
}
/* The blocked instances (v3, NDT-BLOCK): written in intrinsics, so the register block is the
 * block that runs on both compilers -- 8 rows x 48 columns of accumulators on AVX-512 (24 of
 * the 32 vector registers), 6 x 16 on AVX2 (12 of 16); per depth the tile is three (two) vector
 * loads and every A term one broadcast from memory: eleven loads per 24 fused multiply-adds, and
 * the tile's traffic from L2 per fma a third of the 4 x 64 block's (measured on the step's
 * shapes against 12 x 32, 6 x 64 and 4 x 96; NDT-BLOCK's record). Rows that do not fill a block
 * run 4-row (2-row) and single-row blocks of the same chain. Per element the arithmetic is
 * mm_block_kb's exactly: the block from zero,
 * one fma per term with l ascending, then c = c + alpha*P as a rounded product and a rounded
 * sum; the generic instance keeps the plain-C block, and test_decoder_train.c holds every
 * instance to the fmaf chain. The tile is packed by vector: rows of B through masked loads
 * (the padding columns zero, nothing read past the matrix), columns of B^T through 16 x 16
 * (8 x 8) register transposes with the ragged edges copied scalar. A transposed A is read one
 * cache line per depth, so its next lines are prefetched BCIR_MM_KB_PREFETCH depths ahead. */
#define BCIR_MM_KB_PREFETCH 8
/* c before a K block's terms, straight-line (NDT-BETA): the lanes `load` admits as stored, the
 * rest +0.0, then the lanes `scale` admits times beta -- in the first K block under beta 0 no lane
 * is loaded (zero, c never read), under any other beta every lane is scaled (the rounded product,
 * the caller's pass exactly), and in a later block c is as stored. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512 kb512_c(const float *ci, __mmask16 load,
    __mmask16 scale, __m512 beta) {
  __m512 c=_mm512_maskz_loadu_ps(load,ci);
  return _mm512_mask_mul_ps(c,scale,beta,c);
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_block(size_t mr, const mm_args *x,
    const float *bt, size_t i, size_t jb, size_t nj, size_t lb, size_t nl) {
  __m512 acc0[8],acc1[8],acc2[8],alpha,beta; size_t r,l; __mmask16 keep,scale;
  for (r=0;r<mr;r++) { acc0[r]=_mm512_setzero_ps(); acc1[r]=_mm512_setzero_ps(); acc2[r]=_mm512_setzero_ps(); }
  if (x->ta) {
    const float *ap=x->a+lb*x->lda+i; size_t lda=x->lda;
    for (l=0;l<nl;l++,ap+=lda) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      _mm_prefetch((const char *)(ap+BCIR_MM_KB_PREFETCH*lda),_MM_HINT_T0);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r]);
        acc0[r]=_mm512_fmadd_ps(av,b0,acc0[r]); acc1[r]=_mm512_fmadd_ps(av,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(av,b2,acc2[r]);
      }
    }
  } else {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
    for (l=0;l<nl;l++) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r*lda+l]);
        acc0[r]=_mm512_fmadd_ps(av,b0,acc0[r]); acc1[r]=_mm512_fmadd_ps(av,b1,acc1[r]);
        acc2[r]=_mm512_fmadd_ps(av,b2,acc2[r]);
      }
    }
  }
  /* set after the depth loop: the block's 24 accumulators, three tile rows and a broadcast hold
   * 28 of the 32 vector registers through it, and a value live across it costs a spill a depth */
  alpha=_mm512_set1_ps(x->alpha); beta=_mm512_set1_ps(x->beta);
  keep=(__mmask16)(x->fold && lb==0 && x->beta==0.0f ? 0u : 0xffffu);
  scale=(__mmask16)(x->fold && lb==0 && x->beta!=0.0f ? 0xffffu : 0u);
  if (nj==48) {
    for (r=0;r<mr;r++) {
      float *ci=x->c+(i+r)*x->ldc+jb;
      _mm512_storeu_ps(ci,_mm512_add_ps(kb512_c(ci,keep,scale,beta),_mm512_mul_ps(alpha,acc0[r])));
      _mm512_storeu_ps(ci+16,_mm512_add_ps(kb512_c(ci+16,keep,scale,beta),_mm512_mul_ps(alpha,acc1[r])));
      _mm512_storeu_ps(ci+32,_mm512_add_ps(kb512_c(ci+32,keep,scale,beta),_mm512_mul_ps(alpha,acc2[r])));
    }
  } else {
    __mmask16 m0=(__mmask16)(nj>=16 ? 0xffffu : (1u<<nj)-1u);
    __mmask16 m1=(__mmask16)(nj>=32 ? 0xffffu : nj>16 ? (1u<<(nj-16))-1u : 0u);
    __mmask16 m2=(__mmask16)(nj>32 ? (1u<<(nj-32))-1u : 0u);
    for (r=0;r<mr;r++) {
      float *ci=x->c+(i+r)*x->ldc+jb;
      _mm512_mask_storeu_ps(ci,m0,_mm512_add_ps(kb512_c(ci,m0&keep,scale,beta),_mm512_mul_ps(alpha,acc0[r])));
      _mm512_mask_storeu_ps(ci+16,m1,_mm512_add_ps(kb512_c(ci+16,m1&keep,scale,beta),_mm512_mul_ps(alpha,acc1[r])));
      _mm512_mask_storeu_ps(ci+32,m2,_mm512_add_ps(kb512_c(ci+32,m2&keep,scale,beta),_mm512_mul_ps(alpha,acc2[r])));
    }
  }
}
/* 16 depths x 16 columns of B^T (src: one row per column, ld apart) into the tile (dst: one row
 * per depth, ldd apart): unpack pairs, shuffle quads, then the 128-bit lanes twice. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_transpose16(const float *src,
    size_t ld, float *dst, size_t ldd) {
  __m512 u[16],v[16]; int q;
  for (q=0;q<16;q++) u[q]=_mm512_loadu_ps(src+(size_t)q*ld);
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
  for (q=0;q<16;q++) _mm512_storeu_ps(dst+(size_t)q*ldd,u[q]);
}
__attribute__((target("avx512f"))) static void kb512_pack(const mm_args *x, float *bt, size_t jb,
    size_t nj, size_t lb, size_t nl) {
  size_t j,l;
  if (x->tb) {
    size_t jw=nj/16*16,lw=nl/16*16;  /* the whole 16 x 16 blocks, then the ragged edges */
    for (j=0;j<jw;j+=16) for (l=0;l<lw;l+=16) kb512_transpose16(x->b+(jb+j)*x->ldb+lb+l,x->ldb,bt+l*48+j,48);
    for (j=0;j<nj;j++) {
      const float *column=x->b+(jb+j)*x->ldb+lb;
      for (l=j<jw ? lw : 0;l<nl;l++) bt[l*48+j]=column[l];
    }
    for (j=nj;j<48;j++) for (l=0;l<nl;l++) bt[l*48+j]=0.0f;
  } else {
    __mmask16 m0=(__mmask16)(nj>=16 ? 0xffffu : (1u<<nj)-1u);
    __mmask16 m1=(__mmask16)(nj>=32 ? 0xffffu : nj>16 ? (1u<<(nj-16))-1u : 0u);
    __mmask16 m2=(__mmask16)(nj>32 ? (1u<<(nj-32))-1u : 0u);
    for (l=0;l<nl;l++) {
      const float *row=x->b+(lb+l)*x->ldb+jb;
      _mm512_store_ps(bt+l*48,_mm512_maskz_loadu_ps(m0,row));
      _mm512_store_ps(bt+l*48+16,_mm512_maskz_loadu_ps(m1,row+16));
      _mm512_store_ps(bt+l*48+32,_mm512_maskz_loadu_ps(m2,row+32));
    }
  }
}
__attribute__((target("avx512f"))) static void mm_avx512_kb(const mm_args *x, const mm_range *r) {
  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; size_t i,jb,lb,nl;
  for (jb=r->jb0;jb<r->jb1;jb+=48) {
    size_t nj=x->n-jb<48 ? x->n-jb : 48;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      kb512_pack(x,bt,jb,nj,lb,nl);
      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block(8,x,bt,i,jb,nj,lb,nl);
      for (;i+4<=r->r1;i+=4) kb512_block(4,x,bt,i,jb,nj,lb,nl);
      for (;i<r->r1;i++) kb512_block(1,x,bt,i,jb,nj,lb,nl);
    }
  }
}
/* kb512_c's AVX2 twin: the lanes `load` admits as stored (the rest +0.0), then the product with
 * beta where `scale` is all ones (a blend, so a later block's c is as stored). */
__attribute__((target("avx2,fma"))) BCIR_MM_INLINE __m256 kb256_c(const float *ci, __m256i load,
    __m256 scale, __m256 beta) {
  __m256 c=_mm256_maskload_ps(ci,load);
  return _mm256_blendv_ps(c,_mm256_mul_ps(beta,c),scale);
}
__attribute__((target("avx2,fma"))) BCIR_MM_INLINE void kb256_block(size_t mr, const mm_args *x,
    const float *bt, size_t i, size_t jb, size_t nj, size_t lb, size_t nl) {
  __m256 acc0[6],acc1[6],alpha,beta,scale; size_t r,l; __m256i keep;
  for (r=0;r<mr;r++) { acc0[r]=_mm256_setzero_ps(); acc1[r]=_mm256_setzero_ps(); }
  if (x->ta) {
    const float *ap=x->a+lb*x->lda+i; size_t lda=x->lda;
    for (l=0;l<nl;l++,ap+=lda) {
      __m256 b0=_mm256_load_ps(bt+l*16),b1=_mm256_load_ps(bt+l*16+8);
      _mm_prefetch((const char *)(ap+BCIR_MM_KB_PREFETCH*lda),_MM_HINT_T0);
      for (r=0;r<mr;r++) {
        __m256 av=_mm256_set1_ps(ap[r]);
        acc0[r]=_mm256_fmadd_ps(av,b0,acc0[r]); acc1[r]=_mm256_fmadd_ps(av,b1,acc1[r]);
      }
    }
  } else {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
    for (l=0;l<nl;l++) {
      __m256 b0=_mm256_load_ps(bt+l*16),b1=_mm256_load_ps(bt+l*16+8);
      for (r=0;r<mr;r++) {
        __m256 av=_mm256_set1_ps(ap[r*lda+l]);
        acc0[r]=_mm256_fmadd_ps(av,b0,acc0[r]); acc1[r]=_mm256_fmadd_ps(av,b1,acc1[r]);
      }
    }
  }
  /* set after the depth loop, as kb512_block's */
  alpha=_mm256_set1_ps(x->alpha); beta=_mm256_set1_ps(x->beta);
  keep=x->fold && lb==0 && x->beta==0.0f ? _mm256_setzero_si256() : _mm256_set1_epi32(-1);
  scale=_mm256_castsi256_ps(x->fold && lb==0 && x->beta!=0.0f ? _mm256_set1_epi32(-1) : _mm256_setzero_si256());
  if (nj==16) {
    for (r=0;r<mr;r++) {
      float *ci=x->c+(i+r)*x->ldc+jb;
      _mm256_storeu_ps(ci,_mm256_add_ps(kb256_c(ci,keep,scale,beta),_mm256_mul_ps(alpha,acc0[r])));
      _mm256_storeu_ps(ci+8,_mm256_add_ps(kb256_c(ci+8,keep,scale,beta),_mm256_mul_ps(alpha,acc1[r])));
    }
  } else {
    __m256i m0=kb256_mask(nj>=8 ? 8 : nj),m1=kb256_mask(nj>8 ? nj-8 : 0);
    for (r=0;r<mr;r++) {
      float *ci=x->c+(i+r)*x->ldc+jb;
      _mm256_maskstore_ps(ci,m0,_mm256_add_ps(kb256_c(ci,_mm256_and_si256(m0,keep),scale,beta),_mm256_mul_ps(alpha,acc0[r])));
      _mm256_maskstore_ps(ci+8,m1,_mm256_add_ps(kb256_c(ci+8,_mm256_and_si256(m1,keep),scale,beta),_mm256_mul_ps(alpha,acc1[r])));
    }
  }
}
__attribute__((target("avx2,fma"))) static void kb256_pack(const mm_args *x, float *bt, size_t jb,
    size_t nj, size_t lb, size_t nl) {
  size_t j,l;
  if (x->tb) {
    for (j=0;j<nj;j++) { const float *column=x->b+(jb+j)*x->ldb+lb; for (l=0;l<nl;l++) bt[l*16+j]=column[l]; }
    for (j=nj;j<16;j++) for (l=0;l<nl;l++) bt[l*16+j]=0.0f;
  } else {
    __m256i m0=kb256_mask(nj>=8 ? 8 : nj),m1=kb256_mask(nj>8 ? nj-8 : 0);
    for (l=0;l<nl;l++) {
      const float *row=x->b+(lb+l)*x->ldb+jb;
      _mm256_store_ps(bt+l*16,_mm256_maskload_ps(row,m0));
      _mm256_store_ps(bt+l*16+8,_mm256_maskload_ps(row+8,m1));
    }
  }
}
__attribute__((target("avx2,fma"))) static void mm_avx2_kb(const mm_args *x, const mm_range *r) {
  _Alignas(32) float bt[BCIR_MM_KBLOCK*16]; size_t i,jb,lb,nl;
  for (jb=r->jb0;jb<r->jb1;jb+=16) {
    size_t nj=x->n-jb<16 ? x->n-jb : 16;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      kb256_pack(x,bt,jb,nj,lb,nl);
      for (i=r->r0;i+6<=r->r1;i+=6) kb256_block(6,x,bt,i,jb,nj,lb,nl);
      for (;i+2<=r->r1;i+=2) kb256_block(2,x,bt,i,jb,nj,lb,nl);
      for (;i<r->r1;i++) kb256_block(1,x,bt,i,jb,nj,lb,nl);
    }
  }
}
#endif
/* The column width of a variant's register block under each contract: a task's column range
 * is whole blocks. */
static size_t mm_nr(int variant, int fused) {
  if (fused==MM_KBLOCK) return variant==4 ? 48 : variant==2 ? 16 : 32;
  return variant==4 ? 64 : 32;
}
int bcir_tensor_mm_variants(void) {
  int v=1;
#ifdef BCIR_MM_DISPATCH
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) v|=2;
  if (BCIR_TENSOR_MAX_VARIANT>=4 && __builtin_cpu_supports("avx512f")) v|=4;
#endif
  return v;
}
int bcir_tensor_mm_fused_variants(void) {
  int v=1;
#ifdef BCIR_MM_DISPATCH
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) v|=2;
  if (BCIR_TENSOR_MAX_VARIANT>=4 && __builtin_cpu_supports("avx512f")) v|=4;
#endif
  return v;
}
static void mm_run_range(int variant, int fused, const mm_args *x, const mm_range *r) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) {
    if (fused==MM_KBLOCK) mm_avx512_kb(x,r); else if (fused) mm_avx512_fused(x,r); else mm_avx512(x,r);
    return;
  }
  if (variant==2) {
    if (fused==MM_KBLOCK) mm_avx2_kb(x,r); else if (fused) mm_avx2_fused(x,r); else mm_avx2(x,r);
    return;
  }
#endif
  (void)variant;
  if (fused==MM_KBLOCK) mm_generic_kb(x,r); else if (fused) mm_generic_fused(x,r); else mm_generic(x,r);
}
static void mm_run(int variant, const mm_args *x) {
  mm_range r;
  if (!x->m || !x->n || !x->k) return;
  r.jb0=0; r.jb1=x->n; r.r0=0; r.r1=x->m;
  mm_run_range(variant,MM_V1,x,&r);
}
static int mm_widest(void) {
  int v=bcir_tensor_mm_variants();
  return v&4 ? 4 : v&2 ? 2 : 1;
}
static int mm_widest_fused(void) {
  int v=bcir_tensor_mm_fused_variants();
  return v&4 ? 4 : v&2 ? 2 : 1;
}
/* --- the BCIR exponential (NDT-EXP) ------------------------------------------------------------
 * Branch-free: the argument is clamped into the range the reduction handles, a maximum and then a
 * minimum in the comparison order of the vector instructions (`x > lo ? x : lo`), so a NaN clamps to
 * the low end and no lane converts a NaN to an integer, which C leaves undefined (a UBSan build of the
 * trainer's harness found the nested selects passing it through); every other argument clamps as
 * before. Every lane runs the same operations, and the three special cases are selected at the end,
 * the NaN among them. LN2_HI has 16
 * significant bits, so k*LN2_HI is exact for |k| < 256; 2^k is applied as two exact power-of-two
 * scalings (|k| <= 151, so each half is in range): a normal result is exact under both, a
 * subnormal one is rounded once, by the second. The instances below inline this body under
 * their target, and the build's -ffp-contract=off keeps every multiply and add its own. */
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
  y=x>88.72283172607421875f ? exp_bits(0x7f800000u) : y;   /* above ln(FLT_MAX): +inf */
  y=x<-103.972084045410156f ? 0.0f : y;                     /* below the last subnormal: 0 */
  return x==x ? y : x;                                      /* NaN stays NaN */
}
/* the v2 sigmoid: one exponential of -|x|, the two halves selected */
BCIR_MM_INLINE float sigmoid2(float x) {
  float e=exp1(x<0.0f ? x : -x),s=1.0f/(1.0f+e);
  return x>=0.0f ? s : e*s;
}
BCIR_MM_INLINE void exp_body(size_t n, const float *x, float *y) { size_t i; for (i=0;i<n;i++) y[i]=exp1(x[i]); }
/* The same exponential over a row in place (the softmax's rows): one pointer, so the loop
 * vectorizes without an alias check, and exp1 per element as exp_body applies it. */
BCIR_MM_INLINE void exp_inplace_body(size_t n, float *x) { size_t i; for (i=0;i<n;i++) x[i]=exp1(x[i]); }
BCIR_MM_INLINE void silu2_body(size_t n, const float *x, float *y) { size_t i; for (i=0;i<n;i++) y[i]=x[i]*sigmoid2(x[i]); }
BCIR_MM_INLINE void swiglu2_body(size_t n, const float *g, const float *u, const float *dy,
    float *dg, float *du) {
  size_t i;
  for (i=0;i<n;i++) {
    float s=sigmoid2(g[i]);
    dg[i]=dy[i]*u[i]*(s+g[i]*s*(1.0f-s)); du[i]=dy[i]*g[i]*s;
  }
}
static void exp_generic(size_t n, const float *x, float *y) { exp_body(n,x,y); }
static void exp_inplace_generic(size_t n, float *x) { exp_inplace_body(n,x); }
static void silu2_generic(size_t n, const float *x, float *y) { silu2_body(n,x,y); }
static void swiglu2_generic(size_t n, const float *g, const float *u, const float *dy, float *dg,
    float *du) { swiglu2_body(n,g,u,dy,dg,du); }
#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx2"))) static void exp_avx2(size_t n, const float *x, float *y) { exp_body(n,x,y); }
__attribute__((target("avx512f"))) static void exp_avx512(size_t n, const float *x, float *y) { exp_body(n,x,y); }
__attribute__((target("avx2"))) static void exp_inplace_avx2(size_t n, float *x) { exp_inplace_body(n,x); }
__attribute__((target("avx512f"))) static void exp_inplace_avx512(size_t n, float *x) { exp_inplace_body(n,x); }
__attribute__((target("avx2"))) static void silu2_avx2(size_t n, const float *x, float *y) { silu2_body(n,x,y); }
__attribute__((target("avx512f"))) static void silu2_avx512(size_t n, const float *x, float *y) { silu2_body(n,x,y); }
__attribute__((target("avx2"))) static void swiglu2_avx2(size_t n, const float *g, const float *u,
    const float *dy, float *dg, float *du) { swiglu2_body(n,g,u,dy,dg,du); }
__attribute__((target("avx512f"))) static void swiglu2_avx512(size_t n, const float *g, const float *u,
    const float *dy, float *dg, float *du) { swiglu2_body(n,g,u,dy,dg,du); }
#endif
static void exp_run(int variant, size_t n, const float *x, float *y) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { exp_avx512(n,x,y); return; }
  if (variant==2) { exp_avx2(n,x,y); return; }
#endif
  (void)variant; exp_generic(n,x,y);
}
static void exp_inplace_run(int variant, size_t n, float *x) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { exp_inplace_avx512(n,x); return; }
  if (variant==2) { exp_inplace_avx2(n,x); return; }
#endif
  (void)variant; exp_inplace_generic(n,x);
}
static void silu2_run(int variant, size_t n, const float *x, float *y) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { silu2_avx512(n,x,y); return; }
  if (variant==2) { silu2_avx2(n,x,y); return; }
#endif
  (void)variant; silu2_generic(n,x,y);
}
static void swiglu2_run(int variant, size_t n, const float *g, const float *u, const float *dy,
    float *dg, float *du) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { swiglu2_avx512(n,g,u,dy,dg,du); return; }
  if (variant==2) { swiglu2_avx2(n,g,u,dy,dg,du); return; }
#endif
  (void)variant; swiglu2_generic(n,g,u,dy,dg,du);
}
float bcir_tensor_expf(float x) { return exp1(x); }
void bcir_tensor_exp(size_t n, const float *x, float *y) { exp_run(mm_widest(),n,x,y); }
int bcir_tensor_exp_variant(int variant, size_t n, const float *x, float *y) {
  if ((variant!=1 && variant!=2 && variant!=4) || !(bcir_tensor_mm_variants()&variant)) return 0;
  exp_run(variant,n,x,y); return variant;
}
size_t bcir_tensor_exp_sweep(unsigned stride, size_t *mismatch, unsigned *ulp) {
  static float xs[4096],ys[3][4096]; size_t n=0,swept=0,bad=0,i; unsigned worst=0,bits;
  int mask=bcir_tensor_mm_variants(),variants[3],count=0,v;
  if (!stride || (stride&(stride-1))) { *mismatch=0; *ulp=0; return 0; }
  for (v=1;v<=4;v*=2) if (mask&v) variants[count++]=v;
  for (bits=0;;bits+=stride) {
    uint32_t pos=bits&0x7fffffffu; float xp=exp_bits(pos),xn=exp_bits(pos|0x80000000u);
    if (xp<=89.0f) xs[n++]=xp;
    if (xn>=-104.0f) xs[n++]=xn;
    if (n>=4094 || bits>=0x7fffffffu-stride) {
      for (v=0;v<count;v++) exp_run(variants[v],n,xs,ys[v]);
      for (i=0;i<n;i++) {
        float want=(float)exp((double)xs[i]); uint32_t wb,yb; int differ=0;
        for (v=1;v<count;v++) differ|=memcmp(&ys[0][i],&ys[v][i],4)!=0;
        bad+=differ!=0;
        memcpy(&wb,&want,4); memcpy(&yb,&ys[0][i],4);
        if (wb!=yb && wb!=0x7f800000u && yb!=0x7f800000u) {
          unsigned d=wb>yb ? wb-yb : yb-wb;
          if (d>worst) worst=d;
        } else if ((wb==0x7f800000u)!=(yb==0x7f800000u)) worst=0xffffffffu;
      }
      swept+=n; n=0;
      if (bits>=0x7fffffffu-stride) break;
    }
  }
  *mismatch=bad; *ulp=worst; return swept;
}
static int mm_variant_any(int variant, int fused, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  mm_args x; mm_range r; size_t i; int fold=fused==MM_KBLOCK && m && n && k;
  int have=fused ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
  if ((variant!=1 && variant!=2 && variant!=4) || !(have&variant)) return 0;
  if (!fold) for (i=0;i<m*n;i++) c[i]=mm_scaled(beta,c[i]);
  if (!m || !n || !k) return variant;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.lda=ta ? m : k;
  x.b=b; x.ldb=tb ? k : n; x.c=c; x.ldc=n; x.mode=MM_FULL; x.beta=beta; x.fold=fold;
  r.jb0=0; r.jb1=n; r.r0=0; r.r1=m;
  mm_run_range(variant,fused,&x,&r);
  return variant;
}
int bcir_tensor_mm_variant(int variant, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  return mm_variant_any(variant,MM_V1,ta,tb,m,n,k,alpha,a,b,beta,c);
}
int bcir_tensor_mm_fused_variant(int variant, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  return mm_variant_any(variant,MM_FUSED,ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  (void)ctx;
  mm_variant_any(mm_widest(),MM_V1,ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_fused(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  (void)ctx;
  mm_variant_any(mm_widest_fused(),MM_FUSED,ta,tb,m,n,k,alpha,a,b,beta,c);
}
int bcir_tensor_mm_kblock_variants(void) { return bcir_tensor_mm_fused_variants(); }
int bcir_tensor_mm_kblock_variant(int variant, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  return mm_variant_any(variant,MM_KBLOCK,ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_kblock(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  (void)ctx;
  mm_variant_any(mm_widest_fused(),MM_KBLOCK,ta,tb,m,n,k,alpha,a,b,beta,c);
}
/* --- the deterministic pool (NDT-THREADS, NDT-POOL) --------------------------------------------
 * Thread 0 is the caller; the others wait for a generation, run tasks of it and report. Where the
 * compiler has atomics (GCC's and Clang's builtins) every thread claims its tasks one at a time
 * from the generation's counter, in ascending order, so a thread that wakes late or is preempted
 * takes fewer and none waits on another's fixed share; elsewhere task t goes to thread
 * t % threads. Either way every task runs once and no task is taken by two threads. Every task
 * writes outputs no other task writes and reads nothing a task of the same job writes, so which
 * thread runs a task, and when, does not reach the arithmetic; the partitions below make it so.
 * With atomics the hot path takes no lock: the caller publishes a job with the generation's
 * release store and a worker reports with one atomic decrement. A worker waits for the next
 * generation BCIR_POOL_SPINS rounds on the processor, then BCIR_POOL_YIELDS rounds yielding it,
 * then asleep on the condition variable; the caller waits for the reports the same way. A job
 * that follows another closely starts without a wake-up, and on a host with fewer free
 * processors than threads the yields hand the processor over. The reports stay the barrier --
 * every worker reports every generation -- so no thread touches a job after its caller returns.
 * The pool, its thread handles and its worker records live in one block the caller provides
 * (bcir_tensor_pool_size): the unit allocates nothing, as the trainer's arenas allocate nothing. */
#if defined(__GNUC__) || defined(__clang__)
#define BCIR_POOL_ATOMIC 1
#define POOL_LOAD(x) __atomic_load_n(&(x),__ATOMIC_ACQUIRE)
#define POOL_STORE(x,v) __atomic_store_n(&(x),(v),__ATOMIC_RELEASE)
#if defined(__x86_64__) || defined(__i386__)
#define POOL_RELAX() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define POOL_RELAX() __asm__ __volatile__("yield")
#else
#define POOL_RELAX() ((void)0)
#endif
#ifndef BCIR_POOL_SPINS
#define BCIR_POOL_SPINS 64
#endif
#ifndef BCIR_POOL_YIELDS
#define BCIR_POOL_YIELDS 512
#endif
#endif
typedef struct pool_worker { bcir_tensor_pool *pool; unsigned id; } pool_worker;
struct bcir_tensor_pool {
  unsigned threads,pending; unsigned long generation; int stop;
  bcir_tensor_task_fn fn; void *ctx; size_t tasks,next;
#if defined(BCIR_POOL_PTHREADS)
  pthread_mutex_t mutex; pthread_cond_t wake,done; pthread_t *workers;
#elif defined(BCIR_POOL_WIN32)
  CRITICAL_SECTION mutex; CONDITION_VARIABLE wake,done; HANDLE *workers;
#endif
  pool_worker *records;
};
static size_t pool_align_up(size_t n, size_t a) { return (n+a-1)/a*a; }
/* The block: the pool, then the thread handles, then the worker records, each at its alignment. */
static size_t pool_layout(unsigned threads, size_t *handles, size_t *records) {
  size_t off=sizeof(bcir_tensor_pool),n=threads-1;
#if defined(BCIR_POOL_PTHREADS)
  off=pool_align_up(off,_Alignof(pthread_t)); *handles=off; off+=n*sizeof(pthread_t);
#elif defined(BCIR_POOL_WIN32)
  off=pool_align_up(off,_Alignof(HANDLE)); *handles=off; off+=n*sizeof(HANDLE);
#else
  *handles=off;
#endif
  off=pool_align_up(off,_Alignof(pool_worker)); *records=off; off+=n*sizeof(pool_worker);
  return pool_align_up(off,_Alignof(max_align_t));
}
size_t bcir_tensor_pool_size(unsigned threads) {
  size_t h,r;
  if (!threads || threads>4096) return 0;
#if !defined(BCIR_POOL_PTHREADS) && !defined(BCIR_POOL_WIN32)
  if (threads>1) return 0;
#endif
  return pool_layout(threads,&h,&r);
}
int bcir_tensor_pool_rails(void) {
#if defined(BCIR_POOL_PTHREADS)
  return 1;
#elif defined(BCIR_POOL_WIN32)
  return 2;
#else
  return 0;
#endif
}
unsigned bcir_tensor_pool_threads(const bcir_tensor_pool *pool) { return pool ? pool->threads : 1; }
static void pool_tasks(bcir_tensor_pool *p, unsigned id) {
  size_t t;
#if defined(BCIR_POOL_ATOMIC)
  bcir_tensor_task_fn fn=p->fn; void *ctx=p->ctx; size_t tasks=p->tasks;
  (void)id;
  while ((t=__atomic_fetch_add(&p->next,(size_t)1,__ATOMIC_RELAXED))<tasks) fn(ctx,t);
#else
  for (t=id;t<p->tasks;t+=p->threads) p->fn(p->ctx,t);
#endif
}
#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
#if defined(BCIR_POOL_PTHREADS)
#define POOL_LOCK(p) pthread_mutex_lock(&(p)->mutex)
#define POOL_UNLOCK(p) pthread_mutex_unlock(&(p)->mutex)
#define POOL_WAIT(p,c) pthread_cond_wait(&(p)->c,&(p)->mutex)
#define POOL_SIGNAL(p,c) pthread_cond_signal(&(p)->c)
#define POOL_BROADCAST(p,c) pthread_cond_broadcast(&(p)->c)
#define POOL_YIELD() sched_yield()
static void *pool_main(void *arg)
#else
#define POOL_LOCK(p) EnterCriticalSection(&(p)->mutex)
#define POOL_UNLOCK(p) LeaveCriticalSection(&(p)->mutex)
#define POOL_WAIT(p,c) SleepConditionVariableCS(&(p)->c,&(p)->mutex,INFINITE)
#define POOL_SIGNAL(p,c) WakeConditionVariable(&(p)->c)
#define POOL_BROADCAST(p,c) WakeAllConditionVariable(&(p)->c)
#define POOL_YIELD() SwitchToThread()
static DWORD WINAPI pool_main(LPVOID arg)
#endif
{
  pool_worker *w=(pool_worker *)arg; bcir_tensor_pool *p=w->pool; unsigned id=w->id;
  unsigned long seen=0;
  for (;;) {
#if defined(BCIR_POOL_ATOMIC)
    unsigned spin=0;
    while (POOL_LOAD(p->generation)==seen && !POOL_LOAD(p->stop)) {
      if (spin<BCIR_POOL_SPINS) POOL_RELAX();
      else if (spin<BCIR_POOL_SPINS+BCIR_POOL_YIELDS) POOL_YIELD();
      else {
        POOL_LOCK(p);
        while (POOL_LOAD(p->generation)==seen && !POOL_LOAD(p->stop)) POOL_WAIT(p,wake);
        POOL_UNLOCK(p);
      }
      spin++;
    }
    if (POOL_LOAD(p->stop)) break;
    seen=POOL_LOAD(p->generation);
    pool_tasks(p,id);
    if (__atomic_sub_fetch(&p->pending,1u,__ATOMIC_RELEASE)==0) {
      POOL_LOCK(p); POOL_SIGNAL(p,done); POOL_UNLOCK(p);
    }
#else
    POOL_LOCK(p);
    while (p->generation==seen && !p->stop) POOL_WAIT(p,wake);
    if (p->stop) { POOL_UNLOCK(p); break; }
    seen=p->generation;
    POOL_UNLOCK(p);
    pool_tasks(p,id);
    POOL_LOCK(p);
    if (--p->pending==0) POOL_SIGNAL(p,done);
    POOL_UNLOCK(p);
#endif
  }
  return 0;
}
/* Under the lock: stop, seen by a sleeping worker when it wakes and by a waiting one at once. */
static void pool_signal_stop(bcir_tensor_pool *p) {
#if defined(BCIR_POOL_ATOMIC)
  POOL_STORE(p->stop,1);
#else
  p->stop=1;
#endif
  POOL_BROADCAST(p,wake);
}
static void pool_sync_destroy(bcir_tensor_pool *p) {
#if defined(BCIR_POOL_PTHREADS)
  pthread_cond_destroy(&p->done); pthread_cond_destroy(&p->wake); pthread_mutex_destroy(&p->mutex);
#else
  DeleteCriticalSection(&p->mutex);
#endif
}
static int pool_start(bcir_tensor_pool *p) {
  unsigned i;
#if defined(BCIR_POOL_PTHREADS)
  if (pthread_mutex_init(&p->mutex,NULL)) return 0;
  if (pthread_cond_init(&p->wake,NULL)) { pthread_mutex_destroy(&p->mutex); return 0; }
  if (pthread_cond_init(&p->done,NULL)) {
    pthread_cond_destroy(&p->wake); pthread_mutex_destroy(&p->mutex); return 0;
  }
#else
  InitializeCriticalSection(&p->mutex); InitializeConditionVariable(&p->wake);
  InitializeConditionVariable(&p->done);
#endif
  for (i=1;i<p->threads;i++) {
    pool_worker *w=&p->records[i-1];
    w->pool=p; w->id=i;
#if defined(BCIR_POOL_PTHREADS)
    if (pthread_create(&p->workers[i-1],NULL,pool_main,w)) break;
#else
    p->workers[i-1]=CreateThread(NULL,0,pool_main,w,0,NULL);
    if (!p->workers[i-1]) break;
#endif
  }
  if (i<p->threads) {
    /* a thread that could not start: stop the ones that did, release everything, refuse */
    unsigned started=i-1,j;
    POOL_LOCK(p); pool_signal_stop(p); POOL_UNLOCK(p);
    for (j=0;j<started;j++) {
#if defined(BCIR_POOL_PTHREADS)
      pthread_join(p->workers[j],NULL);
#else
      WaitForSingleObject(p->workers[j],INFINITE); CloseHandle(p->workers[j]);
#endif
    }
    pool_sync_destroy(p);
    return 0;
  }
  return 1;
}
static void pool_join(bcir_tensor_pool *p) {
  unsigned j;
  POOL_LOCK(p); pool_signal_stop(p); POOL_UNLOCK(p);
  for (j=0;j+1<p->threads;j++) {
#if defined(BCIR_POOL_PTHREADS)
    pthread_join(p->workers[j],NULL);
#else
    WaitForSingleObject(p->workers[j],INFINITE); CloseHandle(p->workers[j]);
#endif
  }
  pool_sync_destroy(p);
}
#endif
bcir_tensor_pool *bcir_tensor_pool_init(void *memory, size_t bytes, unsigned threads) {
  size_t need=bcir_tensor_pool_size(threads),h,r; bcir_tensor_pool *p;
  if (!need || !memory || bytes<need || (uintptr_t)memory%_Alignof(max_align_t)) return NULL;
  pool_layout(threads,&h,&r);
  memset(memory,0,need);
  p=(bcir_tensor_pool *)memory; p->threads=threads;
#if defined(BCIR_POOL_PTHREADS)
  p->workers=(pthread_t *)(void *)((unsigned char *)memory+h);
#elif defined(BCIR_POOL_WIN32)
  p->workers=(HANDLE *)(void *)((unsigned char *)memory+h);
#endif
  p->records=(pool_worker *)(void *)((unsigned char *)memory+r);
#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  if (threads>1 && !pool_start(p)) return NULL;
#endif
  return p;
}
/* After stop the block still reads as a pool of one, so a kernel handed it runs serially. */
void bcir_tensor_pool_stop(bcir_tensor_pool *p) {
  if (!p) return;
#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  if (p->threads>1) pool_join(p);
#endif
  p->threads=1;
}
void bcir_tensor_pool_run(bcir_tensor_pool *p, bcir_tensor_task_fn fn, void *ctx, size_t tasks) {
  size_t t;
  if (!tasks) return;
  if (!p || p->threads==1 || tasks==1) { for (t=0;t<tasks;t++) fn(ctx,t); return; }
#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  POOL_LOCK(p);
  p->fn=fn; p->ctx=ctx; p->tasks=tasks;
#if defined(BCIR_POOL_ATOMIC)
  __atomic_store_n(&p->pending,p->threads-1,__ATOMIC_RELAXED);
  __atomic_store_n(&p->next,(size_t)0,__ATOMIC_RELAXED);
  POOL_STORE(p->generation,p->generation+1);
  POOL_BROADCAST(p,wake);
  POOL_UNLOCK(p);
  pool_tasks(p,0);
  {
    unsigned spin;
    for (spin=0;spin<BCIR_POOL_SPINS+BCIR_POOL_YIELDS && POOL_LOAD(p->pending);spin++) {
      if (spin<BCIR_POOL_SPINS) POOL_RELAX(); else POOL_YIELD();
    }
  }
  if (POOL_LOAD(p->pending)) {
    POOL_LOCK(p);
    while (POOL_LOAD(p->pending)) POOL_WAIT(p,done);
    POOL_UNLOCK(p);
  }
#else
  p->pending=p->threads-1; p->generation++;
  POOL_BROADCAST(p,wake);
  POOL_UNLOCK(p);
  pool_tasks(p,0);
  POOL_LOCK(p);
  while (p->pending) POOL_WAIT(p,done);
  POOL_UNLOCK(p);
#endif
#else
  for (t=0;t<tasks;t++) fn(ctx,t);
#endif
}
#define pool_run bcir_tensor_pool_run
/* --- the partitions: each a function of the shapes alone ----------------------------------- */
/* GEMM: one task per (column block of the variant's width, row range of BCIR_MM_ROWS rows). The
 * task's elements of C are scaled by beta -- by the task under the v1 and fused contracts (and a
 * blocked call without terms), by the blocked kernel's first K block otherwise (NDT-BETA) -- then
 * the serial nest runs over its block and rows, every depth chunk in order, so each element of C
 * receives its terms in the serial order; the task packs its own tiles. */
#ifndef BCIR_MM_ROWS
#define BCIR_MM_ROWS 128
#endif
typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;
static void mm_task(void *ctx, size_t task) {
  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;
  r.jb0=col*j->nr; r.jb1=r.jb0+j->nr<j->x.n ? r.jb0+j->nr : j->x.n;
  r.r0=row*BCIR_MM_ROWS; r.r1=r.r0+BCIR_MM_ROWS<j->x.m ? r.r0+BCIR_MM_ROWS : j->x.m;
  if (!j->x.fold) for (i=r.r0;i<r.r1;i++) {
    float *ci=j->x.c+i*j->x.ldc;
    for (jj=r.jb0;jj<r.jb1;jj++) ci[jj]=mm_scaled(j->beta,ci[jj]);
  }
  if (j->x.k) mm_run_range(j->variant,j->fused,&j->x,&r);
}
/* The job of one GEMM under a contract: its partition's task count (0 for an empty product). */
static size_t mm_job_init(mm_job *j, int fused, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  if (!m || !n) return 0;
  j->beta=beta; j->fused=fused; j->variant=fused ? mm_widest_fused() : mm_widest(); j->nr=mm_nr(j->variant,fused);
  j->x.ta=ta; j->x.tb=tb; j->x.m=m; j->x.n=n; j->x.k=k; j->x.alpha=alpha; j->x.a=a; j->x.lda=ta ? m : k;
  j->x.b=b; j->x.ldb=tb ? k : n; j->x.c=c; j->x.ldc=n; j->x.mode=MM_FULL;
  j->x.beta=beta; j->x.fold=fused==MM_KBLOCK && k;
  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;
  return j->cols*j->rowsets;
}
static void mm_serial_any(int fused, int ta, int tb, size_t m, size_t n, size_t k, float alpha,
    const float *a, const float *b, float beta, float *c) {
  (fused==MM_KBLOCK ? bcir_tensor_mm_kblock : fused ? bcir_tensor_mm_fused : bcir_tensor_mm)
    (NULL,ta,tb,m,n,k,alpha,a,b,beta,c);
}
static void mm_pool_any(bcir_tensor_pool *pool, int fused, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_job j; size_t tasks;
  if (!pool || pool->threads==1) { mm_serial_any(fused,ta,tb,m,n,k,alpha,a,b,beta,c); return; }
  tasks=mm_job_init(&j,fused,ta,tb,m,n,k,alpha,a,b,beta,c);
  pool_run(pool,mm_task,&j,tasks);
}
/* Two GEMMs, one job (NDT-SCHED-0): tasks [0, split) are the first's, the rest the second's,
 * each with its own partition, so the thread that runs task t runs the same work it would have
 * run for that GEMM alone. */
typedef struct mm_pair_job { mm_job a,b; size_t split; } mm_pair_job;
static void mm_pair_task(void *ctx, size_t task) {
  mm_pair_job *j=(mm_pair_job *)ctx;
  if (task<j->split) mm_task(&j->a,task); else mm_task(&j->b,task-j->split);
}
void bcir_tensor_mm_pair_pool(bcir_tensor_pool *pool, int contract, const bcir_tensor_mm_call *x,
    const bcir_tensor_mm_call *y) {
  mm_pair_job j; size_t second;
  int fused=contract==2 ? MM_KBLOCK : contract==1 ? MM_FUSED : MM_V1;
  if (!pool || pool->threads==1) {
    mm_serial_any(fused,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
    mm_serial_any(fused,y->ta,y->tb,y->m,y->n,y->k,y->alpha,y->a,y->b,y->beta,y->c);
    return;
  }
  j.split=mm_job_init(&j.a,fused,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
  second=mm_job_init(&j.b,fused,y->ta,y->tb,y->m,y->n,y->k,y->alpha,y->a,y->b,y->beta,y->c);
  pool_run(pool,mm_pair_task,&j,j.split+second);
}
/* A chain into one C and independent GEMMs as one job (NDT-SCHED-1): tasks [0, split) are the
 * chain's tiles -- a tile's task runs each chained call's task for that tile in turn, which
 * scales the tile by the call's beta and adds its terms, the order the calls would run whole --
 * and the rest the independent calls' tasks, each its own partition. Every chained call has the
 * first's m, n and contract, so one task index names one tile in all of them. */
typedef struct mm_group_job {
  mm_job x[BCIR_TENSOR_MM_GROUP],y[BCIR_TENSOR_MM_GROUP];
  size_t nx,split,start[BCIR_TENSOR_MM_GROUP+1];
} mm_group_job;
static void mm_group_task(void *ctx, size_t task) {
  mm_group_job *j=(mm_group_job *)ctx; size_t q;
  if (task<j->split) { for (q=0;q<j->nx;q++) mm_task(&j->x[q],task); return; }
  task-=j->split;
  for (q=0;task>=j->start[q+1];q++) {}
  mm_task(&j->y[q],task-j->start[q]);
}
static void mm_call_serial(int fused, const bcir_tensor_mm_call *x) {
  mm_serial_any(fused,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
}
static size_t mm_call_init(mm_job *j, int fused, const bcir_tensor_mm_call *x) {
  return mm_job_init(j,fused,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
}
void bcir_tensor_mm_group_pool(bcir_tensor_pool *pool, int contract, const bcir_tensor_mm_call *x,
    size_t nx, const bcir_tensor_mm_call *y, size_t ny) {
  mm_group_job j; size_t q;
  int fused=contract==2 ? MM_KBLOCK : contract==1 ? MM_FUSED : MM_V1;
  int serial=!pool || pool->threads==1 || nx>BCIR_TENSOR_MM_GROUP || ny>BCIR_TENSOR_MM_GROUP;
  for (q=1;q<nx && !serial;q++) serial=x[q].c!=x[0].c || x[q].m!=x[0].m || x[q].n!=x[0].n;
  if (serial) {
    for (q=0;q<nx;q++) mm_call_serial(fused,x+q);
    for (q=0;q<ny;q++) mm_call_serial(fused,y+q);
    return;
  }
  j.nx=nx; j.split=0; j.start[0]=0;
  for (q=0;q<nx;q++) { size_t tasks=mm_call_init(&j.x[q],fused,x+q); if (!q) j.split=tasks; }
  for (q=0;q<ny;q++) j.start[q+1]=j.start[q]+mm_call_init(&j.y[q],fused,y+q);
  pool_run(pool,mm_group_task,&j,j.split+j.start[ny]);
}
void bcir_tensor_mm_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_V1,ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_fused_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_FUSED,ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_kblock_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_KBLOCK,ta,tb,m,n,k,alpha,a,b,beta,c);
}
/* Elements in ranges of BCIR_EW_CHUNK: y[i] depends on x[i] alone. */
#define BCIR_EW_CHUNK 8192
typedef struct ew_job { size_t n; const float *x,*u,*dy; float *y,*du; int op,variant; } ew_job;
enum { EW_SILU, EW_ADD, EW_MUL, EW_SWIGLU, EW_SILU2, EW_SWIGLU2 };
static void ew_task(void *ctx, size_t task) {
  ew_job *j=(ew_job *)ctx; size_t i0=task*BCIR_EW_CHUNK,i1=i0+BCIR_EW_CHUNK<j->n ? i0+BCIR_EW_CHUNK : j->n;
  switch (j->op) {
    case EW_SILU: bcir_tensor_silu(NULL,i1-i0,j->x+i0,j->y+i0); break;
    case EW_ADD: { size_t i; for (i=i0;i<i1;i++) j->y[i]+=j->x[i]; } break;
    case EW_MUL: { size_t i; for (i=i0;i<i1;i++) j->y[i]*=j->x[i]; } break;
    case EW_SILU2: silu2_run(j->variant,i1-i0,j->x+i0,j->y+i0); break;
    case EW_SWIGLU2: swiglu2_run(j->variant,i1-i0,j->x+i0,j->u+i0,j->dy+i0,j->y+i0,j->du+i0); break;
    default: bcir_tensor_swiglu_backward(i1-i0,j->x+i0,j->u+i0,j->dy+i0,j->y+i0,j->du+i0); break;
  }
}
static void ew_run(bcir_tensor_pool *pool, ew_job *j) {
  j->variant=mm_widest();
  pool_run(pool,ew_task,j,(j->n+BCIR_EW_CHUNK-1)/BCIR_EW_CHUNK);
}
void bcir_tensor_silu_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=NULL; j.dy=NULL; j.du=NULL; j.op=EW_SILU; ew_run(pool,&j);
}
void bcir_tensor_add_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=NULL; j.dy=NULL; j.du=NULL; j.op=EW_ADD; ew_run(pool,&j);
}
void bcir_tensor_mul_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=NULL; j.dy=NULL; j.du=NULL; j.op=EW_MUL; ew_run(pool,&j);
}
void bcir_tensor_swiglu_backward_pool(bcir_tensor_pool *pool, size_t n, const float *g,
    const float *u, const float *dy, float *dg, float *du) {
  ew_job j; j.n=n; j.x=g; j.u=u; j.dy=dy; j.y=dg; j.du=du; j.op=EW_SWIGLU; ew_run(pool,&j);
}
void bcir_tensor_silu_v2_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=NULL; j.dy=NULL; j.du=NULL; j.op=EW_SILU2; ew_run(pool,&j);
}
void bcir_tensor_swiglu_backward_v2_pool(bcir_tensor_pool *pool, size_t n, const float *g,
    const float *u, const float *dy, float *dg, float *du) {
  ew_job j; j.n=n; j.x=g; j.u=u; j.dy=dy; j.y=dg; j.du=du; j.op=EW_SWIGLU2; ew_run(pool,&j);
}
/* Rows in ranges of BCIR_ROW_CHUNK: each row's statistics are its own. */
#define BCIR_ROW_CHUNK 16
#ifdef BCIR_MM_DISPATCH
/* A chunk of 16 rows' statistics as the 16 lanes of a vector (NDT-RMS): lane r runs row r's chain
 * -- the columns in order, one rounded product and one rounded sum per column -- so each row's
 * sum of squares (and, with dy, the VJP's dot) is the scalar loop's value bit for bit. Per 16
 * columns the rows' block is transposed in registers (the GEMM pack's 16 x 16 transpose); the
 * last width%16 columns continue each row's chain one row at a time. The scalar loops spent four
 * cycles per column on one dependent add; the lanes spend them on sixteen rows at once. */
__attribute__((target("avx512f"))) static void rms16_stats(size_t w, const float *x, const float *dy,
    const float *g, float *sum, float *dot) {
  _Alignas(64) float bx[256],bd[256];
  __m512 s=_mm512_setzero_ps(),d=_mm512_setzero_ps(); size_t c0,c,r,w16=w/16*16;
  for (c0=0;c0<w16;c0+=16) {
    kb512_transpose16(x+c0,w,bx,16);
    if (dy) kb512_transpose16(dy+c0,w,bd,16);
    for (c=0;c<16;c++) {
      __m512 xv=_mm512_load_ps(bx+c*16);
      s=_mm512_add_ps(s,_mm512_mul_ps(xv,xv));
      if (dy) d=_mm512_add_ps(d,_mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(bd+c*16),_mm512_set1_ps(g[c0+c])),xv));
    }
  }
  _mm512_storeu_ps(sum,s);
  if (dy) _mm512_storeu_ps(dot,d);
  for (r=0;r<16;r++) for (c=w16;c<w;c++) {
    sum[r]+=x[r*w+c]*x[r*w+c];
    if (dy) dot[r]+=dy[r*w+c]*g[c]*x[r*w+c];
  }
}
#endif
typedef struct rms_job { size_t rows,width; float eps; const float *x,*g; float *y; int variant; } rms_job;
static void rms_task(void *ctx, size_t task) {
  rms_job *j=(rms_job *)ctx; size_t r0=task*BCIR_ROW_CHUNK,r1=r0+BCIR_ROW_CHUNK<j->rows ? r0+BCIR_ROW_CHUNK : j->rows;
#ifdef BCIR_MM_DISPATCH
  if (j->variant==4 && r1-r0==16) {
    float sum[16]; size_t r,c,w=j->width;
    rms16_stats(w,j->x+r0*w,NULL,NULL,sum,NULL);
    for (r=0;r<16;r++) {
      const float *xr=j->x+(r0+r)*w; float *yr=j->y+(r0+r)*w,inv=1.0f/sqrtf(sum[r]/(float)w+j->eps);
      for (c=0;c<w;c++) yr[c]=xr[c]*inv*j->g[c];
    }
    return;
  }
#endif
  bcir_tensor_rms(r1-r0,j->width,j->eps,j->x+r0*j->width,j->g,j->y+r0*j->width);
}
void bcir_tensor_rms_pool(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, float *y) {
  rms_job j; j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.y=y; j.variant=mm_widest();
  pool_run(pool,rms_task,&j,(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
}
/* The RMSNorm VJP: rows in ranges of BCIR_ROW_CHUNK (each row's dx and 1/rms its own), then
 * columns in ranges of BCIR_COL_CHUNK, a column's gamma adjoint summed over the rows in row order
 * from the value it holds -- the serial kernel's operations, element for element. */
#define BCIR_COL_CHUNK 16
typedef struct rmsb_job {
  size_t rows,width; float eps; const float *x,*g,*dy; float *dx,*dg,*inv; int variant;
} rmsb_job;
static void rmsb_rows_task(void *ctx, size_t task) {
  rmsb_job *j=(rmsb_job *)ctx; size_t r,c,w=j->width;
  size_t r0=task*BCIR_ROW_CHUNK,r1=r0+BCIR_ROW_CHUNK<j->rows ? r0+BCIR_ROW_CHUNK : j->rows;
#ifdef BCIR_MM_DISPATCH
  if (j->variant==4 && r1-r0==16) {
    float sum[16],dot[16];
    rms16_stats(w,j->x+r0*w,j->dy+r0*w,j->g,sum,dot);
    for (r=0;r<16;r++) {
      const float *xr=j->x+(r0+r)*w,*dyr=j->dy+(r0+r)*w; float *dxr=j->dx+(r0+r)*w;
      float inv=1.0f/sqrtf(sum[r]/(float)w+j->eps),coeff=dot[r]*inv*inv*inv/(float)w;
      for (c=0;c<w;c++) dxr[c]=dyr[c]*j->g[c]*inv-xr[c]*coeff;
      j->inv[r0+r]=inv;
    }
    return;
  }
#endif
  for (r=r0;r<r1;r++) {
    const float *xr=j->x+r*w,*dyr=j->dy+r*w; float *dxr=j->dx+r*w;
    float sum=0.0f,dot=0.0f,inv,coeff;
    for (c=0;c<w;c++) { sum+=xr[c]*xr[c]; dot+=dyr[c]*j->g[c]*xr[c]; }
    inv=1.0f/sqrtf(sum/(float)w+j->eps); coeff=dot*inv*inv*inv/(float)w;
    for (c=0;c<w;c++) dxr[c]=dyr[c]*j->g[c]*inv-xr[c]*coeff;
    j->inv[r]=inv;
  }
}
/* A column chunk's sums live in the task's registers over the rows and reach dg once at the end:
 * the same rounded terms in the same row order, without a load and a store of dg per row. One
 * length n bounds the fill, the sums and the store (zero for a task past the width, which the
 * pool never runs: the analyzer's path in which c1 - c0 wraps). */
static void rmsb_cols_task(void *ctx, size_t task) {
  rmsb_job *j=(rmsb_job *)ctx; size_t r,c,w=j->width; float acc[BCIR_COL_CHUNK];
  size_t c0=task*BCIR_COL_CHUNK,c1=c0+BCIR_COL_CHUNK<w ? c0+BCIR_COL_CHUNK : w,n=c1>c0 ? c1-c0 : 0;
  for (c=0;c<n;c++) acc[c]=j->dg[c0+c];
  for (r=0;r<j->rows;r++) {
    const float *xr=j->x+r*w+c0,*dyr=j->dy+r*w+c0; float inv=j->inv[r];
    for (c=0;c<n;c++) acc[c]+=dyr[c]*xr[c]*inv;
  }
  for (c=0;c<n;c++) j->dg[c0+c]=acc[c];
}
void bcir_tensor_rms_backward_pool(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, const float *dy, float *dx, float *dg, float *inv) {
  rmsb_job j; j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.dy=dy; j.dx=dx; j.dg=dg; j.inv=inv;
  j.variant=mm_widest();
  pool_run(pool,rmsb_rows_task,&j,(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
  pool_run(pool,rmsb_cols_task,&j,(width+BCIR_COL_CHUNK-1)/BCIR_COL_CHUNK);
}
/* Positions in ranges of BCIR_ROW_CHUNK: a position's angles and rotations are its own. Each pair
 * (pos, j) takes angle = pos * base^(-2j/dim) in double and its cosine and sine rounded to float,
 * and every head's pair at that position the same rotation; the frequencies are computed once per
 * BCIR_ROPE_J pairs and range (a pow per pair and position before NDT-ROPE), each position's
 * angles once, and the rotation runs over a head's pairs contiguously -- every value the same. */
#define BCIR_ROPE_J 64
typedef struct rope_job { size_t b,t,heads,dim; double base; float *x; int inverse; } rope_job;
static void rope_rows(size_t b, size_t t, size_t heads, size_t dim, double base, float *x,
    int inverse, size_t p0, size_t p1) {
  size_t r,pos,h,j,j0,half=dim/2;
  for (j0=0;j0<half;j0+=BCIR_ROPE_J) {
    size_t nj=half-j0<BCIR_ROPE_J ? half-j0 : BCIR_ROPE_J; double f[BCIR_ROPE_J];
    for (j=0;j<nj;j++) f[j]=pow(base,-2.0*(double)(j0+j)/(double)dim);
    for (pos=p0;pos<p1;pos++) {
      float c[BCIR_ROPE_J],s[BCIR_ROPE_J];
      for (j=0;j<nj;j++) {
        double angle=(double)pos*f[j];
        c[j]=(float)cos(angle); s[j]=(float)sin(angle)*(inverse ? -1.0f : 1.0f);
      }
      for (r=0;r<b;r++) for (h=0;h<heads;h++) {
        float *lo=x+((r*t+pos)*heads+h)*dim+j0,*hi=lo+half;
        for (j=0;j<nj;j++) { float a=lo[j],z=hi[j]; lo[j]=a*c[j]-z*s[j]; hi[j]=z*c[j]+a*s[j]; }
      }
    }
  }
}
static void rope_task(void *ctx, size_t task) {
  rope_job *j=(rope_job *)ctx; size_t p0=task*BCIR_ROW_CHUNK,p1=p0+BCIR_ROW_CHUNK<j->t ? p0+BCIR_ROW_CHUNK : j->t;
  rope_rows(j->b,j->t,j->heads,j->dim,j->base,j->x,j->inverse,p0,p1);
}
void bcir_tensor_rope_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t heads, size_t dim,
    double base, float *x, int inverse) {
  rope_job j; j.b=b; j.t=t; j.heads=heads; j.dim=dim; j.base=base; j.x=x; j.inverse=inverse;
  pool_run(pool,rope_task,&j,(t+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
}
static bcir_tensor_mm_fn mm(const bcir_tensor_provider *p) {
  return p && p->mm ? p->mm : bcir_tensor_mm;
}
void bcir_tensor_linear(const bcir_tensor_provider *p, size_t r, size_t in,
    size_t out, const float *x, const float *w, float *y) {
  mm(p)(p ? p->ctx : NULL,0,1,r,out,in,1.0f,x,w,0.0f,y);
}
void bcir_tensor_linear_backward(const bcir_tensor_provider *p, size_t r,
    size_t in, size_t out, const float *x, const float *w, const float *dy,
    float *dx, float *dw, float beta) {
  mm(p)(p ? p->ctx : NULL,0,0,r,in,out,1.0f,dy,w,beta,dx);
  mm(p)(p ? p->ctx : NULL,1,0,out,in,r,1.0f,dy,x,1.0f,dw);
}
static float sigmoid(float x) {
  if (x>=0.0f) return 1.0f/(1.0f+expf(-x));
  { float e=expf(x); return e/(1.0f+e); }
}
void bcir_tensor_silu(void *ctx, size_t n, const float *x, float *y) {
  size_t i; (void)ctx;
  for (i=0;i<n;i++) y[i]=x[i]*sigmoid(x[i]);
}
void bcir_tensor_swiglu_backward(size_t n, const float *g, const float *u,
    const float *dy, float *dg, float *du) {
  size_t i;
  for (i=0;i<n;i++) {
    float s=sigmoid(g[i]);
    dg[i]=dy[i]*u[i]*(s+g[i]*s*(1.0f-s)); du[i]=dy[i]*g[i]*s;
  }
}
void bcir_tensor_rms(size_t rows, size_t width, float eps, const float *x,
    const float *g, float *y) {
  size_t r,j;
  for (r=0;r<rows;r++) {
    float sum=0.0f,inv;
    for (j=0;j<width;j++) sum+=x[r*width+j]*x[r*width+j];
    inv=1.0f/sqrtf(sum/(float)width+eps);
    for (j=0;j<width;j++) y[r*width+j]=x[r*width+j]*inv*g[j];
  }
}
void bcir_tensor_rms_backward(size_t rows, size_t width, float eps,
    const float *x, const float *g, const float *dy, float *dx, float *dg) {
  size_t r,j;
  for (r=0;r<rows;r++) {
    float sum=0.0f,dot=0.0f,inv,coeff;
    for (j=0;j<width;j++) {
      sum+=x[r*width+j]*x[r*width+j]; dot+=dy[r*width+j]*g[j]*x[r*width+j];
    }
    inv=1.0f/sqrtf(sum/(float)width+eps); coeff=dot*inv*inv*inv/(float)width;
    for (j=0;j<width;j++) {
      dx[r*width+j]=dy[r*width+j]*g[j]*inv-x[r*width+j]*coeff;
      dg[j]+=dy[r*width+j]*x[r*width+j]*inv;
    }
  }
}
void bcir_tensor_rope(size_t b, size_t t, size_t heads, size_t dim,
    double base, float *x, int inverse) {
  rope_rows(b,t,heads,dim,base,x,inverse,0,t);
}
/* Causal attention as two masked products per (batch, head) through the GEMM kernel, then the
 * reference loop's softmax: each score is the dot product accumulated from zero over the channels
 * in order and then scaled, each context row adds p_ij v_j for j ascending -- the operations the
 * reference loop (test_decoder_train.c keeps it) performs, in its order, so the result is
 * bit-identical; future scores stay exactly zero. Under the exp contract the softmax row goes
 * through the exponential's vector instance after the max-subtraction (the same exp1 per
 * element, the same sum order; the scalar loop cost the forward a quarter more than libm). On
 * AVX-512 a whole block of 16 rows runs its maxima and sums as the lanes of one vector, and under
 * the exp contract its exponentials in those transposed blocks (NDT-ATTN-4: att512_rowchain,
 * att512_expsum) -- each row's chain the loop's, in its column order. */
typedef struct att_job {
  size_t b,t,nh,nk,d,group; const float *q,*k,*v,*dy; float *p,*y,*dq,*dk,*dv; int variant,exp2;
} att_job;
#ifdef BCIR_MM_DISPATCH
/* The attention's masked products on AVX-512 (NDT-ATTN-2: the forward's scores and context;
 * NDT-ATTN-3: the backward's dv, dq and dk) as register blocks of eight rows, with mm_block's v1
 * arithmetic exactly -- an element starts at the value it holds (the zero its task filled, or the
 * sum a KV head's earlier query heads left in dk and dv) and takes one rounded product and one
 * rounded sum per term in the term order: a score its channels in order, stored where j <= i; a
 * context element, and dq's, its keys j <= i in order; dv's and dk's their terms i >= j in order.
 * The v1 blocks computed the scores' 64-wide column blocks whole above the diagonal and padded the
 * 48 channels to 64 with a mask test per term; these compute a 32-wide block of scores only from
 * its diagonal on and take each term only where the mask admits it. Whole blocks of eight rows
 * only: att_rows runs a ragged tail through the v1 kernel, the same arithmetic. */
#define BCIR_ATT_DC 128
__attribute__((target("avx512f"))) BCIR_MM_INLINE __mmask16 att512_lanes(size_t count) {
  return (__mmask16)(count>=16 ? 0xffffu : (1u<<count)-1u);
}
__attribute__((target("avx512f"))) static void att512_scores(size_t rows, size_t d,
    const float *q, size_t ldq, const float *k, size_t ldk, float *s, size_t lds) {
  _Alignas(64) float kt[BCIR_ATT_DC*32];
  size_t j0,c0,i0,r,c,j;
  for (j0=0;j0<rows;j0+=32) {
    size_t nj=rows-j0<32 ? rows-j0 : 32;
    __mmask16 m0=att512_lanes(nj),m1=att512_lanes(nj>16 ? nj-16 : 0);
    for (c0=0;c0<d;c0+=BCIR_ATT_DC) {
      size_t nc=d-c0<BCIR_ATT_DC ? d-c0 : BCIR_ATT_DC;
      for (c=0;c<nc;c++) for (j=0;j<32;j++) kt[c*32+j]=j<nj ? k[(j0+j)*ldk+c0+c] : 0.0f;
      for (i0=j0;i0<rows;i0+=8) {
        __m512 a0[8],a1[8];
        for (r=0;r<8;r++) {
          const float *sr=s+(i0+r)*lds+j0;
          a0[r]=_mm512_maskz_loadu_ps(m0,sr); a1[r]=_mm512_maskz_loadu_ps(m1,sr+16);
        }
        for (c=0;c<nc;c++) {
          __m512 b0=_mm512_load_ps(kt+c*32),b1=_mm512_load_ps(kt+c*32+16);
          for (r=0;r<8;r++) {
            __m512 qv=_mm512_set1_ps(q[(i0+r)*ldq+c0+c]);
            a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(qv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(qv,b1));
          }
        }
        for (r=0;r<8;r++) {
          size_t i=i0+r,keep=i-j0+1<nj ? i-j0+1 : nj;
          float *sr=s+i*lds+j0;
          _mm512_mask_storeu_ps(sr,att512_lanes(keep),a0[r]);
          _mm512_mask_storeu_ps(sr+16,att512_lanes(keep>16 ? keep-16 : 0),a1[r]);
        }
      }
    }
  }
}
__attribute__((target("avx512f"))) static void att512_context(size_t rows, size_t d,
    const float *p, size_t ldp, const float *v, size_t ldv, float *y, size_t ldy) {
  size_t c0,i0,r,j,dj;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (i0=0;i0<rows;i0+=8) {
      __m512 a0[8],a1[8],a2[8];
      for (r=0;r<8;r++) {
        const float *yr=y+(i0+r)*ldy+c0;
        a0[r]=_mm512_maskz_loadu_ps(m0,yr); a1[r]=_mm512_maskz_loadu_ps(m1,yr+16);
        a2[r]=_mm512_maskz_loadu_ps(m2,yr+32);
      }
      for (j=0;j<=i0;j++) {  /* every row of the block takes key j */
        const float *vj=v+j*ldv+c0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_maskz_loadu_ps(m1,vj+16),
            b2=_mm512_maskz_loadu_ps(m2,vj+32);
        for (r=0;r<8;r++) {
          __m512 pv=_mm512_set1_ps(p[(i0+r)*ldp+j]);
          a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(pv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(pv,b1));
          a2[r]=_mm512_add_ps(a2[r],_mm512_mul_ps(pv,b2));
        }
      }
      for (dj=1;dj<8;dj++) {  /* key i0+dj: the rows r >= dj */
        const float *vj=v+(i0+dj)*ldv+c0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_maskz_loadu_ps(m1,vj+16),
            b2=_mm512_maskz_loadu_ps(m2,vj+32);
        for (r=dj;r<8;r++) {
          __m512 pv=_mm512_set1_ps(p[(i0+r)*ldp+i0+dj]);
          a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(pv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(pv,b1));
          a2[r]=_mm512_add_ps(a2[r],_mm512_mul_ps(pv,b2));
        }
      }
      for (r=0;r<8;r++) {
        float *yr=y+(i0+r)*ldy+c0;
        _mm512_mask_storeu_ps(yr,m0,a0[r]); _mm512_mask_storeu_ps(yr+16,m1,a1[r]);
        _mm512_mask_storeu_ps(yr+32,m2,a2[r]);
      }
    }
  }
}
/* out_j += sum over i >= j, ascending, of m_ij x_i (the KGE products: dv from P and dY, dk from dS
 * and Q): blocks of eight output rows j0..j0+7, the channels 48 at a time; term i reaches the rows
 * j <= i -- the block's diagonal terms some rows, every later term all eight. */
__attribute__((target("avx512f"))) static void att512_transposed(size_t rows, size_t t, size_t d,
    const float *m, size_t ldm, const float *x, size_t ldx, float *out, size_t ldo) {
  size_t c0,j0,r,i,di;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (j0=0;j0<rows;j0+=8) {
      __m512 a0[8],a1[8],a2[8];
      for (r=0;r<8;r++) {
        const float *o=out+(j0+r)*ldo+c0;
        a0[r]=_mm512_maskz_loadu_ps(m0,o); a1[r]=_mm512_maskz_loadu_ps(m1,o+16);
        a2[r]=_mm512_maskz_loadu_ps(m2,o+32);
      }
      for (di=0;di<7;di++) {  /* term j0+di: the rows r <= di */
        const float *xi=x+(j0+di)*ldx+c0,*mi=m+(j0+di)*ldm+j0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_maskz_loadu_ps(m1,xi+16),
            b2=_mm512_maskz_loadu_ps(m2,xi+32);
        for (r=0;r<=di;r++) {
          __m512 mv=_mm512_set1_ps(mi[r]);
          a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(mv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(mv,b1));
          a2[r]=_mm512_add_ps(a2[r],_mm512_mul_ps(mv,b2));
        }
      }
      for (i=j0+7;i<t;i++) {  /* every row of the block takes term i */
        const float *xi=x+i*ldx+c0,*mi=m+i*ldm+j0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_maskz_loadu_ps(m1,xi+16),
            b2=_mm512_maskz_loadu_ps(m2,xi+32);
        for (r=0;r<8;r++) {
          __m512 mv=_mm512_set1_ps(mi[r]);
          a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(mv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(mv,b1));
          a2[r]=_mm512_add_ps(a2[r],_mm512_mul_ps(mv,b2));
        }
      }
      for (r=0;r<8;r++) {
        float *o=out+(j0+r)*ldo+c0;
        _mm512_mask_storeu_ps(o,m0,a0[r]); _mm512_mask_storeu_ps(o+16,m1,a1[r]);
        _mm512_mask_storeu_ps(o+32,m2,a2[r]);
      }
    }
  }
}
#endif
#ifdef BCIR_MM_DISPATCH
/* The attention's row chains for 16 rows at once (NDT-ATTN-4): the 16 rows' blocks of 16 columns
 * transposed in registers so lane r carries row r's chain -- the softmax's running maximum (the
 * scalar `x > max ? x : max`, which is VMAXPS's operand rule exactly) and its sum, and the
 * backward's dot -- over the columns in order, a lane taking a column only where the row has it;
 * every row's value is the scalar loop's, bit for bit. */
__attribute__((target("avx512f"))) static void att512_rowchain(size_t t, size_t i0, const float *p,
    float *out, int sum) {
  _Alignas(64) float blk[256]; __m512 v=sum ? _mm512_setzero_ps() : _mm512_set1_ps(-INFINITY);
  size_t jb,c;
  for (jb=0;jb<=i0;jb+=16) {
    kb512_transpose16(p+i0*t+jb,t,blk,16);
    for (c=0;c<16;c++) {
      __mmask16 k=(__mmask16)(jb<i0 ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_load_ps(blk+c*16);
      v=sum ? _mm512_mask_add_ps(v,k,v,x) : _mm512_mask_max_ps(v,k,x,v);
    }
  }
  _mm512_storeu_ps(out,v);
}
/* The v2 softmax's exponentials and sum for the same 16 rows, in the transposed blocks: each block
 * less its lanes' maxima, through the exponential's vector instance whole (no row's remainder runs
 * one element at a time), summed by lane where the row has the column, the rest zeroed, and
 * transposed back; each element is exp1(x - max) and each sum its row's chain in column order. */
__attribute__((target("avx512f"))) static void att512_expsum(size_t t, size_t i0, float *p,
    const float *maxv, float *sum) {
  _Alignas(64) float blk[256]; __m512 mx=_mm512_loadu_ps(maxv),s=_mm512_setzero_ps(); size_t jb,c;
  for (jb=0;jb<=i0;jb+=16) {
    kb512_transpose16(p+i0*t+jb,t,blk,16);
    for (c=0;c<16;c++) _mm512_store_ps(blk+c*16,_mm512_sub_ps(_mm512_load_ps(blk+c*16),mx));
    exp_inplace_avx512(256,blk);
    for (c=0;c<16;c++) {
      __mmask16 k=(__mmask16)(jb<i0 ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_maskz_mov_ps(k,_mm512_load_ps(blk+c*16));
      s=_mm512_mask_add_ps(s,k,s,x);
      _mm512_store_ps(blk+c*16,x);
    }
    kb512_transpose16(blk,16,p+i0*t+jb,t);
  }
  _mm512_storeu_ps(sum,s);
}
/* The backward's dot over one tile for 16 rows (rows i0+ig.. of the block, columns [jt, lim)):
 * the whole 16-column blocks as lanes, the rest of each row's columns after them in order. */
__attribute__((target("avx512f"))) static void att512_dot(size_t t, size_t i0, size_t ig, size_t jt,
    size_t lim, const float *ph, const float *tile, size_t ldt, float *dot) {
  _Alignas(64) float bp[256],bd[256]; __m512 v=_mm512_loadu_ps(dot+ig); size_t j0,c,r,j,row=i0+ig;
  for (j0=jt;j0+16<=lim && j0<row+16;j0+=16) {
    kb512_transpose16(ph+row*t+j0,t,bp,16);
    kb512_transpose16(tile+ig*ldt+(j0-jt),ldt,bd,16);
    for (c=0;c<16;c++) {
      size_t col=j0+c;
      __mmask16 k=(__mmask16)(col<=row ? 0xffffu : (0xffffu<<(col-row))&0xffffu);
      v=_mm512_mask_add_ps(v,k,v,_mm512_mul_ps(_mm512_load_ps(bp+c*16),_mm512_load_ps(bd+c*16)));
    }
  }
  _mm512_storeu_ps(dot+ig,v);
  for (r=0;r<16;r++) {
    const float *pi=ph+(row+r)*t; size_t end=row+r+1<lim ? row+r+1 : lim;
    for (j=j0;j<end;j++) dot[ig+r]+=pi[j]*tile[(ig+r)*ldt+j-jt];
  }
}
/* A KV head's values transposed once (NDT-ATTN-5): vt[c*t + j] = v[j][c], 16 x 16 blocks through the
 * pack's register transpose and the ragged edges copied; the backward's probability-adjoint tiles
 * of every query head of the KV head read their columns from it. */
__attribute__((target("avx512f"))) static void att512_panel(size_t t, size_t d, const float *v,
    size_t ldv, float *vt) {
  size_t j,c,tw=t/16*16,dw=d/16*16;
  for (j=0;j<tw;j+=16) for (c=0;c<dw;c+=16) kb512_transpose16(v+j*ldv+c,ldv,vt+c*t+j,t);
  for (j=0;j<t;j++) for (c=j<tw ? dw : 0;c<d;c++) vt[c*t+j]=v[j*ldv+c];
}
/* The probability-adjoint tile dp_ij = dy_i . v_j for rows i0..i0+rows-1 and columns jt..jt+cols-1
 * from the panel: blocks of eight rows by 32 columns, only those holding some j <= i (the others
 * are never read), each element from the zero the tile holds with one rounded product and one
 * rounded sum per channel in order -- the v1 kernel's arithmetic for these operands; rows past the
 * last whole eight one at a time, the same chain. */
__attribute__((target("avx512f"))) static void att512_dp(size_t rows, size_t i0, size_t jt,
    size_t cols, size_t d, const float *dy, size_t ldy, const float *vt, size_t t, float *tile,
    size_t ldt) {
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
        const float *vc=vt+c*t+jt+jb;
        __m512 b0=_mm512_maskz_loadu_ps(m0,vc),b1=_mm512_maskz_loadu_ps(m1,vc+16);
        for (r=0;r<8;r++) {
          __m512 qv=_mm512_set1_ps(dy[(i0+ig+r)*ldy+c]);
          a0[r]=_mm512_add_ps(a0[r],_mm512_mul_ps(qv,b0)); a1[r]=_mm512_add_ps(a1[r],_mm512_mul_ps(qv,b1));
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
        for (c=0;c<d;c++) acc=acc+dy[(i0+ig)*ldy+c]*vt[c*t+jt+jb+j];
        tile[ig*ldt+jb+j]=acc;
      }
    }
  }
}
#endif
enum { ATT_SCORES, ATT_CONTEXT, ATT_TRANSPOSED };
/* One of the attention's masked products over its t rows: the AVX-512 register blocks for the
 * whole blocks of eight, the rest (or every row, on another instance) through the v1 kernel. */
static void att_rows(int variant, const mm_args *x, size_t t, int which) {
  mm_range rest; size_t whole=0;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) {
    whole=t/8*8;
    if (which==ATT_SCORES) att512_scores(whole,x->k,x->a,x->lda,x->b,x->ldb,x->c,x->ldc);
    else if (which==ATT_CONTEXT) att512_context(whole,x->n,x->a,x->lda,x->b,x->ldb,x->c,x->ldc);
    else att512_transposed(whole,t,x->n,x->a,x->lda,x->b,x->ldb,x->c,x->ldc);
  }
#endif
  (void)which;
  if (whole==t || !x->m || !x->n || !x->k) return;
  rest.jb0=0; rest.jb1=x->n; rest.r0=whole; rest.r1=t;
  mm_run_range(variant,MM_V1,x,&rest);
}
/* One (batch, query head): its scores, softmax rows and context; nothing shared with another. */
static void att_head(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),i,j; float scale=1.0f/sqrtf((float)d);
  float *ph=a->p+(r*nh+h)*t*t; mm_args x;
  /* the products accumulate into what they write: this head's probabilities and its context
   * columns start at zero here, in the task that owns them (NDT-SCAN; the caller's fill was serial) */
  memset(ph,0,t*t*sizeof(float));
  for (i=0;i<t;i++) memset(a->y+((r*t+i)*nh+h)*d,0,d*sizeof(float));
  x.ta=0; x.tb=1; x.m=t; x.n=t; x.k=d; x.alpha=1.0f; x.a=a->q+(r*t*nh+h)*d; x.lda=nh*d;
  x.b=a->k+(r*t*nk+kh)*d; x.ldb=nk*d; x.c=ph; x.ldc=t; x.mode=MM_LOWER;
  att_rows(a->variant,&x,t,ATT_SCORES);
  i=0;
#ifdef BCIR_MM_DISPATCH
  if (a->variant==4) for (;i+16<=t;i+=16) {
    float maxv[16],sum[16]; size_t u;
    for (u=0;u<16;u++) { float *pi=ph+(i+u)*t; for (j=0;j<=i+u;j++) pi[j]*=scale; }
    att512_rowchain(t,i,ph,maxv,0);
    if (a->exp2) att512_expsum(t,i,ph,maxv,sum);
    else {
      for (u=0;u<16;u++) { float *pi=ph+(i+u)*t; for (j=0;j<=i+u;j++) pi[j]=expf(pi[j]-maxv[u]); }
      att512_rowchain(t,i,ph,sum,1);
    }
    for (u=0;u<16;u++) { float *pi=ph+(i+u)*t; for (j=0;j<=i+u;j++) pi[j]/=sum[u]; }
  }
#endif
  for (;i<t;i++) {
    float *pi=ph+i*t,maxv=-INFINITY,sum=0.0f;
    for (j=0;j<=i;j++) { pi[j]*=scale; if (pi[j]>maxv) maxv=pi[j]; }
    if (a->exp2) {  /* v2: the row through the vector instance, then the sum in index order */
      for (j=0;j<=i;j++) pi[j]-=maxv;
      exp_inplace_run(a->variant,i+1,pi);
      for (j=0;j<=i;j++) sum+=pi[j];
    } else for (j=0;j<=i;j++) { pi[j]=expf(pi[j]-maxv); sum+=pi[j]; }
    for (j=0;j<=i;j++) pi[j]/=sum;
  }
  x.tb=0; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=a->v+(r*t*nk+kh)*d; x.ldb=nk*d;
  x.c=a->y+(r*t*nh+h)*d; x.ldc=nh*d; x.mode=MM_KLE;
  att_rows(a->variant,&x,t,ATT_CONTEXT);
}
static void att_task(void *ctx, size_t task) {
  att_job *a=(att_job *)ctx; att_head(a,task/a->nh,task%a->nh);
}
static void attention_any(bcir_tensor_pool *pool, int exp2, size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, float *y) {
  att_job a;
  a.b=b; a.t=t; a.nh=nh; a.nk=nk; a.d=d; a.group=1; a.q=q; a.k=k; a.v=v; a.dy=NULL; a.p=p; a.y=y;
  a.dq=NULL; a.dk=NULL; a.dv=NULL; a.variant=mm_widest(); a.exp2=exp2;
  pool_run(pool,att_task,&a,b*nh);
}
void bcir_tensor_attention_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, float *y) {
  attention_any(pool,0,b,t,nh,nk,d,q,k,v,p,y);
}
void bcir_tensor_attention_v2_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, float *y) {
  attention_any(pool,1,b,t,nh,nk,d,q,k,v,p,y);
}
void bcir_tensor_attention(size_t b, size_t t, size_t nh, size_t nk, size_t d,
    const float *q, const float *k, const float *v, float *p, float *y) {
  bcir_tensor_attention_pool(NULL,b,t,nh,nk,d,q,k,v,p,y);
}
void bcir_tensor_attention_backward(size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, const float *p,
    const float *dy, float *dq, float *dk, float *dv) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(dq,0,b*t*nh*d*sizeof(float)); memset(dk,0,b*t*nk*d*sizeof(float));
  memset(dv,0,b*t*nk*d*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t;
    float dot=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f;
      for (c=0;c<d;c++) dp+=dy[qi+c]*v[kj+c];
      dot+=p[pi+j]*dp;
    }
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f,ds;
      for (c=0;c<d;c++) dp+=dy[qi+c]*v[kj+c];
      ds=p[pi+j]*(dp-dot)*scale;
      for (c=0;c<d;c++) {
        dq[qi+c]+=ds*k[kj+c]; dk[kj+c]+=ds*q[qi+c]; dv[kj+c]+=p[pi+j]*dy[qi+c];
      }
    }
  }
}
/* The reference backward above recomputes each probability adjoint dp_ij = dy_i . v_j twice, a
 * scalar dot product at a time. This one computes the same values through the GEMM kernel and
 * rewrites p in place with the score adjoint ds_ij = p_ij (dp_ij - dot_i) scale, which the two
 * remaining products then read; nothing beyond p and the outputs is written. Per (batch, head):
 * dv_j += p_ij dy_i (i >= j, before p is rewritten); then, a chunk of rows at a time, dot_i =
 * sum p_ij dp_ij in j order and ds_ij from a second evaluation of the same dp tiles, as the
 * reference does; then dq_i += ds_ij k_j (j <= i) and dk_j += ds_ij q_i (i >= j). Every
 * accumulator receives the reference's terms in the reference's order -- heads in order for the
 * shared GQA dk/dv, i ascending within a head -- so dq, dk and dv are bit-identical to it. On
 * AVX-512 the dp tiles are register blocks over the KV head's values transposed once per task
 * (NDT-ATTN-5: att512_panel, att512_dp), the GEMM kernel's v1 arithmetic for these operands. */
#define BCIR_ATT_ROWS 32
#define BCIR_ATT_COLS 64
/* The probability adjoint's tiles a row block keeps between its two passes (NDT-ATTN-1): its first
 * BCIR_ATT_KEEP columns, so the second pass reads the values the first computed instead of
 * evaluating the same product again; a longer context evaluates its further tiles twice, as every
 * tile was evaluated before. */
#define BCIR_ATT_KEEP 256
/* The largest KV head (d x t floats) whose values the backward transposes once into its task's
 * stack (NDT-ATTN-5), 64 KB; a longer context's tiles take the GEMM kernel as before. */
#define BCIR_ATT_PANEL 16384
/* One (batch, query head) of the backward; the heads of one KV head share dk and dv, so a task
 * is a (batch, KV head) and runs its `group` = heads/kvheads query heads in order -- the order
 * the serial loop adds. A task of fewer heads would put one KV head's query heads on two threads,
 * racing on its dk and dv (the TSan section's injected fault). */
static void att_head_backward(const att_job *a, size_t r, size_t h, const float *vt) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,i,j,i0; float scale=1.0f/sqrtf((float)d);
  (void)vt;  /* the AVX-512 instance's panel; NULL elsewhere */
  int variant=a->variant; const float *q=a->q,*k=a->k,*v=a->v,*dy=a->dy; float *p=a->p,*dq=a->dq,*dk=a->dk,*dv=a->dv;
  {
    size_t kh=h/(nh/nk),ldh=nh*d,ldk=nk*d; float *ph=p+(r*nh+h)*t*t;
    const float *qh=q+(r*t*nh+h)*d,*dyh=dy+(r*t*nh+h)*d;
    const float *kk=k+(r*t*nk+kh)*d,*vk=v+(r*t*nk+kh)*d;
    mm_args x;
    x.ta=1; x.tb=0; x.m=t; x.n=d; x.k=t; x.alpha=1.0f; x.a=ph; x.lda=t; x.b=dyh; x.ldb=ldh;
    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    att_rows(variant,&x,t,ATT_TRANSPOSED);
    for (i0=0;i0<t;i0+=BCIR_ATT_ROWS) {
      size_t rows=t-i0<BCIR_ATT_ROWS ? t-i0 : BCIR_ATT_ROWS,pass,jt;
      float dot[BCIR_ATT_ROWS],kept[BCIR_ATT_KEEP/BCIR_ATT_COLS][BCIR_ATT_ROWS*BCIR_ATT_COLS],
          spare[BCIR_ATT_ROWS*BCIR_ATT_COLS];
      for (i=0;i<rows;i++) dot[i]=0.0f;
      for (pass=0;pass<2;pass++) for (jt=0;jt<i0+rows;jt+=BCIR_ATT_COLS) {
        size_t cols=i0+rows-jt<BCIR_ATT_COLS ? i0+rows-jt : BCIR_ATT_COLS;
        float *tile=jt<BCIR_ATT_KEEP ? kept[jt/BCIR_ATT_COLS] : spare;
        if (!pass || jt>=BCIR_ATT_KEEP) {
          memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));
#ifdef BCIR_MM_DISPATCH
          if (vt) att512_dp(rows,i0,jt,cols,d,dyh,ldh,vt,t,tile,BCIR_ATT_COLS);
          else
#endif
          {
            x.ta=0; x.tb=1; x.m=rows; x.n=cols; x.k=d; x.a=dyh+i0*ldh; x.lda=ldh; x.b=vk+jt*ldk;
            x.ldb=ldk; x.c=tile; x.ldc=BCIR_ATT_COLS; x.mode=MM_FULL;
            mm_run(variant,&x);
          }
        }
        i=0;
#ifdef BCIR_MM_DISPATCH
        if (!pass && variant==4) for (;i+16<=rows;i+=16) att512_dot(t,i0,i,jt,jt+cols,ph,tile,BCIR_ATT_COLS,dot);
#endif
        for (;i<rows;i++) {
          float *pi=ph+(i0+i)*t; size_t end=i0+i+1<jt+cols ? i0+i+1 : jt+cols;
          for (j=jt;j<end;j++) {
            float dp=tile[i*BCIR_ATT_COLS+j-jt];
            if (!pass) dot[i]+=pi[j]*dp;
            else pi[j]=pi[j]*(dp-dot[i])*scale;
          }
        }
      }
    }
    x.ta=0; x.tb=0; x.m=t; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=kk; x.ldb=ldk;
    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;
    att_rows(variant,&x,t,ATT_CONTEXT);
    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    att_rows(variant,&x,t,ATT_TRANSPOSED);
  }
}
static void att_task_backward(void *ctx, size_t task) {
  att_job *a=(att_job *)ctx; size_t per=a->nh/a->group,r=task/per,first=(task%per)*a->group,h,i;
  size_t t=a->t,d=a->d,nh=a->nh,nk=a->nk,share=nh/nk,kh=first/share;
  /* the adjoints accumulate, so they start at zero here, in the task that owns them (NDT-SCAN; the
   * caller's fill was serial): this task's query heads' dq columns, and its KV head's dk and dv
   * columns where the task holds that KV head's first query head -- the task that runs before the
   * KV head's every accumulation, which is the whole group when group is the share */
  for (i=0;i<t;i++) {
    for (h=first;h<first+a->group;h++) memset(a->dq+((r*t+i)*nh+h)*d,0,d*sizeof(float));
    if (first%share==0) {
      memset(a->dk+((r*t+i)*nk+kh)*d,0,d*sizeof(float));
      memset(a->dv+((r*t+i)*nk+kh)*d,0,d*sizeof(float));
    }
  }
  {
    const float *vt=NULL;
#ifdef BCIR_MM_DISPATCH
    /* the KV head's values transposed once for its query heads' probability-adjoint tiles */
    _Alignas(64) float panel[BCIR_ATT_PANEL];
    if (a->variant==4 && d*t<=BCIR_ATT_PANEL) { att512_panel(t,d,a->v+(r*t*nk+kh)*d,nk*d,panel); vt=panel; }
#endif
    for (h=first;h<first+a->group;h++) att_head_backward(a,r,h,vt);
  }
}
void bcir_tensor_attention_backward_inplace_pool(bcir_tensor_pool *pool, size_t b, size_t t,
    size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v, float *p,
    const float *dy, float *dq, float *dk, float *dv) {
  att_job a;
  a.b=b; a.t=t; a.nh=nh; a.nk=nk; a.d=d; a.q=q; a.k=k; a.v=v; a.dy=dy; a.p=p; a.y=NULL;
  a.dq=dq; a.dk=dk; a.dv=dv; a.variant=mm_widest(); a.exp2=0;
  a.group=nh/nk;
  pool_run(pool,att_task_backward,&a,b*nh/a.group);
}
void bcir_tensor_attention_backward_inplace(size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, const float *dy,
    float *dq, float *dk, float *dv) {
  bcir_tensor_attention_backward_inplace_pool(NULL,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv);
}
