#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <float.h>
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
 * BCIR_TENSOR_MAX_VARIANT caps the choice (1, 2, 4 or 8, the bf16 contract's AMX instance) so a
 * build can run a narrower one. */
#ifndef BCIR_TENSOR_MAX_VARIANT
#define BCIR_TENSOR_MAX_VARIANT 8
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
static int mm_widest(void) {
  int v=bcir_tensor_mm_variants();
  return v&4 ? 4 : v&2 ? 2 : 1;
}
static int mm_widest_fused(void) {
  int v=bcir_tensor_mm_fused_variants();
  return v&4 ? 4 : v&2 ? 2 : 1;
}
/* PX-3b (DISPATCH): the instance a job of a binary32 family runs for a request -- 0 the widest the
 * family admits (every entry's default), else the one bit asked when the family admits it -- or
 * 0, refused: v1's instances are bcir_tensor_mm_variants', the fused and blocked contracts'
 * bcir_tensor_mm_fused_variants', and every instance of a family is the same bits. */
static int mm_instance(int fused, int variant) {
  int have=fused ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
  if (!variant) return fused ? mm_widest_fused() : mm_widest();
  return (variant==1 || variant==2 || variant==4) && (have&variant) ? variant : 0;
}
/* The harness's record of the instances jobs ran since the last read, by family -- bit v of family
 * f set when a task of a family-f job ran instance v -- so a witness holds a program's named
 * instance to the kernels that ran it. A build defines BCIR_TENSOR_INSTANCE_PROBE to have it (the
 * harness's builds do; the library does not); a read while no job runs. */
enum { INSTANCE_GEMM=0, INSTANCE_BF16=1, INSTANCE_ATTN=2, INSTANCE_ATTN16=3 };
#ifdef BCIR_TENSOR_INSTANCE_PROBE
static unsigned instance_seen[4];
#if defined(__GNUC__) || defined(__clang__)
#define INSTANCE_SAW(f,v) ((void)__atomic_fetch_or(&instance_seen[f],(unsigned)(v),__ATOMIC_RELAXED))
#else
#define INSTANCE_SAW(f,v) ((void)(instance_seen[f]|=(unsigned)(v)))
#endif
BCIR_DT_API unsigned bcir_tensor_instance_probe(int family);
unsigned bcir_tensor_instance_probe(int family) {
  unsigned seen;
  if (family<0 || family>3) return 0u;
  seen=instance_seen[family]; instance_seen[family]=0u;
  return seen;
}
#else
#define INSTANCE_SAW(f,v) ((void)0)
#endif
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
/* The SwiGLU forward's product in the same pass (NDT-SWIGLU-FWD): the SiLU rounded to binary32 as
 * silu2_body stores it, then its product with u rounded -- the two passes' value bit for bit. */
BCIR_MM_INLINE void silu2mul_body(size_t n, const float *x, const float *u, float *y) {
  size_t i; for (i=0;i<n;i++) { float s=x[i]*sigmoid2(x[i]); y[i]=s*u[i]; }
}
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
static void silu2mul_generic(size_t n, const float *x, const float *u, float *y) { silu2mul_body(n,x,u,y); }
static void swiglu2_generic(size_t n, const float *g, const float *u, const float *dy, float *dg,
    float *du) { swiglu2_body(n,g,u,dy,dg,du); }
#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx2"))) static void exp_avx2(size_t n, const float *x, float *y) { exp_body(n,x,y); }
__attribute__((target("avx512f"))) static void exp_avx512(size_t n, const float *x, float *y) { exp_body(n,x,y); }
__attribute__((target("avx2"))) static void exp_inplace_avx2(size_t n, float *x) { exp_inplace_body(n,x); }
__attribute__((target("avx512f"))) static void exp_inplace_avx512(size_t n, float *x) { exp_inplace_body(n,x); }
__attribute__((target("avx2"))) static void silu2_avx2(size_t n, const float *x, float *y) { silu2_body(n,x,y); }
__attribute__((target("avx512f"))) static void silu2_avx512(size_t n, const float *x, float *y) { silu2_body(n,x,y); }
__attribute__((target("avx2"))) static void silu2mul_avx2(size_t n, const float *x, const float *u, float *y) {
  silu2mul_body(n,x,u,y);
}
__attribute__((target("avx512f"))) static void silu2mul_avx512(size_t n, const float *x, const float *u,
    float *y) { silu2mul_body(n,x,u,y); }
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
static void silu2mul_run(int variant, size_t n, const float *x, const float *u, float *y) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { silu2mul_avx512(n,x,u,y); return; }
  if (variant==2) { silu2mul_avx2(n,x,u,y); return; }
#endif
  (void)variant; silu2mul_generic(n,x,u,y);
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
  INSTANCE_SAW(INSTANCE_GEMM,variant);
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
/* A task that is told its worker (BF16-ATTN): 0 the caller's thread, 1 .. threads-1 the pool's. A
 * worker runs one task at a time, so scratch indexed by the worker is the task's own while it runs;
 * no value may depend on which worker ran a task. */
typedef void (*pool_task_w)(void *ctx, size_t task, unsigned worker);
struct bcir_tensor_pool {
  unsigned threads,pending,cap; unsigned long generation; int stop;
  bcir_tensor_task_fn fn; pool_task_w fnw; void *ctx; size_t tasks,next;
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
/* PX-3b (DISPATCH): the workers the jobs after it run on. The cap is the caller's, set between jobs:
 * a worker reads it only after it sees a generation, which the caller publishes after setting it,
 * and every worker reports each generation before the caller returns, so the cap a worker reads is
 * the one of the job it reports for. */
unsigned bcir_tensor_pool_cap(bcir_tensor_pool *pool, unsigned workers) {
  unsigned prev;
  if (!pool) return 0u;
  prev=pool->cap;
  pool->cap=workers>=pool->threads ? 0u : workers;
  return prev;
}
unsigned bcir_tensor_pool_workers(const bcir_tensor_pool *pool) {
  return !pool ? 1u : pool->cap ? pool->cap : pool->threads;
}
static void pool_tasks(bcir_tensor_pool *p, unsigned id) {
  size_t t; unsigned workers=p->cap ? p->cap : p->threads;
  if (id>=workers) return;  /* past the job's cap: the worker reports without claiming a task */
#if defined(BCIR_POOL_ATOMIC)
  {
    bcir_tensor_task_fn fn=p->fn; pool_task_w fnw=p->fnw; void *ctx=p->ctx; size_t tasks=p->tasks;
    while ((t=__atomic_fetch_add(&p->next,(size_t)1,__ATOMIC_RELAXED))<tasks) {
      if (fnw) fnw(ctx,t,id); else fn(ctx,t);
    }
  }
#else
  for (t=id;t<p->tasks;t+=workers) { if (p->fnw) p->fnw(p->ctx,t,id); else p->fn(p->ctx,t); }
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
/* One job: fn(ctx, task) for every task, or with fnw fnw(ctx, task, worker). */
static void pool_dispatch(bcir_tensor_pool *p, bcir_tensor_task_fn fn, pool_task_w fnw, void *ctx,
    size_t tasks) {
  size_t t;
  if (!tasks) return;
  if (!p || p->threads==1 || tasks==1 || p->cap==1) {  /* a cap of one: the caller alone */
    for (t=0;t<tasks;t++) { if (fnw) fnw(ctx,t,0u); else fn(ctx,t); }
    return;
  }
#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
  POOL_LOCK(p);
  p->fn=fn; p->fnw=fnw; p->ctx=ctx; p->tasks=tasks;
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
  for (t=0;t<tasks;t++) { if (fnw) fnw(ctx,t,0u); else fn(ctx,t); }
#endif
}
void bcir_tensor_pool_run(bcir_tensor_pool *p, bcir_tensor_task_fn fn, void *ctx, size_t tasks) {
  pool_dispatch(p,fn,NULL,ctx,tasks);
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
  INSTANCE_SAW(INSTANCE_GEMM,j->variant);
  if (!j->x.fold) for (i=r.r0;i<r.r1;i++) {
    float *ci=j->x.c+i*j->x.ldc;
    for (jj=r.jb0;jj<r.jb1;jj++) ci[jj]=mm_scaled(j->beta,ci[jj]);
  }
  if (j->x.k) mm_run_range(j->variant,j->fused,&j->x,&r);
}
/* The job of one GEMM under a contract: its partition's task count (0 for an empty product). */
static size_t mm_job_init(mm_job *j, int fused, int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  if (!m || !n) return 0;
  j->beta=beta; j->fused=fused; j->variant=variant; j->nr=mm_nr(j->variant,fused);
  j->x.ta=ta; j->x.tb=tb; j->x.m=m; j->x.n=n; j->x.k=k; j->x.alpha=alpha; j->x.a=a; j->x.lda=ta ? m : k;
  j->x.b=b; j->x.ldb=tb ? k : n; j->x.c=c; j->x.ldc=n; j->x.mode=MM_FULL;
  j->x.beta=beta; j->x.fold=fused==MM_KBLOCK && k;
  j->cols=(n+j->nr-1)/j->nr; j->rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;
  return j->cols*j->rowsets;
}
/* `variant` resolved (mm_instance): the instance every call of the job runs */
static void mm_serial_any(int fused, int variant, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  mm_variant_any(variant,fused,ta,tb,m,n,k,alpha,a,b,beta,c);
}
static void mm_pool_any(bcir_tensor_pool *pool, int fused, int variant, int ta, int tb, size_t m,
    size_t n, size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_job j; size_t tasks;
  if (!pool || pool->threads==1) { mm_serial_any(fused,variant,ta,tb,m,n,k,alpha,a,b,beta,c); return; }
  tasks=mm_job_init(&j,fused,variant,ta,tb,m,n,k,alpha,a,b,beta,c);
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
static void mm_pair_any(bcir_tensor_pool *pool, int fused, int variant, const bcir_tensor_mm_call *x,
    const bcir_tensor_mm_call *y) {
  mm_pair_job j; size_t second;
  if (!pool || pool->threads==1) {
    mm_serial_any(fused,variant,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
    mm_serial_any(fused,variant,y->ta,y->tb,y->m,y->n,y->k,y->alpha,y->a,y->b,y->beta,y->c);
    return;
  }
  j.split=mm_job_init(&j.a,fused,variant,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
  second=mm_job_init(&j.b,fused,variant,y->ta,y->tb,y->m,y->n,y->k,y->alpha,y->a,y->b,y->beta,y->c);
  pool_run(pool,mm_pair_task,&j,j.split+second);
}
static int mm_family(int contract) {
  return contract==2 ? MM_KBLOCK : contract==1 ? MM_FUSED : MM_V1;
}
void bcir_tensor_mm_pair_pool(bcir_tensor_pool *pool, int contract, const bcir_tensor_mm_call *x,
    const bcir_tensor_mm_call *y) {
  int fused=mm_family(contract);
  mm_pair_any(pool,fused,mm_instance(fused,0),x,y);
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
static void mm_call_serial(int fused, int variant, const bcir_tensor_mm_call *x) {
  mm_serial_any(fused,variant,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
}
static size_t mm_call_init(mm_job *j, int fused, int variant, const bcir_tensor_mm_call *x) {
  return mm_job_init(j,fused,variant,x->ta,x->tb,x->m,x->n,x->k,x->alpha,x->a,x->b,x->beta,x->c);
}
static void mm_group_any(bcir_tensor_pool *pool, int fused, int variant, const bcir_tensor_mm_call *x,
    size_t nx, const bcir_tensor_mm_call *y, size_t ny) {
  mm_group_job j; size_t q;
  int serial=!pool || pool->threads==1 || nx>BCIR_TENSOR_MM_GROUP || ny>BCIR_TENSOR_MM_GROUP;
  for (q=1;q<nx && !serial;q++) serial=x[q].c!=x[0].c || x[q].m!=x[0].m || x[q].n!=x[0].n;
  if (serial) {
    for (q=0;q<nx;q++) mm_call_serial(fused,variant,x+q);
    for (q=0;q<ny;q++) mm_call_serial(fused,variant,y+q);
    return;
  }
  j.nx=nx; j.split=0; j.start[0]=0;
  for (q=0;q<nx;q++) { size_t tasks=mm_call_init(&j.x[q],fused,variant,x+q); if (!q) j.split=tasks; }
  for (q=0;q<ny;q++) j.start[q+1]=j.start[q]+mm_call_init(&j.y[q],fused,variant,y+q);
  pool_run(pool,mm_group_task,&j,j.split+j.start[ny]);
}
void bcir_tensor_mm_group_pool(bcir_tensor_pool *pool, int contract, const bcir_tensor_mm_call *x,
    size_t nx, const bcir_tensor_mm_call *y, size_t ny) {
  int fused=mm_family(contract);
  mm_group_any(pool,fused,mm_instance(fused,0),x,nx,y,ny);
}
void bcir_tensor_mm_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_V1,mm_instance(MM_V1,0),ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_fused_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_FUSED,mm_instance(MM_FUSED,0),ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_kblock_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_KBLOCK,mm_instance(MM_KBLOCK,0),ta,tb,m,n,k,alpha,a,b,beta,c);
}
/* --- the bf16 contract (NDT-BF16) --------------------------------------------------------------
 * Every instance accumulates from one packed layout (bcir_tensor.h): A as tiles of 16 rows x 32
 * depths, row-major, 64 bytes a row; B as tiles of 16 depth pairs x 16 columns, a column's two
 * depths of a pair adjacent (TDPBF16PS's B operand); the tiles of one 16-row (16-column) block
 * follow one another in depth order. A pack covers a chunk of rows [i0, i0+mp) and columns
 * [j0, j0+np), multiples of 32 (the AMX instance's four 16 x 16 accumulators cover 32 x 32), so row
 * i's tile q is ap + (((i-i0)/16)*kt + q)*BF16_TILE + ((i-i0)%16)*32. Padding is +0.0: rows past
 * m, columns past n, depths past k. A pool job packs, another computes: a compute task is a
 * BF16_ROWS x BF16_COLS region of C, every element of it there in its whole depth order, so any
 * partition, and any chunking, is the same result. */
#define BF16_TILE 512   /* bf16 elements in a tile: 16 x 32, 1 KB */
#define BF16_ROWS 32    /* a compute region: one 32-row strip of A ... */
#define BF16_PANEL (1024u*1024u)  /* ... over columns whose B panel is about 1 MB (in L2 beside it) */
#define BF16_TASKS 16   /* narrower regions where the rows alone give fewer */
#define BF16_ALIGN 64   /* the workspace's packs start on a 64-byte boundary */
#define BF16_CALLS (2*BCIR_TENSOR_MM_GROUP)
static size_t bf16_pad(size_t x) { return (x+31)/32*32; }
/* VCVTNEPS2BF16's rounding, which BF16-0 held to this on every one of the 2^32 inputs */
static uint16_t bf16_of(float x) {
  uint32_t u; memcpy(&u,&x,sizeof u);
  if ((u&0x7f800000u)==0x7f800000u && (u&0x007fffffu)) return (uint16_t)((u>>16)|0x0040u);
  if (!(u&0x7f800000u)) return (uint16_t)((u>>16)&0x8000u);  /* zero or subnormal: its zero */
  return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);  /* nearest, ties to even; past FLT_MAX, inf */
}
unsigned bcir_tensor_bf16_bits(float x) { return bf16_of(x); }
static float bf16_value(uint16_t h) {
  uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,sizeof f); return f;
}
/* The portable instance's arithmetic. A lane's step: the exact e + a*b -- a*b has at most 16
 * significant bits, so in double the product is exact and the sum rounded once to 53 bits, and
 * rounding that to binary32 is rounding the exact sum (53 >= 2*24 + 2) -- tiny when below
 * 2^-126 - 2^-151, the least magnitude 24 bits round to 2^-126 with the exponent unbounded (the
 * tininess MXCSR's FTZ and the AMX unit apply; a lane is +0.0 or normal, so the sum can be no
 * nearer that bound than an exact tie on it, which rounds to even, up). A sum of two such values is
 * a multiple of 2^-149, so it is tiny exactly when it is subnormal. */
static float bf16_fma(float e, float a, float b) {
  double d=(double)e+(double)a*(double)b;
  return fabs(d)<0x1.ffffffp-127 ? (float)copysign(0.0,d) : (float)d;
}
static float bf16_ftz(float x) { return fabsf(x)<FLT_MIN ? copysignf(0.0f,x) : x; }
/* One call of a job: its operands (mm_args, x.beta its beta), its chunk and its panels. */
typedef struct bf16_call {
  mm_args x;
  const uint16_t *ap,*bp;   /* the chunk's panels */
  uint16_t *wa,*wb;         /* where this call's pack tasks write them; NULL: packed by another */
  size_t kt,i0,mp,j0,np;    /* depth tiles; the chunk: rows [i0, i0+mp), columns [j0, j0+np) */
  size_t pa,pb,tr,tc,cols;  /* pack tasks; regions, rows x columns of them; a region's width */
  size_t q0,q1;             /* the depth tiles the kernels run, [q0, q1): all kt but in the
                               attention's causal products (BF16-ATTN) */
  int prefetch;             /* the AMX kernel prefetches the next block's C (a GEMM's call) */
} bf16_call;
typedef struct bf16_job {
  bf16_call call[BF16_CALLS]; size_t ncall,nchain,split; int variant;
  size_t packs[BF16_CALLS+1],tasks[BF16_CALLS+1];  /* prefix sums: every call's pack tasks; the
                                                      independent calls' regions after the chain */
} bf16_job;
/* A region's width: whole 32-column blocks whose B panel, over the whole depth, is at most
 * BF16_PANEL bytes (one at least), narrowed until the regions number BF16_TASKS where the rows
 * alone are fewer; regions are numbered column-major, so consecutive tasks share a B panel. A
 * function of the shapes alone, like every partition here -- and no partition reaches a result.
 * Counted in blocks, so a width is never zero: a call with no rows, no columns or no depth has one
 * too (a call without rows or columns has no regions), which the grouped entries give every call
 * they take. */
static size_t bf16_cols(size_t kt, size_t m, size_t n) {
  size_t panel=kt*32*32*2,width=panel ? BF16_PANEL/panel : 1,tr=(m+BF16_ROWS-1)/BF16_ROWS,nb=(n+31)/32;
  if (!width) width=1;
  if (tr && tr<BF16_TASKS) {  /* fewer row strips than tasks: `want` regions across, `per` blocks each */
    size_t want=(BF16_TASKS+tr-1)/tr,per=(nb+want-1)/want;
    if ((nb+width-1)/width<want && per && per<width) width=per;
  }
  return width*32;
}
static void bf16_args(bf16_call *c, const bcir_tensor_mm_call *x) {
  memset(c,0,sizeof *c);
  c->x.ta=x->ta; c->x.tb=x->tb; c->x.m=x->m; c->x.n=x->n; c->x.k=x->k; c->x.alpha=x->alpha;
  c->x.a=x->a; c->x.lda=x->ta ? x->m : x->k; c->x.b=x->b; c->x.ldb=x->tb ? x->k : x->n;
  c->x.c=x->c; c->x.ldc=x->n; c->x.mode=MM_FULL; c->x.beta=x->beta;
  c->kt=bf16_pad(x->k)/32; c->mp=bf16_pad(x->m); c->np=bf16_pad(x->n);
  c->q0=0; c->q1=c->kt; c->prefetch=1;
  c->cols=bf16_cols(c->kt,x->m,x->n);
  c->tr=(x->m+BF16_ROWS-1)/BF16_ROWS; c->tc=(x->n+c->cols-1)/c->cols;
}
/* A pack task s: the chunk's 32-row strip s over every depth tile, or (A transposed, its rows read
 * along the depth) depth tile s over every row block; a B pack task: the 32-column strip s (B
 * transposed) or depth tile s over every column block. One partition for every instance. A
 * GEMM's call packs every tile; the attention's (BF16-ATTN) bound a strip of A to its depth tiles
 * [q0, q1) and a depth tile of B to its first np columns, the ones its products read. */
static void bf16_pack_a_generic(const bf16_call *c, size_t s) {
  const mm_args *x=&c->x; size_t lb,q,r,l,kt=c->kt,lb0=x->ta ? 0 : 2*s,lb1=x->ta ? c->mp/16 : 2*s+2;
  size_t q0=x->ta ? s : c->q0,q1=x->ta ? s+1 : c->q1;
  for (lb=lb0;lb<lb1;lb++) for (q=q0;q<q1;q++) {
    uint16_t *t=c->wa+(lb*kt+q)*BF16_TILE;
    for (r=0;r<16;r++) {
      size_t i=c->i0+lb*16+r;
      for (l=0;l<32;l++) {
        size_t d=q*32+l;
        t[r*32+l]=i<x->m && d<x->k ?
          bf16_of(x->alpha*x->a[x->ta ? d*x->lda+i : i*x->lda+d]) : (uint16_t)0;
      }
    }
  }
}
static void bf16_pack_b_generic(const bf16_call *c, size_t s) {
  const mm_args *x=&c->x; size_t lc,q,p,jj,e,kt=c->kt,lc0=x->tb ? 2*s : 0,lc1=x->tb ? 2*s+2 : c->np/16;
  size_t q0=x->tb ? 0 : s,q1=x->tb ? kt : s+1;
  for (lc=lc0;lc<lc1;lc++) for (q=q0;q<q1;q++) {
    uint16_t *t=c->wb+(lc*kt+q)*BF16_TILE;
    for (p=0;p<16;p++) for (jj=0;jj<16;jj++) for (e=0;e<2;e++) {
      size_t j=c->j0+lc*16+jj,d=q*32+2*p+e;
      t[p*32+2*jj+e]=j<x->n && d<x->k ? bf16_of(x->b[x->tb ? j*x->ldb+d : d*x->ldb+j]) : (uint16_t)0;
    }
  }
}
/* Rows [r0, r1) x columns [c0, c1) of C over the chunk's panels: c from +0.0 (zero: beta 0, C not
 * read) or as C holds it (already beta-scaled), then every depth tile in order -- tiles [q0, q1),
 * which a GEMM's call sets to all kt (the panels' stride stays kt). */
static void bf16_generic(const bf16_call *c, size_t r0, size_t r1, size_t c0, size_t c1, int zero) {
  const mm_args *x=&c->x; size_t i,j,q,p,kt=c->kt;
  for (i=r0;i<r1;i++) {
    const uint16_t *arow=c->ap+((i-c->i0)/16)*kt*BF16_TILE+((i-c->i0)%16)*32;
    for (j=c0;j<c1;j++) {
      const uint16_t *bcol=c->bp+((j-c->j0)/16)*kt*BF16_TILE+2*((j-c->j0)%16);
      float acc=zero ? 0.0f : x->c[i*x->ldc+j];
      for (q=c->q0;q<c->q1;q++) {
        const uint16_t *at=arow+q*BF16_TILE,*bt=bcol+q*BF16_TILE; float e=0.0f,o=0.0f;
        for (p=0;p<16;p++) {
          e=bf16_fma(e,bf16_value(at[2*p]),bf16_value(bt[p*32]));
          o=bf16_fma(o,bf16_value(at[2*p+1]),bf16_value(bt[p*32+1]));
        }
        acc=bf16_ftz(bf16_ftz(acc)+bf16_ftz(e+o));
      }
      x->c[i*x->ldc+j]=acc;
    }
  }
}
#ifdef BCIR_MM_DISPATCH
/* 16 binary32 rounded to bf16, each in its lane's upper half (bf16_of on every lane; the lower
 * halves are not part of the value) */
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512i bf512_round(__m512 v) {
  const __m512i u=_mm512_castps_si512(v);
  __m512i r=_mm512_add_epi32(u,_mm512_add_epi32(_mm512_set1_epi32(0x7fff),
      _mm512_and_si512(_mm512_srli_epi32(u,16),_mm512_set1_epi32(1))));
  __mmask16 nan=_mm512_cmpgt_epu32_mask(_mm512_and_si512(u,_mm512_set1_epi32(0x7fffffff)),
      _mm512_set1_epi32(0x7f800000));
  __mmask16 sub=_mm512_testn_epi32_mask(u,_mm512_set1_epi32(0x7f800000));
  r=_mm512_mask_mov_epi32(r,nan,_mm512_or_si512(u,_mm512_set1_epi32(0x00400000)));
  return _mm512_mask_mov_epi32(r,sub,_mm512_and_si512(u,_mm512_set1_epi32((int)0x80000000u)));
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m256i bf512_halves(__m512i r) {
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(r,16));
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE __mmask16 bf512_mask(size_t count) {
  return (__mmask16)(count>=16 ? 0xffffu : (1u<<count)-1u);
}
/* The packs by vector, bf16_pack_*_generic's tiles byte for byte, in their partition: rows of A
 * read along, A^T's 32 depth rows of a tile and B^T's 16 columns through kb512_transpose16 (a
 * transpose moves bit patterns only), B's depth pairs interleaved; padding is never read. BF16_PACK
 * is one body over two conversions of 32 binary32 (lo, hi) to 32 bf16 in one vector: the AVX-512F
 * one (bf512_round on each half), and the AMX host's VCVTNE2PS2BF16 (AVX512-BF16, which every
 * AMX-BF16 CPU carries; BF16-0 held it to bf16_of on all 2^32 inputs, and the admission
 * differential holds the packs to bf16_pack_*_generic's on the conversion's edges). */
__attribute__((target("avx512f"))) BCIR_MM_INLINE __m512i bf512_cvt(__m512 lo, __m512 hi) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(bf512_halves(bf512_round(lo))),
      bf512_halves(bf512_round(hi)),1);
}
#define BF16_PACK_BODIES(NAME, TARGET, CVT, PAIRS) \
TARGET static void NAME##_pack_a(const bf16_call *c, size_t s) { \
  const mm_args *x=&c->x; size_t lb,q,r,ll,kt=c->kt; const __m512 alpha=_mm512_set1_ps(x->alpha); \
  const int scale=x->alpha!=1.0f; _Alignas(64) float st[512],tr[512]; \
  if (!x->ta) { \
    for (lb=2*s;lb<2*s+2;lb++) for (r=0;r<16;r++) { \
      size_t i=c->i0+lb*16+r; \
      for (q=c->q0;q<c->q1;q++) { \
        uint16_t *t=c->wa+(lb*kt+q)*BF16_TILE+r*32; size_t d=q*32; __m512 v0,v1; \
        __mmask16 m0=bf512_mask(x->k-d),m1=bf512_mask(x->k>d+16 ? x->k-d-16 : 0); \
        if (i>=x->m) { _mm512_storeu_si512(t,_mm512_setzero_si512()); continue; } \
        v0=_mm512_maskz_loadu_ps(m0,x->a+i*x->lda+d); \
        v1=m1 ? _mm512_maskz_loadu_ps(m1,x->a+i*x->lda+d+16) : _mm512_setzero_ps(); \
        if (scale) { v0=_mm512_maskz_mul_ps(m0,v0,alpha); v1=_mm512_maskz_mul_ps(m1,v1,alpha); } \
        _mm512_storeu_si512(t,CVT(v0,v1)); \
      } \
    } \
    return; \
  } \
  q=s; \
  for (lb=0;lb<c->mp/16;lb++) { \
    size_t ib=c->i0+lb*16,d=q*32; uint16_t *t=c->wa+(lb*kt+q)*BF16_TILE; \
    __mmask16 mi=bf512_mask(x->m>ib ? x->m-ib : 0),m0=bf512_mask(x->k-d),m1=bf512_mask(x->k>d+16 ? x->k-d-16 : 0); \
    if (!mi) { memset(t,0,BF16_TILE*sizeof *t); continue; } \
    for (ll=0;ll<32;ll++) \
      _mm512_store_ps(st+ll*16,d+ll<x->k ? _mm512_maskz_loadu_ps(mi,x->a+(d+ll)*x->lda+ib) : _mm512_setzero_ps()); \
    kb512_transpose16(st,16,tr,16); kb512_transpose16(st+256,16,tr+256,16); \
    for (r=0;r<16;r++) { \
      __m512 v0=_mm512_load_ps(tr+r*16),v1=_mm512_load_ps(tr+256+r*16); \
      if (ib+r>=x->m) { v0=_mm512_setzero_ps(); v1=_mm512_setzero_ps(); } \
      else if (scale) { v0=_mm512_maskz_mul_ps(m0,v0,alpha); v1=_mm512_maskz_mul_ps(m1,v1,alpha); } \
      _mm512_storeu_si512(t+r*32,CVT(v0,v1)); \
    } \
  } \
} \
TARGET static void NAME##_pack_b(const bf16_call *c, size_t s) { \
  const mm_args *x=&c->x; size_t lc,q,p,jj,kt=c->kt; _Alignas(64) float st[256]; \
  if (!x->tb) { \
    q=s; \
    for (p=0;p<16;p++) for (lc=0;lc<c->np/16;lc++) { \
      size_t jb=c->j0+lc*16,d=q*32+2*p; uint16_t *t=c->wb+(lc*kt+q)*BF16_TILE+p*32; \
      __mmask16 mj=bf512_mask(x->n>jb ? x->n-jb : 0); __m512 ve,vo; \
      if (!mj) { _mm512_storeu_si512(t,_mm512_setzero_si512()); continue; } \
      ve=d<x->k ? _mm512_maskz_loadu_ps(mj,x->b+d*x->ldb+jb) : _mm512_setzero_ps(); \
      vo=d+1<x->k ? _mm512_maskz_loadu_ps(mj,x->b+(d+1)*x->ldb+jb) : _mm512_setzero_ps(); \
      _mm512_storeu_si512(t,PAIRS(CVT(ve,vo))); \
    } \
    return; \
  } \
  for (lc=2*s;lc<2*s+2;lc++) { \
    size_t jb=c->j0+lc*16; \
    for (q=0;q<kt;q++) { \
      uint16_t *t=c->wb+(lc*kt+q)*BF16_TILE; size_t d=q*32; \
      __mmask16 m0=bf512_mask(x->k-d),m1=bf512_mask(x->k>d+16 ? x->k-d-16 : 0); \
      if (jb>=x->n) { memset(t,0,BF16_TILE*sizeof *t); continue; } \
      for (jj=0;jj<16;jj++) {  /* st[jj*16+p]: column jb+jj's pair p, one 32-bit word */ \
        size_t j=jb+jj; \
        if (j>=x->n) { _mm512_store_ps(st+jj*16,_mm512_setzero_ps()); continue; } \
        _mm512_store_si512(st+jj*16,CVT(_mm512_maskz_loadu_ps(m0,x->b+j*x->ldb+d), \
            m1 ? _mm512_maskz_loadu_ps(m1,x->b+j*x->ldb+d+16) : _mm512_setzero_ps())); \
      } \
      kb512_transpose16(st,16,(float *)(void *)t,16); \
    } \
  } \
}
/* depth pairs: lanes 0-15 hold the even depths' bf16, 16-31 the odd; the B layout interleaves
 * them, even first (VPERMW, AVX512-BW) */
#define BF16_PAIR_INDEX _mm512_set_epi16(31,15,30,14,29,13,28,12,27,11,26,10,25,9,24,8,23,7,22,6, \
    21,5,20,4,19,3,18,2,17,1,16,0)
#define BF512_PAIRS_BW(v) _mm512_permutexvar_epi16(BF16_PAIR_INDEX,(v))
#define BF16_AVX512BW __attribute__((target("avx512f,avx512bw")))
BF16_PACK_BODIES(bf512, BF16_AVX512BW, bf512_cvt, BF512_PAIRS_BW)
/* The AVX-512 instance: per 16 columns, rows in register blocks; E and O lanes as vectors of the
 * tile's 16 columns, an A pair broadcast as one 32-bit word and split by shift and mask (a bf16 is
 * the upper half of its binary32). Its flushes are MXCSR's (bf16_avx512 sets FTZ and DAZ around it;
 * BF16-0 held a vector FMA under them to the AMX unit's lanes): DAZ reads a subnormal c as its zero,
 * FTZ flushes every fma and sum the contract flushes. Out of line, so no operation of it can move
 * across the MXCSR writes. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void bf512_rows(size_t mr, const bf16_call *c,
    size_t i, size_t j, __mmask16 keep, int zero) {
  const mm_args *x=&c->x; const size_t kt=c->kt; size_t r,q,p;
  const uint16_t *arow=c->ap+((i-c->i0)/16)*kt*BF16_TILE+((i-c->i0)%16)*32;
  const uint16_t *bcol=c->bp+((j-c->j0)/16)*kt*BF16_TILE;
  const __m512i hi=_mm512_set1_epi32((int)0xffff0000u);
  __m512 acc[8],e[8],o[8];
  for (r=0;r<mr;r++) acc[r]=zero ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(keep,x->c+(i+r)*x->ldc+j);
  for (q=c->q0;q<c->q1;q++) {
    const uint16_t *at=arow+q*BF16_TILE,*bt=bcol+q*BF16_TILE;
    for (r=0;r<mr;r++) { e[r]=_mm512_setzero_ps(); o[r]=_mm512_setzero_ps(); }
    for (p=0;p<16;p++) {
      __m512i bv=_mm512_loadu_si512(bt+p*32);
      __m512 be=_mm512_castsi512_ps(_mm512_slli_epi32(bv,16)),bo=_mm512_castsi512_ps(_mm512_and_si512(bv,hi));
      for (r=0;r<mr;r++) {
        uint32_t w; __m512i aw;
        memcpy(&w,at+r*32+2*p,sizeof w); aw=_mm512_set1_epi32((int)w);
        e[r]=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(aw,16)),be,e[r]);
        o[r]=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_and_si512(aw,hi)),bo,o[r]);
      }
    }
    for (r=0;r<mr;r++) acc[r]=_mm512_add_ps(acc[r],_mm512_add_ps(e[r],o[r]));
  }
  for (r=0;r<mr;r++) _mm512_mask_storeu_ps(x->c+(i+r)*x->ldc+j,keep,acc[r]);
}
__attribute__((target("avx512f"),noinline)) static void bf512_region(const bf16_call *c, size_t r0,
    size_t r1, size_t c0, size_t c1, int zero) {
  size_t i,j;
  for (j=c0;j<c1;j+=16) {
    __mmask16 keep=bf512_mask(c1-j);
    for (i=r0;i+8<=r1;i+=8) bf512_rows(8,c,i,j,keep,zero);
    for (;i+4<=r1;i+=4) bf512_rows(4,c,i,j,keep,zero);
    for (;i<r1;i++) bf512_rows(1,c,i,j,keep,zero);
  }
}
static void bf16_avx512(const bf16_call *c, size_t r0, size_t r1, size_t c0, size_t c1, int zero) {
  unsigned int csr=_mm_getcsr();
  _mm_setcsr(csr|0x8040u);  /* FTZ (bit 15) and DAZ (bit 6) */
  bf512_region(c,r0,r1,c0,c1,zero);
  _mm_setcsr(csr);
}
#if BCIR_TENSOR_MAX_VARIANT>=8 && !defined(BCIR_TENSOR_NO_AMX)
#define BCIR_BF16_AMX 1
#include <cpuid.h>
#define BF16_AMXHOST __attribute__((target("avx512f,avx512bw,avx512bf16")))
BF16_AMXHOST BCIR_MM_INLINE __m512i bfne_cvt(__m512 lo, __m512 hi) {
  return (__m512i)_mm512_cvtne2ps_pbh(hi,lo);  /* bf16 of lo in lanes 0-15, of hi in 16-31 */
}
BF16_PACK_BODIES(bfne, BF16_AMXHOST, bfne_cvt, BF512_PAIRS_BW)
/* The AMX instance: four 16 x 16 accumulator tiles (0-3) over a 32 x 32 block of C, two A tiles
 * (4, 5) and two B tiles (6, 7) a depth tile; TDPBF16PS is the contract's block step exactly. A
 * block that the region's edge cuts goes through a zero-padded copy. The tiles are configured at
 * the start and released at the end of every region, so no thread keeps tile state between tasks.
 * Where the operands sit is not part of the arithmetic (BF16-KERN): the region's blocks share one A
 * strip and read each B block once, so the B tiles are loaded as streamed (TILELOADDT1), leaving L1
 * to the strip; and the next block's lines of C are prefetched for writing during the depth loop,
 * a few a depth tile, so the stores that end a block find them owned (PREFETCHW, a hint: where a
 * CPU lacks it, a no-op). Every tile operation is the one before, in the same order. */
typedef struct bf16_tilecfg {
  unsigned char palette,start,reserved[14]; unsigned short colsb[16]; unsigned char rows[16];
} bf16_tilecfg;
/* The tile unit's configuration, one for every region: eight tiles of 16 rows by 64 bytes. A region
 * runs between bf16_amx_begin and bf16_amx_end; the attention's tasks (BF16-ATTN) run many regions
 * between one pair, a GEMM's task one. */
__attribute__((target("amx-tile"))) static void bf16_amx_begin(void) {
  bf16_tilecfg cfg; size_t q;
  memset(&cfg,0,sizeof cfg); cfg.palette=1;
  for (q=0;q<8;q++) { cfg.rows[q]=16; cfg.colsb[q]=64; }
  _tile_loadconfig(&cfg);
}
__attribute__((target("amx-tile"))) static void bf16_amx_end(void) { _tile_release(); }
__attribute__((target("amx-tile,amx-bf16,prfchw"))) static void bf16_amx_blocks(const bf16_call *c,
    size_t r0, size_t r1, size_t c0, size_t c1, int zero) {
  const mm_args *x=&c->x; const size_t kt=c->kt,ldc=x->ldc,nq=c->q1-c->q0;
  const size_t per=c->prefetch ? (64+nq-1)/(nq ? nq : 1) : 0;
  size_t i,j,q,r,cc; _Alignas(64) float edge[32*32];
  for (i=r0;i<r1;i+=32) for (j=c0;j<c1;j+=32) {
    const uint16_t *a0=c->ap+((i-c->i0)/16)*kt*BF16_TILE,*a1=a0+kt*BF16_TILE;
    const uint16_t *b0=c->bp+((j-c->j0)/16)*kt*BF16_TILE,*b1=b0+kt*BF16_TILE;
    size_t rows=r1-i<32 ? r1-i : 32,cols=c1-j<32 ? c1-j : 32; float *cp=x->c+i*ldc+j;
    int whole=rows==32 && cols==32;
    /* the next block in this order, and its lines of C: row l/2's half l%2 of 32 x 32, where in C */
    size_t ni=j+32<c1 ? i : i+32,nj=j+32<c1 ? j+32 : c0,nrows=ni<r1 ? (r1-ni<32 ? r1-ni : 32) : 0;
    size_t ncols=nj<c1 ? (c1-nj<32 ? c1-nj : 32) : 0; const float *np=nrows && ncols ? x->c+ni*ldc+nj : cp;
    if (zero) { _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); }
    else if (whole) {
      _tile_loadd(0,cp,ldc*4); _tile_loadd(1,cp+16,ldc*4);
      _tile_loadd(2,cp+16*ldc,ldc*4); _tile_loadd(3,cp+16*ldc+16,ldc*4);
    } else {
      memset(edge,0,sizeof edge);
      for (r=0;r<rows;r++) for (cc=0;cc<cols;cc++) edge[r*32+cc]=cp[r*ldc+cc];
      _tile_loadd(0,edge,128); _tile_loadd(1,edge+16,128);
      _tile_loadd(2,edge+16*32,128); _tile_loadd(3,edge+16*32+16,128);
    }
    for (q=c->q0;q<c->q1;q++) {
      size_t l,first=(q-c->q0)*per;
      for (l=first;l<first+per && l<64;l++)
        if ((l>>1)<nrows && (l&1)*16<ncols) __builtin_prefetch(np+(l>>1)*ldc+(l&1)*16,1,3);
      _tile_loadd(4,a0+q*BF16_TILE,64); _tile_loadd(5,a1+q*BF16_TILE,64);
      _tile_stream_loadd(6,b0+q*BF16_TILE,64); _tile_stream_loadd(7,b1+q*BF16_TILE,64);
      _tile_dpbf16ps(0,4,6); _tile_dpbf16ps(1,4,7); _tile_dpbf16ps(2,5,6); _tile_dpbf16ps(3,5,7);
    }
    if (whole) {
      _tile_stored(0,cp,ldc*4); _tile_stored(1,cp+16,ldc*4);
      _tile_stored(2,cp+16*ldc,ldc*4); _tile_stored(3,cp+16*ldc+16,ldc*4);
    } else {
      _tile_stored(0,edge,128); _tile_stored(1,edge+16,128);
      _tile_stored(2,edge+16*32,128); _tile_stored(3,edge+16*32+16,128);
      for (r=0;r<rows;r++) for (cc=0;cc<cols;cc++) cp[r*ldc+cc]=edge[r*32+cc];
    }
  }
}
__attribute__((target("amx-tile,amx-bf16,prfchw"))) static void bf16_amx(const bf16_call *c, size_t r0,
    size_t r1, size_t c0, size_t c1, int zero) {
  bf16_amx_begin(); bf16_amx_blocks(c,r0,r1,c0,c1,zero); bf16_amx_end();
}
/* The host's AMX: CPUID's AMX-TILE and AMX-BF16 (and the AVX512-BW and -BF16 its packs use),
 * XCR0's tile configuration and data (the OS saves them), and Linux's grant of the tile data to this process (arch_prctl(ARCH_REQ_XCOMP_PERM,
 * XFEATURE_XTILEDATA), as a raw system call: x86-64 Linux, no feature macros). */
static int bf16_amx_host(void) {
  unsigned a,b,c,d,lo,hi; long rc;
  if (!__get_cpuid(1,&a,&b,&c,&d) || !(c&(1u<<27))) return 0;
  if (!__get_cpuid_count(7,0,&a,&b,&c,&d) || !(d&(1u<<22)) || !(d&(1u<<24)) || !(b&(1u<<30))) return 0;
  if (!__get_cpuid_count(7,1,&a,&b,&c,&d) || !(a&(1u<<5))) return 0;  /* AVX512-BF16 */
  __asm__ __volatile__("xgetbv" : "=a"(lo),"=d"(hi) : "c"(0u));
  (void)hi;
  if ((lo&(3u<<17))!=(3u<<17)) return 0;
  __asm__ __volatile__("syscall" : "=a"(rc) : "a"(158L),"D"(0x1023L),"S"(18L) : "rcx","r11","memory");
  return rc==0;
}
/* every pack task of one call into its panels, by the portable packs (amx 0) or the AMX host's */
static void bf16_pack_all(bf16_call *c, int amx) {
  size_t s,pa=c->x.ta ? c->kt : c->mp/32,pb=c->x.tb ? c->np/32 : c->kt;
  for (s=0;s<pa;s++) { if (amx) bfne_pack_a(c,s); else bf16_pack_a_generic(c,s); }
  for (s=0;s<pb;s++) { if (amx) bfne_pack_b(c,s); else bf16_pack_b_generic(c,s); }
}
/* The admission differential, both halves of the instance. The packs: every layout (A and B, each
 * transposed or not) packed by the AMX host's packs and by the portable ones from operands whose
 * low halves reach the conversion's edges -- ties to even both ways, near-ties, subnormals, NaN
 * payloads, the largest finite values (which round to infinity), infinities, zeros, a shape whose
 * rows, columns and depth are all ragged -- byte for byte. The arithmetic: one 32 x 64 x 32
 * product on operands that reach the contract's edges -- a lane landing in [2^-126 - 2^-150,
 * 2^-126 - 2^-151) (tiny, though 2^-126 on the subnormal grid), lanes of -0.0 terms from +0.0
 * under c = -0.0, a subnormal c, products below the normal range, infinities -- and mixed
 * magnitudes elsewhere, through the AMX kernel and the portable one from the same panels, with c
 * read and from zero; every element agreeing (NaN with NaN). Otherwise the instance is not offered. */
static int bf16_amx_differential(void) {
  enum { M=32, N=32, K=64, RM=27, RN=21, RK=45 };
  _Alignas(64) uint16_t pa[M*K],pb[K*N],qa[M*K],qb[K*N]; float a[M*K],b[K*N],c0[M*N],c1[M*N],c2[M*N];
  bf16_call cl; uint32_t s=0x2545f491u; size_t i,j,l,pass; int layout;
  for (i=0;i<M*K+K*N;i++) {  /* the conversion's edges, in the packs */
    uint32_t u,r; float f;
    s=s*1664525u+1013904223u; r=s;
    switch (i%8) {
    case 0: u=(r&0x80000000u)|((uint32_t)(100+(r>>8)%56)<<23)|((r>>3)&0x007fffffu); break;
    case 1: u=(r&0xffff0000u&~0x7f800000u)|((uint32_t)(120+(r>>9)%16)<<23)|0x8000u; break;
    case 2: u=(r&0xffff0000u&~0x7f800000u)|((uint32_t)(120+(r>>9)%16)<<23)|(r&1 ? 0x7fffu : 0x8001u); break;
    case 3: u=r&0x807fffffu; break;
    case 4: u=(r&0x80000000u)|0x7f800000u|(((r>>7)&0x7fffffu)|1u); break;
    case 5: u=(r&0x80000000u)|0x7f7f8000u|(r&0x7fffu); break;
    case 6: u=(r&0x80000000u)|0x7f800000u; break;
    default: u=r&0x80000000u; break;
    }
    memcpy(&f,&u,sizeof f);
    if (i<M*K) a[i]=f; else b[i-M*K]=f;
  }
  for (layout=0;layout<4;layout++) {
    bcir_tensor_mm_call x;
    x.ta=layout&1; x.tb=layout>>1; x.m=RM; x.n=RN; x.k=RK; x.alpha=layout==3 ? -1.5f : 1.0f;
    x.a=a; x.b=b; x.beta=0.0f; x.c=c0;
    bf16_args(&cl,&x);
    cl.wa=pa; cl.ap=pa; cl.wb=pb; cl.bp=pb; bf16_pack_all(&cl,0);
    cl.wa=qa; cl.ap=qa; cl.wb=qb; cl.bp=qb; bf16_pack_all(&cl,1);
    if (memcmp(pa,qa,cl.mp*cl.kt*32*sizeof *pa) || memcmp(pb,qb,cl.np*cl.kt*32*sizeof *pb)) return 0;
  }
  for (i=0;i<M*K+K*N+M*N;i++) {  /* the arithmetic's operands: bf16 values, mixed magnitudes */
    uint32_t u; float f;
    s=s*1664525u+1013904223u;
    u=i<M*K+K*N ? (s&0x80000000u)|((uint32_t)(119+(s>>8)%17)<<23)|((s>>1)&0x007f0000u)
                : (s&0x80000000u)|((uint32_t)(121+(s>>8)%13)<<23)|((s>>1)&0x007fffffu);
    memcpy(&f,&u,sizeof f);
    if (i<M*K) a[i]=f; else if (i<M*K+K*N) b[i-M*K]=f; else c0[i-M*K-K*N]=f;
  }
  for (l=0;l<K;l++) { a[0*K+l]=0.0f; b[l*N+0]=0.0f; a[1*K+l]=1.0f; b[l*N+1]=-0.0f; a[2*K+l]=0.0f;
                      a[3*K+l]=0x1p-70f; b[l*N+3]=0x1p-70f; }
  a[0*K+0]=1.0f; b[0*N+0]=0x1p-126f; a[0*K+2]=-0x1.4p-75f; b[2*N+0]=0x1.1p-76f; c0[0*N+0]=0.0f;
  c0[1*N+1]=-0.0f; a[2*K+0]=1.0f; b[0*N+2]=0x1p-126f; c0[2*N+2]=0x1p-149f; c0[3*N+3]=1.0f;
  a[4*K+0]=INFINITY; b[0*N+4]=1.0f;
  {
    bcir_tensor_mm_call x;
    x.ta=0; x.tb=0; x.m=M; x.n=N; x.k=K; x.alpha=1.0f; x.a=a; x.b=b; x.beta=1.0f; x.c=c1;
    bf16_args(&cl,&x);
  }
  cl.wa=pa; cl.ap=pa; cl.wb=pb; cl.bp=pb; bf16_pack_all(&cl,0);
  for (pass=0;pass<2;pass++) {
    memcpy(c1,c0,sizeof c1); memcpy(c2,c0,sizeof c2);
    cl.x.c=c1; bf16_amx(&cl,0,M,0,N,(int)pass);
    cl.x.c=c2; bf16_generic(&cl,0,M,0,N,(int)pass);
    for (i=0;i<M;i++) for (j=0;j<N;j++) {
      float u=c1[i*N+j],v=c2[i*N+j];
      if (!(u!=u && v!=v) && memcmp(&u,&v,sizeof u)) return 0;
    }
  }
  return 1;
}
static int bf16_amx_admitted(void) {
  static int answer;  /* 0 not asked, 1 offered, 2 not: the unit's one value kept between calls */
  int v=__atomic_load_n(&answer,__ATOMIC_ACQUIRE);
  if (!v) {
    v=bf16_amx_host() && bf16_amx_differential() ? 1 : 2;
    __atomic_store_n(&answer,v,__ATOMIC_RELEASE);
  }
  return v==1;
}
#endif
#endif
int bcir_tensor_mm_bf16_variants(void) {
  int v=1;
#ifdef BCIR_MM_DISPATCH
  __builtin_cpu_init();
  if (BCIR_TENSOR_MAX_VARIANT>=4 && __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw")) {
    v|=4;
#ifdef BCIR_BF16_AMX
    if (bf16_amx_admitted()) v|=8;
#endif
  }
#endif
  return v;
}
#ifdef BCIR_DT_PANELS_PROBE
/* The trainer harness's question of the AMX instance's admission (BF16-KERN): whether this build and
 * this host offer the instance -- the vector instance's gate and the host's AMX, everything the
 * admission asks but its differential -- so the harness can hold the differential to passing
 * wherever the instance is offered. A wrong AMX kernel is withdrawn by its admission, never run, so
 * no value shows it: here it is an offered instance not admitted. A build defines
 * BCIR_DT_PANELS_PROBE to have it (the harness's own builds do; the library does not). */
int bcir_tensor_bf16_amx_offered(void);
int bcir_tensor_bf16_amx_offered(void) {
#if defined(BCIR_MM_DISPATCH) && defined(BCIR_BF16_AMX)
  return (bcir_tensor_mm_bf16_variants()&4) && bf16_amx_host();
#else
  return 0;
#endif
}
#endif
static void bf16_pack_task(void *ctx, size_t task) {
  bf16_job *j=(bf16_job *)ctx; size_t q=0; const bf16_call *c;
  while (task>=j->packs[q+1]) q++;
  c=&j->call[q]; task-=j->packs[q];
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (j->variant==8) { if (task<c->pa) bfne_pack_a(c,task); else bfne_pack_b(c,task-c->pa); return; }
#endif
  if (j->variant>=4) { if (task<c->pa) bf512_pack_a(c,task); else bf512_pack_b(c,task-c->pa); return; }
#endif
  if (task<c->pa) bf16_pack_a_generic(c,task); else bf16_pack_b_generic(c,task-c->pa);
}
/* A region of C: beta first (a product other than by 0 or 1 rounded into C, the default
 * environment's arithmetic), then the instance's kernel; with no depth, beta alone. */
static void bf16_region(int variant, const bf16_call *c, size_t t) {
  const mm_args *x=&c->x; size_t r0,r1,c0,c1,rend,cend,i,j; int zero=x->beta==0.0f;
  r0=c->i0+(t%c->tr)*BF16_ROWS; c0=c->j0+(t/c->tr)*c->cols;
  rend=c->i0+c->mp<x->m ? c->i0+c->mp : x->m; cend=c->j0+c->np<x->n ? c->j0+c->np : x->n;
  r1=r0+BF16_ROWS<rend ? r0+BF16_ROWS : rend; c1=c0+c->cols<cend ? c0+c->cols : cend;
  if (!zero && x->beta!=1.0f)
    for (i=r0;i<r1;i++) for (j=c0;j<c1;j++) x->c[i*x->ldc+j]=x->beta*x->c[i*x->ldc+j];
  if (!c->kt) {
    if (zero) for (i=r0;i<r1;i++) for (j=c0;j<c1;j++) x->c[i*x->ldc+j]=0.0f;
    return;
  }
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (variant==8) { bf16_amx(c,r0,r1,c0,c1,zero); return; }
#endif
  if (variant>=4) { bf16_avx512(c,r0,r1,c0,c1,zero); return; }
#endif
  (void)variant;
  bf16_generic(c,r0,r1,c0,c1,zero);
}
static void bf16_compute_task(void *ctx, size_t task) {
  bf16_job *j=(bf16_job *)ctx; size_t q;
  INSTANCE_SAW(INSTANCE_BF16,j->variant);
  if (task<j->split) { for (q=0;q<j->nchain;q++) bf16_region(j->variant,&j->call[q],task); return; }
  task-=j->split;
  for (q=0;task>=j->tasks[q+1];q++) {}
  bf16_region(j->variant,&j->call[j->nchain+q],task-j->tasks[q]);
}
static void bf16_prefix(bf16_job *j) {
  size_t q;
  j->packs[0]=0;
  for (q=0;q<j->ncall;q++) j->packs[q+1]=j->packs[q]+j->call[q].pa+j->call[q].pb;
  j->split=j->nchain ? j->call[0].tr*j->call[0].tc : 0;
  j->tasks[0]=0;
  for (q=j->nchain;q<j->ncall;q++)
    j->tasks[q-j->nchain+1]=j->tasks[q-j->nchain]+j->call[q].tr*j->call[q].tc;
}
static int bf16_packs(const bcir_tensor_mm_call *x) { return x->m && x->n && x->k; }
static const bcir_tensor_mm_call *bf16_nth(const bcir_tensor_mm_call *chain, size_t nchain,
    const bcir_tensor_mm_call *y, size_t q) { return q<nchain ? chain+q : y+(q-nchain); }
static int bf16_same_a(const bcir_tensor_mm_call *x, const bcir_tensor_mm_call *y) {
  return x->a==y->a && x->ta==y->ta && x->m==y->m && x->k==y->k && !memcmp(&x->alpha,&y->alpha,sizeof x->alpha);
}
static int bf16_same_b(const bcir_tensor_mm_call *x, const bcir_tensor_mm_call *y) {
  return x->b==y->b && x->tb==y->tb && x->k==y->k && x->n==y->n;
}
static size_t bf16_sum(size_t a, size_t b) { return a>SIZE_MAX-b ? SIZE_MAX : a+b; }
/* bytes of `rows` rows (or columns) packed over k depths, padded; SIZE_MAX past size_t */
static size_t bf16_strip(size_t rows, size_t k) {
  size_t r,kp;
  if (rows>SIZE_MAX-31 || k>SIZE_MAX-31) return SIZE_MAX;
  r=bf16_pad(rows); kp=bf16_pad(k);
  return kp && r>SIZE_MAX/2/kp ? SIZE_MAX : 2*r*kp;
}
size_t bcir_tensor_mm_bf16_workspace(size_t m, size_t n, size_t k) {
  if (!m || !n || !k) return 0;
  return bf16_sum(bf16_sum(bf16_strip(m,k),bf16_strip(n,k)),BF16_ALIGN);
}
/* the bytes a group packs: an operand an earlier call packs is shared, an operand pre names (a
 * panel, BF16-PANELS) neither packed nor shared */
static size_t bf16_need(const bcir_tensor_mm_call *chain, size_t nchain,
    const bcir_tensor_mm_call *y, size_t ny, const bcir_tensor_bf16_panels *pre) {
  size_t q,r,total=0,calls=nchain+ny;
  for (q=0;q<calls;q++) {
    const bcir_tensor_mm_call *x=bf16_nth(chain,nchain,y,q);
    int sa=pre && pre[q].a,sb=pre && pre[q].b;
    if (!bf16_packs(x)) continue;
    for (r=0;r<q;r++) {
      const bcir_tensor_mm_call *w=bf16_nth(chain,nchain,y,r);
      if (bf16_packs(w)) {
        sa|=!(pre && pre[r].a) && bf16_same_a(w,x); sb|=!(pre && pre[r].b) && bf16_same_b(w,x);
      }
    }
    if (!sa) total=bf16_sum(total,bf16_strip(x->m,x->k));
    if (!sb) total=bf16_sum(total,bf16_strip(x->n,x->k));
  }
  return total ? bf16_sum(total,BF16_ALIGN) : 0;
}
size_t bcir_tensor_mm_bf16_group_workspace(const bcir_tensor_mm_call *chain, size_t nchain,
    const bcir_tensor_mm_call *y, size_t ny) {
  return bf16_need(chain,nchain,y,ny,NULL);
}
size_t bcir_tensor_mm_bf16_group_panels_workspace(const bcir_tensor_mm_call *chain, size_t nchain,
    const bcir_tensor_mm_call *y, size_t ny, const bcir_tensor_bf16_panels *pre) {
  return bf16_need(chain,nchain,y,ny,pre);
}
/* the workspace's first 64-byte boundary and the bytes from it */
static uint16_t *bf16_base(void *workspace, size_t *bytes) {
  uintptr_t p=(uintptr_t)workspace,q=(p+BF16_ALIGN-1)/BF16_ALIGN*BF16_ALIGN;
  if (!workspace || *bytes<q-p) { *bytes=0; return NULL; }
  *bytes-=q-p; return (uint16_t *)(void *)((unsigned char *)workspace+(q-p));
}
/* One call, alone: whole when its packs fit, else in chunks -- all of A and columns of B as many
 * as fit, or rows and columns both -- each chunk packed, then computed; the caller has admitted
 * the workspace (bf16_least). */
static void bf16_call_run(bcir_tensor_pool *pool, int variant, const bcir_tensor_mm_call *x,
    void *workspace, size_t bytes) {
  bf16_job j; bf16_call *c=&j.call[0]; uint16_t *base; size_t kp,budget,rows,cols,i0,j0;
  if (!x->m || !x->n) return;
  bf16_args(c,x); j.ncall=1; j.nchain=0; j.variant=variant;
  if (!x->k) { bf16_prefix(&j); pool_run(pool,bf16_compute_task,&j,j.tasks[1]); return; }
  base=bf16_base(workspace,&bytes); kp=bf16_pad(x->k); budget=bytes/2/kp;
  if (c->mp+c->np<=budget) { rows=c->mp; cols=c->np; }
  else if (c->mp+32<=budget) { rows=c->mp; cols=(budget-c->mp)/32*32; }
  else { rows=budget/64*32; cols=(budget-rows)/32*32; }
  for (i0=0;i0<x->m;i0+=rows) {
    c->i0=i0; c->mp=bf16_pad(x->m)-i0<rows ? bf16_pad(x->m)-i0 : rows;
    c->wa=base; c->ap=base; c->pa=x->ta ? c->kt : c->mp/32; c->wb=NULL; c->pb=0; c->tr=c->tc=0;
    bf16_prefix(&j); pool_run(pool,bf16_pack_task,&j,j.packs[1]);
    for (j0=0;j0<x->n;j0+=cols) {
      c->j0=j0; c->np=bf16_pad(x->n)-j0<cols ? bf16_pad(x->n)-j0 : cols;
      c->wa=NULL; c->pa=0; c->wb=base+c->mp*kp; c->bp=c->wb; c->pb=x->tb ? c->np/32 : c->kt; c->tr=c->tc=0;
      bf16_prefix(&j); pool_run(pool,bf16_pack_task,&j,j.packs[1]);
      c->pb=0;
      c->tr=((x->m-i0<c->mp ? x->m-i0 : c->mp)+BF16_ROWS-1)/BF16_ROWS;
      c->tc=((x->n-j0<c->np ? x->n-j0 : c->np)+c->cols-1)/c->cols;
      bf16_prefix(&j); pool_run(pool,bf16_compute_task,&j,j.tasks[1]);
    }
  }
}
/* the least workspace a call runs in: a 32-row strip of A and a 32-column strip of B */
static size_t bf16_least(const bcir_tensor_mm_call *x) {
  return bf16_packs(x) ? bf16_sum(bf16_strip(64,x->k),BF16_ALIGN) : 0;
}
/* a call a panel feeds: whole tiles every way (BF16-PANELS) */
static int bf16_whole(const bcir_tensor_mm_call *x) {
  return x->m && x->n && x->k && !(x->m%32) && !(x->n%32) && !(x->k%32);
}
static int bf16_group_any(bcir_tensor_pool *pool, int variant, const bcir_tensor_mm_call *chain,
    size_t nchain, const bcir_tensor_mm_call *y, size_t ny, const bcir_tensor_bf16_panels *pre,
    void *workspace, size_t bytes) {
  size_t q,r,calls=nchain+ny,need,usable=bytes; int grouped,panels=0; bf16_job j; uint16_t *base;
  for (q=0;q<calls;q++) {  /* refused before anything is written */
    const bcir_tensor_mm_call *x=bf16_nth(chain,nchain,y,q);
    if (pre && (pre[q].a || pre[q].b)) {  /* the A pack converts alpha A, an A panel A alone */
      if (!bf16_whole(x) || (pre[q].a && x->alpha!=1.0f)) return 0;
      panels=1;
    }
    else if (bf16_packs(x) && (!workspace || bf16_least(x)>bytes)) return 0;
  }
  need=bf16_need(chain,nchain,y,ny,pre);
  grouped=nchain<=BCIR_TENSOR_MM_GROUP && ny<=BCIR_TENSOR_MM_GROUP && need<=bytes && (workspace || !need);
  for (q=1;q<nchain && grouped;q++) grouped=chain[q].c==chain[0].c && chain[q].m==chain[0].m && chain[q].n==chain[0].n;
  if (!grouped) {
    if (panels) return 0;  /* a call a panel feeds runs whole, in the one job */
    for (q=0;q<calls;q++) bf16_call_run(pool,variant,bf16_nth(chain,nchain,y,q),workspace,bytes);
    return 1;
  }
  base=bf16_base(workspace,&usable);
  j.ncall=calls; j.nchain=nchain; j.variant=variant;
  for (q=0;q<calls;q++) {
    const bcir_tensor_mm_call *x=bf16_nth(chain,nchain,y,q); bf16_call *c=&j.call[q];
    bf16_args(c,x);
    if (!bf16_packs(x)) continue;
    if (pre && pre[q].a) c->ap=pre[q].a;
    for (r=0;r<q && !c->ap;r++)
      if (!(pre && pre[r].a) && bf16_packs(bf16_nth(chain,nchain,y,r)) &&
          bf16_same_a(bf16_nth(chain,nchain,y,r),x)) c->ap=j.call[r].ap;
    if (!c->ap) { c->wa=base; c->ap=base; base+=c->mp*c->kt*32; c->pa=x->ta ? c->kt : c->mp/32; }
    if (pre && pre[q].b) c->bp=pre[q].b;
    for (r=0;r<q && !c->bp;r++)
      if (!(pre && pre[r].b) && bf16_packs(bf16_nth(chain,nchain,y,r)) &&
          bf16_same_b(bf16_nth(chain,nchain,y,r),x)) c->bp=j.call[r].bp;
    if (!c->bp) { c->wb=base; c->bp=base; base+=c->np*c->kt*32; c->pb=x->tb ? c->np/32 : c->kt; }
  }
  bf16_prefix(&j);
  pool_run(pool,bf16_pack_task,&j,j.packs[calls]);
  pool_run(pool,bf16_compute_task,&j,j.split+j.tasks[calls-nchain]);
  return 1;
}
static int bf16_widest(void) {
  int v=bcir_tensor_mm_bf16_variants();
  return v&8 ? 8 : v&4 ? 4 : 1;
}
/* PX-3b: the bf16 contract's instance for a request (mm_instance's rule over
 * bcir_tensor_mm_bf16_variants: 0 the widest, else 1 portable, 4 AVX-512 or 8 AMX when admitted). */
static int bf16_instance(int variant) {
  if (!variant) return bf16_widest();
  return (variant==1 || variant==4 || variant==8) && (bcir_tensor_mm_bf16_variants()&variant) ?
    variant : 0;
}
int bcir_tensor_mm_bf16_group_pool(bcir_tensor_pool *pool, const bcir_tensor_mm_call *chain,
    size_t nchain, const bcir_tensor_mm_call *y, size_t ny, void *workspace, size_t bytes) {
  return bf16_group_any(pool,bf16_widest(),chain,nchain,y,ny,NULL,workspace,bytes);
}
int bcir_tensor_mm_bf16_group_panels_pool(bcir_tensor_pool *pool,
    const bcir_tensor_mm_call *chain, size_t nchain, const bcir_tensor_mm_call *y, size_t ny,
    const bcir_tensor_bf16_panels *pre, void *workspace, size_t bytes) {
  return bf16_group_any(pool,bf16_widest(),chain,nchain,y,ny,pre,workspace,bytes);
}
int bcir_tensor_mm_bf16_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c, void *workspace,
    size_t bytes) {
  bcir_tensor_mm_call x;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.b=b; x.beta=beta; x.c=c;
  return bf16_group_any(pool,bf16_widest(),NULL,0,&x,1,NULL,workspace,bytes);
}
int bcir_tensor_mm_bf16_variant(int variant, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c, void *workspace,
    size_t bytes) {
  bcir_tensor_mm_call x;
  if ((variant!=1 && variant!=4 && variant!=8) || !(bcir_tensor_mm_bf16_variants()&variant)) return 0;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.b=b; x.beta=beta; x.c=c;
  return bf16_group_any(NULL,variant,NULL,0,&x,1,NULL,workspace,bytes) ? variant : 0;
}
/* --- PX-3b (DISPATCH): the GEMM entries with the instance named (bcir_tensor.h) ---------------- */
int bcir_tensor_mm_contract_variants(int contract) {
  return contract==3 ? bcir_tensor_mm_bf16_variants() : contract==1 || contract==2 ?
    bcir_tensor_mm_fused_variants() : contract==0 ? bcir_tensor_mm_variants() : 0;
}
int bcir_tensor_mm_contract_variant_pool(bcir_tensor_pool *pool, int contract, int variant, int ta,
    int tb, size_t m, size_t n, size_t k, float alpha, const float *a, const float *b, float beta,
    float *c, void *workspace, size_t bytes) {
  int fused,v;
  if (contract==3) {
    bcir_tensor_mm_call x;
    if (!(v=bf16_instance(variant))) return 0;
    x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.b=b; x.beta=beta; x.c=c;
    return bf16_group_any(pool,v,NULL,0,&x,1,NULL,workspace,bytes) ? v : 0;
  }
  if (contract<0 || contract>2) return 0;
  fused=mm_family(contract);
  if (!(v=mm_instance(fused,variant))) return 0;
  mm_pool_any(pool,fused,v,ta,tb,m,n,k,alpha,a,b,beta,c);
  return v;
}
int bcir_tensor_mm_pair_variant_pool(bcir_tensor_pool *pool, int contract, int variant,
    const bcir_tensor_mm_call *x, const bcir_tensor_mm_call *y) {
  int fused,v;
  if (contract<0 || contract>2) return 0;
  fused=mm_family(contract);
  if (!(v=mm_instance(fused,variant))) return 0;
  mm_pair_any(pool,fused,v,x,y);
  return v;
}
int bcir_tensor_mm_group_variant_pool(bcir_tensor_pool *pool, int contract, int variant,
    const bcir_tensor_mm_call *chain, size_t nchain, const bcir_tensor_mm_call *y, size_t ny,
    const bcir_tensor_bf16_panels *pre, void *workspace, size_t bytes) {
  int fused,v;
  if (contract==3) {
    if (!(v=bf16_instance(variant))) return 0;
    return bf16_group_any(pool,v,chain,nchain,y,ny,pre,workspace,bytes) ? v : 0;
  }
  if (contract<0 || contract>2 || pre) return 0;
  fused=mm_family(contract);
  if (!(v=mm_instance(fused,variant))) return 0;
  mm_group_any(pool,fused,v,chain,nchain,y,ny);
  return v;
}
/* --- the writers' verdicts (NDT-SCAN-W) --------------------------------------------------------
 * A task that wrote a region reports whether it is finite into its own byte of the caller's
 * BCIR_TENSOR_SCAN_TASKS bytes (on the caller's stack; the caller ORs them), reading it while it is
 * in the core's cache. A job of more tasks than that runs without the bytes, and the regions it
 * reads are scanned after it on the pool, in at most BCIR_TENSOR_SCAN_TASKS chunks -- the same
 * verdict. !(|x| <= FLT_MAX) is exactly !isfinite(x); a block with no exit vectorizes. */
#ifndef BCIR_TENSOR_SCAN_TASKS
#define BCIR_TENSOR_SCAN_TASKS 4096
#endif
#define BCIR_TENSOR_SCAN_CHUNK 16384
static int span_finite(const float *x, size_t n) {
  size_t i=0,j,end;
  while (i<n) {
    int bad=0;
    end=n-i<1024 ? n : i+1024;
    for (j=i;j<end;j++) bad|=!(fabsf(x[j])<=FLT_MAX);
    if (bad) return 0;
    i=end;
  }
  return 1;
}
#ifdef BCIR_MM_DISPATCH
/* The same verdict 16 and 8 lanes at a time: NLE_UQ is "not <=, or unordered" */
__attribute__((target("avx512f"))) static int span_finite_avx512(const float *x, size_t n) {
  const __m512 lim=_mm512_set1_ps(FLT_MAX); size_t i=0;
  for (;i+64<=n;i+=64) {
    __mmask16 m=_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x+i)),lim,_CMP_NLE_UQ);
    m|=_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x+i+16)),lim,_CMP_NLE_UQ);
    m|=_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x+i+32)),lim,_CMP_NLE_UQ);
    m|=_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x+i+48)),lim,_CMP_NLE_UQ);
    if (m) return 0;
  }
  for (;i<n;i+=16) {
    __mmask16 lanes=(__mmask16)(n-i>=16 ? 0xffffu : (1u<<(n-i))-1u);
    if (_mm512_mask_cmp_ps_mask(lanes,_mm512_abs_ps(_mm512_maskz_loadu_ps(lanes,x+i)),lim,_CMP_NLE_UQ))
      return 0;
  }
  return 1;
}
__attribute__((target("avx2"))) static int span_finite_avx2(const float *x, size_t n) {
  const __m256 lim=_mm256_set1_ps(FLT_MAX),abs=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  size_t i=0;
  for (;i+32<=n;i+=32) {
    __m256 m=_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x+i),abs),lim,_CMP_NLE_UQ);
    m=_mm256_or_ps(m,_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x+i+8),abs),lim,_CMP_NLE_UQ));
    m=_mm256_or_ps(m,_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x+i+16),abs),lim,_CMP_NLE_UQ));
    m=_mm256_or_ps(m,_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x+i+24),abs),lim,_CMP_NLE_UQ));
    if (_mm256_movemask_ps(m)) return 0;
  }
  return span_finite(x+i,n-i);
}
#endif
static int span_finite_run(int variant, const float *x, size_t n) {
#ifdef BCIR_MM_DISPATCH
  if (variant==4) return span_finite_avx512(x,n);
  if (variant==2) return span_finite_avx2(x,n);
#endif
  (void)variant; return span_finite(x,n);
}
#ifdef BCIR_MM_DISPATCH
/* rows of `width` floats `ld` apart, one verdict for them all: the attention's heads, d at a time */
__attribute__((target("avx512f"))) static int rows_finite_avx512(const float *x, size_t rows,
    size_t width, size_t ld) {
  const __m512 lim=_mm512_set1_ps(FLT_MAX); size_t r,c,w16=width/16*16;
  __mmask16 tail=(__mmask16)((1u<<(width%16))-1u),m=0;
  for (r=0;r<rows;r++) {
    const float *row=x+r*ld;
    for (c=0;c<w16;c+=16) m|=_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(row+c)),lim,_CMP_NLE_UQ);
    if (tail) m|=_mm512_mask_cmp_ps_mask(tail,_mm512_abs_ps(_mm512_maskz_loadu_ps(tail,row+w16)),lim,_CMP_NLE_UQ);
  }
  return !m;
}
#endif
static int rows_finite(int variant, const float *x, size_t rows, size_t width, size_t ld) {
  size_t r;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) return rows_finite_avx512(x,rows,width,ld);
#endif
  for (r=0;r<rows;r++) if (!span_finite_run(variant,x+r*ld,width)) return 0;
  return 1;
}
typedef struct scan_job { const float *x; size_t n,chunk; unsigned char *bad; int variant; } scan_job;
static void scan_task(void *ctx, size_t task) {
  const scan_job *j=(const scan_job *)ctx; size_t i0=task*j->chunk;
  j->bad[task]=(unsigned char)!span_finite_run(j->variant,j->x+i0,j->n-i0<j->chunk ? j->n-i0 : j->chunk);
}
static int scan_pool(bcir_tensor_pool *pool, const float *x, size_t n) {
  unsigned char bad[BCIR_TENSOR_SCAN_TASKS]; scan_job j; size_t i,tasks; int any=0;
  if (!n) return 1;
  j.x=x; j.n=n; j.chunk=n/BCIR_TENSOR_SCAN_TASKS+(n%BCIR_TENSOR_SCAN_TASKS!=0); j.bad=bad;
  j.variant=mm_widest();
  if (j.chunk<BCIR_TENSOR_SCAN_CHUNK) j.chunk=BCIR_TENSOR_SCAN_CHUNK;
  tasks=n/j.chunk+(n%j.chunk!=0);
  pool_run(pool,scan_task,&j,tasks);
  for (i=0;i<tasks;i++) any|=bad[i];
  return !any;
}
/* The bytes a job of `tasks` tasks reports into: none without a verdict to give, or past the bound */
static unsigned char *scan_bytes(const int *bad, unsigned char *bytes, size_t tasks) {
  return bad && tasks<=BCIR_TENSOR_SCAN_TASKS ? bytes : NULL;
}
static void scan_fold(int *bad, const unsigned char *bytes, size_t tasks) {
  size_t i; unsigned char any=0;
  for (i=0;i<tasks;i++) any|=bytes[i];
  if (any) *bad=1;
}
/* Elements in ranges of BCIR_EW_CHUNK: y[i] depends on x[i] alone. */
#define BCIR_EW_CHUNK 8192
typedef struct ew_job {
  size_t n; const float *x,*u,*dy; float *y,*du; int op,variant; unsigned char *bad;
} ew_job;
enum { EW_SILU, EW_ADD, EW_MUL, EW_SWIGLU, EW_SILU2, EW_SWIGLU2, EW_SILUMUL, EW_SILU2MUL };
static float sigmoid(float x);
static void ew_task(void *ctx, size_t task) {
  ew_job *j=(ew_job *)ctx; size_t i0=task*BCIR_EW_CHUNK,i1=i0+BCIR_EW_CHUNK<j->n ? i0+BCIR_EW_CHUNK : j->n;
  switch (j->op) {
    case EW_SILU: bcir_tensor_silu(NULL,i1-i0,j->x+i0,j->y+i0); break;
    case EW_ADD: { size_t i; for (i=i0;i<i1;i++) j->y[i]+=j->x[i]; } break;
    case EW_MUL: { size_t i; for (i=i0;i<i1;i++) j->y[i]*=j->x[i]; } break;
    case EW_SILU2: silu2_run(j->variant,i1-i0,j->x+i0,j->y+i0); break;
    case EW_SWIGLU2: swiglu2_run(j->variant,i1-i0,j->x+i0,j->u+i0,j->dy+i0,j->y+i0,j->du+i0); break;
    case EW_SILUMUL: { size_t i; for (i=i0;i<i1;i++) { float s=j->x[i]*sigmoid(j->x[i]); j->y[i]=s*j->u[i]; } } break;
    case EW_SILU2MUL: silu2mul_run(j->variant,i1-i0,j->x+i0,j->u+i0,j->y+i0); break;
    default: bcir_tensor_swiglu_backward(i1-i0,j->x+i0,j->u+i0,j->dy+i0,j->y+i0,j->du+i0); break;
  }
  if (j->bad) j->bad[task]=(unsigned char)!span_finite_run(j->variant,j->y+i0,i1-i0);
}
static void ew_run(bcir_tensor_pool *pool, ew_job *j) {
  j->variant=mm_widest(); j->bad=NULL;
  pool_run(pool,ew_task,j,(j->n+BCIR_EW_CHUNK-1)/BCIR_EW_CHUNK);
}
/* The same job with its verdict on y (NDT-SCAN-W). */
static void ew_scan_run(bcir_tensor_pool *pool, ew_job *j, int *bad) {
  unsigned char bytes[BCIR_TENSOR_SCAN_TASKS]; size_t tasks=(j->n+BCIR_EW_CHUNK-1)/BCIR_EW_CHUNK;
  j->variant=mm_widest(); j->bad=scan_bytes(bad,bytes,tasks);
  pool_run(pool,ew_task,j,tasks);
  if (!bad) return;
  if (j->bad) scan_fold(bad,bytes,tasks);
  else if (!scan_pool(pool,j->y,j->n)) *bad=1;
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
void bcir_tensor_silu_mul_pool(bcir_tensor_pool *pool, size_t n, const float *x, const float *u,
    float *y) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=u; j.dy=NULL; j.du=NULL; j.op=EW_SILUMUL; ew_run(pool,&j);
}
void bcir_tensor_silu_mul_v2_pool(bcir_tensor_pool *pool, size_t n, const float *x, const float *u,
    float *y) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=u; j.dy=NULL; j.du=NULL; j.op=EW_SILU2MUL; ew_run(pool,&j);
}
void bcir_tensor_silu_mul_scan_pool(bcir_tensor_pool *pool, size_t n, const float *x,
    const float *u, float *y, int *bad) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=u; j.dy=NULL; j.du=NULL; j.op=EW_SILUMUL; ew_scan_run(pool,&j,bad);
}
void bcir_tensor_silu_mul_v2_scan_pool(bcir_tensor_pool *pool, size_t n, const float *x,
    const float *u, float *y, int *bad) {
  ew_job j; j.n=n; j.x=x; j.y=y; j.u=u; j.dy=NULL; j.du=NULL; j.op=EW_SILU2MUL; ew_scan_run(pool,&j,bad);
}
void bcir_tensor_swiglu_backward_v2_pool(bcir_tensor_pool *pool, size_t n, const float *g,
    const float *u, const float *dy, float *dg, float *du) {
  ew_job j; j.n=n; j.x=g; j.u=u; j.dy=dy; j.y=dg; j.du=du; j.op=EW_SWIGLU2; ew_run(pool,&j);
}
/* --- the bf16 contract's operands written by their producers (BF16-PANELS) ---------------------
 * A writer computes an r x c tensor's binary32 values a row at a time (PANEL_COLS columns a call),
 * by the functions the binary32 entries call -- each element its value bit for bit, they being
 * elementwise -- and stores each value's conversion where the bf16 entries' pack puts it for a whole
 * call (bf16_pack_*_generic): as A, row i's depth l at tile (i/16, l/32), row i%16, lane l%32. A
 * 32-row block done, its second panel comes from that A panel, bf16 to bf16: as B, depth l's column
 * j at tile (j/16, l/32), pair (l%32)/2, lane 2*(j%16)+l%2 -- a pair of rows' 16 columns as one tile
 * row of 16 words -- or as A^T, A's place of (j, l): those 16 x 16 words transposed. A task is a range
 * of 32-row blocks, at most BCIR_TENSOR_SCAN_TASKS of them, so the verdict is always its tasks' bytes,
 * and every element of every panel is written by the one task that owns its rows. */
#define PANEL_COLS 1024
static size_t panel_a_at(size_t depth, size_t i, size_t l) {
  return ((i/16)*(depth/32)+l/32)*BF16_TILE+(i%16)*32+l%32;
}
static size_t panel_b_at(size_t depth, size_t l, size_t j) {
  return ((j/16)*(depth/32)+l/32)*BF16_TILE+((l%32)/2)*32+2*(j%16)+l%2;
}
enum { PANEL_SILU1, PANEL_SILU2, PANEL_SWIGLU1, PANEL_SWIGLU2 };
typedef struct panel_job {
  size_t r,c,per;          /* rows, columns, 32-row blocks a task */
  int op,variant,conv,at;  /* the values; their kernels' variant; the conversion's; second panel A^T */
  const float *x,*u,*dy;
  uint16_t *pa[2],*pb[2];  /* each value's panel as A and its second panel (as B, or as A^T) */
  unsigned char *bad;
} panel_job;
/* row i's columns [c0, c0+w): value 0 into v0, value 1 into v1 */
static void panel_values(const panel_job *j, size_t i, size_t c0, size_t w, float *v0, float *v1) {
  size_t o=i*j->c+c0,q;
  switch (j->op) {
    case PANEL_SILU1: for (q=0;q<w;q++) { float s=j->x[o+q]*sigmoid(j->x[o+q]); v0[q]=s*j->u[o+q]; } break;
    case PANEL_SILU2: silu2mul_run(j->variant,w,j->x+o,j->u+o,v0); break;
    case PANEL_SWIGLU1: bcir_tensor_swiglu_backward(w,j->x+o,j->u+o,j->dy+o,v0,v1); break;
    default: swiglu2_run(j->variant,w,j->x+o,j->u+o,j->dy+o,v0,v1); break;
  }
}
static void panel_a_generic(const panel_job *j, int k, size_t i, size_t c0, size_t w, const float *v) {
  size_t q;
  for (q=0;q<w;q++) j->pa[k][panel_a_at(j->c,i,c0+q)]=bf16_of(v[q]);
}
/* the second panel of rows [r0, r0+32), element by element from the A panel */
static void panel_second_generic(const panel_job *j, int k, size_t r0) {
  size_t rr,l;
  for (rr=0;rr<32;rr++) for (l=0;l<j->c;l++) {
    uint16_t h=j->pa[k][panel_a_at(j->c,r0+rr,l)];
    if (j->at) j->pb[k][panel_a_at(j->r,l,r0+rr)]=h; else j->pb[k][panel_b_at(j->r,r0+rr,l)]=h;
  }
}
#ifdef BCIR_MM_DISPATCH
/* By vector: 32 columns of a row as one A tile row (the packs' conversion, BF16_PACK_BODIES); a pair
 * of rows' 16 columns read back from the A panel and interleaved as 16 words (the B pack's pairs, the
 * even row first) -- B's tile row, or, 16 of them transposed (kb512_transpose16 moves bit patterns
 * only), A^T's tile. */
#define PANEL_A_BODY(NAME, TARGET, CVT) \
TARGET static void NAME##_panel_a(const panel_job *j, int k, size_t i, size_t c0, size_t w, \
    const float *v) { \
  size_t q; \
  for (q=0;q<w;q+=32) \
    _mm512_storeu_si512(j->pa[k]+panel_a_at(j->c,i,c0+q),CVT(_mm512_loadu_ps(v+q),_mm512_loadu_ps(v+q+16))); \
}
PANEL_A_BODY(bf512, BF16_AVX512BW, bf512_cvt)
#ifdef BCIR_BF16_AMX
PANEL_A_BODY(bfne, BF16_AMXHOST, bfne_cvt)
#endif
BF16_AVX512BW static void bf512_panel_second(const panel_job *j, int k, size_t r0) {
  size_t jb,p; _Alignas(64) float wd[256];
  for (jb=0;jb<j->c/16;jb++) {
    for (p=0;p<16;p++) {
      const uint16_t *e=j->pa[k]+panel_a_at(j->c,r0+2*p,16*jb),*o=j->pa[k]+panel_a_at(j->c,r0+2*p+1,16*jb);
      __m512i pr=BF512_PAIRS_BW(_mm512_inserti64x4(_mm512_castsi256_si512(
          _mm256_loadu_si256((const __m256i *)(const void *)e)),_mm256_loadu_si256((const __m256i *)(const void *)o),1));
      if (j->at) _mm512_store_si512(wd+16*p,pr);
      else _mm512_storeu_si512(j->pb[k]+panel_b_at(j->r,r0+2*p,16*jb),pr);
    }
    if (j->at) kb512_transpose16(wd,16,(float *)(void *)(j->pb[k]+panel_a_at(j->r,16*jb,r0)),16);
  }
}
#endif
static void panel_a(const panel_job *j, int k, size_t i, size_t c0, size_t w, const float *v) {
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (j->conv==8) { bfne_panel_a(j,k,i,c0,w,v); return; }
#endif
  if (j->conv>=4) { bf512_panel_a(j,k,i,c0,w,v); return; }
#endif
  panel_a_generic(j,k,i,c0,w,v);
}
static void panel_second(const panel_job *j, int k, size_t r0) {
#ifdef BCIR_MM_DISPATCH
  if (j->conv>=4) { bf512_panel_second(j,k,r0); return; }  /* bit moves: one body for both hosts */
#endif
  panel_second_generic(j,k,r0);
}
static void panel_task(void *ctx, size_t t) {
  const panel_job *j=(const panel_job *)ctx; size_t b0=t*j->per,nb=j->r/32,b,rr,c0;
  size_t b1=b0+j->per<nb ? b0+j->per : nb; int two=j->op>=PANEL_SWIGLU1,bad=0;
  _Alignas(64) float v0[PANEL_COLS],v1[PANEL_COLS];
  for (b=b0;b<b1;b++) {
    for (rr=0;rr<32;rr++) for (c0=0;c0<j->c;c0+=PANEL_COLS) {
      size_t w=j->c-c0<PANEL_COLS ? j->c-c0 : PANEL_COLS,i=b*32+rr;
      panel_values(j,i,c0,w,v0,v1);
      if (j->bad) bad|=!span_finite_run(j->variant,v0,w);
      panel_a(j,0,i,c0,w,v0);
      if (two) panel_a(j,1,i,c0,w,v1);
    }
    panel_second(j,0,b*32);
    if (two) panel_second(j,1,b*32);
  }
  if (j->bad) j->bad[t]=(unsigned char)bad;
}
#ifdef BCIR_DT_PANELS_PROBE
/* The trainer harness's record of the values the panel writers' jobs computed -- from each job's own
 * fields, not what a caller meant to pass: bit 1 << op for every job that ran since the last read
 * (1 v1's SiLU product, 2 v2's, 4 v1's SwiGLU VJP, 8 v2's); a read starts the record again. A build
 * defines BCIR_DT_PANELS_PROBE to have it (the harness's own builds do; the library does not). */
static unsigned panel_ops_seen;
unsigned bcir_tensor_panels_probe(void);
unsigned bcir_tensor_panels_probe(void) { unsigned u=panel_ops_seen; panel_ops_seen=0; return u; }
#endif
static int panel_run(bcir_tensor_pool *pool, panel_job *j, int *bad) {
  unsigned char bytes[BCIR_TENSOR_SCAN_TASKS]; size_t blocks,tasks;
  if (j->r%32 || j->c%32) return 0;
  if (!j->r || !j->c) return 1;
  blocks=j->r/32; tasks=blocks<BCIR_TENSOR_SCAN_TASKS ? blocks : BCIR_TENSOR_SCAN_TASKS;
  j->per=(blocks+tasks-1)/tasks; tasks=(blocks+j->per-1)/j->per;
  j->variant=mm_widest(); j->conv=bf16_widest(); j->bad=bad ? bytes : NULL;
#ifdef BCIR_DT_PANELS_PROBE
  panel_ops_seen|=1u<<j->op;
#endif
  pool_run(pool,panel_task,j,tasks);
  if (bad) scan_fold(bad,bytes,tasks);
  return 1;
}
int bcir_tensor_silu_mul_panels_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *x, const float *u, unsigned short *a, unsigned short *b, int *bad) {
  panel_job j;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.op=v2 ? PANEL_SILU2 : PANEL_SILU1; j.x=x; j.u=u;
  j.pa[0]=a; j.pb[0]=b; j.at=0;
  return panel_run(pool,&j,bad);
}
int bcir_tensor_swiglu_backward_panels_pool(bcir_tensor_pool *pool, int v2, size_t r, size_t c,
    const float *g, const float *u, const float *dy, unsigned short *dga, unsigned short *dgt,
    unsigned short *dua, unsigned short *dut) {
  panel_job j;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.op=v2 ? PANEL_SWIGLU2 : PANEL_SWIGLU1; j.x=g; j.u=u;
  j.dy=dy; j.pa[0]=dga; j.pb[0]=dgt; j.pa[1]=dua; j.pb[1]=dut; j.at=1;
  return panel_run(pool,&j,NULL);
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
typedef struct rms_job {
  size_t rows,width; float eps; const float *x,*g; float *y; int variant; unsigned char *bad;
} rms_job;
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
  } else
#endif
  bcir_tensor_rms(r1-r0,j->width,j->eps,j->x+r0*j->width,j->g,j->y+r0*j->width);
  if (j->bad) j->bad[task]=(unsigned char)!span_finite_run(j->variant,j->y+r0*j->width,(r1-r0)*j->width);
}
static void rms_any(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, float *y, int *bad) {
  rms_job j; unsigned char bytes[BCIR_TENSOR_SCAN_TASKS]; size_t tasks=(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK;
  j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.y=y; j.variant=mm_widest();
  j.bad=scan_bytes(bad,bytes,tasks);
  pool_run(pool,rms_task,&j,tasks);
  if (!bad) return;
  if (j.bad) scan_fold(bad,bytes,tasks);
  else if (!scan_pool(pool,y,rows*width)) *bad=1;
}
void bcir_tensor_rms_pool(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, float *y) {
  rms_any(pool,rows,width,eps,x,g,y,NULL);
}
void bcir_tensor_rms_scan_pool(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, float *y, int *bad) {
  rms_any(pool,rows,width,eps,x,g,y,bad);
}
/* The RMSNorm VJP: rows in ranges of BCIR_ROW_CHUNK (each row's dx and 1/rms its own), then
 * columns in ranges of BCIR_COL_CHUNK, a column's gamma adjoint summed over the rows in row order
 * from the value it holds -- the serial kernel's operations, element for element. */
#define BCIR_COL_CHUNK 16
typedef struct rmsb_job {
  size_t rows,width; float eps; const float *x,*g,*dy; float *dx,*dg,*inv; int variant; unsigned how;
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
      if (j->how&BCIR_TENSOR_RMSB_ADD) for (c=0;c<w;c++) dxr[c]=dxr[c]+(dyr[c]*j->g[c]*inv-xr[c]*coeff);
      else for (c=0;c<w;c++) dxr[c]=dyr[c]*j->g[c]*inv-xr[c]*coeff;
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
    if (j->how&BCIR_TENSOR_RMSB_ADD) for (c=0;c<w;c++) dxr[c]=dxr[c]+(dyr[c]*j->g[c]*inv-xr[c]*coeff);
    else for (c=0;c<w;c++) dxr[c]=dyr[c]*j->g[c]*inv-xr[c]*coeff;
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
  if (j->how&BCIR_TENSOR_RMSB_FIRST) for (c=0;c<n;c++) acc[c]=0.0f;
  else for (c=0;c<n;c++) acc[c]=j->dg[c0+c];
  for (r=0;r<j->rows;r++) {
    const float *xr=j->x+r*w+c0,*dyr=j->dy+r*w+c0; float inv=j->inv[r];
    for (c=0;c<n;c++) acc[c]+=dyr[c]*xr[c]*inv;
  }
  for (c=0;c<n;c++) j->dg[c0+c]=acc[c];
}
void bcir_tensor_rms_backward_into_pool(bcir_tensor_pool *pool, size_t rows, size_t width,
    float eps, const float *x, const float *g, const float *dy, float *dx, float *dg, float *inv,
    unsigned how) {
  rmsb_job j; j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.dy=dy; j.dx=dx; j.dg=dg; j.inv=inv;
  j.variant=mm_widest(); j.how=how;
  pool_run(pool,rmsb_rows_task,&j,(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
  pool_run(pool,rmsb_cols_task,&j,(width+BCIR_COL_CHUNK-1)/BCIR_COL_CHUNK);
}
void bcir_tensor_rms_backward_pool(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, const float *dy, float *dx, float *dg, float *inv) {
  bcir_tensor_rms_backward_into_pool(pool,rows,width,eps,x,g,dy,dx,dg,inv,0u);
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
 * AVX-512 a whole block of 16 rows runs its maxima and sums as the lanes of one vector
 * (NDT-ATTN-4: att512_rowchain, att512_expsum) -- each row's chain the loop's, in its column order;
 * the maxima are taken over the raw scores and scaled, the scale folded into each exponential's
 * argument (ATTN-SOFT: the same values, one pass over the scores fewer). */
typedef struct att_job {
  size_t b,t,nh,nk,d,group; const float *q,*k,*v,*dy; float *p,*y,*dq,*dk,*dv; int variant,exp2;
  int fused;  /* ATTN-FMA: the products' every multiply-add one fmaf */
  unsigned char *bad;  /* the forward's verdict bytes, one a task (NDT-SCAN-W), or NULL */
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
 * only: att_rows runs a ragged tail through the v1 kernel, the same arithmetic. The context's and
 * the transposed products' diagonal terms -- the ones only some rows of a block take -- run as
 * loops over all eight rows with a row that does not take the term masked out of its sum
 * (merge-masked: its accumulator stays the one it held, NaN or not), so every register array is
 * indexed by a constant (NDT-ATTN-7); a loop bounded by the term made the arrays addressable,
 * and the block stored 21 of its 24 accumulators to the stack every step of its main loop. A
 * skipped row's term reads the probability or adjoint past its diagonal inside the block, the
 * +0.0 the scores stored there, and its product is discarded. */
#define BCIR_ATT_DC 128
__attribute__((target("avx512f"))) BCIR_MM_INLINE __mmask16 att512_lanes(size_t count) {
  return (__mmask16)(count>=16 ? 0xffffu : (1u<<count)-1u);
}
/* One term of a product: v1's rounded product and rounded sum, or under the fused attention
 * contract (ATTN-FMA) one fused multiply-add -- the product unrounded, one rounding. `f` is a
 * constant of each instantiation (the blocks are written once and inlined into a v1 and a fused
 * copy), so neither copy tests it. The masked form leaves a lane the mask excludes exactly as it
 * was, NaN or not, under both. */
#define ATT_MADD(f,acc,a,b) ((f) ? _mm512_fmadd_ps((a),(b),(acc)) : _mm512_add_ps((acc),_mm512_mul_ps((a),(b))))
#define ATT_MADD_MASK(f,acc,take,a,b) ((f) ? _mm512_mask3_fmadd_ps((a),(b),(acc),(take)) : \
    _mm512_mask_add_ps((acc),(take),(acc),_mm512_mul_ps((a),(b))))
#define ATT_MADD1(f,acc,a,b) ((f) ? fmaf((a),(b),(acc)) : (acc)+(a)*(b))
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_scores_any(int f, size_t rows, size_t d,
    const float *q, size_t ldq, const float *k, size_t ldk, float *s, size_t lds) {
  _Alignas(64) float kt[BCIR_ATT_DC*32];
  size_t j0,c0,i0,r,c,j;
  for (j0=0;j0<rows;j0+=32) {
    size_t nj=rows-j0<32 ? rows-j0 : 32;
    __mmask16 m0=att512_lanes(nj),m1=att512_lanes(nj>16 ? nj-16 : 0);
    for (c0=0;c0<d;c0+=BCIR_ATT_DC) {
      size_t nc=d-c0<BCIR_ATT_DC ? d-c0 : BCIR_ATT_DC;
      {  /* the keys' panel: whole 16 x 16 blocks by register transpose, the rest one by one */
        size_t jw=nj/16*16,cw=nc/16*16,jb,cb;
        for (jb=0;jb<jw;jb+=16) for (cb=0;cb<cw;cb+=16) kb512_transpose16(k+(j0+jb)*ldk+c0+cb,ldk,kt+cb*32+jb,32);
        for (c=0;c<nc;c++) for (j=c<cw ? jw : 0;j<32;j++) kt[c*32+j]=j<nj ? k[(j0+j)*ldk+c0+c] : 0.0f;
      }
      for (i0=j0;i0<rows;i0+=8) {
        __m512 a0[8],a1[8];
        for (r=0;r<8;r++) {
          const float *sr=s+(i0+r)*lds+j0; size_t keep=i0+r-j0+1<nj ? i0+r-j0+1 : nj;
          a0[r]=_mm512_maskz_loadu_ps(c0 ? att512_lanes(keep) : 0,sr);
          a1[r]=_mm512_maskz_loadu_ps(c0 ? att512_lanes(keep>16 ? keep-16 : 0) : 0,sr+16);
        }
        for (c=0;c<nc;c++) {
          __m512 b0=_mm512_load_ps(kt+c*32),b1=_mm512_load_ps(kt+c*32+16);
          for (r=0;r<8;r++) {
            __m512 qv=_mm512_set1_ps(q[(i0+r)*ldq+c0+c]);
            a0[r]=ATT_MADD(f,a0[r],qv,b0); a1[r]=ATT_MADD(f,a1[r],qv,b1);
          }
        }
        for (r=0;r<8;r++) {
          size_t i=i0+r,keep=i-j0+1<nj ? i-j0+1 : nj;
          float *sr=s+i*lds+j0;
          __mmask16 k0=att512_lanes(keep),k1=att512_lanes(keep>16 ? keep-16 : 0);
          if (c0+nc<d) { _mm512_mask_storeu_ps(sr,k0,a0[r]); _mm512_mask_storeu_ps(sr+16,k1,a1[r]); }
          else {  /* the last channel block: the block's future lanes as +0.0 */
            _mm512_mask_storeu_ps(sr,m0,_mm512_maskz_mov_ps(k0,a0[r]));
            _mm512_mask_storeu_ps(sr+16,m1,_mm512_maskz_mov_ps(k1,a1[r]));
          }
        }
      }
    }
  }
}
__attribute__((target("avx512f"))) static void att512_scores(size_t rows, size_t d,
    const float *q, size_t ldq, const float *k, size_t ldk, float *s, size_t lds) {
  att512_scores_any(0,rows,d,q,ldq,k,ldk,s,lds);
}
__attribute__((target("avx512f"))) static void att512_scores_fma(size_t rows, size_t d,
    const float *q, size_t ldq, const float *k, size_t ldk, float *s, size_t lds) {
  att512_scores_any(1,rows,d,q,ldq,k,ldk,s,lds);
}
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_context_any(int f, size_t rows, size_t d,
    const float *p, size_t ldp, const float *v, size_t ldv, float *y, size_t ldy, int first) {
  size_t c0,i0,r,j,dj;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (i0=0;i0<rows;i0+=8) {
      __m512 a0[8],a1[8],a2[8];
      for (r=0;r<8;r++) {
        const float *yr=y+(i0+r)*ldy+c0;
        a0[r]=_mm512_maskz_loadu_ps(first ? 0 : m0,yr); a1[r]=_mm512_maskz_loadu_ps(first ? 0 : m1,yr+16);
        a2[r]=_mm512_maskz_loadu_ps(first ? 0 : m2,yr+32);
      }
      for (j=0;j<=i0;j++) {  /* every row of the block takes key j */
        const float *vj=v+j*ldv+c0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_maskz_loadu_ps(m1,vj+16),
            b2=_mm512_maskz_loadu_ps(m2,vj+32);
        for (r=0;r<8;r++) {
          __m512 pv=_mm512_set1_ps(p[(i0+r)*ldp+j]);
          a0[r]=ATT_MADD(f,a0[r],pv,b0); a1[r]=ATT_MADD(f,a1[r],pv,b1);
          a2[r]=ATT_MADD(f,a2[r],pv,b2);
        }
      }
      for (dj=1;dj<8;dj++) {  /* key i0+dj: the rows r >= dj, the others masked out */
        const float *vj=v+(i0+dj)*ldv+c0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,vj),b1=_mm512_maskz_loadu_ps(m1,vj+16),
            b2=_mm512_maskz_loadu_ps(m2,vj+32);
        for (r=0;r<8;r++) {
          __mmask16 take=(__mmask16)(r>=dj ? 0xffffu : 0u);
          __m512 pv=_mm512_set1_ps(p[(i0+r)*ldp+i0+dj]);
          a0[r]=ATT_MADD_MASK(f,a0[r],take,pv,b0);
          a1[r]=ATT_MADD_MASK(f,a1[r],take,pv,b1);
          a2[r]=ATT_MADD_MASK(f,a2[r],take,pv,b2);
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
__attribute__((target("avx512f"))) static void att512_context(size_t rows, size_t d,
    const float *p, size_t ldp, const float *v, size_t ldv, float *y, size_t ldy, int first) {
  att512_context_any(0,rows,d,p,ldp,v,ldv,y,ldy,first);
}
__attribute__((target("avx512f"))) static void att512_context_fma(size_t rows, size_t d,
    const float *p, size_t ldp, const float *v, size_t ldv, float *y, size_t ldy, int first) {
  att512_context_any(1,rows,d,p,ldp,v,ldv,y,ldy,first);
}
/* out_j += sum over i >= j, ascending, of m_ij x_i (the KGE products: dv from P and dY, dk from dS
 * and Q): blocks of eight output rows j0..j0+7, the channels 48 at a time; term i reaches the rows
 * j <= i -- the block's diagonal terms some rows, every later term all eight. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_transposed_any(int f, size_t rows,
    size_t t, size_t d, const float *m, size_t ldm, const float *x, size_t ldx, float *out,
    size_t ldo, int first) {
  size_t c0,j0,r,i,di;
  for (c0=0;c0<d;c0+=48) {
    size_t nc=d-c0<48 ? d-c0 : 48;
    __mmask16 m0=att512_lanes(nc),m1=att512_lanes(nc>16 ? nc-16 : 0),m2=att512_lanes(nc>32 ? nc-32 : 0);
    for (j0=0;j0<rows;j0+=8) {
      __m512 a0[8],a1[8],a2[8];
      for (r=0;r<8;r++) {
        const float *o=out+(j0+r)*ldo+c0;
        a0[r]=_mm512_maskz_loadu_ps(first ? 0 : m0,o); a1[r]=_mm512_maskz_loadu_ps(first ? 0 : m1,o+16);
        a2[r]=_mm512_maskz_loadu_ps(first ? 0 : m2,o+32);
      }
      for (di=0;di<7;di++) {  /* term j0+di: the rows r <= di */
        const float *xi=x+(j0+di)*ldx+c0,*mi=m+(j0+di)*ldm+j0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_maskz_loadu_ps(m1,xi+16),
            b2=_mm512_maskz_loadu_ps(m2,xi+32);
        for (r=0;r<8;r++) {  /* the rows r <= di, the others masked out */
          __mmask16 take=(__mmask16)(r<=di ? 0xffffu : 0u);
          __m512 mv=_mm512_set1_ps(mi[r]);
          a0[r]=ATT_MADD_MASK(f,a0[r],take,mv,b0);
          a1[r]=ATT_MADD_MASK(f,a1[r],take,mv,b1);
          a2[r]=ATT_MADD_MASK(f,a2[r],take,mv,b2);
        }
      }
      for (i=j0+7;i<t;i++) {  /* every row of the block takes term i */
        const float *xi=x+i*ldx+c0,*mi=m+i*ldm+j0;
        __m512 b0=_mm512_maskz_loadu_ps(m0,xi),b1=_mm512_maskz_loadu_ps(m1,xi+16),
            b2=_mm512_maskz_loadu_ps(m2,xi+32);
        for (r=0;r<8;r++) {
          __m512 mv=_mm512_set1_ps(mi[r]);
          a0[r]=ATT_MADD(f,a0[r],mv,b0); a1[r]=ATT_MADD(f,a1[r],mv,b1);
          a2[r]=ATT_MADD(f,a2[r],mv,b2);
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
__attribute__((target("avx512f"))) static void att512_transposed(size_t rows, size_t t, size_t d,
    const float *m, size_t ldm, const float *x, size_t ldx, float *out, size_t ldo, int first) {
  att512_transposed_any(0,rows,t,d,m,ldm,x,ldx,out,ldo,first);
}
__attribute__((target("avx512f"))) static void att512_transposed_fma(size_t rows, size_t t,
    size_t d, const float *m, size_t ldm, const float *x, size_t ldx, float *out, size_t ldo,
    int first) {
  att512_transposed_any(1,rows,t,d,m,ldm,x,ldx,out,ldo,first);
}
#endif
#ifdef BCIR_MM_DISPATCH
/* The attention's row chains for 16 rows at once (NDT-ATTN-4): rows i0 .. i0+15, the first at `rows`,
 * the next ld floats on (the scores' t, or a scratch block's stride, BF16-ATTN); the 16 rows' blocks
 * of 16 columns transposed in registers so lane r carries row r's chain -- the softmax's running maximum (the
 * scalar `x > max ? x : max`, which is VMAXPS's operand rule exactly) and its sum, and the
 * backward's dot -- over the columns in order, a lane taking a column only where the row has it;
 * every row's value is the scalar loop's, bit for bit. */
__attribute__((target("avx512f"))) static void att512_rowchain(size_t ld, size_t i0, const float *rows,
    float *out, int sum) {
  _Alignas(64) float blk[256]; __m512 v=sum ? _mm512_setzero_ps() : _mm512_set1_ps(-INFINITY);
  size_t jb,c;
  for (jb=0;jb<=i0;jb+=16) {
    kb512_transpose16(rows+jb,ld,blk,16);
    for (c=0;c<16;c++) {
      __mmask16 k=(__mmask16)(jb<i0 ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_load_ps(blk+c*16);
      v=sum ? _mm512_mask_add_ps(v,k,v,x) : _mm512_mask_max_ps(v,k,x,v);
    }
  }
  _mm512_storeu_ps(out,v);
}
/* The v2 softmax's exponentials and sum for the same 16 rows, in the transposed blocks: each block's
 * scores scaled and less their lanes' scaled maxima -- the scaled score's rounding, then the
 * difference's, as when the scores were scaled in place first -- through the exponential's vector
 * instance whole (no row's remainder runs one element at a time), summed by lane where the row has
 * the column, the rest zeroed, and transposed back; each element is exp1(x*scale - max) and each
 * sum its row's chain in column order. */
__attribute__((target("avx512f"))) static void att512_expsum(size_t ld, size_t i0, float *rows,
    float scale, const float *maxv, float *sum) {
  _Alignas(64) float blk[256]; __m512 mx=_mm512_loadu_ps(maxv),sc=_mm512_set1_ps(scale),s=_mm512_setzero_ps();
  size_t jb,c;
  for (jb=0;jb<=i0;jb+=16) {
    kb512_transpose16(rows+jb,ld,blk,16);
    for (c=0;c<16;c++) _mm512_store_ps(blk+c*16,_mm512_sub_ps(_mm512_mul_ps(_mm512_load_ps(blk+c*16),sc),mx));
    exp_inplace_avx512(256,blk);
    for (c=0;c<16;c++) {
      __mmask16 k=(__mmask16)(jb<i0 ? 0xffffu : (0xffffu<<c)&0xffffu);
      __m512 x=_mm512_maskz_mov_ps(k,_mm512_load_ps(blk+c*16));
      s=_mm512_mask_add_ps(s,k,s,x);
      _mm512_store_ps(blk+c*16,x);
    }
    kb512_transpose16(blk,16,rows+jb,ld);
  }
  _mm512_storeu_ps(sum,s);
}
/* The backward's dot over one tile for 16 rows (rows i0+ig.. of the block, columns [jt, lim)):
 * the whole 16-column blocks as lanes, the rest of each row's columns after them in order. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_dot_any(int f, size_t t, size_t i0,
    size_t ig, size_t jt, size_t lim, const float *ph, const float *tile, size_t ldt, float *dot) {
  _Alignas(64) float bp[256],bd[256]; __m512 v=_mm512_loadu_ps(dot+ig); size_t j0,c,r,j,row=i0+ig;
  for (j0=jt;j0+16<=lim && j0<row+16;j0+=16) {
    kb512_transpose16(ph+row*t+j0,t,bp,16);
    kb512_transpose16(tile+ig*ldt+(j0-jt),ldt,bd,16);
    for (c=0;c<16;c++) {
      size_t col=j0+c;
      __mmask16 k=(__mmask16)(col<=row ? 0xffffu : (0xffffu<<(col-row))&0xffffu);
      v=ATT_MADD_MASK(f,v,k,_mm512_load_ps(bp+c*16),_mm512_load_ps(bd+c*16));
    }
  }
  _mm512_storeu_ps(dot+ig,v);
  for (r=0;r<16;r++) {
    const float *pi=ph+(row+r)*t; size_t end=row+r+1<lim ? row+r+1 : lim;
    for (j=j0;j<end;j++) dot[ig+r]=ATT_MADD1(f,dot[ig+r],pi[j],tile[(ig+r)*ldt+j-jt]);
  }
}
__attribute__((target("avx512f"))) static void att512_dot(size_t t, size_t i0, size_t ig, size_t jt,
    size_t lim, const float *ph, const float *tile, size_t ldt, float *dot) {
  att512_dot_any(0,t,i0,ig,jt,lim,ph,tile,ldt,dot);
}
__attribute__((target("avx512f"))) static void att512_dot_fma(size_t t, size_t i0, size_t ig,
    size_t jt, size_t lim, const float *ph, const float *tile, size_t ldt, float *dot) {
  att512_dot_any(1,t,i0,ig,jt,lim,ph,tile,ldt,dot);
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
__attribute__((target("avx512f"))) BCIR_MM_INLINE void att512_dp_any(int f, size_t rows, size_t i0,
    size_t jt, size_t cols, size_t d, const float *dy, size_t ldy, const float *vt, size_t t,
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
        const float *vc=vt+c*t+jt+jb;
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
        for (c=0;c<d;c++) acc=ATT_MADD1(f,acc,dy[(i0+ig)*ldy+c],vt[c*t+jt+jb+j]);
        tile[ig*ldt+jb+j]=acc;
      }
    }
  }
}
__attribute__((target("avx512f"))) static void att512_dp(size_t rows, size_t i0, size_t jt,
    size_t cols, size_t d, const float *dy, size_t ldy, const float *vt, size_t t, float *tile,
    size_t ldt) {
  att512_dp_any(0,rows,i0,jt,cols,d,dy,ldy,vt,t,tile,ldt);
}
__attribute__((target("avx512f"))) static void att512_dp_fma(size_t rows, size_t i0, size_t jt,
    size_t cols, size_t d, const float *dy, size_t ldy, const float *vt, size_t t, float *tile,
    size_t ldt) {
  att512_dp_any(1,rows,i0,jt,cols,d,dy,ldy,vt,t,tile,ldt);
}
#endif
enum { ATT_SCORES, ATT_CONTEXT, ATT_TRANSPOSED };
/* One of the attention's masked products over its t rows: the AVX-512 register blocks for the
 * whole blocks of eight, the rest (or every row, on another instance) through the v1 kernel.
 * With `first` the blocks start from +0.0 instead of what C holds -- the value a zero fill gave
 * them (NDT-ATTN-ZERO) -- so the caller fills only att_v1_rows' rows; the scores' blocks always
 * start a row from +0.0 and store their future lanes as +0.0. */
static size_t att_v1_rows(int variant, size_t t) {  /* the first row the v1 kernel takes */
  return variant==4 ? t/8*8 : 0;
}
static void att_rows(int variant, int fused, const mm_args *x, size_t t, int which, int first) {
  mm_range rest; size_t whole=0;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) {
    whole=t/8*8;
    if (which==ATT_SCORES) (fused ? att512_scores_fma : att512_scores)(whole,x->k,x->a,x->lda,x->b,x->ldb,x->c,x->ldc);
    else if (which==ATT_CONTEXT)
      (fused ? att512_context_fma : att512_context)(whole,x->n,x->a,x->lda,x->b,x->ldb,x->c,x->ldc,first);
    else (fused ? att512_transposed_fma : att512_transposed)(whole,t,x->n,x->a,x->lda,x->b,x->ldb,x->c,x->ldc,first);
  }
#endif
  (void)which; (void)first;
  if (whole==t || !x->m || !x->n || !x->k) return;
  rest.jb0=0; rest.jb1=x->n; rest.r0=whole; rest.r1=t;
  mm_run_range(variant,fused ? MM_FUSED : MM_V1,x,&rest);
}
/* A head's softmax rows over its scores ph (t x t): each row's scores j <= i scaled by 1/sqrtf(d),
 * less their running maximum, through the exponential (libm's, or the BCIR exponential's vector
 * instance under the exp contract), summed in index order and divided by the sum -- 16 rows at once
 * on AVX-512 (NDT-ATTN-4), each row's chain the scalar loop's; the future columns are not read. A
 * row's values depend on that row alone, so a block of n rows runs alone wherever it sits
 * (BF16-ATTN): its row u is the attention's row i0+u, at s + u*ld, i0 a multiple of 16. */
static void att_softmax_block(const att_job *a, float *s, size_t ld, size_t i0, size_t n) {
  size_t u0=0,j; float scale=1.0f/sqrtf((float)a->d);
#ifdef BCIR_MM_DISPATCH
  if (a->variant==4) for (;u0+16<=n;u0+=16) {
    float maxv[16],sum[16],*rows=s+u0*ld; size_t u,i=i0+u0;
    /* the raw scores' maxima, then scaled: rounding is monotonic and the scale positive, so a
     * row's scaled maximum is its scaled scores' (the scalar loop's), NaN skipped alike; each
     * exponential's argument is the scaled score less it, the same two roundings */
    att512_rowchain(ld,i,rows,maxv,0);
    for (u=0;u<16;u++) maxv[u]*=scale;
    if (a->exp2) att512_expsum(ld,i,rows,scale,maxv,sum);
    else {
      for (u=0;u<16;u++) { float *pi=rows+u*ld; for (j=0;j<=i+u;j++) pi[j]=expf(pi[j]*scale-maxv[u]); }
      att512_rowchain(ld,i,rows,sum,1);
    }
    for (u=0;u<16;u++) { float *pi=rows+u*ld; for (j=0;j<=i+u;j++) pi[j]/=sum[u]; }
  }
#endif
  for (;u0<n;u0++) {
    float *pi=s+u0*ld,maxv=-INFINITY,sum=0.0f; size_t i=i0+u0;
    for (j=0;j<=i;j++) { pi[j]*=scale; if (pi[j]>maxv) maxv=pi[j]; }
    if (a->exp2) {  /* v2: the row through the vector instance, then the sum in index order */
      for (j=0;j<=i;j++) pi[j]-=maxv;
      exp_inplace_run(a->variant,i+1,pi);
      for (j=0;j<=i;j++) sum+=pi[j];
    } else for (j=0;j<=i;j++) { pi[j]=expf(pi[j]-maxv); sum+=pi[j]; }
    for (j=0;j<=i;j++) pi[j]/=sum;
  }
}
static void att_softmax(const att_job *a, float *ph) { att_softmax_block(a,ph,a->t,0,a->t); }
/* A forward task's verdict (NDT-SCAN-W): the head's context rows, and its KV head's keys from the
 * group's first query head. */
static void att_verdict(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk);
  int ok=rows_finite(a->variant,a->y+(r*t*nh+h)*d,t,d,nh*d);
  if (ok && h%(nh/nk)==0) ok=rows_finite(a->variant,a->k+(r*t*nk+kh)*d,t,d,nk*d);
  a->bad[r*nh+h]=(unsigned char)!ok;
}
/* One (batch, query head): its scores, softmax rows and context; nothing shared with another. */
static void att_head(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),i;
  float *ph=a->p+(r*nh+h)*t*t; mm_args x;
  /* the products accumulate into what they write: this head's probabilities and its context
   * columns start at zero here, in the task that owns them (NDT-SCAN; the caller's fill was serial)
   * -- where the v1 kernel writes them; the AVX-512 blocks start from +0.0 (NDT-ATTN-ZERO), and of
   * a row's probabilities the fill covers the future columns past the blocks the row reaches */
  {
    size_t v1=att_v1_rows(a->variant,t);
    for (i=0;i<v1;i++) { size_t end=(i/32+1)*32<v1 ? (i/32+1)*32 : v1; memset(ph+i*t+end,0,(t-end)*sizeof(float)); }
    memset(ph+v1*t,0,(t-v1)*t*sizeof(float));
    for (i=v1;i<t;i++) memset(a->y+((r*t+i)*nh+h)*d,0,d*sizeof(float));
  }
  x.ta=0; x.tb=1; x.m=t; x.n=t; x.k=d; x.alpha=1.0f; x.a=a->q+(r*t*nh+h)*d; x.lda=nh*d;
  x.b=a->k+(r*t*nk+kh)*d; x.ldb=nk*d; x.c=ph; x.ldc=t; x.mode=MM_LOWER;
  att_rows(a->variant,a->fused,&x,t,ATT_SCORES,1);
  att_softmax(a,ph);
  x.tb=0; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=a->v+(r*t*nk+kh)*d; x.ldb=nk*d;
  x.c=a->y+(r*t*nh+h)*d; x.ldc=nh*d; x.mode=MM_KLE;
  att_rows(a->variant,a->fused,&x,t,ATT_CONTEXT,1);
  if (a->bad) att_verdict(a,r,h);
}
static void att_task(void *ctx, size_t task) {
  att_job *a=(att_job *)ctx;
  INSTANCE_SAW(INSTANCE_ATTN,a->variant);
  att_head(a,task/a->nh,task%a->nh);
}
#ifdef BCIR_DT_ATTENTION_PROBE
/* The trainer harness's record of the contract the attention's jobs ran (ATTN-FMA) -- what each
 * job computed with, from its own fields, not what a caller meant to pass: [0] the forwards', [1]
 * the backwards' (EXP is the forward's alone). Every job since the last read must have run one
 * contract, or the read is ~0u, as it is when no job ran; a read starts the record again. A build
 * defines BCIR_DT_ATTENTION_PROBE to have it (the harness's own builds do; the library does not). */
static unsigned attention_any_seen[2],attention_all_seen[2]={~0u,~0u};
static void attention_saw(int backward, unsigned contract) {
  attention_any_seen[backward]|=contract; attention_all_seen[backward]&=contract;
}
unsigned bcir_tensor_attention_probe(int backward);
unsigned bcir_tensor_attention_probe(int backward) {
  int d=backward ? 1 : 0;
  unsigned c=attention_any_seen[d]==attention_all_seen[d] ? attention_any_seen[d] : ~0u;
  attention_any_seen[d]=0u; attention_all_seen[d]=~0u;
  return c;
}
#endif
/* `variant` resolved (mm_instance over the products' family: the fused GEMM's under FUSED, v1's
 * otherwise) */
static void attention_any(bcir_tensor_pool *pool, int exp2, int fused, int variant, size_t b, size_t t,
    size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v, float *p, float *y,
    int *bad) {
  att_job a; unsigned char bytes[BCIR_TENSOR_SCAN_TASKS]; size_t tasks=b*nh;
  a.b=b; a.t=t; a.nh=nh; a.nk=nk; a.d=d; a.group=1; a.q=q; a.k=k; a.v=v; a.dy=NULL; a.p=p; a.y=y;
  a.dq=NULL; a.dk=NULL; a.dv=NULL; a.exp2=exp2; a.fused=fused;
  a.variant=variant;
#ifdef BCIR_DT_ATTENTION_PROBE
  attention_saw(0,(a.exp2 ? BCIR_TENSOR_ATTENTION_EXP : 0u)|(a.fused ? BCIR_TENSOR_ATTENTION_FUSED : 0u));
#endif
  a.bad=scan_bytes(bad,bytes,tasks);
  pool_run(pool,att_task,&a,tasks);
  if (!bad) return;
  if (a.bad) scan_fold(bad,bytes,tasks);
  else if (!scan_pool(pool,y,b*t*nh*d) || !scan_pool(pool,k,b*t*nk*d)) *bad=1;
}
void bcir_tensor_attention_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, float *y) {
  attention_any(pool,0,0,mm_instance(MM_V1,0),b,t,nh,nk,d,q,k,v,p,y,NULL);
}
void bcir_tensor_attention_v2_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, float *y) {
  attention_any(pool,1,0,mm_instance(MM_V1,0),b,t,nh,nk,d,q,k,v,p,y,NULL);
}
void bcir_tensor_attention_scan_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t nh,
    size_t nk, size_t d, const float *q, const float *k, const float *v, float *p, float *y,
    int *bad) {
  attention_any(pool,0,0,mm_instance(MM_V1,0),b,t,nh,nk,d,q,k,v,p,y,bad);
}
void bcir_tensor_attention_v2_scan_pool(bcir_tensor_pool *pool, size_t b, size_t t, size_t nh,
    size_t nk, size_t d, const float *q, const float *k, const float *v, float *p, float *y,
    int *bad) {
  attention_any(pool,1,0,mm_instance(MM_V1,0),b,t,nh,nk,d,q,k,v,p,y,bad);
}
void bcir_tensor_attention_contract_scan_pool(bcir_tensor_pool *pool, unsigned contract, size_t b,
    size_t t, size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v,
    float *p, float *y, int *bad) {
  bcir_tensor_attention_contract_variant_scan_pool(pool,0,contract,b,t,nh,nk,d,q,k,v,p,y,bad);
}
int bcir_tensor_attention_contract_variant_scan_pool(bcir_tensor_pool *pool, int variant,
    unsigned contract, size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q,
    const float *k, const float *v, float *p, float *y, int *bad) {
  int fused=(contract&BCIR_TENSOR_ATTENTION_FUSED)!=0,iv=mm_instance(fused ? MM_FUSED : MM_V1,variant);
  if (!iv) return 0;
  attention_any(pool,(contract&BCIR_TENSOR_ATTENTION_EXP)!=0,fused,iv,b,t,nh,nk,d,q,k,v,p,y,bad);
  return iv;
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
    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,h%(nh/nk)==0);
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
          if (vt) (a->fused ? att512_dp_fma : att512_dp)(rows,i0,jt,cols,d,dyh,ldh,vt,t,tile,BCIR_ATT_COLS);
          else
#endif
          {
            mm_range all;
            x.ta=0; x.tb=1; x.m=rows; x.n=cols; x.k=d; x.a=dyh+i0*ldh; x.lda=ldh; x.b=vk+jt*ldk;
            x.ldb=ldk; x.c=tile; x.ldc=BCIR_ATT_COLS; x.mode=MM_FULL;
            all.jb0=0; all.jb1=cols; all.r0=0; all.r1=rows;
            mm_run_range(variant,a->fused ? MM_FUSED : MM_V1,&x,&all);
          }
        }
        i=0;
#ifdef BCIR_MM_DISPATCH
        if (!pass && variant==4) for (;i+16<=rows;i+=16)
          (a->fused ? att512_dot_fma : att512_dot)(t,i0,i,jt,jt+cols,ph,tile,BCIR_ATT_COLS,dot);
#endif
        for (;i<rows;i++) {
          float *pi=ph+(i0+i)*t; size_t end=i0+i+1<jt+cols ? i0+i+1 : jt+cols;
          for (j=jt;j<end;j++) {
            float dp=tile[i*BCIR_ATT_COLS+j-jt];
            if (!pass) dot[i]=a->fused ? fmaf(pi[j],dp,dot[i]) : dot[i]+pi[j]*dp;
            else pi[j]=pi[j]*(dp-dot[i])*scale;
          }
        }
      }
    }
    x.ta=0; x.tb=0; x.m=t; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=kk; x.ldb=ldk;
    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;
    att_rows(variant,a->fused,&x,t,ATT_CONTEXT,1);
    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    att_rows(variant,a->fused,&x,t,ATT_TRANSPOSED,h%(nh/nk)==0);
  }
}
static void att_task_backward(void *ctx, size_t task) {
  att_job *a=(att_job *)ctx; size_t per=a->nh/a->group,r=task/per,first=(task%per)*a->group,h,i;
  size_t t=a->t,d=a->d,nh=a->nh,nk=a->nk,share=nh/nk,kh=first/share;
  INSTANCE_SAW(INSTANCE_ATTN,a->variant);
  /* the adjoints accumulate, so they start at zero here, in the task that owns them (NDT-SCAN; the
   * caller's fill was serial): this task's query heads' dq columns, and its KV head's dk and dv
   * columns where the task holds that KV head's first query head -- the task that runs before the
   * KV head's every accumulation, which is the whole group when group is the share */
  for (i=att_v1_rows(a->variant,t);i<t;i++) {  /* the v1 kernel's rows (NDT-ATTN-ZERO) */
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
void bcir_tensor_attention_backward_contract_pool(bcir_tensor_pool *pool, unsigned contract,
    size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q, const float *k,
    const float *v, float *p, const float *dy, float *dq, float *dk, float *dv) {
  bcir_tensor_attention_backward_contract_variant_pool(pool,0,contract,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,
      dv);
}
int bcir_tensor_attention_backward_contract_variant_pool(bcir_tensor_pool *pool, int variant,
    unsigned contract, size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q,
    const float *k, const float *v, float *p, const float *dy, float *dq, float *dk, float *dv) {
  att_job a; int fused=(contract&BCIR_TENSOR_ATTENTION_FUSED)!=0;
  int iv=mm_instance(fused ? MM_FUSED : MM_V1,variant);
  if (!iv) return 0;
  a.b=b; a.t=t; a.nh=nh; a.nk=nk; a.d=d; a.q=q; a.k=k; a.v=v; a.dy=dy; a.p=p; a.y=NULL;
  a.dq=dq; a.dk=dk; a.dv=dv; a.exp2=0; a.bad=NULL;
  a.fused=fused; a.variant=iv;
  a.group=nh/nk;
#ifdef BCIR_DT_ATTENTION_PROBE
  attention_saw(1,a.fused ? BCIR_TENSOR_ATTENTION_FUSED : 0u);
#endif
  pool_run(pool,att_task_backward,&a,b*nh/a.group);
  return iv;
}
void bcir_tensor_attention_backward_inplace_pool(bcir_tensor_pool *pool, size_t b, size_t t,
    size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v, float *p,
    const float *dy, float *dq, float *dk, float *dv) {
  bcir_tensor_attention_backward_contract_pool(pool,0,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv);
}
void bcir_tensor_attention_backward_inplace(size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, const float *dy,
    float *dq, float *dk, float *dv) {
  bcir_tensor_attention_backward_inplace_pool(NULL,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv);
}
/* --- the attention on bfloat16 operands (BF16-ATTN) ---------------------------------------------
 * The attention's six products -- the forward's scores S = Q K^T and context Y = P V, the backward's
 * dV = P^T dY, dP = dY V^T, dQ = dS K and dK = dS^T Q -- under the bf16 contract: each element is the
 * bf16 GEMM's (the operands rounded to bfloat16; the depth in 32-term tiles, each tile's two lanes
 * from +0.0 and their sum added to c, flushed as the AMX unit flushes), over the depth tiles the
 * causal mask reaches from the element's 32-row block: a query block's key tiles 0 .. its own (Y,
 * dQ), a key block's query tiles from its own on (dV, dK), every channel tile for S and dP, whose
 * blocks past the diagonal are never computed. A tile's terms past the mask are the +0.0
 * probabilities and score adjoints the arena holds there. The scaling, the softmax (v1, or the
 * BCIR exponential under EXP), the dot and the score adjoint are the v1 attention's, in its order;
 * the future scores and adjoints are +0.0. A KV head's dk and dv take its query heads in order, the
 * first from +0.0. Every instance -- portable, AVX-512 under MXCSR's FTZ and DAZ, AMX -- is the same
 * bits at every thread count.
 * A task is a (batch, KV head) in both directions: its keys and values are packed once for its
 * query heads, and each query block runs its scores, softmax, probabilities' strip and context (its
 * dP, dot, score adjoint, dS's strips and dQ) while its rows are in the core's cache, packing only
 * the depth tiles the mask reaches. dV and dK are computed transposed -- dV^T = dY^T P, dK^T = Q^T dS,
 * the same terms in the same lanes, a product of two bfloat16 being exact either way round -- so the
 * transposed packs are of dY and Q (dim x time) instead of P and dS (time x time); they accumulate
 * in the task's region over its query heads and are transposed into dv and dk at the end, values
 * moved, none computed. Every product runs whole 32 x 32 tiles where the shapes allow, its context
 * and dq rows through a scratch block whose padded columns are never copied. A task's region is a
 * worker's (or a task's, where a pool has more workers than tasks): no value depends on which. */
typedef struct att16_job {
  att_job a;            /* shapes, operands, the exponential bit, the verdict bytes */
  int bv;               /* the products' instance: 1 portable, 4 AVX-512, 8 AMX */
  unsigned char *ws;    /* the regions, from the workspace's first 64-byte boundary */
  size_t per;           /* bytes of one region */
  int by_worker;        /* a region a worker (threads <= tasks), else a region a task */
} att16_job;
static size_t att16_mul(size_t a, size_t b) { return a && b>SIZE_MAX/a ? SIZE_MAX : a*b; }
/* bytes of rows x cols binary32 scratch, a multiple of 64 */
static size_t att16_floats(size_t rows, size_t cols) {
  size_t n=att16_mul(att16_mul(rows,cols),4);
  return n>SIZE_MAX-63 ? SIZE_MAX : (n+63)/64*64;
}
/* a forward region: K^T's panel (B) and V's (B), a query head's Q (A) and P (A), a context block */
static size_t att16_forward_bytes(size_t t, size_t d) {
  size_t kv=bf16_sum(bf16_strip(t,d),bf16_strip(d,t));
  size_t head=bf16_sum(bf16_strip(t,d),bf16_strip(t,t));
  return bf16_sum(bf16_sum(kv,head),att16_floats(32,bf16_pad(d)));
}
/* a backward region: the KV head's V^T (B, for dP) and K (B, for dQ), dV^T and dK^T, a block of dP
 * and one of dQ, and a query head's dY (A), dY^T (A), P (B), dS (A), dS (B) and Q^T (A) */
static size_t att16_backward_bytes(size_t t, size_t d) {
  size_t tp=bf16_pad(t),dp=bf16_pad(d);
  size_t kv=bf16_sum(bf16_sum(bf16_strip(t,d),bf16_strip(d,t)),bf16_sum(att16_floats(dp,tp),att16_floats(dp,tp)));
  size_t blocks=bf16_sum(att16_floats(32,tp),att16_floats(32,dp));
  size_t head=bf16_sum(bf16_sum(bf16_strip(t,d),bf16_strip(d,t)),bf16_sum(bf16_strip(t,t),bf16_strip(t,t)));
  head=bf16_sum(head,bf16_sum(bf16_strip(t,t),bf16_strip(d,t)));
  return bf16_sum(bf16_sum(kv,blocks),head);
}
/* the regions of `regions` tasks at once, the larger direction's, and the alignment */
static size_t att16_regions(size_t regions, size_t t, size_t d) {
  size_t f=att16_mul(regions,att16_forward_bytes(t,d)),w=att16_mul(regions,att16_backward_bytes(t,d));
  return bf16_sum(f>w ? f : w,BF16_ALIGN);
}
size_t bcir_tensor_attention_bf16_workspace(size_t b, size_t t, size_t nh, size_t nk, size_t d) {
  if (!b || !t || !nh || !nk || !d) return 0;
  return att16_regions(att16_mul(b,nk),t,d);
}
/* a pool's workers no more than the tasks: a region a worker (the entries' by_worker), else a task */
size_t bcir_tensor_attention_bf16_pool_workspace(size_t b, size_t t, size_t nh, size_t nk,
    size_t d, unsigned threads) {
  size_t tasks;
  if (!b || !t || !nh || !nk || !d) return 0;
  tasks=att16_mul(b,nk);
  return att16_regions(threads && threads<=tasks ? threads : tasks,t,d);
}
/* one product as the bf16 GEMM's call (alpha 1) over the attention's strides, its panels pa, pb */
static void att16_call(bf16_call *c, int ta, int tb, size_t m, size_t n, size_t k, const float *a,
    size_t lda, const float *b, size_t ldb, float *cc, size_t ldc, uint16_t *pa, uint16_t *pb) {
  bcir_tensor_mm_call x;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=1.0f; x.a=a; x.b=b; x.beta=0.0f; x.c=cc;
  bf16_args(c,&x);
  c->x.lda=lda; c->x.ldb=ldb; c->x.ldc=ldc;
  c->wa=pa; c->ap=pa; c->wb=pb; c->bp=pb;
  c->prefetch=0;  /* its blocks are few and small: the prefetch cost the scores a quarter */
}
/* pack task s of a call's A or B panel by the instance's packs */
static void att16_pack_a(int bv, const bf16_call *c, size_t s) {
  (void)bv;
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (bv==8) { bfne_pack_a(c,s); return; }
#endif
  if (bv>=4) { bf512_pack_a(c,s); return; }
#endif
  bf16_pack_a_generic(c,s);
}
static void att16_pack_b(int bv, const bf16_call *c, size_t s) {
  (void)bv;
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (bv==8) { bfne_pack_b(c,s); return; }
#endif
  if (bv>=4) { bf512_pack_b(c,s); return; }
#endif
  bf16_pack_b_generic(c,s);
}
/* a call's whole panels, serially in the task: A where pack_a, B where pack_b */
static void att16_pack(int bv, const bf16_call *c, int pack_a, int pack_b) {
  size_t s,na=c->x.ta ? c->kt : c->mp/32,nb=c->x.tb ? c->np/32 : c->kt;
  for (s=0;pack_a && s<na;s++) att16_pack_a(bv,c,s);
  for (s=0;pack_b && s<nb;s++) att16_pack_b(bv,c,s);
}
/* depth tile s of a call's B panel (B not transposed) over the first np columns its products read */
static void att16_pack_depth(int bv, const bf16_call *c, size_t s, size_t np) {
  bf16_call x=*c;
  x.np=np<c->np ? np : c->np; att16_pack_b(bv,&x,s);
}
/* strip ib of a call's A panel (A not transposed) from a block of `rows` rows at src (lds) -- the
 * strip's rows from 0 -- over the depth tiles [0, q1) its products read */
static void att16_pack_rows(int bv, const bf16_call *c, size_t ib, const float *src, size_t lds,
    size_t rows, size_t q1) {
  bf16_call x=*c;
  x.x.a=src; x.x.lda=lds; x.x.m=rows; x.i0=0; x.wa=c->wa+2*ib*c->kt*BF16_TILE; x.q0=0; x.q1=q1;
  att16_pack_a(bv,&x,0);
}
/* depth tile ib of a call's B panel (B not transposed) from a block of `rows` depth rows at src
 * (lds), over its first np columns */
static void att16_pack_rows_b(int bv, const bf16_call *c, size_t ib, const float *src, size_t lds,
    size_t rows, size_t np) {
  bf16_call x=*c;
  x.x.b=src; x.x.ldb=lds; x.x.k=rows; x.wb=c->wb+ib*BF16_TILE; x.np=np<c->np ? np : c->np;
  att16_pack_b(bv,&x,0);
}
/* row block ib of a call as its own call: A's strip ib from row 0, C from out (ldo) */
static void att16_block(bf16_call *blk, const bf16_call *c, size_t ib, float *out, size_t ldo) {
  *blk=*c; blk->ap=c->ap+2*ib*c->kt*BF16_TILE; blk->i0=0; blk->x.c=out; blk->x.ldc=ldo;
}
/* rows [r0, r1) x columns [c0, c1) of a product over its depth tiles [q0, q1), from +0.0 (zero) or
 * from what C holds; the AMX instance runs between its task's bf16_amx_begin and bf16_amx_end */
static void att16_region(int bv, bf16_call *c, size_t r0, size_t r1, size_t c0, size_t c1,
    size_t q0, size_t q1, int zero) {
  c->q0=q0; c->q1=q1;
#ifdef BCIR_MM_DISPATCH
#ifdef BCIR_BF16_AMX
  if (bv==8) { bf16_amx_blocks(c,r0,r1,c0,c1,zero); return; }
#endif
  if (bv>=4) { bf16_avx512(c,r0,r1,c0,c1,zero); return; }
#endif
  (void)bv;
  bf16_generic(c,r0,r1,c0,c1,zero);
}
#ifdef BCIR_MM_DISPATCH
/* rows [i0, i0+n) of a head's scores (t x t): each row's columns past its diagonal +0.0 */
__attribute__((target("avx512f"))) static void att16_future512(size_t t, size_t i0, size_t n, float *ph) {
  size_t u,j;
  for (u=0;u<n;u++) {
    size_t i=i0+u,j0=(i+1)/16*16; float *row=ph+i*t;
    if (j0<t) _mm512_mask_storeu_ps(row+j0,(__mmask16)(att512_lanes(t-j0)&~att512_lanes(i+1-j0)),
        _mm512_setzero_ps());
    for (j=j0+16;j<t;j+=16) _mm512_mask_storeu_ps(row+j,att512_lanes(t-j),_mm512_setzero_ps());
  }
}
/* rows [0, n) of a dP block (row u the attention's row i0+u, at dpb + u*ld) become its score
 * adjoints in place -- p_ij (dp_ij - dot_u) scale, the scalar loop's three roundings, for j <= i --
 * and +0.0 past the diagonal to `edge` (a multiple of 16) */
__attribute__((target("avx512f"))) static void att16_ds512(size_t t, size_t i0, size_t n, const float *ph,
    float *dpb, size_t ld, const float *dot, float scale, size_t edge) {
  size_t u,j; const __m512 sc=_mm512_set1_ps(scale);
  for (u=0;u<n;u++) {
    size_t i=i0+u; const float *pi=ph+i*t; float *ds=dpb+u*ld; __m512 dv=_mm512_set1_ps(dot[u]);
    for (j=0;j<edge;j+=16) {
      __mmask16 take=j>i ? 0 : att512_lanes(i-j+1);
      __m512 pv=_mm512_maskz_loadu_ps(take,pi+j),dpv=_mm512_maskz_loadu_ps(take,ds+j);
      _mm512_storeu_ps(ds+j,_mm512_maskz_mul_ps(take,_mm512_mul_ps(pv,_mm512_sub_ps(dpv,dv)),sc));
    }
  }
}
/* n rows of `cols` floats from src (lds) to dst (ldd) */
__attribute__((target("avx512f"))) static void att16_copy512(size_t n, size_t cols, const float *src,
    size_t lds, float *dst, size_t ldd) {
  size_t u,j;
  for (u=0;u<n;u++) for (j=0;j<cols;j+=16) {
    __mmask16 m=att512_lanes(cols-j);
    _mm512_mask_storeu_ps(dst+u*ldd+j,m,_mm512_maskz_loadu_ps(m,src+u*lds+j));
  }
}
__attribute__((target("avx512f"))) static void att16_transpose512(size_t rw, size_t cw, const float *in,
    size_t ldi, float *out, size_t ldo) {
  size_t i,c;
  for (i=0;i<rw;i+=16) for (c=0;c<cw;c+=16) kb512_transpose16(in+c*ldi+i,ldi,out+i*ldo+c,ldo);
}
#endif
/* rows [i0, i0+n) of a head's scores: each row's columns past its diagonal +0.0 */
static void att16_future(int variant, size_t t, size_t i0, size_t n, float *ph) {
  size_t u;
  (void)variant;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { att16_future512(t,i0,n,ph); return; }
#endif
  for (u=0;u<n;u++) if (i0+u+1<t) memset(ph+(i0+u)*t+i0+u+1,0,(t-i0-u-1)*sizeof(float));
}
/* n rows of `cols` floats from src (lds) to dst (ldd) */
static void att16_copy(int variant, size_t n, size_t cols, const float *src, size_t lds, float *dst,
    size_t ldd) {
  size_t u;
  (void)variant;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { att16_copy512(n,cols,src,lds,dst,ldd); return; }
#endif
  for (u=0;u<n;u++) memcpy(dst+u*ldd,src+u*lds,cols*sizeof(float));
}
/* out (rows x cols, ldo) as the transpose of in (cols x rows, ldi): values moved, none computed */
static void att16_transpose(int variant, size_t rows, size_t cols, const float *in, size_t ldi,
    float *out, size_t ldo) {
  size_t i,c,rw=0,cw=0;
  (void)variant;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { rw=rows/16*16; cw=cols/16*16; att16_transpose512(rw,cw,in,ldi,out,ldo); }
#endif
  for (i=0;i<rows;i++) for (c=i<rw ? cw : 0;c<cols;c++) out[i*ldo+c]=in[c*ldi+i];
}
/* One query head of a forward task: kb and vb are its KV head's K^T and V panels. A query block's
 * scores land in p, its future +0.0, its softmax in place, its strip of P packed from those rows. */
static void att16_forward_head(const att16_job *j, size_t r, size_t h, uint16_t *kb, uint16_t *vb,
    uint16_t *qa, uint16_t *pa, float *ys) {
  const att_job *a=&j->a; size_t t=a->t,nh=a->nh,d=a->d,dp=bf16_pad(d),nb=(t+31)/32,ib;
  float *ph=a->p+(r*nh+h)*t*t,*yh=a->y+(r*t*nh+h)*d; bf16_call cs,cy,blk;
  att16_call(&cs,0,1,t,t,d,a->q+(r*t*nh+h)*d,nh*d,NULL,0,NULL,t,qa,kb);
  att16_pack(j->bv,&cs,1,0);
  att16_call(&cy,0,0,t,d,t,NULL,0,NULL,0,NULL,dp,pa,vb);
  for (ib=0;ib<nb;ib++) {
    size_t r0=ib*32,rows=t-r0<32 ? t-r0 : 32,cols=r0+rows;
    att16_block(&blk,&cs,ib,ph+r0*t,t);
    att16_region(j->bv,&blk,0,rows,0,cols,0,cs.kt,1);  /* the block's scores to its diagonal */
    att16_future(a->variant,t,r0,rows,ph);
    att_softmax_block(a,ph+r0*t,t,r0,rows);
    att16_pack_rows(j->bv,&cy,ib,ph+r0*t,t,rows,ib+1);  /* the block's strip of P, key tiles 0 .. ib */
    att16_block(&blk,&cy,ib,ys,dp);
    att16_region(j->bv,&blk,0,rows,0,dp,0,ib+1,1);  /* its context, through ys */
    att16_copy(a->variant,rows,d,ys,dp,yh+r0*nh*d,nh*d);
  }
  if (a->bad) att_verdict(a,r,h);
}
static void att16_task(void *ctx, size_t task, unsigned worker) {
  const att16_job *j=(const att16_job *)ctx; const att_job *a=&j->a;
  size_t t=a->t,d=a->d,nh=a->nh,nk=a->nk,share=nh/nk,r=task/nk,kh=task%nk,h;
  unsigned char *region=j->ws+(j->by_worker ? (size_t)worker : task)*j->per;
  INSTANCE_SAW(INSTANCE_ATTN16,j->bv);
  uint16_t *kb=(uint16_t *)(void *)region,*vb=kb+bf16_strip(t,d)/2,*qa=vb+bf16_strip(d,t)/2;
  uint16_t *pa=qa+bf16_strip(t,d)/2; float *ys=(float *)(void *)(pa+bf16_strip(t,t)/2);
  bf16_call c;
#ifdef BCIR_BF16_AMX
  if (j->bv==8) bf16_amx_begin();
#endif
  /* the KV head's keys (the scores' B, transposed) and values (the context's B), once */
  att16_call(&c,0,1,t,t,d,NULL,0,a->k+(r*t*nk+kh)*d,nk*d,NULL,t,NULL,kb);
  att16_pack(j->bv,&c,0,1);
  att16_call(&c,0,0,t,d,t,NULL,0,a->v+(r*t*nk+kh)*d,nk*d,NULL,d,NULL,vb);
  att16_pack(j->bv,&c,0,1);
  for (h=kh*share;h<(kh+1)*share;h++) att16_forward_head(j,r,h,kb,vb,qa,pa,ys);
#ifdef BCIR_BF16_AMX
  if (j->bv==8) bf16_amx_end();
#endif
}
/* The backward region's parts, in order. */
typedef struct att16_back {
  uint16_t *vt,*kb0; float *dvt,*dkt,*dpb,*dqs; uint16_t *dya,*dyta,*pb,*dsa,*dsb,*qta;
} att16_back;
static att16_back att16_parts(unsigned char *region, size_t t, size_t d) {
  att16_back w; size_t tp=bf16_pad(t),dp=bf16_pad(d); unsigned char *x=region;
  w.vt=(uint16_t *)(void *)x; x+=bf16_strip(t,d);
  w.kb0=(uint16_t *)(void *)x; x+=bf16_strip(d,t);
  w.dvt=(float *)(void *)x; x+=att16_floats(dp,tp);
  w.dkt=(float *)(void *)x; x+=att16_floats(dp,tp);
  w.dpb=(float *)(void *)x; x+=att16_floats(32,tp);
  w.dqs=(float *)(void *)x; x+=att16_floats(32,dp);
  w.dya=(uint16_t *)(void *)x; x+=bf16_strip(t,d);
  w.dyta=(uint16_t *)(void *)x; x+=bf16_strip(d,t);
  w.pb=(uint16_t *)(void *)x; x+=bf16_strip(t,t);
  w.dsa=(uint16_t *)(void *)x; x+=bf16_strip(t,t);
  w.dsb=(uint16_t *)(void *)x; x+=bf16_strip(t,t);
  w.qta=(uint16_t *)(void *)x;
  return w;
}
/* One query head of a backward task; `first` is its KV head's first query head. p is read, never
 * written: a query block's dP becomes its dS in dpb, whose strips the products read. */
static void att16_backward_head(const att16_job *j, size_t r, size_t h, int first,
    const att16_back *w) {
  const att_job *a=&j->a; size_t t=a->t,nh=a->nh,d=a->d,dp=bf16_pad(d),tp=bf16_pad(t),ldh=nh*d;
  size_t nb=(t+31)/32,ib,i,s; float scale=1.0f/sqrtf((float)d);
  const float *ph=a->p+(r*nh+h)*t*t,*qh=a->q+(r*t*nh+h)*d,*dyh=a->dy+(r*t*nh+h)*d;
  float *dqh=a->dq+(r*t*nh+h)*d; bf16_call cv,cp,cq,ck,blk;
  /* dV^T += dY^T P: a key block's query tiles from its own on */
  att16_call(&cv,1,0,d,t,t,dyh,ldh,ph,t,w->dvt,tp,w->dyta,w->pb);
  att16_pack(j->bv,&cv,1,0);
  for (s=0;s<nb;s++) att16_pack_depth(j->bv,&cv,s,32*(s+1));
  for (ib=0;ib<nb;ib++) att16_region(j->bv,&cv,0,dp,32*ib,32*ib+32,ib,nb,first);
  /* a query block at a time: its dP, dot and dS (in dpb), dS's strip (A, for dQ) and depth tile
   * (B, for dK^T), and its dQ */
  att16_call(&cp,0,1,t,t,d,dyh,ldh,NULL,0,NULL,tp,w->dya,w->vt);
  att16_pack(j->bv,&cp,1,0);
  att16_call(&cq,0,0,t,d,t,NULL,0,NULL,0,NULL,dp,w->dsa,w->kb0);
  att16_call(&ck,1,0,d,t,t,qh,ldh,NULL,0,w->dkt,tp,w->qta,w->dsb);
  for (ib=0;ib<nb;ib++) {
    size_t r0=ib*32,rows=t-r0<32 ? t-r0 : 32,cols=r0+rows,edge=r0+32; float dot[32];
    att16_block(&blk,&cp,ib,w->dpb,tp);
    att16_region(j->bv,&blk,0,rows,0,cols,0,cp.kt,1);
    for (i=0;i<rows;i++) dot[i]=0.0f;
    i=0;
#ifdef BCIR_MM_DISPATCH
    if (a->variant==4) for (;i+16<=rows;i+=16) att512_dot(t,r0,i,0,cols,ph,w->dpb,tp,dot);
#endif
    for (;i<rows;i++) {
      const float *pi=ph+(r0+i)*t,*dpi=w->dpb+i*tp; size_t jj;
      for (jj=0;jj<=r0+i;jj++) dot[i]=dot[i]+pi[jj]*dpi[jj];
    }
#ifdef BCIR_MM_DISPATCH
    if (a->variant==4) att16_ds512(t,r0,rows,ph,w->dpb,tp,dot,scale,edge); else
#endif
    for (i=0;i<rows;i++) {  /* dS over dP, the future +0.0 to the tile's edge */
      const float *pi=ph+(r0+i)*t; float *dsi=w->dpb+i*tp; size_t jj;
      for (jj=0;jj<=r0+i;jj++) dsi[jj]=pi[jj]*(dsi[jj]-dot[i])*scale;
      memset(dsi+r0+i+1,0,(edge-r0-i-1)*sizeof(float));
    }
    att16_pack_rows(j->bv,&cq,ib,w->dpb,tp,rows,ib+1);
    att16_pack_rows_b(j->bv,&ck,ib,w->dpb,tp,rows,edge);
    att16_block(&blk,&cq,ib,w->dqs,dp);
    att16_region(j->bv,&blk,0,rows,0,dp,0,ib+1,1);
    att16_copy(a->variant,rows,d,w->dqs,dp,dqh+r0*ldh,ldh);
  }
  /* dK^T += Q^T dS: a key block's query tiles from its own on */
  att16_pack(j->bv,&ck,1,0);
  for (ib=0;ib<nb;ib++) att16_region(j->bv,&ck,0,dp,32*ib,32*ib+32,ib,nb,first);
}
/* A backward task is a (batch, KV head), its query heads in order: they share dk and dv. */
static void att16_task_backward(void *ctx, size_t task, unsigned worker) {
  const att16_job *j=(const att16_job *)ctx; const att_job *a=&j->a;
  size_t t=a->t,d=a->d,nh=a->nh,nk=a->nk,share=nh/nk,r=task/nk,kh=task%nk,tp=bf16_pad(t),h;
  att16_back w=att16_parts(j->ws+(j->by_worker ? (size_t)worker : task)*j->per,t,d);
  INSTANCE_SAW(INSTANCE_ATTN16,j->bv);
  bf16_call c;
#ifdef BCIR_BF16_AMX
  if (j->bv==8) bf16_amx_begin();
#endif
  /* the KV head's values transposed (dP's B) and keys (dQ's B), once for its query heads */
  att16_call(&c,0,1,t,t,d,NULL,0,a->v+(r*t*nk+kh)*d,nk*d,NULL,t,NULL,w.vt);
  att16_pack(j->bv,&c,0,1);
  att16_call(&c,0,0,t,d,t,NULL,0,a->k+(r*t*nk+kh)*d,nk*d,NULL,d,NULL,w.kb0);
  att16_pack(j->bv,&c,0,1);
  for (h=kh*share;h<(kh+1)*share;h++) att16_backward_head(j,r,h,h==kh*share,&w);
#ifdef BCIR_BF16_AMX
  if (j->bv==8) bf16_amx_end();
#endif
  att16_transpose(a->variant,t,d,w.dvt,tp,a->dv+(r*t*nk+kh)*d,nk*d);
  att16_transpose(a->variant,t,d,w.dkt,tp,a->dk+(r*t*nk+kh)*d,nk*d);
}
/* the instance a call names (0: the widest admitted), or 0 where it is not admitted */
static int att16_instance(int variant) { return bf16_instance(variant); }
static void att16_fill(att_job *a, size_t b, size_t t, size_t nh, size_t nk, size_t d) {
  memset(a,0,sizeof *a);
  a->b=b; a->t=t; a->nh=nh; a->nk=nk; a->d=d; a->variant=mm_widest(); a->group=nh/nk;
}
int bcir_tensor_attention_bf16_scan_pool(bcir_tensor_pool *pool, int variant, unsigned contract,
    size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v,
    float *p, float *y, int *bad, void *workspace, size_t bytes) {
  att16_job j; unsigned char verdicts[BCIR_TENSOR_SCAN_TASKS]; size_t tasks=b*nk,heads=b*nh,usable=bytes;
  int bv=att16_instance(variant);
  if (!bv || !nk || nh%nk) return 0;
  if (!b || !t || !nh || !d) return bv;
  if (!workspace || bytes<bcir_tensor_attention_bf16_pool_workspace(b,t,nh,nk,d,
      bcir_tensor_pool_threads(pool))) return 0;
  att16_fill(&j.a,b,t,nh,nk,d);
  j.a.q=q; j.a.k=k; j.a.v=v; j.a.p=p; j.a.y=y;
  j.a.exp2=(contract&BCIR_TENSOR_ATTENTION_EXP)!=0;
  j.bv=bv; j.ws=(unsigned char *)(void *)bf16_base(workspace,&usable);
  j.per=att16_forward_bytes(t,d); j.by_worker=bcir_tensor_pool_threads(pool)<=tasks;
#ifdef BCIR_DT_ATTENTION_PROBE
  attention_saw(0,(j.a.exp2 ? BCIR_TENSOR_ATTENTION_EXP : 0u)|BCIR_TENSOR_ATTENTION_BF16);
#endif
  j.a.bad=scan_bytes(bad,verdicts,heads);
  pool_dispatch(pool,NULL,att16_task,&j,tasks);
  if (bad) {
    if (j.a.bad) scan_fold(bad,verdicts,heads);
    else if (!scan_pool(pool,y,b*t*nh*d) || !scan_pool(pool,k,b*t*nk*d)) *bad=1;
  }
  return bv;
}
int bcir_tensor_attention_bf16_backward_pool(bcir_tensor_pool *pool, int variant, size_t b,
    size_t t, size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v,
    const float *p, const float *dy, float *dq, float *dk, float *dv, void *workspace, size_t bytes) {
  att16_job j; size_t tasks=b*nk,usable=bytes; int bv=att16_instance(variant);
  if (!bv || !nk || nh%nk) return 0;
  if (!b || !t || !nh || !d) return bv;
  if (!workspace || bytes<bcir_tensor_attention_bf16_pool_workspace(b,t,nh,nk,d,
      bcir_tensor_pool_threads(pool))) return 0;
  att16_fill(&j.a,b,t,nh,nk,d);
  j.a.q=q; j.a.k=k; j.a.v=v; j.a.p=(float *)(uintptr_t)p; j.a.dy=dy; j.a.dq=dq; j.a.dk=dk; j.a.dv=dv;
  j.bv=bv; j.ws=(unsigned char *)(void *)bf16_base(workspace,&usable);
  j.per=att16_backward_bytes(t,d); j.by_worker=bcir_tensor_pool_threads(pool)<=tasks;
#ifdef BCIR_DT_ATTENTION_PROBE
  attention_saw(1,BCIR_TENSOR_ATTENTION_BF16);
#endif
  pool_dispatch(pool,NULL,att16_task_backward,&j,tasks);
  return bv;
}

/* --- PX-5 (LAYOUT): the packers' layouts, read back ---------------------------------------------
 * Each packer whose layout bcir/hosted/models/native_layouts.py states, run alone on an operand the
 * caller lends, so a witness reads back what it writes and holds every element to the statement; a
 * packer this build or host has not answers 0, nothing written. A build defines
 * BCIR_DT_LAYOUT_PROBE to have them (the witness's builds do; the library does not). */
#ifdef BCIR_DT_LAYOUT_PROBE
BCIR_DT_API int bcir_tensor_probe_bf16_pack(int variant, int which, int trans, size_t rows,
    size_t cols, float alpha, const float *x, size_t ld, uint16_t *out);
BCIR_DT_API int bcir_tensor_probe_kb_pack(int tb, size_t k, size_t n, const float *b, size_t ldb,
    size_t jb, size_t nj, size_t lb, size_t nl, float *bt);
BCIR_DT_API int bcir_tensor_probe_transpose16(const float *src, size_t ld, float *dst, size_t ldd);
BCIR_DT_API int bcir_tensor_probe_panels(int conv, int swiglu, int v2, size_t r, size_t c,
    const float *x, const float *u, const float *dy, unsigned short *a0, unsigned short *b0,
    unsigned short *a1, unsigned short *b1);
/* The bf16 contract's pack of a whole operand by the instance `variant` names (1 the portable C,
 * 4 AVX-512, 8 AMX: bcir_tensor_mm_bf16_variants' bits): which 0 packs an A of `rows` x `cols`
 * (the depth; trans: stored cols x rows, ld apart) times alpha, which 1 a B of `rows` (the depth) x
 * `cols` (trans: stored cols x rows); `out` holds pad32(rows) x pad32(cols) elements. */
int bcir_tensor_probe_bf16_pack(int variant, int which, int trans, size_t rows, size_t cols,
    float alpha, const float *x, size_t ld, uint16_t *out) {
  bf16_call c; size_t s,tasks;
  if (!x || !out || !rows || !cols || (variant!=1 && variant!=4 && variant!=8) ||
      !(bcir_tensor_mm_bf16_variants()&variant)) return 0;
  memset(&c,0,sizeof c);
  if (!which) {
    c.x.ta=trans!=0; c.x.m=rows; c.x.k=cols; c.x.alpha=alpha; c.x.a=x; c.x.lda=ld;
    c.kt=bf16_pad(cols)/32; c.mp=bf16_pad(rows); c.q0=0; c.q1=c.kt; c.wa=out;
    tasks=c.x.ta ? c.kt : c.mp/32;
    for (s=0;s<tasks;s++) att16_pack_a(variant,&c,s);
  } else {
    c.x.tb=trans!=0; c.x.k=rows; c.x.n=cols; c.x.alpha=1.0f; c.x.b=x; c.x.ldb=ld;
    c.kt=bf16_pad(rows)/32; c.np=bf16_pad(cols); c.q0=0; c.q1=c.kt; c.wb=out;
    tasks=c.x.tb ? c.np/32 : c.kt;
    for (s=0;s<tasks;s++) att16_pack_b(variant,&c,s);
  }
  return 1;
}
#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx512f"))) static void probe_kb_pack(const mm_args *x, float *bt,
    size_t jb, size_t nj, size_t lb, size_t nl) {
  kb512_pack(x,bt,jb,nj,lb,nl);
}
__attribute__((target("avx512f"))) static void probe_transpose16(const float *src, size_t ld,
    float *dst, size_t ldd) {
  kb512_transpose16(src,ld,dst,ldd);
}
#endif
/* The AVX-512 blocked GEMM's packed B tile: columns [jb, jb+nj) (nj at most 48) and depths
 * [lb, lb+nl) (nl at most BCIR_MM_KBLOCK) of a k x n B (tb: stored n x k), into bt -- nl rows of 48
 * floats, 64-byte aligned (the tile's stores are aligned). */
int bcir_tensor_probe_kb_pack(int tb, size_t k, size_t n, const float *b, size_t ldb, size_t jb,
    size_t nj, size_t lb, size_t nl, float *bt) {
#ifdef BCIR_MM_DISPATCH
  mm_args x;
  if (!(bcir_tensor_mm_variants()&4) || !b || !bt || (uintptr_t)bt%64 || !nj || nj>48 || !nl ||
      nl>BCIR_MM_KBLOCK || jb>n || nj>n-jb || lb>k || nl>k-lb) return 0;
  memset(&x,0,sizeof x);
  x.tb=tb!=0; x.k=k; x.n=n; x.b=b; x.ldb=ldb;
  probe_kb_pack(&x,bt,jb,nj,lb,nl);
  return 1;
#else
  (void)tb; (void)k; (void)n; (void)b; (void)ldb; (void)jb; (void)nj; (void)lb; (void)nl; (void)bt;
  return 0;
#endif
}
/* The 16 x 16 register transpose: 16 rows of 16 floats, ld apart, to their transpose, ldd apart. */
int bcir_tensor_probe_transpose16(const float *src, size_t ld, float *dst, size_t ldd) {
#ifdef BCIR_MM_DISPATCH
  if (!(bcir_tensor_mm_variants()&4) || !src || !dst || ld<16 || ldd<16) return 0;
  probe_transpose16(src,ld,dst,ldd);
  return 1;
#else
  (void)src; (void)ld; (void)dst; (void)ldd;
  return 0;
#endif
}
/* The panel writers (BF16-PANELS) by the conversion instance `conv` names (1 the portable C, 4
 * AVX-512, 8 AMX), serially, where the library's entries run the host's widest: swiglu 0 the SiLU
 * product's panels as A (a0) and as B (b0), swiglu 1 the SwiGLU VJP's two adjoints as A (a0, a1)
 * and as A^T (b0, b1); r and c multiples of 32, as the entries take them. */
int bcir_tensor_probe_panels(int conv, int swiglu, int v2, size_t r, size_t c, const float *x,
    const float *u, const float *dy, unsigned short *a0, unsigned short *b0, unsigned short *a1,
    unsigned short *b1) {
  panel_job j; size_t t;
  if ((conv!=1 && conv!=4 && conv!=8) || !(bcir_tensor_mm_bf16_variants()&conv) || !r || !c ||
      r%32 || c%32 || !x || !u || !a0 || !b0 || (swiglu && (!dy || !a1 || !b1))) return 0;
  memset(&j,0,sizeof j); j.r=r; j.c=c; j.x=x; j.u=u; j.dy=dy; j.at=swiglu!=0;
  j.op=swiglu ? (v2 ? PANEL_SWIGLU2 : PANEL_SWIGLU1) : (v2 ? PANEL_SILU2 : PANEL_SILU1);
  j.pa[0]=a0; j.pb[0]=b0; j.pa[1]=a1; j.pb[1]=b1;
  j.per=1; j.variant=mm_widest(); j.conv=conv;
  for (t=0;t<r/32;t++) panel_task(&j,t);
  return 1;
}
#endif
