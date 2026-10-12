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
} mm_args;
/* The contract an instance keeps, a compile-time constant of the instance: v1 adds one rounded
 * product per term, v2 (NDT-FMA) one fmaf. */
enum { MM_V1=0, MM_FUSED=1 };
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
/* The column width of a variant's register block: a task's column range is whole blocks. */
typedef struct mm_range { size_t jb0,jb1,r0,r1; } mm_range;
static void mm_generic(const mm_args *x, const mm_range *r) { mm_kernel(2,32,MM_V1,x,r->jb0,r->jb1,r->r0,r->r1); }
static void mm_generic_fused(const mm_args *x, const mm_range *r) { mm_kernel(2,32,MM_FUSED,x,r->jb0,r->jb1,r->r0,r->r1); }
/* x86-64 Linux with GCC or Clang also carries AVX2 and AVX-512 variants of the same kernel,
 * chosen per call from what the CPU and OS support; anything else runs mm_generic.
 * BCIR_TENSOR_MAX_VARIANT caps the choice (1, 2 or 4) so a build can run a narrower one. */
#ifndef BCIR_TENSOR_MAX_VARIANT
#define BCIR_TENSOR_MAX_VARIANT 4
#endif
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(BCIR_TENSOR_NO_DISPATCH) && BCIR_TENSOR_MAX_VARIANT>1
#define BCIR_MM_DISPATCH 1
__attribute__((target("avx2"))) static void mm_avx2(const mm_args *x, const mm_range *r) {
  mm_kernel(2,32,MM_V1,x,r->jb0,r->jb1,r->r0,r->r1);
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
#endif
static size_t mm_nr(int variant) { return variant==4 ? 64 : 32; }
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
  if (variant==4) { if (fused) mm_avx512_fused(x,r); else mm_avx512(x,r); return; }
  if (variant==2) { if (fused) mm_avx2_fused(x,r); else mm_avx2(x,r); return; }
#endif
  (void)variant;
  if (fused) mm_generic_fused(x,r); else mm_generic(x,r);
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
 * Branch-free: the argument is clamped into the range the reduction handles, every lane runs the
 * same operations, and the three special cases are selected at the end. LN2_HI has 16
 * significant bits, so k*LN2_HI is exact for |k| < 256; 2^k is applied as two exact power-of-two
 * scalings (|k| <= 151, so each half is in range): a normal result is exact under both, a
 * subnormal one is rounded once, by the second. The instances below inline this body under
 * their target, and the build's -ffp-contract=off keeps every multiply and add its own. */
#define BCIR_EXP_LOG2E 1.44269504088896341f
#define BCIR_EXP_LN2_HI 0.693145751953125f
#define BCIR_EXP_LN2_LO 1.428606765330187e-06f
BCIR_MM_INLINE float exp_bits(uint32_t b) { float f; memcpy(&f,&b,4); return f; }
BCIR_MM_INLINE float exp1(float x) {
  float xc=x<-104.0f ? -104.0f : x>89.0f ? 89.0f : x,kf,r,p,s1,s2,y; int32_t k,k1,k2;
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
static void silu2_generic(size_t n, const float *x, float *y) { silu2_body(n,x,y); }
static void swiglu2_generic(size_t n, const float *g, const float *u, const float *dy, float *dg,
    float *du) { swiglu2_body(n,g,u,dy,dg,du); }
#ifdef BCIR_MM_DISPATCH
__attribute__((target("avx2"))) static void exp_avx2(size_t n, const float *x, float *y) { exp_body(n,x,y); }
__attribute__((target("avx512f"))) static void exp_avx512(size_t n, const float *x, float *y) { exp_body(n,x,y); }
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
  mm_args x; mm_range r; size_t i;
  int have=fused ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
  if ((variant!=1 && variant!=2 && variant!=4) || !(have&variant)) return 0;
  for (i=0;i<m*n;i++) c[i]=beta==0.0f ? 0.0f : beta*c[i];
  if (!m || !n || !k) return variant;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.lda=ta ? m : k;
  x.b=b; x.ldb=tb ? k : n; x.c=c; x.ldc=n; x.mode=MM_FULL;
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
/* --- the deterministic pool (NDT-THREADS) ------------------------------------------------------
 * Thread 0 is the caller; the others wait for a generation, run their share of its tasks and
 * report. Task t goes to thread t % threads: the assignment is fixed, no task is taken by two
 * threads, and no thread takes a task the serial run would not have run in the same position.
 * Every task writes outputs no other task writes and reads nothing a task of the same job
 * writes, so the run order does not reach the arithmetic; the partitions below make it so.
 * The pool, its thread handles and its worker records live in one block the caller provides
 * (bcir_tensor_pool_size): the unit allocates nothing, as the trainer's arenas allocate nothing. */
typedef struct pool_worker { bcir_tensor_pool *pool; unsigned id; } pool_worker;
struct bcir_tensor_pool {
  unsigned threads,pending; unsigned long generation; int stop;
  bcir_tensor_task_fn fn; void *ctx; size_t tasks;
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
  for (t=id;t<p->tasks;t+=p->threads) p->fn(p->ctx,t);
}
#if defined(BCIR_POOL_PTHREADS) || defined(BCIR_POOL_WIN32)
#if defined(BCIR_POOL_PTHREADS)
#define POOL_LOCK(p) pthread_mutex_lock(&(p)->mutex)
#define POOL_UNLOCK(p) pthread_mutex_unlock(&(p)->mutex)
#define POOL_WAIT(p,c) pthread_cond_wait(&(p)->c,&(p)->mutex)
#define POOL_SIGNAL(p,c) pthread_cond_signal(&(p)->c)
#define POOL_BROADCAST(p,c) pthread_cond_broadcast(&(p)->c)
static void *pool_main(void *arg)
#else
#define POOL_LOCK(p) EnterCriticalSection(&(p)->mutex)
#define POOL_UNLOCK(p) LeaveCriticalSection(&(p)->mutex)
#define POOL_WAIT(p,c) SleepConditionVariableCS(&(p)->c,&(p)->mutex,INFINITE)
#define POOL_SIGNAL(p,c) WakeConditionVariable(&(p)->c)
#define POOL_BROADCAST(p,c) WakeAllConditionVariable(&(p)->c)
static DWORD WINAPI pool_main(LPVOID arg)
#endif
{
  pool_worker *w=(pool_worker *)arg; bcir_tensor_pool *p=w->pool; unsigned id=w->id;
  unsigned long seen=0;
  for (;;) {
    POOL_LOCK(p);
    while (p->generation==seen && !p->stop) POOL_WAIT(p,wake);
    if (p->stop) { POOL_UNLOCK(p); break; }
    seen=p->generation;
    POOL_UNLOCK(p);
    pool_tasks(p,id);
    POOL_LOCK(p);
    if (--p->pending==0) POOL_SIGNAL(p,done);
    POOL_UNLOCK(p);
  }
  return 0;
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
    POOL_LOCK(p); p->stop=1; POOL_BROADCAST(p,wake); POOL_UNLOCK(p);
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
  POOL_LOCK(p); p->stop=1; POOL_BROADCAST(p,wake); POOL_UNLOCK(p);
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
  p->fn=fn; p->ctx=ctx; p->tasks=tasks; p->pending=p->threads-1; p->generation++;
  POOL_BROADCAST(p,wake);
  POOL_UNLOCK(p);
  pool_tasks(p,0);
  POOL_LOCK(p);
  while (p->pending) POOL_WAIT(p,done);
  POOL_UNLOCK(p);
#else
  for (t=0;t<tasks;t++) fn(ctx,t);
#endif
}
#define pool_run bcir_tensor_pool_run
/* --- the partitions: each a function of the shapes alone ----------------------------------- */
/* GEMM: one task per (column block of the variant's width, row range of BCIR_MM_ROWS rows). The
 * task scales its own elements of C by beta, then runs the serial nest over its block and rows
 * -- every depth chunk, in order -- so each element of C receives its terms in the serial order;
 * the task packs its own tiles. */
#define BCIR_MM_ROWS 64
typedef struct mm_job { mm_args x; float beta; int variant,fused; size_t nr,cols,rowsets; } mm_job;
static void mm_task(void *ctx, size_t task) {
  mm_job *j=(mm_job *)ctx; mm_range r; size_t col=task%j->cols,row=task/j->cols,i,jj;
  r.jb0=col*j->nr; r.jb1=r.jb0+j->nr<j->x.n ? r.jb0+j->nr : j->x.n;
  r.r0=row*BCIR_MM_ROWS; r.r1=r.r0+BCIR_MM_ROWS<j->x.m ? r.r0+BCIR_MM_ROWS : j->x.m;
  for (i=r.r0;i<r.r1;i++) {
    float *ci=j->x.c+i*j->x.ldc;
    for (jj=r.jb0;jj<r.jb1;jj++) ci[jj]=j->beta==0.0f ? 0.0f : j->beta*ci[jj];
  }
  if (j->x.k) mm_run_range(j->variant,j->fused,&j->x,&r);
}
static void mm_pool_any(bcir_tensor_pool *pool, int fused, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_job j;
  if (!pool || pool->threads==1) {
    (fused ? bcir_tensor_mm_fused : bcir_tensor_mm)(NULL,ta,tb,m,n,k,alpha,a,b,beta,c); return;
  }
  if (!m || !n) return;
  j.beta=beta; j.fused=fused; j.variant=fused ? mm_widest_fused() : mm_widest(); j.nr=mm_nr(j.variant);
  j.x.ta=ta; j.x.tb=tb; j.x.m=m; j.x.n=n; j.x.k=k; j.x.alpha=alpha; j.x.a=a; j.x.lda=ta ? m : k;
  j.x.b=b; j.x.ldb=tb ? k : n; j.x.c=c; j.x.ldc=n; j.x.mode=MM_FULL;
  j.cols=(n+j.nr-1)/j.nr; j.rowsets=(m+BCIR_MM_ROWS-1)/BCIR_MM_ROWS;
  pool_run(pool,mm_task,&j,j.cols*j.rowsets);
}
void bcir_tensor_mm_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_V1,ta,tb,m,n,k,alpha,a,b,beta,c);
}
void bcir_tensor_mm_fused_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c) {
  mm_pool_any(pool,MM_FUSED,ta,tb,m,n,k,alpha,a,b,beta,c);
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
typedef struct rms_job { size_t rows,width; float eps; const float *x,*g; float *y; } rms_job;
static void rms_task(void *ctx, size_t task) {
  rms_job *j=(rms_job *)ctx; size_t r0=task*BCIR_ROW_CHUNK,r1=r0+BCIR_ROW_CHUNK<j->rows ? r0+BCIR_ROW_CHUNK : j->rows;
  bcir_tensor_rms(r1-r0,j->width,j->eps,j->x+r0*j->width,j->g,j->y+r0*j->width);
}
void bcir_tensor_rms_pool(bcir_tensor_pool *pool, size_t rows, size_t width, float eps,
    const float *x, const float *g, float *y) {
  rms_job j; j.rows=rows; j.width=width; j.eps=eps; j.x=x; j.g=g; j.y=y;
  pool_run(pool,rms_task,&j,(rows+BCIR_ROW_CHUNK-1)/BCIR_ROW_CHUNK);
}
/* Positions in ranges of BCIR_ROW_CHUNK: a position's angles and rotations are its own. */
typedef struct rope_job { size_t b,t,heads,dim; double base; float *x; int inverse; } rope_job;
static void rope_rows(size_t b, size_t t, size_t heads, size_t dim, double base, float *x,
    int inverse, size_t p0, size_t p1) {
  size_t r,pos,h,j;
  for (pos=p0;pos<p1;pos++) for (j=0;j<dim/2;j++) {
    double angle=(double)pos*pow(base,-2.0*(double)j/(double)dim);
    float c=(float)cos(angle),s=(float)sin(angle)*(inverse ? -1.0f : 1.0f);
    for (r=0;r<b;r++) for (h=0;h<heads;h++) {
      size_t off=((r*t+pos)*heads+h)*dim+j;
      float a=x[off],z=x[off+dim/2]; x[off]=a*c-z*s; x[off+dim/2]=z*c+a*s;
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
 * bit-identical; future scores stay exactly zero. */
typedef struct att_job {
  size_t b,t,nh,nk,d,group; const float *q,*k,*v,*dy; float *p,*y,*dq,*dk,*dv; int variant,exp2;
} att_job;
/* One (batch, query head): its scores, softmax rows and context; nothing shared with another. */
static void att_head(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,kh=h/(nh/nk),i,j; float scale=1.0f/sqrtf((float)d);
  float *ph=a->p+(r*nh+h)*t*t; mm_args x;
  x.ta=0; x.tb=1; x.m=t; x.n=t; x.k=d; x.alpha=1.0f; x.a=a->q+(r*t*nh+h)*d; x.lda=nh*d;
  x.b=a->k+(r*t*nk+kh)*d; x.ldb=nk*d; x.c=ph; x.ldc=t; x.mode=MM_LOWER;
  mm_run(a->variant,&x);
  for (i=0;i<t;i++) {
    float *pi=ph+i*t,maxv=-INFINITY,sum=0.0f;
    for (j=0;j<=i;j++) { pi[j]*=scale; if (pi[j]>maxv) maxv=pi[j]; }
    if (a->exp2) { for (j=0;j<=i;j++) { pi[j]=exp1(pi[j]-maxv); sum+=pi[j]; } }
    else for (j=0;j<=i;j++) { pi[j]=expf(pi[j]-maxv); sum+=pi[j]; }
    for (j=0;j<=i;j++) pi[j]/=sum;
  }
  x.tb=0; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=a->v+(r*t*nk+kh)*d; x.ldb=nk*d;
  x.c=a->y+(r*t*nh+h)*d; x.ldc=nh*d; x.mode=MM_KLE;
  mm_run(a->variant,&x);
}
static void att_task(void *ctx, size_t task) {
  att_job *a=(att_job *)ctx; att_head(a,task/a->nh,task%a->nh);
}
static void attention_any(bcir_tensor_pool *pool, int exp2, size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, float *y) {
  att_job a;
  memset(y,0,b*t*nh*d*sizeof(float)); memset(p,0,b*nh*t*t*sizeof(float));
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
 * shared GQA dk/dv, i ascending within a head -- so dq, dk and dv are bit-identical to it. */
#define BCIR_ATT_ROWS 32
#define BCIR_ATT_COLS 64
/* One (batch, query head) of the backward; the heads of one KV head share dk and dv, so a task
 * is a (batch, KV head) and runs its `group` = heads/kvheads query heads in order -- the order
 * the serial loop adds. A task of fewer heads would put one KV head's query heads on two threads,
 * racing on its dk and dv (the TSan section's injected fault). */
static void att_head_backward(const att_job *a, size_t r, size_t h) {
  size_t t=a->t,nh=a->nh,nk=a->nk,d=a->d,i,j,i0; float scale=1.0f/sqrtf((float)d);
  int variant=a->variant; const float *q=a->q,*k=a->k,*v=a->v,*dy=a->dy; float *p=a->p,*dq=a->dq,*dk=a->dk,*dv=a->dv;
  {
    size_t kh=h/(nh/nk),ldh=nh*d,ldk=nk*d; float *ph=p+(r*nh+h)*t*t;
    const float *qh=q+(r*t*nh+h)*d,*dyh=dy+(r*t*nh+h)*d;
    const float *kk=k+(r*t*nk+kh)*d,*vk=v+(r*t*nk+kh)*d;
    mm_args x;
    x.ta=1; x.tb=0; x.m=t; x.n=d; x.k=t; x.alpha=1.0f; x.a=ph; x.lda=t; x.b=dyh; x.ldb=ldh;
    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    mm_run(variant,&x);
    for (i0=0;i0<t;i0+=BCIR_ATT_ROWS) {
      size_t rows=t-i0<BCIR_ATT_ROWS ? t-i0 : BCIR_ATT_ROWS,pass,jt;
      float dot[BCIR_ATT_ROWS],tile[BCIR_ATT_ROWS*BCIR_ATT_COLS];
      for (i=0;i<rows;i++) dot[i]=0.0f;
      for (pass=0;pass<2;pass++) for (jt=0;jt<i0+rows;jt+=BCIR_ATT_COLS) {
        size_t cols=i0+rows-jt<BCIR_ATT_COLS ? i0+rows-jt : BCIR_ATT_COLS;
        memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));
        x.ta=0; x.tb=1; x.m=rows; x.n=cols; x.k=d; x.a=dyh+i0*ldh; x.lda=ldh; x.b=vk+jt*ldk;
        x.ldb=ldk; x.c=tile; x.ldc=BCIR_ATT_COLS; x.mode=MM_FULL;
        mm_run(variant,&x);
        for (i=0;i<rows;i++) {
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
    mm_run(variant,&x);
    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    mm_run(variant,&x);
  }
}
static void att_task_backward(void *ctx, size_t task) {
  att_job *a=(att_job *)ctx; size_t per=a->nh/a->group,r=task/per,first=(task%per)*a->group,h;
  for (h=first;h<first+a->group;h++) att_head_backward(a,r,h);
}
void bcir_tensor_attention_backward_inplace_pool(bcir_tensor_pool *pool, size_t b, size_t t,
    size_t nh, size_t nk, size_t d, const float *q, const float *k, const float *v, float *p,
    const float *dy, float *dq, float *dk, float *dv) {
  att_job a;
  memset(dq,0,b*t*nh*d*sizeof(float)); memset(dk,0,b*t*nk*d*sizeof(float));
  memset(dv,0,b*t*nk*d*sizeof(float));
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
