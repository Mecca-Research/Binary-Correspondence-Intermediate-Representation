/* NDT-GEMM-L1's probe (2026-10-16): the AVX-512 kblock kernel's one-thread rate against variants that
 * keep its arithmetic (every element's K block from zero, one fma per term in ascending depth, then
 * c + alpha*P) over the pool's task partition (128 rows x 48 columns, tasks in order), on the step's
 * GEMM shapes, every variant's C compared with the unit kernel's bit for bit. The unit is included
 * whole, so its static kernels are reachable; the probe scales c itself before each task, so it is
 * built against the unit as it stood before NDT-BETA folded that scaling into the kernel:
 *
 *   git show ad77c84:runtime/c/bcir_tensor.c > U/bcir_tensor.c   (and bcir_tensor.h beside it)
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I U -DBCIR_TENSOR_C='"U/bcir_tensor.c"' gemm_probe.c -lm
 *   ./a.out ceiling     -- the inner loop alone over one tile, depth 64 (L1) .. 448 (L2): probe-ceiling.jsonl
 *   ./a.out REPS        -- one thread, every variant, interleaved by round: probe-t1.jsonl
 *   ./a.out pool REPS   -- the same on a pool of four: probe-t4.jsonl
 *   ./a.out split       -- the pack's and the beta pass's share of a task (rdtsc): probe-split.jsonl
 */
#define _GNU_SOURCE 1
#ifndef BCIR_TENSOR_C
#define BCIR_TENSOR_C "bcir_tensor.c"
#endif
#include BCIR_TENSOR_C
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

static double now_ns(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec;
}
typedef void (*kern_fn)(const mm_args *x, const mm_range *r);

/* V0: the tree's kernel. */
static void k_tree(const mm_args *x, const mm_range *r) { mm_avx512_kb(x,r); }

/* A depth slice [l0, l1) of one K block for rows [i, i+mr): the accumulators from zero (first) or
 * from acc, one fma per term in ascending depth, back to acc. Spilling a register to memory and
 * loading it back is exact, so the K block's P is kb512_block's bit for bit. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_slice(size_t mr, const mm_args *x,
    const float *bt, size_t i, size_t lb, size_t l0, size_t l1, float *acc, int first) {
  __m512 a0[8],a1[8],a2[8]; size_t r,l;
  for (r=0;r<mr;r++) {
    if (first) { a0[r]=_mm512_setzero_ps(); a1[r]=_mm512_setzero_ps(); a2[r]=_mm512_setzero_ps(); }
    else { a0[r]=_mm512_load_ps(acc+r*48); a1[r]=_mm512_load_ps(acc+r*48+16); a2[r]=_mm512_load_ps(acc+r*48+32); }
  }
  if (x->ta) {
    const float *ap=x->a+(lb+l0)*x->lda+i; size_t lda=x->lda;
    for (l=l0;l<l1;l++,ap+=lda) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      _mm_prefetch((const char *)(ap+BCIR_MM_KB_PREFETCH*lda),_MM_HINT_T0);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r]);
        a0[r]=_mm512_fmadd_ps(av,b0,a0[r]); a1[r]=_mm512_fmadd_ps(av,b1,a1[r]); a2[r]=_mm512_fmadd_ps(av,b2,a2[r]);
      }
    }
  } else {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
    for (l=l0;l<l1;l++) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r*lda+l]);
        a0[r]=_mm512_fmadd_ps(av,b0,a0[r]); a1[r]=_mm512_fmadd_ps(av,b1,a1[r]); a2[r]=_mm512_fmadd_ps(av,b2,a2[r]);
      }
    }
  }
  for (r=0;r<mr;r++) {
    _mm512_store_ps(acc+r*48,a0[r]); _mm512_store_ps(acc+r*48+16,a1[r]); _mm512_store_ps(acc+r*48+32,a2[r]);
  }
}
/* c = c + alpha*P for rows [i, i+mr) from acc: kb512_block's epilogue exactly. */
__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_commit(size_t mr, const mm_args *x,
    size_t i, size_t jb, size_t nj, const float *acc) {
  __m512 alpha=_mm512_set1_ps(x->alpha); size_t r;
  __mmask16 m0=(__mmask16)(nj>=16 ? 0xffffu : (1u<<nj)-1u);
  __mmask16 m1=(__mmask16)(nj>=32 ? 0xffffu : nj>16 ? (1u<<(nj-16))-1u : 0u);
  __mmask16 m2=(__mmask16)(nj>32 ? (1u<<(nj-32))-1u : 0u);
  for (r=0;r<mr;r++) {
    float *ci=x->c+(i+r)*x->ldc+jb;
    __m512 p0=_mm512_load_ps(acc+r*48),p1=_mm512_load_ps(acc+r*48+16),p2=_mm512_load_ps(acc+r*48+32);
    if (nj==48) {
      _mm512_storeu_ps(ci,_mm512_add_ps(_mm512_loadu_ps(ci),_mm512_mul_ps(alpha,p0)));
      _mm512_storeu_ps(ci+16,_mm512_add_ps(_mm512_loadu_ps(ci+16),_mm512_mul_ps(alpha,p1)));
      _mm512_storeu_ps(ci+32,_mm512_add_ps(_mm512_loadu_ps(ci+32),_mm512_mul_ps(alpha,p2)));
    } else {
      _mm512_mask_storeu_ps(ci,m0,_mm512_add_ps(_mm512_maskz_loadu_ps(m0,ci),_mm512_mul_ps(alpha,p0)));
      _mm512_mask_storeu_ps(ci+16,m1,_mm512_add_ps(_mm512_maskz_loadu_ps(m1,ci+16),_mm512_mul_ps(alpha,p1)));
      _mm512_mask_storeu_ps(ci+32,m2,_mm512_add_ps(_mm512_maskz_loadu_ps(m2,ci+32),_mm512_mul_ps(alpha,p2)));
    }
  }
}
/* V1(S, G): each K block in depth slices of S over groups of G row blocks, the group's
 * accumulators spilled between slices; the tile's slice (S x 48) is read from L1 by the group's
 * blocks after the first. */
static size_t SLICE=96, GROUP=4;
__attribute__((target("avx512f"))) static void k_slice(const mm_args *x, const mm_range *r) {
  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; _Alignas(64) float acc[32*8*48];
  size_t jb,lb,nl,i0,l0;
  for (jb=r->jb0;jb<r->jb1;jb+=48) {
    size_t nj=x->n-jb<48 ? x->n-jb : 48;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      kb512_pack(x,bt,jb,nj,lb,nl);
      for (i0=r->r0;i0<r->r1;i0+=GROUP*8) {
        size_t i1=i0+GROUP*8<r->r1 ? i0+GROUP*8 : r->r1,i,g;
        for (l0=0;l0<nl;l0+=SLICE) {
          size_t l1=l0+SLICE<nl ? l0+SLICE : nl;
          for (i=i0,g=0;i+8<=i1;i+=8,g++) kb512_slice(8,x,bt,i,lb,l0,l1,acc+g*384,l0==0);
          for (;i+4<=i1;i+=4,g++) kb512_slice(4,x,bt,i,lb,l0,l1,acc+g*384,l0==0);
          for (;i<i1;i++,g++) kb512_slice(1,x,bt,i,lb,l0,l1,acc+g*384,l0==0);
        }
        { size_t i2=i0; g=0;
          for (;i2+8<=i1;i2+=8,g++) kb512_commit(8,x,i2,jb,nj,acc+g*384);
          for (;i2+4<=i1;i2+=4,g++) kb512_commit(4,x,i2,jb,nj,acc+g*384);
          for (;i2<i1;i2++,g++) kb512_commit(1,x,i2,jb,nj,acc+g*384); }
      }
    }
  }
}
/* V2: the tree's loop with the tile's next depth lines prefetched PF depths ahead. */
static size_t PF=6;
__attribute__((target("avx512f"))) BCIR_MM_INLINE void kb512_block_pf(size_t mr, const mm_args *x,
    const float *bt, size_t i, size_t jb, size_t nj, size_t lb, size_t nl) {
  _Alignas(64) float acc[8*48];
  size_t l,pf=PF*48;
  (void)jb; (void)nj;
  /* the slice with a prefetch: written out so the prefetch sits in the loop */
  __m512 a0[8],a1[8],a2[8]; size_t r;
  for (r=0;r<mr;r++) { a0[r]=_mm512_setzero_ps(); a1[r]=_mm512_setzero_ps(); a2[r]=_mm512_setzero_ps(); }
  if (x->ta) {
    const float *ap=x->a+lb*x->lda+i; size_t lda=x->lda;
    for (l=0;l<nl;l++,ap+=lda) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      _mm_prefetch((const char *)(ap+BCIR_MM_KB_PREFETCH*lda),_MM_HINT_T0);
      _mm_prefetch((const char *)(bt+l*48+pf),_MM_HINT_T0); _mm_prefetch((const char *)(bt+l*48+pf+16),_MM_HINT_T0);
      _mm_prefetch((const char *)(bt+l*48+pf+32),_MM_HINT_T0);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r]);
        a0[r]=_mm512_fmadd_ps(av,b0,a0[r]); a1[r]=_mm512_fmadd_ps(av,b1,a1[r]); a2[r]=_mm512_fmadd_ps(av,b2,a2[r]);
      }
    }
  } else {
    const float *ap=x->a+i*x->lda+lb; size_t lda=x->lda;
    for (l=0;l<nl;l++) {
      __m512 b0=_mm512_load_ps(bt+l*48),b1=_mm512_load_ps(bt+l*48+16),b2=_mm512_load_ps(bt+l*48+32);
      _mm_prefetch((const char *)(bt+l*48+pf),_MM_HINT_T0); _mm_prefetch((const char *)(bt+l*48+pf+16),_MM_HINT_T0);
      _mm_prefetch((const char *)(bt+l*48+pf+32),_MM_HINT_T0);
      for (r=0;r<mr;r++) {
        __m512 av=_mm512_set1_ps(ap[r*lda+l]);
        a0[r]=_mm512_fmadd_ps(av,b0,a0[r]); a1[r]=_mm512_fmadd_ps(av,b1,a1[r]); a2[r]=_mm512_fmadd_ps(av,b2,a2[r]);
      }
    }
  }
  for (r=0;r<mr;r++) { _mm512_store_ps(acc+r*48,a0[r]); _mm512_store_ps(acc+r*48+16,a1[r]); _mm512_store_ps(acc+r*48+32,a2[r]); }
  kb512_commit(mr,x,i,jb,nj,acc);
}
__attribute__((target("avx512f"))) static void k_pf(const mm_args *x, const mm_range *r) {
  _Alignas(64) float bt[BCIR_MM_KBLOCK*48+16*48]; size_t i,jb,lb,nl;
  for (jb=r->jb0;jb<r->jb1;jb+=48) {
    size_t nj=x->n-jb<48 ? x->n-jb : 48;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      kb512_pack(x,bt,jb,nj,lb,nl);
      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block_pf(8,x,bt,i,jb,nj,lb,nl);
      for (;i+4<=r->r1;i+=4) kb512_block_pf(4,x,bt,i,jb,nj,lb,nl);
      for (;i<r->r1;i++) kb512_block_pf(1,x,bt,i,jb,nj,lb,nl);
    }
  }
}

/* V3: the tree's loop with the tile's source prefetched into L2 PD rows (depths, or for a
 * transposed B 16-column steps) ahead inside the pack. */
static size_t PD=16;
__attribute__((target("avx512f"))) static void kb512_pack_pf(const mm_args *x, float *bt, size_t jb,
    size_t nj, size_t lb, size_t nl) {
  size_t j,l;
  if (x->tb) {
    size_t jw=nj/16*16,lw=nl/16*16;
    for (j=0;j<jw;j+=16) for (l=0;l<lw;l+=16) {
      size_t q;
      if (l+16*PD/16<nl) for (q=0;q<16;q++) _mm_prefetch((const char *)(x->b+(jb+j+q)*x->ldb+lb+l+PD),_MM_HINT_T1);
      kb512_transpose16(x->b+(jb+j)*x->ldb+lb+l,x->ldb,bt+l*48+j,48);
    }
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
      if (l+PD<nl) {
        const char *ahead=(const char *)(row+PD*x->ldb);
        _mm_prefetch(ahead,_MM_HINT_T1); _mm_prefetch(ahead+64,_MM_HINT_T1);
        _mm_prefetch(ahead+128,_MM_HINT_T1); _mm_prefetch(ahead+191,_MM_HINT_T1);
      }
      _mm512_store_ps(bt+l*48,_mm512_maskz_loadu_ps(m0,row));
      _mm512_store_ps(bt+l*48+16,_mm512_maskz_loadu_ps(m1,row+16));
      _mm512_store_ps(bt+l*48+32,_mm512_maskz_loadu_ps(m2,row+32));
    }
  }
}
__attribute__((target("avx512f"))) static void k_pfpack(const mm_args *x, const mm_range *r) {
  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; size_t i,jb,lb,nl;
  for (jb=r->jb0;jb<r->jb1;jb+=48) {
    size_t nj=x->n-jb<48 ? x->n-jb : 48;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      kb512_pack_pf(x,bt,jb,nj,lb,nl);
      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block(8,x,bt,i,jb,nj,lb,nl);
      for (;i+4<=r->r1;i+=4) kb512_block(4,x,bt,i,jb,nj,lb,nl);
      for (;i<r->r1;i++) kb512_block(1,x,bt,i,jb,nj,lb,nl);
    }
  }
}
/* The tree's loop with the pack and the beta scaling timed (rdtsc; one thread). */
#include <x86intrin.h>
static unsigned long long T_PACK, T_BETA, T_ALL;
__attribute__((target("avx512f"))) static void k_timed(const mm_args *x, const mm_range *r) {
  _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; size_t i,jb,lb,nl; unsigned long long t0;
  for (jb=r->jb0;jb<r->jb1;jb+=48) {
    size_t nj=x->n-jb<48 ? x->n-jb : 48;
    for (lb=0;lb<x->k;lb+=nl) {
      nl=kblock_next(x->k-lb);
      t0=__rdtsc(); kb512_pack(x,bt,jb,nj,lb,nl); T_PACK+=__rdtsc()-t0;
      for (i=r->r0;i+8<=r->r1;i+=8) kb512_block(8,x,bt,i,jb,nj,lb,nl);
      for (;i+4<=r->r1;i+=4) kb512_block(4,x,bt,i,jb,nj,lb,nl);
      for (;i<r->r1;i++) kb512_block(1,x,bt,i,jb,nj,lb,nl);
    }
  }
}
/* The pool's partition, run in task order on the caller (gemm_tasks.py's one-thread routing). */
static size_t ROWS=BCIR_MM_ROWS;
static void run_tasks(kern_fn k, int ta, int tb, size_t m, size_t n, size_t kk, float alpha,
    const float *a, const float *b, float beta, float *c) {
  mm_job j; size_t tasks,t;
  mm_job_init(&j,MM_KBLOCK,ta,tb,m,n,kk,alpha,a,b,beta,c);
  j.rowsets=(m+ROWS-1)/ROWS; tasks=j.cols*j.rowsets;
  for (t=0;t<tasks;t++) {
    mm_range r; size_t col=t%j.cols,row=t/j.cols,i,jj;
    r.jb0=col*j.nr; r.jb1=r.jb0+j.nr<j.x.n ? r.jb0+j.nr : j.x.n;
    r.r0=row*ROWS; r.r1=r.r0+ROWS<j.x.m ? r.r0+ROWS : j.x.m;
    { unsigned long long t0=__rdtsc();
      for (i=r.r0;i<r.r1;i++) { float *ci=j.x.c+i*j.x.ldc; for (jj=r.jb0;jj<r.jb1;jj++) ci[jj]=j.beta==0.0f ? 0.0f : j.beta*ci[jj]; }
      T_BETA+=__rdtsc()-t0; t0=__rdtsc();
      if (j.x.k) k(&j.x,&r);
      T_ALL+=__rdtsc()-t0; }
  }
}
/* The same partition on a pool of POOL_THREADS: one pool task per GEMM task. */
typedef struct bench_job { kern_fn k; mm_job j; } bench_job;
static void bench_task(void *ctx, size_t t) {
  bench_job *b=(bench_job *)ctx; mm_job *j=&b->j; mm_range r; size_t col=t%j->cols,row=t/j->cols,i,jj;
  r.jb0=col*j->nr; r.jb1=r.jb0+j->nr<j->x.n ? r.jb0+j->nr : j->x.n;
  r.r0=row*ROWS; r.r1=r.r0+ROWS<j->x.m ? r.r0+ROWS : j->x.m;
  for (i=r.r0;i<r.r1;i++) { float *ci=j->x.c+i*j->x.ldc; for (jj=r.jb0;jj<r.jb1;jj++) ci[jj]=j->beta==0.0f ? 0.0f : j->beta*ci[jj]; }
  if (j->x.k) b->k(&j->x,&r);
}
static bcir_tensor_pool *POOL;
static void run_pool(kern_fn k, int ta, int tb, size_t m, size_t n, size_t kk, float alpha,
    const float *a, const float *b, float beta, float *c) {
  bench_job bj; bj.k=k; mm_job_init(&bj.j,MM_KBLOCK,ta,tb,m,n,kk,alpha,a,b,beta,c);
  bj.j.rowsets=(m+ROWS-1)/ROWS;
  bcir_tensor_pool_run(POOL,bench_task,&bj,bj.j.cols*bj.j.rowsets);
}
static float *big(size_t n) {
  size_t bytes=(n*sizeof(float)+(2u<<20)-1)/(2u<<20)*(2u<<20); void *p=NULL;
  if (posix_memalign(&p,2u<<20,bytes)) { perror("posix_memalign"); exit(2); }
  madvise(p,bytes,MADV_HUGEPAGE); memset(p,0,bytes); return (float *)p;
}
static int cmp_d(const void *x, const void *y) { double a=*(const double *)x,b=*(const double *)y; return (a>b)-(a<b); }

/* the step's GEMM shapes (gemm-tasks-t4.json): ta, tb, m, n, k, calls a step */
static const int SH[][6]={
  {0,1,1024,384,384,12},{0,1,1024,192,384,12},{0,1,1024,1024,384,12},{0,1,1024,384,1024,6},
  {0,0,1024,1024,384,6},{1,0,384,1024,1024,6},{0,0,1024,384,1024,12},{1,0,1024,384,1024,12},
  {0,0,1024,384,384,12},{1,0,384,384,1024,12},{0,0,1024,384,192,12},{1,0,192,384,1024,12},
};

__attribute__((target("avx512f"))) static void ceiling(void) {
  /* the inner loop over one tile, no C traffic: rows from a 1024 x 1024 A (ta=0, lda 1024 and 448) */
  static _Alignas(64) float bt[BCIR_MM_KBLOCK*48]; static _Alignas(64) float acc[8*48];
  size_t nls[]={64,96,128,192,288,384,448},q,lda_i; float *a=big(1024*1024);
  for (q=0;q<BCIR_MM_KBLOCK*48;q++) bt[q]=0.001f*(float)(q%97);
  for (q=0;q<1024*1024;q++) a[q]=0.001f*(float)(q%89);
  for (lda_i=0;lda_i<2;lda_i++) for (q=0;q<sizeof(nls)/sizeof(nls[0]);q++) {
    mm_args x; size_t nl=nls[q],rep,i; double t0,best=1e30; int round;
    memset(&x,0,sizeof(x)); x.ta=0; x.a=a; x.lda=lda_i ? 448 : 1024; x.alpha=1.0f;
    for (round=0;round<7;round++) {
      t0=now_ns();
      for (rep=0;rep<40;rep++) for (i=0;i<128;i+=8) kb512_slice(8,&x,bt,i,0,0,nl,acc,1);
      { double t=now_ns()-t0; if (t<best) best=t; }
    }
    printf("{\"probe\": \"ceiling\", \"lda\": %zu, \"depth\": %zu, \"tile_kb\": %.1f, \"gflops\": %.1f}\n",
        x.lda,nl,nl*48*4/1024.0,2.0*40*128*48*(double)nl/best);
  }
}

int main(int argc, char **argv) {
  int reps=argc>1 ? atoi(argv[1]) : 9; size_t s,v;
  struct { const char *name; kern_fn k; size_t slice, group, pf, rows; } V[]={
    {"tree",k_tree,0,0,0,128},{"tree-r256",k_tree,0,0,0,256},{"tree-r64",k_tree,0,0,0,64},
    {"slice96g4",k_slice,96,4,0,128},{"slice128g2",k_slice,128,2,0,128},{"slice64g8",k_slice,64,8,0,128},
    {"slice144g4",k_slice,144,4,0,128},{"slice96g16",k_slice,96,16,0,128},
    {"pf4",k_pf,0,0,4,128},{"pf8",k_pf,0,0,8,128},
    {"tree-r512",k_tree,0,0,0,512},{"pack-pd8",k_pfpack,0,0,8,128},{"pack-pd16",k_pfpack,0,0,16,128},
    {"pack-pd32",k_pfpack,0,0,32,128},{"pack-pd16-r256",k_pfpack,0,0,16,256},
  };
  size_t nv=sizeof(V)/sizeof(V[0]);
  float *a,*b,*c0,*c1,*init;
  void (*runner)(kern_fn,int,int,size_t,size_t,size_t,float,const float *,const float *,float,float *)=run_tasks;
  if (argc>1 && !strcmp(argv[1],"ceiling")) { ceiling(); return 0; }
  if (argc>1 && !strcmp(argv[1],"split")) {  /* ./bench split: the pack's and the beta scaling's share of the tree loop, one thread */
    float *a2=big(1024*1024),*b2=big(1024*1024),*c2=big(1024*1024); size_t q; int rep;
    for (q=0;q<1024*1024;q++) { a2[q]=0.01f*(float)((q*7)%113)-0.5f; b2[q]=0.01f*(float)((q*5)%127)-0.6f; }
    for (s=0;s<sizeof(SH)/sizeof(SH[0]);s++) {
      int ta=SH[s][0],tb=SH[s][1]; size_t m=(size_t)SH[s][2],n=(size_t)SH[s][3],kk=(size_t)SH[s][4];
      ROWS=BCIR_MM_ROWS; run_tasks(k_timed,ta,tb,m,n,kk,1.0f,a2,b2,0.0f,c2);
      T_PACK=T_BETA=T_ALL=0;
      for (rep=0;rep<9;rep++) run_tasks(k_timed,ta,tb,m,n,kk,1.0f,a2,b2,0.0f,c2);
      printf("{\"ta\": %d, \"tb\": %d, \"m\": %zu, \"n\": %zu, \"k\": %zu, \"pack_share\": %.4f, \"beta_share\": %.4f, \"kernel_mcycles\": %.2f}\n",
          ta,tb,m,n,kk,(double)T_PACK/(double)(T_ALL+T_BETA),(double)T_BETA/(double)(T_ALL+T_BETA),(double)T_ALL/9e6);
    }
    return 0;
  }
  if (argc>2 && !strcmp(argv[1],"pool")) {  /* ./bench pool REPS: the same rows on a pool of four */
    static _Alignas(max_align_t) unsigned char mem[65536];
    size_t need=bcir_tensor_pool_size(4);
    if (!need || need>sizeof(mem)) { fprintf(stderr,"no pool\n"); return 3; }
    POOL=bcir_tensor_pool_init(mem,sizeof(mem),4); runner=run_pool; reps=atoi(argv[2]);
  }
  if (bcir_tensor_mm_fused_variants()<4) { fprintf(stderr,"no AVX-512\n"); return 3; }
  a=big(1024*1024); b=big(1024*1024); c0=big(1024*1024); c1=big(1024*1024); init=big(1024*1024);
  for (s=0;s<1024*1024;s++) { a[s]=0.01f*(float)((s*7)%113)-0.5f; b[s]=0.01f*(float)((s*5)%127)-0.6f; init[s]=0.001f*(float)(s%31); }
  for (s=0;s<sizeof(SH)/sizeof(SH[0]);s++) {
    int ta=SH[s][0],tb=SH[s][1]; size_t m=(size_t)SH[s][2],n=(size_t)SH[s][3],kk=(size_t)SH[s][4];
    double flop=2.0*(double)m*(double)n*(double)kk,*t=malloc(sizeof(double)*nv*(size_t)reps); int rr;
    ROWS=BCIR_MM_ROWS; memcpy(c0,init,m*n*sizeof(float)); run_tasks(k_tree,ta,tb,m,n,kk,1.0f,a,b,1.0f,c0);
    for (v=0;v<nv;v++) {
      SLICE=V[v].slice ? V[v].slice : 96; GROUP=V[v].group ? V[v].group : 4; PF=V[v].pf ? V[v].pf : 6; PD=V[v].pf ? V[v].pf : 16; ROWS=V[v].rows;
      memcpy(c1,init,m*n*sizeof(float)); runner(V[v].k,ta,tb,m,n,kk,1.0f,a,b,1.0f,c1);
      if (memcmp(c0,c1,m*n*sizeof(float))) { printf("{\"shape\": %zu, \"variant\": \"%s\", \"identical\": false}\n",s,V[v].name); return 1; }
    }
    for (rr=0;rr<reps;rr++) for (v=0;v<nv;v++) {  /* interleaved: round rr runs every variant once */
      double t0;
      SLICE=V[v].slice ? V[v].slice : 96; GROUP=V[v].group ? V[v].group : 4; PF=V[v].pf ? V[v].pf : 6; PD=V[v].pf ? V[v].pf : 16; ROWS=V[v].rows;
      memcpy(c1,init,m*n*sizeof(float));
      t0=now_ns(); runner(V[v].k,ta,tb,m,n,kk,1.0f,a,b,0.0f,c1); t[v*(size_t)reps+(size_t)rr]=now_ns()-t0;
    }
    for (v=0;v<nv;v++) {
      double med; qsort(t+v*(size_t)reps,(size_t)reps,sizeof(double),cmp_d); med=t[v*(size_t)reps+(size_t)reps/2];
      printf("{\"threads\": %d, \"ta\": %d, \"tb\": %d, \"m\": %zu, \"n\": %zu, \"k\": %zu, \"calls\": %d, \"variant\": \"%s\", \"identical\": true, \"median_ms\": %.4f, \"gflops\": %.1f}\n",
          POOL ? 4 : 1,ta,tb,m,n,kk,SH[s][5],V[v].name,med/1e6,flop/med);
    }
    free(t); fflush(stdout);
  }
  return 0;
}
