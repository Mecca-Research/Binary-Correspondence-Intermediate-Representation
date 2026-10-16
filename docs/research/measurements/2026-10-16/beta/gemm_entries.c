/* NDT-BETA's A/B instrument (2026-10-16). The step's GEMM shapes through the tensor unit's public kblock entries -- bcir_tensor_mm_kblock
 * (serial) and bcir_tensor_mm_kblock_pool (four threads) -- at the betas the trainer passes (0 for
 * the forward and a first input adjoint, 1 for the weight gradients), for one build of the unit.
 * Built twice (two trees' bcir_tensor.c), run alternately; every C is digested so the two builds'
 * results can be compared bit for bit.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I TREE gemm_entries.c TREE/bcir_tensor.c -lm
 *   ./a.out REPS THREADS    -- one JSON line per shape: median ms and an FNV-1a digest of C
 */
#define _GNU_SOURCE 1
#include "bcir_tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now_ns(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec;
}
static float *big(size_t n) {
  size_t bytes=(n*sizeof(float)+(2u<<20)-1)/(2u<<20)*(2u<<20); void *p=NULL;
  if (posix_memalign(&p,2u<<20,bytes)) { perror("posix_memalign"); exit(2); }
  madvise(p,bytes,MADV_HUGEPAGE); memset(p,0,bytes); return (float *)p;
}
static int cmp_d(const void *x, const void *y) { double a=*(const double *)x,b=*(const double *)y; return (a>b)-(a<b); }
static uint64_t fnv(const float *c, size_t n) {
  uint64_t h=1469598103934665603ull; size_t i; const unsigned char *p=(const unsigned char *)c;
  for (i=0;i<n*sizeof(float);i++) { h^=p[i]; h*=1099511628211ull; }
  return h;
}
/* ta, tb, m, n, k, calls a step, beta */
static const int SH[][7]={
  {0,1,1024,384,384,12,0},{0,1,1024,192,384,12,0},{0,1,1024,1024,384,12,0},{0,1,1024,384,1024,6,0},
  {0,0,1024,1024,384,6,0},{1,0,384,1024,1024,6,1},{0,0,1024,384,1024,12,0},{1,0,1024,384,1024,12,1},
  {0,0,1024,384,384,12,0},{1,0,384,384,1024,12,1},{0,0,1024,384,192,12,1},{1,0,192,384,1024,12,1},
};
int main(int argc, char **argv) {
  int reps=argc>1 ? atoi(argv[1]) : 9,threads=argc>2 ? atoi(argv[2]) : 1,r; size_t s,i;
  static _Alignas(max_align_t) unsigned char mem[65536]; bcir_tensor_pool *pool=NULL;
  float *a=big(1024*1024),*b=big(1024*1024),*c=big(1024*1024),*init=big(1024*1024);
  double *t=malloc(sizeof(double)*(size_t)reps);
  if (threads>1) {
    size_t need=bcir_tensor_pool_size((unsigned)threads);
    if (!need || need>sizeof(mem)) { fprintf(stderr,"no pool\n"); return 3; }
    pool=bcir_tensor_pool_init(mem,sizeof(mem),(unsigned)threads);
  }
  for (i=0;i<1024*1024;i++) { a[i]=0.01f*(float)((i*7)%113)-0.5f; b[i]=0.01f*(float)((i*5)%127)-0.6f; init[i]=0.001f*(float)(i%31); }
  for (s=0;s<sizeof(SH)/sizeof(SH[0]);s++) {
    int ta=SH[s][0],tb=SH[s][1]; size_t m=(size_t)SH[s][2],n=(size_t)SH[s][3],k=(size_t)SH[s][4];
    float beta=(float)SH[s][6]; uint64_t h; double flop=2.0*(double)m*(double)n*(double)k;
    memcpy(c,init,m*n*sizeof(float));
    if (pool) bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,a,b,beta,c);
    else bcir_tensor_mm_kblock(NULL,ta,tb,m,n,k,1.0f,a,b,beta,c);
    h=fnv(c,m*n);
    for (r=0;r<reps;r++) {
      double t0;
      memcpy(c,init,m*n*sizeof(float));
      t0=now_ns();
      if (pool) bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,a,b,beta,c);
      else bcir_tensor_mm_kblock(NULL,ta,tb,m,n,k,1.0f,a,b,beta,c);
      t[r]=now_ns()-t0;
    }
    qsort(t,(size_t)reps,sizeof(double),cmp_d);
    printf("{\"threads\": %d, \"ta\": %d, \"tb\": %d, \"m\": %zu, \"n\": %zu, \"k\": %zu, \"beta\": %g, \"calls\": %d, "
        "\"median_ms\": %.4f, \"gflops\": %.1f, \"c_fnv1a\": \"%016llx\"}\n",
        threads,ta,tb,m,n,k,(double)beta,SH[s][5],t[reps/2]/1e6,flop/t[reps/2],(unsigned long long)h);
    fflush(stdout);
  }
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
