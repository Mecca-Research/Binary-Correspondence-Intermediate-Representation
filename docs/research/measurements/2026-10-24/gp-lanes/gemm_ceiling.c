/* The k-blocked GEMM's all-core ceiling on this host: W threads started together, each running the
 * same shape CALLS times back to back on operands of its own (bcir_tensor_mm_kblock, serial, no
 * pool), against one thread alone (W 1) and against the pool's one GEMM split over W threads
 * (bcir_tensor_mm_kblock_pool). Median of ROUNDS rounds. Every value is each entry's own; nothing
 * here is checked but the arithmetic each entry already holds.
 * usage: gemm_ceiling W TA TB M N K CALLS ROUNDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n)) exit(3); memset(p,0,n); return p; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
typedef struct { int ta,tb,calls; size_t m,n,k; float *a,*b,*c; double t0,t1; pthread_barrier_t *bar; } arm;
static void *run(void *p) {
  arm *x=(arm *)p; int c;
  pthread_barrier_wait(x->bar); x->t0=now();
  for (c=0;c<x->calls;c++) bcir_tensor_mm_kblock(NULL,x->ta,x->tb,x->m,x->n,x->k,1.0f,x->a,x->b,0.0f,x->c);
  x->t1=now(); return NULL;
}
static void fill(float *p,size_t n,unsigned s) { size_t i; for (i=0;i<n;i++) p[i]=(float)((i*s)%13)*0.1f-0.6f; }
int main(int argc,char **argv) {
  if (argc!=9) { fprintf(stderr,"usage: gemm_ceiling W TA TB M N K CALLS ROUNDS\n"); return 2; }
  unsigned w=(unsigned)atoi(argv[1]),q; int ta=atoi(argv[2]),tb=atoi(argv[3]),calls=atoi(argv[7]),rounds=atoi(argv[8]),r,c;
  size_t m=strtoul(argv[4],0,10),n=strtoul(argv[5],0,10),k=strtoul(argv[6],0,10);
  double *own=malloc(sizeof(double)*rounds),*one=malloc(sizeof(double)*rounds),*pool_ms=malloc(sizeof(double)*rounds),fl=2.0*m*n*k;
  arm *x=calloc(w,sizeof(arm)); pthread_t *th=calloc(w,sizeof(pthread_t)); pthread_barrier_t bar;
  for (q=0;q<w;q++) {
    x[q].ta=ta; x[q].tb=tb; x[q].calls=calls; x[q].m=m; x[q].n=n; x[q].k=k; x[q].bar=&bar;
    x[q].a=alloc64(m*k*4); x[q].b=alloc64(k*n*4); x[q].c=alloc64(m*n*4);
    fill(x[q].a,m*k,3u+q); fill(x[q].b,k*n,7u+q);
  }
  size_t ps=bcir_tensor_pool_size(w); void *pm=alloc64((ps+63)/64*64);
  bcir_tensor_pool *pool=w>1 ? bcir_tensor_pool_init(pm,ps,w) : NULL;
  for (r=-1;r<rounds;r++) {
    double t0,end=0.0,start=1e300;
    /* one thread alone, the caller, on arm 0's operands */
    t0=now(); for (c=0;c<calls;c++) bcir_tensor_mm_kblock(NULL,ta,tb,m,n,k,1.0f,x[0].a,x[0].b,0.0f,x[0].c);
    if (r>=0) one[r]=(now()-t0)*1e3/calls;
    /* W threads at once, each its own operands */
    pthread_barrier_init(&bar,NULL,w);
    for (q=0;q<w;q++) if (pthread_create(&th[q],NULL,run,&x[q])) return 4;
    for (q=0;q<w;q++) pthread_join(th[q],NULL);
    pthread_barrier_destroy(&bar);
    for (q=0;q<w;q++) { if (x[q].t0<start) start=x[q].t0; if (x[q].t1>end) end=x[q].t1; }
    if (r>=0) own[r]=(end-start)*1e3/calls;  /* ms for W GEMMs, one a thread */
    /* the pool's one GEMM over W threads, back to back */
    t0=now(); for (c=0;c<calls;c++) bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,x[0].a,x[0].b,0.0f,x[0].c);
    if (r>=0) pool_ms[r]=(now()-t0)*1e3/calls;
  }
  qsort(one,(size_t)rounds,sizeof(double),cmpd); qsort(own,(size_t)rounds,sizeof(double),cmpd); qsort(pool_ms,(size_t)rounds,sizeof(double),cmpd);
  double g1=fl/(one[rounds/2]*1e-3)/1e9,gw=w*fl/(own[rounds/2]*1e-3)/1e9,gp=fl/(pool_ms[rounds/2]*1e-3)/1e9;
  printf("{\"w\": %u, \"ta\": %d, \"tb\": %d, \"m\": %zu, \"n\": %zu, \"k\": %zu, \"one_gflops\": %.1f, "
         "\"own_gflops\": %.1f, \"own_per_thread\": %.1f, \"pool_gflops\": %.1f, \"pool_over_own\": %.3f, "
         "\"own_over_w_one\": %.3f}\n",w,ta,tb,m,n,k,g1,gw,gw/w,gp,gp/gw,gw/(w*g1));
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
