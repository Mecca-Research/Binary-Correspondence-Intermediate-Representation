/* The GEO step's GEMM jobs at one context, each as the trainer issues it (bcir_tensor_mm_group_pool
 * and bcir_tensor_mm_pair_pool under the k-blocked contract, bcir_tensor_mm_kblock_pool alone):
 * the job on a pool of W threads back to back, against the job's ceiling -- W threads started
 * together, each running the whole job serially on operands of its own -- and one thread alone.
 * Median of ROUNDS rounds of CALLS jobs. usage: gemm_jobs W ROWS CALLS ROUNDS (d 128, kv 128,
 * ffn 256, vocab 260: the GEO dimensions) */
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
static float *buf(size_t n,unsigned s) { float *p=alloc64(n*4); size_t i; for (i=0;i<n;i++) p[i]=(float)((i*s)%13)*0.1f-0.6f; return p; }
enum { ALONE, PAIR, GROUP };
typedef struct { const char *name; int kind; size_t nx,ny; bcir_tensor_mm_call x[3],y[3]; } job;
static bcir_tensor_mm_call call(int ta,int tb,size_t m,size_t n,size_t k,float beta,unsigned s) {
  bcir_tensor_mm_call c; c.ta=ta; c.tb=tb; c.m=m; c.n=n; c.k=k; c.alpha=1.0f; c.beta=beta;
  c.a=buf(m*k,s); c.b=buf(k*n,s+1); c.c=buf(m*n,s+2); return c;
}
static void run_job(bcir_tensor_pool *pool,const job *j) {
  if (j->kind==ALONE) { const bcir_tensor_mm_call *c=&j->y[0]; bcir_tensor_mm_kblock_pool(pool,c->ta,c->tb,c->m,c->n,c->k,c->alpha,c->a,c->b,c->beta,c->c); }
  else if (j->kind==PAIR) bcir_tensor_mm_pair_pool(pool,2,&j->x[0],&j->y[0]);
  else bcir_tensor_mm_group_pool(pool,2,j->nx ? j->x : NULL,j->nx,j->y,j->ny);
}
static double flops(const job *j) {
  double f=0; size_t q;
  for (q=0;q<j->nx;q++) f+=2.0*j->x[q].m*j->x[q].n*j->x[q].k;
  for (q=0;q<j->ny;q++) f+=2.0*j->y[q].m*j->y[q].n*j->y[q].k;
  return f;
}
/* a private copy of a job: every operand its own */
static job copy(const job *j,unsigned s) {
  job c=*j; size_t q;
  for (q=0;q<j->nx;q++) c.x[q]=call(j->x[q].ta,j->x[q].tb,j->x[q].m,j->x[q].n,j->x[q].k,j->x[q].beta,s+3*q);
  for (q=1;q<j->nx;q++) c.x[q].c=c.x[0].c;  /* a chain writes one C */
  for (q=0;q<j->ny;q++) c.y[q]=call(j->y[q].ta,j->y[q].tb,j->y[q].m,j->y[q].n,j->y[q].k,j->y[q].beta,s+17+3*q);
  return c;
}
typedef struct { const job *j; int calls; double t0,t1; pthread_barrier_t *bar; } arm;
static void *run(void *p) { arm *a=(arm *)p; int c; pthread_barrier_wait(a->bar); a->t0=now(); for (c=0;c<a->calls;c++) run_job(NULL,a->j); a->t1=now(); return NULL; }
int main(int argc,char **argv) {
  if (argc!=5) { fprintf(stderr,"usage: gemm_jobs W ROWS CALLS ROUNDS\n"); return 2; }
  unsigned w=(unsigned)atoi(argv[1]),q; size_t n=strtoul(argv[2],0,10),d=128,kv=128,f=256,v=260; int calls=atoi(argv[3]),rounds=atoi(argv[4]),r,c,i;
  job js[10]; int nj=0;
  memset(js,0,sizeof js);
  /* the forward: q, k, v one job; gate and up one job; o, down and the head alone */
  js[nj].name="fwd.qkv (rest.nt)"; js[nj].kind=GROUP; js[nj].ny=3; js[nj].y[0]=call(0,1,n,d,d,0,1); js[nj].y[1]=call(0,1,n,kv,d,0,5); js[nj].y[2]=call(0,1,n,kv,d,0,9);
  js[nj].y[1].a=js[nj].y[0].a; js[nj].y[2].a=js[nj].y[0].a; nj++;
  js[nj].name="fwd.gate_up (rest.nt)"; js[nj].kind=GROUP; js[nj].ny=2; js[nj].y[0]=call(0,1,n,f,d,0,13); js[nj].y[1]=call(0,1,n,f,d,0,17); js[nj].y[1].a=js[nj].y[0].a; nj++;
  js[nj].name="fwd.o (alone.nt)"; js[nj].kind=ALONE; js[nj].ny=1; js[nj].y[0]=call(0,1,n,d,d,0,21); nj++;
  js[nj].name="fwd.down (alone.nt)"; js[nj].kind=ALONE; js[nj].ny=1; js[nj].y[0]=call(0,1,n,d,f,0,25); nj++;
  js[nj].name="fwd.head (alone.nt)"; js[nj].kind=ALONE; js[nj].ny=1; js[nj].y[0]=call(0,1,n,v,d,0,29); nj++;
  /* the backward: a linear's dx then dW as one pair; q/k/v's and gate/up's input adjoints a chain into one dx beside their weight adjoints */
  js[nj].name="bwd.head (pair.nn)"; js[nj].kind=PAIR; js[nj].nx=1; js[nj].ny=1; js[nj].x[0]=call(0,0,n,d,v,0,33); js[nj].y[0]=call(1,0,v,d,n,0,37); nj++;
  js[nj].name="bwd.o (pair.nn)"; js[nj].kind=PAIR; js[nj].nx=1; js[nj].ny=1; js[nj].x[0]=call(0,0,n,d,d,0,41); js[nj].y[0]=call(1,0,d,d,n,0,45); nj++;
  js[nj].name="bwd.down (pair.nn)"; js[nj].kind=PAIR; js[nj].nx=1; js[nj].ny=1; js[nj].x[0]=call(0,0,n,f,d,0,49); js[nj].y[0]=call(1,0,d,f,n,0,53); nj++;
  js[nj].name="bwd.qkv (chain.nn)"; js[nj].kind=GROUP; js[nj].nx=3; js[nj].ny=3;
  js[nj].x[0]=call(0,0,n,d,d,0,57); js[nj].x[1]=call(0,0,n,d,kv,1,61); js[nj].x[2]=call(0,0,n,d,kv,1,65); js[nj].x[1].c=js[nj].x[0].c; js[nj].x[2].c=js[nj].x[0].c;
  js[nj].y[0]=call(1,0,d,d,n,0,69); js[nj].y[1]=call(1,0,kv,d,n,0,73); js[nj].y[2]=call(1,0,kv,d,n,0,77); nj++;
  js[nj].name="bwd.gate_up (chain.nn)"; js[nj].kind=GROUP; js[nj].nx=2; js[nj].ny=2;
  js[nj].x[0]=call(0,0,n,d,f,0,81); js[nj].x[1]=call(0,0,n,d,f,1,85); js[nj].x[1].c=js[nj].x[0].c;
  js[nj].y[0]=call(1,0,f,d,n,0,89); js[nj].y[1]=call(1,0,f,d,n,0,93); nj++;
  size_t ps=bcir_tensor_pool_size(w); void *pm=alloc64((ps+63)/64*64);
  bcir_tensor_pool *pool=bcir_tensor_pool_init(pm,ps,w);
  double *tp=malloc(sizeof(double)*rounds),*to=malloc(sizeof(double)*rounds),*t1=malloc(sizeof(double)*rounds);
  for (i=0;i<nj;i++) {
    job priv[8]; arm a[8]; pthread_t th[8]; pthread_barrier_t bar;
    for (q=0;q<w;q++) priv[q]=copy(&js[i],100u+7u*q);
    for (r=-1;r<rounds;r++) {
      double t0=now(),start=1e300,end=0;
      for (c=0;c<calls;c++) run_job(pool,&js[i]);
      if (r>=0) tp[r]=(now()-t0)*1e6/calls;
      t0=now(); for (c=0;c<calls;c++) run_job(NULL,&js[i]);
      if (r>=0) t1[r]=(now()-t0)*1e6/calls;
      pthread_barrier_init(&bar,NULL,w);
      for (q=0;q<w;q++) { a[q].j=&priv[q]; a[q].calls=calls; a[q].bar=&bar; if (pthread_create(&th[q],NULL,run,&a[q])) return 4; }
      for (q=0;q<w;q++) pthread_join(th[q],NULL);
      pthread_barrier_destroy(&bar);
      for (q=0;q<w;q++) { if (a[q].t0<start) start=a[q].t0; if (a[q].t1>end) end=a[q].t1; }
      if (r>=0) to[r]=(end-start)*1e6/calls/w;  /* the ceiling's time a job */
    }
    qsort(tp,(size_t)rounds,sizeof(double),cmpd); qsort(to,(size_t)rounds,sizeof(double),cmpd); qsort(t1,(size_t)rounds,sizeof(double),cmpd);
    double fl=flops(&js[i]);
    printf("{\"rows\": %zu, \"job\": \"%s\", \"mflop\": %.1f, \"pool_us\": %.1f, \"ceiling_us\": %.1f, \"one_us\": %.1f, "
           "\"pool_gflops\": %.1f, \"ceiling_gflops\": %.1f, \"pool_over_ceiling_time\": %.3f}\n",
           n,js[i].name,fl/1e6,tp[rounds/2],to[rounds/2],t1[rounds/2],fl/tp[rounds/2]/1e3,fl/to[rounds/2]/1e3,tp[rounds/2]/to[rounds/2]);
    fflush(stdout);
  }
  bcir_tensor_pool_stop(pool);
  return 0;
}
