/* ROPE-BOUND: why the table's rotation ran at x0.47-0.65 of the per-call one under clang where ROPE-TABLE-0's
 * prototype ran at x0.22-0.31. Five forms of the same rotation reading a table -- every value checked byte for
 * byte against the first before any is timed -- each run on the same pool over the same buffers, each the median
 * of ROUNDS rounds of CALLS calls, forward and inverse alternating:
 *   first  -- a6f8feb1's rope_lent_task: ranges of BCIR_ROPE_J pairs, the inner loop bounded by the range's count
 *   fix    -- ROPE-BOUND's: the same through rope_turn, bounded by the head's half where the head fits one range
 *   proto  -- ROPE-TABLE-0's lent_task (../gp-lanes/rope_bench.c): no ranges, bounded by the head's half
 *   half_k -- first with its bound half-k0 (no min with the range: the same count where the head fits one range)
 *   min64  -- proto with its bound min(half, BCIR_ROPE_J) (the same count where the head fits one range)
 * The inner loop's operations are the same in all five; only its bound's expression differs.
 * usage: bound_cause THREADS T CALLS ROUNDS (batch 4, 4 heads of 32: the GEO step's rotation) */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define BCIR_ROW_CHUNK 16
#define BCIR_ROPE_J 64
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
typedef struct lent_job { size_t b,t,heads,dim; const float *table; float *x; int inverse; } lent_job;
static void first_task(void *ctx, size_t task) {
  lent_job *j=(lent_job *)ctx; size_t r,pos,h,k,k0,half=j->dim/2;
  size_t p0=task*BCIR_ROW_CHUNK,p1=p0+BCIR_ROW_CHUNK<j->t ? p0+BCIR_ROW_CHUNK : j->t;
  for (pos=p0;pos<p1;pos++) for (k0=0;k0<half;k0+=BCIR_ROPE_J) {
    size_t nk=half-k0<BCIR_ROPE_J ? half-k0 : BCIR_ROPE_J; float s[BCIR_ROPE_J];
    const float *c=j->table+pos*j->dim+k0,*sin0=c+half;
    for (k=0;k<nk;k++) s[k]=sin0[k]*(j->inverse ? -1.0f : 1.0f);
    for (r=0;r<j->b;r++) for (h=0;h<j->heads;h++) {
      float *lo=j->x+((r*j->t+pos)*j->heads+h)*j->dim+k0,*hi=lo+half;
      for (k=0;k<nk;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
    }
  }
}
static inline __attribute__((always_inline)) void rope_turn(size_t b, size_t t, size_t heads, size_t dim,
    size_t pos, size_t k0, size_t n, const float *c, const float *s, float *x) {
  size_t r,h,k,half=dim/2;
  for (r=0;r<b;r++) for (h=0;h<heads;h++) {
    float *lo=x+((r*t+pos)*heads+h)*dim+k0,*hi=lo+half;
    for (k=0;k<n;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
  }
}
static void fix_task(void *ctx, size_t task) {
  lent_job *j=(lent_job *)ctx; size_t pos,k,k0,half=j->dim/2;
  size_t p0=task*BCIR_ROW_CHUNK,p1=p0+BCIR_ROW_CHUNK<j->t ? p0+BCIR_ROW_CHUNK : j->t;
  for (pos=p0;pos<p1;pos++) for (k0=0;k0<half;k0+=BCIR_ROPE_J) {
    size_t nk=half-k0<BCIR_ROPE_J ? half-k0 : BCIR_ROPE_J; float s[BCIR_ROPE_J];
    const float *c=j->table+pos*j->dim+k0,*sin0=c+half;
    for (k=0;k<nk;k++) s[k]=sin0[k]*(j->inverse ? -1.0f : 1.0f);
    if (half<=BCIR_ROPE_J) rope_turn(j->b,j->t,j->heads,j->dim,pos,0,half,c,s,j->x);
    else rope_turn(j->b,j->t,j->heads,j->dim,pos,k0,nk,c,s,j->x);
  }
}
static void proto_task(void *ctx, size_t task) {
  lent_job *j=(lent_job *)ctx; size_t half=j->dim/2,p0=task*16,p1=p0+16<j->t ? p0+16 : j->t,pos,r,h,k;
  for (pos=p0;pos<p1;pos++) {
    const float *c=j->table+pos*j->dim,*s0=c+half; float s[64];
    for (k=0;k<half;k++) s[k]=s0[k]*(j->inverse ? -1.0f : 1.0f);
    for (r=0;r<j->b;r++) for (h=0;h<j->heads;h++) {
      float *lo=j->x+((r*j->t+pos)*j->heads+h)*j->dim,*hi=lo+half;
      for (k=0;k<half;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
    }
  }
}
static void half_k_task(void *ctx, size_t task) {
  lent_job *j=(lent_job *)ctx; size_t r,pos,h,k,k0,half=j->dim/2;
  size_t p0=task*BCIR_ROW_CHUNK,p1=p0+BCIR_ROW_CHUNK<j->t ? p0+BCIR_ROW_CHUNK : j->t;
  for (pos=p0;pos<p1;pos++) for (k0=0;k0<half;k0+=BCIR_ROPE_J) {
    size_t nk=half-k0; float s[BCIR_ROPE_J];  /* the head fits one range here: nk <= BCIR_ROPE_J */
    const float *c=j->table+pos*j->dim+k0,*sin0=c+half;
    for (k=0;k<nk;k++) s[k]=sin0[k]*(j->inverse ? -1.0f : 1.0f);
    for (r=0;r<j->b;r++) for (h=0;h<j->heads;h++) {
      float *lo=j->x+((r*j->t+pos)*j->heads+h)*j->dim+k0,*hi=lo+half;
      for (k=0;k<nk;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
    }
  }
}
static void min64_task(void *ctx, size_t task) {
  lent_job *j=(lent_job *)ctx; size_t half=j->dim/2,p0=task*16,p1=p0+16<j->t ? p0+16 : j->t,pos,r,h,k;
  size_t nk=half<BCIR_ROPE_J ? half : BCIR_ROPE_J;
  for (pos=p0;pos<p1;pos++) {
    const float *c=j->table+pos*j->dim,*s0=c+half; float s[64];
    for (k=0;k<nk;k++) s[k]=s0[k]*(j->inverse ? -1.0f : 1.0f);
    for (r=0;r<j->b;r++) for (h=0;h<j->heads;h++) {
      float *lo=j->x+((r*j->t+pos)*j->heads+h)*j->dim,*hi=lo+half;
      for (k=0;k<nk;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
    }
  }
}
int main(int argc, char **argv) {
  static const char *names[5]={"first","fix","proto","half_k","min64"};
  bcir_tensor_task_fn fns[5]={first_task,fix_task,proto_task,half_k_task,min64_task};
  unsigned th; size_t b=4,t,h=4,dim=32,n,i; int calls,rounds,r,c,v; double *ms[5],t0,med[5];
  float *x0,*xs[5],*tab; void *pm=NULL; size_t ps; bcir_tensor_pool *pool;
  if (argc!=5) return 2;
  th=(unsigned)atoi(argv[1]); t=strtoul(argv[2],0,10); calls=atoi(argv[3]); rounds=atoi(argv[4]); n=b*t*h*dim;
  x0=malloc(n*4); tab=malloc(t*dim*4);
  for (v=0;v<5;v++) { xs[v]=malloc(n*4); ms[v]=malloc(sizeof(double)*rounds); }
  ps=bcir_tensor_pool_size(th); if (posix_memalign(&pm,64,(ps+63)/64*64)) return 3;
  pool=bcir_tensor_pool_init(pm,ps,th); if (!pool) return 3;
  for (i=0;i<n;i++) x0[i]=(float)((i*2654435761u)%2001)/1000.0f-1.0f;
  bcir_tensor_rope_table_pool(pool,t,dim,10000.0,tab);
  for (v=0;v<5;v++) {  /* the values: 8 chained calls each way from the same input, each form's the first's */
    lent_job j; j.b=b; j.t=t; j.heads=h; j.dim=dim; j.table=tab; j.x=xs[v];
    memcpy(xs[v],x0,n*4);
    for (c=0;c<8;c++) { j.inverse=c&1; bcir_tensor_pool_run(pool,fns[v],&j,(t+15)/16); }
    if (v && memcmp(xs[v],xs[0],n*4)) { printf("MISMATCH %s\n",names[v]); return 1; }
  }
  for (r=-1;r<rounds;r++) for (v=0;v<5;v++) {
    lent_job j; j.b=b; j.t=t; j.heads=h; j.dim=dim; j.table=tab; j.x=xs[v];
    t0=now(); for (c=0;c<calls;c++) { j.inverse=c&1; bcir_tensor_pool_run(pool,fns[v],&j,(t+15)/16); }
    if (r>=0) ms[v][r]=(now()-t0)*1e6/calls;
  }
  for (v=0;v<5;v++) { qsort(ms[v],(size_t)rounds,sizeof(double),cmpd); med[v]=ms[v][rounds/2]; }
  printf("{\"threads\": %u, \"t\": %zu, \"bytes_equal\": true",th,t);
  for (v=0;v<5;v++) printf(", \"%s_us\": %.2f",names[v],med[v]);
  printf("}\n");
  bcir_tensor_pool_stop(pool);
  return 0;
}
