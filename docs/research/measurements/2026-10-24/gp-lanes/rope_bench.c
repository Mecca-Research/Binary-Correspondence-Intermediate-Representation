/* ROPE-TABLE-0: the tensor unit's RoPE (bcir_tensor_rope_pool: each call takes every position's
 * cosines and sines from libm) against the same rotation reading a table of them computed once with
 * the same calls -- every value checked byte for byte, forward and inverse -- on a pool of THREADS,
 * B batches of T positions, H heads of width DIM; median of ROUNDS rounds of CALLS calls.
 * usage: rope_bench THREADS B T H DIM CALLS ROUNDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
/* the table: pos-major, a position's half cosines then its half sines, each the unit's expression */
static void table(size_t t,size_t dim,double base,float *cs) {
  size_t pos,j,half=dim/2;
  for (pos=0;pos<t;pos++) for (j=0;j<half;j++) {
    double f=pow(base,-2.0*(double)j/(double)dim),angle=(double)pos*f;
    cs[pos*dim+j]=(float)cos(angle); cs[pos*dim+half+j]=(float)sin(angle);
  }
}
typedef struct { size_t b,t,heads,dim; const float *cs; float *x; int inverse; } lent_job;
static void lent_task(void *ctx,size_t task) {
  lent_job *j=(lent_job *)ctx; size_t half=j->dim/2,p0=task*16,p1=p0+16<j->t ? p0+16 : j->t,pos,r,h,k;
  for (pos=p0;pos<p1;pos++) {
    const float *c=j->cs+pos*j->dim,*s0=c+half; float s[64];
    for (k=0;k<half;k++) s[k]=s0[k]*(j->inverse ? -1.0f : 1.0f);
    for (r=0;r<j->b;r++) for (h=0;h<j->heads;h++) {
      float *lo=j->x+((r*j->t+pos)*j->heads+h)*j->dim,*hi=lo+half;
      for (k=0;k<half;k++) { float a=lo[k],z=hi[k]; lo[k]=a*c[k]-z*s[k]; hi[k]=z*c[k]+a*s[k]; }
    }
  }
}
int main(int argc,char **argv) {
  if (argc!=8) return 2;
  unsigned th=(unsigned)atoi(argv[1]); size_t b=strtoul(argv[2],0,10),t=strtoul(argv[3],0,10),h=strtoul(argv[4],0,10),dim=strtoul(argv[5],0,10),n=b*t*h*dim,i;
  int calls=atoi(argv[6]),rounds=atoi(argv[7]),r,c,inv; double base=10000.0,*ta=malloc(sizeof(double)*rounds),*tb=malloc(sizeof(double)*rounds),t0;
  float *x0=malloc(n*4),*xa=malloc(n*4),*xb=malloc(n*4),*cs=malloc(t*dim*4);
  size_t ps=bcir_tensor_pool_size(th); void *pm=NULL; if (posix_memalign(&pm,64,(ps+63)/64*64)) return 3;
  bcir_tensor_pool *pool=bcir_tensor_pool_init(pm,ps,th);
  for (i=0;i<n;i++) x0[i]=(float)((i*2654435761u)%2001)/1000.0f-1.0f;
  table(t,dim,base,cs);
  for (inv=0;inv<2;inv++) {  /* the values: one call each way from the same input, then 7 more chained */
    lent_job j={b,t,h,dim,cs,xb,inv};
    memcpy(xa,x0,n*4); memcpy(xb,x0,n*4);
    for (c=0;c<8;c++) { bcir_tensor_rope_pool(pool,b,t,h,dim,base,xa,inv); bcir_tensor_pool_run(pool,lent_task,&j,(t+15)/16); }
    if (memcmp(xa,xb,n*4)) { printf("MISMATCH inverse=%d\n",inv); return 1; }
  }
  for (r=-1;r<rounds;r++) {
    lent_job j={b,t,h,dim,cs,xb,0};
    t0=now(); for (c=0;c<calls;c++) bcir_tensor_rope_pool(pool,b,t,h,dim,base,xa,c&1); if (r>=0) ta[r]=(now()-t0)*1e6/calls;
    t0=now(); for (c=0;c<calls;c++) { j.inverse=c&1; bcir_tensor_pool_run(pool,lent_task,&j,(t+15)/16); } if (r>=0) tb[r]=(now()-t0)*1e6/calls;
  }
  qsort(ta,(size_t)rounds,sizeof(double),cmpd); qsort(tb,(size_t)rounds,sizeof(double),cmpd);
  t0=now(); table(t,dim,base,cs); double tt=(now()-t0)*1e6;
  printf("{\"threads\": %u, \"b\": %zu, \"t\": %zu, \"heads\": %zu, \"dim\": %zu, \"bytes_equal\": true, \"unit_us\": %.2f, \"lent_us\": %.2f, "
         "\"lent_over_unit\": %.3f, \"table_once_us\": %.1f}\n",th,b,t,h,dim,ta[rounds/2],tb[rounds/2],tb[rounds/2]/ta[rounds/2],tt);
  bcir_tensor_pool_stop(pool);
  return 0;
}
