/* ROPE-TABLE: the tensor unit's rotation evaluating its cosines and sines every call (bcir_tensor_rope_pool)
 * against the rotation reading a table of them (bcir_tensor_rope_lent_pool) filled once
 * (bcir_tensor_rope_table_pool) -- every value checked byte for byte first, forward and inverse, on the
 * committed entries -- on a pool of THREADS, B batches of T positions, H heads of width DIM: each the median
 * microseconds of ROUNDS rounds of CALLS calls, forward and inverse alternating, and the fill's own.
 * usage: rope_lent_bench THREADS B T H DIM CALLS ROUNDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
int main(int argc,char **argv) {
  unsigned th; size_t b,t,h,dim,n,i; int calls,rounds,r,c,inv; double base=10000.0,*ta,*tb,*tf,t0;
  float *x0,*xa,*xb,*tab; void *pm=NULL; size_t ps; bcir_tensor_pool *pool;
  if (argc!=8) return 2;
  th=(unsigned)atoi(argv[1]); b=strtoul(argv[2],0,10); t=strtoul(argv[3],0,10); h=strtoul(argv[4],0,10);
  dim=strtoul(argv[5],0,10); calls=atoi(argv[6]); rounds=atoi(argv[7]); n=b*t*h*dim;
  ta=malloc(sizeof(double)*rounds); tb=malloc(sizeof(double)*rounds); tf=malloc(sizeof(double)*rounds);
  x0=malloc(n*4); xa=malloc(n*4); xb=malloc(n*4); tab=malloc(t*dim*4);
  ps=bcir_tensor_pool_size(th); if (posix_memalign(&pm,64,(ps+63)/64*64)) return 3;
  pool=bcir_tensor_pool_init(pm,ps,th); if (!pool) return 3;
  for (i=0;i<n;i++) x0[i]=(float)((i*2654435761u)%2001)/1000.0f-1.0f;
  bcir_tensor_rope_table_pool(pool,t,dim,base,tab);
  for (inv=0;inv<2;inv++) {  /* the values: one call each way from the same input, then 7 more chained */
    memcpy(xa,x0,n*4); memcpy(xb,x0,n*4);
    for (c=0;c<8;c++) { bcir_tensor_rope_pool(pool,b,t,h,dim,base,xa,inv); bcir_tensor_rope_lent_pool(pool,b,t,h,dim,tab,xb,inv); }
    if (memcmp(xa,xb,n*4)) { printf("MISMATCH inverse=%d\n",inv); return 1; }
  }
  for (r=-1;r<rounds;r++) {
    t0=now(); for (c=0;c<calls;c++) bcir_tensor_rope_pool(pool,b,t,h,dim,base,xa,c&1); if (r>=0) ta[r]=(now()-t0)*1e6/calls;
    t0=now(); for (c=0;c<calls;c++) bcir_tensor_rope_lent_pool(pool,b,t,h,dim,tab,xb,c&1); if (r>=0) tb[r]=(now()-t0)*1e6/calls;
    t0=now(); for (c=0;c<calls;c++) bcir_tensor_rope_table_pool(pool,t,dim,base,tab); if (r>=0) tf[r]=(now()-t0)*1e6/calls;
  }
  qsort(ta,(size_t)rounds,sizeof(double),cmpd); qsort(tb,(size_t)rounds,sizeof(double),cmpd); qsort(tf,(size_t)rounds,sizeof(double),cmpd);
  printf("{\"threads\": %u, \"b\": %zu, \"t\": %zu, \"heads\": %zu, \"dim\": %zu, \"bytes_equal\": true, \"unit_us\": %.2f, "
         "\"lent_us\": %.2f, \"lent_over_unit\": %.3f, \"fill_us\": %.2f}\n",th,b,t,h,dim,ta[rounds/2],tb[rounds/2],
         tb[rounds/2]/ta[rounds/2],tf[rounds/2]);
  bcir_tensor_pool_stop(pool);
  return 0;
}
