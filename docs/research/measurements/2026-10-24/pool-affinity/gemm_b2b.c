/* One pool, one k-blocked GEMM shape, CALLS calls back to back (the step's jobs follow one another without a gap),
 * then the same with a GAP_US microsecond pause before each call (a serial stretch between jobs); median of ROUNDS
 * rounds of the mean call. usage: gemm_b2b THREADS TA TB M N K CALLS ROUNDS GAP_US */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n)) exit(3); memset(p,0,n); return p; }
static int cmpd(const void *a, const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
static void spin_us(double us) { double t0=now(); while ((now()-t0)*1e6<us) {} }
int main(int argc, char **argv) {
  if (argc!=10) return 2;
  unsigned th=(unsigned)atoi(argv[1]); int ta=atoi(argv[2]),tb=atoi(argv[3]);
  size_t m=strtoul(argv[4],0,10),n=strtoul(argv[5],0,10),k=strtoul(argv[6],0,10),i;
  int calls=atoi(argv[7]),rounds=atoi(argv[8]),r,c; double gap=atof(argv[9]),*t=malloc(sizeof(double)*rounds),*tg=malloc(sizeof(double)*rounds);
  float *a=alloc64(m*k*4),*b=alloc64(k*n*4),*cc=alloc64(m*n*4);
  for (i=0;i<m*k;i++) a[i]=(float)(i%13)*0.1f; for (i=0;i<k*n;i++) b[i]=(float)(i%7)*0.2f;
  size_t ps=bcir_tensor_pool_size(th); void *pm=alloc64((ps+63)/64*64);
  bcir_tensor_pool *pool=th>1 ? bcir_tensor_pool_init(pm,ps,th) : NULL;
  for (r=-1;r<rounds;r++) {
    double t0=now();
    for (c=0;c<calls;c++) bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,a,b,0.0f,cc);
    if (r>=0) t[r]=(now()-t0)*1e3/calls;
    double sum=0;
    for (c=0;c<calls;c++) { spin_us(gap); double t1=now(); bcir_tensor_mm_kblock_pool(pool,ta,tb,m,n,k,1.0f,a,b,0.0f,cc); sum+=now()-t1; }
    if (r>=0) tg[r]=sum*1e3/calls;
  }
  qsort(t,(size_t)rounds,sizeof(double),cmpd); qsort(tg,(size_t)rounds,sizeof(double),cmpd);
  double fl=2.0*m*n*k;
  printf("threads %u  %d,%d,%zu,%zu,%zu  back-to-back %.4f ms (%.1f GF/s)  after a %.0f us gap %.4f ms (%.1f GF/s)\n",
         th,ta,tb,m,n,k,t[rounds/2],fl/(t[rounds/2]*1e-3)/1e9,gap,tg[rounds/2],fl/(tg[rounds/2]*1e-3)/1e9);
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
