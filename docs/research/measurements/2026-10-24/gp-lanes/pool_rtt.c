/* The tensor pool's cost a job: JOBS jobs of TASKS trivial tasks back to back (each task writes one
 * byte of its own line), then the same with a GAP_US pause before each job; median of ROUNDS rounds
 * of the mean job, in microseconds. usage: pool_rtt THREADS TASKS JOBS ROUNDS GAP_US */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmpd(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
static void spin_us(double us) { double t0=now(); while ((now()-t0)*1e6<us) {} }
static unsigned char lines[4096*64];
static void touch(void *ctx,size_t task) { (void)ctx; lines[(task&4095)*64]++; }
int main(int argc,char **argv) {
  if (argc!=6) return 2;
  unsigned th=(unsigned)atoi(argv[1]); size_t tasks=strtoul(argv[2],0,10); int jobs=atoi(argv[3]),rounds=atoi(argv[4]),r,c;
  double gap=atof(argv[5]),*b2b=malloc(sizeof(double)*rounds),*after=malloc(sizeof(double)*rounds);
  size_t ps=bcir_tensor_pool_size(th); void *pm=NULL; if (posix_memalign(&pm,64,(ps+63)/64*64)) return 3;
  bcir_tensor_pool *pool=bcir_tensor_pool_init(pm,ps,th);
  for (r=-1;r<rounds;r++) {
    double t0=now(),sum=0.0;
    for (c=0;c<jobs;c++) bcir_tensor_pool_run(pool,touch,NULL,tasks);
    if (r>=0) b2b[r]=(now()-t0)*1e6/jobs;
    for (c=0;c<jobs;c++) { spin_us(gap); double t1=now(); bcir_tensor_pool_run(pool,touch,NULL,tasks); sum+=now()-t1; }
    if (r>=0) after[r]=sum*1e6/jobs;
  }
  qsort(b2b,(size_t)rounds,sizeof(double),cmpd); qsort(after,(size_t)rounds,sizeof(double),cmpd);
  printf("{\"threads\": %u, \"tasks\": %zu, \"back_to_back_us\": %.2f, \"gap_us\": %.0f, \"after_gap_us\": %.2f}\n",
         th,tasks,b2b[rounds/2],gap,after[rounds/2]);
  bcir_tensor_pool_stop(pool);
  return 0;
}
