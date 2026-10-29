/* The pool's own cost a job: JOBS back-to-back pool_run calls of TASKS empty tasks (and of tasks that spin WORK_NS
 * nanoseconds each), median of ROUNDS rounds, microseconds a job; plus the elementwise jobs at one shape (RMSNorm,
 * its VJP, SwiGLU's VJP, RoPE) back to back on the same pool. usage: pool_cost THREADS TASKS JOBS ROUNDS WORK_NS ROWS WIDTH */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n)) exit(3); memset(p,0,n); return p; }
static int cmpd(const void *a, const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
static double work_ns;
static void empty_task(void *ctx, size_t t) { (void)ctx; (void)t; }
static void spin_task(void *ctx, size_t t) { (void)ctx; (void)t; double t0=now(); while ((now()-t0)*1e9<work_ns) {} }
int main(int argc, char **argv) {
  if (argc!=8) return 2;
  unsigned th=(unsigned)atoi(argv[1]); size_t tasks=strtoul(argv[2],0,10); int jobs=atoi(argv[3]),rounds=atoi(argv[4]),r,j,k;
  size_t rows=strtoul(argv[6],0,10),w=strtoul(argv[7],0,10),i,n=rows*w; work_ns=atof(argv[5]);
  size_t ps=bcir_tensor_pool_size(th); void *pm=alloc64((ps+63)/64*64);
  bcir_tensor_pool *pool=th>1 ? bcir_tensor_pool_init(pm,ps,th) : NULL;
  double *t=malloc(sizeof(double)*rounds*8);
  float *x=alloc64(n*4),*dy=alloc64(n*4),*dx=alloc64(n*4),*y=alloc64(n*4),*g=alloc64(w*4),*dg=alloc64(w*4),*inv=alloc64(rows*4),*u=alloc64(n*4),*du=alloc64(n*4);
  for (i=0;i<n;i++) { x[i]=sinf((float)i*0.37f); dy[i]=cosf((float)i*0.11f); u[i]=sinf((float)i*0.13f); }
  for (i=0;i<w;i++) g[i]=1.0f+0.01f*(float)i;
  static const char *nm[6]={"empty","spin","rms","rms_vjp","swiglu_vjp","rope"};
  double res[6];
  for (k=0;k<6;k++) {
    for (r=-1;r<rounds;r++) {
      double t0=now();
      for (j=0;j<jobs;j++) {
        if (k==0) bcir_tensor_pool_run(pool,empty_task,NULL,tasks);
        else if (k==1) bcir_tensor_pool_run(pool,spin_task,NULL,tasks);
        else if (k==2) bcir_tensor_rms_pool(pool,rows,w,1e-5f,x,g,y);
        else if (k==3) bcir_tensor_rms_backward_into_pool(pool,rows,w,1e-5f,x,g,dy,dx,dg,inv,0u);
        else if (k==4) bcir_tensor_swiglu_backward_pool(pool,n,x,u,dy,dx,du);
        else bcir_tensor_rope_pool(pool,rows/512>0 ? rows/512 : 1,512,w/32,32,10000.0,y,0);
      }
      if (r>=0) t[r]=(now()-t0)*1e6/jobs;
    }
    qsort(t,(size_t)rounds,sizeof(double),cmpd); res[k]=t[rounds/2];
  }
  printf("threads %u tasks %zu: ",th,tasks);
  for (k=0;k<6;k++) printf("%s %.2f us  ",nm[k],res[k]);
  printf("(spin task %.0f ns; rows %zu width %zu)\n",work_ns,rows,w);
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
