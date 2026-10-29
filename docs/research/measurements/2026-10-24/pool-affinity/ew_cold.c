/* An elementwise job of the step at one shape, warm (back to back) and cold (a 256 MiB read before each call, every
 * operand out of the caches), and "dirty" (each operand written by a pool job on the other threads just before:
 * a row-chunked fill whose tasks the workers claim, as the step's previous job left them); median microseconds.
 * usage: ew_cold THREADS ROWS WIDTH ROUNDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n)) exit(3); memset(p,0,n); return p; }
static int cmpd(const void *a, const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y ? -1 : x>y; }
static volatile float sink;
static void evict(const float *e, size_t n) { float s=0; size_t i; for (i=0;i<n;i+=16) s+=e[i]; sink=s; }
typedef struct { float *a; size_t n; float v; } fill_job;
static void fill_task(void *ctx, size_t t) { fill_job *f=ctx; size_t i0=t*4096,i1=i0+4096<f->n ? i0+4096 : f->n,i; for (i=i0;i<i1;i++) f->a[i]=f->v+(float)(i%97)*1e-3f; }
int main(int argc, char **argv) {
  if (argc!=5) return 2;
  unsigned th=(unsigned)atoi(argv[1]); size_t rows=strtoul(argv[2],0,10),w=strtoul(argv[3],0,10),n=rows*w,i,ne=(size_t)1<<26;
  int rounds=atoi(argv[4]),r,k,mode; double *t=malloc(sizeof(double)*rounds),res[3][3];
  size_t ps=bcir_tensor_pool_size(th); void *pm=alloc64((ps+63)/64*64);
  bcir_tensor_pool *pool=th>1 ? bcir_tensor_pool_init(pm,ps,th) : NULL;
  float *x=alloc64(n*4),*dy=alloc64(n*4),*dx=alloc64(n*4),*y=alloc64(n*4),*g=alloc64(w*4),*dg=alloc64(w*4),*inv=alloc64(rows*4),*e=alloc64(ne*4);
  int bad=0;
  for (i=0;i<n;i++) { x[i]=sinf((float)i*0.37f); dy[i]=cosf((float)i*0.11f); }
  for (i=0;i<w;i++) g[i]=1.0f+0.01f*(float)i;
  static const char *mn[3]={"warm","cold","dirty"},*kn[3]={"rms_scan","rms_vjp_add_first","add"};
  for (k=0;k<3;k++) for (mode=0;mode<3;mode++) {
    for (r=-1;r<rounds;r++) {
      double t0;
      if (mode==1) evict(e,ne);
      if (mode==2) { fill_job f1={x,n,0.5f},f2={dy,n,0.25f}; bcir_tensor_pool_run(pool,fill_task,&f1,(n+4095)/4096); bcir_tensor_pool_run(pool,fill_task,&f2,(n+4095)/4096); }
      t0=now();
      if (k==0) bcir_tensor_rms_scan_pool(pool,rows,w,1e-5f,x,g,y,&bad);
      else if (k==1) bcir_tensor_rms_backward_into_pool(pool,rows,w,1e-5f,x,g,dy,dx,dg,inv,BCIR_TENSOR_RMSB_ADD|BCIR_TENSOR_RMSB_FIRST);
      else bcir_tensor_add_pool(pool,n,x,dy);
      if (r>=0) t[r]=(now()-t0)*1e6;
    }
    qsort(t,(size_t)rounds,sizeof(double),cmpd); res[k][mode]=t[rounds/2];
  }
  printf("threads %u rows %zu width %zu:",th,rows,w);
  for (k=0;k<3;k++) { printf("  %s",kn[k]); for (mode=0;mode<3;mode++) printf(" %s %.1f",mn[mode],res[k][mode]); }
  printf(" us\n");
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
