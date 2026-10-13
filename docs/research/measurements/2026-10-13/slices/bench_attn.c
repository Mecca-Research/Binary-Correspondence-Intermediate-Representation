/* The attention kernels alone on the 9.84M decoder's shape (batch 8, T 128, 8 heads over 4 KV heads,
 * 48 channels): the forward and the in-place backward (p restored from a copy before each call, the
 * copy timed apart and subtracted), at the threads given; best and median of R calls each. */
#define _POSIX_C_SOURCE 199309L
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static float rnd(unsigned *s) { *s=*s*1664525u+1013904223u; return ((float)(*s>>8)/16777216.0f-0.5f); }
int main(int argc, char **argv) {
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 1; int reps=argc>2 ? atoi(argv[2]) : 21;
  size_t B=8,T=128,H=8,K=4,D=48,i; unsigned seed=2026;
  size_t nq=B*T*H*D,nk=B*T*K*D,np=B*H*T*T;
  float *q=malloc(nq*4),*k=malloc(nk*4),*v=malloc(nk*4),*p=malloc(np*4),*p0=malloc(np*4),*y=malloc(nq*4),
      *dy=malloc(nq*4),*dq=malloc(nq*4),*dk=malloc(nk*4),*dv=malloc(nk*4);
  double *tf=malloc(reps*sizeof(double)),*tb=malloc(reps*sizeof(double)),*tc=malloc(reps*sizeof(double));
  static _Alignas(64) unsigned char memory[1<<16]; bcir_tensor_pool *pool=NULL; int r;
  if (threads>1) { pool=bcir_tensor_pool_init(memory,sizeof memory,threads); if (!pool) return 1; }
  for (i=0;i<nq;i++) { q[i]=rnd(&seed); dy[i]=rnd(&seed); }
  for (i=0;i<nk;i++) { k[i]=rnd(&seed); v[i]=rnd(&seed); }
  bcir_tensor_attention_pool(pool,B,T,H,K,D,q,k,v,p0,y);
  for (r=0;r<reps;r++) {
    double t0=now_ns(); bcir_tensor_attention_pool(pool,B,T,H,K,D,q,k,v,p,y); tf[r]=now_ns()-t0;
    t0=now_ns(); memcpy(p,p0,np*4); tc[r]=now_ns()-t0;
    t0=now_ns(); memcpy(p,p0,np*4);
    bcir_tensor_attention_backward_inplace_pool(pool,B,T,H,K,D,q,k,v,p,dy,dq,dk,dv); tb[r]=now_ns()-t0-tc[r];
  }
  qsort(tf,reps,sizeof(double),cmp); qsort(tb,reps,sizeof(double),cmp);
  printf("threads %u forward best %.2f median %.2f ms backward best %.2f median %.2f ms (checksum %.6g)\n",threads,
      tf[0]/1e6,tf[reps/2]/1e6,tb[0]/1e6,tb[reps/2]/1e6,(double)(dq[7]+dk[11]+dv[13]+y[5]));
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
