/* NDT-SCAN-W's cost in the writers: each scan entry against its plain entry, in turn, 60 calls of
 * each at the language model's shapes on a pool of four -- the attention (v2 softmax: batch 8,
 * context 128, 8 query heads over 4 KV heads, 48 channels), the SwiGLU product (v2) over 1,024 x 1,024
 * and the RMSNorm over 1,024 x 384 -- the medians, and their difference times the calls a step
 * makes (6 attentions, 6 SwiGLU products, 13 RMSNorms).
 *
 *   cc -std=c2x -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c scan_overhead.c runtime/c/bcir_tensor.c -lm
 */
#define _POSIX_C_SOURCE 199309L
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }
int main(void) {
  size_t b=8,t=128,nh=8,nk=4,d=48,i,r,nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t; int reps=60;
  float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*p=malloc(np*4),*y=malloc(nq*4);
  double plain[60],scan[60],sil0[60],sil1[60],rms0[60],rms1[60];
  static _Alignas(64) unsigned char mem[65536]; bcir_tensor_pool *pool=bcir_tensor_pool_init(mem,sizeof mem,4);
  size_t nf=1024*1024; float *x=malloc(nf*4),*u=malloc(nf*4),*h=malloc(nf*4),*g=malloc(384*4);
  for (i=0;i<nq;i++) q[i]=sinf(i*0.37f);
  for (i=0;i<nkv;i++) { k[i]=cosf(i*0.53f); v[i]=sinf(i*0.29f); }
  for (i=0;i<nf;i++) { x[i]=sinf(i*0.11f)*3; u[i]=cosf(i*0.07f); }
  for (i=0;i<384;i++) g[i]=1.0f;
  for (r=0;r<(size_t)reps;r++) {
    int bad=0; double t0;
    t0=now(); bcir_tensor_attention_v2_pool(pool,b,t,nh,nk,d,q,k,v,p,y); plain[r]=now()-t0;
    t0=now(); bcir_tensor_attention_v2_scan_pool(pool,b,t,nh,nk,d,q,k,v,p,y,&bad); scan[r]=now()-t0;
    t0=now(); bcir_tensor_silu_mul_v2_pool(pool,nf,x,u,h); sil0[r]=now()-t0;
    t0=now(); bcir_tensor_silu_mul_v2_scan_pool(pool,nf,x,u,h,&bad); sil1[r]=now()-t0;
    t0=now(); bcir_tensor_rms_pool(pool,1024,384,1e-5f,x,g,h); rms0[r]=now()-t0;
    t0=now(); bcir_tensor_rms_scan_pool(pool,1024,384,1e-5f,x,g,h,&bad); rms1[r]=now()-t0;
    if (bad) { puts("unexpected verdict"); return 1; }
  }
  qsort(plain,reps,8,cmp); qsort(scan,reps,8,cmp); qsort(sil0,reps,8,cmp); qsort(sil1,reps,8,cmp); qsort(rms0,reps,8,cmp); qsort(rms1,reps,8,cmp);
  printf("attention v2 plain %.3f scan %.3f ms (medians; x6 layers: +%.3f ms/step)\n",plain[reps/2],scan[reps/2],6*(scan[reps/2]-plain[reps/2]));
  printf("silu_mul v2 plain %.3f scan %.3f ms (x6: +%.3f)\n",sil0[reps/2],sil1[reps/2],6*(sil1[reps/2]-sil0[reps/2]));
  printf("rms plain %.3f scan %.3f ms (x13: +%.3f)\n",rms0[reps/2],rms1[reps/2],13*(rms1[reps/2]-rms0[reps/2]));
  bcir_tensor_pool_stop(pool); return 0;
}
