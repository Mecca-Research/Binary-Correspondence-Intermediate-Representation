/* The attention alone at the language model's shapes (batch 8, context 128, 8 query heads over 4 KV
 * heads, 48 channels), the v2 softmax: the forward and the in-place backward of one layer, timed per
 * call; prints one JSON line of the medians of REPS calls after two warm-up calls.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I TREE attention_time.c TREE/bcir_tensor.c -lm
 *   ./a.out REPS THREADS
 */
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
int main(int argc, char **argv) {
  size_t b=8,t=128,nh=8,nk=4,d=48,i; int reps=argc>1 ? atoi(argv[1]) : 30; unsigned threads=argc>2 ? (unsigned)atoi(argv[2]) : 4;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t; int r;
  float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*p=malloc(np*4),*y=malloc(nq*4),*dy=malloc(nq*4),
        *dq=malloc(nq*4),*dk=malloc(nkv*4),*dv=malloc(nkv*4);
  double *tf=malloc(sizeof(double)*(size_t)reps),*tb=malloc(sizeof(double)*(size_t)reps);
  static _Alignas(64) unsigned char mem[1<<16]; bcir_tensor_pool *pool=NULL;
  if (!q || !k || !v || !p || !y || !dy || !dq || !dk || !dv || !tf || !tb || reps<1) return 2;
  for (i=0;i<nq;i++) { q[i]=0.3f*sinf((float)i*0.37f); dy[i]=0.01f*cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { k[i]=0.3f*cosf((float)i*0.53f); v[i]=0.5f*sinf((float)i*0.29f); }
  if (threads>1) pool=bcir_tensor_pool_init(mem,sizeof(mem),threads);
  for (r=-2;r<reps;r++) {
    double a0=now(),a1,a2;
    bcir_tensor_attention_v2_pool(pool,b,t,nh,nk,d,q,k,v,p,y); a1=now();
    bcir_tensor_attention_backward_inplace_pool(pool,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv); a2=now();
    if (r>=0) { tf[r]=1e3*(a1-a0); tb[r]=1e3*(a2-a1); }
  }
  qsort(tf,(size_t)reps,sizeof(double),cmp); qsort(tb,(size_t)reps,sizeof(double),cmp);
  printf("{\"threads\": %u, \"reps\": %d, \"forward_ms\": %.4f, \"backward_ms\": %.4f}\n",threads,reps,tf[reps/2],tb[reps/2]);
  bcir_tensor_pool_stop(pool); return 0;
}
