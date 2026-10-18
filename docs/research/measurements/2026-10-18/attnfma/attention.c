/* ATTN-FMA (2026-10-18): the attention alone at the language model's shapes (batch 8, 8 query heads
 * over 4 KV heads, 48 channels; context TIME, 128 the model's), BCIR's exponential under both: one
 * layer's forward and in-place backward under v1's products and under the fused ones
 * (BCIR_TENSOR_ATTENTION_FUSED), the two contracts alternated round by round in one process, their
 * order swapped every round. As the trainer calls them: back to back on a pool kept busy (nothing
 * between the timed calls -- each contract has its own outputs, hashed only after the first warm-up
 * round and after the last round), the operands on one mapping advised for 2 MB transparent huge
 * pages (NDT-PAGES's placement of the trainer's state). Prints one JSON line: each contract's
 * median forward and backward over REPS rounds after two warm-up rounds, the ratios (v1 over
 * fused), whether each contract's outputs hashed alike in its first and last rounds, and whether
 * the two contracts' differ.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c attention.c runtime/c/bcir_tensor.c -lm
 *   ./a.out REPS THREADS [TIME]
 */
#define _DEFAULT_SOURCE
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static uint64_t fnv(uint64_t h, const float *x, size_t n) {
  const unsigned char *p=(const unsigned char *)x; size_t i;
  for (i=0;i<n*sizeof(float);i++) { h^=p[i]; h*=1099511628211ull; }
  return h;
}
static unsigned char *at;  /* the next free byte of the mapping */
static float *take(size_t n) { float *x=(float *)(void *)at; at+=(n*sizeof(float)+4095)/4096*4096; return x; }
int main(int argc, char **argv) {
  size_t b=8,t=argc>3 ? (size_t)atoi(argv[3]) : 128,nh=8,nk=4,d=48,i,total;
  int reps=argc>1 ? atoi(argv[1]) : 30; unsigned threads=argc>2 ? (unsigned)atoi(argv[2]) : 4;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t; int r,c;
  static const unsigned contract[2]={BCIR_TENSOR_ATTENTION_EXP,BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED};
  uint64_t first[2]={0,0},last[2]={0,0};
  float *q,*k,*v,*dy,*p[2],*y[2],*dq[2],*dk[2],*dv[2];
  double *tf[2],*tb[2];
  unsigned char *base;
  static _Alignas(64) unsigned char mem[1<<16]; bcir_tensor_pool *pool=NULL;
  if (reps<1 || t<1) return 2;
  total=(2*nq+2*nkv+2*(np+nq+nq+2*nkv))*sizeof(float)+16*4096;
  base=mmap(NULL,total+(2u<<20),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if (base==MAP_FAILED) return 2;
  at=(unsigned char *)(((uintptr_t)base+(2u<<20)-1)&~(uintptr_t)((2u<<20)-1));
  (void)madvise(at,total,MADV_HUGEPAGE);  /* advice: where the host has no THP the pages stay 4 KB */
  q=take(nq); k=take(nkv); v=take(nkv); dy=take(nq);
  for (c=0;c<2;c++) { p[c]=take(np); y[c]=take(nq); dq[c]=take(nq); dk[c]=take(nkv); dv[c]=take(nkv); }
  for (c=0;c<2;c++) { tf[c]=malloc(sizeof(double)*(size_t)reps); tb[c]=malloc(sizeof(double)*(size_t)reps); }
  if (!tf[0] || !tb[0] || !tf[1] || !tb[1]) return 2;
  for (i=0;i<nq;i++) { q[i]=0.3f*sinf((float)i*0.37f); dy[i]=0.01f*cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { k[i]=0.3f*cosf((float)i*0.53f); v[i]=0.5f*sinf((float)i*0.29f); }
  if (threads>1) pool=bcir_tensor_pool_init(mem,sizeof(mem),threads);
  if (threads>1 && !pool) return 2;
  for (r=-2;r<reps;r++) {
    int j;
    for (j=0;j<2;j++) {
      double a0,a1,a2;
      c=(r&1) ? 1-j : j;
      a0=now();
      bcir_tensor_attention_contract_scan_pool(pool,contract[c],b,t,nh,nk,d,q,k,v,p[c],y[c],NULL); a1=now();
      bcir_tensor_attention_backward_contract_pool(pool,contract[c],b,t,nh,nk,d,q,k,v,p[c],dy,dq[c],dk[c],dv[c]);
      a2=now();
      if (r>=0) { tf[c][r]=1e3*(a1-a0); tb[c][r]=1e3*(a2-a1); }
    }
    if (r==-2 || r==reps-1) for (c=0;c<2;c++) {
      uint64_t h=fnv(fnv(fnv(fnv(1469598103934665603ull,y[c],nq),dq[c],nq),dk[c],nkv),dv[c],nkv);
      if (r==-2) first[c]=h; else last[c]=h;
    }
  }
  for (c=0;c<2;c++) { qsort(tf[c],(size_t)reps,sizeof(double),cmp); qsort(tb[c],(size_t)reps,sizeof(double),cmp); }
  printf("{\"time\": %zu, \"threads\": %u, \"reps\": %d, \"v1\": {\"forward_ms\": %.4f, \"backward_ms\": %.4f}, "
      "\"fused\": {\"forward_ms\": %.4f, \"backward_ms\": %.4f}, \"forward_ratio\": %.4f, \"backward_ratio\": %.4f, "
      "\"layer_ratio\": %.4f, \"each_contract_one_result\": %s, \"contracts_apart\": %s}\n",
      t,threads,reps,tf[0][reps/2],tb[0][reps/2],tf[1][reps/2],tb[1][reps/2],
      tf[0][reps/2]/tf[1][reps/2],tb[0][reps/2]/tb[1][reps/2],
      (tf[0][reps/2]+tb[0][reps/2])/(tf[1][reps/2]+tb[1][reps/2]),
      first[0]==last[0] && first[1]==last[1] ? "true" : "false",first[0]!=first[1] ? "true" : "false");
  bcir_tensor_pool_stop(pool); return 0;
}
