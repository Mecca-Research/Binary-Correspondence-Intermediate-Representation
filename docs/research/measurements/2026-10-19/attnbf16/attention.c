/* BF16-ATTN (2026-10-19): one layer's attention of the 9.84M decoder (8 x 128 tokens, 8 query heads
 * over 4 KV heads of 48 channels) alone, forward and backward, under the fused binary32 contract
 * (BCIR_TENSOR_ATTENTION_FUSED with the BCIR exponential: bf16+exp+reduce+attn's attention) and under
 * the bf16 contract (bcir_tensor_attention_bf16_*, the widest instance the host admits, with the
 * exponential: bf16+exp+reduce+attnbf16's), on one pool of THREADS threads. Each round runs both
 * arms, forward then backward, in an order that alternates round by round, so the host's drift
 * reaches them alike; the first two rounds are warm-up. Prints one JSON line: each arm's median
 * forward, backward and layer (the two summed) in ms, and per measure the round-by-round ratios
 * (fused over bf16: above 1 means bf16 is faster) as their median, a 95% percentile-bootstrap
 * interval of the median (2000 resamples, a fixed seed) and the rounds bf16 won. Wall time:
 * indicative, never a gate.
 *
 *   cc -std=c11 -O3 -ffp-contract=off -fno-math-errno -pthread -I runtime/c attention.c \
 *      runtime/c/bcir_tensor.c -lm -o attention && ./attention THREADS ROUNDS
 */
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 512
static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}
static int cmpd(const void *a,const void *b) {
  double x=*(const double *)a,y=*(const double *)b; return x<y ? -1 : x>y;
}
static double median(const double *x,int n) {
  double s[MAXR]; memcpy(s,x,(size_t)n*sizeof(double)); qsort(s,(size_t)n,sizeof(double),cmpd);
  return n%2 ? s[n/2] : 0.5*(s[n/2-1]+s[n/2]);
}
static uint64_t rng=0x9e3779b97f4a7c15ull;
static uint64_t next(void) { rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; }
/* the median of x and a 95% percentile-bootstrap interval of it */
static void boot(const double *x,int n,double *mid,double *lo,double *hi) {
  static double meds[2000]; double r[MAXR]; int i,j;
  for (i=0;i<2000;i++) { for (j=0;j<n;j++) r[j]=x[next()%(uint64_t)n]; meds[i]=median(r,n); }
  qsort(meds,2000,sizeof(double),cmpd);
  *mid=median(x,n); *lo=meds[49]; *hi=meds[1949];
}
int main(int argc,char **argv) {
  size_t b=8,t=128,nh=8,nk=4,d=48,nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i,ws;
  unsigned th=argc>1 ? (unsigned)atoi(argv[1]) : 4; int rounds=argc>2 ? atoi(argv[2]) : 31,r,arm,bad,inst=0;
  float *q,*k,*v,*dy,*p,*y,*dq,*dk,*dv; unsigned char *w; bcir_tensor_pool *pool=NULL;
  static _Alignas(64) unsigned char pm[65536];
  static double f[2][MAXR],g[2][MAXR],rf[MAXR],rg[MAXR],rl[MAXR];
  const char *names[3]={"forward","backward","layer"}; double *ratios[3]={rf,rg,rl};
  if (rounds<3 || rounds>MAXR || th<1) { fprintf(stderr,"usage: attention THREADS ROUNDS (3..%d)\n",MAXR); return 2; }
  q=malloc(nq*4); k=malloc(nkv*4); v=malloc(nkv*4); dy=malloc(nq*4); p=malloc(np*4); y=malloc(nq*4);
  dq=malloc(nq*4); dk=malloc(nkv*4); dv=malloc(nkv*4);
  ws=bcir_tensor_attention_bf16_workspace(b,t,nh,nk,d); w=aligned_alloc(64,(ws+63)/64*64);
  if (!q || !k || !v || !dy || !p || !y || !dq || !dk || !dv || !w) { fprintf(stderr,"out of memory\n"); return 2; }
  if (th>1) {
    if (!bcir_tensor_pool_size(th) || bcir_tensor_pool_size(th)>sizeof(pm)) { fprintf(stderr,"no pool of %u\n",th); return 2; }
    pool=bcir_tensor_pool_init(pm,sizeof(pm),th);
  }
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
  for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=0;r<rounds+2;r++) for (arm=0;arm<2;arm++) {
    int a=(r&1) ? 1-arm : arm; double t0,t1,t2;
    t0=now();
    if (a==0) bcir_tensor_attention_contract_scan_pool(pool,BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED,b,t,
        nh,nk,d,q,k,v,p,y,&bad);
    else inst=bcir_tensor_attention_bf16_scan_pool(pool,0,BCIR_TENSOR_ATTENTION_EXP,b,t,nh,nk,d,q,k,v,p,y,&bad,w,ws);
    t1=now();
    if (a==0) bcir_tensor_attention_backward_contract_pool(pool,BCIR_TENSOR_ATTENTION_FUSED,b,t,nh,nk,d,q,k,v,p,dy,
        dq,dk,dv);
    else (void)bcir_tensor_attention_bf16_backward_pool(pool,0,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv,w,ws);
    t2=now();
    if (r>=2) { f[a][r-2]=(t1-t0)*1e3; g[a][r-2]=(t2-t1)*1e3; }
  }
  if (!inst) { fprintf(stderr,"the bf16 attention refused\n"); return 1; }
  for (r=0;r<rounds;r++) {
    rf[r]=f[0][r]/f[1][r]; rg[r]=g[0][r]/g[1][r]; rl[r]=(f[0][r]+g[0][r])/(f[1][r]+g[1][r]);
  }
  printf("{\"threads\": %u, \"rounds\": %d, \"shape\": [8, 128, 8, 4, 48], \"bf16_instance\": %d, "
      "\"workspace_bytes\": %zu, \"fused_ms\": {\"forward\": %.4f, \"backward\": %.4f}, \"bf16_ms\": "
      "{\"forward\": %.4f, \"backward\": %.4f}, \"fused_over_bf16\": {",th,rounds,inst,ws,median(f[0],rounds),
      median(g[0],rounds),median(f[1],rounds),median(g[1],rounds));
  for (i=0;i<3;i++) {
    double mid,lo,hi; int wins=0;
    boot(ratios[i],rounds,&mid,&lo,&hi);
    for (r=0;r<rounds;r++) wins+=ratios[i][r]>1.0;
    printf("%s\"%s\": {\"median\": %.4f, \"ci95\": [%.4f, %.4f], \"bf16_won\": %d}",i ? ", " : "",names[i],mid,lo,hi,wins);
  }
  printf("}}\n");
  bcir_tensor_pool_stop(pool);
  return 0;
}
