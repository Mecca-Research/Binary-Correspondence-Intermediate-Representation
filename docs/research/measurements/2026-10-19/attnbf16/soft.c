/* ATTN-SOFT (2026-10-19): one layer's binary32 attention of the 9.84M decoder (8 x 128 tokens, 8
 * query heads over 4 KV heads of 48 channels), forward and backward under the fused contract with the
 * BCIR exponential (bf16+exp+reduce+attn's attention), the parent commit's library against this
 * tree's, both loaded into one process (dlopen, RTLD_LOCAL: each library its own symbols and its own
 * pool of THREADS threads). Each round runs both arms, forward then backward, in an order that
 * alternates round by round; the first two rounds are warm-up. The two arms' outputs must be equal
 * bit for bit every round (ATTN-SOFT changes no value). Prints one JSON line: each arm's median
 * forward and backward in ms, and per measure the round-by-round ratios (base over soft: above 1
 * means this tree is faster) as their median, a 95% percentile-bootstrap interval of the median
 * (2000 resamples, a fixed seed) and the rounds this tree won. Wall time: indicative, never a gate.
 *
 *   cc -std=c11 -O2 soft.c -ldl -lm -o soft && ./soft BASE.so THIS.so THREADS ROUNDS
 *
 * each library built as the comparison harness builds the trainer:
 *   cc -std=c11 -O3 -ffp-contract=off -fno-math-errno -DBCIR_DT_BUILD_SHARED -shared -fPIC -pthread \
 *      -I runtime/c runtime/c/bcir_tensor.c runtime/c/bcir_decoder_train.c -o X.so -lm
 */
#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 512
typedef void (*scan_fn)(void *, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
                        const float *, const float *, float *, float *, int *);
typedef void (*back_fn)(void *, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
                        const float *, const float *, float *, const float *, float *, float *, float *);
typedef size_t (*psize_fn)(unsigned);
typedef void *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(void *);
typedef struct { scan_fn scan; back_fn back; void *pool; } arm;

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
static void boot(const double *x,int n,double *mid,double *lo,double *hi) {
  static double meds[2000]; double r[MAXR]; int i,j;
  for (i=0;i<2000;i++) { for (j=0;j<n;j++) r[j]=x[next()%(uint64_t)n]; meds[i]=median(r,n); }
  qsort(meds,2000,sizeof(double),cmpd);
  *mid=median(x,n); *lo=meds[49]; *hi=meds[1949];
}
static int load(const char *path,unsigned th,arm *a,void *memory,size_t bytes) {
  void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL); psize_fn ps; pinit_fn pi;
  if (!h) { fprintf(stderr,"%s\n",dlerror()); return 0; }
  a->scan=(scan_fn)dlsym(h,"bcir_tensor_attention_contract_scan_pool");
  a->back=(back_fn)dlsym(h,"bcir_tensor_attention_backward_contract_pool");
  ps=(psize_fn)dlsym(h,"bcir_tensor_pool_size"); pi=(pinit_fn)dlsym(h,"bcir_tensor_pool_init");
  if (!a->scan || !a->back || !ps || !pi) { fprintf(stderr,"%s: missing an entry\n",path); return 0; }
  a->pool=NULL;
  if (th>1) {
    if (!ps(th) || ps(th)>bytes) { fprintf(stderr,"%s: no pool of %u\n",path,th); return 0; }
    a->pool=pi(memory,bytes,th);
  }
  return 1;
}
int main(int argc,char **argv) {
  const size_t b=8,t=128,nh=8,nk=4,d=48,nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
  const unsigned exp_=1u,fused=2u;
  static _Alignas(64) unsigned char pm[2][65536];
  static double f[2][MAXR],g[2][MAXR],rf[MAXR],rg[MAXR],rl[MAXR];
  const char *names[3]={"forward","backward","layer"}; double *ratios[3]={rf,rg,rl};
  float *q,*k,*v,*dy,*p[2],*y[2],*dq[2],*dk[2],*dv[2]; arm arms[2]; unsigned th; int rounds,r,s,bad,same=1; size_t i;
  if (argc!=5) { fprintf(stderr,"usage: soft BASE.so THIS.so THREADS ROUNDS\n"); return 2; }
  th=(unsigned)atoi(argv[3]); rounds=atoi(argv[4]);
  if (rounds<3 || rounds>MAXR || th<1) { fprintf(stderr,"ROUNDS 3..%d, THREADS >= 1\n",MAXR); return 2; }
  if (!load(argv[1],th,&arms[0],pm[0],sizeof pm[0]) || !load(argv[2],th,&arms[1],pm[1],sizeof pm[1])) return 2;
  q=malloc(nq*4); k=malloc(nkv*4); v=malloc(nkv*4); dy=malloc(nq*4);
  for (s=0;s<2;s++) { p[s]=malloc(np*4); y[s]=malloc(nq*4); dq[s]=malloc(nq*4); dk[s]=malloc(nkv*4); dv[s]=malloc(nkv*4);
    if (!p[s] || !y[s] || !dq[s] || !dk[s] || !dv[s]) return 2; }
  if (!q || !k || !v || !dy) return 2;
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
  for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=0;r<rounds+2;r++) {
    for (s=0;s<2;s++) {
      int a=(r&1) ? 1-s : s; double t0,t1,t2;
      t0=now(); arms[a].scan(arms[a].pool,exp_|fused,b,t,nh,nk,d,q,k,v,p[a],y[a],&bad);
      t1=now(); arms[a].back(arms[a].pool,fused,b,t,nh,nk,d,q,k,v,p[a],dy,dq[a],dk[a],dv[a]);
      t2=now();
      if (r>=2) { f[a][r-2]=(t1-t0)*1e3; g[a][r-2]=(t2-t1)*1e3; }
    }
    same&=!memcmp(y[0],y[1],nq*4) && !memcmp(p[0],p[1],np*4) && !memcmp(dq[0],dq[1],nq*4) &&
          !memcmp(dk[0],dk[1],nkv*4) && !memcmp(dv[0],dv[1],nkv*4);
  }
  for (r=0;r<rounds;r++) {
    rf[r]=f[0][r]/f[1][r]; rg[r]=g[0][r]/g[1][r]; rl[r]=(f[0][r]+g[0][r])/(f[1][r]+g[1][r]);
  }
  printf("{\"threads\": %u, \"rounds\": %d, \"shape\": [8, 128, 8, 4, 48], \"same_bits\": %s, "
      "\"base_ms\": {\"forward\": %.4f, \"backward\": %.4f}, \"soft_ms\": {\"forward\": %.4f, "
      "\"backward\": %.4f}, \"base_over_soft\": {",th,rounds,same ? "true" : "false",median(f[0],rounds),
      median(g[0],rounds),median(f[1],rounds),median(g[1],rounds));
  for (i=0;i<3;i++) {
    double mid,lo,hi; int wins=0;
    boot(ratios[i],rounds,&mid,&lo,&hi);
    for (r=0;r<rounds;r++) wins+=ratios[i][r]>1.0;
    printf("%s\"%s\": {\"median\": %.4f, \"ci95\": [%.4f, %.4f], \"soft_won\": %d}",i ? ", " : "",names[i],mid,lo,hi,wins);
  }
  printf("}}\n");
  return same ? 0 : 1;
}
