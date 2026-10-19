/* BF16-ATTN (2026-10-19): the bf16 GEMM alone, the parent commit's library against this tree's, both
 * loaded into one process (dlopen, RTLD_LOCAL: each its own symbols and its own pool of THREADS
 * threads) -- whether splitting the AMX kernel into begin, blocks and end (so an attention task
 * configures its tiles once) costs the GEMMs anything. Five of the 9.84M decoder's GEMM shapes at
 * 8 x 128 tokens (the gate/up and QKV projections, the down and output projections, a weight
 * adjoint), each call under bf16_mm's widest instance; each round runs both arms over all five, in
 * an order that alternates round by round; the first two rounds are warm-up. The arms' C must be
 * equal bit for bit every round. Prints one JSON line: each arm's median round in ms and the
 * round-by-round ratio (base over this: above 1 means this tree is faster) as its median, a 95%
 * percentile-bootstrap interval (2000 resamples, a fixed seed) and the rounds this tree won.
 * Wall time: indicative, never a gate.
 *
 *   cc -std=c11 -O2 gemm.c -ldl -lm -o gemm && ./gemm BASE.so THIS.so THREADS ROUNDS
 */
#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 512
typedef int (*mm_fn)(void *, int, int, size_t, size_t, size_t, float, const float *, const float *, float,
                     float *, void *, size_t);
typedef size_t (*ws_fn)(size_t, size_t, size_t);
typedef size_t (*psize_fn)(unsigned);
typedef void *(*pinit_fn)(void *, size_t, unsigned);
typedef struct { mm_fn mm; ws_fn ws; void *pool; } arm;
static const size_t shapes[5][5]={  /* ta, tb, m, n, k */
  {0,1,1024,2048,384}, {0,1,1024,768,384}, {0,1,1024,384,1024}, {0,1,1024,384,384}, {1,0,384,1024,1024}};

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
static int load(const char *path,unsigned th,arm *a,void *memory,size_t bytes) {
  void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL); psize_fn ps; pinit_fn pi;
  if (!h) { fprintf(stderr,"%s\n",dlerror()); return 0; }
  a->mm=(mm_fn)dlsym(h,"bcir_tensor_mm_bf16_pool"); a->ws=(ws_fn)dlsym(h,"bcir_tensor_mm_bf16_workspace");
  ps=(psize_fn)dlsym(h,"bcir_tensor_pool_size"); pi=(pinit_fn)dlsym(h,"bcir_tensor_pool_init");
  if (!a->mm || !a->ws || !ps || !pi) { fprintf(stderr,"%s: missing an entry\n",path); return 0; }
  a->pool=NULL;
  if (th>1) {
    if (!ps(th) || ps(th)>bytes) { fprintf(stderr,"%s: no pool of %u\n",path,th); return 0; }
    a->pool=pi(memory,bytes,th);
  }
  return 1;
}
int main(int argc,char **argv) {
  static _Alignas(64) unsigned char pm[2][65536];
  static double t[2][MAXR],ratio[MAXR];
  float *a,*b,*c[2]; void *w[2]; size_t big=0,i,ws=0; arm arms[2]; unsigned th; int rounds,r,s,x,same=1;
  if (argc!=5) { fprintf(stderr,"usage: gemm BASE.so THIS.so THREADS ROUNDS\n"); return 2; }
  th=(unsigned)atoi(argv[3]); rounds=atoi(argv[4]);
  if (rounds<3 || rounds>MAXR || th<1) { fprintf(stderr,"ROUNDS 3..%d, THREADS >= 1\n",MAXR); return 2; }
  if (!load(argv[1],th,&arms[0],pm[0],sizeof pm[0]) || !load(argv[2],th,&arms[1],pm[1],sizeof pm[1])) return 2;
  for (x=0;x<5;x++) {
    size_t m=shapes[x][2],n=shapes[x][3],k=shapes[x][4],e=m*k>n*k ? m*k : n*k;
    if (e>big) big=e;
    if (m*n>big) big=m*n;
    for (s=0;s<2;s++) { size_t q=arms[s].ws(m,n,k); if (q>ws) ws=q; }
  }
  a=malloc(big*4); b=malloc(big*4); c[0]=malloc(big*4); c[1]=malloc(big*4);
  w[0]=aligned_alloc(64,(ws+63)/64*64); w[1]=aligned_alloc(64,(ws+63)/64*64);
  if (!a || !b || !c[0] || !c[1] || !w[0] || !w[1]) return 2;
  for (i=0;i<big;i++) { a[i]=(float)((int)(next()%2001)-1000)*1e-3f; b[i]=(float)((int)(next()%2001)-1000)*1e-3f; }
  for (r=0;r<rounds+2;r++) {
    for (s=0;s<2;s++) {
      int arm_=(r&1) ? 1-s : s; double t0=now();
      for (x=0;x<5;x++) {
        const size_t *q=shapes[x];
        if (!arms[arm_].mm(arms[arm_].pool,(int)q[0],(int)q[1],q[2],q[3],q[4],1.0f,a,b,0.0f,c[arm_],w[arm_],ws)) {
          fprintf(stderr,"a GEMM refused\n"); return 1;
        }
      }
      if (r>=2) t[arm_][r-2]=(now()-t0)*1e3;
    }
    same&=!memcmp(c[0],c[1],shapes[4][2]*shapes[4][3]*4);
  }
  for (r=0;r<rounds;r++) ratio[r]=t[0][r]/t[1][r];
  {
    static double meds[2000]; double rs[MAXR],lo,hi; int j,wins=0;
    for (i=0;i<2000;i++) { for (j=0;j<rounds;j++) rs[j]=ratio[next()%(uint64_t)rounds]; meds[i]=median(rs,rounds); }
    qsort(meds,2000,sizeof(double),cmpd); lo=meds[49]; hi=meds[1949];
    for (r=0;r<rounds;r++) wins+=ratio[r]>1.0;
    printf("{\"threads\": %u, \"rounds\": %d, \"same_bits\": %s, \"base_ms\": %.4f, \"this_ms\": %.4f, "
        "\"base_over_this\": {\"median\": %.4f, \"ci95\": [%.4f, %.4f], \"this_won\": %d}}\n",th,rounds,
        same ? "true" : "false",median(t[0],rounds),median(t[1],rounds),median(ratio,rounds),lo,hi,wins);
  }
  return same ? 0 : 1;
}
