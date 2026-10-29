/* ATTN-RATE's kernels in one process: the tensor unit before (LIB_A) and after (LIB_B), each built as
 * a shared library and opened RTLD_LOCAL, their entries timed in rotated rounds -- A, B and B again
 * (the A/A control) -- on the same inputs, so the three arms share the process, its core and its
 * pages; every round's outputs are held equal across the three before a time is kept. Prints each
 * arm's median ms a call and the paired ratios B/A and B'/B (geometric mean, 95% interval).
 * usage: ab_bench LIB_A LIB_B ROUNDS elem exp|exp_avx2|silu_mul_v2|swiglu_vjp_v2 N
 *        ab_bench LIB_A LIB_B ROUNDS attn B T H KH D THREADS CONTRACT SCRATCH
 * (elementwise: one thread, each sample enough calls to pass 3 ms; attention: a pool of THREADS made
 * and joined around each sample, outside its time, so one library's workers never wait beside the
 * other's) */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct opaque_pool pool_t;
typedef void (*exp_fn)(size_t, const float *, float *);
typedef int (*expv_fn)(int, size_t, const float *, float *);
typedef void (*silu_fn)(pool_t *, size_t, const float *, const float *, float *);
typedef void (*vjp_fn)(pool_t *, size_t, const float *, const float *, const float *, float *, float *);
typedef int (*attn_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *);
typedef size_t (*psize_fn)(unsigned);
typedef pool_t *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(pool_t *);
typedef size_t (*sfl_fn)(size_t, size_t, size_t);
typedef struct lib { exp_fn exp; expv_fn expv; silu_fn silu; vjp_fn vjp; attn_fn attn; psize_fn psize; pinit_fn pinit; pstop_fn pstop; sfl_fn sfl; } lib;
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
static void *alloc64(size_t bytes) { void *p=NULL; if (posix_memalign(&p,64,bytes ? bytes : 64)) return NULL; memset(p,0,bytes); return p; }
static void *sym(void *h,const char *name) {
  void *f=dlsym(h,name);
  if (!f) { fprintf(stderr,"missing %s\n",name); exit(1); }
  return f;
}
static lib open_lib(const char *path) {
  lib l; void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if (!h) { fprintf(stderr,"dlopen %s: %s\n",path,dlerror()); exit(1); }
  *(void **)&l.exp=sym(h,"bcir_tensor_exp"); *(void **)&l.expv=sym(h,"bcir_tensor_exp_variant"); *(void **)&l.silu=sym(h,"bcir_tensor_silu_mul_v2_pool");
  *(void **)&l.vjp=sym(h,"bcir_tensor_swiglu_backward_v2_pool");
  *(void **)&l.attn=sym(h,"bcir_tensor_attention_contract_variant_scan_pool");
  *(void **)&l.psize=sym(h,"bcir_tensor_pool_size"); *(void **)&l.pinit=sym(h,"bcir_tensor_pool_init");
  *(void **)&l.pstop=sym(h,"bcir_tensor_pool_stop"); *(void **)&l.sfl=sym(h,"bcir_tensor_attention_scratch_floats");
  return l;
}
static void interval(const double *r,int n,double out[3]) {
  double m=0,sd=0,h,t; int i;
  for (i=0;i<n;i++) m+=log(r[i]);
  m/=n;
  for (i=0;i<n;i++) sd+=(log(r[i])-m)*(log(r[i])-m);
  sd=sqrt(sd/(n-1));
  t=n==16 ? 2.131 : n==12 ? 2.201 : n==24 ? 2.069 : 2.0;  /* t(0.975, n-1) */
  h=t*sd/sqrt((double)n);
  out[0]=exp(m); out[1]=exp(m-h); out[2]=exp(m+h);
}
int main(int argc,char **argv) {
  lib L[3]; int rounds,r,j,arm,kind,which=-1; double ms[3][64],ratio[2][64],ci[2][3],med[3];
  size_t n=0,b=0,t=0,nh=0,nk=0,d=0,i,reps=1,k2,nq=0,nkv=0,np=0; unsigned threads=1,ct=0; int scratch=0,bad;
  float *x=NULL,*u=NULL,*dy=NULL,*y[3]={0},*z[3]={0},*q=NULL,*kk=NULL,*v=NULL,*p=NULL;
  const char *names[3]={"A","B","B-AA"};
  if (argc<6) return 2;
  L[0]=open_lib(argv[1]); L[1]=open_lib(argv[2]); L[2]=L[1];
  rounds=atoi(argv[3]); if (rounds<2 || rounds>64) return 2;
  kind=!strcmp(argv[4],"elem") ? 0 : !strcmp(argv[4],"attn") ? 1 : -1;
  if (kind==0 && argc==7) {
    which=!strcmp(argv[5],"exp") ? 0 : !strcmp(argv[5],"silu_mul_v2") ? 1 : !strcmp(argv[5],"swiglu_vjp_v2") ? 2 :
        !strcmp(argv[5],"exp_avx2") ? 3 : -1;
    n=strtoul(argv[6],0,10);
    if (which<0 || !n) return 2;
    x=alloc64(n*4); u=alloc64(n*4); dy=alloc64(n*4);
    for (j=0;j<3;j++) { y[j]=alloc64(n*4); z[j]=alloc64(n*4); }
    for (i=0;i<n;i++) { x[i]=sinf((float)i*0.31f)*6.0f; u[i]=cosf((float)i*0.17f); dy[i]=sinf((float)i*0.11f); }
  } else if (kind==1 && argc==13) {
    b=strtoul(argv[5],0,10); t=strtoul(argv[6],0,10); nh=strtoul(argv[7],0,10); nk=strtoul(argv[8],0,10);
    d=strtoul(argv[9],0,10); threads=(unsigned)atoi(argv[10]); ct=(unsigned)atoi(argv[11]); scratch=atoi(argv[12]);
    if (scratch) ct|=8u;  /* BCIR_TENSOR_ATTENTION_SCRATCH */
    nq=b*t*nh*d; nkv=b*t*nk*d; np=scratch ? L[1].sfl(b,t,nh) : b*nh*t*t;
    if (scratch && L[0].sfl(b,t,nh)!=np) return 3;
    q=alloc64(nq*4); kk=alloc64(nkv*4); v=alloc64(nkv*4); p=alloc64(np*4);
    for (j=0;j<3;j++) y[j]=alloc64(nq*4);
    for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f)*1.7f;
    for (i=0;i<nkv;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  } else return 2;
  if (kind==0) {  /* reps: enough calls to pass 3 ms, from the after library's warm call */
    double t0;
    L[1].exp(n,x,y[1]); t0=now(); L[1].exp(n,x,y[1]);
    reps=(size_t)(0.003/fmax(now()-t0,1e-9))+1; if (reps>100000) reps=100000;
  }
  for (r=0;r<rounds;r++) {
    for (j=0;j<3;j++) {
      double t0; pool_t *pool=NULL; void *mem=NULL;
      arm=(j+r)%3;
      if (kind==0) {
        if (which==0) L[arm].exp(n,x,y[arm]);
        else if (which==3) L[arm].expv(2,n,x,y[arm]);
        else if (which==1) L[arm].silu(NULL,n,x,u,y[arm]);
        else L[arm].vjp(NULL,n,x,u,dy,y[arm],z[arm]);
        t0=now();
        for (k2=0;k2<reps;k2++) {
          if (which==0) L[arm].exp(n,x,y[arm]);
          else if (which==3) L[arm].expv(2,n,x,y[arm]);
          else if (which==1) L[arm].silu(NULL,n,x,u,y[arm]);
          else L[arm].vjp(NULL,n,x,u,dy,y[arm],z[arm]);
        }
        ms[arm][r]=1000*(now()-t0)/(double)reps;
      } else {
        if (threads>1) { size_t ps=L[arm].psize(threads); mem=alloc64(ps); pool=L[arm].pinit(mem,ps,threads); }
        memset(y[arm],0,nq*4);
        L[arm].attn(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,y[arm],&bad);
        t0=now(); L[arm].attn(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,y[arm],&bad); ms[arm][r]=1000*(now()-t0);
        if (pool) L[arm].pstop(pool);
        free(mem);
      }
    }
    if (kind==0) {
      if (memcmp(y[0],y[1],n*4) || memcmp(y[0],y[2],n*4) || (which==2 && (memcmp(z[0],z[1],n*4) || memcmp(z[0],z[2],n*4)))) { fprintf(stderr,"outputs differ\n"); return 4; }
    } else if (memcmp(y[0],y[1],nq*4) || memcmp(y[0],y[2],nq*4)) { fprintf(stderr,"outputs differ\n"); return 4; }
    ratio[0][r]=ms[1][r]/ms[0][r]; ratio[1][r]=ms[2][r]/ms[1][r];
  }
  for (j=0;j<3;j++) { double s[64]; memcpy(s,ms[j],rounds*sizeof(double)); qsort(s,rounds,sizeof(double),cmp); med[j]=rounds%2 ? s[rounds/2] : 0.5*(s[rounds/2-1]+s[rounds/2]); }
  interval(ratio[0],rounds,ci[0]); interval(ratio[1],rounds,ci[1]);
  printf("{\"rounds\":%d,\"calls_per_sample\":%zu",rounds,reps);
  for (j=0;j<3;j++) printf(",\"%s_ms_median\":%.6f",names[j],med[j]);
  printf(",\"after_over_before\":[%.4f,%.4f,%.4f],\"aa\":[%.4f,%.4f,%.4f],\"raw_ms\":{",ci[0][0],ci[0][1],ci[0][2],ci[1][0],ci[1][1],ci[1][2]);
  for (j=0;j<3;j++) { printf("%s\"%s\":[",j ? "," : "",names[j]); for (r=0;r<rounds;r++) printf("%s%.6f",r ? "," : "",ms[j][r]); printf("]"); }
  puts("}}");
  return 0;
}
