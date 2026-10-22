/* The attention backward from two tensor units in one process (A before, B after; B again the A/A
 * control), rotated rounds; each sample runs its unit's stored forward untimed (the backward
 * overwrites p) and times its backward; every round's dq, dk, dv held equal across the arms.
 * usage: ab_bwd LIB_A LIB_B ROUNDS B T H KH D THREADS CONTRACT */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct opaque_pool pool_t;
typedef int (*fwd_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *);
typedef int (*bwd_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, const float *, float *, float *, float *);
typedef size_t (*psize_fn)(unsigned);
typedef pool_t *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(pool_t *);
typedef struct lib { fwd_fn fwd; bwd_fn bwd; psize_fn psize; pinit_fn pinit; pstop_fn pstop; } lib;
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
static void *alloc64(size_t bytes) { void *p=NULL; if (posix_memalign(&p,64,bytes ? bytes : 64)) return NULL; memset(p,0,bytes); return p; }
static void *sym(void *h,const char *n) { void *f=dlsym(h,n); if (!f) { fprintf(stderr,"missing %s\n",n); exit(1); } return f; }
static lib open_lib(const char *path) {
  lib l; void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if (!h) { fprintf(stderr,"dlopen %s: %s\n",path,dlerror()); exit(1); }
  *(void **)&l.fwd=sym(h,"bcir_tensor_attention_contract_variant_scan_pool");
  *(void **)&l.bwd=sym(h,"bcir_tensor_attention_backward_contract_variant_pool");
  *(void **)&l.psize=sym(h,"bcir_tensor_pool_size"); *(void **)&l.pinit=sym(h,"bcir_tensor_pool_init");
  *(void **)&l.pstop=sym(h,"bcir_tensor_pool_stop");
  return l;
}
static void interval(const double *r,int n,double out[3]) {
  double m=0,sd=0,h,t; int i;
  for (i=0;i<n;i++) m+=log(r[i]);
  m/=n;
  for (i=0;i<n;i++) sd+=(log(r[i])-m)*(log(r[i])-m);
  sd=sqrt(sd/(n-1)); t=n==24 ? 2.069 : n==16 ? 2.131 : n==12 ? 2.201 : 2.0; h=t*sd/sqrt((double)n);
  out[0]=exp(m); out[1]=exp(m-h); out[2]=exp(m+h);
}
int main(int argc,char **argv) {
  lib L[3]; int rounds,r,j,arm,bad; size_t b,t,nh,nk,d,nq,nkv,np,i; unsigned threads,ct;
  double ms[3][64],ratio[2][64],ci[2][3],med[3]; const char *names[3]={"A","B","B-AA"};
  float *q,*k,*v,*y,*p,*dy,*dq[3],*dk[3],*dv[3];
  if (argc!=11) return 2;
  L[0]=open_lib(argv[1]); L[1]=open_lib(argv[2]); L[2]=L[1]; rounds=atoi(argv[3]);
  b=strtoul(argv[4],0,10); t=strtoul(argv[5],0,10); nh=strtoul(argv[6],0,10); nk=strtoul(argv[7],0,10);
  d=strtoul(argv[8],0,10); threads=(unsigned)atoi(argv[9]); ct=(unsigned)atoi(argv[10]);
  if (rounds<2 || rounds>64) return 2;
  nq=b*t*nh*d; nkv=b*t*nk*d; np=b*nh*t*t;
  q=alloc64(nq*4); k=alloc64(nkv*4); v=alloc64(nkv*4); y=alloc64(nq*4); p=alloc64(np*4); dy=alloc64(nq*4);
  for (j=0;j<3;j++) { dq[j]=alloc64(nq*4); dk[j]=alloc64(nkv*4); dv[j]=alloc64(nkv*4); }
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=-1;r<rounds;r++) {
    for (j=0;j<3;j++) {
      pool_t *pool=NULL; void *mem=NULL; double t0;
      arm=r<0 ? j : (j+r)%3;
      if (threads>1) { size_t ps=L[arm].psize(threads); mem=alloc64(ps); pool=L[arm].pinit(mem,ps,threads); }
      L[arm].fwd(pool,0,ct,b,t,nh,nk,d,q,k,v,p,y,&bad);
      t0=now(); L[arm].bwd(pool,0,ct,b,t,nh,nk,d,q,k,v,p,dy,dq[arm],dk[arm],dv[arm]);
      if (r>=0) ms[arm][r]=1000*(now()-t0);
      if (pool) L[arm].pstop(pool);
      free(mem);
    }
    for (j=1;j<3;j++) if (memcmp(dq[0],dq[j],nq*4) || memcmp(dk[0],dk[j],nkv*4) || memcmp(dv[0],dv[j],nkv*4)) { fprintf(stderr,"adjoints differ\n"); return 4; }
    if (r>=0) { ratio[0][r]=ms[1][r]/ms[0][r]; ratio[1][r]=ms[2][r]/ms[1][r]; }
  }
  for (j=0;j<3;j++) { double s[64]; memcpy(s,ms[j],rounds*sizeof(double)); qsort(s,rounds,sizeof(double),cmp); med[j]=rounds%2 ? s[rounds/2] : 0.5*(s[rounds/2-1]+s[rounds/2]); }
  interval(ratio[0],rounds,ci[0]); interval(ratio[1],rounds,ci[1]);
  printf("{\"rounds\":%d",rounds);
  for (j=0;j<3;j++) printf(",\"%s_ms_median\":%.4f",names[j],med[j]);
  printf(",\"after_over_before\":[%.4f,%.4f,%.4f],\"aa\":[%.4f,%.4f,%.4f],\"raw_ms\":{",ci[0][0],ci[0][1],ci[0][2],ci[1][0],ci[1][1],ci[1][2]);
  for (j=0;j<3;j++) { printf("%s\"%s\":[",j ? "," : "",names[j]); for (r=0;r<rounds;r++) printf("%s%.4f",r ? "," : "",ms[j][r]); printf("]"); }
  puts("}}");
  return 0;
}
