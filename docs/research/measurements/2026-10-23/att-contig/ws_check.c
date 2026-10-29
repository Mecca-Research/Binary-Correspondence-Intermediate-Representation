/* ATT-CONTIG's check: one tensor unit, the attention through its entries without a workspace (A) and with
 * one (B: the pool's need; C: a byte short of it, so no task copies), y, the verdict, p (the stored forward's,
 * by FNV-1a), dq, dk and dv byte for byte; a non-finite key in round 2 so the verdict's path is held too.
 * usage: ws_check LIB B T H KH D THREADS CONTRACT -> prints "equal copies=<0|1>" or the first difference */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct opaque_pool pool_t;
typedef int (*fwd_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *);
typedef int (*fwdw_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *, void *, size_t);
typedef int (*bwd_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, const float *, float *, float *, float *);
typedef int (*bwdw_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, const float *, float *, float *, float *, void *, size_t);
typedef size_t (*need_fn)(size_t, size_t, size_t, size_t, size_t, unsigned);
typedef size_t (*psize_fn)(unsigned);
typedef pool_t *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(pool_t *);
static void *sym(void *h,const char *n) { void *f=dlsym(h,n); if (!f) { fprintf(stderr,"missing %s\n",n); exit(1); } return f; }
static void *alloc64(size_t bytes) { void *p=NULL; if (posix_memalign(&p,64,bytes ? bytes : 64)) exit(1); memset(p,0,bytes); return p; }
static unsigned long long fnv(const void *x, size_t n) {
  const unsigned char *b=(const unsigned char *)x; unsigned long long h=1469598103934665603ull; size_t i;
  for (i=0;i<n;i++) { h^=b[i]; h*=1099511628211ull; }
  return h;
}
int main(int argc,char **argv) {
  void *h; fwd_fn fwd; fwdw_fn fwdw; bwd_fn bwd; bwdw_fn bwdw; need_fn need; psize_fn psize; pinit_fn pinit; pstop_fn pstop;
  size_t b,t,nh,nk,d,nq,nkv,np,i,ws_need; unsigned threads,ct; int arm,round,copies;
  float *q,*k,*v,*dy,*p[3],*y[3],*dq[3],*dk[3],*dv[3]; int bad[3]; unsigned long long ph[3];
  pool_t *pool=NULL; void *pmem=NULL; unsigned char *ws;
  if (argc!=9) return 2;
  h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if (!h) { fprintf(stderr,"%s\n",dlerror()); return 1; }
  *(void **)&fwd=sym(h,"bcir_tensor_attention_contract_variant_scan_pool");
  *(void **)&fwdw=sym(h,"bcir_tensor_attention_contract_variant_scan_ws_pool");
  *(void **)&bwd=sym(h,"bcir_tensor_attention_backward_contract_variant_pool");
  *(void **)&bwdw=sym(h,"bcir_tensor_attention_backward_contract_variant_ws_pool");
  *(void **)&need=sym(h,"bcir_tensor_attention_pool_workspace");
  *(void **)&psize=sym(h,"bcir_tensor_pool_size"); *(void **)&pinit=sym(h,"bcir_tensor_pool_init");
  *(void **)&pstop=sym(h,"bcir_tensor_pool_stop");
  b=strtoul(argv[2],0,10); t=strtoul(argv[3],0,10); nh=strtoul(argv[4],0,10); nk=strtoul(argv[5],0,10);
  d=strtoul(argv[6],0,10); threads=(unsigned)atoi(argv[7]); ct=(unsigned)atoi(argv[8]);
  nq=b*t*nh*d; nkv=b*t*nk*d; np=b*nh*t*t;
  q=alloc64(nq*4); k=alloc64(nkv*4); v=alloc64(nkv*4); dy=alloc64(nq*4);
  for (arm=0;arm<3;arm++) { p[arm]=alloc64(np*4); y[arm]=alloc64(nq*4); dq[arm]=alloc64(nq*4); dk[arm]=alloc64(nkv*4); dv[arm]=alloc64(nkv*4); }
  if (threads>1) { size_t ps=psize(threads); pmem=alloc64(ps); pool=pinit(pmem,ps,threads); }
  ws_need=need(b,t,nh,nk,d,threads ? threads : 1); copies=ws_need!=0;
  ws=alloc64(ws_need+64);
  for (round=0;round<2;round++) {
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f); }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    if (round) k[nkv/2]=NAN;  /* a non-finite key: the verdict's path */
    for (arm=0;arm<3;arm++) {
      size_t lend=arm==1 ? ws_need : ws_need ? ws_need-1 : 0; unsigned char *base=ws+1;  /* off a 64-byte boundary */
      for (i=0;i<np;i++) p[arm][i]=NAN;
      bad[arm]=0;
      if (!arm) fwd(pool,0,ct,b,t,nh,nk,d,q,k,v,p[arm],y[arm],&bad[arm]);
      else fwdw(pool,0,ct,b,t,nh,nk,d,q,k,v,p[arm],y[arm],&bad[arm],base,lend);
      ph[arm]=(ct&8u) ? 0 : fnv(p[arm],np*4);
      if (round || (ct&8u)) continue;  /* the backward reads the stored forward's probabilities */
      if (!arm) bwd(pool,0,ct,b,t,nh,nk,d,q,k,v,p[arm],dy,dq[arm],dk[arm],dv[arm]);
      else bwdw(pool,0,ct,b,t,nh,nk,d,q,k,v,p[arm],dy,dq[arm],dk[arm],dv[arm],base,lend);
    }
    for (arm=1;arm<3;arm++) {
      if (memcmp(y[0],y[arm],nq*4)) { printf("round %d arm %d: y differs\n",round,arm); return 4; }
      if (bad[0]!=bad[arm]) { printf("round %d arm %d: verdict %d vs %d\n",round,arm,bad[arm],bad[0]); return 4; }
      if (ph[0]!=ph[arm]) { printf("round %d arm %d: p differs\n",round,arm); return 4; }
      if (!round && !(ct&8u) && (memcmp(dq[0],dq[arm],nq*4) || memcmp(dk[0],dk[arm],nkv*4) || memcmp(dv[0],dv[arm],nkv*4))) {
        printf("round %d arm %d: dq %d dk %d dv %d\n",round,arm,!!memcmp(dq[0],dq[arm],nq*4),!!memcmp(dk[0],dk[arm],nkv*4),
            !!memcmp(dv[0],dv[arm],nkv*4));
        return 4;
      }
    }
    if (round && !bad[0]) { printf("round 1: the non-finite key was not reported\n"); return 4; }
  }
  if (pool) pstop(pool);
  printf("equal copies=%d need=%zu\n",copies,ws_need);
  return 0;
}
