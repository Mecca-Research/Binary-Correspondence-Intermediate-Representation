/* INFER-ATTN: the attention forward stored (P written) against scratch (P a strip at a time), and
 * the stored arm again as an A/A control, rounds in rotated order; the outputs held equal bit for
 * bit first. usage: attn_bench B T H KH D THREADS ROUNDS CONTRACT */
#include "bcir_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
int main(int argc,char **argv) {
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]); int rounds=atoi(argv[7]); unsigned ct=(unsigned)atoi(argv[8]);
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i; int r,arm,bad;
  float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*y0=malloc(nq*4),*y1=malloc(nq*4),*y2=malloc(nq*4);
  float *p0=malloc(np*4),*p2=malloc(np*4),*p1=malloc(bcir_tensor_attention_scratch_floats(b,t,nh)*4);
  size_t psize=bcir_tensor_pool_size(threads); void *pm=aligned_alloc(64,(psize+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,psize,threads) : NULL;
  double ms[3][64],ratio[2][64]; const char *name[3]={"stored","scratch","stored-AA"};
  for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f)*1.7f;
  for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  memset(p0,0,np*4); memset(p2,0,np*4); memset(p1,0,bcir_tensor_attention_scratch_floats(b,t,nh)*4);
  bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct,b,t,nh,nk,d,q,k,v,p0,y0,&bad);
  bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p1,y1,&bad);
  assert(!memcmp(y0,y1,nq*4));
  for (r=0;r<rounds;r++) for (arm=0;arm<3;arm++) {
    int a=(arm+r)%3; double t0=now();
    if (a==0) bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct,b,t,nh,nk,d,q,k,v,p0,y0,&bad);
    else if (a==1) bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p1,y1,&bad);
    else bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct,b,t,nh,nk,d,q,k,v,p2,y2,&bad);
    ms[a][r]=1000*(now()-t0);
  }
  assert(!memcmp(y0,y1,nq*4) && !memcmp(y0,y2,nq*4));
  for (r=0;r<rounds;r++) { ratio[0][r]=ms[1][r]/ms[0][r]; ratio[1][r]=ms[2][r]/ms[0][r]; }
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"rounds\":%d,\"scratch_floats\":%zu,\"stored_floats\":%zu",
      b,t,nh,nk,d,threads,ct,rounds,bcir_tensor_attention_scratch_floats(b,t,nh),np);
  for (arm=0;arm<3;arm++) {
    double s[64]; memcpy(s,ms[arm],rounds*sizeof(double)); qsort(s,rounds,sizeof(double),cmp);
    printf(",\"%s_ms_median\":%.4f",name[arm],s[rounds/2]);
  }
  for (arm=0;arm<2;arm++) {
    double m=0,sd=0,h; for (r=0;r<rounds;r++) m+=log(ratio[arm][r]); m/=rounds;
    for (r=0;r<rounds;r++) sd+=(log(ratio[arm][r])-m)*(log(ratio[arm][r])-m); sd=sqrt(sd/(rounds-1));
    h=2.069*sd/sqrt((double)rounds);  /* t(0.975, 23) for 24 rounds */
    printf(",\"%s_over_stored\":[%.4f,%.4f,%.4f]",arm ? "aa" : "scratch",exp(m),exp(m-h),exp(m+h));
  }
  puts("}");
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
