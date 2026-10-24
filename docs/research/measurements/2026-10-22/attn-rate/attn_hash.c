/* The attention's outputs (y; and p for the stored forward) over shapes, contracts and both paths,
 * one FNV-1a digest a case: two builds are one arithmetic when every line matches. */
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fnv(const void *b, size_t n, uint64_t h) { const unsigned char *c=b; while (n--) { h^=*c++; h*=1099511628211ull; } return h; }
int main(void) {
  static const size_t sh[][5]={{1,1,1,1,2},{1,5,2,1,3},{2,33,4,2,17},{1,64,2,2,8},{1,40,2,1,130},{1,70,2,1,32},
    {1,97,4,2,48},{2,130,2,2,33},{1,300,2,1,64},{1,77,3,3,16},{1,129,2,1,47}};
  unsigned cts[]={0,1,2,3}; size_t s,c,i; int sc,bad;
  for (s=0;s<sizeof sh/sizeof sh[0];s++) for (c=0;c<4;c++) for (sc=0;sc<2;sc++) {
    size_t b=sh[s][0],t=sh[s][1],nh=sh[s][2],nk=sh[s][3],d=sh[s][4],nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
    float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*y=calloc(nq,4),*p=calloc(np,4);
    for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f+(float)s)*1.7f;
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    bcir_tensor_attention_contract_variant_scan_pool(NULL,0,cts[c]|(sc ? BCIR_TENSOR_ATTENTION_SCRATCH : 0u),b,t,nh,nk,d,q,k,v,p,y,&bad);
    printf("%zux%zux%zux%zux%zu ct%u %s y=%016llx%s",b,t,nh,nk,d,cts[c],sc ? "scratch" : "stored ",(unsigned long long)fnv(y,nq*4,1469598103934665603ull),sc ? "\n" : "");
    if (!sc) printf(" p=%016llx\n",(unsigned long long)fnv(p,np*4,1469598103934665603ull));
    free(q); free(k); free(v); free(y); free(p);
  }
  return 0;
}
