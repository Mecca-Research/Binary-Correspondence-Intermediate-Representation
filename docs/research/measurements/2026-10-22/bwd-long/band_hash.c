/* The attention backward's outputs (dq, dk, dv, and the score adjoints it leaves in p) after the
 * stored forward, v1 and fused, one FNV-1a digest a case: ../bwd-width/bwd_hash.c's 17 shapes, then
 * contexts past BWD-BAND's threshold (a row of P of 4 KB or more, 1024 floats), whose transposed
 * products run in bands -- whole bands, a context ending inside a band, a GQA group sharing dk and
 * dv, head widths of one, two and three vectors. Two builds are one arithmetic when every line
 * matches. usage: band_hash [short|long|all] */
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fnv(const void *b, size_t n, uint64_t h) { const unsigned char *c=b; while (n--) { h^=*c++; h*=1099511628211ull; } return h; }
int main(int argc, char **argv) {
  static const size_t sh[][5]={{1,1,1,1,2},{1,5,2,1,3},{2,33,4,2,17},{1,64,2,2,8},{1,40,2,1,130},{1,70,2,1,32},
    {1,97,4,2,48},{2,130,2,2,33},{1,300,2,1,64},{1,77,3,3,16},{1,129,2,1,47},{1,20,1,1,32},{1,9,2,1,33},{1,11,2,2,81},
    {2,90,4,4,24},{1,50,2,2,49},{1,60,4,2,15},
    /* past the threshold */
    {1,1024,2,1,32},{1,1040,2,1,32},{1,1027,4,2,16},{2,1100,2,2,48},{1,1500,2,1,64},{1,2048,2,2,32},{1,1031,1,1,81}};
  size_t nshort=17,nall=sizeof sh/sizeof sh[0],lo=0,hi=nall,s,c,i; unsigned cts[]={0,2}; int bad;
  const char *mode=argc>1 ? argv[1] : "all";
  if (!strcmp(mode,"short")) hi=nshort; else if (!strcmp(mode,"long")) lo=nshort; else if (strcmp(mode,"all")) return 2;
  for (s=lo;s<hi;s++) for (c=0;c<2;c++) {
    size_t b=sh[s][0],t=sh[s][1],nh=sh[s][2],nk=sh[s][3],d=sh[s][4],nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
    float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*y=calloc(nq,4),*p=calloc(np,4),*dy=malloc(nq*4);
    float *dq=calloc(nq,4),*dk=calloc(nkv,4),*dv=calloc(nkv,4);
    if (!q || !k || !v || !y || !p || !dy || !dq || !dk || !dv) return 1;
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f+(float)s)*1.7f; dy[i]=cosf((float)i*0.23f)-0.2f; }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    bcir_tensor_attention_contract_variant_scan_pool(NULL,0,cts[c],b,t,nh,nk,d,q,k,v,p,y,&bad);
    bcir_tensor_attention_backward_contract_variant_pool(NULL,0,cts[c],b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv);
    printf("%zux%zux%zux%zux%zu ct%u dq=%016llx dk=%016llx dv=%016llx p=%016llx\n",b,t,nh,nk,d,cts[c],
        (unsigned long long)fnv(dq,nq*4,1469598103934665603ull),(unsigned long long)fnv(dk,nkv*4,1469598103934665603ull),
        (unsigned long long)fnv(dv,nkv*4,1469598103934665603ull),(unsigned long long)fnv(p,np*4,1469598103934665603ull));
    free(q); free(k); free(v); free(y); free(p); free(dy); free(dq); free(dk); free(dv);
  }
  return 0;
}
