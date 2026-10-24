/* The banded stored forward's phases at one shape (instrument_fwd.py's copy of the unit), lent the pool's
 * workspace: ROUNDS calls after one warm one, wall TSC cycles a call and the phases summed over the threads.
 * usage: fwd_bench B T H KH D THREADS ROUNDS CONTRACT */
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
extern unsigned long long bcir_fw_cycles[16];
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n ? n : 64)) exit(3); memset(p,0,n ? n : 64); return p; }
int main(int argc, char **argv) {
  if (argc!=9) return 2;
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]),ct=(unsigned)atoi(argv[8]); int rounds=atoi(argv[7]),r,bad,k;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i;
  float *q=alloc64(nq*4),*kk=alloc64(nkv*4),*v=alloc64(nkv*4),*y=alloc64(nq*4),*p=alloc64(np*4);
  size_t psize=bcir_tensor_pool_size(threads); void *pm=alloc64((psize+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,psize,threads) : NULL;
  size_t wsb=bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,threads); void *ws=wsb ? alloc64(wsb) : NULL;
  unsigned long long wall=0;
  static const char *names[6]={"tq","scores","softmax","context","zero","head"};
  for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f)*1.7f;
  for (i=0;i<nkv;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=-1;r<rounds;r++) {
    unsigned long long c0;
    if (r==0) memset(bcir_fw_cycles,0,sizeof bcir_fw_cycles);
    c0=__rdtsc();
    bcir_tensor_attention_contract_variant_scan_ws_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,y,&bad,ws,wsb);
    if (r>=0) wall+=__rdtsc()-c0;
  }
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"rounds\":%d,\"wall_cycles_per_call\":%.0f",
         b,t,nh,nk,d,threads,ct,rounds,(double)wall/rounds);
  for (k=0;k<6;k++) printf(",\"%s\":%.0f",names[k],(double)bcir_fw_cycles[k]/rounds);
  printf(",\"y0\":%.9g}\n",y[nq/3]);
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
