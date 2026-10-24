/* The attention backward's phases at one shape (instrument_bwd3.py's copy of the unit), lent the pool's
 * workspace as the trainer lends it (ATT-CONTIG's copies and BWD-KEEP's tiles): each round the stored
 * forward untimed, then the backward timed whole and by phase; TSC cycles a call over ROUNDS (the phases
 * summed over the threads). usage: bwd_bound_bench B T H KH D THREADS ROUNDS CONTRACT [WS 0|1] */
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
extern unsigned long long bcir_bw_cycles[8];
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n ? n : 64)) exit(3); memset(p,0,n ? n : 64); return p; }
int main(int argc, char **argv) {
  if (argc!=9 && argc!=10) return 2;
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]),ct=(unsigned)atoi(argv[8]); int rounds=atoi(argv[7]),r,bad,k,use_ws=argc==10 ? atoi(argv[9]) : 1;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i;
  float *q=alloc64(nq*4),*kk=alloc64(nkv*4),*v=alloc64(nkv*4),*y=alloc64(nq*4),*p=alloc64(np*4),*dy=alloc64(nq*4);
  float *dq=alloc64(nq*4),*dk=alloc64(nkv*4),*dv=alloc64(nkv*4);
  size_t psize=bcir_tensor_pool_size(threads); void *pm=alloc64((psize+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,psize,threads) : NULL;
  size_t wsb=use_ws ? bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,threads) : 0; void *ws=wsb ? alloc64(wsb) : NULL;
  unsigned long long wall=0;
  static const char *names[8]={"dv","dp","dot","ds","dq","dk","contig_task","dp_again"};
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=-1;r<rounds;r++) {  /* round -1 warms */
    unsigned long long c0;
    bcir_tensor_attention_contract_variant_scan_ws_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,y,&bad,ws,wsb);
    if (r==0) memset(bcir_bw_cycles,0,sizeof bcir_bw_cycles);
    c0=__rdtsc();
    bcir_tensor_attention_backward_contract_variant_ws_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,dy,dq,dk,dv,ws,wsb);
    if (r>=0) wall+=__rdtsc()-c0;
  }
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"workspace_bytes\":%zu,\"rounds\":%d,"
         "\"wall_cycles_per_call\":%.0f",b,t,nh,nk,d,threads,ct,wsb,rounds,(double)wall/rounds);
  for (k=0;k<8;k++) printf(",\"%s\":%.0f",names[k],(double)bcir_bw_cycles[k]/rounds);
  puts("}");
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
