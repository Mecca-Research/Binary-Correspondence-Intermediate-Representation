/* ATT-BOUND: the scratch attention's phases at one shape (../../2026-10-22/attn-rate/phase_bench.c with the softmax
 * split by instrument_phases.py): cycles per phase summed over ROUNDS calls, then per
 * call. usage: phase_bench B T H KH D THREADS ROUNDS CONTRACT */
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern unsigned long long bcir_ar_cycles[8];
int main(int argc, char **argv) {
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]); int rounds=atoi(argv[7]),r,bad; unsigned ct=(unsigned)atoi(argv[8]);
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,sf=bcir_tensor_attention_scratch_floats(b,t,nh),i;
  float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*y=malloc(nq*4),*p=calloc(sf,4);
  size_t psize=bcir_tensor_pool_size(threads); void *pm=aligned_alloc(64,(psize+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,psize,threads) : NULL;
  for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f)*1.7f;
  for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p,y,&bad);
  memset(bcir_ar_cycles,0,sizeof bcir_ar_cycles);
  unsigned long long t0=__builtin_ia32_rdtsc();
  for (r=0;r<rounds;r++) bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p,y,&bad);
  unsigned long long all=__builtin_ia32_rdtsc()-t0;
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"wall_cycles_per_call\":%.0f,\"scores\":%.0f,\"softmax\":%.0f,\"context\":%.0f,\"max\":%.0f,\"expsum\":%.0f,\"divide\":%.0f}\n",
         b,t,nh,nk,d,threads,(double)all/rounds,(double)bcir_ar_cycles[0]/rounds,(double)bcir_ar_cycles[1]/rounds,(double)bcir_ar_cycles[2]/rounds,(double)bcir_ar_cycles[3]/rounds,(double)bcir_ar_cycles[4]/rounds,(double)bcir_ar_cycles[5]/rounds);
  return 0;
}
