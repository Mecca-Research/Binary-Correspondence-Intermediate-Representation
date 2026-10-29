/* The attention backward's phases at one shape (instrument_bwd.py's copy of the unit): each round
 * the stored forward untimed, then the backward timed whole and by phase; cycles per call over
 * ROUNDS. usage: bwd_phase_bench B T H KH D THREADS ROUNDS CONTRACT */
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
extern unsigned long long bcir_bw_cycles[8];
int main(int argc, char **argv) {
  if (argc!=9) return 2;
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]),ct=(unsigned)atoi(argv[8]); int rounds=atoi(argv[7]),r,bad,k;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i;
  float *q=malloc(nq*4),*kk=malloc(nkv*4),*v=malloc(nkv*4),*y=malloc(nq*4),*p=calloc(np,4),*dy=malloc(nq*4);
  float *dq=calloc(nq,4),*dk=calloc(nkv,4),*dv=calloc(nkv,4);
  size_t psize=bcir_tensor_pool_size(threads); void *pm=aligned_alloc(64,(psize+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,psize,threads) : NULL;
  unsigned long long wall=0;
  static const char *names[7]={"dv","dp_pass0","dot","dp_pass1","ds","dq","dk"};
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=-1;r<rounds;r++) {  /* round -1 warms */
    unsigned long long c0;
    bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,y,&bad);
    if (r==0) memset(bcir_bw_cycles,0,sizeof bcir_bw_cycles);
    c0=__rdtsc();
    bcir_tensor_attention_backward_contract_variant_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,dy,dq,dk,dv);
    if (r>=0) wall+=__rdtsc()-c0;
  }
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"rounds\":%d,\"wall_cycles_per_call\":%.0f",
         b,t,nh,nk,d,threads,ct,rounds,(double)wall/rounds);
  for (k=0;k<7;k++) printf(",\"%s\":%.0f",names[k],(double)bcir_bw_cycles[k]/rounds);
  puts("}");
  return 0;
}
