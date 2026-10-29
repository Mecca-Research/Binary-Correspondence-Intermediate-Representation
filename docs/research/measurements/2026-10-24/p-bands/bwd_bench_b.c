/* The attention backward's phases at one shape (instrument_bwd4.py's copy of the unit), lent the pool's
 * workspace as the trainer lends it: each round the stored forward untimed, optionally a 1 GiB read after it
 * (EVICT 1: P and the operands out of the caches, as in a step where other layers ran between), then the
 * backward timed whole and by phase; TSC cycles a call over ROUNDS (the phases summed over the threads).
 * usage: bwd_bench4 B T H KH D THREADS ROUNDS CONTRACT [EVICT 0|1] */
#include "bcir_tensor.h"
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
extern unsigned long long bcir_bw_cycles[16];
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n ? n : 64)) exit(3); memset(p,0,n ? n : 64); return p; }
static volatile float evict_sink;
__attribute__((target("avx512f"))) static void evict(const float *e, size_t n) {
  __m512 a=_mm512_setzero_ps(); size_t i;
  for (i=0;i<n;i+=16) a=_mm512_add_ps(a,_mm512_load_ps(e+i));
  evict_sink=_mm512_reduce_add_ps(a);
}
int main(int argc, char **argv) {
  if (argc!=9 && argc!=10) return 2;
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]),ct=(unsigned)atoi(argv[8]); int rounds=atoi(argv[7]),r,bad,k,ev=argc==10 ? atoi(argv[9]) : 0;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i,ne=(size_t)1<<28;
  float *q=alloc64(nq*4),*kk=alloc64(nkv*4),*v=alloc64(nkv*4),*y=alloc64(nq*4),*p=alloc64(np*4),*dy=alloc64(nq*4);
  float *dq=alloc64(nq*4),*dk=alloc64(nkv*4),*dv=alloc64(nkv*4),*e=ev==1 ? alloc64(ne*4) : NULL;
  size_t psize=bcir_tensor_pool_size(threads); void *pm=alloc64((psize+63)/64*64);
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pm,psize,threads) : NULL;
  size_t wsb=bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,threads); void *ws=wsb ? alloc64(wsb) : NULL;
  unsigned long long wall=0;
  static const char *names[16]={"dv","dp","dot","ds","dq","dk","contig_task","dp_again","tile_zero","tile_panel",
                                "kv_in","qdy_in","dq_out","dkdv_out","n14","n15"};
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=-1;r<rounds;r++) {  /* round -1 warms */
    unsigned long long c0;
    bcir_tensor_attention_contract_variant_scan_ws_pool(pool,0,ct|32u,b,t,nh,nk,d,q,kk,v,p,y,&bad,ws,wsb);
    if (ev==1) evict(e,ne);
    if (ev==2) evict(p,np);  /* P read once: from L3 where it fits */
    if (r==0) memset(bcir_bw_cycles,0,sizeof bcir_bw_cycles);
    c0=__rdtsc();
    bcir_tensor_attention_backward_contract_variant_ws_pool(pool,0,ct|32u,b,t,nh,nk,d,q,kk,v,p,dy,dq,dk,dv,ws,wsb);
    if (r>=0) wall+=__rdtsc()-c0;
  }
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"evict\":%d,\"workspace_bytes\":%zu,\"rounds\":%d,"
         "\"wall_cycles_per_call\":%.0f",b,t,nh,nk,d,threads,ct,ev,wsb,rounds,(double)wall/rounds);
  for (k=0;k<14;k++) printf(",\"%s\":%.0f",names[k],(double)bcir_bw_cycles[k]/rounds);
  { double h=0; for (k=0;k<14;k++) if (k!=6) h+=(double)bcir_bw_cycles[k]; (void)h; }
  printf(",\"dq0\":%.9g,\"dk0\":%.9g,\"dv0\":%.9g}\n",dq[nq/3],dk[nkv/3],dv[nkv/3]);
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
