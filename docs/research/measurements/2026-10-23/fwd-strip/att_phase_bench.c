/* The binary32 attention's phases at one shape (instrument_att3.py's copy of the unit): each round the
 * forward then the backward through the workspace entries, lent the pool's need, each timed whole and by
 * phase; cycles per call over ROUNDS after a warm round.
 * usage: att_phase_bench B T H KH D THREADS ROUNDS CONTRACT LEND(0|1) */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
extern unsigned long long bcir_bw_cycles[8], bcir_fw_cycles[8];
static void *alloc64(size_t n) { void *p=NULL; if (posix_memalign(&p,64,n?n:64)) return NULL; memset(p,0,n?n:64); return p; }
int main(int argc, char **argv) {
  if (argc!=10) { fprintf(stderr,"usage: att_phase_bench B T H KH D THREADS ROUNDS CONTRACT LEND\n"); return 2; }
  size_t b=strtoul(argv[1],0,10),t=strtoul(argv[2],0,10),nh=strtoul(argv[3],0,10),nk=strtoul(argv[4],0,10),d=strtoul(argv[5],0,10);
  unsigned threads=(unsigned)atoi(argv[6]),ct=(unsigned)atoi(argv[8]); int rounds=atoi(argv[7]),lend=atoi(argv[9]),r,bad,k;
  size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,i,need,psize;
  float *q=alloc64(nq*4),*kk=alloc64(nkv*4),*v=alloc64(nkv*4),*y=alloc64(nq*4),*p=alloc64(np*4),*dy=alloc64(nq*4);
  float *dq=alloc64(nq*4),*dk=alloc64(nkv*4),*dv=alloc64(nkv*4); void *ws,*pm; bcir_tensor_pool *pool=NULL;
  unsigned long long fw=0,bw=0;
  static const char *fn[6]={"fill","scores","softmax","context","copy_in","copy_out"};
  static const char *bn[8]={"dv","dp_pass0","dot","dp_pass1","ds","dq","dk","copies"};
  if (!q||!kk||!v||!y||!p||!dy||!dq||!dk||!dv) return 1;
  psize=bcir_tensor_pool_size(threads); pm=alloc64(psize);
  if (threads>1) pool=bcir_tensor_pool_init(pm,psize,threads);
  need=lend ? bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,threads) : 0; ws=alloc64(need);
  for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f); }
  for (i=0;i<nkv;i++) { kk[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
  for (r=-1;r<rounds;r++) {  /* round -1 warms */
    unsigned long long c0,c1,c2;
    if (r==0) { memset(bcir_bw_cycles,0,sizeof bcir_bw_cycles); memset(bcir_fw_cycles,0,sizeof bcir_fw_cycles); }
    c0=__rdtsc();
    if (!bcir_tensor_attention_contract_variant_scan_ws_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,y,&bad,ws,need)) return 3;
    c1=__rdtsc();
    if (!bcir_tensor_attention_backward_contract_variant_ws_pool(pool,0,ct,b,t,nh,nk,d,q,kk,v,p,dy,dq,dk,dv,ws,need)) return 4;
    c2=__rdtsc();
    if (r>=0) { fw+=c1-c0; bw+=c2-c1; }
  }
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"rounds\":%d,\"workspace\":%zu,"
         "\"forward_wall\":%.0f,\"backward_wall\":%.0f",b,t,nh,nk,d,threads,ct,rounds,need,(double)fw/rounds,(double)bw/rounds);
  for (k=0;k<6;k++) printf(",\"f_%s\":%.0f",fn[k],(double)bcir_fw_cycles[k]/rounds);
  for (k=0;k<8;k++) printf(",\"b_%s\":%.0f",bn[k],(double)bcir_bw_cycles[k]/rounds);
  puts("}");
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
