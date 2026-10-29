/* INFER-ATTN end to end: bcir_decoder_forward on one decoder geometry, K timed forwards after a
 * warm one; prints the median ms and the logits' digest (FNV-1a over their bytes), so two builds
 * are compared on the same bits. usage: fwd_bench VOCAB WIDTH HEADS KVHEADS LAYERS FF BATCH TIME
 * CONTRACT THREADS K */
#define _POSIX_C_SOURCE 200112L  /* posix_memalign under -std=c11 */
#include "bcir_decoder_train.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
static void *alloc64(size_t bytes) { void *p=NULL; if (posix_memalign(&p,64,bytes ? bytes : 64)) return NULL; memset(p,0,bytes); return p; }
int main(int argc,char **argv) {
  bcir_decoder_spec spec; bcir_decoder_state st; size_t i,n; unsigned threads; int k,r;
  uint32_t *ids; double ms[256]; uint64_t h=1469598103934665603ull; const unsigned char *lb;
  if (argc!=12) return 2;
  memset(&spec,0,sizeof spec);
  spec.vocab=(uint32_t)atoi(argv[1]); spec.width=(uint32_t)atoi(argv[2]); spec.heads=(uint32_t)atoi(argv[3]);
  spec.kvheads=(uint32_t)atoi(argv[4]); spec.layers=(uint32_t)atoi(argv[5]); spec.ff=(uint32_t)atoi(argv[6]);
  spec.batch=(uint32_t)atoi(argv[7]); spec.time=(uint32_t)atoi(argv[8]); spec.tied=1; spec.rope_base=10000.0; spec.rms_eps=1e-5;
  memset(&st,0,sizeof st);
  if (bcir_decoder_make_plan(&spec,&st.plan)) { fprintf(stderr,"plan refused\n"); return 1; }
  st.contract=(uint32_t)atoi(argv[9]); threads=(unsigned)atoi(argv[10]); k=atoi(argv[11]);
  st.weights=alloc64(st.plan.parameters*4); st.gradients=alloc64(st.plan.parameters*4);
  st.moment1=alloc64(st.plan.parameters*4); st.moment2=alloc64(st.plan.parameters*4);
  st.activations=alloc64(st.plan.activations*4); st.scratch=alloc64(st.plan.scratch*4);
  st.weight_capacity=st.gradient_capacity=st.moment1_capacity=st.moment2_capacity=st.plan.parameters;
  st.activation_capacity=st.plan.activations; st.scratch_capacity=st.plan.scratch;
  st.beta1_power=st.beta2_power=1.0;
  for (i=0;i<st.plan.parameters;i++) st.weights[i]=0.05f*sinf((float)(i+1)*0.37f)+0.02f;
  st.workspace_bytes=bcir_decoder_workspace_bytes(&st.plan,st.contract);
  st.workspace=st.workspace_bytes ? alloc64(st.workspace_bytes) : NULL;
  if (threads>1) { size_t ps=bcir_tensor_pool_size(threads); st.pool=bcir_tensor_pool_init(alloc64(ps),ps,threads); }
  n=(size_t)spec.batch*spec.time; ids=malloc(n*4);
  for (i=0;i<n;i++) ids[i]=(uint32_t)((i*2654435761u)%spec.vocab);
  if (bcir_decoder_forward(&st,ids,n)) { fprintf(stderr,"forward refused\n"); return 1; }
  for (r=0;r<k;r++) { double t0=now(); if (bcir_decoder_forward(&st,ids,n)) return 1; ms[r]=1000*(now()-t0); }
  lb=(const unsigned char *)(st.activations+st.plan.activations-n*spec.vocab);
  for (i=0;i<n*spec.vocab*4;i++) { h^=lb[i]; h*=1099511628211ull; }
  qsort(ms,(size_t)k,sizeof(double),cmp);
  printf("{\"median_ms\":%.4f,\"logits_fnv1a\":\"%016llx\",\"activations_floats\":%zu}\n",ms[k/2],(unsigned long long)h,st.plan.activations);
  if (st.pool) bcir_tensor_pool_stop(st.pool);
  return 0;
}
