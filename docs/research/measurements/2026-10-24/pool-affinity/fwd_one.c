/* The inference forward (bcir_decoder_forward) on one state and one pool, STEPS forwards back to back after WARM
 * untimed ones -- step_one.c with the step replaced by the forward; prints the median and the least ms a forward and
 * the last forward's logits' digest (the activations' last batch x time x vocab floats, as fwd_ab.c holds them). One
 * library a process, so builds that differ in how the pool deals its tasks run as training runs them, one pool on the
 * host. usage: fwd_one LIB VOCAB WIDTH HEADS KVHEADS LAYERS FF BATCH TIME CONTRACT THREADS STEPS WARM */
#define _POSIX_C_SOURCE 200809L
#include "bcir_decoder_train.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef int (*plan_fn)(const bcir_decoder_spec *, bcir_decoder_plan *);
typedef size_t (*wsb_fn)(const bcir_decoder_plan *, uint32_t);
typedef int (*lb_fn)(bcir_decoder_state *, const uint32_t *, size_t);
typedef size_t (*psize_fn)(unsigned);
typedef bcir_tensor_pool *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(bcir_tensor_pool *);
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
static void *alloc64(size_t bytes) { void *p=NULL; if (posix_memalign(&p,64,bytes ? bytes : 64)) return NULL; memset(p,0,bytes); return p; }
static void *sym(void *h,const char *name) { void *f=dlsym(h,name); if (!f) { fprintf(stderr,"missing %s\n",name); exit(1); } return f; }
int main(int argc,char **argv) {
  bcir_decoder_spec spec; bcir_decoder_state st; uint32_t *ids; size_t i,n;
  unsigned threads; int steps,warm,r; double *ms; uint64_t h=1469598103934665603ull;
  plan_fn plan; wsb_fn wsb; lb_fn lb; psize_fn psize; pinit_fn pinit; pstop_fn pstop; void *lh;
  if (argc!=14) return 2;
  lh=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if (!lh) { fprintf(stderr,"dlopen: %s\n",dlerror()); return 1; }
  *(void **)&plan=sym(lh,"bcir_decoder_make_plan"); *(void **)&wsb=sym(lh,"bcir_decoder_workspace_bytes");
  *(void **)&lb=sym(lh,"bcir_decoder_forward");
  *(void **)&psize=sym(lh,"bcir_tensor_pool_size"); *(void **)&pinit=sym(lh,"bcir_tensor_pool_init"); *(void **)&pstop=sym(lh,"bcir_tensor_pool_stop");
  memset(&spec,0,sizeof spec);
  spec.vocab=(uint32_t)atoi(argv[2]); spec.width=(uint32_t)atoi(argv[3]); spec.heads=(uint32_t)atoi(argv[4]);
  spec.kvheads=(uint32_t)atoi(argv[5]); spec.layers=(uint32_t)atoi(argv[6]); spec.ff=(uint32_t)atoi(argv[7]);
  spec.batch=(uint32_t)atoi(argv[8]); spec.time=(uint32_t)atoi(argv[9]); spec.tied=1; spec.rope_base=10000.0; spec.rms_eps=1e-5;
  threads=(unsigned)atoi(argv[11]); steps=atoi(argv[12]); warm=atoi(argv[13]); ms=malloc(sizeof(double)*steps);
  memset(&st,0,sizeof st);
  if (plan(&spec,&st.plan)) { fprintf(stderr,"plan refused\n"); return 1; }
  st.contract=(uint32_t)atoi(argv[10]);
  st.weights=alloc64(st.plan.parameters*4); st.gradients=alloc64(st.plan.parameters*4);
  st.moment1=alloc64(st.plan.parameters*4); st.moment2=alloc64(st.plan.parameters*4);
  st.activations=alloc64(st.plan.activations*4); st.scratch=alloc64(st.plan.scratch*4);
  st.weight_capacity=st.gradient_capacity=st.moment1_capacity=st.moment2_capacity=st.plan.parameters;
  st.activation_capacity=st.plan.activations; st.scratch_capacity=st.plan.scratch; st.beta1_power=st.beta2_power=1.0;
  for (i=0;i<st.plan.parameters;i++) st.weights[i]=0.05f*sinf((float)(i+1)*0.37f)+0.02f;
  st.workspace_bytes=wsb(&st.plan,st.contract); st.workspace=st.workspace_bytes ? alloc64(st.workspace_bytes) : NULL;
  if (threads>1) { size_t ps=psize(threads); st.pool=pinit(alloc64(ps),ps,threads); if (!st.pool) return 1; }
  n=(size_t)spec.batch*spec.time; ids=malloc(n*4);
  for (i=0;i<n;i++) ids[i]=(uint32_t)((i*2654435761u)%spec.vocab);
  for (r=-warm;r<steps;r++) {
    double t0=now();
    if (lb(&st,ids,n)) { fprintf(stderr,"forward refused\n"); return 1; }
    if (r>=0) ms[r]=1000*(now()-t0);
  }
  { const unsigned char *lg=(const unsigned char *)(st.activations+st.plan.activations-n*spec.vocab);
    for (i=0;i<n*spec.vocab*4;i++) { h^=lg[i]; h*=1099511628211ull; } }
  qsort(ms,(size_t)steps,sizeof(double),cmp);
  printf("{\"lib\":\"%s\",\"batch\":%u,\"time\":%u,\"forwards\":%d,\"median_ms\":%.4f,\"min_ms\":%.4f,\"logits_fnv1a\":\"%016llx\"}\n",
         argv[1],spec.batch,spec.time,steps,ms[steps/2],ms[0],(unsigned long long)h);
  if (st.pool) pstop(st.pool);
  return 0;
}
