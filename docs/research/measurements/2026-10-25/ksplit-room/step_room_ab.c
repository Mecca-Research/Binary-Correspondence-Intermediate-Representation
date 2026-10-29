/* KSPLIT-ROOM-0 (step_ab with B's arms lent ROOM_BYTES of workspace at least): the training step (bcir_decoder_loss_backward + bcir_decoder_update) in one process for
 * the trainer before (LIB_A) and after (LIB_B), each built as a shared library and opened RTLD_LOCAL:
 * three states from the same weights -- A, B and B again (the A/A control) -- each with its own pool
 * of THREADS for the whole run (an idle pool's workers spin 64 pauses and yield 512 times, then
 * sleep), each round one step on each in rotated order; after every round the three states' weights
 * are held equal byte for byte before its times are kept. Prints each arm's median ms a step and the paired
 * ratios B/A and B'/B (geometric mean, 95% interval), and the weights' digest at the end.
 * usage: step_ab LIB_A LIB_B VOCAB WIDTH HEADS KVHEADS LAYERS FF BATCH TIME CONTRACT THREADS ROUNDS */
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
typedef int (*lb_fn)(bcir_decoder_state *, const uint32_t *, const uint32_t *, size_t, int, double *);
typedef int (*up_fn)(bcir_decoder_state *, const bcir_decoder_adamw *, double, double *);
typedef size_t (*psize_fn)(unsigned);
typedef bcir_tensor_pool *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(bcir_tensor_pool *);
typedef struct lib { plan_fn plan; wsb_fn wsb; lb_fn lb; up_fn up; psize_fn psize; pinit_fn pinit; pstop_fn pstop; } lib;
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }
static void *alloc64(size_t bytes) { void *p=NULL; if (posix_memalign(&p,64,bytes ? bytes : 64)) return NULL; memset(p,0,bytes); return p; }
static void *sym(void *h,const char *name) {
  void *f=dlsym(h,name);
  if (!f) { fprintf(stderr,"missing %s\n",name); exit(1); }
  return f;
}
static lib open_lib(const char *path) {
  lib l; void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if (!h) { fprintf(stderr,"dlopen %s: %s\n",path,dlerror()); exit(1); }
  *(void **)&l.plan=sym(h,"bcir_decoder_make_plan"); *(void **)&l.wsb=sym(h,"bcir_decoder_workspace_bytes");
  *(void **)&l.lb=sym(h,"bcir_decoder_loss_backward"); *(void **)&l.up=sym(h,"bcir_decoder_update");
  *(void **)&l.psize=sym(h,"bcir_tensor_pool_size"); *(void **)&l.pinit=sym(h,"bcir_tensor_pool_init");
  *(void **)&l.pstop=sym(h,"bcir_tensor_pool_stop");
  return l;
}
static void interval(const double *r,int n,double out[3]) {
  double m=0,sd=0,h,t; int i;
  for (i=0;i<n;i++) m+=log(r[i]);
  m/=n;
  for (i=0;i<n;i++) sd+=(log(r[i])-m)*(log(r[i])-m);
  sd=sqrt(sd/(n-1));
  t=n==24 ? 2.069 : n==16 ? 2.131 : n==12 ? 2.201 : 2.0;  /* t(0.975, n-1) */
  h=t*sd/sqrt((double)n);
  out[0]=exp(m); out[1]=exp(m-h); out[2]=exp(m+h);
}
int main(int argc,char **argv) {
  lib L[3]; bcir_decoder_spec spec; bcir_decoder_state st[3]; bcir_decoder_adamw opt; uint32_t *ids,*tg;
  size_t i,n; unsigned threads; int rounds,r,j,arm; double ms[3][64],ratio[2][64],ci[2][3],med[3],loss,norm;
  const char *names[3]={"A","B","B-AA"}; uint64_t h=1469598103934665603ull; const unsigned char *wb;
  if (argc!=14) return 2;
  L[0]=open_lib(argv[1]); L[1]=open_lib(argv[2]); L[2]=L[1];
  memset(&spec,0,sizeof spec);
  spec.vocab=(uint32_t)atoi(argv[3]); spec.width=(uint32_t)atoi(argv[4]); spec.heads=(uint32_t)atoi(argv[5]);
  spec.kvheads=(uint32_t)atoi(argv[6]); spec.layers=(uint32_t)atoi(argv[7]); spec.ff=(uint32_t)atoi(argv[8]);
  spec.batch=(uint32_t)atoi(argv[9]); spec.time=(uint32_t)atoi(argv[10]); spec.tied=1; spec.rope_base=10000.0; spec.rms_eps=1e-5;
  threads=(unsigned)atoi(argv[12]); rounds=atoi(argv[13]);
  if (rounds<2 || rounds>64) return 2;
  for (j=0;j<3;j++) {
    bcir_decoder_state *s=&st[j];
    memset(s,0,sizeof *s);
    if (L[j].plan(&spec,&s->plan)) { fprintf(stderr,"plan refused\n"); return 1; }
    s->contract=(uint32_t)atoi(argv[11]);
    s->weights=alloc64(s->plan.parameters*4); s->gradients=alloc64(s->plan.parameters*4);
    s->moment1=alloc64(s->plan.parameters*4); s->moment2=alloc64(s->plan.parameters*4);
    s->activations=alloc64(s->plan.activations*4); s->scratch=alloc64(s->plan.scratch*4);
    if (!s->weights || !s->gradients || !s->moment1 || !s->moment2 || !s->activations || !s->scratch) return 1;
    s->weight_capacity=s->gradient_capacity=s->moment1_capacity=s->moment2_capacity=s->plan.parameters;
    s->activation_capacity=s->plan.activations; s->scratch_capacity=s->plan.scratch;
    s->beta1_power=s->beta2_power=1.0;
    for (i=0;i<s->plan.parameters;i++) s->weights[i]=0.05f*sinf((float)(i+1)*0.37f)+0.02f;
    s->workspace_bytes=L[j].wsb(&s->plan,s->contract);
    if (j && getenv("ROOM_BYTES")) { size_t room=(size_t)strtoull(getenv("ROOM_BYTES"),NULL,10); if (room>s->workspace_bytes) s->workspace_bytes=room; }  /* KSPLIT-ROOM-0: B's arms lent the room */
    s->workspace=s->workspace_bytes ? alloc64(s->workspace_bytes) : NULL;
    if (threads>1) { size_t ps=L[j].psize(threads); s->pool=L[j].pinit(alloc64(ps),ps,threads); if (!s->pool) return 1; }
  }
  n=(size_t)spec.batch*spec.time; ids=malloc(n*4); tg=malloc(n*4);
  for (i=0;i<n;i++) { ids[i]=(uint32_t)((i*2654435761u)%spec.vocab); tg[i]=(uint32_t)(((i+1)*2654435761u)%spec.vocab); }
  opt.lr=1e-3; opt.beta1=0.9; opt.beta2=0.999; opt.epsilon=1e-8; opt.weight_decay=0.01; opt.grad_clip=1.0;
  for (r=-1;r<rounds;r++) {  /* round -1 warms every state, its times not kept */
    for (j=0;j<3;j++) {
      double t0;
      arm=r<0 ? j : (j+r)%3;
      t0=now();
      if (L[arm].lb(&st[arm],ids,tg,n,0,&loss) || L[arm].up(&st[arm],&opt,1.0,&norm)) { fprintf(stderr,"step refused\n"); return 1; }
      if (r>=0) ms[arm][r]=1000*(now()-t0);
    }
    if (memcmp(st[0].weights,st[1].weights,st[0].plan.parameters*4) || memcmp(st[0].weights,st[2].weights,st[0].plan.parameters*4)) {
      fprintf(stderr,"weights differ after round %d\n",r); return 4;
    }
    if (r>=0) { ratio[0][r]=ms[1][r]/ms[0][r]; ratio[1][r]=ms[2][r]/ms[1][r]; }
  }
  for (j=0;j<3;j++) { double s[64]; memcpy(s,ms[j],rounds*sizeof(double)); qsort(s,rounds,sizeof(double),cmp); med[j]=rounds%2 ? s[rounds/2] : 0.5*(s[rounds/2-1]+s[rounds/2]); }
  interval(ratio[0],rounds,ci[0]); interval(ratio[1],rounds,ci[1]);
  wb=(const unsigned char *)st[1].weights;
  for (i=0;i<st[1].plan.parameters*4;i++) { h^=wb[i]; h*=1099511628211ull; }
  printf("{\"rounds\":%d,\"steps_each\":%d,\"weights_fnv1a\":\"%016llx\"",rounds,rounds+1,(unsigned long long)h);
  for (j=0;j<3;j++) printf(",\"%s_ms_median\":%.4f",names[j],med[j]);
  printf(",\"after_over_before\":[%.4f,%.4f,%.4f],\"aa\":[%.4f,%.4f,%.4f],\"raw_ms\":{",ci[0][0],ci[0][1],ci[0][2],ci[1][0],ci[1][1],ci[1][2]);
  for (j=0;j<3;j++) { printf("%s\"%s\":[",j ? "," : "",names[j]); for (r=0;r<rounds;r++) printf("%s%.4f",r ? "," : "",ms[j][r]); printf("]"); }
  puts("}}");
  for (j=0;j<3;j++) if (st[j].pool) L[j].pstop(st[j].pool);
  return 0;
}
