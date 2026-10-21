#include "bcir_decoder_train.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
    "native decoder ABI requires IEEE binary32");
#ifdef BCIR_DT_TRACE_PROBE
/* PX-1 (TRACE): the harness's record of every launch the step issues -- each call that reaches the
 * tensor unit, a provider or the pool, in issue order, its operands as (space, byte offset, bytes)
 * in the state the trace was begun on -- so a witness holds the C step to the launches its step
 * program states (bcir/hosted/models/native_launch.py; train.plan.executed.mismatch). A build
 * defines BCIR_DT_TRACE_PROBE to have it (the witness's builds do; the library does not): every
 * dispatch is then the same call after its record, so the state is the library's bit for bit.
 * A record is BCIR_DT_TRACE_WORDS words -- kind, launch (the calls of one job share it), flags,
 * eight dimensions, ten operands -- and records past the lent capacity are counted, never dropped
 * silently. The spaces: 1-4 the parameter arenas, 5 activations, 6 scratch, 7 the bf16 workspace,
 * 8 outside the state (the caller's tokens and targets). */
#define BCIR_DT_TRACE_WORDS BCIR_DT_RECORD_WORDS
static const bcir_decoder_state *trace_state;
static uint64_t *trace_words,trace_launch;
static size_t trace_capacity,trace_count,trace_dropped,trace_mark;
BCIR_DT_API void bcir_decoder_trace_probe_begin(const bcir_decoder_state *s,uint64_t *words,
    size_t capacity);
BCIR_DT_API size_t bcir_decoder_trace_probe_end(size_t *dropped);
BCIR_DT_API size_t bcir_decoder_trace_probe_words(void);
static void tr_rope(size_t b,size_t t,size_t h,size_t d,const float *x,int inverse);
static void tr_rms(size_t rows,size_t width,float eps,const float *x,const float *g,const float *y);
static void tr_add(size_t n,const float *x,const float *y);
static void tr_mul(size_t n,const float *x,const float *y);
static void tr_swiglu_vjp(int v2,size_t n,const float *g,const float *u,const float *dy,
    const float *dg,const float *du);
static void tr_swiglu_vjp_panels(int v2,size_t r,size_t c,const float *g,const float *u,
    const float *dy,const void *dga,const void *dgt,const void *dua,const void *dut);
static void tr_silu_mul(int v2,size_t n,const float *x,const float *u,const float *y);
static void tr_silu_mul_panels(int v2,size_t r,size_t c,const float *x,const float *u,
    const void *a,const void *b);
static void tr_rms_vjp(size_t rows,size_t width,float eps,const float *x,const float *g,
    const float *dy,const float *dx,const float *dg,const float *inv,unsigned how);
static void tr_mm(unsigned entry,int ta,int tb,size_t m,size_t n,size_t k,float alpha,
    const float *a,const float *b,float beta,const float *c,const void *ws,size_t bytes);
static void tr_mm_group(unsigned entry,int contract,const bcir_tensor_mm_call *chain,size_t nchain,
    const bcir_tensor_mm_call *y,size_t ny,const bcir_tensor_bf16_panels *pre,const void *ws,
    size_t bytes);
static void tr_attention(unsigned kind,unsigned contract,int bf16,size_t b,size_t t,size_t h,
    size_t kh,size_t d,const float *q,const float *k,const float *v,const float *p,const float *y,
    const float *dq,const float *dk,const float *dv,const void *ws,size_t bytes);
static int tr_ran(int ran);
static void tr_run(bcir_tensor_task_fn fn,const void *ctx);
static void tr_provider_silu(size_t n,const float *x,const float *y);
static void tr_embed(size_t n,size_t d,const uint32_t *tokens,const float *table,const float *x);
#define bcir_tensor_rope_pool(pool,b,t,h,d,base,x,inv) \
  (tr_rope(b,t,h,d,x,inv),bcir_tensor_rope_pool(pool,b,t,h,d,base,x,inv))
#define bcir_tensor_rms_scan_pool(pool,r,w,eps,x,g,y,bad) \
  (tr_rms(r,w,eps,x,g,y),bcir_tensor_rms_scan_pool(pool,r,w,eps,x,g,y,bad))
#define bcir_tensor_add_pool(pool,n,x,y) (tr_add(n,x,y),bcir_tensor_add_pool(pool,n,x,y))
#define bcir_tensor_mul_pool(pool,n,x,y) (tr_mul(n,x,y),bcir_tensor_mul_pool(pool,n,x,y))
#define bcir_tensor_swiglu_backward_v2_pool(pool,n,g,u,dy,dg,du) \
  (tr_swiglu_vjp(1,n,g,u,dy,dg,du),bcir_tensor_swiglu_backward_v2_pool(pool,n,g,u,dy,dg,du))
#define bcir_tensor_swiglu_backward_pool(pool,n,g,u,dy,dg,du) \
  (tr_swiglu_vjp(0,n,g,u,dy,dg,du),bcir_tensor_swiglu_backward_pool(pool,n,g,u,dy,dg,du))
#define bcir_tensor_swiglu_backward_panels_pool(pool,v2,r,c,g,u,dy,dga,dgt,dua,dut) \
  (tr_swiglu_vjp_panels(v2,r,c,g,u,dy,dga,dgt,dua,dut), \
   bcir_tensor_swiglu_backward_panels_pool(pool,v2,r,c,g,u,dy,dga,dgt,dua,dut))
#define bcir_tensor_silu_mul_v2_scan_pool(pool,n,x,u,y,bad) \
  (tr_silu_mul(1,n,x,u,y),bcir_tensor_silu_mul_v2_scan_pool(pool,n,x,u,y,bad))
#define bcir_tensor_silu_mul_scan_pool(pool,n,x,u,y,bad) \
  (tr_silu_mul(0,n,x,u,y),bcir_tensor_silu_mul_scan_pool(pool,n,x,u,y,bad))
#define bcir_tensor_silu_mul_panels_pool(pool,v2,r,c,x,u,a,b,bad) \
  (tr_silu_mul_panels(v2,r,c,x,u,a,b),bcir_tensor_silu_mul_panels_pool(pool,v2,r,c,x,u,a,b,bad))
#define bcir_tensor_rms_backward_into_pool(pool,r,w,eps,x,g,dy,dx,dg,inv,how) \
  (tr_rms_vjp(r,w,eps,x,g,dy,dx,dg,inv,how), \
   bcir_tensor_rms_backward_into_pool(pool,r,w,eps,x,g,dy,dx,dg,inv,how))
/* PX-3b (DISPATCH): the GEMM and attention entries the step calls name the instance, and each
 * returns the one it ran; tr_ran writes it into the launch's records just written (word 41), so a
 * trace holds the instance that ran, the entry's own answer, never the one asked. */
#define bcir_tensor_mm_contract_variant_pool(pool,ct,va,ta,tb,m,n,k,al,a,b,be,c,ws,bytes) \
  (tr_mm((unsigned)(ct),ta,tb,m,n,k,al,a,b,be,c,ws,bytes), \
   tr_ran(bcir_tensor_mm_contract_variant_pool(pool,ct,va,ta,tb,m,n,k,al,a,b,be,c,ws,bytes)))
#define bcir_tensor_mm_pair_variant_pool(pool,ct,va,x,y) \
  (tr_mm_group(4,ct,x,1,y,1,NULL,NULL,0),tr_ran(bcir_tensor_mm_pair_variant_pool(pool,ct,va,x,y)))
#define bcir_tensor_mm_group_variant_pool(pool,ct,va,chain,nchain,y,ny,pre,ws,bytes) \
  (tr_mm_group((ct)==3 ? 6u : 5u,ct,chain,nchain,y,ny,pre,ws,bytes), \
   tr_ran(bcir_tensor_mm_group_variant_pool(pool,ct,va,chain,nchain,y,ny,pre,ws,bytes)))
#define bcir_tensor_attention_contract_variant_scan_pool(pool,va,ct,b,t,h,kh,d,q,k,v,p,y,bad) \
  (tr_attention(7,ct,0,b,t,h,kh,d,q,k,v,p,y,NULL,NULL,NULL,NULL,0), \
   tr_ran(bcir_tensor_attention_contract_variant_scan_pool(pool,va,ct,b,t,h,kh,d,q,k,v,p,y,bad)))
#define bcir_tensor_attention_bf16_scan_pool(pool,va,ct,b,t,h,kh,d,q,k,v,p,y,bad,ws,bytes) \
  (tr_attention(7,ct,1,b,t,h,kh,d,q,k,v,p,y,NULL,NULL,NULL,ws,bytes), \
   tr_ran(bcir_tensor_attention_bf16_scan_pool(pool,va,ct,b,t,h,kh,d,q,k,v,p,y,bad,ws,bytes)))
#define bcir_tensor_attention_backward_contract_variant_pool(pool,va,ct,b,t,h,kh,d,q,k,v,p,dy,dq,dk,dv) \
  (tr_attention(8,ct,0,b,t,h,kh,d,q,k,v,p,dy,dq,dk,dv,NULL,0), \
   tr_ran(bcir_tensor_attention_backward_contract_variant_pool(pool,va,ct,b,t,h,kh,d,q,k,v,p,dy,dq, \
       dk,dv)))
#define bcir_tensor_attention_bf16_backward_pool(pool,va,b,t,h,kh,d,q,k,v,p,dy,dq,dk,dv,ws,bytes) \
  (tr_attention(8,0,1,b,t,h,kh,d,q,k,v,p,dy,dq,dk,dv,ws,bytes), \
   tr_ran(bcir_tensor_attention_bf16_backward_pool(pool,va,b,t,h,kh,d,q,k,v,p,dy,dq,dk,dv,ws,bytes)))
#define bcir_tensor_pool_run(pool,fn,ctx,tasks) (tr_run(fn,ctx),bcir_tensor_pool_run(pool,fn,ctx,tasks))
#define DT_TRACE_PROVIDER_SILU(n,x,y) tr_provider_silu(n,x,y)
#define DT_TRACE_PROVIDER_MM(ta,tb,m,n,k,al,a,b,be,c) tr_mm(7,ta,tb,m,n,k,al,a,b,be,c,NULL,0)
#define DT_TRACE_EMBED(n,d,tokens,table,x) tr_embed(n,d,tokens,table,x)
#else
#define DT_TRACE_PROVIDER_SILU(n,x,y) ((void)0)
#define DT_TRACE_PROVIDER_MM(ta,tb,m,n,k,al,a,b,be,c) ((void)0)
#define DT_TRACE_EMBED(n,d,tokens,table,x) ((void)0)
#endif
#ifdef BCIR_DT_TIME_PROBE
/* PX-3 (CHOOSE): a measurement's record of what each job of a program run takes -- the nanoseconds
 * from its dispatch to its return, added into the word of its first record's index in words the
 * caller lends (zeroed by the caller), so a measurement reads every choice site's launches in
 * place (bcir/hosted/models/native_choose.py). ISO C11 timespec_get, no POSIX feature macros. A
 * build defines BCIR_DT_TIME_PROBE to have it (the measurement's builds do; the library does not):
 * every job is then the same call between two clock reads, so the state is the library's bit for
 * bit. A job past the lent words is counted, never dropped silently; lent none, the clock writes
 * and counts nothing. */
#include <time.h>
static uint64_t *time_words;
static size_t time_capacity,time_dropped;
BCIR_DT_API void bcir_decoder_time_probe_begin(uint64_t *ns,size_t capacity);
BCIR_DT_API size_t bcir_decoder_time_probe_end(void);
void bcir_decoder_time_probe_begin(uint64_t *ns,size_t capacity) {
  time_words=ns; time_capacity=ns ? capacity : 0; time_dropped=0;
}
size_t bcir_decoder_time_probe_end(void) {
  size_t dropped=time_dropped;
  time_words=NULL; time_capacity=time_dropped=0;
  return dropped;
}
static uint64_t time_now(void) {
  struct timespec t;
  if (timespec_get(&t,TIME_UTC)!=TIME_UTC || t.tv_sec<0 || t.tv_nsec<0) return 0;
  return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec;
}
static void time_job(size_t at,uint64_t t0) {
  uint64_t t1;
  if (!time_words) return;
  t1=time_now();
  if (at<time_capacity) time_words[at]+=t1>t0 ? t1-t0 : 0; else time_dropped++;
}
#define DT_TIME_START(t0) uint64_t t0=time_now()
#define DT_TIME_JOB(at,t0) time_job(at,t0)
#else
#define DT_TIME_START(t0) ((void)0)
#define DT_TIME_JOB(at,t0) ((void)0)
#endif
uint32_t bcir_decoder_abi_version(void) { return 5; }
static int add(size_t *n,size_t a) {
  if (a>SIZE_MAX-*n) return 0;
  *n+=a; return 1;
}
static int product(size_t a,size_t b,size_t *out) {
  if (b && a>SIZE_MAX/b) return 0;
  *out=a*b; return 1;
}
static int term(size_t *n,size_t a,size_t b) {
  size_t v; return product(a,b,&v) && add(n,v);
}
int bcir_decoder_make_plan(const bcir_decoder_spec *s,bcir_decoder_plan *out) {
  bcir_decoder_plan p;
  size_t nd,nk,nf,nv,att,lp=0,la=0,n,k;
  if (!s || !out || !s->vocab || !s->width || !s->heads || !s->kvheads ||
      !s->layers || !s->ff || !s->batch || !s->time || s->tied>1 ||
      s->vocab>(1u<<20) || s->width>65536 || s->ff>(1u<<20) || s->layers>4096 ||
      s->batch>65536 || s->time>65536 || s->width%s->heads ||
      (s->width/s->heads)%2 || s->heads%s->kvheads ||
      !isfinite(s->rope_base) || s->rope_base<=0 || !isfinite(s->rms_eps) ||
      s->rms_eps>FLT_MAX || (float)s->rms_eps<=0) return BCIR_DT_INVALID;
  p.spec=*s; p.parameters=0; p.activations=0; p.scratch=0;
  k=(size_t)s->kvheads*(s->width/s->heads);
  if (!product(s->batch,s->time,&n) || !product(n,s->width,&nd) ||
      !product(n,k,&nk) || !product(n,s->ff,&nf) || !product(n,s->vocab,&nv) ||
      !product(n,s->time,&att) || !product(att,s->heads,&att) ||
      !term(&lp,2,s->width) || !term(&lp,2*(size_t)s->width,s->width) ||
      !term(&lp,2*(size_t)s->width,k) || !term(&lp,3*(size_t)s->width,s->ff) ||
      !term(&p.parameters,s->vocab,s->width) || !term(&p.parameters,s->layers,lp) ||
      !add(&p.parameters,s->width) || (!s->tied && !term(&p.parameters,s->vocab,s->width)) ||
      !term(&la,5,nd) || !term(&la,2,nk) || !add(&la,att) || !term(&la,3,nf) ||
      !term(&p.activations,(size_t)s->layers+1,nd) || !term(&p.activations,s->layers,la) ||
      !add(&p.activations,nd) || !add(&p.activations,nv) ||
      !term(&p.scratch,5,nd) || !term(&p.scratch,2,nk) || !term(&p.scratch,3,nf) ||
      !add(&p.scratch,nv) || p.parameters>SIZE_MAX/sizeof(float) ||
      p.activations>SIZE_MAX/sizeof(float) || p.scratch>SIZE_MAX/sizeof(float))
    return BCIR_DT_CAPACITY;
  *out=p; return BCIR_DT_OK;
}
static int overlap(const void *a,size_t na,const void *b,size_t nb) {
  uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
  if (na>UINTPTR_MAX-x || nb>UINTPTR_MAX-y) return 1;
  return na && nb && x<y+nb && y<x+na;
}
#ifdef BCIR_DT_WORKSPACE_PROBE
/* The harness's floor (a build defining BCIR_DT_WORKSPACE_PROBE has it; the library does not):
 * SIZE_MAX the plan's, any other value the least valid() admits -- so a product that refuses the
 * workspace, which no admitted state reaches, is a verdict the harness can see (WS-PLAN). */
static size_t workspace_admit=SIZE_MAX;
void bcir_decoder_workspace_probe_admit(size_t bytes);
void bcir_decoder_workspace_probe_admit(size_t bytes) { workspace_admit=bytes; }
#endif
/* The state checked for a run. `placed` (PX-4, a program's run): the activation and scratch arenas
 * are as long as the caller lent them, any length, where the fixed step needs the plan's spans -- a
 * program's checker holds each operand inside its arena's lent length, so the two transient arenas
 * need hold only what the program places (bcir_decoder_run_program). */
static int valid_with(bcir_decoder_state *s,int placed) {
  bcir_decoder_plan p; const float *ptr[6]; size_t len[6],cap[6],i,j,acts;
  int rc;
  if (!s) return BCIR_DT_INVALID;
  rc=bcir_decoder_make_plan(&s->plan.spec,&p);
  if (rc) return rc;
  if (p.parameters!=s->plan.parameters || p.activations!=s->plan.activations ||
      p.scratch!=s->plan.scratch) return BCIR_DT_INVALID;
  /* REMAT: the recompute set names layers of this plan; the arena holds what it keeps */
  acts=bcir_decoder_activation_count(&p,s->recompute);
  if (acts==SIZE_MAX) return BCIR_DT_INVALID;
  if (s->contract&~(uint32_t)(BCIR_DT_CONTRACT_FUSED|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_KBLOCK|
      BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTN|BCIR_DT_CONTRACT_ATTNBF16))
    return BCIR_DT_INVALID;
  if (((s->contract&BCIR_DT_CONTRACT_FUSED)!=0)+((s->contract&BCIR_DT_CONTRACT_KBLOCK)!=0)+
      ((s->contract&BCIR_DT_CONTRACT_BF16)!=0)>1) return BCIR_DT_INVALID;  /* one GEMM contract */
  /* the bf16 attention (BF16-ATTN) packs into the bf16 contract's workspace, and its products are
   * the attention's one contract: never the fused attention's too */
  if ((s->contract&BCIR_DT_CONTRACT_ATTNBF16) && (!(s->contract&BCIR_DT_CONTRACT_BF16) ||
      (s->contract&BCIR_DT_CONTRACT_ATTN))) return BCIR_DT_INVALID;
  ptr[0]=s->weights; ptr[1]=s->gradients; ptr[2]=s->moment1; ptr[3]=s->moment2;
  ptr[4]=s->activations; ptr[5]=s->scratch;
  len[0]=len[1]=len[2]=len[3]=p.parameters; len[4]=acts; len[5]=p.scratch;
  cap[0]=s->weight_capacity; cap[1]=s->gradient_capacity;
  cap[2]=s->moment1_capacity; cap[3]=s->moment2_capacity;
  cap[4]=s->activation_capacity; cap[5]=s->scratch_capacity;
  if (placed) { len[4]=cap[4]; len[5]=cap[5]; }
  for (i=0;i<6;i++) {
    if (!ptr[i] || (uintptr_t)ptr[i]%_Alignof(float)) return BCIR_DT_INVALID;
    if (cap[i]<len[i]) return BCIR_DT_CAPACITY;
    if (len[i]>(UINTPTR_MAX-(uintptr_t)ptr[i])/sizeof(float)) return BCIR_DT_CAPACITY;
    if (overlap(ptr[i],len[i]*sizeof(float),s,sizeof(*s))) return BCIR_DT_INVALID;
    for (j=0;j<i;j++) if (overlap(ptr[i],len[i]*sizeof(float),ptr[j],len[j]*sizeof(float)))
      return BCIR_DT_INVALID;
  }
  if (s->contract&BCIR_DT_CONTRACT_BF16) {  /* the GEMMs' workspace: present, at the floor, apart */
    size_t least=bcir_decoder_workspace_floor(&p,s->contract,bcir_tensor_pool_threads(s->pool));
#ifdef BCIR_DT_WORKSPACE_PROBE
    if (workspace_admit!=SIZE_MAX) least=workspace_admit;
#endif
    if (!s->workspace) return BCIR_DT_INVALID;
    if (s->workspace_bytes<least || s->workspace_bytes>UINTPTR_MAX-(uintptr_t)s->workspace)
      return BCIR_DT_CAPACITY;
    if (overlap(s->workspace,s->workspace_bytes,s,sizeof(*s))) return BCIR_DT_INVALID;
    for (i=0;i<6;i++) if (overlap(s->workspace,s->workspace_bytes,ptr[i],len[i]*sizeof(float)))
      return BCIR_DT_INVALID;
  }
  return BCIR_DT_OK;
}
static int valid(bcir_decoder_state *s) { return valid_with(s,0); }
/* the activation arena's floats under the state's recompute set (valid() admitted the set) */
static size_t arena_floats(const bcir_decoder_state *s) {
  size_t n=bcir_decoder_activation_count(&s->plan,s->recompute);
  return n==SIZE_MAX ? s->plan.activations : n;
}
/* the bytes of the activation arena (scratch 0) or the scratch arena (1) a run reaches: the plan's
 * spans under the fixed step, the lent lengths under a program (valid_with admitted either) */
static size_t transient_bytes(const bcir_decoder_state *s,int placed,int scratch) {
  if (placed) return (scratch ? s->scratch_capacity : s->activation_capacity)*sizeof(float);
  return (scratch ? s->plan.scratch : arena_floats(s))*sizeof(float);
}
/* p's `bytes` lie outside the state and every arena the run reaches */
static int external_in(bcir_decoder_state *s,int placed,const void *p,size_t bytes) {
  if (!p || bytes>UINTPTR_MAX-(uintptr_t)p) return 0;
  return !overlap(p,bytes,s,sizeof(*s)) &&
    !overlap(p,bytes,s->weights,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->gradients,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->moment1,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->moment2,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->activations,transient_bytes(s,placed,0)) &&
    !overlap(p,bytes,s->scratch,transient_bytes(s,placed,1)) &&
    !((s->contract&BCIR_DT_CONTRACT_BF16) && overlap(p,bytes,s->workspace,s->workspace_bytes));
}
static int external(bcir_decoder_state *s,const void *p,size_t bytes) {
  return external_in(s,0,p,bytes);
}
static int ids_in(bcir_decoder_state *s,int placed,const uint32_t *x,size_t n) {
  size_t i;
  if (n>SIZE_MAX/sizeof(uint32_t) || !external_in(s,placed,x,n*sizeof(uint32_t)) ||
      (uintptr_t)x%_Alignof(uint32_t)) return 0;
  for (i=0;i<n;i++) if (x[i]>=s->plan.spec.vocab) return 0;
  return 1;
}
static int ids(bcir_decoder_state *s,const uint32_t *x,size_t n) { return ids_in(s,0,x,n); }
static int finite_span(const float *x,size_t n) {
  /* !(|x| <= FLT_MAX) is exactly !isfinite(x); a block with no exit vectorizes */
  size_t i=0,j,end;
  while (i<n) {
    int bad=0;
    end=n-i<1024 ? n : i+1024;
    for (j=i;j<end;j++) bad|=!(fabsf(x[j])<=FLT_MAX);
    if (bad) return 0;
    i=end;
  }
  return 1;
}
/* The step's whole-buffer passes on the pool (NDT-SCAN): the finiteness scans of the weights, the
 * activations and the gradients, and the gradients' zero fill, were the caller's alone -- a tenth
 * of the four-thread step, measured in place, with the other threads waiting. A scan's verdict is
 * "some element is not finite" and a fill writes zeros, the same whatever the partition, so each
 * runs as one task per chunk of scan_chunk(n) elements -- BCIR_DT_SCAN_CHUNK, or n over
 * BCIR_DT_SCAN_TASKS rounded up when that is more, so there are never more than
 * BCIR_DT_SCAN_TASKS chunks -- and a scan's chunks report into the caller's stack, which the
 * caller ORs. */
#define BCIR_DT_SCAN_CHUNK 16384
#define BCIR_DT_SCAN_TASKS 1024
static size_t scan_chunk(size_t n) {
  size_t chunk=n/BCIR_DT_SCAN_TASKS+(n%BCIR_DT_SCAN_TASKS!=0);
  return chunk<BCIR_DT_SCAN_CHUNK ? BCIR_DT_SCAN_CHUNK : chunk;
}
typedef struct span_job { const float *x; float *y; size_t n,chunk; unsigned char *bad; } span_job;
static void scan_task(void *ctx,size_t task) {
  const span_job *j=(const span_job *)ctx; size_t i0=task*j->chunk;
  j->bad[task]=(unsigned char)!finite_span(j->x+i0,j->n-i0<j->chunk ? j->n-i0 : j->chunk);
}
static void fill_task(void *ctx,size_t task) {
  const span_job *j=(const span_job *)ctx; size_t i0=task*j->chunk;
  memset(j->y+i0,0,(j->n-i0<j->chunk ? j->n-i0 : j->chunk)*sizeof(float));
}
/* Whether x[0, n) is finite, on the state's pool; finite_span's verdict exactly. */
static int finite_pool(bcir_decoder_state *s,const float *x,size_t n) {
  unsigned char bad[BCIR_DT_SCAN_TASKS]; span_job j; size_t i,tasks; int any=0;
  j.x=x; j.y=NULL; j.n=n; j.chunk=scan_chunk(n); j.bad=bad;
  tasks=n/j.chunk+(n%j.chunk!=0);
  if (tasks>BCIR_DT_SCAN_TASKS) return finite_span(x,n);  /* the reports' bound, held here */
  bcir_tensor_pool_run(s->pool,scan_task,&j,tasks);
  for (i=0;i<tasks;i++) any|=bad[i];
  return !any;
}
/* y[0, n) = 0 on the state's pool. */
static void zero_pool(bcir_decoder_state *s,float *y,size_t n) {
  span_job j; j.x=NULL; j.y=y; j.n=n; j.chunk=scan_chunk(n); j.bad=NULL;
  bcir_tensor_pool_run(s->pool,fill_task,&j,n/j.chunk+(n%j.chunk!=0));
}
typedef struct geometry { size_t n,d,k,f,v,b,t,h,kh,hd,l,nd,nk,nf,nv,a,lp,la; } geometry;
static geometry geom(const bcir_decoder_spec *s) {
  geometry g;
  g.b=s->batch; g.t=s->time; g.n=g.b*g.t; g.d=s->width; g.h=s->heads;
  g.kh=s->kvheads; g.hd=g.d/g.h; g.k=g.kh*g.hd; g.f=s->ff; g.v=s->vocab;
  g.l=s->layers; g.nd=g.n*g.d; g.nk=g.n*g.k; g.nf=g.n*g.f; g.nv=g.n*g.v;
  g.a=g.b*g.h*g.t*g.t; g.lp=2*g.d+2*g.d*g.d+2*g.d*g.k+3*g.d*g.f;
  g.la=5*g.nd+2*g.nk+g.a+3*g.nf; return g;
}
/* The bf16 workspace (WS-PLAN): what the step's GEMM groups pack, exactly. Every group as the
 * forward and backward issue it (linears, linear, linear_backward, linears_backward and the SwiGLU
 * block's panel groups below), every operand a distinct address but those the step's calls share
 * -- the input q, k and v read alike, the activation two weight adjoints read alike -- and a panel
 * operand NULL and named in pre, as the step names it: counted by the tensor unit's own rule
 * (bcir_tensor_mm_bf16_group_panels_workspace), so the figure is the bytes the group packs, no more
 * (the plan's figure was once every call packed whole, sharing nothing: 2.6 times this for the
 * 9.84M decoder). `panels` counts the SwiGLU block with its panels; `which` 1 counts its panel
 * groups alone, the least at which the block runs with them (a call a panel feeds runs whole). */
static size_t ws_most(size_t most,size_t need) { return need>most ? need : most; }
static bcir_tensor_mm_call ws_call(int ta,int tb,size_t m,size_t n,size_t k,const float *a,
    const float *b) {
  bcir_tensor_mm_call x;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=1.0f; x.a=a; x.b=b; x.beta=0.0f; x.c=NULL;
  return x;
}
static size_t ws_need(geometry g,int panels,int which) {
  enum { N1,N2,CTX,HID,NORM,DL,CUR,DG,DU,DQ,DK,DV,WQ,WK,WV,WO,WG,WU,WD,HEAD,SYMBOLS };
  static const float at[SYMBOLS]; static const unsigned short panel[1];
  const size_t n=g.n,d=g.d,k=g.k,f=g.f,v=g.v; size_t most=0,q;
  bcir_tensor_mm_call c[6]; bcir_tensor_bf16_panels pre[6];
  memset(pre,0,sizeof pre);
  if (!which) {
    /* the forward: q, k and v read one input; o; gate and up read one input; the head */
    c[0]=ws_call(0,1,n,d,d,at+N1,at+WQ); c[1]=ws_call(0,1,n,k,d,at+N1,at+WK);
    c[2]=ws_call(0,1,n,k,d,at+N1,at+WV);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(NULL,0,c,3));
    c[0]=ws_call(0,1,n,d,d,at+CTX,at+WO); most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(NULL,0,c,1));
    c[0]=ws_call(0,1,n,f,d,at+N2,at+WG); c[1]=ws_call(0,1,n,f,d,at+N2,at+WU);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(NULL,0,c,2));
    c[0]=ws_call(0,1,n,v,d,at+NORM,at+HEAD); most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(NULL,0,c,1));
    /* the backward: a linear's dx then dW as one group, q/k/v's input adjoints a chain into dn */
    c[0]=ws_call(0,0,n,d,v,at+DL,at+HEAD); c[1]=ws_call(1,0,v,d,n,at+DL,at+NORM);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(c,1,c+1,1));
    c[0]=ws_call(0,0,n,d,d,at+CUR,at+WO); c[1]=ws_call(1,0,d,d,n,at+CUR,at+CTX);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(c,1,c+1,1));
    c[0]=ws_call(0,0,n,d,d,at+DQ,at+WQ); c[1]=ws_call(0,0,n,d,k,at+DK,at+WK);
    c[2]=ws_call(0,0,n,d,k,at+DV,at+WV); c[3]=ws_call(1,0,d,d,n,at+DQ,at+N1);
    c[4]=ws_call(1,0,k,d,n,at+DK,at+N1); c[5]=ws_call(1,0,k,d,n,at+DV,at+N1);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(c,3,c+3,3));
  }
  if (!panels) {  /* the SwiGLU block's products on binary32 operands, every one packed */
    c[0]=ws_call(0,1,n,d,f,at+HID,at+WD); most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(NULL,0,c,1));
    c[0]=ws_call(0,0,n,f,d,at+CUR,at+WD); c[1]=ws_call(1,0,d,f,n,at+CUR,at+HID);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(c,1,c+1,1));
    c[0]=ws_call(0,0,n,d,f,at+DG,at+WG); c[1]=ws_call(0,0,n,d,f,at+DU,at+WU);
    c[2]=ws_call(1,0,f,d,n,at+DG,at+N2); c[3]=ws_call(1,0,f,d,n,at+DU,at+N2);
    most=ws_most(most,bcir_tensor_mm_bf16_group_workspace(c,2,c+2,2));
  }
  else {  /* its panel groups: the down projection's A, its weight adjoint's B, the adjoints' A, A^T */
    c[0]=ws_call(0,1,n,d,f,NULL,at+WD); pre[0].a=panel;
    most=ws_most(most,bcir_tensor_mm_bf16_group_panels_workspace(NULL,0,c,1,pre));
    pre[0].a=NULL;
    c[0]=ws_call(0,0,n,f,d,at+CUR,at+WD); c[1]=ws_call(1,0,d,f,n,at+CUR,NULL); pre[1].b=panel;
    most=ws_most(most,bcir_tensor_mm_bf16_group_panels_workspace(c,1,c+1,1,pre));
    pre[1].b=NULL;
    for (q=0;q<2;q++) {
      c[q]=ws_call(0,0,n,d,f,NULL,at+(q ? WU : WG)); c[2+q]=ws_call(1,0,f,d,n,NULL,at+N2);
      pre[q].a=panel; pre[2+q].a=panel;
    }
    most=ws_most(most,bcir_tensor_mm_bf16_group_panels_workspace(c,2,c+2,2,pre));
  }
  return most;
}
/* the shapes admit the SwiGLU block's panels (the rest of panels() below is the state's); a build
 * without them (BCIR_DT_NO_PANELS) plans the block packed, as it runs */
static int ws_panels(geometry g) {
#ifdef BCIR_DT_NO_PANELS
  (void)g; return 0;
#else
  return g.n && g.d && g.f && !(g.n%32) && !(g.d%32) && !(g.f%32);
#endif
}
static size_t ws_attention(geometry g,unsigned threads) {
  return bcir_tensor_attention_bf16_pool_workspace(g.b,g.t,g.h,g.kh,g.hd,threads);
}
size_t bcir_decoder_workspace_need(const bcir_decoder_plan *plan,uint32_t contract,unsigned threads) {
  bcir_decoder_plan p; geometry g; size_t most;
  if (!(contract&BCIR_DT_CONTRACT_BF16)) return 0;
  if (!plan || bcir_decoder_make_plan(&plan->spec,&p)) return SIZE_MAX;
  g=geom(&p.spec); most=ws_need(g,ws_panels(g),0);
  if (contract&BCIR_DT_CONTRACT_ATTNBF16) most=ws_most(most,ws_attention(g,threads));
  return most;
}
size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan,uint32_t contract) {
  return bcir_decoder_workspace_need(plan,contract,0);
}
size_t bcir_decoder_workspace_panels(const bcir_decoder_plan *plan,uint32_t contract) {
  bcir_decoder_plan p; geometry g;
  if (!(contract&BCIR_DT_CONTRACT_BF16)) return 0;
  if (!plan || bcir_decoder_make_plan(&plan->spec,&p)) return SIZE_MAX;
  g=geom(&p.spec);
  return ws_panels(g) ? ws_need(g,1,1) : 0;
}
/* the least: below a call's 32-row strip of A and 32-column strip of B (bcir_tensor_mm_bf16_workspace
 * of a 32 x 32 call at its depth) its entry refuses; the step's depths are the width, the hidden
 * width, the KV width, the vocabulary and the rows */
size_t bcir_decoder_workspace_floor(const bcir_decoder_plan *plan,uint32_t contract,unsigned threads) {
  bcir_decoder_plan p; geometry g; size_t depth[5],least=0,i;
  if (!(contract&BCIR_DT_CONTRACT_BF16)) return 0;
  if (!plan || bcir_decoder_make_plan(&plan->spec,&p)) return SIZE_MAX;
  g=geom(&p.spec); depth[0]=g.d; depth[1]=g.f; depth[2]=g.k; depth[3]=g.v; depth[4]=g.n;
  for (i=0;i<5;i++) least=ws_most(least,bcir_tensor_mm_bf16_workspace(32,32,depth[i]));
  if (contract&BCIR_DT_CONTRACT_ATTNBF16) least=ws_most(least,ws_attention(g,threads));
  return least;
}
/* REMAT: layer l is recomputed when its bit is set (bits past 63 name no layer). The kept layers'
 * slots come first in layer order, the recomputed layers' one shared slot after them. */
static int recomputes(uint64_t r,size_t l) { return l<64 && ((r>>l)&1u); }
static size_t recomputed_below(uint64_t r,size_t l) {
  uint64_t x=l>=64 ? r : r&(((uint64_t)1<<l)-1u); size_t c=0;
  while (x) { x&=x-1u; c++; }
  return c;
}
static size_t cache_slots(geometry g,uint64_t r) {
  size_t gone=recomputed_below(r,g.l);
  return g.l-gone+(gone!=0);
}
static size_t slot_of(geometry g,uint64_t r,size_t l) {
  return recomputes(r,l) ? g.l-recomputed_below(r,g.l) : l-recomputed_below(r,l);
}
/* the highest recomputed layer: the shared slot still holds its cache when the backward reaches it */
static size_t top_recomputed(geometry g,uint64_t r) {
  size_t l=g.l<64 ? g.l : 64;
  while (l-->0) if (recomputes(r,l)) return l;
  return SIZE_MAX;
}
size_t bcir_decoder_activation_count(const bcir_decoder_plan *plan,uint64_t recompute) {
  bcir_decoder_plan p; geometry g;
  if (!plan || bcir_decoder_make_plan(&plan->spec,&p)) return SIZE_MAX;
  g=geom(&p.spec);
  if (g.l<64 && (recompute>>g.l)) return SIZE_MAX;
  return p.activations-(g.l-cache_slots(g,recompute))*g.la;
}
typedef struct layer { float *norm1,*q,*k,*v,*o,*norm2,*gate,*up,*down; } layer;
static layer params(float *x,geometry g) {
  layer p;
  p.norm1=x; x+=g.d; p.q=x; x+=g.d*g.d; p.k=x; x+=g.k*g.d;
  p.v=x; x+=g.k*g.d; p.o=x; x+=g.d*g.d; p.norm2=x; x+=g.d;
  p.gate=x; x+=g.f*g.d; p.up=x; x+=g.f*g.d; p.down=x; return p;
}
typedef struct cache { float *n1,*q,*k,*v,*p,*ctx,*x2,*n2,*gate,*up,*hidden; } cache;
static cache activations(float *x,geometry g) {
  cache a;
  a.n1=x; x+=g.nd; a.q=x; x+=g.nd; a.k=x; x+=g.nk; a.v=x; x+=g.nk;
  a.p=x; x+=g.a; a.ctx=x; x+=g.nd; a.x2=x; x+=g.nd; a.n2=x; x+=g.nd;
  a.gate=x; x+=g.nf; a.up=x; x+=g.nf; a.hidden=x; return a;
}
/* The step's kernels: a provider's GEMM or SiLU replaces BCIR's (threaded as the provider
 * threads it); BCIR's GEMM keeps the state's contract (v1, or the fused v2), as do SiLU, the
 * SwiGLU VJP and the attention softmax (v1 libm, or v2 the BCIR exponential); every other kernel
 * runs on the state's pool, serial when it is NULL. The pool
 * kernels' partitions are fixed by the shapes (bcir_tensor.h), so the step is bit-identical at
 * every thread count: the loss's rows are tasks whose terms the caller adds in row order, the
 * gamma VJP's row sums and the embedding scatter are tasks over column ranges, each column's
 * sum in row order (NDT-REDUCE); the update's first pass is tasks over fixed chunks whose maxima
 * and verdicts are exact, with the clipping norm one index-order chain unless the reduce
 * contract names the chunks. */
/* The GEMM contract the state names, as the tensor unit's pair entry numbers them (0 v1, 1 the
 * fused v2, 2 the blocked v3), and 3 for the bf16 contract, whose entries take the state's
 * workspace: one predicate for the forward's linears and the backward's pairs. */
static int gemm_contract(const bcir_decoder_state *s) {
  return (s->contract&BCIR_DT_CONTRACT_BF16) ? 3 : (s->contract&BCIR_DT_CONTRACT_KBLOCK) ? 2 :
    (s->contract&BCIR_DT_CONTRACT_FUSED) ? 1 : 0;
}
#ifdef BCIR_DT_WORKSPACE_PROBE
/* The harness's record of the most workspace a bf16 GEMM group of the step packs, against which
 * it holds bcir_decoder_workspace_bytes: a build defines BCIR_DT_WORKSPACE_PROBE to have it (the
 * harness's own builds do; the library does not). Read and reset by the probe. */
static size_t workspace_seen;
static void workspace_saw(size_t need) { if (need>workspace_seen) workspace_seen=need; }
size_t bcir_decoder_workspace_probe(void);
size_t bcir_decoder_workspace_probe(void) { size_t seen=workspace_seen; workspace_seen=0; return seen; }
#endif
/* calls under the state's contract as one pool job: the chain into one C, then the rest (the
 * bf16 entries' workspace is the state's, admitted by valid(); the others need none); pre names
 * the operands already packed (BF16-PANELS, the bf16 contract only), NULL none. The probe records
 * what the group packs, its panels counted as the entry counts them (WS-PLAN: a panel operand is
 * NULL, which a count without pre reads as one operand two calls share). Every product returns
 * whether it ran: a refusal writes nothing, and the step it belongs to fails (WS-PLAN). */
/* PX-3b (DISPATCH): `variant` the instance the job runs (bcir_tensor.h: 0 the widest its family
 * admits -- the fixed step's -- else one bit of it, a program's); each returns whether it ran. */
static int gemm_group_v(bcir_decoder_state *s,int variant,const bcir_tensor_mm_call *chain,
    size_t nchain,const bcir_tensor_mm_call *y,size_t ny,const bcir_tensor_bf16_panels *pre) {
  int bf=gemm_contract(s)==3;
#ifdef BCIR_DT_WORKSPACE_PROBE
  if (bf) workspace_saw(bcir_tensor_mm_bf16_group_panels_workspace(chain,nchain,y,ny,pre));
#endif
  return bcir_tensor_mm_group_variant_pool(s->pool,gemm_contract(s),variant,chain,nchain,y,ny,
      bf ? pre : NULL,bf ? s->workspace : NULL,bf ? s->workspace_bytes : 0)!=0;
}
static int gemm_group_with(bcir_decoder_state *s,const bcir_tensor_mm_call *chain,size_t nchain,
    const bcir_tensor_mm_call *y,size_t ny,const bcir_tensor_bf16_panels *pre) {
  return gemm_group_v(s,0,chain,nchain,y,ny,pre);
}
static int gemm_group(bcir_decoder_state *s,const bcir_tensor_mm_call *chain,size_t nchain,
    const bcir_tensor_mm_call *y,size_t ny) { return gemm_group_with(s,chain,nchain,y,ny,NULL); }
/* BF16-PANELS: under the bf16 contract the SwiGLU block's outputs -- the hidden activation and the
 * gate and up adjoints -- are read by bf16 GEMMs alone, so their writers store them as the GEMMs'
 * panels (each element the conversion of its binary32 value, the pack's): in the binary32 tensor's
 * slot, the panel as A in its first half and the other (the hidden activation as its weight
 * adjoint's B, an adjoint as its weight adjoint's A^T) in its second -- when the rows and the hidden
 * width are multiples of 32, so the two panels fill the slot exactly, the model's width is too, so
 * every GEMM they feed is whole 32-tiles every way (the grouped entry refuses any other call a panel
 * feeds, and the GEMM's output would then go unwritten), and the slot is 64-byte aligned;
 * otherwise binary32 and the GEMMs' packs, as before. The products are the packed ones bit for bit,
 * so the state is the same either way (a build defining BCIR_DT_NO_PANELS has no panels: the
 * harness's witness). A call a panel feeds runs whole, so the workspace must hold its group's other
 * packs (WS-PLAN): below the panel groups' need the block runs without panels, its calls in chunks. */
static int panels(const bcir_decoder_state *s,geometry g,const float *a,const float *b) {
#ifdef BCIR_DT_NO_PANELS
  (void)s; (void)g; (void)a; (void)b; return 0;
#else
  return gemm_contract(s)==3 && !s->provider.mm && !s->provider.silu && g.n && g.d && g.f &&
    !(g.n%32) && !(g.d%32) && !(g.f%32) && !((uintptr_t)a%64) && !((uintptr_t)b%64) &&
    s->workspace_bytes>=ws_need(g,1,1);
#endif
}
static unsigned short *panel_of(float *slot) { return (unsigned short *)(void *)slot; }
#ifdef BCIR_DT_PANELS_PROBE
/* The harness's record of the panel paths a run took since the last read (a build defines
 * BCIR_DT_PANELS_PROBE to have it): 1 the forward's hidden activation written as panels, 2 the
 * down projection's weight adjoint reading its B panel, 4 the gate and up adjoints as panels.
 * Exported from a shared build, so the Python witness reads it too. */
static unsigned panels_seen;
BCIR_DT_API unsigned bcir_decoder_panels_probe(void);
BCIR_DT_API unsigned bcir_decoder_panels_probe(void) { unsigned u=panels_seen; panels_seen=0; return u; }
#define PANELS_SAW(bit) (panels_seen|=(bit))
#else
#define PANELS_SAW(bit) ((void)0)
#endif
static int gemm_v(bcir_decoder_state *s,int variant,int ta,int tb,size_t m,size_t n,size_t k,
    float alpha,const float *a,const float *b,float beta,float *c) {
  int bf=gemm_contract(s)==3;
  if (s->provider.mm) {  /* a provider's call: not BCIR's instance to name (variant 0) */
    DT_TRACE_PROVIDER_MM(ta,tb,m,n,k,alpha,a,b,beta,c);
    s->provider.mm(s->provider.ctx,ta,tb,m,n,k,alpha,a,b,beta,c);
    return 1;
  }
#ifdef BCIR_DT_WORKSPACE_PROBE
  if (bf) workspace_saw(bcir_tensor_mm_bf16_workspace(m,n,k));
#endif
  return bcir_tensor_mm_contract_variant_pool(s->pool,gemm_contract(s),variant,ta,tb,m,n,k,alpha,a,
      b,beta,c,bf ? s->workspace : NULL,bf ? s->workspace_bytes : 0)!=0;
}
static int gemm(bcir_decoder_state *s,int ta,int tb,size_t m,size_t n,size_t k,float alpha,
    const float *a,const float *b,float beta,float *c) {
  return gemm_v(s,0,ta,tb,m,n,k,alpha,a,b,beta,c);
}
static int linear(bcir_decoder_state *s,size_t r,size_t in,size_t out,const float *x,
    const float *w,float *y) { return gemm(s,0,1,r,out,in,1.0f,x,w,0.0f,y); }
/* A linear's two VJPs -- dx = dy W (beta-scaled) and dW += dy^T x -- read one adjoint and write
 * apart, so BCIR's kernels run them as one pool job (NDT-SCHED-0: one barrier, the narrow
 * product's idle threads on the other's tasks); a provider's GEMM runs them one after the other,
 * threaded as the provider threads it. The results are the two calls', bit for bit. */
static int linear_backward(bcir_decoder_state *s,size_t r,size_t in,size_t out,
    const float *x,const float *w,const float *dy,float *dx,float *dw,float beta,float wbeta) {
  bcir_tensor_mm_call cx,cw;
  if (s->provider.mm) {
    gemm(s,0,0,r,in,out,1.0f,dy,w,beta,dx);
    gemm(s,1,0,out,in,r,1.0f,dy,x,wbeta,dw);
    return 1;
  }
  cx.ta=0; cx.tb=0; cx.m=r; cx.n=in; cx.k=out; cx.alpha=1.0f; cx.a=dy; cx.b=w; cx.beta=beta; cx.c=dx;
  cw.ta=1; cw.tb=0; cw.m=out; cw.n=in; cw.k=r; cw.alpha=1.0f; cw.a=dy; cw.b=x; cw.beta=wbeta; cw.c=dw;
  if (gemm_contract(s)==3) return gemm_group(s,&cx,1,&cw,1);
  return bcir_tensor_mm_pair_variant_pool(s->pool,gemm_contract(s),0,&cx,&cw)!=0;
}
/* The linears that read one input -- the attention's q, k and v, the MLP's gate and up -- as one
 * pool job (NDT-SCHED-1): their outputs are disjoint, so each is its own entry's bit for bit; and
 * their VJPs as one job, the input adjoints a chain into one dx (the first at the call's beta, the
 * rest adding, in the order the separate calls added) and the weight adjoints apart. A provider's
 * GEMM runs the calls one after another, a linear's two VJPs together, as before. */
static int linears(bcir_decoder_state *s,size_t r,size_t in,const float *x,size_t count,
    const size_t *out,const float *const *w,float *const *y) {
  bcir_tensor_mm_call c[BCIR_TENSOR_MM_GROUP]; size_t q; int ran=1;
  if (s->provider.mm || count>BCIR_TENSOR_MM_GROUP) {
    for (q=0;q<count;q++) ran&=linear(s,r,in,out[q],x,w[q],y[q]);
    return ran;
  }
  for (q=0;q<count;q++) {
    c[q].ta=0; c[q].tb=1; c[q].m=r; c[q].n=out[q]; c[q].k=in; c[q].alpha=1.0f; c[q].a=x;
    c[q].b=w[q]; c[q].beta=0.0f; c[q].c=y[q];
  }
  return gemm_group(s,NULL,0,c,count);
}
static int linears_backward(bcir_decoder_state *s,size_t r,size_t in,const float *x,size_t count,
    const size_t *out,const float *const *w,const float *const *dy,float *dx,float *const *dw,
    float beta,float wbeta) {
  bcir_tensor_mm_call cx[BCIR_TENSOR_MM_GROUP],cw[BCIR_TENSOR_MM_GROUP]; size_t q; int ran=1;
  if (s->provider.mm || count>BCIR_TENSOR_MM_GROUP) {
    for (q=0;q<count;q++)
      ran&=linear_backward(s,r,in,out[q],x,w[q],dy[q],dx,dw[q],q ? 1.0f : beta,wbeta);
    return ran;
  }
  for (q=0;q<count;q++) {
    cx[q].ta=0; cx[q].tb=0; cx[q].m=r; cx[q].n=in; cx[q].k=out[q]; cx[q].alpha=1.0f; cx[q].a=dy[q];
    cx[q].b=w[q]; cx[q].beta=q ? 1.0f : beta; cx[q].c=dx;
    cw[q].ta=1; cw[q].tb=0; cw[q].m=out[q]; cw[q].n=in; cw[q].k=r; cw[q].alpha=1.0f; cw[q].a=dy[q];
    cw[q].b=x; cw[q].beta=wbeta; cw[q].c=dw[q];
  }
  return gemm_group(s,cx,count,cw,count);
}
/* y = silu(x) * u (NDT-SWIGLU-FWD): BCIR's SiLU under the state's contract and the product in one
 * pass, the two passes' values; a provider's SiLU into y, then the product, as before. *bad
 * answers for x, u and y (NDT-SCAN-W): BCIR's product from its tasks, a provider's by scanning the
 * three after it. */
static void silu_mul(bcir_decoder_state *s,size_t n,const float *x,const float *u,float *y,
    int *bad) {
  if (s->provider.silu) {
    DT_TRACE_PROVIDER_SILU(n,x,y);
    s->provider.silu(s->provider.ctx,n,x,y); bcir_tensor_mul_pool(s->pool,n,u,y);
    if (!finite_pool(s,x,n) || !finite_pool(s,u,n) || !finite_pool(s,y,n)) *bad=1;
  }
  else if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_silu_mul_v2_scan_pool(s->pool,n,x,u,y,bad);
  else bcir_tensor_silu_mul_scan_pool(s->pool,n,x,u,y,bad);
}
static int swiglu_v2(const bcir_decoder_state *s) { return (s->contract&BCIR_DT_CONTRACT_EXP)!=0; }
static void swiglu_backward(bcir_decoder_state *s,size_t n,const float *g,const float *u,
    const float *dy,float *dg,float *du) {
  if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_swiglu_backward_v2_pool(s->pool,n,g,u,dy,dg,du);
  else bcir_tensor_swiglu_backward_pool(s->pool,n,g,u,dy,dg,du);
}
/* The attention's contract as the tensor unit names it: the softmax's exponential (EXP) and the
 * products' multiply-adds (ATTN-FMA) -- one predicate for the forward and the backward. */
static unsigned attention_contract(const bcir_decoder_state *s) {
  return ((s->contract&BCIR_DT_CONTRACT_EXP) ? BCIR_TENSOR_ATTENTION_EXP : 0u)|
    ((s->contract&BCIR_DT_CONTRACT_ATTN) ? BCIR_TENSOR_ATTENTION_FUSED : 0u);
}
/* Under the bf16 attention (BF16-ATTN) its widest instance, packing into the state's workspace,
 * which valid() admitted for its pool; otherwise the binary32 attention under its contract. Each
 * returns whether it ran (an entry that refuses writes nothing). */
static int attention_v(bcir_decoder_state *s,int variant,size_t b,size_t t,size_t h,size_t kh,
    size_t d,const float *q,const float *k,const float *v,float *p,float *y,int *bad) {
  if (s->contract&BCIR_DT_CONTRACT_ATTNBF16)
    return bcir_tensor_attention_bf16_scan_pool(s->pool,variant,attention_contract(s),b,t,h,kh,d,q,k,
        v,p,y,bad,s->workspace,s->workspace_bytes)!=0;
  return bcir_tensor_attention_contract_variant_scan_pool(s->pool,variant,attention_contract(s),b,t,
      h,kh,d,q,k,v,p,y,bad)!=0;
}
static int attention(bcir_decoder_state *s,size_t b,size_t t,size_t h,size_t kh,size_t d,
    const float *q,const float *k,const float *v,float *p,float *y,int *bad) {
  return attention_v(s,0,b,t,h,kh,d,q,k,v,p,y,bad);
}
static int attention_backward_v(bcir_decoder_state *s,int variant,geometry g,const float *q,
    const float *k,const float *v,float *p,const float *dy,float *dq,float *dk,float *dv) {
  if (s->contract&BCIR_DT_CONTRACT_ATTNBF16)
    return bcir_tensor_attention_bf16_backward_pool(s->pool,variant,g.b,g.t,g.h,g.kh,g.hd,q,k,v,p,dy,
        dq,dk,dv,s->workspace,s->workspace_bytes)!=0;
  return bcir_tensor_attention_backward_contract_variant_pool(s->pool,variant,attention_contract(s),
      g.b,g.t,g.h,g.kh,g.hd,q,k,v,p,dy,dq,dk,dv)!=0;
}
static int attention_backward(bcir_decoder_state *s,geometry g,const float *q,const float *k,
    const float *v,float *p,const float *dy,float *dq,float *dk,float *dv) {
  return attention_backward_v(s,0,g,q,k,v,p,dy,dq,dk,dv);
}
/* The forward, and whether every activation it wrote is finite (NDT-SCAN-W). One scan of the
 * whole arena after it read the arena back from memory, 171 MB for the 9.84M decoder; now the
 * writers answer for it from their tasks (bcir_tensor_rms_scan_pool): each layer's two RMSNorms
 * for their input rows -- the residual stream, and the attention's sum -- and their outputs, the
 * attention for q, k, v, the probabilities and the context, the SwiGLU product for the gate, up
 * and hidden activations, the final norm for the last layer's output and its own; the logits are
 * scanned after the head. Every element of the arena is in exactly one call's regions, each
 * called after the region's last write, so the verdict is the one scan's, and every layer runs
 * whatever an earlier one held.
 * One layer, into its cache slot from its input x, the next layer's input after it (REMAT: the
 * backward's recompute, `full` 0, stops at the hidden activation -- the last value the layer's
 * backward reads -- so the down projection and the residual add run in the forward alone, and
 * its verdict is the forward's, whose values it writes again). Returns 0 when a product refused
 * its workspace, which wrote nothing. */
static int layer_forward(bcir_decoder_state *s,geometry g,size_t l,float *slot,float *x,int full,
    int *bad) {
  const bcir_decoder_spec *sp=&s->plan.spec; int ran=1;
  layer p=params(s->weights+g.v*g.d+l*g.lp,g);
  cache a=activations(slot,g); float *y=x+g.nd;
  bcir_tensor_rms_scan_pool(s->pool,g.n,g.d,(float)sp->rms_eps,x,p.norm1,a.n1,bad);
  {
    size_t out[3]; const float *w[3]; float *y3[3];
    out[0]=g.d; out[1]=g.k; out[2]=g.k; w[0]=p.q; w[1]=p.k; w[2]=p.v; y3[0]=a.q; y3[1]=a.k; y3[2]=a.v;
    ran&=linears(s,g.n,g.d,a.n1,3,out,w,y3);
  }
  bcir_tensor_rope_pool(s->pool,g.b,g.t,g.h,g.hd,sp->rope_base,a.q,0);
  bcir_tensor_rope_pool(s->pool,g.b,g.t,g.kh,g.hd,sp->rope_base,a.k,0);
  ran&=attention(s,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,a.ctx,bad);
  ran&=linear(s,g.n,g.d,g.d,a.ctx,p.o,a.x2);
  bcir_tensor_add_pool(s->pool,g.nd,x,a.x2);
  bcir_tensor_rms_scan_pool(s->pool,g.n,g.d,(float)sp->rms_eps,a.x2,p.norm2,a.n2,bad);
  {
    size_t out[2]; const float *w[2]; float *y2[2];
    out[0]=g.f; out[1]=g.f; w[0]=p.gate; w[1]=p.up; y2[0]=a.gate; y2[1]=a.up;
    ran&=linears(s,g.n,g.d,a.n2,2,out,w,y2);
  }
  if (panels(s,g,a.hidden,a.hidden)) {
    bcir_tensor_mm_call c; bcir_tensor_bf16_panels pre;
    ran&=bcir_tensor_silu_mul_panels_pool(s->pool,swiglu_v2(s),g.n,g.f,a.gate,a.up,
        panel_of(a.hidden),panel_of(a.hidden)+g.nf,bad);
    c.ta=0; c.tb=1; c.m=g.n; c.n=g.d; c.k=g.f; c.alpha=1.0f; c.a=NULL; c.b=p.down; c.beta=0.0f; c.c=y;
    pre.a=panel_of(a.hidden); pre.b=NULL;
    if (full) ran&=gemm_group_with(s,NULL,0,&c,1,&pre);
    PANELS_SAW(1u);
  }
  else {
    silu_mul(s,g.nf,a.gate,a.up,a.hidden,bad);
    if (full) ran&=linear(s,g.n,g.f,g.d,a.hidden,p.down,y);
  }
  if (full) bcir_tensor_add_pool(s->pool,g.nd,a.x2,y);
  return ran;
}
/* The embedding: each of the n tokens' rows of the d-wide table into x, in order (the forward's
 * first launch, and a program's EMBED). */
static void embed(size_t n,size_t d,const uint32_t *tokens,const float *table,float *x) {
  size_t i;
  DT_TRACE_EMBED(n,d,tokens,table,x);
  for (i=0;i<n;i++) memcpy(x+i*d,table+(size_t)tokens[i]*d,d*sizeof(float));
}
/* BCIR_DT_OK, BCIR_DT_NUMERIC for a value not finite, BCIR_DT_CAPACITY for a product that ran
 * nowhere (WS-PLAN: valid() admits no workspace that refuses, so this is a defect's verdict) */
static int forward(bcir_decoder_state *s,const uint32_t *tokens,geometry g) {
  const bcir_decoder_spec *sp=&s->plan.spec;
  size_t l; float *xs=s->activations,*cs=xs+(g.l+1)*g.nd;
  float *norm=cs+cache_slots(g,s->recompute)*g.la,*logits=norm+g.nd;
  float *final=s->weights+g.v*g.d+g.l*g.lp;
  const float *head=sp->tied ? s->weights : final+g.d; int bad=0,ran=1;
  embed(g.n,g.d,tokens,s->weights,xs);
  for (l=0;l<g.l;l++)
    ran&=layer_forward(s,g,l,cs+slot_of(g,s->recompute,l)*g.la,xs+l*g.nd,1,&bad);
  bcir_tensor_rms_scan_pool(s->pool,g.n,g.d,(float)sp->rms_eps,xs+g.l*g.nd,final,norm,&bad);
  ran&=linear(s,g.n,g.d,g.v,norm,head,logits);
  if (!finite_pool(s,logits,g.nv)) bad=1;
  return !ran ? BCIR_DT_CAPACITY : bad ? BCIR_DT_NUMERIC : BCIR_DT_OK;
}
int bcir_decoder_forward(bcir_decoder_state *s,const uint32_t *tokens,size_t n) {
  geometry g; int rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n)) return BCIR_DT_INVALID;
  if (!finite_pool(s,s->weights,s->plan.parameters)) return BCIR_DT_NUMERIC;
  return forward(s,tokens,g);
}
/* The loss's rows in ranges of BCIR_DT_ROW_CHUNK: a row's softmax, its adjoint and its term
 * (log-sum-exp plus the maximum less the target's logit, in double, as the serial loop computed
 * it) are its own; the terms are kept as bytes in the scratch arena and added in row order by the
 * caller, so the mean is the serial chain. */
#define BCIR_DT_ROW_CHUNK 16
#define BCIR_DT_COL_CHUNK 16
typedef struct loss_job {
  size_t n,v; const float *logits; const uint32_t *targets; float *dl; unsigned char *terms;
} loss_job;
static void loss_task(void *ctx,size_t task) {
  loss_job *jb=(loss_job *)ctx; size_t i,j,v=jb->v;
  size_t i0=task*BCIR_DT_ROW_CHUNK,i1=i0+BCIR_DT_ROW_CHUNK<jb->n ? i0+BCIR_DT_ROW_CHUNK : jb->n;
  for (i=i0;i<i1;i++) {
    const float *row=jb->logits+i*v; float *d=jb->dl+i*v; float maxv=row[0]; double sum=0.0,term;
    for (j=1;j<v;j++) if (row[j]>maxv) maxv=row[j];
    for (j=0;j<v;j++) sum+=exp((double)row[j]-maxv);
    term=log(sum)+(double)maxv-row[jb->targets[i]];
    for (j=0;j<v;j++) d[j]=(float)((exp((double)row[j]-maxv)/sum-
        (j==jb->targets[i] ? 1.0 : 0.0))/(double)jb->n);
    memcpy(jb->terms+i*sizeof(double),&term,sizeof(double));
  }
}
/* The loss's g.n rows over g.v logits on the pool -- their adjoints into dl, their terms into
 * `terms` -- and the terms added in row order: the sum the step's mean divides (the backward's,
 * and a program's LOSS). */
static double loss_rows(bcir_decoder_state *s,geometry g,const float *logits,
    const uint32_t *targets,float *dl,unsigned char *terms) {
  loss_job lj; size_t i; double ce=0.0;
  lj.n=g.n; lj.v=g.v; lj.logits=logits; lj.targets=targets; lj.dl=dl; lj.terms=terms;
  bcir_tensor_pool_run(s->pool,loss_task,&lj,(g.n+BCIR_DT_ROW_CHUNK-1)/BCIR_DT_ROW_CHUNK);
  for (i=0;i<g.n;i++) {
    double term; memcpy(&term,lj.terms+i*sizeof(double),sizeof(double)); ce+=term;
  }
  return ce;
}
/* The embedding scatter in column ranges: a task adds every row's columns to its token's row in
 * row order, so a token the batch repeats accumulates as the serial loop did. */
typedef struct scatter_job { size_t n,d; const uint32_t *tokens; const float *dx; float *grad; } scatter_job;
static void scatter_task(void *ctx,size_t task) {
  scatter_job *jb=(scatter_job *)ctx; size_t i,c,d=jb->d;
  size_t c0=task*BCIR_DT_COL_CHUNK,c1=c0+BCIR_DT_COL_CHUNK<d ? c0+BCIR_DT_COL_CHUNK : d;
  for (i=0;i<jb->n;i++) {
    float *row=jb->grad+(size_t)jb->tokens[i]*d; const float *src=jb->dx+i*d;
    for (c=c0;c<c1;c++) row[c]+=src[c];
  }
}
/* the n rows of dx added into the table adjoint's token rows (the backward's last launch, and a
 * program's SCATTER) */
static void scatter(bcir_decoder_state *s,size_t n,size_t d,const uint32_t *tokens,const float *dx,
    float *grad) {
  scatter_job sj;
  sj.n=n; sj.d=d; sj.tokens=tokens; sj.dx=dx; sj.grad=grad;
  bcir_tensor_pool_run(s->pool,scatter_task,&sj,(d+BCIR_DT_COL_CHUNK-1)/BCIR_DT_COL_CHUNK);
}
static void rms_backward(bcir_decoder_state *s,size_t rows,size_t width,const float *x,
    const float *g,const float *dy,float *dx,float *dg,float *inv,unsigned how) {
  bcir_tensor_rms_backward_into_pool(s->pool,rows,width,(float)s->plan.spec.rms_eps,x,g,dy,dx,dg,
      inv,how);
}
/* What a run knows about its step that a lone call does not (NDT-FACTS-CALL; no caller code runs
 * between a run's micro-batches and its update, and nothing it writes overlaps the state):
 * DT_KNOWN -- the weights, and the gradient being accumulated, were scanned finite by this step
 * and are unchanged, so the opening scans would pass; DT_DEFER -- the closing gradient scan is the
 * update's first pass, which the run calls next with the same verdict on every element. */
#define DT_KNOWN 1u
#define DT_DEFER 2u
/* DT_COMMITTED (FACTS-RUN) -- the weights are the run's own last commit's, unchanged since, so the
 * opening weight scan would pass: an update commits only once every candidate is admitted -- no
 * larger than FLT_MAX, NaN excluded, by the preflight or by adamw_bounded's proof -- and the commit
 * stores those candidates, value for value; between that commit and the next step's first
 * micro-batch the run writes only its events, outside the state. A lone call never knows it: its
 * caller may have written the weights since. */
#define DT_COMMITTED 4u
#ifdef BCIR_DT_SCAN_PROBE
/* The harness's count of the opening weight scans since the last read (a build defines
 * BCIR_DT_SCAN_PROBE to have it; the library does not): FACTS-RUN's witness, a run of several
 * steps scanning once. Exported from a shared build, so the Python witness reads it too. */
static unsigned long weight_scans;
BCIR_DT_API unsigned long bcir_decoder_scan_probe(void);
BCIR_DT_API unsigned long bcir_decoder_scan_probe(void) {
  unsigned long n=weight_scans; weight_scans=0; return n;
}
#endif
/* the opening scan of the weights (counted where the probe is built in) */
static int weights_finite(bcir_decoder_state *s) {
#ifdef BCIR_DT_SCAN_PROBE
  weight_scans++;
#endif
  return finite_pool(s,s->weights,s->plan.parameters);
}
static int loss_backward_with(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t n,int accumulate,double *loss,unsigned how) {
  geometry g; size_t l; double ce;
  float *xs,*cs,*norm,*logits,*cur,*tmp,*dn,*dq,*dk,*dv,*dc,*dg,*du,*dh,*dl;
  float *final,*dfinal,*head,*dhead,wb; unsigned gf; int first,ran=1,rc=valid(s); size_t top;
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n) || !ids(s,targets,n) ||
      (accumulate!=0 && accumulate!=1) || !external(s,loss,sizeof(*loss)) ||
      (uintptr_t)loss%_Alignof(double)) return BCIR_DT_INVALID;
  if ((!(how&(DT_KNOWN|DT_COMMITTED)) && !weights_finite(s)) ||
      (!(how&DT_KNOWN) && accumulate && !finite_pool(s,s->gradients,s->plan.parameters)))
    return BCIR_DT_NUMERIC;
  rc=forward(s,tokens,g);
  if (rc) return rc;
  /* NDT-ZERO: a fresh gradient's first writers write it -- each weight adjoint's GEMM at beta 0,
   * which adds its terms to +0.0 without reading C (what the fill and beta 1 gave), each gamma
   * VJP from +0.0 -- so only the region the scatter writes first, an untied embedding, is filled;
   * a provider's GEMM keeps the fill and beta 1. */
  first=!accumulate && !s->provider.mm; wb=first ? 0.0f : 1.0f; gf=first ? BCIR_TENSOR_RMSB_FIRST : 0u;
  if (!accumulate && !first) zero_pool(s,s->gradients,s->plan.parameters);
  if (!accumulate && first && !s->plan.spec.tied) zero_pool(s,s->gradients,g.v*g.d);
  xs=s->activations; cs=xs+(g.l+1)*g.nd; norm=cs+cache_slots(g,s->recompute)*g.la; logits=norm+g.nd;
  top=top_recomputed(g,s->recompute);
  cur=s->scratch; tmp=cur+g.nd; dn=tmp+g.nd; dq=dn+g.nd; dk=dq+g.nd;
  dv=dk+g.nk; dc=dv+g.nk; dg=dc+g.nd; du=dg+g.nf; dh=du+g.nf; dl=dh+g.nf;
  final=s->weights+g.v*g.d+g.l*g.lp; dfinal=s->gradients+g.v*g.d+g.l*g.lp;
  head=s->plan.spec.tied ? s->weights : final+g.d;
  dhead=s->plan.spec.tied ? s->gradients : dfinal+g.d;
  /* the rows' terms live at the arena's start (8n bytes within cur, tmp and dn: regions the
   * head's VJP writes only after they are added) */
  ce=loss_rows(s,g,logits,targets,dl,(unsigned char *)s->scratch);
  ran&=linear_backward(s,g.n,g.d,g.v,norm,head,dl,dn,dhead,0.0f,wb);
  /* each gamma VJP keeps its rows' 1/rms in dq, free until the layer's query VJP */
  rms_backward(s,g.n,g.d,xs+g.l*g.nd,final,dn,cur,dfinal,dq,gf);
  for (l=g.l;l-->0;) {
    layer p=params(s->weights+g.v*g.d+l*g.lp,g);
    layer grad=params(s->gradients+g.v*g.d+l*g.lp,g);
    float *slot=cs+slot_of(g,s->recompute,l)*g.la; cache a=activations(slot,g); float *x=xs+l*g.nd;
    /* REMAT: a recomputed layer's cache, from its input, into the shared slot -- except the highest,
     * whose forward left it there (the slot's other layers are below it); the scratch the backward
     * keeps between layers is not a value the forward writes */
    if (recomputes(s->recompute,l) && l!=top) {
      int seen=0;
      ran&=layer_forward(s,g,l,slot,x,0,&seen);
    }
    /* the hidden activation is a panel exactly when the forward wrote it as one (one predicate on
     * the same slot); the adjoints are panels when their own slots admit them, whichever the hidden
     * activation is (their GEMMs read the gate, the up projection and dh, never the hidden slot) */
    if (panels(s,g,a.hidden,a.hidden)) {
      bcir_tensor_mm_call cx,cw; bcir_tensor_bf16_panels pre[2];
      cx.ta=0; cx.tb=0; cx.m=g.n; cx.n=g.f; cx.k=g.d; cx.alpha=1.0f; cx.a=cur; cx.b=p.down;
      cx.beta=0.0f; cx.c=dh;
      cw.ta=1; cw.tb=0; cw.m=g.d; cw.n=g.f; cw.k=g.n; cw.alpha=1.0f; cw.a=cur; cw.b=NULL;
      cw.beta=wb; cw.c=grad.down;
      pre[0].a=NULL; pre[0].b=NULL; pre[1].a=NULL; pre[1].b=panel_of(a.hidden)+g.nf;
      ran&=gemm_group_with(s,&cx,1,&cw,1,pre);
      PANELS_SAW(2u);
    }
    else ran&=linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f,wb);
    if (panels(s,g,dg,du)) {
      bcir_tensor_mm_call cx[2],cw[2]; bcir_tensor_bf16_panels pre[4]; size_t q;
      ran&=bcir_tensor_swiglu_backward_panels_pool(s->pool,swiglu_v2(s),g.n,g.f,a.gate,a.up,dh,
          panel_of(dg),panel_of(dg)+g.nf,panel_of(du),panel_of(du)+g.nf);
      for (q=0;q<2;q++) {
        float *slot=q ? du : dg;
        cx[q].ta=0; cx[q].tb=0; cx[q].m=g.n; cx[q].n=g.d; cx[q].k=g.f; cx[q].alpha=1.0f; cx[q].a=NULL;
        cx[q].b=q ? p.up : p.gate; cx[q].beta=q ? 1.0f : 0.0f; cx[q].c=dn;
        cw[q].ta=1; cw[q].tb=0; cw[q].m=g.f; cw[q].n=g.d; cw[q].k=g.n; cw[q].alpha=1.0f; cw[q].a=NULL;
        cw[q].b=a.n2; cw[q].beta=wb; cw[q].c=q ? grad.up : grad.gate;
        pre[q].a=panel_of(slot); pre[q].b=NULL; pre[2+q].a=panel_of(slot)+g.nf; pre[2+q].b=NULL;
      }
      ran&=gemm_group_with(s,cx,2,cw,2,pre);
      PANELS_SAW(4u);
    }
    else {
      swiglu_backward(s,g.nf,a.gate,a.up,dh,dg,du);
      {
        size_t out[2]; const float *w[2],*d2[2]; float *dw[2];
        out[0]=g.f; out[1]=g.f; w[0]=p.gate; w[1]=p.up; d2[0]=dg; d2[1]=du; dw[0]=grad.gate; dw[1]=grad.up;
        ran&=linears_backward(s,g.n,g.d,a.n2,2,out,w,d2,dn,dw,0.0f,wb);
      }
    }
    /* NDT-RMS-ACC: the VJP adds its dx into the residual's adjoint, one rounded sum an element */
    rms_backward(s,g.n,g.d,a.x2,p.norm2,dn,cur,grad.norm2,dq,BCIR_TENSOR_RMSB_ADD|gf);
    ran&=linear_backward(s,g.n,g.d,g.d,a.ctx,p.o,cur,dc,grad.o,0.0f,wb);
    /* the layer's probabilities are read for the last time here: the in-place backward
     * leaves its score adjoint in their place (same dq/dk/dv as the reference, bit for bit), the
     * bf16 attention's leaves them as they were */
    ran&=attention_backward(s,g,a.q,a.k,a.v,a.p,dc,dq,dk,dv);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.h,g.hd,s->plan.spec.rope_base,dq,1);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.kh,g.hd,s->plan.spec.rope_base,dk,1);
    {
      size_t out[3]; const float *w[3],*d3[3]; float *dw[3];
      out[0]=g.d; out[1]=g.k; out[2]=g.k; w[0]=p.q; w[1]=p.k; w[2]=p.v;
      d3[0]=dq; d3[1]=dk; d3[2]=dv; dw[0]=grad.q; dw[1]=grad.k; dw[2]=grad.v;
      ran&=linears_backward(s,g.n,g.d,a.n1,3,out,w,d3,dn,dw,0.0f,wb);
    }
    rms_backward(s,g.n,g.d,x,p.norm1,dn,cur,grad.norm1,dq,BCIR_TENSOR_RMSB_ADD|gf);
  }
  scatter(s,g.n,g.d,tokens,cur,s->gradients);
  if (!ran) return BCIR_DT_CAPACITY;  /* a product that ran nowhere: its gradient is no gradient */
  if ((!(how&DT_DEFER) && !finite_pool(s,s->gradients,s->plan.parameters)) || !isfinite(ce))
    return BCIR_DT_NUMERIC;
  *loss=ce/(double)g.n; return BCIR_DT_OK;
}
int bcir_decoder_loss_backward(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t n,int accumulate,double *loss) {
  return loss_backward_with(s,tokens,targets,n,accumulate,loss,0u);
}
/* The AdamW candidate for one element, exactly as the update has always computed it: the
 * moments rounded to float before they are used, the step in double. The preflight and the
 * commit both evaluate it; contraction is off by the build contract, so every lane of a vector
 * variant performs the same rounded operations as the scalar loop. */
typedef struct adamw_terms { double scale,d1,d2,decay,lr,eps,b1,b2; } adamw_terms;
#if defined(__GNUC__) || defined(__clang__)
#define BCIR_DT_INLINE static inline __attribute__((always_inline))
#else
#define BCIR_DT_INLINE static inline
#endif
BCIR_DT_INLINE int adamw_preflight(const adamw_terms *t,size_t n,const float *gr,
    const float *wt,const float *m1,const float *m2) {
  /* One flag for every refusal: !(|x| <= FLT_MAX) is exactly "not finite or above FLT_MAX",
   * NaN included, so the loop has no early exit and no branch per element. A moment out of
   * float's range is refused, and only an in-range one is converted (C11 6.3.1.5 leaves the
   * conversion undefined outside Annex F). */
  size_t i; int bad=0;
  for (i=0;i<n;i++) {
    double g=gr[i]*t->scale,mc=t->d1*m1[i]+(1-t->d1)*g,vc=t->d2*m2[i]+(1-t->d2)*g*g,wc;
    int in=fabs(mc)<=FLT_MAX && fabs(vc)<=FLT_MAX;
    float m=in ? (float)mc : 0.0f,v=in ? (float)vc : 0.0f;
    wc=(double)wt[i]*t->decay-t->lr*((double)m/(1-t->b1))/(sqrt((double)v/(1-t->b2))+t->eps);
    bad|=!in | !(fabs(wc)<=FLT_MAX);
  }
  return bad;
}
BCIR_DT_INLINE void adamw_commit(const adamw_terms *t,size_t n,const float *gr,float *wt,
    float *m1,float *m2) {
  size_t i;
  for (i=0;i<n;i++) {
    double g=gr[i]*t->scale,mc=t->d1*m1[i]+(1-t->d1)*g,vc=t->d2*m2[i]+(1-t->d2)*g*g;
    float m=(float)mc,v=(float)vc;
    wt[i]=(float)((double)wt[i]*t->decay-t->lr*((double)m/(1-t->b1))/
        (sqrt((double)v/(1-t->b2))+t->eps));
    m1[i]=m; m2[i]=v;
  }
}
static int preflight_generic(const adamw_terms *t,size_t n,const float *gr,const float *wt,
    const float *m1,const float *m2) { return adamw_preflight(t,n,gr,wt,m1,m2); }
static void commit_generic(const adamw_terms *t,size_t n,const float *gr,float *wt,
    float *m1,float *m2) { adamw_commit(t,n,gr,wt,m1,m2); }
/* The same AVX2/AVX-512 variants bcir_tensor_mm carries; they need the build's
 * -fno-math-errno to vectorize sqrt, and are bit-identical with or without it. */
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(BCIR_TENSOR_NO_DISPATCH)
#define BCIR_DT_DISPATCH 1
#include <immintrin.h>
/* x/d for the divisor every element of a commit shares, rounded as the division rounds it
 * (NDT-UPDATE), given y = RN(1/d): q0 = x*y is within 1.5 ulp of x/d, one FMA correction brings it
 * within half an ulp and a vanishing term -- one of the two floating-point neighbours of x/d -- and
 * a second is Markstein's: for such a q and a y within a relative 2^-53 of 1/d, r = x - q*d is
 * exact and RN(q + r*y) is RN(x/d) (Handbook of Floating-Point Arithmetic, 2nd ed., Theorem 4.10),
 * for x a float and d in [2^-53, 1] -- one minus a bias power, so every operand stays far from
 * under- and overflow. The quotient has x's sign, which the corrections' sums lose for a zero x.
 * Only the instances below, whose FMA is an instruction, take it, so it is defined among them. */
BCIR_DT_INLINE double shared_reciprocal(double d) { return 1/d; }
BCIR_DT_INLINE double div_shared(double x,double d,double y) {
  double q0=x*y,r0=fma(-q0,d,x),q1=fma(r0,y,q0),r1=fma(-q1,d,x);
  return copysign(fma(r1,y,q1),x);
}
/* The commit with its two divisions by the bias corrections as div_shared: the same candidates,
 * bit for bit, for the instances whose FMA is an instruction (the divider's two divisions an
 * element were most of the commit's time); the third division and the root are the element's own. */
BCIR_DT_INLINE void adamw_commit_fma(const adamw_terms *t,size_t n,const float *gr,float *wt,
    float *m1,float *m2) {
  size_t i; double c1=1-t->b1,c2=1-t->b2,y1=shared_reciprocal(c1),y2=shared_reciprocal(c2);
  for (i=0;i<n;i++) {
    double g=gr[i]*t->scale,mc=t->d1*m1[i]+(1-t->d1)*g,vc=t->d2*m2[i]+(1-t->d2)*g*g;
    float m=(float)mc,v=(float)vc;
    wt[i]=(float)((double)wt[i]*t->decay-t->lr*div_shared((double)m,c1,y1)/
        (sqrt(div_shared((double)v,c2,y2))+t->eps));
    m1[i]=m; m2[i]=v;
  }
}
__attribute__((target("avx2"))) static int preflight_avx2(const adamw_terms *t,size_t n,
    const float *gr,const float *wt,const float *m1,const float *m2) {
  return adamw_preflight(t,n,gr,wt,m1,m2);
}
__attribute__((target("avx2"))) static void commit_avx2(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) { adamw_commit(t,n,gr,wt,m1,m2); }
__attribute__((target("avx2,fma"))) static void commit_avx2_fma(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) { adamw_commit_fma(t,n,gr,wt,m1,m2); }
__attribute__((target("avx512f"))) static int preflight_avx512(const adamw_terms *t,size_t n,
    const float *gr,const float *wt,const float *m1,const float *m2) {
  return adamw_preflight(t,n,gr,wt,m1,m2);
}
__attribute__((target("avx512f"))) static void commit_avx512(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) { adamw_commit_fma(t,n,gr,wt,m1,m2); }
#endif
/* The commit's variant: the GEMM's bits, and bit 8 where the AVX2 instance may take its FMAs (the
 * fused GEMM's variants carry AVX2 only with FMA). */
static int commit_variants(void) {
  return bcir_tensor_mm_variants()|(bcir_tensor_mm_fused_variants()&2 ? 8 : 0);
}
#ifdef BCIR_DT_DIVISION_PROBE
/* The commit's shared division alone, dispatched as the commit dispatches it, for the harness's
 * check of it against the division over every float dividend: a build defines
 * BCIR_DT_DIVISION_PROBE to have the symbol (the harness's own builds do; the library does not). */
#ifdef BCIR_DT_DISPATCH
__attribute__((target("avx512f"))) static void probe_avx512(size_t n,const float *x,double d,
    double y,double *q) { size_t i; for (i=0;i<n;i++) q[i]=div_shared((double)x[i],d,y); }
__attribute__((target("avx2,fma"))) static void probe_avx2_fma(size_t n,const float *x,double d,
    double y,double *q) { size_t i; for (i=0;i<n;i++) q[i]=div_shared((double)x[i],d,y); }
#endif
void bcir_decoder_division_probe(size_t n,const float *x,double d,double *q);
void bcir_decoder_division_probe(size_t n,const float *x,double d,double *q) {
  size_t i; int v=commit_variants();
  (void)v;
#ifdef BCIR_DT_DISPATCH
  if (v&4) { probe_avx512(n,x,d,shared_reciprocal(d),q); return; }
  if ((v&2) && (v&8)) { probe_avx2_fma(n,x,d,shared_reciprocal(d),q); return; }
#endif
  for (i=0;i<n;i++) q[i]=(double)x[i]/d;
}
#endif
static int adamw_preflight_variant(int v,const adamw_terms *t,size_t n,const float *gr,
    const float *wt,const float *m1,const float *m2) {
  (void)v;
#ifdef BCIR_DT_DISPATCH
  if (v&4) return preflight_avx512(t,n,gr,wt,m1,m2);
  if (v&2) return preflight_avx2(t,n,gr,wt,m1,m2);
#endif
  return preflight_generic(t,n,gr,wt,m1,m2);
}
static void adamw_commit_variant(int v,const adamw_terms *t,size_t n,const float *gr,float *wt,
    float *m1,float *m2) {
  (void)v;
#ifdef BCIR_DT_DISPATCH
  if (v&4) { commit_avx512(t,n,gr,wt,m1,m2); return; }
  if ((v&2) && (v&8)) { commit_avx2_fma(t,n,gr,wt,m1,m2); return; }
  if (v&2) { commit_avx2(t,n,gr,wt,m1,m2); return; }
#endif
  commit_generic(t,n,gr,wt,m1,m2);
}
/* The first pass and the preflight over the pool, in the reduce contract's chunks (NDT-REDUCE):
 * a task's finiteness verdict and maxima are exact whatever the partition, and its chunk's
 * squares summed in index order in double from zero are the reduce contract's partial; the caller
 * ORs the verdicts, takes the maxima and, under the contract, adds the partials in chunk order
 * from zero. Under the other contracts the norm stays the one index-order chain over every
 * element, serial. The chunk is a function of the count alone, so the state is the same at every
 * thread count; at most BCIR_DT_PASS_TASKS tasks, their results on the caller's stack. */
#define BCIR_DT_PASS_CHUNK 1024
#define BCIR_DT_PASS_TASKS 1024
size_t bcir_decoder_reduce_chunk(size_t n) {
  size_t chunk=BCIR_DT_PASS_CHUNK;
  while ((n+chunk-1)/chunk>BCIR_DT_PASS_TASKS) chunk*=2;
  return chunk;
}
typedef struct pass_part { double norm; float grmax,wmax,mmax,vmax; int bad; } pass_part;
typedef struct pass_job {
  const float *gr,*wt,*m1,*m2; size_t n,chunk,pair; double gs; int reduce,variant;
  pass_part *parts;
} pass_job;
/* One chunk's verdict and maxima, without a branch, so the instances below are vector loops: one
 * flag for every refusal, as the preflight's; the maxima as the largest bit patterns read as
 * int32 -- a non-negative float orders as its bits, and the one negative value the pass admits,
 * a -0.0 second moment, reads as INT32_MIN and never wins, as `x > max ? x : max` from +0.0 never
 * takes it -- so on every chunk the pass admits they are that loop's maxima, bit for bit (a chunk
 * it refuses has no maxima the update reads). */
BCIR_DT_INLINE void pass_span(size_t n,const float *gr,const float *wt,const float *m1,
    const float *m2,double gs,pass_part *p) {
  size_t i; int bad=0; int32_t bg=0,bw=0,bm=0,bv=0;
  for (i=0;i<n;i++) {
    float ag=fabsf(gr[i]),aw=fabsf(wt[i]),am=fabsf(m1[i]),av=m2[i]; int32_t xg,xw,xm,xv;
    bad|=!(fabs((double)gr[i]*gs)<=DBL_MAX) | !(aw<=FLT_MAX) | !(am<=FLT_MAX) | !(av>=0.0f) |
        !(av<=FLT_MAX);
    memcpy(&xg,&ag,4); memcpy(&xw,&aw,4); memcpy(&xm,&am,4); memcpy(&xv,&av,4);
    bg=xg>bg ? xg : bg; bw=xw>bw ? xw : bw; bm=xm>bm ? xm : bm; bv=xv>bv ? xv : bv;
  }
  p->bad=bad; memcpy(&p->grmax,&bg,4); memcpy(&p->wmax,&bw,4); memcpy(&p->mmax,&bm,4);
  memcpy(&p->vmax,&bv,4);
}
static void pass_generic(size_t n,const float *gr,const float *wt,const float *m1,const float *m2,
    double gs,pass_part *p) { pass_span(n,gr,wt,m1,m2,gs,p); }
#ifdef BCIR_DT_DISPATCH
__attribute__((target("avx2"))) static void pass_avx2(size_t n,const float *gr,const float *wt,
    const float *m1,const float *m2,double gs,pass_part *p) { pass_span(n,gr,wt,m1,m2,gs,p); }
__attribute__((target("avx512f"))) static void pass_avx512(size_t n,const float *gr,
    const float *wt,const float *m1,const float *m2,double gs,pass_part *p) {
  pass_span(n,gr,wt,m1,m2,gs,p);
}
/* Two whole chunks, n elements each and contiguous, under the reduce contract (NDT-UPDATE): both
 * chunks' flag and maxima a vector at a time, and each chunk's squares added to its own chain in
 * index order between the vector loads -- (double)g*gs squared and added in double from zero,
 * pass_task's chain -- so the chains run under the four streams' reads instead of after them, and
 * each under the other's latency. The vector flag is pass_span's element for element (each
 * comparison true for NaN where pass_span's is false, negated); the maxima are the int32 maxima of
 * the same bit patterns, the pair's in its first part. A chunk is 1,024 elements doubled
 * (bcir_decoder_reduce_chunk), so whole vectors. */
__attribute__((target("avx512f"))) static void pair_avx512(size_t n,const float *gr,
    const float *wt,const float *m1,const float *m2,double gs,pass_part *p,double *norms) {
  const __m512i magnitude=_mm512_set1_epi32(0x7fffffff);
  const __m512 top=_mm512_set1_ps(FLT_MAX),zero=_mm512_setzero_ps();
  const __m512d scale=_mm512_set1_pd(gs),dtop=_mm512_set1_pd(DBL_MAX);
  __m512i bg=_mm512_setzero_si512(),bw=bg,bm=bg,bv=bg; __mmask16 bad=0; __mmask8 gbad=0;
  double n0=0.0,n1=0.0,sq[32]; int32_t lane[16],mx[4]={0,0,0,0}; size_t i,c,k;
  for (i=0;i<n;i+=16) {
    for (c=0;c<2;c++) {
      size_t at=c*n+i;
      __m512 g=_mm512_loadu_ps(gr+at),w=_mm512_loadu_ps(wt+at),a=_mm512_loadu_ps(m1+at);
      __m512 v=_mm512_loadu_ps(m2+at);
      __m512i ig=_mm512_and_si512(_mm512_castps_si512(g),magnitude);
      __m512i iw=_mm512_and_si512(_mm512_castps_si512(w),magnitude);
      __m512i ia=_mm512_and_si512(_mm512_castps_si512(a),magnitude);
      __m512d lo=_mm512_mul_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(g)),scale);
      __m512d hi=_mm512_mul_pd(_mm512_cvtps_pd(_mm256_castpd_ps(
          _mm512_extractf64x4_pd(_mm512_castps_pd(g),1))),scale);
      gbad|=_mm512_cmp_pd_mask(_mm512_abs_pd(lo),dtop,_CMP_NLE_UQ)|
          _mm512_cmp_pd_mask(_mm512_abs_pd(hi),dtop,_CMP_NLE_UQ);
      bad|=_mm512_cmp_ps_mask(_mm512_castsi512_ps(iw),top,_CMP_NLE_UQ)|
          _mm512_cmp_ps_mask(_mm512_castsi512_ps(ia),top,_CMP_NLE_UQ)|
          _mm512_cmp_ps_mask(v,zero,_CMP_NGE_UQ)|_mm512_cmp_ps_mask(v,top,_CMP_NLE_UQ);
      bg=_mm512_max_epi32(bg,ig); bw=_mm512_max_epi32(bw,iw); bm=_mm512_max_epi32(bm,ia);
      bv=_mm512_max_epi32(bv,_mm512_castps_si512(v));
      _mm512_storeu_pd(sq+16*c,_mm512_mul_pd(lo,lo)); _mm512_storeu_pd(sq+16*c+8,_mm512_mul_pd(hi,hi));
    }
    for (k=0;k<16;k++) { n0+=sq[k]; n1+=sq[16+k]; }
  }
  _mm512_storeu_si512(lane,bg); for (k=0;k<16;k++) mx[0]=lane[k]>mx[0] ? lane[k] : mx[0];
  _mm512_storeu_si512(lane,bw); for (k=0;k<16;k++) mx[1]=lane[k]>mx[1] ? lane[k] : mx[1];
  _mm512_storeu_si512(lane,bm); for (k=0;k<16;k++) mx[2]=lane[k]>mx[2] ? lane[k] : mx[2];
  _mm512_storeu_si512(lane,bv); for (k=0;k<16;k++) mx[3]=lane[k]>mx[3] ? lane[k] : mx[3];
  p->bad=bad!=0 || gbad!=0; memcpy(&p->grmax,&mx[0],4); memcpy(&p->wmax,&mx[1],4);
  memcpy(&p->mmax,&mx[2],4); memcpy(&p->vmax,&mx[3],4); norms[0]=n0; norms[1]=n1;
}
__attribute__((target("avx2"))) static void pair_avx2(size_t n,const float *gr,const float *wt,
    const float *m1,const float *m2,double gs,pass_part *p,double *norms) {
  const __m256i magnitude=_mm256_set1_epi32(0x7fffffff);
  const __m256 top=_mm256_set1_ps(FLT_MAX),zero=_mm256_setzero_ps();
  const __m256d scale=_mm256_set1_pd(gs),dtop=_mm256_set1_pd(DBL_MAX),sign=_mm256_set1_pd(-0.0);
  __m256i bg=_mm256_setzero_si256(),bw=bg,bm=bg,bv=bg; __m256 bad=_mm256_setzero_ps();
  __m256d gbad=_mm256_setzero_pd(); double n0=0.0,n1=0.0,sq[16];
  int32_t lane[8],mx[4]={0,0,0,0}; size_t i,c,k;
  for (i=0;i<n;i+=8) {
    for (c=0;c<2;c++) {
      size_t at=c*n+i;
      __m256 g=_mm256_loadu_ps(gr+at),w=_mm256_loadu_ps(wt+at),a=_mm256_loadu_ps(m1+at);
      __m256 v=_mm256_loadu_ps(m2+at);
      __m256i ig=_mm256_and_si256(_mm256_castps_si256(g),magnitude);
      __m256i iw=_mm256_and_si256(_mm256_castps_si256(w),magnitude);
      __m256i ia=_mm256_and_si256(_mm256_castps_si256(a),magnitude);
      __m256d lo=_mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(g)),scale);
      __m256d hi=_mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(g,1)),scale);
      gbad=_mm256_or_pd(gbad,_mm256_or_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign,lo),dtop,_CMP_NLE_UQ),
          _mm256_cmp_pd(_mm256_andnot_pd(sign,hi),dtop,_CMP_NLE_UQ)));
      bad=_mm256_or_ps(bad,_mm256_or_ps(_mm256_or_ps(
          _mm256_cmp_ps(_mm256_castsi256_ps(iw),top,_CMP_NLE_UQ),
          _mm256_cmp_ps(_mm256_castsi256_ps(ia),top,_CMP_NLE_UQ)),_mm256_or_ps(
          _mm256_cmp_ps(v,zero,_CMP_NGE_UQ),_mm256_cmp_ps(v,top,_CMP_NLE_UQ))));
      bg=_mm256_max_epi32(bg,ig); bw=_mm256_max_epi32(bw,iw); bm=_mm256_max_epi32(bm,ia);
      bv=_mm256_max_epi32(bv,_mm256_castps_si256(v));
      _mm256_storeu_pd(sq+8*c,_mm256_mul_pd(lo,lo)); _mm256_storeu_pd(sq+8*c+4,_mm256_mul_pd(hi,hi));
    }
    for (k=0;k<8;k++) { n0+=sq[k]; n1+=sq[8+k]; }
  }
  _mm256_storeu_si256((__m256i *)lane,bg); for (k=0;k<8;k++) mx[0]=lane[k]>mx[0] ? lane[k] : mx[0];
  _mm256_storeu_si256((__m256i *)lane,bw); for (k=0;k<8;k++) mx[1]=lane[k]>mx[1] ? lane[k] : mx[1];
  _mm256_storeu_si256((__m256i *)lane,bm); for (k=0;k<8;k++) mx[2]=lane[k]>mx[2] ? lane[k] : mx[2];
  _mm256_storeu_si256((__m256i *)lane,bv); for (k=0;k<8;k++) mx[3]=lane[k]>mx[3] ? lane[k] : mx[3];
  p->bad=_mm256_movemask_ps(bad)!=0 || _mm256_movemask_pd(gbad)!=0;
  memcpy(&p->grmax,&mx[0],4); memcpy(&p->wmax,&mx[1],4); memcpy(&p->mmax,&mx[2],4);
  memcpy(&p->vmax,&mx[3],4); norms[0]=n0; norms[1]=n1;
}
#endif
/* A task is a chunk, or j->pair of them where first_pass pairs them (the reduce contract on a
 * vector variant, from the contract and the variant alone): two whole chunks take pair_*, their
 * flag and maxima in the first chunk's part and zeros in the second's, which the caller's fold
 * reads as it read the chunks' own; any other chunk the loop below. Every chunk's partial is its
 * chain's either way. */
static void pass_task(void *ctx,size_t task) {
  pass_job *j=(pass_job *)ctx; size_t c,i,chunks=(j->n+j->chunk-1)/j->chunk;
  size_t c0=task*j->pair,c1=c0+j->pair<chunks ? c0+j->pair : chunks;
  const float *gr=j->gr; double gs=j->gs;
#ifdef BCIR_DT_DISPATCH
  if (j->pair==2 && c1-c0==2 && c1*j->chunk<=j->n) {
    size_t i0=c0*j->chunk; double norms[2]; pass_part p;
    if (j->variant&4) pair_avx512(j->chunk,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p,norms);
    else pair_avx2(j->chunk,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p,norms);
    p.norm=norms[0]; j->parts[c0]=p;
    p.bad=0; p.grmax=p.wmax=p.mmax=p.vmax=0.0f; p.norm=norms[1]; j->parts[c0+1]=p;
    return;
  }
#endif
  for (c=c0;c<c1;c++) {
    pass_part p; size_t i0=c*j->chunk,i1=i0+j->chunk<j->n ? i0+j->chunk : j->n;
#ifdef BCIR_DT_DISPATCH
    if (j->variant&4) pass_avx512(i1-i0,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p);
    else if (j->variant&2) pass_avx2(i1-i0,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p);
    else
#endif
    pass_generic(i1-i0,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p);
    p.norm=0.0;
    if (j->reduce) for (i=i0;i<i1;i++) { double g=(double)gr[i]*gs; p.norm+=g*g; }
    j->parts[c]=p;
  }
}
typedef struct preflight_job {
  const adamw_terms *t; size_t n,chunk; const float *gr,*wt,*m1,*m2; int variant; pass_part *parts;
} preflight_job;
static void preflight_task(void *ctx,size_t task) {
  preflight_job *j=(preflight_job *)ctx;
  size_t i0=task*j->chunk,i1=i0+j->chunk<j->n ? i0+j->chunk : j->n;
  j->parts[task].bad=adamw_preflight_variant(j->variant,j->t,i1-i0,j->gr+i0,j->wt+i0,j->m1+i0,
      j->m2+i0);
}
/* The commit over the pool: elements in ranges of BCIR_DT_CHUNK, each element's update its own
 * and the variant chosen once, so the partition cannot reach the arithmetic. */
#define BCIR_DT_CHUNK 65536
typedef struct commit_job {
  const adamw_terms *t; size_t n; const float *gr; float *wt,*m1,*m2; int variant;
} commit_job;
static void commit_task(void *ctx,size_t task) {
  commit_job *j=(commit_job *)ctx;
  size_t i0=task*BCIR_DT_CHUNK,i1=i0+BCIR_DT_CHUNK<j->n ? i0+BCIR_DT_CHUNK : j->n;
  adamw_commit_variant(j->variant,j->t,i1-i0,j->gr+i0,j->wt+i0,j->m1+i0,j->m2+i0);
}
static void adamw_commit_pool(bcir_tensor_pool *pool,const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) {
  commit_job j;
  j.t=t; j.n=n; j.gr=gr; j.wt=wt; j.m1=m1; j.m2=m2; j.variant=commit_variants();
  bcir_tensor_pool_run(pool,commit_task,&j,(n+BCIR_DT_CHUNK-1)/BCIR_DT_CHUNK);
}
/* Whether the magnitudes the first pass saw prove that no candidate can leave float's range,
 * so the preflight would admit every element and may be skipped. With G = max|g| after scaling,
 * M = max|m1|, V = max m2 and W = max|w|: |mc| <= M + G and |vc| <= V + G*G (convex combinations
 * of non-negative magnitudes), |m| <= 2|mc| after rounding to float, the step's denominator is
 * at least eps, and |wc| <= W|decay| + lr*(|m|/(1-b1))/eps. Every rounded operation is
 * monotone in its operands, so evaluating those bounds in the same operation order bounds the
 * rounded values; any overflow in the bound itself (inf, or NaN from it) fails the comparisons
 * and keeps the preflight. Half of FLT_MAX is left as margin throughout. */
static int adamw_bounded(const adamw_terms *t,double grmax,double mmax,double vmax,double wmax) {
  double half=0.5*(double)FLT_MAX,g=grmax*t->scale,mb=mmax>g ? mmax : g,gg=g*g;
  double vb=vmax>gg ? vmax : gg,m=4.0*mb,step=t->lr*(m/(1-t->b1))/t->eps;
  double w=wmax*fabs(t->decay);
  return 2.0*mb<=half && 2.0*vb<=half && step<=0.5*half && w<=0.5*half && w+step<=half;
}
/* The first pass, one task per chunk: every element finite (the second moment also not
 * negative) and the largest magnitudes adamw_bounded reads, exact whatever the partition; the
 * gradient norm's square the contract's -- the chunks' partials added in chunk order under the
 * reduce contract, else summed in double in index order over every element, as always. One flag
 * per task replaces the early exit: the verdict and the untouched state are the same. */
typedef struct pass_result { double norm2,grmax,wmax,mmax,vmax; size_t chunk,tasks; } pass_result;
static int first_pass(bcir_decoder_state *s,double gs,pass_part *parts,pass_result *out) {
  pass_job pj; size_t i; int bad=0; double norm=0.0;
  pj.gr=s->gradients; pj.wt=s->weights; pj.m1=s->moment1; pj.m2=s->moment2;
  pj.n=s->plan.parameters; pj.chunk=bcir_decoder_reduce_chunk(pj.n); pj.gs=gs;
  pj.reduce=(s->contract&BCIR_DT_CONTRACT_REDUCE)!=0; pj.parts=parts;
  pj.variant=bcir_tensor_mm_variants();
  out->tasks=(pj.n+pj.chunk-1)/pj.chunk; out->chunk=pj.chunk;
  out->grmax=out->wmax=out->mmax=out->vmax=0.0;
  pj.pair=1;
#ifdef BCIR_DT_DISPATCH
  if (pj.reduce && (pj.variant&6)) pj.pair=2;
#endif
  bcir_tensor_pool_run(s->pool,pass_task,&pj,(out->tasks+pj.pair-1)/pj.pair);
  for (i=0;i<out->tasks;i++) {
    const pass_part *p=parts+i;
    bad|=p->bad;
    out->grmax=p->grmax>out->grmax ? p->grmax : out->grmax;
    out->wmax=p->wmax>out->wmax ? p->wmax : out->wmax;
    out->mmax=p->mmax>out->mmax ? p->mmax : out->mmax;
    out->vmax=p->vmax>out->vmax ? p->vmax : out->vmax;
  }
  if (bad) return BCIR_DT_NUMERIC;
  if (pj.reduce) { for (i=0;i<out->tasks;i++) norm+=parts[i].norm; }
  else { const float *gr=s->gradients; for (i=0;i<pj.n;i++) { double g=(double)gr[i]*gs; norm+=g*g; } }
  out->norm2=norm;
  return BCIR_DT_OK;
}
int bcir_decoder_gradient_norm(bcir_decoder_state *s,double gs,double *squared) {
  pass_part parts[BCIR_DT_PASS_TASKS]; pass_result pass; int rc=valid(s);
  if (rc) return rc;
  if (!isfinite(gs) || gs<=0 || !external(s,squared,sizeof(*squared)) ||
      (uintptr_t)squared%_Alignof(double)) return BCIR_DT_INVALID;
  rc=first_pass(s,gs,parts,&pass);
  if (rc) return rc;
  *squared=pass.norm2; return BCIR_DT_OK;
}
static int optimizer_valid(const bcir_decoder_adamw *o) {
  return o && isfinite(o->lr) && o->lr>=0 && isfinite(o->beta1) && o->beta1>=0 && o->beta1<1 &&
    isfinite(o->beta2) && o->beta2>=0 && o->beta2<1 && isfinite(o->epsilon) && o->epsilon>0 &&
    isfinite(o->weight_decay) && o->weight_decay>=0 && isfinite(o->grad_clip) && o->grad_clip>=0;
}
/* The update a run calls with its step's closing gradient scan deferred to it (NDT-FACTS-CALL:
 * the first pass's verdict on every gradient element is the scan's, gs being 1/accumulation):
 * a refusal of the update's own arguments is reported only after that scan, so a step whose
 * gradient is not finite reports the backward's verdict first, as the separate calls do.
 * `placed` is the run's view of the transient arenas (valid_with): a program's run checks the
 * state as its run_args did -- the arenas as lent -- and the update reads neither arena. */
static int update_with(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    double gs,double *grad_norm,int deferred,int placed) {
  size_t i; double norm,scale,b1,b2; int rc=valid_with(s,placed);
  bcir_decoder_adamw options; pass_part parts[BCIR_DT_PASS_TASKS]; pass_result pass;
  if (rc) return rc;
  if (!o) return deferred && !finite_pool(s,s->gradients,s->plan.parameters) ? BCIR_DT_NUMERIC :
    BCIR_DT_INVALID;
  options=*o; o=&options;
  if (!optimizer_valid(o) || !isfinite(gs) || gs<=0 || s->step==UINT64_MAX ||
      !isfinite(s->beta1_power) || !isfinite(s->beta2_power) ||
      s->beta1_power<0 || s->beta1_power>1 || s->beta2_power<0 || s->beta2_power>1 ||
      (!s->step && (s->beta1_power!=1 || s->beta2_power!=1)) ||
      !external_in(s,placed,grad_norm,sizeof(*grad_norm)) ||
      (uintptr_t)grad_norm%_Alignof(double))
    return deferred && !finite_pool(s,s->gradients,s->plan.parameters) ? BCIR_DT_NUMERIC :
      BCIR_DT_INVALID;
  rc=first_pass(s,gs,parts,&pass);
  if (rc) return rc;
  norm=sqrt(pass.norm2); if (!isfinite(norm)) return BCIR_DT_NUMERIC;
  scale=gs;
  if (o->grad_clip>0 && norm>o->grad_clip) scale*=o->grad_clip/(norm+1e-6);
  b1=s->beta1_power*o->beta1; b2=s->beta2_power*o->beta2;
  /* Round moments before using them; preflight every candidate before any commit. The commit
   * recomputes the candidates the preflight admitted, value for value (its shared divisions as
   * div_shared's corrected products where the FMA is an instruction); the preflight is skipped
   * only when adamw_bounded proves it would admit them all. */
  {
    adamw_terms t;
    t.scale=scale; t.d1=o->beta1; t.d2=o->beta2; t.decay=1-o->lr*o->weight_decay;
    t.lr=o->lr; t.eps=o->epsilon; t.b1=b1; t.b2=b2;
    if (!adamw_bounded(&t,pass.grmax,pass.mmax,pass.vmax,pass.wmax)) {
      preflight_job fj; int bad=0;
      fj.t=&t; fj.n=s->plan.parameters; fj.chunk=pass.chunk; fj.gr=s->gradients;
      fj.wt=s->weights; fj.m1=s->moment1; fj.m2=s->moment2;
      fj.variant=bcir_tensor_mm_variants(); fj.parts=parts;
      bcir_tensor_pool_run(s->pool,preflight_task,&fj,pass.tasks);
      for (i=0;i<pass.tasks;i++) bad|=parts[i].bad;
      if (bad) return BCIR_DT_NUMERIC;
    }
    adamw_commit_pool(s->pool,&t,s->plan.parameters,s->gradients,s->weights,s->moment1,
        s->moment2);
  }
  s->step++; s->beta1_power=b1; s->beta2_power=b2; *grad_norm=norm;
  return BCIR_DT_OK;
}
int bcir_decoder_update(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    double gs,double *grad_norm) {
  return update_with(s,o,gs,grad_norm,0,0);
}
int bcir_decoder_run(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    const uint32_t *tokens,const uint32_t *targets,size_t count,
    const double *rates,size_t steps,bcir_decoder_event *events,size_t capacity,size_t *completed) {
  return bcir_decoder_run_accum(s,o,tokens,targets,count,rates,steps,1,events,capacity,completed);
}
/* A run's arguments, the multi-step run's and a program's alike: the state, the optimizer, the
 * tokens and targets of steps x accumulation micro-batches, the rates, events and count -- every
 * output apart from every input. BCIR_DT_OK when the run may start. */
static int run_args(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    const uint32_t *tokens,const uint32_t *targets,size_t count,
    const double *rates,size_t steps,size_t accumulation,bcir_decoder_event *events,
    size_t capacity,size_t *completed,int placed) {
  size_t n,total,batches,i; int rc=valid_with(s,placed);
  if (rc) return rc;
  if (!optimizer_valid(o) || !accumulation || !completed || (uintptr_t)completed%_Alignof(size_t) ||
      !external_in(s,placed,completed,sizeof(*completed))) return BCIR_DT_INVALID;
  n=(size_t)s->plan.spec.batch*s->plan.spec.time;
  if (!product(steps,accumulation,&batches) || !product(n,batches,&total) ||
      steps>SIZE_MAX/sizeof(*events) || steps>SIZE_MAX/sizeof(*rates) ||
      s->step>UINT64_MAX-steps) return BCIR_DT_CAPACITY;
  if (count!=total || capacity<steps) return BCIR_DT_CAPACITY;
  if (steps && (!ids_in(s,placed,tokens,total) || !ids_in(s,placed,targets,total) ||
      !external_in(s,placed,events,steps*sizeof(*events)) ||
      (uintptr_t)events%_Alignof(bcir_decoder_event) || (rates &&
      (!external_in(s,placed,rates,steps*sizeof(*rates)) || (uintptr_t)rates%_Alignof(double)))))
    return BCIR_DT_INVALID;
  /* Outputs cannot overwrite a future input, rate or completion count. */
  if (steps && (overlap(events,steps*sizeof(*events),tokens,total*sizeof(*tokens)) ||
      overlap(events,steps*sizeof(*events),targets,total*sizeof(*targets)) ||
      (rates && overlap(events,steps*sizeof(*events),rates,steps*sizeof(*rates))) ||
      overlap(completed,sizeof(*completed),events,steps*sizeof(*events)) ||
      (rates && overlap(completed,sizeof(*completed),rates,steps*sizeof(*rates))) ||
      overlap(completed,sizeof(*completed),tokens,total*sizeof(*tokens)) ||
      overlap(completed,sizeof(*completed),targets,total*sizeof(*targets)))) return BCIR_DT_INVALID;
  if (rates) for (i=0;i<steps;i++) if (!isfinite(rates[i]) || rates[i]<0) return BCIR_DT_INVALID;
  return BCIR_DT_OK;
}
int bcir_decoder_run_accum(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    const uint32_t *tokens,const uint32_t *targets,size_t count,
    const double *rates,size_t steps,size_t accumulation,bcir_decoder_event *events,
    size_t capacity,size_t *completed) {
  size_t n,i,j; bcir_decoder_adamw opt;
  int rc=run_args(s,o,tokens,targets,count,rates,steps,accumulation,events,capacity,completed,0);
  if (rc) return rc;
  n=(size_t)s->plan.spec.batch*s->plan.spec.time;
  opt=*o; *completed=0;
  for (i=0;i<steps;i++) {
    double loss=0.0,norm;
    for (j=0;j<accumulation;j++) {
      double part; size_t off=(i*accumulation+j)*n;
      rc=loss_backward_with(s,tokens+off,targets+off,n,j!=0,&part,
          (j ? DT_KNOWN : i ? DT_COMMITTED : 0u)|(j+1==accumulation ? DT_DEFER : 0u));
      if (rc) return rc;
      loss+=part/(double)accumulation;
    }
    if (rates) opt.lr=rates[i];
    rc=update_with(s,&opt,1.0/(double)accumulation,&norm,1,0); if (rc) return rc;
    events[i].loss=loss; events[i].grad_norm=norm; events[i].step=s->step; *completed=i+1;
  }
  return BCIR_DT_OK;
}
/* PX-2 (PACK): the run from a launch program (bcir_decoder_train.h). The check walks the program
 * once with what the run will know at each record -- the micro-batch it belongs to, whether it has
 * its loss, whether the weights are known finite -- and refuses the whole program at its first
 * fault; the run walks it again, each record the call the step makes for it. An operand's use is
 * read, written, both (its kernel's in-place operand: one slot) or a panel, and its job's spans
 * are disjoint wherever one is written, a chain's shared C excepted (bcir_tensor.h). */
#define PX_IN ((1u<<1)|(1u<<2)|(1u<<5)|(1u<<6))  /* a kernel reads weights, gradients, acts, scratch */
#define PX_OUT ((1u<<2)|(1u<<5)|(1u<<6))         /* and writes gradients, activations, scratch */
#define PX_EXT (1u<<8)                           /* the micro-batch's tokens or targets */
#define PX_ONE 0x3f800000u                        /* 1.0f: the step's alpha, an accumulating beta */
#define PX_OPS (2*BCIR_TENSOR_MM_GROUP*3)
enum { PX_R=1u, PX_W=2u, PX_RW=3u, PX_PANEL=5u };
typedef struct px_op { unsigned space,use,chain; size_t off,bytes; } px_op;
typedef struct px_ctx {
  bcir_decoder_state *s; geometry g; unsigned char *base[8]; size_t len[8];
  uint64_t eps; unsigned v2,gc,attn,attn_vjp;
} px_ctx;
static void px_init(px_ctx *c,bcir_decoder_state *s) {
  float eps=(float)s->plan.spec.rms_eps; uint32_t bits; unsigned i;
  c->s=s; c->g=geom(&s->plan.spec); memcpy(&bits,&eps,sizeof bits); c->eps=bits;
  c->base[0]=NULL; c->len[0]=0;
  c->base[1]=(unsigned char *)s->weights; c->base[2]=(unsigned char *)s->gradients;
  c->base[3]=(unsigned char *)s->moment1; c->base[4]=(unsigned char *)s->moment2;
  for (i=1;i<5;i++) c->len[i]=s->plan.parameters*sizeof(float);
  c->base[5]=(unsigned char *)s->activations; c->len[5]=transient_bytes(s,1,0);
  c->base[6]=(unsigned char *)s->scratch; c->len[6]=transient_bytes(s,1,1);
  c->base[7]=(unsigned char *)s->workspace;
  c->len[7]=(s->contract&BCIR_DT_CONTRACT_BF16) ? s->workspace_bytes : 0;
  c->v2=(unsigned)swiglu_v2(s); c->gc=(unsigned)gemm_contract(s);
  c->attn=attention_contract(s)|((s->contract&BCIR_DT_CONTRACT_ATTNBF16) ? 256u : 0u);
  c->attn_vjp=(s->contract&BCIR_DT_CONTRACT_ATTNBF16) ? 256u : attention_contract(s);
}
static int px_fits(uint64_t x) {
#if SIZE_MAX < UINT64_MAX
  return x<=SIZE_MAX;
#else
  (void)x; return 1;
#endif
}
/* a*b*c*e, or 0 for a factor 0 or a product past size_t */
static size_t px_size(size_t a,size_t b,size_t c,size_t e) {
  size_t x;
  return product(a,b,&x) && product(x,c,&x) && product(x,e,&x) ? x : 0;
}
/* Operand slot q of record w: `bytes` bytes (0: the slot is absent, all three words 0) of a space
 * `mask` admits -- inside it at an offset a multiple of 4, or the micro-batch's tokens or targets
 * whole -- kept in ops with its use. */
static int px_slot(const px_ctx *c,const uint64_t *w,unsigned q,unsigned mask,size_t bytes,
    unsigned use,px_op *ops,size_t *n) {
  const uint64_t *r=w+11+3*q;
  if (!bytes) return !r[0] && !r[1] && !r[2];
  if (r[0]>8 || !((mask>>r[0])&1u) || r[2]!=bytes) return 0;
  if (r[0]==8) return !r[1];
  if (r[1]%4 || r[2]>c->len[r[0]] || r[1]>c->len[r[0]]-r[2]) return 0;
  ops[*n].space=(unsigned)r[0]; ops[*n].use=use; ops[*n].chain=0;
  ops[*n].off=(size_t)r[1]; ops[*n].bytes=bytes; (*n)++;
  return 1;
}
/* the workspace slot q: the state's whole bf16 workspace where the call packs into it, else absent */
static int px_ws(const px_ctx *c,const uint64_t *w,unsigned q,int packs) {
  const uint64_t *r=w+11+3*q;
  if (!packs) return !r[0] && !r[1] && !r[2];
  return r[0]==7 && !r[1] && c->len[7] && r[2]==c->len[7];
}
/* the dimensions from `dims` on, the slots from `slots` on and the five spare words: zero (the two
 * dispatch words are px_dispatch's) */
static int px_rest(const uint64_t *w,unsigned dims,unsigned slots) {
  unsigned i;
  for (i=3+dims;i<11;i++) if (w[i]) return 0;
  for (i=11+3*slots;i<BCIR_DT_RECORD_WORDS;i++)
    if (w[i] && i!=BCIR_DT_WORD_INSTANCE && i!=BCIR_DT_WORD_WORKERS) return 0;
  return 1;
}
/* PX-3b (DISPATCH): the instances a record's job may name -- a mask of bcir_tensor.h's bits: a GEMM
 * job of BCIR's entries its contract's family, an attention its products' (the bf16 family under
 * the bf16 attention, the fused GEMM's under ATTN-FMA, v1's otherwise), every other record none */
static int px_instances(const px_ctx *c,const uint64_t *w) {
  const bcir_decoder_state *s=c->s;
  if (w[0]==BCIR_DT_REC_GEMM)
    return (w[2]&15u)==7u ? 0 : bcir_tensor_mm_contract_variants(gemm_contract(s));
  if (w[0]==BCIR_DT_REC_ATTN || w[0]==BCIR_DT_REC_ATTN_VJP)
    return (s->contract&BCIR_DT_CONTRACT_ATTNBF16) ? bcir_tensor_mm_bf16_variants() :
      (s->contract&BCIR_DT_CONTRACT_ATTN) ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
  return 0;
}
/* a job's two dispatch words, alike in each of its records: the instance 0 or one bit its family
 * admits, the workers 0 or 1 .. the pool's threads (0 on a CHECK) */
static int px_dispatch(const px_ctx *c,const uint64_t *w,size_t job) {
  uint64_t v=w[BCIR_DT_WORD_INSTANCE],k=w[BCIR_DT_WORD_WORKERS]; size_t q;
  int mask=px_instances(c,w);
  if (v && ((v&(v-1)) || !((uint64_t)(unsigned)mask&v))) return 0;  /* one bit of the family's */
  if (k>(uint64_t)bcir_tensor_pool_threads(c->s->pool) || (k && w[0]==BCIR_DT_REC_CHECK)) return 0;
  for (q=1;q<job;q++)
    if (w[q*BCIR_DT_RECORD_WORDS+BCIR_DT_WORD_INSTANCE]!=v ||
        w[q*BCIR_DT_RECORD_WORDS+BCIR_DT_WORD_WORKERS]!=k) return 0;
  return 1;
}
/* no span of a record (or of a GEMM job) that is written meets another of its spans, a chain's
 * shared C excepted, and no two panels meet (bcir_tensor.h: a panel is never shared) */
static int px_apart(const px_op *ops,size_t n) {
  size_t i,j;
  for (i=0;i<n;i++) for (j=i+1;j<n;j++) {
    const px_op *a=ops+i,*b=ops+j;
    if (a->space!=b->space || !(a->off<b->off+b->bytes && b->off<a->off+a->bytes)) continue;
    if (((a->use|b->use)&PX_W) && !(a->chain && b->chain)) return 0;
    if (a->use==PX_PANEL && b->use==PX_PANEL) return 0;
  }
  return 1;
}
/* A record that is no GEMM and not the host's: its grammar, operands and the state it runs under
 * (a contract field the state's, a provider's call only with the provider, BCIR's SiLU only
 * without one). */
static int px_kernel(const px_ctx *c,const uint64_t *w,px_op *ops,size_t *n) {
  const bcir_decoder_state *s=c->s; const uint64_t f=w[2]; size_t d[8],x,y,z; unsigned i;
  *n=0;
  for (i=0;i<8;i++) { if (!px_fits(w[3+i])) return 0; d[i]=(size_t)w[3+i]; }
  switch (w[0]) {
  case BCIR_DT_REC_SCAN: case BCIR_DT_REC_ZERO:  /* n | x */
    x=px_size(d[0],1,1,4);
    return !f && x && px_rest(w,1,1) && (w[0]==BCIR_DT_REC_SCAN ?
        px_slot(c,w,0,PX_IN,x,PX_R,ops,n) : px_slot(c,w,0,PX_OUT,x,PX_W,ops,n));
  case BCIR_DT_REC_EMBED:  /* n, d | tokens, the table, x */
    x=px_size(d[0],d[1],1,4); y=px_size(c->g.v,d[1],1,4);
    return !f && d[0]==c->g.n && x && y && px_rest(w,2,3) &&
        px_slot(c,w,0,PX_EXT,px_size(d[0],1,1,4),PX_R,ops,n) &&
        px_slot(c,w,1,PX_IN,y,PX_R,ops,n) && px_slot(c,w,2,PX_OUT,x,PX_W,ops,n);
  case BCIR_DT_REC_RMS:  /* rows, width, eps | x, gamma, y */
    x=px_size(d[0],d[1],1,4); y=px_size(d[1],1,1,4);
    return !f && w[5]==c->eps && x && px_rest(w,3,3) && px_slot(c,w,0,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,1,PX_IN,y,PX_R,ops,n) && px_slot(c,w,2,PX_OUT,x,PX_W,ops,n);
  case BCIR_DT_REC_ROPE:  /* batch, time, heads, dim (even), inverse | x in place */
    x=px_size(px_size(d[0],d[1],d[2],1),d[3],1,4);
    return !f && x && !(d[3]%2) && d[4]<=1 && px_rest(w,5,1) &&
        px_slot(c,w,0,PX_OUT,x,PX_RW,ops,n);
  case BCIR_DT_REC_ATTN: case BCIR_DT_REC_ATTN_VJP: {
    /* batch, time, heads, kv heads (dividing heads), dim | q k v p y, or q k v p dy dq dk dv; ws */
    int vjp=w[0]==BCIR_DT_REC_ATTN_VJP,bf=(s->contract&BCIR_DT_CONTRACT_ATTNBF16)!=0;
    x=px_size(px_size(d[0],d[1],d[2],1),d[4],1,4); y=px_size(px_size(d[0],d[1],d[3],1),d[4],1,4);
    z=px_size(px_size(d[0],d[2],d[1],1),d[1],1,4);
    if (f!=(vjp ? c->attn_vjp : c->attn) || !x || !y || !z || d[2]%d[3] || !px_rest(w,5,9) ||
        !px_slot(c,w,0,PX_IN,x,PX_R,ops,n) || !px_slot(c,w,1,PX_IN,y,PX_R,ops,n) ||
        !px_slot(c,w,2,PX_IN,y,PX_R,ops,n) || !px_ws(c,w,8,bf)) return 0;
    if (!vjp)
      return px_slot(c,w,3,PX_OUT,z,PX_W,ops,n) && px_slot(c,w,4,PX_OUT,x,PX_W,ops,n) &&
          px_slot(c,w,5,0,0,0,ops,n) && px_slot(c,w,6,0,0,0,ops,n) && px_slot(c,w,7,0,0,0,ops,n);
    /* the binary32 backward leaves the score adjoint in p; the bf16 one reads p alone */
    return (bf ? px_slot(c,w,3,PX_IN,z,PX_R,ops,n) : px_slot(c,w,3,PX_OUT,z,PX_RW,ops,n)) &&
        px_slot(c,w,4,PX_IN,x,PX_R,ops,n) && px_slot(c,w,5,PX_OUT,x,PX_W,ops,n) &&
        px_slot(c,w,6,PX_OUT,y,PX_W,ops,n) && px_slot(c,w,7,PX_OUT,y,PX_W,ops,n);
  }
  case BCIR_DT_REC_ADD: case BCIR_DT_REC_MUL:  /* n | x, y (y += x, y *= x) */
    x=px_size(d[0],1,1,4);
    return !f && x && px_rest(w,1,2) && px_slot(c,w,0,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,1,PX_OUT,x,PX_RW,ops,n);
  case BCIR_DT_REC_PROVIDER_SILU:  /* n | x, y: the provider's SiLU */
    x=px_size(d[0],1,1,4);
    return !f && s->provider.silu && x && px_rest(w,1,2) && px_slot(c,w,0,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,1,PX_OUT,x,PX_W,ops,n);
  case BCIR_DT_REC_SILU_MUL:  /* n | x, u, y: BCIR's SiLU */
    x=px_size(d[0],1,1,4);
    return f==c->v2 && !s->provider.silu && x && px_rest(w,1,3) &&
        px_slot(c,w,0,PX_IN,x,PX_R,ops,n) && px_slot(c,w,1,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,2,PX_OUT,x,PX_W,ops,n);
  case BCIR_DT_REC_SILU_MUL_PANELS:  /* rows, cols (multiples of 32) | x, u, as A, as B */
    x=px_size(d[0],d[1],1,4); y=px_size(d[0],d[1],1,2);
    return f==c->v2 && !s->provider.silu && x && !(d[0]%32) && !(d[1]%32) && px_rest(w,2,4) &&
        px_slot(c,w,0,PX_IN,x,PX_R,ops,n) && px_slot(c,w,1,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,2,PX_OUT,y,PX_W,ops,n) && px_slot(c,w,3,PX_OUT,y,PX_W,ops,n);
  case BCIR_DT_REC_SWIGLU_VJP:  /* n | gate, up, dy, dgate, dup */
    x=px_size(d[0],1,1,4);
    return f==c->v2 && x && px_rest(w,1,5) && px_slot(c,w,0,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,1,PX_IN,x,PX_R,ops,n) && px_slot(c,w,2,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,3,PX_OUT,x,PX_W,ops,n) && px_slot(c,w,4,PX_OUT,x,PX_W,ops,n);
  case BCIR_DT_REC_SWIGLU_VJP_PANELS:  /* rows, cols | gate, up, dy; each adjoint as A and A^T */
    x=px_size(d[0],d[1],1,4); y=px_size(d[0],d[1],1,2);
    if (f!=c->v2 || !x || d[0]%32 || d[1]%32 || !px_rest(w,2,7)) return 0;
    for (i=0;i<3;i++) if (!px_slot(c,w,i,PX_IN,x,PX_R,ops,n)) return 0;
    for (i=3;i<7;i++) if (!px_slot(c,w,i,PX_OUT,y,PX_W,ops,n)) return 0;
    return 1;
  case BCIR_DT_REC_RMS_VJP:  /* rows, width, eps | x, gamma, dy, dx, dgamma, 1/rms; how */
    x=px_size(d[0],d[1],1,4); y=px_size(d[1],1,1,4); z=px_size(d[0],1,1,4);
    return f<=(BCIR_TENSOR_RMSB_ADD|BCIR_TENSOR_RMSB_FIRST) && w[5]==c->eps && x &&
        px_rest(w,3,6) && px_slot(c,w,0,PX_IN,x,PX_R,ops,n) && px_slot(c,w,1,PX_IN,y,PX_R,ops,n) &&
        px_slot(c,w,2,PX_IN,x,PX_R,ops,n) &&
        px_slot(c,w,3,PX_OUT,x,(f&BCIR_TENSOR_RMSB_ADD) ? PX_RW : PX_W,ops,n) &&
        px_slot(c,w,4,PX_OUT,y,(f&BCIR_TENSOR_RMSB_FIRST) ? PX_W : PX_RW,ops,n) &&
        px_slot(c,w,5,PX_OUT,z,PX_W,ops,n);
  case BCIR_DT_REC_LOSS:  /* rows, vocabulary | logits, targets, dlogits, the rows' terms */
    x=px_size(d[0],d[1],1,4);
    return !f && d[0]==c->g.n && d[1]==c->g.v && x && px_rest(w,2,4) &&
        px_slot(c,w,0,PX_IN,x,PX_R,ops,n) && px_slot(c,w,1,PX_EXT,px_size(d[0],1,1,4),PX_R,ops,n) &&
        px_slot(c,w,2,PX_OUT,x,PX_W,ops,n) && px_slot(c,w,3,PX_OUT,px_size(d[0],1,1,8),PX_W,ops,n);
  case BCIR_DT_REC_SCATTER:  /* rows, d | tokens, dx, the table's adjoint */
    x=px_size(d[0],d[1],1,4); y=px_size(c->g.v,d[1],1,4);
    return !f && d[0]==c->g.n && x && y && px_rest(w,2,3) &&
        px_slot(c,w,0,PX_EXT,px_size(d[0],1,1,4),PX_R,ops,n) &&
        px_slot(c,w,1,PX_IN,x,PX_R,ops,n) && px_slot(c,w,2,PX_OUT,y,PX_RW,ops,n);
  default:
    return 0;
  }
}
/* A GEMM job: `job` records of one launch, each a call -- ta, tb, m, n, k, alpha 1, beta 0 or 1 --
 * whose flags name the job's entry, the call's role and index, the job's contract field and its
 * panels: the job the state's kernels take. A call alone is the state's contract's entry, or the
 * provider's when it has one (and then nothing else is); a pair (NDT-SCHED-0) is two calls; a group
 * (NDT-SCHED-1) a chain of at most BCIR_TENSOR_MM_GROUP calls into one C (one m, n and C) then at
 * most as many others; the bf16 group the same under the bf16 contract, a call's A or B a panel of
 * whole 32-tiles (BF16-PANELS). */
static int px_gemm(const px_ctx *c,const uint64_t *w,size_t job,px_op *ops,size_t *n) {
  const uint64_t *chain=NULL; unsigned e=(unsigned)(w[2]&15u); size_t q,chained=0;
  int provider=c->s->provider.mm!=NULL;
  *n=0;
  switch (e) {
  case 0: case 1: case 2: case 3: if (job!=1 || e!=c->gc || provider) return 0; break;
  case 4: if (job!=2 || c->gc>2 || provider) return 0; break;
  case 5: if (c->gc>2 || provider) return 0; break;
  case 6: if (c->gc!=3 || provider) return 0; break;
  case 7: if (job!=1 || !provider) return 0; break;
  default: return 0;
  }
  for (q=0;q<job;q++) {
    const uint64_t *r=w+q*BCIR_DT_RECORD_WORDS; const uint64_t f=r[2];
    unsigned role=(unsigned)(f>>4&15u),pa=(unsigned)(f>>24&1u),pb=(unsigned)(f>>25&1u);
    unsigned field=e<4 ? e : e==7 ? 15u : c->gc; size_t m,nn,k,sa,sb,sc;
    if ((f&15u)!=e || (f>>8&255u)!=q || (f>>16&255u)!=field || f>>26 || r[3]>1 || r[4]>1 ||
        r[8]!=PX_ONE || (r[9] && r[9]!=PX_ONE) || !px_rest(r,7,4) || !px_fits(r[5]) ||
        !px_fits(r[6]) || !px_fits(r[7])) return 0;
    if (e==4 ? role!=q+1 : e==5 || e==6 ? role<1 || role>2 || (role==1 && q!=chained) : role!=0)
      return 0;
    m=(size_t)r[5]; nn=(size_t)r[6]; k=(size_t)r[7];
    if ((pa || pb) && (e!=6 || m%32 || nn%32 || k%32)) return 0;
    sa=px_size(m,k,1,pa ? 2 : 4); sb=px_size(k,nn,1,pb ? 2 : 4); sc=px_size(m,nn,1,4);
    if (!sa || !sb || !sc || !px_slot(c,r,0,PX_IN,sa,pa ? PX_PANEL : PX_R,ops,n) ||
        !px_slot(c,r,1,PX_IN,sb,pb ? PX_PANEL : PX_R,ops,n) ||
        !px_slot(c,r,2,PX_OUT,sc,r[9] ? PX_RW : PX_W,ops,n) || !px_ws(c,r,3,e==3 || e==6))
      return 0;
    if (role==1 && e!=4) {  /* a chain's call: the chain's C, m and n */
      ops[*n-1].chain=1; chained++;
      if (!chain) chain=r;
      else if (r[5]!=chain[5] || r[6]!=chain[6] || r[17]!=chain[17] || r[18]!=chain[18]) return 0;
    }
  }
  if (chained>BCIR_TENSOR_MM_GROUP || job-chained>BCIR_TENSOR_MM_GROUP) return 0;
  return px_apart(ops,*n);
}
/* the update record: the pass's as the trace holds it -- the reduce contract's flag, the
 * parameters and their chunk, the gradients, the weights and the two moments whole */
static int px_update(const px_ctx *c,const uint64_t *w) {
  static const unsigned order[4]={2,1,3,4}; size_t p=c->s->plan.parameters; unsigned q;
  if (w[2]!=((c->s->contract&BCIR_DT_CONTRACT_REDUCE) ? 1u : 0u) || w[3]!=p ||
      w[4]!=bcir_decoder_reduce_chunk(p) || !px_rest(w,2,4)) return 0;
  for (q=0;q<4;q++)
    if (w[11+3*q]!=order[q] || w[12+3*q] || w[13+3*q]!=c->len[1]) return 0;
  return 1;
}
/* the records of a job starting at w (of `left` records): a GEMM's calls share its launch */
static size_t px_job(const uint64_t *w,size_t left) {
  size_t job=1;
  if (w[0]==BCIR_DT_REC_GEMM)
    while (job<left && job<2*BCIR_TENSOR_MM_GROUP && w[job*BCIR_DT_RECORD_WORDS]==BCIR_DT_REC_GEMM &&
        w[job*BCIR_DT_RECORD_WORDS+1]==w[1]) job++;
  return job;
}
/* The program checked whole: 1 when the run may start (bcir_decoder_train.h's rules). */
static int px_check(const px_ctx *c,const uint64_t *p,size_t records,size_t steps,
    size_t accumulation) {
  px_op ops[PX_OPS]; size_t i=0,job,n,q,total,micro=0,batches=0,updates=0,open=0;
  uint64_t launch=0; int known=0,loss=0;
  if (!product(steps,accumulation,&total)) return 0;
  while (i<records) {
    const uint64_t *w=p+i*BCIR_DT_RECORD_WORDS;
    if (w[1]<=launch) return 0;  /* each job's launch exceeds the one before */
    launch=w[1]; job=px_job(w,records-i); n=0;
    if (!px_dispatch(c,w,job)) return 0;
    if (w[0]==BCIR_DT_REC_CHECK) {
      if (!px_rest(w,0,0) || (w[2]!=BCIR_DT_CHECK_FORWARD && w[2]!=BCIR_DT_CHECK_RAN &&
          w[2]!=BCIR_DT_CHECK_LOSS)) return 0;
      if (w[2]==BCIR_DT_CHECK_LOSS) {  /* closes the micro-batch its one loss record names */
        if (!loss) return 0;
        loss=0; open=0; micro++; batches++; i++; continue;
      }
    }
    else if (w[0]==BCIR_DT_REC_UPDATE) {  /* after exactly `accumulation` closed micro-batches */
      if (!px_update(c,w) || open || micro!=accumulation || updates==steps) return 0;
      micro=0; updates++; known=1; i++; continue;
    }
    else if (w[0]==BCIR_DT_REC_GEMM) { if (!px_gemm(c,w,job,ops,&n)) return 0; }
    else {
      if (!px_kernel(c,w,ops,&n) || !px_apart(ops,n)) return 0;
      if ((w[0]==BCIR_DT_REC_EMBED || w[0]==BCIR_DT_REC_LOSS || w[0]==BCIR_DT_REC_SCATTER) &&
          batches>=total) return 0;  /* no tokens past the run's */
      if (w[0]==BCIR_DT_REC_LOSS) {  /* one loss a micro-batch */
        if (loss) return 0;
        loss=1;
      }
    }
    if (!known && w[0]!=BCIR_DT_REC_SCAN)  /* FACTS-RUN: the weights read once known finite */
      for (q=0;q<n;q++) if (ops[q].space==1) return 0;
    if (w[0]==BCIR_DT_REC_SCAN && w[11]==1 && !w[12] && w[13]==c->len[1]) known=1;
    open++; i+=job;
  }
  return !open && !micro && updates==steps;
}
static void *px_at(const px_ctx *c,const uint64_t *w,unsigned q) {
  return c->base[w[11+3*q]]+(size_t)w[12+3*q];
}
/* a checked GEMM job of `job` calls: the entry its flags name, as the step calls it */
static int px_gemm_run(const px_ctx *c,const uint64_t *w,size_t job) {
  bcir_decoder_state *s=c->s; bcir_tensor_mm_call call[2*BCIR_TENSOR_MM_GROUP];
  bcir_tensor_bf16_panels pre[2*BCIR_TENSOR_MM_GROUP]; size_t q,chain=0; int panels=0;
  int variant=(int)w[BCIR_DT_WORD_INSTANCE];
  if (!job || job>2*BCIR_TENSOR_MM_GROUP) return 0; /* px_job's 1..2*GROUP calls */
  for (q=0;q<job;q++) {
    const uint64_t *r=w+q*BCIR_DT_RECORD_WORDS; bcir_tensor_mm_call *x=call+q;
    pre[q].a=(r[2]>>24&1u) ? (const unsigned short *)px_at(c,r,0) : NULL;
    pre[q].b=(r[2]>>25&1u) ? (const unsigned short *)px_at(c,r,1) : NULL;
    panels|=pre[q].a || pre[q].b;
    x->ta=(int)r[3]; x->tb=(int)r[4]; x->m=(size_t)r[5]; x->n=(size_t)r[6]; x->k=(size_t)r[7];
    x->alpha=1.0f; x->beta=r[9] ? 1.0f : 0.0f;
    x->a=pre[q].a ? NULL : (const float *)px_at(c,r,0);
    x->b=pre[q].b ? NULL : (const float *)px_at(c,r,1);
    x->c=(float *)px_at(c,r,2);
    if ((r[2]>>4&15u)==1u) chain++;
  }
  switch (w[2]&15u) {  /* the job's instance: PX-3b's word, checked against its family */
  case 4: return bcir_tensor_mm_pair_variant_pool(s->pool,gemm_contract(s),variant,call,call+1)!=0;
  case 5: case 6:
    return gemm_group_v(s,variant,chain ? call : NULL,chain,call+chain,job-chain,
        panels ? pre : NULL);
  default:
    return gemm_v(s,variant,call->ta,call->tb,call->m,call->n,call->k,1.0f,call->a,call->b,
        call->beta,call->c);
  }
}
/* a checked kernel record: the call the step makes for it, its verdicts into ran, bad and ce */
static int px_kernel_run(const px_ctx *c,const uint64_t *w,const uint32_t *tokens,
    const uint32_t *targets,int *ran,int *bad,double *ce) {
  bcir_decoder_state *s=c->s;
  size_t d0=(size_t)w[3],d1=(size_t)w[4],d2=(size_t)w[5],d3=(size_t)w[6],d4=(size_t)w[7];
  switch (w[0]) {
  case BCIR_DT_REC_SCAN:  /* a parameter arena's fails the run; another's reports to the check */
    if (finite_pool(s,px_at(c,w,0),d0)) break;
    if (w[11]<5) return BCIR_DT_NUMERIC;
    *bad=1; break;
  case BCIR_DT_REC_ZERO: zero_pool(s,px_at(c,w,0),d0); break;
  case BCIR_DT_REC_EMBED: embed(d0,d1,tokens,px_at(c,w,1),px_at(c,w,2)); break;
  case BCIR_DT_REC_RMS:
    bcir_tensor_rms_scan_pool(s->pool,d0,d1,(float)s->plan.spec.rms_eps,px_at(c,w,0),px_at(c,w,1),
        px_at(c,w,2),bad);
    break;
  case BCIR_DT_REC_ROPE:
    bcir_tensor_rope_pool(s->pool,d0,d1,d2,d3,s->plan.spec.rope_base,px_at(c,w,0),(int)d4);
    break;
  case BCIR_DT_REC_ATTN:
    *ran&=attention_v(s,(int)w[BCIR_DT_WORD_INSTANCE],d0,d1,d2,d3,d4,px_at(c,w,0),px_at(c,w,1),
        px_at(c,w,2),px_at(c,w,3),px_at(c,w,4),bad);
    break;
  case BCIR_DT_REC_ATTN_VJP: {
    geometry g=c->g;
    g.b=d0; g.t=d1; g.h=d2; g.kh=d3; g.hd=d4;
    *ran&=attention_backward_v(s,(int)w[BCIR_DT_WORD_INSTANCE],g,px_at(c,w,0),px_at(c,w,1),
        px_at(c,w,2),px_at(c,w,3),px_at(c,w,4),px_at(c,w,5),px_at(c,w,6),px_at(c,w,7));
    break;
  }
  case BCIR_DT_REC_ADD: bcir_tensor_add_pool(s->pool,d0,px_at(c,w,0),px_at(c,w,1)); break;
  case BCIR_DT_REC_MUL: bcir_tensor_mul_pool(s->pool,d0,px_at(c,w,0),px_at(c,w,1)); break;
  case BCIR_DT_REC_PROVIDER_SILU:
    DT_TRACE_PROVIDER_SILU(d0,px_at(c,w,0),px_at(c,w,1));
    s->provider.silu(s->provider.ctx,d0,px_at(c,w,0),px_at(c,w,1));
    break;
  case BCIR_DT_REC_SILU_MUL: silu_mul(s,d0,px_at(c,w,0),px_at(c,w,1),px_at(c,w,2),bad); break;
  case BCIR_DT_REC_SILU_MUL_PANELS:
    *ran&=bcir_tensor_silu_mul_panels_pool(s->pool,(int)c->v2,d0,d1,px_at(c,w,0),px_at(c,w,1),
        px_at(c,w,2),px_at(c,w,3),bad);
    break;
  case BCIR_DT_REC_SWIGLU_VJP:
    swiglu_backward(s,d0,px_at(c,w,0),px_at(c,w,1),px_at(c,w,2),px_at(c,w,3),px_at(c,w,4));
    break;
  case BCIR_DT_REC_SWIGLU_VJP_PANELS:
    *ran&=bcir_tensor_swiglu_backward_panels_pool(s->pool,(int)c->v2,d0,d1,px_at(c,w,0),
        px_at(c,w,1),px_at(c,w,2),px_at(c,w,3),px_at(c,w,4),px_at(c,w,5),px_at(c,w,6));
    break;
  case BCIR_DT_REC_RMS_VJP:
    rms_backward(s,d0,d1,px_at(c,w,0),px_at(c,w,1),px_at(c,w,2),px_at(c,w,3),px_at(c,w,4),
        px_at(c,w,5),(unsigned)w[2]);
    break;
  case BCIR_DT_REC_LOSS: *ce=loss_rows(s,c->g,px_at(c,w,0),targets,px_at(c,w,2),px_at(c,w,3)); break;
  case BCIR_DT_REC_SCATTER: scatter(s,d0,d1,tokens,px_at(c,w,1),px_at(c,w,2)); break;
  default: break;  /* none: the check admits no other kind here */
  }
  return BCIR_DT_OK;
}
/* The run of a checked program: its records in order, the host's the step's own checks, update
 * and event (bcir_decoder_run_accum's arithmetic: the micro-batch's mean, then its share). */
static int px_jobs(const px_ctx *c,const bcir_decoder_adamw *o,const uint64_t *p,size_t records,
    const uint32_t *tokens,const uint32_t *targets,const double *rates,size_t accumulation,
    bcir_decoder_event *events,size_t *completed) {
  bcir_decoder_state *s=c->s; bcir_decoder_adamw opt=*o; size_t i=0,job,batch=0,step=0;
  int ran=1,bad=0,rc; double ce=0.0,loss=0.0;
  while (i<records) {
    const uint64_t *w=p+i*BCIR_DT_RECORD_WORDS;
    DT_TIME_START(t0);
    job=px_job(w,records-i);
    bcir_tensor_pool_cap(s->pool,(unsigned)w[BCIR_DT_WORD_WORKERS]);  /* PX-3b: the job's workers */
    if (w[0]==BCIR_DT_REC_GEMM) ran&=px_gemm_run(c,w,job);
    else if (w[0]==BCIR_DT_REC_CHECK) {
      if (!ran) return BCIR_DT_CAPACITY;  /* a product that ran nowhere */
      if (w[2]==BCIR_DT_CHECK_FORWARD && bad) return BCIR_DT_NUMERIC;
      if (w[2]==BCIR_DT_CHECK_LOSS) {
        double part;
        if (!isfinite(ce)) return BCIR_DT_NUMERIC;
        part=ce/(double)c->g.n; loss+=part/(double)accumulation; ce=0.0; bad=0; batch++;
      }
    }
    else if (w[0]==BCIR_DT_REC_UPDATE) {
      double norm;
      if (rates) opt.lr=rates[step];
      rc=update_with(s,&opt,1.0/(double)accumulation,&norm,1,1); if (rc) return rc;
      events[step].loss=loss; events[step].grad_norm=norm; events[step].step=s->step;
      *completed=++step; loss=0.0;
    }
    else {
      size_t off=batch*c->g.n;
      int ext=w[0]==BCIR_DT_REC_EMBED || w[0]==BCIR_DT_REC_LOSS || w[0]==BCIR_DT_REC_SCATTER;
      rc=px_kernel_run(c,w,ext ? tokens+off : NULL,ext ? targets+off : NULL,&ran,&bad,&ce);
      if (rc) return rc;
    }
    DT_TIME_JOB(i,t0);
    i+=job;
  }
  return BCIR_DT_OK;
}
/* the program's jobs on the workers each names, then the pool's cap as it was before the run,
 * however the run ended (PX-3b) */
static int px_run(const px_ctx *c,const bcir_decoder_adamw *o,const uint64_t *p,size_t records,
    const uint32_t *tokens,const uint32_t *targets,const double *rates,size_t accumulation,
    bcir_decoder_event *events,size_t *completed) {
  unsigned before=bcir_tensor_pool_cap(c->s->pool,0u);
  int rc=px_jobs(c,o,p,records,tokens,targets,rates,accumulation,events,completed);
  bcir_tensor_pool_cap(c->s->pool,before);
  return rc;
}
int bcir_decoder_run_program(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    const uint64_t *program,size_t records,const uint32_t *tokens,const uint32_t *targets,
    size_t count,const double *rates,size_t steps,size_t accumulation,bcir_decoder_event *events,
    size_t capacity,size_t *completed) {
  px_ctx c; size_t bytes;
  int rc=run_args(s,o,tokens,targets,count,rates,steps,accumulation,events,capacity,completed,1);
  if (rc) return rc;
  if (!product(records,BCIR_DT_RECORD_WORDS*sizeof(uint64_t),&bytes)) return BCIR_DT_INVALID;
  if (records && (!program || (uintptr_t)program%_Alignof(uint64_t) ||
      !external_in(s,1,program,bytes) ||
      overlap(program,bytes,completed,sizeof(*completed)) ||
      (steps && overlap(program,bytes,events,steps*sizeof(*events))))) return BCIR_DT_INVALID;
  px_init(&c,s);
  if (!px_check(&c,program,records,steps,accumulation)) return BCIR_DT_INVALID;
  *completed=0;
  return px_run(&c,o,program,records,tokens,targets,rates,accumulation,events,completed);
}
#ifdef BCIR_DT_TRACE_PROBE
/* PX-1's records (the declarations at the top say what they are; the kinds are the header's). */
void bcir_decoder_trace_probe_begin(const bcir_decoder_state *s,uint64_t *words,size_t capacity) {
  trace_state=s; trace_words=words; trace_capacity=words ? capacity : 0;
  trace_count=trace_dropped=trace_mark=0; trace_launch=0;
}
size_t bcir_decoder_trace_probe_end(size_t *dropped) {
  size_t n=trace_count;
  if (dropped) *dropped=trace_dropped;
  trace_state=NULL; trace_words=NULL; trace_capacity=trace_count=trace_dropped=trace_mark=0;
  trace_launch=0;
  return n;
}
size_t bcir_decoder_trace_probe_words(void) { return BCIR_DT_TRACE_WORDS; }
static uint64_t tr_bits(float f) { uint32_t u; memcpy(&u,&f,sizeof u); return u; }
/* A record of `kind`: a new launch's, or (fresh 0) the next call of the launch before; NULL when no
 * trace is begun or the lent words are full (counted as dropped). */
static uint64_t *tr_record(unsigned kind,int fresh) {
  uint64_t *w;
  if (!trace_state) return NULL;
  if (fresh) { trace_launch++; trace_mark=trace_count; }
  if (trace_count>=trace_capacity) { trace_dropped++; return NULL; }
  w=trace_words+trace_count*BCIR_DT_TRACE_WORDS; trace_count++;
  memset(w,0,BCIR_DT_TRACE_WORDS*sizeof *w);
  w[0]=kind; w[1]=trace_launch;
  w[BCIR_DT_WORD_WORKERS]=bcir_tensor_pool_workers(trace_state->pool);  /* PX-3b: the job's workers */
  return w;
}
/* The instance the launch just recorded ran (an entry's return), into each of its records. */
static int tr_ran(int ran) {
  size_t i;
  for (i=trace_mark;i<trace_count;i++)
    trace_words[i*BCIR_DT_TRACE_WORDS+BCIR_DT_WORD_INSTANCE]=(uint64_t)(unsigned)ran;
  return ran;
}
/* Operand `slot` of record w: its space, its byte offset in it and its bytes. A NULL operand -- a
 * panel call's A or B, whose panel is recorded in its place -- records nothing. */
static void tr_ref(uint64_t *w,unsigned slot,const void *p,size_t bytes) {
  const bcir_decoder_state *s=trace_state; const void *base[7]; size_t len[7],i;
  uintptr_t x=(uintptr_t)p; uint64_t *r;
  if (!w || !p || slot>=10) return;
  r=w+11+3*slot;
  base[0]=s->weights; base[1]=s->gradients; base[2]=s->moment1; base[3]=s->moment2;
  base[4]=s->activations; base[5]=s->scratch; base[6]=s->workspace;
  len[0]=len[1]=len[2]=len[3]=s->plan.parameters*sizeof(float);
  len[4]=arena_floats(s)*sizeof(float); len[5]=s->plan.scratch*sizeof(float);
  len[6]=(s->contract&BCIR_DT_CONTRACT_BF16) ? s->workspace_bytes : 0;
  r[0]=8; r[1]=0; r[2]=bytes;
  for (i=0;i<7;i++)
    if (base[i] && len[i] && x>=(uintptr_t)base[i] && x-(uintptr_t)base[i]<len[i]) {
      r[0]=i+1; r[1]=x-(uintptr_t)base[i]; return;
    }
}
static void tr_two(unsigned kind,size_t n,const float *x,const float *y) {
  uint64_t *w=tr_record(kind,1);
  if (!w) return;
  w[3]=n; tr_ref(w,0,x,n*sizeof(float)); tr_ref(w,1,y,n*sizeof(float));
}
static void tr_add(size_t n,const float *x,const float *y) { tr_two(BCIR_DT_REC_ADD,n,x,y); }
static void tr_mul(size_t n,const float *x,const float *y) { tr_two(BCIR_DT_REC_MUL,n,x,y); }
static void tr_provider_silu(size_t n,const float *x,const float *y) {
  tr_two(BCIR_DT_REC_PROVIDER_SILU,n,x,y);
}
static void tr_rope(size_t b,size_t t,size_t h,size_t d,const float *x,int inverse) {
  uint64_t *w=tr_record(BCIR_DT_REC_ROPE,1);
  if (!w) return;
  w[3]=b; w[4]=t; w[5]=h; w[6]=d; w[7]=(uint64_t)(inverse!=0);
  tr_ref(w,0,x,b*t*h*d*sizeof(float));
}
static void tr_rms(size_t rows,size_t width,float eps,const float *x,const float *g,const float *y) {
  uint64_t *w=tr_record(BCIR_DT_REC_RMS,1); size_t b=rows*width*sizeof(float);
  if (!w) return;
  w[3]=rows; w[4]=width; w[5]=tr_bits(eps);
  tr_ref(w,0,x,b); tr_ref(w,1,g,width*sizeof(float)); tr_ref(w,2,y,b);
}
static void tr_swiglu_vjp(int v2,size_t n,const float *g,const float *u,const float *dy,
    const float *dg,const float *du) {
  uint64_t *w=tr_record(BCIR_DT_REC_SWIGLU_VJP,1); size_t b=n*sizeof(float);
  if (!w) return;
  w[2]=(uint64_t)(v2!=0); w[3]=n;
  tr_ref(w,0,g,b); tr_ref(w,1,u,b); tr_ref(w,2,dy,b); tr_ref(w,3,dg,b); tr_ref(w,4,du,b);
}
static void tr_swiglu_vjp_panels(int v2,size_t r,size_t c,const float *g,const float *u,
    const float *dy,const void *dga,const void *dgt,const void *dua,const void *dut) {
  uint64_t *w=tr_record(BCIR_DT_REC_SWIGLU_VJP_PANELS,1); size_t b=r*c*sizeof(float),h=r*c*2;
  if (!w) return;
  w[2]=(uint64_t)(v2!=0); w[3]=r; w[4]=c;
  tr_ref(w,0,g,b); tr_ref(w,1,u,b); tr_ref(w,2,dy,b);
  tr_ref(w,3,dga,h); tr_ref(w,4,dgt,h); tr_ref(w,5,dua,h); tr_ref(w,6,dut,h);
}
static void tr_silu_mul(int v2,size_t n,const float *x,const float *u,const float *y) {
  uint64_t *w=tr_record(BCIR_DT_REC_SILU_MUL,1); size_t b=n*sizeof(float);
  if (!w) return;
  w[2]=(uint64_t)(v2!=0); w[3]=n;
  tr_ref(w,0,x,b); tr_ref(w,1,u,b); tr_ref(w,2,y,b);
}
static void tr_silu_mul_panels(int v2,size_t r,size_t c,const float *x,const float *u,
    const void *a,const void *b) {
  uint64_t *w=tr_record(BCIR_DT_REC_SILU_MUL_PANELS,1); size_t f=r*c*sizeof(float),h=r*c*2;
  if (!w) return;
  w[2]=(uint64_t)(v2!=0); w[3]=r; w[4]=c;
  tr_ref(w,0,x,f); tr_ref(w,1,u,f); tr_ref(w,2,a,h); tr_ref(w,3,b,h);
}
static void tr_rms_vjp(size_t rows,size_t width,float eps,const float *x,const float *g,
    const float *dy,const float *dx,const float *dg,const float *inv,unsigned how) {
  uint64_t *w=tr_record(BCIR_DT_REC_RMS_VJP,1); size_t b=rows*width*sizeof(float);
  if (!w) return;
  w[2]=how; w[3]=rows; w[4]=width; w[5]=tr_bits(eps);
  tr_ref(w,0,x,b); tr_ref(w,1,g,width*sizeof(float)); tr_ref(w,2,dy,b); tr_ref(w,3,dx,b);
  tr_ref(w,4,dg,width*sizeof(float)); tr_ref(w,5,inv,rows*sizeof(float));
}
/* One GEMM call: its entry (0 v1, 1 fused, 2 kblock, 3 bf16 -- each alone; 4 a pair's, 5 a group's,
 * 6 a bf16 group's; 7 a provider's), its role (0 alone, 1 the pair's first or a chain's, 2 the
 * pair's second or an independent call), its index in the job, the job's contract (15 none), which
 * operands are panels, and the workspace it was lent. */
static void tr_call(uint64_t *w,unsigned entry,unsigned role,size_t index,unsigned contract,
    const bcir_tensor_mm_call *c,const unsigned short *pa,const unsigned short *pb,const void *ws,
    size_t bytes) {
  if (!w) return;
  w[2]=entry|(uint64_t)role<<4|(uint64_t)index<<8|(uint64_t)contract<<16|
      (uint64_t)(pa!=NULL)<<24|(uint64_t)(pb!=NULL)<<25;
  w[3]=(uint64_t)c->ta; w[4]=(uint64_t)c->tb; w[5]=c->m; w[6]=c->n; w[7]=c->k;
  w[8]=tr_bits(c->alpha); w[9]=tr_bits(c->beta);
  if (pa) tr_ref(w,0,pa,c->m*c->k*2); else tr_ref(w,0,c->a,c->m*c->k*sizeof(float));
  if (pb) tr_ref(w,1,pb,c->k*c->n*2); else tr_ref(w,1,c->b,c->k*c->n*sizeof(float));
  tr_ref(w,2,c->c,c->m*c->n*sizeof(float)); tr_ref(w,3,ws,bytes);
}
static void tr_mm(unsigned entry,int ta,int tb,size_t m,size_t n,size_t k,float alpha,
    const float *a,const float *b,float beta,const float *c,const void *ws,size_t bytes) {
  bcir_tensor_mm_call x;
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.b=b; x.beta=beta; x.c=(float *)c;
  tr_call(tr_record(BCIR_DT_REC_GEMM,1),entry,0,0,entry<4 ? entry : 15u,&x,NULL,NULL,ws,bytes);
}
static void tr_mm_group(unsigned entry,int contract,const bcir_tensor_mm_call *chain,size_t nchain,
    const bcir_tensor_mm_call *y,size_t ny,const bcir_tensor_bf16_panels *pre,const void *ws,
    size_t bytes) {
  size_t q; int fresh=1;
  for (q=0;q<nchain+ny;q++) {
    const bcir_tensor_mm_call *c=q<nchain ? chain+q : y+(q-nchain);
    tr_call(tr_record(BCIR_DT_REC_GEMM,fresh),entry,q<nchain ? 1u : 2u,q,(unsigned)contract,c,
        pre ? pre[q].a : NULL,pre ? pre[q].b : NULL,ws,bytes);
    fresh=0;
  }
}
static void tr_attention(unsigned kind,unsigned contract,int bf16,size_t b,size_t t,size_t h,
    size_t kh,size_t d,const float *q,const float *k,const float *v,const float *p,const float *y,
    const float *dq,const float *dk,const float *dv,const void *ws,size_t bytes) {
  uint64_t *w=tr_record(kind,1);
  size_t qb=b*t*h*d*sizeof(float),kb=b*t*kh*d*sizeof(float);
  if (!w) return;
  w[2]=contract|(uint64_t)(bf16!=0)<<8;
  w[3]=b; w[4]=t; w[5]=h; w[6]=kh; w[7]=d;
  tr_ref(w,0,q,qb); tr_ref(w,1,k,kb); tr_ref(w,2,v,kb); tr_ref(w,3,p,b*h*t*t*sizeof(float));
  tr_ref(w,4,y,qb); tr_ref(w,5,dq,qb); tr_ref(w,6,dk,kb); tr_ref(w,7,dv,kb); tr_ref(w,8,ws,bytes);
}
static void tr_quad(uint64_t *w,const float *gr,const float *wt,const float *m1,const float *m2,
    size_t n) {
  size_t b=n*sizeof(float);
  tr_ref(w,0,gr,b); tr_ref(w,1,wt,b); tr_ref(w,2,m1,b); tr_ref(w,3,m2,b);
}
/* The pool's jobs the trainer writes itself, told apart by their task; a job nothing here names is
 * recorded as kind 0, a finding. */
static void tr_run(bcir_tensor_task_fn fn,const void *ctx) {
  uint64_t *w;
  if (fn==scan_task || fn==fill_task) {
    const span_job *j=(const span_job *)ctx;
    if (!(w=tr_record(fn==scan_task ? BCIR_DT_REC_SCAN : BCIR_DT_REC_ZERO,1))) return;
    w[3]=j->n; tr_ref(w,0,fn==scan_task ? (const void *)j->x : (const void *)j->y,j->n*sizeof(float));
  }
  else if (fn==loss_task) {
    const loss_job *j=(const loss_job *)ctx;
    if (!(w=tr_record(BCIR_DT_REC_LOSS,1))) return;
    w[3]=j->n; w[4]=j->v;
    tr_ref(w,0,j->logits,j->n*j->v*sizeof(float)); tr_ref(w,1,j->targets,j->n*sizeof(uint32_t));
    tr_ref(w,2,j->dl,j->n*j->v*sizeof(float)); tr_ref(w,3,j->terms,j->n*sizeof(double));
  }
  else if (fn==scatter_task) {
    const scatter_job *j=(const scatter_job *)ctx;
    if (!(w=tr_record(BCIR_DT_REC_SCATTER,1))) return;
    w[3]=j->n; w[4]=j->d;
    tr_ref(w,0,j->tokens,j->n*sizeof(uint32_t)); tr_ref(w,1,j->dx,j->n*j->d*sizeof(float));
    tr_ref(w,2,j->grad,(size_t)trace_state->plan.spec.vocab*j->d*sizeof(float));
  }
  else if (fn==pass_task) {
    const pass_job *j=(const pass_job *)ctx;
    if (!(w=tr_record(BCIR_DT_REC_PASS,1))) return;
    w[2]=(uint64_t)(j->reduce!=0); w[3]=j->n; w[4]=j->chunk; tr_quad(w,j->gr,j->wt,j->m1,j->m2,j->n);
  }
  else if (fn==preflight_task) {
    const preflight_job *j=(const preflight_job *)ctx;
    if (!(w=tr_record(BCIR_DT_REC_PREFLIGHT,1))) return;
    w[3]=j->n; w[4]=j->chunk; tr_quad(w,j->gr,j->wt,j->m1,j->m2,j->n);
  }
  else if (fn==commit_task) {
    const commit_job *j=(const commit_job *)ctx;
    if (!(w=tr_record(BCIR_DT_REC_COMMIT,1))) return;
    w[3]=j->n; tr_quad(w,j->gr,j->wt,j->m1,j->m2,j->n);
  }
  else (void)tr_record(0,1);
}
static void tr_embed(size_t n,size_t d,const uint32_t *tokens,const float *table,const float *x) {
  uint64_t *w=tr_record(BCIR_DT_REC_EMBED,1);
  if (!w) return;
  w[3]=n; w[4]=d;
  tr_ref(w,0,tokens,n*sizeof(uint32_t));
  tr_ref(w,1,table,(size_t)trace_state->plan.spec.vocab*d*sizeof(float));
  tr_ref(w,2,x,n*d*sizeof(float));
}
#endif
