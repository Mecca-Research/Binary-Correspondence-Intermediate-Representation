#include "bcir_decoder_train.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
    "native decoder ABI requires IEEE binary32");
uint32_t bcir_decoder_abi_version(void) { return 3; }
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
static int valid(bcir_decoder_state *s) {
  bcir_decoder_plan p; const float *ptr[6]; size_t len[6],cap[6],i,j;
  int rc;
  if (!s) return BCIR_DT_INVALID;
  rc=bcir_decoder_make_plan(&s->plan.spec,&p);
  if (rc) return rc;
  if (p.parameters!=s->plan.parameters || p.activations!=s->plan.activations ||
      p.scratch!=s->plan.scratch) return BCIR_DT_INVALID;
  if (s->contract&~(uint32_t)(BCIR_DT_CONTRACT_FUSED|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_KBLOCK|
      BCIR_DT_CONTRACT_REDUCE)) return BCIR_DT_INVALID;
  if ((s->contract&BCIR_DT_CONTRACT_FUSED) && (s->contract&BCIR_DT_CONTRACT_KBLOCK)) return BCIR_DT_INVALID;  /* one GEMM contract */
  ptr[0]=s->weights; ptr[1]=s->gradients; ptr[2]=s->moment1; ptr[3]=s->moment2;
  ptr[4]=s->activations; ptr[5]=s->scratch;
  len[0]=len[1]=len[2]=len[3]=p.parameters; len[4]=p.activations; len[5]=p.scratch;
  cap[0]=s->weight_capacity; cap[1]=s->gradient_capacity;
  cap[2]=s->moment1_capacity; cap[3]=s->moment2_capacity;
  cap[4]=s->activation_capacity; cap[5]=s->scratch_capacity;
  for (i=0;i<6;i++) {
    if (!ptr[i] || (uintptr_t)ptr[i]%_Alignof(float)) return BCIR_DT_INVALID;
    if (cap[i]<len[i]) return BCIR_DT_CAPACITY;
    if (len[i]*sizeof(float)>UINTPTR_MAX-(uintptr_t)ptr[i]) return BCIR_DT_CAPACITY;
    if (overlap(ptr[i],len[i]*sizeof(float),s,sizeof(*s))) return BCIR_DT_INVALID;
    for (j=0;j<i;j++) if (overlap(ptr[i],len[i]*sizeof(float),ptr[j],len[j]*sizeof(float)))
      return BCIR_DT_INVALID;
  }
  return BCIR_DT_OK;
}
static int external(bcir_decoder_state *s,const void *p,size_t bytes) {
  if (!p || bytes>UINTPTR_MAX-(uintptr_t)p) return 0;
  return !overlap(p,bytes,s,sizeof(*s)) &&
    !overlap(p,bytes,s->weights,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->gradients,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->moment1,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->moment2,s->plan.parameters*sizeof(float)) &&
    !overlap(p,bytes,s->activations,s->plan.activations*sizeof(float)) &&
    !overlap(p,bytes,s->scratch,s->plan.scratch*sizeof(float));
}
static int ids(bcir_decoder_state *s,const uint32_t *x,size_t n) {
  size_t i;
  if (n>SIZE_MAX/sizeof(uint32_t) || !external(s,x,n*sizeof(uint32_t)) ||
      (uintptr_t)x%_Alignof(uint32_t)) return 0;
  for (i=0;i<n;i++) if (x[i]>=s->plan.spec.vocab) return 0;
  return 1;
}
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
 * fused v2, 2 the blocked v3): one predicate for the forward's linears and the backward's pairs. */
static int gemm_contract(const bcir_decoder_state *s) {
  return (s->contract&BCIR_DT_CONTRACT_KBLOCK) ? 2 : (s->contract&BCIR_DT_CONTRACT_FUSED) ? 1 : 0;
}
static void gemm(bcir_decoder_state *s,int ta,int tb,size_t m,size_t n,size_t k,float alpha,
    const float *a,const float *b,float beta,float *c) {
  if (s->provider.mm) s->provider.mm(s->provider.ctx,ta,tb,m,n,k,alpha,a,b,beta,c);
  else if (gemm_contract(s)==2) bcir_tensor_mm_kblock_pool(s->pool,ta,tb,m,n,k,alpha,a,b,beta,c);
  else if (gemm_contract(s)==1) bcir_tensor_mm_fused_pool(s->pool,ta,tb,m,n,k,alpha,a,b,beta,c);
  else bcir_tensor_mm_pool(s->pool,ta,tb,m,n,k,alpha,a,b,beta,c);
}
static void linear(bcir_decoder_state *s,size_t r,size_t in,size_t out,const float *x,
    const float *w,float *y) { gemm(s,0,1,r,out,in,1.0f,x,w,0.0f,y); }
/* A linear's two VJPs -- dx = dy W (beta-scaled) and dW += dy^T x -- read one adjoint and write
 * apart, so BCIR's kernels run them as one pool job (NDT-SCHED-0: one barrier, the narrow
 * product's idle threads on the other's tasks); a provider's GEMM runs them one after the other,
 * threaded as the provider threads it. The results are the two calls', bit for bit. */
static void linear_backward(bcir_decoder_state *s,size_t r,size_t in,size_t out,
    const float *x,const float *w,const float *dy,float *dx,float *dw,float beta) {
  bcir_tensor_mm_call cx,cw;
  if (s->provider.mm) {
    gemm(s,0,0,r,in,out,1.0f,dy,w,beta,dx);
    gemm(s,1,0,out,in,r,1.0f,dy,x,1.0f,dw);
    return;
  }
  cx.ta=0; cx.tb=0; cx.m=r; cx.n=in; cx.k=out; cx.alpha=1.0f; cx.a=dy; cx.b=w; cx.beta=beta; cx.c=dx;
  cw.ta=1; cw.tb=0; cw.m=out; cw.n=in; cw.k=r; cw.alpha=1.0f; cw.a=dy; cw.b=x; cw.beta=1.0f; cw.c=dw;
  bcir_tensor_mm_pair_pool(s->pool,gemm_contract(s),&cx,&cw);
}
/* The linears that read one input -- the attention's q, k and v, the MLP's gate and up -- as one
 * pool job (NDT-SCHED-1): their outputs are disjoint, so each is its own entry's bit for bit; and
 * their VJPs as one job, the input adjoints a chain into one dx (the first at the call's beta, the
 * rest adding, in the order the separate calls added) and the weight adjoints apart. A provider's
 * GEMM runs the calls one after another, a linear's two VJPs together, as before. */
static void linears(bcir_decoder_state *s,size_t r,size_t in,const float *x,size_t count,
    const size_t *out,const float *const *w,float *const *y) {
  bcir_tensor_mm_call c[BCIR_TENSOR_MM_GROUP]; size_t q;
  if (s->provider.mm || count>BCIR_TENSOR_MM_GROUP) {
    for (q=0;q<count;q++) linear(s,r,in,out[q],x,w[q],y[q]);
    return;
  }
  for (q=0;q<count;q++) {
    c[q].ta=0; c[q].tb=1; c[q].m=r; c[q].n=out[q]; c[q].k=in; c[q].alpha=1.0f; c[q].a=x;
    c[q].b=w[q]; c[q].beta=0.0f; c[q].c=y[q];
  }
  bcir_tensor_mm_group_pool(s->pool,gemm_contract(s),NULL,0,c,count);
}
static void linears_backward(bcir_decoder_state *s,size_t r,size_t in,const float *x,size_t count,
    const size_t *out,const float *const *w,const float *const *dy,float *dx,float *const *dw,
    float beta) {
  bcir_tensor_mm_call cx[BCIR_TENSOR_MM_GROUP],cw[BCIR_TENSOR_MM_GROUP]; size_t q;
  if (s->provider.mm || count>BCIR_TENSOR_MM_GROUP) {
    for (q=0;q<count;q++) linear_backward(s,r,in,out[q],x,w[q],dy[q],dx,dw[q],q ? 1.0f : beta);
    return;
  }
  for (q=0;q<count;q++) {
    cx[q].ta=0; cx[q].tb=0; cx[q].m=r; cx[q].n=in; cx[q].k=out[q]; cx[q].alpha=1.0f; cx[q].a=dy[q];
    cx[q].b=w[q]; cx[q].beta=q ? 1.0f : beta; cx[q].c=dx;
    cw[q].ta=1; cw[q].tb=0; cw[q].m=out[q]; cw[q].n=in; cw[q].k=r; cw[q].alpha=1.0f; cw[q].a=dy[q];
    cw[q].b=x; cw[q].beta=1.0f; cw[q].c=dw[q];
  }
  bcir_tensor_mm_group_pool(s->pool,gemm_contract(s),cx,count,cw,count);
}
static void silu(bcir_decoder_state *s,size_t n,const float *x,float *y) {
  if (s->provider.silu) s->provider.silu(s->provider.ctx,n,x,y);
  else if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_silu_v2_pool(s->pool,n,x,y);
  else bcir_tensor_silu_pool(s->pool,n,x,y);
}
static void swiglu_backward(bcir_decoder_state *s,size_t n,const float *g,const float *u,
    const float *dy,float *dg,float *du) {
  if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_swiglu_backward_v2_pool(s->pool,n,g,u,dy,dg,du);
  else bcir_tensor_swiglu_backward_pool(s->pool,n,g,u,dy,dg,du);
}
static void attention(bcir_decoder_state *s,size_t b,size_t t,size_t h,size_t kh,size_t d,
    const float *q,const float *k,const float *v,float *p,float *y) {
  if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_attention_v2_pool(s->pool,b,t,h,kh,d,q,k,v,p,y);
  else bcir_tensor_attention_pool(s->pool,b,t,h,kh,d,q,k,v,p,y);
}
static void forward(bcir_decoder_state *s,const uint32_t *tokens,geometry g) {
  const bcir_decoder_spec *sp=&s->plan.spec;
  size_t i,l; float *xs=s->activations,*cs=xs+(g.l+1)*g.nd;
  float *norm=cs+g.l*g.la,*logits=norm+g.nd;
  float *final=s->weights+g.v*g.d+g.l*g.lp;
  const float *head=sp->tied ? s->weights : final+g.d;
  for (i=0;i<g.n;i++) memcpy(xs+i*g.d,s->weights+(size_t)tokens[i]*g.d,g.d*sizeof(float));
  for (l=0;l<g.l;l++) {
    layer p=params(s->weights+g.v*g.d+l*g.lp,g);
    cache a=activations(cs+l*g.la,g); float *x=xs+l*g.nd,*y=x+g.nd;
    bcir_tensor_rms_pool(s->pool,g.n,g.d,(float)sp->rms_eps,x,p.norm1,a.n1);
    {
      size_t out[3]; const float *w[3]; float *y3[3];
      out[0]=g.d; out[1]=g.k; out[2]=g.k; w[0]=p.q; w[1]=p.k; w[2]=p.v; y3[0]=a.q; y3[1]=a.k; y3[2]=a.v;
      linears(s,g.n,g.d,a.n1,3,out,w,y3);
    }
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.h,g.hd,sp->rope_base,a.q,0);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.kh,g.hd,sp->rope_base,a.k,0);
    attention(s,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,a.ctx);
    linear(s,g.n,g.d,g.d,a.ctx,p.o,a.x2);
    bcir_tensor_add_pool(s->pool,g.nd,x,a.x2);
    bcir_tensor_rms_pool(s->pool,g.n,g.d,(float)sp->rms_eps,a.x2,p.norm2,a.n2);
    {
      size_t out[2]; const float *w[2]; float *y2[2];
      out[0]=g.f; out[1]=g.f; w[0]=p.gate; w[1]=p.up; y2[0]=a.gate; y2[1]=a.up;
      linears(s,g.n,g.d,a.n2,2,out,w,y2);
    }
    silu(s,g.nf,a.gate,a.hidden);
    bcir_tensor_mul_pool(s->pool,g.nf,a.up,a.hidden);
    linear(s,g.n,g.f,g.d,a.hidden,p.down,y);
    bcir_tensor_add_pool(s->pool,g.nd,a.x2,y);
  }
  bcir_tensor_rms_pool(s->pool,g.n,g.d,(float)sp->rms_eps,xs+g.l*g.nd,final,norm);
  linear(s,g.n,g.d,g.v,norm,head,logits);
}
int bcir_decoder_forward(bcir_decoder_state *s,const uint32_t *tokens,size_t n) {
  geometry g; int rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n)) return BCIR_DT_INVALID;
  if (!finite_pool(s,s->weights,s->plan.parameters)) return BCIR_DT_NUMERIC;
  forward(s,tokens,g);
  return finite_pool(s,s->activations,s->plan.activations) ? BCIR_DT_OK : BCIR_DT_NUMERIC;
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
static void rms_backward(bcir_decoder_state *s,size_t rows,size_t width,const float *x,
    const float *g,const float *dy,float *dx,float *dg,float *inv) {
  bcir_tensor_rms_backward_pool(s->pool,rows,width,(float)s->plan.spec.rms_eps,x,g,dy,dx,dg,inv);
}
int bcir_decoder_loss_backward(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t n,int accumulate,double *loss) {
  geometry g; size_t i,l; double ce=0.0; loss_job lj; scatter_job sj;
  float *xs,*cs,*norm,*logits,*cur,*tmp,*dn,*dq,*dk,*dv,*dc,*dg,*du,*dh,*dl;
  float *final,*dfinal,*head,*dhead; int rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n) || !ids(s,targets,n) ||
      (accumulate!=0 && accumulate!=1) || !external(s,loss,sizeof(*loss)) ||
      (uintptr_t)loss%_Alignof(double)) return BCIR_DT_INVALID;
  if (!finite_pool(s,s->weights,s->plan.parameters) ||
      (accumulate && !finite_pool(s,s->gradients,s->plan.parameters))) return BCIR_DT_NUMERIC;
  forward(s,tokens,g);
  if (!finite_pool(s,s->activations,s->plan.activations)) return BCIR_DT_NUMERIC;
  if (!accumulate) zero_pool(s,s->gradients,s->plan.parameters);
  xs=s->activations; cs=xs+(g.l+1)*g.nd; norm=cs+g.l*g.la; logits=norm+g.nd;
  cur=s->scratch; tmp=cur+g.nd; dn=tmp+g.nd; dq=dn+g.nd; dk=dq+g.nd;
  dv=dk+g.nk; dc=dv+g.nk; dg=dc+g.nd; du=dg+g.nf; dh=du+g.nf; dl=dh+g.nf;
  final=s->weights+g.v*g.d+g.l*g.lp; dfinal=s->gradients+g.v*g.d+g.l*g.lp;
  head=s->plan.spec.tied ? s->weights : final+g.d;
  dhead=s->plan.spec.tied ? s->gradients : dfinal+g.d;
  /* the rows' terms live at the arena's start (8n bytes within cur, tmp and dn: regions the
   * head's VJP writes only after they are added) */
  lj.n=g.n; lj.v=g.v; lj.logits=logits; lj.targets=targets; lj.dl=dl;
  lj.terms=(unsigned char *)s->scratch;
  bcir_tensor_pool_run(s->pool,loss_task,&lj,(g.n+BCIR_DT_ROW_CHUNK-1)/BCIR_DT_ROW_CHUNK);
  for (i=0;i<g.n;i++) {
    double term; memcpy(&term,lj.terms+i*sizeof(double),sizeof(double)); ce+=term;
  }
  linear_backward(s,g.n,g.d,g.v,norm,head,dl,dn,dhead,0.0f);
  /* each gamma VJP keeps its rows' 1/rms in dq, free until the layer's query VJP */
  rms_backward(s,g.n,g.d,xs+g.l*g.nd,final,dn,cur,dfinal,dq);
  for (l=g.l;l-->0;) {
    layer p=params(s->weights+g.v*g.d+l*g.lp,g);
    layer grad=params(s->gradients+g.v*g.d+l*g.lp,g);
    cache a=activations(cs+l*g.la,g); float *x=xs+l*g.nd;
    linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f);
    swiglu_backward(s,g.nf,a.gate,a.up,dh,dg,du);
    {
      size_t out[2]; const float *w[2],*d2[2]; float *dw[2];
      out[0]=g.f; out[1]=g.f; w[0]=p.gate; w[1]=p.up; d2[0]=dg; d2[1]=du; dw[0]=grad.gate; dw[1]=grad.up;
      linears_backward(s,g.n,g.d,a.n2,2,out,w,d2,dn,dw,0.0f);
    }
    rms_backward(s,g.n,g.d,a.x2,p.norm2,dn,tmp,grad.norm2,dq);
    bcir_tensor_add_pool(s->pool,g.nd,tmp,cur);
    linear_backward(s,g.n,g.d,g.d,a.ctx,p.o,cur,dc,grad.o,0.0f);
    /* the layer's probabilities are read for the last time here: the in-place backward
     * leaves its score adjoint in their place (same dq/dk/dv as the reference, bit for bit) */
    bcir_tensor_attention_backward_inplace_pool(s->pool,g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,
        dc,dq,dk,dv);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.h,g.hd,s->plan.spec.rope_base,dq,1);
    bcir_tensor_rope_pool(s->pool,g.b,g.t,g.kh,g.hd,s->plan.spec.rope_base,dk,1);
    {
      size_t out[3]; const float *w[3],*d3[3]; float *dw[3];
      out[0]=g.d; out[1]=g.k; out[2]=g.k; w[0]=p.q; w[1]=p.k; w[2]=p.v;
      d3[0]=dq; d3[1]=dk; d3[2]=dv; dw[0]=grad.q; dw[1]=grad.k; dw[2]=grad.v;
      linears_backward(s,g.n,g.d,a.n1,3,out,w,d3,dn,dw,0.0f);
    }
    rms_backward(s,g.n,g.d,x,p.norm1,dn,tmp,grad.norm1,dq);
    bcir_tensor_add_pool(s->pool,g.nd,tmp,cur);
  }
  sj.n=g.n; sj.d=g.d; sj.tokens=tokens; sj.dx=cur; sj.grad=s->gradients;
  bcir_tensor_pool_run(s->pool,scatter_task,&sj,(g.d+BCIR_DT_COL_CHUNK-1)/BCIR_DT_COL_CHUNK);
  if (!finite_pool(s,s->gradients,s->plan.parameters) || !isfinite(ce)) return BCIR_DT_NUMERIC;
  *loss=ce/(double)g.n; return BCIR_DT_OK;
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
__attribute__((target("avx2"))) static int preflight_avx2(const adamw_terms *t,size_t n,
    const float *gr,const float *wt,const float *m1,const float *m2) {
  return adamw_preflight(t,n,gr,wt,m1,m2);
}
__attribute__((target("avx2"))) static void commit_avx2(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) { adamw_commit(t,n,gr,wt,m1,m2); }
__attribute__((target("avx512f"))) static int preflight_avx512(const adamw_terms *t,size_t n,
    const float *gr,const float *wt,const float *m1,const float *m2) {
  return adamw_preflight(t,n,gr,wt,m1,m2);
}
__attribute__((target("avx512f"))) static void commit_avx512(const adamw_terms *t,size_t n,
    const float *gr,float *wt,float *m1,float *m2) { adamw_commit(t,n,gr,wt,m1,m2); }
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
  const float *gr,*wt,*m1,*m2; size_t n,chunk; double gs; int reduce,variant; pass_part *parts;
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
#endif
static void pass_task(void *ctx,size_t task) {
  pass_job *j=(pass_job *)ctx; pass_part p; size_t i;
  size_t i0=task*j->chunk,i1=i0+j->chunk<j->n ? i0+j->chunk : j->n;
  const float *gr=j->gr; double gs=j->gs;
#ifdef BCIR_DT_DISPATCH
  if (j->variant&4) pass_avx512(i1-i0,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p);
  else if (j->variant&2) pass_avx2(i1-i0,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p);
  else
#endif
  pass_generic(i1-i0,gr+i0,j->wt+i0,j->m1+i0,j->m2+i0,gs,&p);
  p.norm=0.0;
  if (j->reduce) for (i=i0;i<i1;i++) { double g=(double)gr[i]*gs; p.norm+=g*g; }
  j->parts[task]=p;
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
  j.t=t; j.n=n; j.gr=gr; j.wt=wt; j.m1=m1; j.m2=m2; j.variant=bcir_tensor_mm_variants();
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
  bcir_tensor_pool_run(s->pool,pass_task,&pj,out->tasks);
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
int bcir_decoder_update(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    double gs,double *grad_norm) {
  size_t i; double norm,scale,b1,b2; int rc=valid(s);
  bcir_decoder_adamw options; pass_part parts[BCIR_DT_PASS_TASKS]; pass_result pass;
  if (rc) return rc;
  if (!o) return BCIR_DT_INVALID;
  options=*o; o=&options;
  if (!optimizer_valid(o) || !isfinite(gs) || gs<=0 || s->step==UINT64_MAX ||
      !isfinite(s->beta1_power) || !isfinite(s->beta2_power) ||
      s->beta1_power<0 || s->beta1_power>1 || s->beta2_power<0 || s->beta2_power>1 ||
      (!s->step && (s->beta1_power!=1 || s->beta2_power!=1)) ||
      !external(s,grad_norm,sizeof(*grad_norm)) || (uintptr_t)grad_norm%_Alignof(double))
    return BCIR_DT_INVALID;
  rc=first_pass(s,gs,parts,&pass);
  if (rc) return rc;
  norm=sqrt(pass.norm2); if (!isfinite(norm)) return BCIR_DT_NUMERIC;
  scale=gs;
  if (o->grad_clip>0 && norm>o->grad_clip) scale*=o->grad_clip/(norm+1e-6);
  b1=s->beta1_power*o->beta1; b2=s->beta2_power*o->beta2;
  /* Round moments before using them; preflight every candidate before any commit. The commit
   * recomputes the candidates the preflight admitted, operation for operation; the preflight
   * is skipped only when adamw_bounded proves it would admit them all. */
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
int bcir_decoder_run(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    const uint32_t *tokens,const uint32_t *targets,size_t count,
    const double *rates,size_t steps,bcir_decoder_event *events,size_t capacity,size_t *completed) {
  return bcir_decoder_run_accum(s,o,tokens,targets,count,rates,steps,1,events,capacity,completed);
}
int bcir_decoder_run_accum(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    const uint32_t *tokens,const uint32_t *targets,size_t count,
    const double *rates,size_t steps,size_t accumulation,bcir_decoder_event *events,
    size_t capacity,size_t *completed) {
  size_t n,total,batches,i,j; int rc=valid(s); bcir_decoder_adamw opt;
  if (rc) return rc;
  if (!optimizer_valid(o) || !accumulation || !completed || (uintptr_t)completed%_Alignof(size_t) ||
      !external(s,completed,sizeof(*completed))) return BCIR_DT_INVALID;
  n=(size_t)s->plan.spec.batch*s->plan.spec.time;
  if (!product(steps,accumulation,&batches) || !product(n,batches,&total) ||
      steps>SIZE_MAX/sizeof(*events) || steps>SIZE_MAX/sizeof(*rates) ||
      s->step>UINT64_MAX-steps) return BCIR_DT_CAPACITY;
  if (count!=total || capacity<steps) return BCIR_DT_CAPACITY;
  if (steps && (!ids(s,tokens,total) || !ids(s,targets,total) ||
      !external(s,events,steps*sizeof(*events)) || (uintptr_t)events%_Alignof(bcir_decoder_event) ||
      (rates && (!external(s,rates,steps*sizeof(*rates)) || (uintptr_t)rates%_Alignof(double)))))
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
  opt=*o; *completed=0;
  for (i=0;i<steps;i++) {
    double loss=0.0,norm;
    for (j=0;j<accumulation;j++) {
      double part; size_t off=(i*accumulation+j)*n;
      rc=bcir_decoder_loss_backward(s,tokens+off,targets+off,n,j!=0,&part);
      if (rc) return rc;
      loss+=part/(double)accumulation;
    }
    if (rates) opt.lr=rates[i];
    rc=bcir_decoder_update(s,&opt,1.0/(double)accumulation,&norm); if (rc) return rc;
    events[i].loss=loss; events[i].grad_norm=norm; events[i].step=s->step; *completed=i+1;
  }
  return BCIR_DT_OK;
}
