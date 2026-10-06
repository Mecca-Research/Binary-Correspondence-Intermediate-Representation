#include "bcir_decoder_train.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
    "native decoder ABI requires IEEE binary32");
uint32_t bcir_decoder_abi_version(void) { return 1; }
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
  size_t i; for (i=0;i<n;i++) if (!isfinite(x[i])) return 0;
  return 1;
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
    bcir_tensor_rms(g.n,g.d,(float)sp->rms_eps,x,p.norm1,a.n1);
    bcir_tensor_linear(&s->provider,g.n,g.d,g.d,a.n1,p.q,a.q);
    bcir_tensor_linear(&s->provider,g.n,g.d,g.k,a.n1,p.k,a.k);
    bcir_tensor_linear(&s->provider,g.n,g.d,g.k,a.n1,p.v,a.v);
    bcir_tensor_rope(g.b,g.t,g.h,g.hd,sp->rope_base,a.q,0);
    bcir_tensor_rope(g.b,g.t,g.kh,g.hd,sp->rope_base,a.k,0);
    bcir_tensor_attention(g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,a.ctx);
    bcir_tensor_linear(&s->provider,g.n,g.d,g.d,a.ctx,p.o,a.x2);
    for (i=0;i<g.nd;i++) a.x2[i]+=x[i];
    bcir_tensor_rms(g.n,g.d,(float)sp->rms_eps,a.x2,p.norm2,a.n2);
    bcir_tensor_linear(&s->provider,g.n,g.d,g.f,a.n2,p.gate,a.gate);
    bcir_tensor_linear(&s->provider,g.n,g.d,g.f,a.n2,p.up,a.up);
    (s->provider.silu ? s->provider.silu : bcir_tensor_silu)(s->provider.ctx,g.nf,a.gate,a.hidden);
    for (i=0;i<g.nf;i++) a.hidden[i]*=a.up[i];
    bcir_tensor_linear(&s->provider,g.n,g.f,g.d,a.hidden,p.down,y);
    for (i=0;i<g.nd;i++) y[i]+=a.x2[i];
  }
  bcir_tensor_rms(g.n,g.d,(float)sp->rms_eps,xs+g.l*g.nd,final,norm);
  bcir_tensor_linear(&s->provider,g.n,g.d,g.v,norm,head,logits);
}
int bcir_decoder_forward(bcir_decoder_state *s,const uint32_t *tokens,size_t n) {
  geometry g; int rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n)) return BCIR_DT_INVALID;
  if (!finite_span(s->weights,s->plan.parameters)) return BCIR_DT_NUMERIC;
  forward(s,tokens,g);
  return finite_span(s->activations,s->plan.activations) ? BCIR_DT_OK : BCIR_DT_NUMERIC;
}
int bcir_decoder_loss_backward(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t n,int accumulate,double *loss) {
  geometry g; size_t i,j,l; double ce=0.0;
  float *xs,*cs,*norm,*logits,*cur,*tmp,*dn,*dq,*dk,*dv,*dc,*dg,*du,*dh,*dl;
  float *final,*dfinal,*head,*dhead; int rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n) || !ids(s,targets,n) ||
      (accumulate!=0 && accumulate!=1) || !external(s,loss,sizeof(*loss)) ||
      (uintptr_t)loss%_Alignof(double)) return BCIR_DT_INVALID;
  if (!finite_span(s->weights,s->plan.parameters) ||
      (accumulate && !finite_span(s->gradients,s->plan.parameters))) return BCIR_DT_NUMERIC;
  forward(s,tokens,g);
  if (!finite_span(s->activations,s->plan.activations)) return BCIR_DT_NUMERIC;
  if (!accumulate) memset(s->gradients,0,s->plan.parameters*sizeof(float));
  xs=s->activations; cs=xs+(g.l+1)*g.nd; norm=cs+g.l*g.la; logits=norm+g.nd;
  cur=s->scratch; tmp=cur+g.nd; dn=tmp+g.nd; dq=dn+g.nd; dk=dq+g.nd;
  dv=dk+g.nk; dc=dv+g.nk; dg=dc+g.nd; du=dg+g.nf; dh=du+g.nf; dl=dh+g.nf;
  final=s->weights+g.v*g.d+g.l*g.lp; dfinal=s->gradients+g.v*g.d+g.l*g.lp;
  head=s->plan.spec.tied ? s->weights : final+g.d;
  dhead=s->plan.spec.tied ? s->gradients : dfinal+g.d;
  for (i=0;i<g.n;i++) {
    float maxv=logits[i*g.v]; double sum=0.0;
    for (j=1;j<g.v;j++) if (logits[i*g.v+j]>maxv) maxv=logits[i*g.v+j];
    for (j=0;j<g.v;j++) sum+=exp((double)logits[i*g.v+j]-maxv);
    ce+=log(sum)+(double)maxv-logits[i*g.v+targets[i]];
    for (j=0;j<g.v;j++) dl[i*g.v+j]=(float)((exp((double)logits[i*g.v+j]-maxv)/sum-
        (j==targets[i] ? 1.0 : 0.0))/(double)g.n);
  }
  bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.v,norm,head,dl,dn,dhead,0.0f);
  bcir_tensor_rms_backward(g.n,g.d,(float)s->plan.spec.rms_eps,xs+g.l*g.nd,final,dn,cur,dfinal);
  for (l=g.l;l-->0;) {
    layer p=params(s->weights+g.v*g.d+l*g.lp,g);
    layer grad=params(s->gradients+g.v*g.d+l*g.lp,g);
    cache a=activations(cs+l*g.la,g); float *x=xs+l*g.nd;
    bcir_tensor_linear_backward(&s->provider,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f);
    bcir_tensor_swiglu_backward(g.nf,a.gate,a.up,dh,dg,du);
    bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.f,a.n2,p.gate,dg,dn,grad.gate,0.0f);
    bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.f,a.n2,p.up,du,dn,grad.up,1.0f);
    bcir_tensor_rms_backward(g.n,g.d,(float)s->plan.spec.rms_eps,a.x2,p.norm2,dn,tmp,grad.norm2);
    for (i=0;i<g.nd;i++) cur[i]+=tmp[i];
    bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.d,a.ctx,p.o,cur,dc,grad.o,0.0f);
    bcir_tensor_attention_backward(g.b,g.t,g.h,g.kh,g.hd,a.q,a.k,a.v,a.p,dc,dq,dk,dv);
    bcir_tensor_rope(g.b,g.t,g.h,g.hd,s->plan.spec.rope_base,dq,1);
    bcir_tensor_rope(g.b,g.t,g.kh,g.hd,s->plan.spec.rope_base,dk,1);
    bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.d,a.n1,p.q,dq,dn,grad.q,0.0f);
    bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.k,a.n1,p.k,dk,dn,grad.k,1.0f);
    bcir_tensor_linear_backward(&s->provider,g.n,g.d,g.k,a.n1,p.v,dv,dn,grad.v,1.0f);
    bcir_tensor_rms_backward(g.n,g.d,(float)s->plan.spec.rms_eps,x,p.norm1,dn,tmp,grad.norm1);
    for (i=0;i<g.nd;i++) cur[i]+=tmp[i];
  }
  for (i=0;i<g.n;i++) for (j=0;j<g.d;j++) s->gradients[(size_t)tokens[i]*g.d+j]+=cur[i*g.d+j];
  if (!finite_span(s->gradients,s->plan.parameters) || !isfinite(ce)) return BCIR_DT_NUMERIC;
  *loss=ce/(double)g.n; return BCIR_DT_OK;
}
static int optimizer_valid(const bcir_decoder_adamw *o) {
  return o && isfinite(o->lr) && o->lr>=0 && isfinite(o->beta1) && o->beta1>=0 && o->beta1<1 &&
    isfinite(o->beta2) && o->beta2>=0 && o->beta2<1 && isfinite(o->epsilon) && o->epsilon>0 &&
    isfinite(o->weight_decay) && o->weight_decay>=0 && isfinite(o->grad_clip) && o->grad_clip>=0;
}
int bcir_decoder_update(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    double gs,double *grad_norm) {
  size_t i,pass; double norm=0.0,scale,b1,b2; int rc=valid(s);
  bcir_decoder_adamw options;
  if (rc) return rc;
  if (!o) return BCIR_DT_INVALID;
  options=*o; o=&options;
  if (!optimizer_valid(o) || !isfinite(gs) || gs<=0 || s->step==UINT64_MAX ||
      !isfinite(s->beta1_power) || !isfinite(s->beta2_power) ||
      s->beta1_power<0 || s->beta1_power>1 || s->beta2_power<0 || s->beta2_power>1 ||
      (!s->step && (s->beta1_power!=1 || s->beta2_power!=1)) ||
      !external(s,grad_norm,sizeof(*grad_norm)) || (uintptr_t)grad_norm%_Alignof(double))
    return BCIR_DT_INVALID;
  for (i=0;i<s->plan.parameters;i++) {
    double g=(double)s->gradients[i]*gs;
    if (!isfinite(g) || !isfinite(s->weights[i]) || !isfinite(s->moment1[i]) ||
        !isfinite(s->moment2[i]) || s->moment2[i]<0) return BCIR_DT_NUMERIC;
    norm+=g*g;
  }
  norm=sqrt(norm); if (!isfinite(norm)) return BCIR_DT_NUMERIC;
  scale=gs;
  if (o->grad_clip>0 && norm>o->grad_clip) scale*=o->grad_clip/(norm+1e-6);
  b1=s->beta1_power*o->beta1; b2=s->beta2_power*o->beta2;
  /* Round moments before using them; preflight every candidate before any commit. */
  for (pass=0;pass<2;pass++) for (i=0;i<s->plan.parameters;i++) {
    double g=s->gradients[i]*scale;
    double mc=o->beta1*s->moment1[i]+(1-o->beta1)*g;
    double vc=o->beta2*s->moment2[i]+(1-o->beta2)*g*g,wc;
    float m,v,w;
    if (!isfinite(mc) || !isfinite(vc) || fabs(mc)>FLT_MAX || vc>FLT_MAX) return BCIR_DT_NUMERIC;
    m=(float)mc; v=(float)vc;
    wc=(double)s->weights[i]*(1-o->lr*o->weight_decay)-
        o->lr*((double)m/(1-b1))/(sqrt((double)v/(1-b2))+o->epsilon);
    if (!isfinite(wc) || fabs(wc)>FLT_MAX) return BCIR_DT_NUMERIC;
    w=(float)wc;
    if (pass) { s->weights[i]=w; s->moment1[i]=m; s->moment2[i]=v; }
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
