#define BCIR_DT_NO_PANELS 1
#include "bcir_decoder_train.h"
#include <math.h>
#include <string.h>
/* Experimental causal prefix geometry. Caller owns all arrays. No allocation.
 * z[0:N*R] is projected/rotated input; z[N*R:2*N*R] is prefix cache.
 * dz uses the same two-block layout, with prefix adjoints in its second block.
 * Prefix scan and reverse scan use fixed order, double accumulators.
 */
#if GP_PROJECTOR
#define GP_P ((GP_RANK*(GP_RANK+1))/2)
#else
#define GP_P ((GP_RANK*(GP_RANK-1))/2)
#endif
#define GP_RIDGE 1e-4
typedef struct gp_job {size_t batch,time;const float *z,*dp;float *p,*dz;} gp_job;

static void gp_prefix_task(void *ctx,size_t batch) {
  gp_job *j=ctx; size_t t,k,n=j->batch*j->time;
  const float *z=j->z+batch*j->time*GP_RANK;
  float *prefix=(float *)j->z+n*GP_RANK+batch*j->time*GP_RANK;
  double sum[GP_RANK]={0};
  for(t=0;t<j->time;t++) {
    double scale=t?1.0/sqrt((double)t):0;
    for(k=0;k<GP_RANK;k++) {
      prefix[t*GP_RANK+k]=(float)(sum[k]*scale);
      sum[k]+=(double)z[t*GP_RANK+k];
    }
  }
}
/* New numerical contract: guard ill-conditioned wedges, compute pair/VJP in
 * FP64 there; a plane with norm <= epsilon has zero feature and zero VJP.
 * FP32 input quantization is not reversed by this fallback.
 */
static int wedge_ill_conditioned(const float *u,const float *v) {
  size_t i;double uu=0,vv=0,uv=0,prod,det;
  for(i=0;i<GP_RANK;i++){uu+=(double)u[i]*u[i];vv+=(double)v[i]*v[i];uv+=(double)u[i]*v[i];}
  prod=uu*vv;det=prod-uv*uv;
  return prod==0 || det<=1e-4*prod;
}
static void wedge_precise_forward(const float *u,const float *v,float *out) {
  double p[GP_P],n=0;size_t i,k,c;
  for(i=0,c=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,c++){
    p[c]=(double)u[i]*v[k]-(double)u[k]*v[i];n+=p[c]*p[c];}
  n=sqrt(n);
  if(n<=1e-6){for(c=0;c<GP_P;c++)out[c]=0;return;}
  for(c=0;c<GP_P;c++)out[c]=(float)(p[c]/n);
}
static void wedge_precise_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {
  double p[GP_P],gu[GP_RANK]={0},gv[GP_RANK]={0},n=0,dot=0;size_t i,k,c;
  for(i=0,c=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,c++){
    p[c]=(double)u[i]*v[k]-(double)u[k]*v[i];n+=p[c]*p[c];dot+=p[c]*dp[c];}
  n=sqrt(n);
  if(n>1e-6)for(i=0,c=0;i<GP_RANK;i++)for(k=i+1;k<GP_RANK;k++,c++){
    double q=((double)dp[c]-p[c]*dot/(n*n))/n;
    gu[i]+=q*v[k];gu[k]-=q*v[i];gv[i]-=q*u[k];gv[k]+=q*u[i];}
  for(i=0;i<GP_RANK;i++){du[i]=(float)gu[i];dv[i]=(float)gv[i];}
}

static void gp_pair_forward(const float *u,const float *v,float *out) {
  if(wedge_ill_conditioned(u,v)){wedge_precise_forward(u,v,out);return;}
  size_t i,k,c; double aa=GP_RIDGE,bb=0,dd=GP_RIDGE;
#if GP_PROJECTOR
  double inv,a,b,d;
  for(i=0;i<GP_RANK;i++){aa+=(double)u[i]*u[i];bb+=(double)u[i]*v[i];dd+=(double)v[i]*v[i];}
  inv=1.0/(aa*dd-bb*bb);a=dd*inv;b=-bb*inv;d=aa*inv;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i;k<GP_RANK;k++,c++)
    out[c]=(float)((u[i]*(a*u[k]+b*v[k])+v[i]*(b*u[k]+d*v[k]))*0.7071067811865475244);
#else
  double norm=0,inv;
  (void)aa;(void)bb;(void)dd;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
#if GP_SYMMETRIC
    out[c]=u[i]*v[k]+u[k]*v[i];
#else
    out[c]=u[i]*v[k]-u[k]*v[i];
#endif
    norm+=(double)out[c]*out[c];
  }
  norm=sqrt(norm);inv=1.0/(norm<1e-6?1e-6:norm);
  for(c=0;c<GP_P;c++)out[c]=(float)((double)out[c]*inv);
#endif
}
static void gp_pair_backward(const float *u,const float *v,const float *dp,float *du,float *dv) {
  if(wedge_ill_conditioned(u,v)){wedge_precise_backward(u,v,dp,du,dv);return;}
  size_t i,k,c;
  for(i=0;i<GP_RANK;i++){du[i]=0;dv[i]=0;}
#if GP_PROJECTOR
  double aa=GP_RIDGE,bb=0,dd=GP_RIDGE,inv,a,b,d;
  double yu[GP_RANK]={0},yv[GP_RANK]={0},d00=0,d01=0,d11=0;
  for(i=0;i<GP_RANK;i++){aa+=(double)u[i]*u[i];bb+=(double)u[i]*v[i];dd+=(double)v[i]*v[i];}
  inv=1.0/(aa*dd-bb*bb);a=dd*inv;b=-bb*inv;d=aa*inv;
  /* S is the symmetric matrix adjoint: packed off-diagonals are halved. */
  for(i=0,c=0;i<GP_RANK;i++) for(k=i;k<GP_RANK;k++,c++) {
    double q=dp[c]*0.7071067811865475244;
    if(i==k){yu[i]+=q*u[i];yv[i]+=q*v[i];}
    else {q*=0.5;yu[i]+=q*u[k];yu[k]+=q*u[i];yv[i]+=q*v[k];yv[k]+=q*v[i];}
  }
  for(i=0;i<GP_RANK;i++){d00+=u[i]*yu[i];d01+=u[i]*yv[i];d11+=v[i]*yv[i];}
  {
    double e00=a*a*d00+2*a*b*d01+b*b*d11;
    double e01=a*b*d00+(a*d+b*b)*d01+b*d*d11;
    double e11=b*b*d00+2*b*d*d01+d*d*d11;
    for(i=0;i<GP_RANK;i++) {
      du[i]=(float)(2*(a*yu[i]+b*yv[i]-u[i]*e00-v[i]*e01));
      dv[i]=(float)(2*(b*yu[i]+d*yv[i]-u[i]*e01-v[i]*e11));
    }
  }
#else
  float p[GP_P];double norm=0,dot=0,inv,scale,correction;int clamped;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
#if GP_SYMMETRIC
    p[c]=u[i]*v[k]+u[k]*v[i];
#else
    p[c]=u[i]*v[k]-u[k]*v[i];
#endif
    norm+=(double)p[c]*p[c];dot+=(double)p[c]*dp[c];
  }
  norm=sqrt(norm);clamped=norm<=1e-6;inv=1.0/(clamped?1e-6:norm);
  scale=inv;correction=clamped?0:dot*inv*inv;
  for(i=0,c=0;i<GP_RANK;i++) for(k=i+1;k<GP_RANK;k++,c++) {
    float q=(float)(((double)dp[c]-p[c]*correction)*scale);
    du[i]+=q*v[k];dv[k]+=q*u[i];
#if GP_SYMMETRIC
    du[k]+=q*v[i];dv[i]+=q*u[k];
#else
    du[k]-=q*v[i];dv[i]-=q*u[k];
#endif
  }
#endif
}
static void gp_forward_task(void *ctx,size_t row) {
  gp_job *j=ctx;size_t n=j->batch*j->time;
  gp_pair_forward(j->z+row*GP_RANK,j->z+(n+row)*GP_RANK,j->p+row*GP_P);
}
static void gp_backward_task(void *ctx,size_t row) {
  gp_job *j=ctx;size_t n=j->batch*j->time;
  gp_pair_backward(j->z+row*GP_RANK,j->z+(n+row)*GP_RANK,j->dp+row*GP_P,
      j->dz+row*GP_RANK,j->dz+(n+row)*GP_RANK);
}
static void gp_reverse_task(void *ctx,size_t batch) {
  gp_job *j=ctx;size_t n=j->batch*j->time,t,k;
  float *dz=j->dz+batch*j->time*GP_RANK;
  const float *dv=j->dz+n*GP_RANK+batch*j->time*GP_RANK;
  double carry[GP_RANK]={0};
  for(t=j->time;t-->0;) {
    double scale=t?1.0/sqrt((double)t):0;
    for(k=0;k<GP_RANK;k++) {
      dz[t*GP_RANK+k]+=(float)carry[k];
      carry[k]+=(double)dv[t*GP_RANK+k]*scale;
    }
  }
}
void gp_wedge_forward(bcir_tensor_pool *pool,size_t batch,size_t time,const float *z,float *p) {
  gp_job j={batch,time,z,NULL,p,NULL};
  bcir_tensor_pool_run(pool,gp_prefix_task,&j,batch);
  bcir_tensor_pool_run(pool,gp_forward_task,&j,batch*time);
}
void gp_wedge_backward(bcir_tensor_pool *pool,size_t batch,size_t time,const float *z,const float *dp,float *dz) {
  gp_job j={batch,time,z,dp,NULL,dz};
  bcir_tensor_pool_run(pool,gp_backward_task,&j,batch*time);
  bcir_tensor_pool_run(pool,gp_reverse_task,&j,batch);
}
/* Deliberately direct formulation, same pair kernel: quadratic prefix building
 * and adjoint scatter. Reference for scan optimization, not a training path. */
void gp_prefix_reference(size_t batch,size_t time,const float *z,const float *dp,float *p,float *dz) {
  size_t b,t,s,k;memset(dz,0,batch*time*GP_RANK*sizeof(float));
  for(b=0;b<batch;b++)for(t=0;t<time;t++) {
    float v[GP_RANK],du[GP_RANK],dv[GP_RANK];double sum[GP_RANK]={0};
    double scale=t?1.0/sqrt((double)t):0;size_t row=b*time+t;
    for(s=0;s<t;s++)for(k=0;k<GP_RANK;k++)sum[k]+=z[(b*time+s)*GP_RANK+k];
    for(k=0;k<GP_RANK;k++)v[k]=(float)(sum[k]*scale);
    gp_pair_forward(z+row*GP_RANK,v,p+row*GP_P);
    gp_pair_backward(z+row*GP_RANK,v,dp+row*GP_P,du,dv);
    for(k=0;k<GP_RANK;k++)dz[row*GP_RANK+k]+=du[k];
    for(s=0;s<t;s++)for(k=0;k<GP_RANK;k++)dz[(b*time+s)*GP_RANK+k]+=(float)(dv[k]*scale);
  }
}

#include "bcir_decoder_train.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
    "native decoder ABI requires IEEE binary32");
uint32_t bcir_decoder_abi_version(void) { return 4; }
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
  bcir_decoder_plan p; size_t n,nd,nf,nv,nr,np,lp=0,la=0,scratch_alt=0;
  if(!s||!out||!s->vocab||!s->width||!s->heads||!s->kvheads||!s->layers||!s->ff||
      !s->batch||!s->time||s->tied>1||s->vocab>(1u<<20)||s->width>65536||
      s->layers>4096||s->batch>65536||s->time>65536||s->ff>(1u<<20)||
      !isfinite(s->rms_eps)||s->rms_eps<=0||s->rms_eps>FLT_MAX) return BCIR_DT_INVALID;
  p.spec=*s;p.parameters=p.activations=p.scratch=0;
  if(!product(s->batch,s->time,&n)||!product(n,s->width,&nd)||!product(n,s->ff,&nf)||
      !product(n,s->vocab,&nv)||!product(n,GP_RANK,&nr)||!product(n,GP_P,&np)||
      !term(&lp,2,s->width)||!term(&lp,(size_t)GP_RANK+GP_P,s->width)||
      !term(&lp,2*(size_t)s->width,s->width)||!term(&lp,3*(size_t)s->width,s->ff)||
      !term(&p.parameters,s->vocab,s->width)||!term(&p.parameters,s->layers,lp)||
      !add(&p.parameters,s->width)||(!s->tied&&!term(&p.parameters,s->vocab,s->width))||
      !term(&la,5,nd)||!term(&la,2,nr)||!add(&la,nf)||
      !term(&p.activations,(size_t)s->layers+1,nd)||!term(&p.activations,s->layers,la)||
      !add(&p.activations,nd)||!add(&p.activations,nv)||!term(&p.scratch,1,nd)||!term(&p.scratch,3,nf)||
      !product(n,3*(size_t)s->width+GP_P+2*GP_RANK,&scratch_alt)||scratch_alt>SIZE_MAX/4||p.parameters>SIZE_MAX/4||p.activations>SIZE_MAX/4||
      p.scratch>SIZE_MAX/4) return BCIR_DT_CAPACITY;
  if(scratch_alt>p.scratch)p.scratch=scratch_alt;
  *out=p;return BCIR_DT_OK;
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
      BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTN)) return BCIR_DT_INVALID;
  if (((s->contract&BCIR_DT_CONTRACT_FUSED)!=0)+((s->contract&BCIR_DT_CONTRACT_KBLOCK)!=0)+
      ((s->contract&BCIR_DT_CONTRACT_BF16)!=0)>1) return BCIR_DT_INVALID;  /* one GEMM contract */
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
  if (s->contract&BCIR_DT_CONTRACT_BF16) {  /* the GEMMs' workspace: present, large enough, apart */
    if (!s->workspace) return BCIR_DT_INVALID;
    if (s->workspace_bytes<bcir_decoder_workspace_bytes(&p,s->contract) ||
        s->workspace_bytes>UINTPTR_MAX-(uintptr_t)s->workspace) return BCIR_DT_CAPACITY;
    if (overlap(s->workspace,s->workspace_bytes,s,sizeof(*s))) return BCIR_DT_INVALID;
    for (i=0;i<6;i++) if (overlap(s->workspace,s->workspace_bytes,ptr[i],len[i]*sizeof(float)))
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
    !overlap(p,bytes,s->scratch,s->plan.scratch*sizeof(float)) &&
    !((s->contract&BCIR_DT_CONTRACT_BF16) && overlap(p,bytes,s->workspace,s->workspace_bytes));
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
  g.a=g.n*GP_P; g.lp=2*g.d+(GP_RANK+GP_P)*g.d+2*g.d*g.d+3*g.d*g.f;
  g.la=5*g.nd+2*g.n*GP_RANK+g.nf; return g;
}
static size_t ws_sum(size_t a,size_t b){return a>SIZE_MAX-b?SIZE_MAX:a+b;}
size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan,uint32_t contract){
  bcir_decoder_plan p;geometry g;size_t q,c,most=0;
  if(!(contract&BCIR_DT_CONTRACT_BF16))return 0;
  if(!plan||bcir_decoder_make_plan(&plan->spec,&p))return SIZE_MAX;g=geom(&p.spec);
  {const size_t n=g.n,d=g.d,f=g.f,v=g.v,r=GP_RANK,a=GP_P;
   /* Upper bound of each actual research-head group, including the query side channel. */
   const size_t groups[][4][3]={
    {{n,r,d},{n,d,d}},{{n,d,a}},{{n,d,r}},{{n,d,d}},
    {{n,f,d},{n,f,d}},{{n,d,f}},{{n,v,d}},
    {{n,d,v},{v,d,n}},{{n,f,d},{d,f,n}},
    {{n,d,f},{n,d,f},{f,d,n},{f,d,n}},
    {{n,d,d},{d,d,n}},{{n,a,d},{d,a,n}},
    {{n,r,d},{d,r,n}},{{n,d,r},{r,d,n}}
   };
   for(q=0;q<sizeof(groups)/sizeof(groups[0]);q++){
    size_t sum=0;for(c=0;c<4&&groups[q][c][0];c++)sum=ws_sum(sum,bcir_tensor_mm_bf16_workspace(groups[q][c][0],groups[q][c][1],groups[q][c][2]));
    if(sum>most)most=sum;
   }
  }return most;
}

typedef struct layer { float *norm1,*q,*k,*v,*o,*norm2,*gate,*up,*down; } layer;
static layer params(float *x,geometry g) {
  layer p;
  p.norm1=x; x+=g.d; p.q=x; x+=GP_RANK*g.d; p.k=x; x+=g.d*GP_P;
  p.v=x; x+=g.d*g.d; p.o=x; x+=g.d*g.d; p.norm2=x; x+=g.d;
  p.gate=x; x+=g.f*g.d; p.up=x; x+=g.f*g.d; p.down=x; return p;
}
typedef struct cache { float *n1,*q,*k,*v,*p,*ctx,*x2,*n2,*gate,*up,*hidden; } cache;
static cache activations(bcir_decoder_state *s,float *x,geometry g) {
  cache a;
  a.n1=x; x+=g.nd; a.q=x; x+=2*g.n*GP_RANK; a.k=s->scratch+3*g.nd; a.v=x; x+=g.nd;
  a.p=x; x+=g.nd; a.ctx=NULL; a.x2=x; x+=g.nd; a.n2=x; x+=g.nd;
  /* Cross alias: dg overwrites up only after its read; du writes gate last. */
  a.up=s->scratch+g.nd+g.nf;
  a.gate=a.up+g.nf;a.hidden=x;return a;
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
 * what the group would pack without them, the most the plan must hold either way. */
static void gemm_group_with(bcir_decoder_state *s,const bcir_tensor_mm_call *chain,size_t nchain,
    const bcir_tensor_mm_call *y,size_t ny,const bcir_tensor_bf16_panels *pre) {
#ifdef BCIR_DT_WORKSPACE_PROBE
  if (gemm_contract(s)==3) workspace_saw(bcir_tensor_mm_bf16_group_workspace(chain,nchain,y,ny));
#endif
  if (gemm_contract(s)==3)
    (void)bcir_tensor_mm_bf16_group_panels_pool(s->pool,chain,nchain,y,ny,pre,s->workspace,
        s->workspace_bytes);
  else bcir_tensor_mm_group_pool(s->pool,gemm_contract(s),chain,nchain,y,ny);
}
static void gemm_group(bcir_decoder_state *s,const bcir_tensor_mm_call *chain,size_t nchain,
    const bcir_tensor_mm_call *y,size_t ny) { gemm_group_with(s,chain,nchain,y,ny,NULL); }
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
 * harness's witness). */
static int panels(const bcir_decoder_state *s,geometry g,const float *a,const float *b) {
#ifdef BCIR_DT_NO_PANELS
  (void)s; (void)g; (void)a; (void)b; return 0;
#else
  return gemm_contract(s)==3 && !s->provider.mm && !s->provider.silu && g.n && g.d && g.f &&
    !(g.n%32) && !(g.d%32) && !(g.f%32) && !((uintptr_t)a%64) && !((uintptr_t)b%64);
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
static void gemm(bcir_decoder_state *s,int ta,int tb,size_t m,size_t n,size_t k,float alpha,
    const float *a,const float *b,float beta,float *c) {
  if (s->provider.mm) s->provider.mm(s->provider.ctx,ta,tb,m,n,k,alpha,a,b,beta,c);
  else if (gemm_contract(s)==3) {
#ifdef BCIR_DT_WORKSPACE_PROBE
    workspace_saw(bcir_tensor_mm_bf16_workspace(m,n,k));
#endif
    (void)bcir_tensor_mm_bf16_pool(s->pool,ta,tb,m,n,k,alpha,a,b,beta,c,s->workspace,s->workspace_bytes);
  }
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
    const float *x,const float *w,const float *dy,float *dx,float *dw,float beta,float wbeta) {
  bcir_tensor_mm_call cx,cw;
  if (s->provider.mm) {
    gemm(s,0,0,r,in,out,1.0f,dy,w,beta,dx);
    gemm(s,1,0,out,in,r,1.0f,dy,x,wbeta,dw);
    return;
  }
  cx.ta=0; cx.tb=0; cx.m=r; cx.n=in; cx.k=out; cx.alpha=1.0f; cx.a=dy; cx.b=w; cx.beta=beta; cx.c=dx;
  cw.ta=1; cw.tb=0; cw.m=out; cw.n=in; cw.k=r; cw.alpha=1.0f; cw.a=dy; cw.b=x; cw.beta=wbeta; cw.c=dw;
  if (gemm_contract(s)==3) gemm_group(s,&cx,1,&cw,1);
  else bcir_tensor_mm_pair_pool(s->pool,gemm_contract(s),&cx,&cw);
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
  gemm_group(s,NULL,0,c,count);
}
static void linears_backward(bcir_decoder_state *s,size_t r,size_t in,const float *x,size_t count,
    const size_t *out,const float *const *w,const float *const *dy,float *dx,float *const *dw,
    float beta,float wbeta) {
  bcir_tensor_mm_call cx[BCIR_TENSOR_MM_GROUP],cw[BCIR_TENSOR_MM_GROUP]; size_t q;
  if (s->provider.mm || count>BCIR_TENSOR_MM_GROUP) {
    for (q=0;q<count;q++) linear_backward(s,r,in,out[q],x,w[q],dy[q],dx,dw[q],q ? 1.0f : beta,wbeta);
    return;
  }
  for (q=0;q<count;q++) {
    cx[q].ta=0; cx[q].tb=0; cx[q].m=r; cx[q].n=in; cx[q].k=out[q]; cx[q].alpha=1.0f; cx[q].a=dy[q];
    cx[q].b=w[q]; cx[q].beta=q ? 1.0f : beta; cx[q].c=dx;
    cw[q].ta=1; cw[q].tb=0; cw[q].m=out[q]; cw[q].n=in; cw[q].k=r; cw[q].alpha=1.0f; cw[q].a=dy[q];
    cw[q].b=x; cw[q].beta=wbeta; cw[q].c=dw[q];
  }
  gemm_group(s,cx,count,cw,count);
}
/* y = silu(x) * u (NDT-SWIGLU-FWD): BCIR's SiLU under the state's contract and the product in one
 * pass, the two passes' values; a provider's SiLU into y, then the product, as before. *bad
 * answers for x, u and y (NDT-SCAN-W): BCIR's product from its tasks, a provider's by scanning the
 * three after it. */
static void silu_mul(bcir_decoder_state *s,size_t n,const float *x,const float *u,float *y,
    int *bad) {
  if (s->provider.silu) {
    s->provider.silu(s->provider.ctx,n,x,y); bcir_tensor_mul_pool(s->pool,n,u,y);
    if (!finite_pool(s,x,n) || !finite_pool(s,u,n) || !finite_pool(s,y,n)) *bad=1;
  }
  else if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_silu_mul_v2_scan_pool(s->pool,n,x,u,y,bad);
  else bcir_tensor_silu_mul_scan_pool(s->pool,n,x,u,y,bad);
}
static int swiglu_v2(const bcir_decoder_state *s){return (s->contract&BCIR_DT_CONTRACT_EXP)!=0;}
static void swiglu_backward(bcir_decoder_state *s,size_t n,const float *g,const float *u,
    const float *dy,float *dg,float *du) {
  if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_swiglu_backward_v2_pool(s->pool,n,g,u,dy,dg,du);
  else bcir_tensor_swiglu_backward_pool(s->pool,n,g,u,dy,dg,du);
}
static void attention(bcir_decoder_state *s,size_t b,size_t t,size_t h,size_t kh,size_t d,
    const float *q,const float *k,const float *v,float *p,float *y,int *bad) {
  if (s->contract&BCIR_DT_CONTRACT_EXP) bcir_tensor_attention_v2_scan_pool(s->pool,b,t,h,kh,d,q,k,v,p,y,bad);
  else bcir_tensor_attention_scan_pool(s->pool,b,t,h,kh,d,q,k,v,p,y,bad);
}

typedef struct wedge_gate_job {size_t n;const float *h,*g,*cur;float *a,*x,*dc,*dv,*dn;} wedge_gate_job;
static void wedge_gate_fwd_task(void *ctx,size_t task) {
  wedge_gate_job *j=ctx;size_t i,end=(task+1)*512;if(end>j->n)end=j->n;
  for(i=task*512;i<end;i++) {
    float e=bcir_tensor_expf(-(j->a[i]+j->x[i]));
    j->a[i]=1.0f/(1.0f+e);j->x[i]=j->a[i]*j->h[i]+(1.0f-j->a[i])*j->g[i];
  }
}
static void wedge_gate_bwd_task(void *ctx,size_t task) {
  wedge_gate_job *j=ctx;size_t i,end=(task+1)*512;if(end>j->n)end=j->n;
  for(i=task*512;i<end;i++) {float alpha=j->a[i];
    j->dc[i]=j->cur[i]*(j->h[i]-j->g[i])*alpha*(1.0f-alpha);
    j->dv[i]=j->cur[i]*(1.0f-alpha);j->dn[i]=j->cur[i]*alpha;
  }
}

/* The forward, and whether every activation it wrote is finite (NDT-SCAN-W). One scan of the
 * whole arena after it read the arena back from memory, 171 MB for the 9.84M decoder; now the
 * writers answer for it from their tasks (bcir_tensor_rms_scan_pool): each layer's two RMSNorms
 * for their input rows -- the residual stream, and the attention's sum -- and their outputs, the
 * attention for q, k, v, the probabilities and the context, the SwiGLU product for the gate, up
 * and hidden activations, the final norm for the last layer's output and its own; the logits are
 * scanned after the head. Every element of the arena is in exactly one call's regions, each
 * called after the region's last write, so the verdict is the one scan's, and every layer runs
 * whatever an earlier one held. */
static int forward(bcir_decoder_state *s,const uint32_t *tokens,geometry g) {
  const bcir_decoder_spec *sp=&s->plan.spec;
  size_t i,l; float *xs=s->activations,*cs=xs+(g.l+1)*g.nd;
  float *norm=cs+g.l*g.la,*logits=norm+g.nd;
  float *final=s->weights+g.v*g.d+g.l*g.lp;
  const float *head=sp->tied ? s->weights : final+g.d; int bad=0;
  for (i=0;i<g.n;i++) memcpy(xs+i*g.d,s->weights+(size_t)tokens[i]*g.d,g.d*sizeof(float));
  for (l=0;l<g.l;l++) {
    layer p=params(s->weights+g.v*g.d+l*g.lp,g);
    cache a=activations(s,cs+l*g.la,g); float *x=xs+l*g.nd,*y=x+g.nd;
    bcir_tensor_rms_scan_pool(s->pool,g.n,g.d,(float)sp->rms_eps,x,p.norm1,a.n1,&bad);
    {
      size_t out[2]={GP_RANK,g.d};const float *w[2]={p.q,p.v};float *yy[2]={a.q,a.p};
      linears(s,g.n,g.d,a.n1,2,out,w,yy);
    }
    /* Position ablation: identity map. */
    gp_wedge_forward(s->pool,g.b,g.t,a.q,a.k);
    linear(s,g.n,GP_P,g.d,a.k,p.k,a.v);
    /* gate_h grouped with reduce */
    linear(s,g.n,g.d,g.d,a.v,p.o,a.x2);
    {wedge_gate_job j={g.nd,a.n1,a.v,NULL,a.p,a.x2,NULL,NULL,NULL};
      bcir_tensor_pool_run(s->pool,wedge_gate_fwd_task,&j,(g.nd+511)/512);}
    if(!finite_pool(s,a.q,g.n*GP_RANK)||!finite_pool(s,a.k,g.n*GP_P)||
        !finite_pool(s,a.v,g.nd)||!finite_pool(s,a.p,g.nd)) bad=1;
    bcir_tensor_add_pool(s->pool,g.nd,x,a.x2);
    bcir_tensor_rms_scan_pool(s->pool,g.n,g.d,(float)sp->rms_eps,a.x2,p.norm2,a.n2,&bad);
    {
      size_t out[2]; const float *w[2]; float *y2[2];
      out[0]=g.f; out[1]=g.f; w[0]=p.gate; w[1]=p.up; y2[0]=a.gate; y2[1]=a.up;
      linears(s,g.n,g.d,a.n2,2,out,w,y2);
    }
    if (panels(s,g,a.hidden,a.hidden)) {
      bcir_tensor_mm_call c; bcir_tensor_bf16_panels pre;
      (void)bcir_tensor_silu_mul_panels_pool(s->pool,swiglu_v2(s),g.n,g.f,a.gate,a.up,
          panel_of(a.hidden),panel_of(a.hidden)+g.nf,&bad);
      c.ta=0; c.tb=1; c.m=g.n; c.n=g.d; c.k=g.f; c.alpha=1.0f; c.a=NULL; c.b=p.down; c.beta=0.0f; c.c=y;
      pre.a=panel_of(a.hidden); pre.b=NULL;
      gemm_group_with(s,NULL,0,&c,1,&pre);
      PANELS_SAW(1u);
    }
    else {
      silu_mul(s,g.nf,a.gate,a.up,a.hidden,&bad);
      linear(s,g.n,g.f,g.d,a.hidden,p.down,y);
    }
    bcir_tensor_add_pool(s->pool,g.nd,a.x2,y);
  }
  bcir_tensor_rms_scan_pool(s->pool,g.n,g.d,(float)sp->rms_eps,xs+g.l*g.nd,final,norm,&bad);
  linear(s,g.n,g.d,g.v,norm,head,logits);
  if (!finite_pool(s,logits,g.nv)) bad=1;
  return !bad;
}
int bcir_decoder_forward(bcir_decoder_state *s,const uint32_t *tokens,size_t n) {
  geometry g; int rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n)) return BCIR_DT_INVALID;
  if (!finite_pool(s,s->weights,s->plan.parameters)) return BCIR_DT_NUMERIC;
  return forward(s,tokens,g) ? BCIR_DT_OK : BCIR_DT_NUMERIC;
}
/* The loss's rows in ranges of BCIR_DT_ROW_CHUNK: a row's softmax, its adjoint and its term
 * (log-sum-exp plus the maximum less the target's logit, in double, as the serial loop computed
 * it) are its own; the terms are kept as bytes in the scratch arena and added in row order by the
 * caller, so the mean is the serial chain. */
#define BCIR_DT_ROW_CHUNK 16
#define BCIR_DT_COL_CHUNK 16
static int target_ids(bcir_decoder_state *s,const uint32_t *x,size_t n) {
  size_t i;
  if(n>SIZE_MAX/sizeof(uint32_t) || !external(s,x,n*sizeof(uint32_t)) ||
      (uintptr_t)x%_Alignof(uint32_t)) return 0;
  for(i=0;i<n;i++) if(x[i]!=UINT32_MAX && x[i]>=s->plan.spec.vocab) return 0;
  return 1;
}
typedef struct loss_job {
  size_t n,v,denom; const float *logits; const uint32_t *targets; float *dl; unsigned char *terms;
} loss_job;
static void loss_task(void *ctx,size_t task) {
  loss_job *jb=(loss_job *)ctx; size_t i,j,v=jb->v;
  size_t i0=task*BCIR_DT_ROW_CHUNK,i1=i0+BCIR_DT_ROW_CHUNK<jb->n ? i0+BCIR_DT_ROW_CHUNK : jb->n;
  for (i=i0;i<i1;i++) {
    const float *row=jb->logits+i*v; float *d=jb->dl+i*v; float maxv=row[0]; double sum=0.0,term;
    if(jb->targets[i]==UINT32_MAX) {
      term=0; memset(d,0,v*sizeof(float));
      memcpy(jb->terms+i*sizeof(double),&term,sizeof(double)); continue;
    }
    for (j=1;j<v;j++) if (row[j]>maxv) maxv=row[j];
    for (j=0;j<v;j++) sum+=exp((double)row[j]-maxv);
    term=log(sum)+(double)maxv-row[jb->targets[i]];
    for (j=0;j<v;j++) d[j]=(float)((exp((double)row[j]-maxv)/sum-
        (j==jb->targets[i] ? 1.0 : 0.0))/(double)jb->denom);
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
static int loss_backward_with(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t n,int accumulate,double *loss,unsigned how) {
  geometry g; size_t i,l,supervised=0; double ce=0.0; loss_job lj; scatter_job sj;
  float *xs,*cs,*norm,*logits,*cur,*tmp,*dn,*dq,*dk,*dv,*dc,*dg,*du,*dh,*dl;
  float *final,*dfinal,*head,*dhead,wb; unsigned gf; int first,rc=valid(s);
  if (rc) return rc;
  g=geom(&s->plan.spec);
  if (n!=g.n || !ids(s,tokens,n) || !target_ids(s,targets,n) ||
      (accumulate!=0 && accumulate!=1) || !external(s,loss,sizeof(*loss)) ||
      (uintptr_t)loss%_Alignof(double)) return BCIR_DT_INVALID;
  for(i=0;i<n;i++) supervised+=(targets[i]!=UINT32_MAX);
  if(!supervised) return BCIR_DT_INVALID;
  if (!(how&DT_KNOWN) && (!finite_pool(s,s->weights,s->plan.parameters) ||
      (accumulate && !finite_pool(s,s->gradients,s->plan.parameters)))) return BCIR_DT_NUMERIC;
  if (!forward(s,tokens,g)) return BCIR_DT_NUMERIC;
  /* NDT-ZERO: a fresh gradient's first writers write it -- each weight adjoint's GEMM at beta 0,
   * which adds its terms to +0.0 without reading C (what the fill and beta 1 gave), each gamma
   * VJP from +0.0 -- so only the region the scatter writes first, an untied embedding, is filled;
   * a provider's GEMM keeps the fill and beta 1. */
  first=!accumulate && !s->provider.mm; wb=first ? 0.0f : 1.0f; gf=first ? BCIR_TENSOR_RMSB_FIRST : 0u;
  if (!accumulate && !first) zero_pool(s,s->gradients,s->plan.parameters);
  if (!accumulate && first && !s->plan.spec.tied) zero_pool(s,s->gradients,g.v*g.d);
  xs=s->activations; cs=xs+(g.l+1)*g.nd; norm=cs+g.l*g.la; logits=norm+g.nd;
  cur=s->scratch;tmp=cur;dn=cur+g.nd;dq=cur+3*g.nd+g.n*GP_P;dk=cur+3*g.nd;
  dv=cur+2*g.nd;dc=cur+3*g.nd;dh=cur+g.nd;dg=dh+g.nf;du=dg+g.nf;dl=logits;
  final=s->weights+g.v*g.d+g.l*g.lp; dfinal=s->gradients+g.v*g.d+g.l*g.lp;
  head=s->plan.spec.tied ? s->weights : final+g.d;
  dhead=s->plan.spec.tied ? s->gradients : dfinal+g.d;
  /* the rows' terms live at the arena's start (8n bytes within cur, tmp and dn: regions the
   * head's VJP writes only after they are added) */
  lj.n=g.n; lj.v=g.v; lj.denom=supervised; lj.logits=logits; lj.targets=targets; lj.dl=dl;
  lj.terms=(unsigned char *)s->scratch;
  bcir_tensor_pool_run(s->pool,loss_task,&lj,(g.n+BCIR_DT_ROW_CHUNK-1)/BCIR_DT_ROW_CHUNK);
  for (i=0;i<g.n;i++) {
    double term; memcpy(&term,lj.terms+i*sizeof(double),sizeof(double)); ce+=term;
  }
  linear_backward(s,g.n,g.d,g.v,norm,head,dl,dn,dhead,0.0f,wb);
  /* each gamma VJP keeps its rows' 1/rms in dq, free until the layer's query VJP */
  rms_backward(s,g.n,g.d,xs+g.l*g.nd,final,dn,cur,dfinal,dq,gf);
  for (l=g.l;l-->0;) {
    layer p=params(s->weights+g.v*g.d+l*g.lp,g);
    layer grad=params(s->gradients+g.v*g.d+l*g.lp,g);
    cache a=activations(s,cs+l*g.la,g); float *x=xs+l*g.nd;
    {
      size_t out[2]={g.f,g.f};const float *w[2]={p.gate,p.up};float *yy[2]={a.gate,a.up};
      linears(s,g.n,g.d,a.n2,2,out,w,yy);
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
      gemm_group_with(s,&cx,1,&cw,1,pre);
      PANELS_SAW(2u);
    }
    else linear_backward(s,g.n,g.f,g.d,a.hidden,p.down,cur,dh,grad.down,0.0f,wb);
    if (panels(s,g,dg,du)) {
      bcir_tensor_mm_call cx[2],cw[2]; bcir_tensor_bf16_panels pre[4]; size_t q;
      (void)bcir_tensor_swiglu_backward_panels_pool(s->pool,swiglu_v2(s),g.n,g.f,a.gate,a.up,dh,
          panel_of(dg),panel_of(dg)+g.nf,panel_of(du),panel_of(du)+g.nf);
      for (q=0;q<2;q++) {
        float *slot=q ? du : dg;
        cx[q].ta=0; cx[q].tb=0; cx[q].m=g.n; cx[q].n=g.d; cx[q].k=g.f; cx[q].alpha=1.0f; cx[q].a=NULL;
        cx[q].b=q ? p.up : p.gate; cx[q].beta=q ? 1.0f : 0.0f; cx[q].c=dn;
        cw[q].ta=1; cw[q].tb=0; cw[q].m=g.f; cw[q].n=g.d; cw[q].k=g.n; cw[q].alpha=1.0f; cw[q].a=NULL;
        cw[q].b=a.n2; cw[q].beta=wb; cw[q].c=q ? grad.up : grad.gate;
        pre[q].a=panel_of(slot); pre[q].b=NULL; pre[2+q].a=panel_of(slot)+g.nf; pre[2+q].b=NULL;
      }
      gemm_group_with(s,cx,2,cw,2,pre);
      PANELS_SAW(4u);
    }
    else {
      swiglu_backward(s,g.nf,a.gate,a.up,dh,dg,du);
      {
        size_t out[2]; const float *w[2],*d2[2]; float *dw[2];
        out[0]=g.f; out[1]=g.f; w[0]=p.gate; w[1]=p.up; d2[0]=dg; d2[1]=du; dw[0]=grad.gate; dw[1]=grad.up;
        linears_backward(s,g.n,g.d,a.n2,2,out,w,d2,dn,dw,0.0f,wb);
      }
    }
    /* NDT-RMS-ACC: the VJP adds its dx into the residual's adjoint, one rounded sum an element */
    rms_backward(s,g.n,g.d,a.x2,p.norm2,dn,cur,grad.norm2,dq,BCIR_TENSOR_RMSB_ADD|gf);
    {wedge_gate_job j={g.nd,a.n1,a.v,cur,a.p,NULL,dc,dv,dn};
      bcir_tensor_pool_run(s->pool,wedge_gate_bwd_task,&j,(g.nd+511)/512);}
    linear_backward(s,g.n,g.d,g.d,a.n1,p.v,dc,dn,grad.v,1.0f,wb);
    linear_backward(s,g.n,g.d,g.d,a.v,p.o,dc,dv,grad.o,1.0f,wb);
    gp_wedge_forward(s->pool,g.b,g.t,a.q,a.k);
    /* Weight VJP consumes p before the input VJP reuses its address. */
    gemm(s,1,0,g.d,GP_P,g.n,1.0f,dv,a.k,wb,grad.k);
    gemm(s,0,0,g.n,GP_P,g.d,1.0f,dv,p.k,0.0f,dk);
    gp_wedge_backward(s->pool,g.b,g.t,a.q,dk,dq);
    /* Position ablation: identity map. */
    linear_backward(s,g.n,g.d,GP_RANK,a.n1,p.q,dq,dn,grad.q,1.0f,wb);
    rms_backward(s,g.n,g.d,x,p.norm1,dn,cur,grad.norm1,dq,BCIR_TENSOR_RMSB_ADD|gf);
  }
  sj.n=g.n; sj.d=g.d; sj.tokens=tokens; sj.dx=cur; sj.grad=s->gradients;
  bcir_tensor_pool_run(s->pool,scatter_task,&sj,(g.d+BCIR_DT_COL_CHUNK-1)/BCIR_DT_COL_CHUNK);
  if ((!(how&DT_DEFER) && !finite_pool(s,s->gradients,s->plan.parameters)) || !isfinite(ce))
    return BCIR_DT_NUMERIC;
  *loss=ce/(double)supervised; return BCIR_DT_OK;
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
 * gradient is not finite reports the backward's verdict first, as the separate calls do. */
static int update_with(bcir_decoder_state *s,const bcir_decoder_adamw *o,
    double gs,double *grad_norm,int deferred) {
  size_t i; double norm,scale,b1,b2; int rc=valid(s);
  bcir_decoder_adamw options; pass_part parts[BCIR_DT_PASS_TASKS]; pass_result pass;
  if (rc) return rc;
  if (!o) return deferred && !finite_pool(s,s->gradients,s->plan.parameters) ? BCIR_DT_NUMERIC :
    BCIR_DT_INVALID;
  options=*o; o=&options;
  if (!optimizer_valid(o) || !isfinite(gs) || gs<=0 || s->step==UINT64_MAX ||
      !isfinite(s->beta1_power) || !isfinite(s->beta2_power) ||
      s->beta1_power<0 || s->beta1_power>1 || s->beta2_power<0 || s->beta2_power>1 ||
      (!s->step && (s->beta1_power!=1 || s->beta2_power!=1)) ||
      !external(s,grad_norm,sizeof(*grad_norm)) || (uintptr_t)grad_norm%_Alignof(double))
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
  return update_with(s,o,gs,grad_norm,0);
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
  if (steps && (!ids(s,tokens,total) || !target_ids(s,targets,total) ||
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
      rc=loss_backward_with(s,tokens+off,targets+off,n,j!=0,&part,
          (j ? DT_KNOWN : 0u)|(j+1==accumulation ? DT_DEFER : 0u));
      if (rc) return rc;
      loss+=part/(double)accumulation;
    }
    if (rates) opt.lr=rates[i];
    rc=update_with(s,&opt,1.0/(double)accumulation,&norm,1); if (rc) return rc;
    events[i].loss=loss; events[i].grad_norm=norm; events[i].step=s->step; *completed=i+1;
  }
  return BCIR_DT_OK;
}
