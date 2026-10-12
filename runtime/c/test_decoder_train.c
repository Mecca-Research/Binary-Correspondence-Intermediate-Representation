/* Complete decoder gradients, memory contracts, atomic updates and learning. */
#include "bcir_decoder_train.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float w[4096],grad[4096],m[4096],v[4096],act[32768],scratch[32768];
static bcir_decoder_state fixture(void) {
  bcir_decoder_state s; bcir_decoder_spec spec={16,8,2,1,2,16,1,4,1,10000.0,1e-6};
  size_t i,l,off=128;
  memset(&s,0,sizeof(s));
  assert(bcir_decoder_make_plan(&spec,&s.plan)==0);
  s.weights=w; s.gradients=grad; s.moment1=m; s.moment2=v;
  s.activations=act; s.scratch=scratch;
  s.weight_capacity=s.gradient_capacity=s.moment1_capacity=s.moment2_capacity=4096;
  s.activation_capacity=s.scratch_capacity=32768;
  s.beta1_power=s.beta2_power=1.0;
  memset(m,0,sizeof(m)); memset(v,0,sizeof(v)); memset(grad,0,sizeof(grad));
  for (i=0;i<4096;i++) w[i]=0.1f*sinf((float)(i+1)*0.37f);
  for (l=0;l<2;l++) {
    for (i=0;i<8;i++) w[off+i]=1.0f;
    off+=8+64+32+32+64;
    for (i=0;i<8;i++) w[off+i]=1.0f;
    off+=8+3*128;
  }
  for (i=0;i<8;i++) w[off+i]=1.0f;
  return s;
}
static void learning(void) {
  bcir_decoder_state s=fixture(); bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,1.0};
  uint32_t x[160],y[160]; bcir_decoder_event events[40]; size_t done,i;
  double initial,final;
  for (i=0;i<160;i++) { x[i]=(uint32_t)(i%4+1); y[i]=(uint32_t)(i%4+2); }
  assert(bcir_decoder_loss_backward(&s,x,y,4,0,&initial)==0);
  assert(bcir_decoder_run(&s,&opt,x,y,160,NULL,40,events,40,&done)==0);
  assert(done==40 && s.step==40 && events[39].step==40);
  assert(bcir_decoder_loss_backward(&s,x,y,4,0,&final)==0);
  assert(final<initial*0.4);
  puts("  PASS native decoder learning: all-parameter GQA/SwiGLU AdamW loop");
}
static void finite_differences(void) {
  bcir_decoder_state s=fixture(); uint32_t x[]={1,2,1,4},y[]={2,3,4,5};
  float expected[4096]; double loss,plus,minus; size_t i;
  assert(bcir_decoder_loss_backward(&s,x,y,4,0,&loss)==0);
  memcpy(expected,grad,sizeof(expected));
  for (i=0;i<s.plan.parameters;i++) {
    float original=w[i]; double numerical;
    w[i]=original+0.002f;
    assert(bcir_decoder_loss_backward(&s,x,y,4,0,&plus)==0);
    w[i]=original-0.002f;
    assert(bcir_decoder_loss_backward(&s,x,y,4,0,&minus)==0);
    w[i]=original; numerical=(plus-minus)/0.004;
    assert(fabs(numerical-expected[i])<0.0004+0.03*fabs(numerical));
  }
  assert(bcir_decoder_loss_backward(&s,x,y,4,0,&loss)==0);
  memcpy(expected,grad,sizeof(expected));
  assert(bcir_decoder_loss_backward(&s,x,y,4,1,&loss)==0);
  for (i=0;i<s.plan.parameters;i++) assert(fabs(grad[i]-2*expected[i])<1e-6);
  puts("  PASS native decoder every-parameter finite differences + accumulation");
}
static void refusals(void) {
  bcir_decoder_state s=fixture(),bad;
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,1.0};
  uint32_t x[]={1,2,3,4},y[]={2,3,4,5},badx[]={1,2,3,16}; double loss,norm;
  float oldw[4096],oldm[4096],oldv[4096]; bcir_decoder_event event;
  union { bcir_decoder_event event; uint32_t ids[6]; } aliased={.ids={1,2,3,4}};
  bcir_decoder_plan oldplan=s.plan,p=s.plan; bcir_decoder_spec shape=p.spec;
  size_t done=99,i;
  shape.heads=3;
  assert(bcir_decoder_make_plan(&shape,&p)==BCIR_DT_INVALID && p.parameters==oldplan.parameters);
  shape=oldplan.spec; shape.rms_eps=1e300;
  assert(bcir_decoder_make_plan(&shape,&p)==BCIR_DT_INVALID && p.parameters==oldplan.parameters);
  shape=(bcir_decoder_spec){1048576,65536,32768,32768,4096,1048576,65536,65536,1,10000.0,1e-6};
  assert(bcir_decoder_make_plan(&shape,&p)==BCIR_DT_CAPACITY && p.parameters==oldplan.parameters);
  bad=s; bad.plan.scratch++;
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  bad=s; bad.activation_capacity=bad.plan.activations-1; act[0]=123;
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_CAPACITY && act[0]==123);
  bad=s; bad.gradients=w;
  assert(bcir_decoder_loss_backward(&bad,x,y,4,0,&loss)==BCIR_DT_INVALID);
  assert(bcir_decoder_forward(&s,badx,4)==BCIR_DT_INVALID && act[0]==123);
  /* a non-finite weight or accumulated gradient in any block of its span, the last element
   * included, is refused before the forward pass runs: the activations stay untouched */
  for (i=0;i<3;i++) {
    size_t at=i==0 ? 5 : i==1 ? 1300 : s.plan.parameters-1; float keep=w[at];
    w[at]=i==1 ? NAN : INFINITY; act[0]=777;
    assert(bcir_decoder_forward(&s,x,4)==BCIR_DT_NUMERIC && act[0]==777);
    w[at]=keep; grad[at]=i==1 ? NAN : -INFINITY;
    assert(bcir_decoder_loss_backward(&s,x,y,4,1,&loss)==BCIR_DT_NUMERIC && act[0]==777);
    grad[at]=0;
  }
  assert(bcir_decoder_loss_backward(&s,x,y,4,0,&loss)==0);
  memcpy(oldw,w,sizeof(w)); memcpy(oldm,m,sizeof(m)); memcpy(oldv,v,sizeof(v));
  opt.lr=1e300; opt.weight_decay=1e300;
  assert(bcir_decoder_update(&s,&opt,1,&norm)==BCIR_DT_NUMERIC);
  assert(!memcmp(oldw,w,sizeof(w)) && !memcmp(oldm,m,sizeof(m)) && !memcmp(oldv,v,sizeof(v)) && s.step==0);
  opt.lr=0.01; opt.weight_decay=0.01; v[0]=-1;
  assert(bcir_decoder_update(&s,&opt,1,&norm)==BCIR_DT_NUMERIC && s.step==0);
  v[0]=0; grad[0]=NAN;
  assert(bcir_decoder_update(&s,&opt,1,&norm)==BCIR_DT_NUMERIC && s.step==0);
  grad[0]=0;
  /* a finite gradient whose second moment leaves float's range: refused, nothing written */
  opt.grad_clip=0; grad[7]=FLT_MAX;
  assert(bcir_decoder_update(&s,&opt,1,&norm)==BCIR_DT_NUMERIC && s.step==0);
  assert(!memcmp(oldw,w,sizeof(w)) && !memcmp(oldm,m,sizeof(m)) && !memcmp(oldv,v,sizeof(v)));
  grad[7]=0; opt.grad_clip=1.0;
  assert(bcir_decoder_run(&s,&opt,badx,y,4,NULL,1,&event,1,&done)==BCIR_DT_INVALID);
  assert(done==99 && !memcmp(oldw,w,sizeof(w)));
  assert(bcir_decoder_run(&s,&opt,x,y,4,NULL,1,&event,0,&done)==BCIR_DT_CAPACITY);
  assert(bcir_decoder_run(&s,&opt,aliased.ids,y,4,NULL,1,&aliased.event,1,&done)==BCIR_DT_INVALID);
  assert(bcir_decoder_run(&s,&opt,x,y,4,NULL,1,&event,1,(size_t *)&event)==BCIR_DT_INVALID);
  assert(s.step==0 && done==99);
  puts("  PASS native decoder bounds/IDs/alias and atomic optimizer refusals");
}
/* The GEMM contract every provider path keeps (bcir_tensor.h): beta first, then alpha*a
 * times b added for k ascending, each operation rounded, no contraction -- checked bit for
 * bit against the naive loop over every transpose pair, the register blocks' and tiles' edges
 * (2/4 rows, 32/64 columns, 64 deep) and every instruction-set variant this host can run. */
static void gemm_contract(void) {
  static float a[140*140],b[140*140],c[140*140],want[140*140],init[140*140];
  static const size_t shapes[][3]={{1,1,1},{3,33,5},{33,31,65},{32,32,32},{7,65,40},{64,1,33},
                                   {5,129,130},{66,64,64},{9,63,128},{4,96,1},{130,7,3}};
  size_t s,i,j,l,ta,tb; int variant,ran=0,mask=bcir_tensor_mm_variants();
  assert(mask&1 && !(mask&~7));
  for (i=0;i<140*140;i++) {
    a[i]=sinf((float)i*0.71f)*(i%3 ? 4.0f : 0.125f); b[i]=cosf((float)i*1.37f)-0.25f;
  }
  for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (ta=0;ta<2;ta++) for (tb=0;tb<2;tb++) {
    size_t m=shapes[s][0],n=shapes[s][1],k=shapes[s][2];
    float alpha=s%2 ? 0.75f : 1.0f,beta=s%3==1 ? 1.0f : s%3==2 ? 0.5f : 0.0f;
    for (i=0;i<m*n;i++) init[i]=0.01f*(float)(i%17)-0.05f;
    for (i=0;i<m*n;i++) want[i]=beta==0.0f ? 0.0f : beta*init[i];
    for (i=0;i<m;i++) for (j=0;j<n;j++) for (l=0;l<k;l++) {
      /* each product rounded on its own, whatever contraction the harness is built with:
       * clang's default -ffp-contract=on fuses a*b+c into one FMA where the target has one
       * (every aarch64), which is not the contract the kernel keeps */
      volatile float prod=alpha*a[ta ? l*m+i : i*k+l]*b[tb ? j*k+l : l*n+j];
      want[i*n+j]+=prod;
    }
    memcpy(c,init,m*n*sizeof(float));
    bcir_tensor_mm(NULL,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c);
    assert(!memcmp(c,want,m*n*sizeof(float)));
    for (variant=1;variant<=4;variant*=2) {
      memcpy(c,init,m*n*sizeof(float));
      if (!(mask&variant)) {
        /* a variant this host cannot run is refused and writes nothing */
        assert(bcir_tensor_mm_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c)==0);
        assert(!memcmp(c,init,m*n*sizeof(float)));
        continue;
      }
      assert(bcir_tensor_mm_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c)==variant);
      assert(!memcmp(c,want,m*n*sizeof(float)));
      ran++;
    }
  }
  assert(bcir_tensor_mm_variant(3,0,0,1,1,1,1.0f,a,b,0.0f,c)==0);
  printf("  PASS native GEMM contract: bit-identical to the ascending-k loop, every transpose, "
      "%d variant run(s) (mask %d)\n",ran,mask);
}
/* The attention kernels as they were written before the GEMM kernel carried them -- one scalar
 * dot product per score and per probability adjoint -- kept here as the reference both the
 * forward and the in-place backward must match bit for bit. Products go through volatile so
 * the harness's own contraction setting cannot fuse them. */
static void ref_attention(size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q,
    const float *k, const float *v, float *p, float *y) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(y,0,b*t*nh*d*sizeof(float)); memset(p,0,b*nh*t*t*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t;
    float maxv=-INFINITY,sum=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float z=0.0f;
      for (c=0;c<d;c++) { volatile float prod=q[qi+c]*k[kj+c]; z+=prod; }
      p[pi+j]=z*scale; if (p[pi+j]>maxv) maxv=p[pi+j];
    }
    for (j=0;j<=i;j++) { p[pi+j]=expf(p[pi+j]-maxv); sum+=p[pi+j]; }
    for (j=0;j<=i;j++) {
      size_t vj=((r*t+j)*nk+kh)*d; p[pi+j]/=sum;
      for (c=0;c<d;c++) { volatile float prod=p[pi+j]*v[vj+c]; y[qi+c]+=prod; }
    }
  }
}
static void ref_attention_backward(size_t b, size_t t, size_t nh, size_t nk, size_t d,
    const float *q, const float *k, const float *v, const float *p, const float *dy, float *dq,
    float *dk, float *dv) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(dq,0,b*t*nh*d*sizeof(float)); memset(dk,0,b*t*nk*d*sizeof(float));
  memset(dv,0,b*t*nk*d*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t; float dot=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f;
      for (c=0;c<d;c++) { volatile float prod=dy[qi+c]*v[kj+c]; dp+=prod; }
      { volatile float prod=p[pi+j]*dp; dot+=prod; }
    }
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f,ds;
      for (c=0;c<d;c++) { volatile float prod=dy[qi+c]*v[kj+c]; dp+=prod; }
      ds=p[pi+j]*(dp-dot)*scale;
      for (c=0;c<d;c++) {
        volatile float pq=ds*k[kj+c],pk=ds*q[qi+c],pv=p[pi+j]*dy[qi+c];
        dq[qi+c]+=pq; dk[kj+c]+=pk; dv[kj+c]+=pv;
      }
    }
  }
}
/* Causal GQA attention through the GEMM kernel against the reference loops above, bit for bit:
 * single rows and channels, head groups of one to four, widths past the 64-deep tile, contexts
 * past the backward's 32-row chunks and 64-column tiles -- and a non-finite operand, which the
 * masked products must keep out of every result the reference keeps it out of. */
static void attention_reference(void) {
  static float q[2*70*8*48],k[2*70*4*130],v[2*70*4*130],dy[2*70*8*48],p0[2*8*70*70],
      p1[2*8*70*70],y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],
      dk0[2*70*4*130],dk1[2*70*4*130],dv0[2*70*4*130],dv1[2*70*4*130];
  static const size_t shapes[][5]={{1,1,1,1,2},{1,2,1,1,1},{2,5,4,2,6},{1,37,2,1,48},
      {2,70,8,4,48},{1,33,3,1,65},{1,40,2,2,130},{2,64,4,1,24}};
  size_t s,i,pass;
  for (pass=0;pass<2;pass++) for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
    size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
    size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
    assert(nq<=sizeof(q)/sizeof(q[0]) && nkv<=sizeof(k)/sizeof(k[0]) && np<=sizeof(p0)/sizeof(p0[0]));
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    if (pass && t>3) { v[(t-1)*nk*d]=INFINITY; dy[d]=NAN; }
    ref_attention(b,t,nh,nk,d,q,k,v,p0,y0); bcir_tensor_attention(b,t,nh,nk,d,q,k,v,p1,y1);
    assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y1,nq*sizeof(float)));
    for (i=0;i<np;i++) if ((i%t)>(i/t)%t) assert(p1[i]==0.0f && !signbit(p1[i]));
    ref_attention_backward(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
    /* the reference backward the header keeps, unchanged, and the in-place one */
    bcir_tensor_attention_backward(b,t,nh,nk,d,q,k,v,p1,dy,dq1,dk1,dv1);
    assert(!memcmp(dq0,dq1,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
        !memcmp(dv0,dv1,nkv*sizeof(float)));
    memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float));
    bcir_tensor_attention_backward_inplace(b,t,nh,nk,d,q,k,v,p1,dy,dq1,dk1,dv1);
    assert(!memcmp(dq0,dq1,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
        !memcmp(dv0,dv1,nkv*sizeof(float)));
  }
  puts("  PASS native causal GQA attention: forward and in-place backward bit-identical to the "
      "scalar reference, finite and not");
}
/* The AdamW update as it was written before its passes were restructured -- the reference the
 * update must match bit for bit, state and verdict, on the fast path (the magnitudes prove the
 * preflight would admit everything) and on the preflight path alike. */
static int ref_update(bcir_decoder_state *s,const bcir_decoder_adamw *o,double gs,double *gn) {
  size_t i,n=s->plan.parameters; double norm=0.0,scale,b1,b2; int bad=0;
  for (i=0;i<n;i++) {
    double g=(double)s->gradients[i]*gs;
    if (!isfinite(g) || !isfinite(s->weights[i]) || !isfinite(s->moment1[i]) ||
        !isfinite(s->moment2[i]) || s->moment2[i]<0) return BCIR_DT_NUMERIC;
    norm+=g*g;
  }
  norm=sqrt(norm); if (!isfinite(norm)) return BCIR_DT_NUMERIC;
  scale=gs;
  if (o->grad_clip>0 && norm>o->grad_clip) scale*=o->grad_clip/(norm+1e-6);
  b1=s->beta1_power*o->beta1; b2=s->beta2_power*o->beta2;
  {
    const float *gr=s->gradients; float *wt=s->weights,*m1=s->moment1,*m2=s->moment2;
    double d1=o->beta1,d2=o->beta2,decay=1-o->lr*o->weight_decay,lr=o->lr,eps=o->epsilon;
    for (i=0;i<n;i++) {
      double g=gr[i]*scale,mc=d1*m1[i]+(1-d1)*g,vc=d2*m2[i]+(1-d2)*g*g,wc;
      int in=fabs(mc)<=FLT_MAX && fabs(vc)<=FLT_MAX;
      float m=in ? (float)mc : 0.0f,v=in ? (float)vc : 0.0f;
      wc=(double)wt[i]*decay-lr*((double)m/(1-b1))/(sqrt((double)v/(1-b2))+eps);
      bad|=!in | !(fabs(wc)<=FLT_MAX);
    }
    if (bad) return BCIR_DT_NUMERIC;
    for (i=0;i<n;i++) {
      double g=gr[i]*scale,mc=d1*m1[i]+(1-d1)*g,vc=d2*m2[i]+(1-d2)*g*g;
      float m=(float)mc,v=(float)vc;
      wt[i]=(float)((double)wt[i]*decay-lr*((double)m/(1-b1))/(sqrt((double)v/(1-b2))+eps));
      m1[i]=m; m2[i]=v;
    }
  }
  s->step++; s->beta1_power=b1; s->beta2_power=b2; *gn=norm;
  return BCIR_DT_OK;
}
static void adamw_reference(void) {
  static float w2[4096],g2[4096],m12[4096],m22[4096];
  bcir_decoder_state s=fixture(),r; size_t n=s.plan.parameters,i,step;
  /* clipping on and off, scaled gradients, decay; a moment of 1e30 with an epsilon of 1e-40
   * fails the magnitude bound (the preflight runs and admits the update), and a learning rate
   * of 1e300 fails it too and is refused by the preflight -- state untouched either way */
  static const double cases[][5]={{0.01,0.01,1.0,1.0,1e-8},{0.003,0.1,0.0,0.5,1e-8},
      {0.02,0.0,0.05,2.0,1e-8},{0.001,0.01,1.0,1.0,1e-40},{1e300,0.0,0.0,1.0,1e-8}};
  r=s; r.weights=w2; r.gradients=g2; r.moment1=m12; r.moment2=m22;
  for (i=0;i<n;i++) {
    w[i]=0.1f*sinf((float)i*0.13f); grad[i]=0.05f*cosf((float)i*0.71f)*(i%5 ? 1.0f : 20.0f);
    m[i]=0.0f; v[i]=0.0f;
  }
  memcpy(w2,w,n*sizeof(float)); memcpy(g2,grad,n*sizeof(float));
  memcpy(m12,m,n*sizeof(float)); memcpy(m22,v,n*sizeof(float));
  for (step=0;step<12;step++) {
    const double *c=cases[step%5]; bcir_decoder_adamw o={c[0],0.9,0.95,c[4],c[1],c[2]};
    double na=-1.0,nb=-2.0; int ra,rb;
    if (step%5==3) { m[3]=m12[3]=1e30f; v[3]=m22[3]=1e30f; }
    ra=bcir_decoder_update(&s,&o,c[3],&na); rb=ref_update(&r,&o,c[3],&nb);
    assert(ra==rb && (ra==BCIR_DT_OK)==(step%5!=4));
    assert(!memcmp(w,w2,n*sizeof(float)) && !memcmp(m,m12,n*sizeof(float)) &&
        !memcmp(v,m22,n*sizeof(float)) && s.step==r.step);
    assert(!memcmp(&s.beta1_power,&r.beta1_power,sizeof(double)) &&
        !memcmp(&s.beta2_power,&r.beta2_power,sizeof(double)));
    if (ra==BCIR_DT_OK) assert(!memcmp(&na,&nb,sizeof(double)));
    for (i=0;i<n;i++) grad[i]=g2[i]=0.05f*sinf((float)(i+step)*0.97f)*(i%7 ? 1.0f : 30.0f);
  }
  puts("  PASS native AdamW: bit-identical to the reference update, bounded and preflight paths");
}
/* NDT-THREADS: the pool kernels at one to four threads against the serial kernels and the
 * scalar references, bit for bit -- GEMM shapes that cross the 64-row tasks and both column-block
 * widths over every transpose pair with a beta that scales, causal GQA attention forward and
 * in-place backward against the reference loops (one task per KV head, its query heads in order),
 * SiLU, the SwiGLU VJP, RMSNorm, RoPE, the residual add and gate multiply over more than one
 * chunk, the task runner itself, and three complete training steps of the fixture through
 * bcir_decoder_run on a state that names the pool. The pool lives in the harness's own block
 * (no memory, too little, or misaligned memory is refused); a host without a thread rail admits
 * one thread only: the kernels still run there, serially, and the wider pools have no size. */
static void count_task(void *ctx, size_t task) { ((unsigned char *)ctx)[task]++; }
static void pool_identity(void) {
  static float a[14000],b[14000],c0[20000],c1[20000],init[20000];
  static float aq[2*70*8*48],ak[2*70*4*130],av[2*70*4*130],ady[2*70*8*48],p0[2*8*70*70],
      p1[2*8*70*70],y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],
      dk0[2*70*4*130],dk1[2*70*4*130],dv0[2*70*4*130],dv1[2*70*4*130];
  static float x[20000],u[20000],g[20000],e0[20000],e1[20000],f0[20000],f1[20000],gamma[37];
  static float w0[4096],m0[4096],v0[4096]; static unsigned char hits[300];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const size_t gemm_shapes[][3]={{200,100,70},{1,1,1},{65,33,64},{129,65,3},{64,130,1},{3,200,2}};
  static const size_t att_shapes[][5]={{2,70,8,4,48},{1,33,3,1,65},{2,64,4,1,24},{1,1,1,1,2}};
  uint32_t ids[12],targets[12]; bcir_decoder_event events[3]; size_t done;
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,1.0};
  unsigned threads,ran=0; size_t i,s,ta,tb; int rails=bcir_tensor_pool_rails();
  assert(bcir_tensor_pool_size(0)==0 && bcir_tensor_pool_size(4097)==0);
  assert(bcir_tensor_pool_init(NULL,sizeof(pool_memory),1)==NULL);
  assert(bcir_tensor_pool_init(pool_memory,bcir_tensor_pool_size(1)-1,1)==NULL);
  assert(bcir_tensor_pool_init(pool_memory+1,sizeof(pool_memory)-1,1)==NULL);
  assert(bcir_tensor_pool_threads(NULL)==1);
  for (i=0;i<14000;i++) { a[i]=sinf((float)i*0.71f)*(i%3 ? 4.0f : 0.125f); b[i]=cosf((float)i*1.37f)-0.25f; }
  for (i=0;i<20000;i++) { x[i]=sinf((float)i*0.37f)*3.0f; u[i]=cosf((float)i*0.53f)*1.3f; g[i]=sinf((float)i*0.29f)-0.1f; }
  for (i=0;i<37;i++) gamma[i]=1.0f+0.1f*sinf((float)i);
  for (i=0;i<12;i++) { ids[i]=(uint32_t)(i%4+1); targets[i]=(uint32_t)(i%4+2); }
  {
    bcir_decoder_state st=fixture();
    assert(bcir_decoder_run(&st,&opt,ids,targets,12,NULL,3,events,3,&done)==0 && done==3);
    memcpy(w0,w,sizeof(w)); memcpy(m0,m,sizeof(m)); memcpy(v0,v,sizeof(v));
  }
  for (threads=1;threads<=4;threads++) {
    size_t need=bcir_tensor_pool_size(threads); bcir_tensor_pool *pool;
    if (!need) { assert(threads>1 && rails==0); continue; }
    assert(need<=sizeof(pool_memory));
    pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads);
    assert(pool!=NULL && bcir_tensor_pool_threads(pool)==threads);
    ran++;
    /* the runner: every task once, whatever the count */
    memset(hits,0,sizeof(hits)); bcir_tensor_pool_run(pool,count_task,hits,300);
    for (i=0;i<300;i++) assert(hits[i]==1);
    bcir_tensor_pool_run(pool,count_task,hits,1); assert(hits[0]==2 && hits[1]==1);
    bcir_tensor_pool_run(pool,count_task,hits,0); assert(hits[0]==2);
    for (s=0;s<sizeof(gemm_shapes)/sizeof(gemm_shapes[0]);s++) for (ta=0;ta<2;ta++) for (tb=0;tb<2;tb++) {
      size_t mm=gemm_shapes[s][0],n=gemm_shapes[s][1],kk=gemm_shapes[s][2];
      float alpha=s%2 ? 0.75f : 1.0f,beta=s%3==1 ? 1.0f : s%3==2 ? 0.5f : 0.0f;
      for (i=0;i<mm*n;i++) init[i]=0.01f*(float)(i%17)-0.05f;
      memcpy(c0,init,mm*n*sizeof(float)); bcir_tensor_mm(NULL,(int)ta,(int)tb,mm,n,kk,alpha,a,b,beta,c0);
      memcpy(c1,init,mm*n*sizeof(float)); bcir_tensor_mm_pool(pool,(int)ta,(int)tb,mm,n,kk,alpha,a,b,beta,c1);
      assert(!memcmp(c0,c1,mm*n*sizeof(float)));
      /* a depth of zero still scales by beta */
      memcpy(c1,init,mm*n*sizeof(float)); bcir_tensor_mm_pool(pool,(int)ta,(int)tb,mm,n,0,alpha,a,b,beta,c1);
      for (i=0;i<mm*n;i++) assert(c1[i]==(beta==0.0f ? 0.0f : beta*init[i]));
    }
    for (s=0;s<sizeof(att_shapes)/sizeof(att_shapes[0]);s++) {
      size_t bb=att_shapes[s][0],t=att_shapes[s][1],nh=att_shapes[s][2],nk=att_shapes[s][3],d=att_shapes[s][4];
      size_t nq=bb*t*nh*d,nkv=bb*t*nk*d,np=bb*nh*t*t;
      for (i=0;i<nq;i++) { aq[i]=sinf((float)i*0.37f)*1.7f; ady[i]=cosf((float)i*0.11f)*0.5f; }
      for (i=0;i<nkv;i++) { ak[i]=cosf((float)i*0.53f)*1.3f; av[i]=sinf((float)i*0.29f)-0.1f; }
      ref_attention(bb,t,nh,nk,d,aq,ak,av,p0,y0);
      memset(p1,0xff,np*sizeof(float)); memset(y1,0xff,nq*sizeof(float));
      bcir_tensor_attention_pool(pool,bb,t,nh,nk,d,aq,ak,av,p1,y1);
      assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y1,nq*sizeof(float)));
      ref_attention_backward(bb,t,nh,nk,d,aq,ak,av,p0,ady,dq0,dk0,dv0);
      memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float)); memset(dv1,0xff,nkv*sizeof(float));
      bcir_tensor_attention_backward_inplace_pool(pool,bb,t,nh,nk,d,aq,ak,av,p1,ady,dq1,dk1,dv1);
      assert(!memcmp(dq0,dq1,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
          !memcmp(dv0,dv1,nkv*sizeof(float)));
    }
    /* elementwise over three chunks, rows over three row chunks, positions over three */
    bcir_tensor_silu(NULL,20000,x,e0); bcir_tensor_silu_pool(pool,20000,x,e1);
    assert(!memcmp(e0,e1,sizeof(e0)));
    bcir_tensor_swiglu_backward(20000,x,u,g,e0,f0); bcir_tensor_swiglu_backward_pool(pool,20000,x,u,g,e1,f1);
    assert(!memcmp(e0,e1,sizeof(e0)) && !memcmp(f0,f1,sizeof(f0)));
    memcpy(e0,u,sizeof(e0)); for (i=0;i<20000;i++) e0[i]+=x[i];
    memcpy(e1,u,sizeof(e1)); bcir_tensor_add_pool(pool,20000,x,e1); assert(!memcmp(e0,e1,sizeof(e0)));
    memcpy(e0,u,sizeof(e0)); for (i=0;i<20000;i++) e0[i]*=x[i];
    memcpy(e1,u,sizeof(e1)); bcir_tensor_mul_pool(pool,20000,x,e1); assert(!memcmp(e0,e1,sizeof(e0)));
    bcir_tensor_rms(40,37,1e-5f,x,gamma,e0); bcir_tensor_rms_pool(pool,40,37,1e-5f,x,gamma,e1);
    assert(!memcmp(e0,e1,40*37*sizeof(float)));
    memcpy(f0,x,sizeof(f0)); bcir_tensor_rope(2,40,3,8,10000.0,f0,0);
    memcpy(f1,x,sizeof(f1)); bcir_tensor_rope_pool(pool,2,40,3,8,10000.0,f1,0);
    assert(!memcmp(f0,f1,2*40*3*8*sizeof(float)));
    bcir_tensor_rope(2,40,3,8,10000.0,f0,1); bcir_tensor_rope_pool(pool,2,40,3,8,10000.0,f1,1);
    assert(!memcmp(f0,f1,2*40*3*8*sizeof(float)));
    /* the state after three steps on this pool is the serial state */
    {
      bcir_decoder_state st=fixture(); st.pool=pool;
      assert(bcir_decoder_run(&st,&opt,ids,targets,12,NULL,3,events,3,&done)==0 && done==3);
      assert(!memcmp(w0,w,sizeof(w)) && !memcmp(m0,m,sizeof(m)) && !memcmp(v0,v,sizeof(v)));
    }
    bcir_tensor_pool_stop(pool); bcir_tensor_pool_stop(pool);
    assert(bcir_tensor_pool_threads(pool)==1);
  }
  bcir_tensor_pool_stop(NULL);
  printf("  PASS native pool kernels and the training step bit-identical to the serial kernels and "
      "references at %u thread count(s) (rail %d)\n",ran,rails);
}
int main(void) {
  gemm_contract(); attention_reference(); adamw_reference(); finite_differences(); refusals();
  learning(); pool_identity(); return 0;
}
