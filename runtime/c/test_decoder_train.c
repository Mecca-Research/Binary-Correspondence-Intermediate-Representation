/* Complete decoder gradients, memory contracts, atomic updates and learning. */
#include "bcir_decoder_train.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float w[4096],grad[4096],m[4096],v[4096],act[32768],scratch[32768];
/* The fixture: vocab 16, width 8, two heads over one KV head, two layers, feed-forward 16; the
 * head tied by default, batch 1 by 4 tokens (1,320 parameters: two of the reduce contract's
 * chunks; four layers make 2,504, three chunks). The wide shape -- batch 5 by 8, 40 rows --
 * crosses the loss's and the scatter's task chunks; an untied head leaves the embedding's
 * gradient to the scatter alone. */
static bcir_decoder_state fixture_of(uint32_t batch,uint32_t time,uint32_t tied,uint32_t layers) {
  bcir_decoder_state s; bcir_decoder_spec spec={16,8,2,1,2,16,1,4,1,10000.0,1e-6};
  size_t i,l,off=128;
  spec.batch=batch; spec.time=time; spec.tied=tied; spec.layers=layers;
  memset(&s,0,sizeof(s));
  assert(bcir_decoder_make_plan(&spec,&s.plan)==0);
  s.weights=w; s.gradients=grad; s.moment1=m; s.moment2=v;
  s.activations=act; s.scratch=scratch;
  s.weight_capacity=s.gradient_capacity=s.moment1_capacity=s.moment2_capacity=4096;
  s.activation_capacity=s.scratch_capacity=32768;
  s.beta1_power=s.beta2_power=1.0;
  memset(m,0,sizeof(m)); memset(v,0,sizeof(v)); memset(grad,0,sizeof(grad));
  for (i=0;i<4096;i++) w[i]=0.1f*sinf((float)(i+1)*0.37f);
  for (l=0;l<layers;l++) {
    for (i=0;i<8;i++) w[off+i]=1.0f;
    off+=8+64+32+32+64;
    for (i=0;i<8;i++) w[off+i]=1.0f;
    off+=8+3*128;
  }
  for (i=0;i<8;i++) w[off+i]=1.0f;
  return s;
}
static bcir_decoder_state fixture(void) { return fixture_of(1,4,1,2); }
/* Token streams the batch repeats without repeating a row: a period coprime to the batch's
 * eight positions and the vocabulary, so the repeats of a token sit at different positions with
 * different neighbours (a periodic stream made every repeat the same row, bit for bit, and a
 * reversed accumulation of identical rows is the same sum). */
static void stream(uint32_t *ids,uint32_t *targets,size_t n) {
  size_t i;
  for (i=0;i<n;i++) { ids[i]=(uint32_t)((i*37+11)%13+1); targets[i]=(uint32_t)((i*29+5)%16); }
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
  bad=s; bad.contract=16;  /* a contract bit the ABI does not define */
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  bad=s; bad.contract=BCIR_DT_CONTRACT_FUSED|BCIR_DT_CONTRACT_KBLOCK;  /* two GEMM contracts at once */
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
  /* the same refusals from the first pass's last chunk (the fixture has two), under the reduce
   * contract too: every task's verdict reaches the caller */
  assert(bcir_decoder_reduce_chunk(s.plan.parameters)==1024 && s.plan.parameters>1024);
  for (i=0;i<2;i++) {
    size_t last=s.plan.parameters-1; bad=s; bad.contract=i ? BCIR_DT_CONTRACT_REDUCE : 0;
    grad[last]=NAN;
    assert(bcir_decoder_update(&bad,&opt,1,&norm)==BCIR_DT_NUMERIC && bad.step==0);
    grad[last]=0; v[last]=-1;
    assert(bcir_decoder_update(&bad,&opt,1,&norm)==BCIR_DT_NUMERIC && bad.step==0);
    v[last]=0; w[last]=INFINITY;
    assert(bcir_decoder_update(&bad,&opt,1,&norm)==BCIR_DT_NUMERIC && bad.step==0);
    w[last]=oldw[last]; opt.grad_clip=0; grad[last]=FLT_MAX;
    assert(bcir_decoder_update(&bad,&opt,1,&norm)==BCIR_DT_NUMERIC && bad.step==0);
    grad[last]=0; opt.grad_clip=1.0;
    assert(!memcmp(oldw,w,sizeof(w)) && !memcmp(oldm,m,sizeof(m)) && !memcmp(oldv,v,sizeof(v)));
  }
  /* the first pass's maxima decide whether the preflight runs (NDT-PASS: they are vector loops'
   * now): a first moment whose step leaves float's range -- its gradient zero, so nothing bounds
   * the step but epsilon -- is one only the preflight refuses; at either end of either chunk and at
   * the vector instances' block edges it is refused and nothing is written, so each element of
   * every chunk reaches the bound */
  for (i=0;i<6;i++) {
    size_t at=i==0 ? 0 : i==1 ? 15 : i==2 ? 16 : i==3 ? 1023 : i==4 ? 1024 : s.plan.parameters-1;
    float keep=grad[at]; grad[at]=0; m[at]=FLT_MAX/4;
    assert(bcir_decoder_update(&s,&opt,1,&norm)==BCIR_DT_NUMERIC && s.step==0);
    m[at]=0; grad[at]=keep;
    assert(!memcmp(oldw,w,sizeof(w)) && !memcmp(oldm,m,sizeof(m)) && !memcmp(oldv,v,sizeof(v)));
  }
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
    const float *k, const float *v, float *p, float *y, float (*ex)(float)) {
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
    for (j=0;j<=i;j++) { p[pi+j]=ex(p[pi+j]-maxv); sum+=p[pi+j]; }
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
 * past the backward's 32-row chunks and 64-column tiles, the forward's register blocks' edges (rows
 * of no whole block of eight, exactly one, one and a row; channels of one, under and over a vector,
 * past the scores' 128-channel chunk), and one past the columns whose
 * probability-adjoint tiles a row block keeps between its passes (330: the tiles at 256 and 320
 * evaluated in each pass, through one spare slot) -- and a non-finite operand, which the masked
 * products must keep out of every result the reference keeps it out of. */
static void attention_reference(void) {
  static float q[2*70*8*48],k[2*70*4*130],v[2*70*4*130],dy[2*70*8*48],p0[2*330*330],
      p1[2*330*330],y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],
      dk0[2*70*4*130],dk1[2*70*4*130],dv0[2*70*4*130],dv1[2*70*4*130];
  static const size_t shapes[][5]={{1,1,1,1,2},{1,2,1,1,1},{2,5,4,2,6},{1,37,2,1,48},
      {2,70,8,4,48},{1,33,3,1,65},{1,40,2,2,130},{2,64,4,1,24},{1,330,2,1,8},
      {1,7,1,1,3},{1,8,1,1,16},{1,9,2,1,17},{1,31,1,1,15},{1,32,2,2,47},{1,33,2,1,49},
      {1,40,1,1,1}};
  size_t s,i,pass;
  for (pass=0;pass<2;pass++) for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
    size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
    size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
    assert(nq<=sizeof(q)/sizeof(q[0]) && nkv<=sizeof(k)/sizeof(k[0]) && np<=sizeof(p0)/sizeof(p0[0]));
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    if (pass && t>3) { v[(t-1)*nk*d]=INFINITY; dy[d]=NAN; }
    /* the outputs start as NaN bytes: every element is the kernel's own write (NDT-SCAN moved the
     * zero fills into the tasks that own them) */
    memset(p1,0xff,np*sizeof(float)); memset(y1,0xff,nq*sizeof(float));
    ref_attention(b,t,nh,nk,d,q,k,v,p0,y0,expf); bcir_tensor_attention(b,t,nh,nk,d,q,k,v,p1,y1);
    assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y1,nq*sizeof(float)));
    for (i=0;i<np;i++) if ((i%t)>(i/t)%t) assert(p1[i]==0.0f && !signbit(p1[i]));
    ref_attention_backward(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
    /* the reference backward the header keeps, unchanged, and the in-place one */
    bcir_tensor_attention_backward(b,t,nh,nk,d,q,k,v,p1,dy,dq1,dk1,dv1);
    assert(!memcmp(dq0,dq1,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
        !memcmp(dv0,dv1,nkv*sizeof(float)));
    memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float));
    memset(dv1,0xff,nkv*sizeof(float));
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
/* The reduce contract's chunk rule and norm, written apart from the unit: 1024 doubled until
 * at most 1024 chunks; each chunk's squares in index order in double from zero, the chunk sums
 * added in chunk order from zero. */
static size_t ref_chunk(size_t n) {
  size_t chunk=1024;
  while ((n+chunk-1)/chunk>1024) chunk*=2;
  return chunk;
}
static double ref_norm(const float *gr,size_t n,double gs,int reduce) {
  size_t i,start,chunk=ref_chunk(n); double norm=0.0;
  if (!reduce) { for (i=0;i<n;i++) { double g=(double)gr[i]*gs; norm+=g*g; } return norm; }
  for (start=0;start<n;start+=chunk) {
    size_t end=start+chunk<n ? start+chunk : n; double part=0.0;
    for (i=start;i<end;i++) { double g=(double)gr[i]*gs; part+=g*g; }
    norm+=part;
  }
  return norm;
}
static int ref_update(bcir_decoder_state *s,const bcir_decoder_adamw *o,double gs,double *gn) {
  size_t i,n=s->plan.parameters; double norm,scale,b1,b2; int bad=0;
  for (i=0;i<n;i++) {
    double g=(double)s->gradients[i]*gs;
    if (!isfinite(g) || !isfinite(s->weights[i]) || !isfinite(s->moment1[i]) ||
        !isfinite(s->moment2[i]) || s->moment2[i]<0) return BCIR_DT_NUMERIC;
  }
  norm=ref_norm(s->gradients,n,gs,(s->contract&BCIR_DT_CONTRACT_REDUCE)!=0);
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
  static float w2[4096],g2[4096],m12[4096],m22[4096]; static double norms[2][12];
  bcir_decoder_state s=fixture_of(1,4,1,4),r; size_t n=s.plan.parameters,i,step,reduce;
  /* clipping on and off, scaled gradients, decay; a moment of 1e30 with an epsilon of 1e-40
   * fails the magnitude bound (the preflight runs and admits the update), and a learning rate
   * of 1e300 fails it too and is refused by the preflight -- state untouched either way; the
   * whole table under v1's chain and under the reduce contract's chunks (NDT-REDUCE), each
   * against its own reference, on a pool of three as well as serially */
  static const double cases[][5]={{0.01,0.01,1.0,1.0,1e-8},{0.003,0.1,0.0,0.5,1e-8},
      {0.02,0.0,0.05,2.0,1e-8},{0.001,0.01,1.0,1.0,1e-40},{1e300,0.0,0.0,1.0,1e-8}};
  static _Alignas(max_align_t) unsigned char pool_memory[4096]; bcir_tensor_pool *pool=NULL;
  unsigned differ=0;
  assert(ref_chunk(1)==1024 && ref_chunk(1048576)==1024 && ref_chunk(1048577)==2048 &&
      ref_chunk(9836928)==16384 && bcir_decoder_reduce_chunk(1)==1024 &&
      bcir_decoder_reduce_chunk(1048576)==1024 && bcir_decoder_reduce_chunk(1048577)==2048 &&
      bcir_decoder_reduce_chunk(9836928)==16384);
  if (bcir_tensor_pool_size(3)) {
    pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3); assert(pool!=NULL);
  }
  assert(n==2504 && bcir_decoder_reduce_chunk(n)==1024);  /* three chunks: an order is visible */
  for (reduce=0;reduce<4;reduce++) {
    s=fixture_of(1,4,1,4); s.contract=reduce&1 ? BCIR_DT_CONTRACT_REDUCE : 0; s.pool=reduce&2 ? pool : NULL;
    r=s; r.weights=w2; r.gradients=g2; r.moment1=m12; r.moment2=m22; r.pool=NULL;
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
      /* the square the update takes its root of is the reference's sum, bit for bit */
      {
        double have=-1.0,want=ref_norm(grad,n,c[3],(int)(reduce&1));
        assert(bcir_decoder_gradient_norm(&s,c[3],&have)==BCIR_DT_OK && !memcmp(&have,&want,sizeof(double)));
      }
      ra=bcir_decoder_update(&s,&o,c[3],&na); rb=ref_update(&r,&o,c[3],&nb);
      assert(ra==rb && (ra==BCIR_DT_OK)==(step%5!=4));
      assert(!memcmp(w,w2,n*sizeof(float)) && !memcmp(m,m12,n*sizeof(float)) &&
          !memcmp(v,m22,n*sizeof(float)) && s.step==r.step);
      assert(!memcmp(&s.beta1_power,&r.beta1_power,sizeof(double)) &&
          !memcmp(&s.beta2_power,&r.beta2_power,sizeof(double)));
      if (ra==BCIR_DT_OK) assert(!memcmp(&na,&nb,sizeof(double)));
      if (reduce<2) norms[reduce][step]=na; else assert(!memcmp(&norms[reduce&1][step],&na,sizeof(double)));
      for (i=0;i<n;i++) grad[i]=g2[i]=0.05f*sinf((float)(i+step)*0.97f)*(i%7 ? 1.0f : 30.0f);
    }
  }
  /* the two contracts' norms are one value to a part in 10^12 and not one double: the chunks
   * round apart from the chain on these gradients at some of the admitted steps */
  for (step=0;step<12;step++) if (step%5!=4) {
    assert(fabs(norms[0][step]-norms[1][step])<=1e-12*norms[0][step]);
    differ+=memcmp(&norms[0][step],&norms[1][step],sizeof(double))!=0;
  }
  assert(differ>0);
  bcir_tensor_pool_stop(pool);
  printf("  PASS native AdamW: bit-identical to the reference update, bounded and preflight paths, "
      "under the chain and the reduce contract (norms apart at %u of 10 steps), serially and pooled\n",differ);
}
/* NDT-THREADS: the pool kernels at one to four threads against the serial kernels and the
 * scalar references, bit for bit -- GEMM shapes that cross the 64-row tasks and both column-block
 * widths over every transpose pair with a beta that scales, causal GQA attention forward and
 * in-place backward against the reference loops (one task per KV head, its query heads in order),
 * SiLU, the SwiGLU VJP, RMSNorm and its VJP (three row chunks and three column chunks of the
 * gamma adjoint, accumulated from what it holds), RoPE, the residual add and gate multiply over
 * more than one chunk, the task runner itself, and three complete training steps of the wide
 * fixture (40 rows: three chunks of the loss's rows, tokens the batch repeats) through
 * bcir_decoder_run on a state that names the pool, under every contract (v1, the fused GEMM, the
 * BCIR exponential, both, the blocked GEMM alone and with the exponential, and each of the six
 * under the reduce contract: twelve states, each its own, with the losses and norms the steps
 * report). The pool lives in the harness's own block (no memory, too little, or misaligned
 * memory is refused); a host without a thread rail admits one thread only: the kernels still run
 * there, serially, and the wider pools have no size. */
static void count_task(void *ctx, size_t task) { ((unsigned char *)ctx)[task]++; }
static void pool_identity(void) {
  static float a[14000],b[14000],c0[20000],c1[20000],init[20000];
  static float aq[2*70*8*48],ak[2*70*4*130],av[2*70*4*130],ady[2*70*8*48],p0[2*8*70*70],
      p1[2*8*70*70],y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],
      dk0[2*70*4*130],dk1[2*70*4*130],dv0[2*70*4*130],dv1[2*70*4*130];
  static float x[20000],u[20000],g[20000],e0[20000],e1[20000],f0[20000],f1[20000],gamma[37];
  static float dg0[37],dg1[37],inv[40];
  static float ref[16][3][4096]; static bcir_decoder_event refev[16][3];
  static unsigned char hits[300]; uint32_t contract,other;
  /* v1; fused; exp; fused+exp; kblock; kblock+exp; each with the reduce bit */
  static const uint32_t contracts[]={0,1,2,3,4,6,8,9,10,11,12,14};
  size_t ci,cj,nc=sizeof(contracts)/sizeof(contracts[0]);
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const size_t gemm_shapes[][3]={{200,100,70},{1,1,1},{65,33,64},{129,65,3},{64,130,1},{3,200,2}};
  static const size_t att_shapes[][5]={{2,70,8,4,48},{1,33,3,1,65},{2,64,4,1,24},{1,1,1,1,2}};
  uint32_t ids[120],targets[120]; bcir_decoder_event events[3]; size_t done; unsigned apart=0;
  /* the clip engages at every step, so a norm that rounds apart leaves a state apart */
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  unsigned threads,ran=0; size_t i,s,ta,tb; int rails=bcir_tensor_pool_rails();
  assert(bcir_tensor_pool_size(0)==0 && bcir_tensor_pool_size(4097)==0);
  assert(bcir_tensor_pool_init(NULL,sizeof(pool_memory),1)==NULL);
  assert(bcir_tensor_pool_init(pool_memory,bcir_tensor_pool_size(1)-1,1)==NULL);
  assert(bcir_tensor_pool_init(pool_memory+1,sizeof(pool_memory)-1,1)==NULL);
  assert(bcir_tensor_pool_threads(NULL)==1);
  for (i=0;i<14000;i++) { a[i]=sinf((float)i*0.71f)*(i%3 ? 4.0f : 0.125f); b[i]=cosf((float)i*1.37f)-0.25f; }
  for (i=0;i<20000;i++) { x[i]=sinf((float)i*0.37f)*3.0f; u[i]=cosf((float)i*0.53f)*1.3f; g[i]=sinf((float)i*0.29f)-0.1f; }
  for (i=0;i<37;i++) gamma[i]=1.0f+0.1f*sinf((float)i);
  stream(ids,targets,120);
  for (ci=0;ci<nc;ci++) {
    bcir_decoder_state st=fixture_of(5,8,1,2); contract=contracts[ci]; st.contract=contract;
    assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
    memcpy(ref[contract][0],w,sizeof(w)); memcpy(ref[contract][1],m,sizeof(m)); memcpy(ref[contract][2],v,sizeof(v));
    memcpy(refev[contract],events,sizeof(events));
  }
  /* each GEMM and exponential contract its own state; the reduce bit changes the norm's last
   * bits and nothing else, so its states are compared through the norms the steps report: a part
   * in 10^12 apart at most, the same loss at the first step (the state is the same until the
   * first update), and apart in some bit at some step -- a difference the float rounding of every
   * update absorbs over these three steps, so the reduce states coincide with their chain's here,
   * while over a long run they need not (bcir_decoder_train.h: a state is its contract's) */
  for (ci=0;ci<nc;ci++) for (cj=0;cj<ci;cj++) {
    contract=contracts[ci]; other=contracts[cj];
    if ((contract^other)==BCIR_DT_CONTRACT_REDUCE) continue;
    assert(memcmp(ref[contract][0],ref[other][0],sizeof(w))!=0);
  }
  for (ci=0;ci<nc;ci++) if (contracts[ci]&BCIR_DT_CONTRACT_REDUCE) {
    contract=contracts[ci]; other=contract&~(uint32_t)BCIR_DT_CONTRACT_REDUCE;
    assert(!memcmp(&refev[contract][0].loss,&refev[other][0].loss,sizeof(double)));
    for (i=0;i<3;i++) {
      double na=refev[contract][i].grad_norm,nb=refev[other][i].grad_norm;
      assert(fabs(na-nb)<=1e-12*nb);
      apart+=memcmp(&na,&nb,sizeof(double))!=0;
    }
  }
  assert(apart>0);
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
      /* the pair job (NDT-SCHED-0): this shape and the next, under each GEMM contract, against
       * the two pool entries one after the other -- and against itself with the two swapped */
      {
        size_t s2=(s+1)%(sizeof(gemm_shapes)/sizeof(gemm_shapes[0]));
        size_t m2=gemm_shapes[s2][0],n2=gemm_shapes[s2][1],k2=gemm_shapes[s2][2]; int contract;
        bcir_tensor_mm_call x,y;
        x.ta=(int)ta; x.tb=(int)tb; x.m=mm; x.n=n; x.k=kk; x.alpha=alpha; x.a=a; x.b=b; x.beta=beta; x.c=c1;
        y.ta=(int)tb; y.tb=(int)ta; y.m=m2; y.n=n2; y.k=k2; y.alpha=1.0f; y.a=b; y.b=a; y.beta=0.5f; y.c=f1;
        for (contract=0;contract<3;contract++) {
          memcpy(c0,init,sizeof(c0)); memcpy(c1,init,sizeof(c1)); memcpy(f0,init,sizeof(f0)); memcpy(f1,init,sizeof(f1));
          (contract==2 ? bcir_tensor_mm_kblock_pool : contract ? bcir_tensor_mm_fused_pool : bcir_tensor_mm_pool)
            (pool,x.ta,x.tb,x.m,x.n,x.k,x.alpha,x.a,x.b,x.beta,c0);
          (contract==2 ? bcir_tensor_mm_kblock_pool : contract ? bcir_tensor_mm_fused_pool : bcir_tensor_mm_pool)
            (pool,y.ta,y.tb,y.m,y.n,y.k,y.alpha,y.a,y.b,y.beta,f0);
          bcir_tensor_mm_pair_pool(pool,contract,&x,&y);
          assert(!memcmp(c0,c1,mm*n*sizeof(float)) && !memcmp(f0,f1,m2*n2*sizeof(float)));
          memcpy(c1,init,sizeof(c1)); memcpy(f1,init,sizeof(f1)); bcir_tensor_mm_pair_pool(pool,contract,&y,&x);
          assert(!memcmp(c0,c1,mm*n*sizeof(float)) && !memcmp(f0,f1,m2*n2*sizeof(float)));
        }
      }
    }
    for (s=0;s<sizeof(att_shapes)/sizeof(att_shapes[0]);s++) {
      size_t bb=att_shapes[s][0],t=att_shapes[s][1],nh=att_shapes[s][2],nk=att_shapes[s][3],d=att_shapes[s][4];
      size_t nq=bb*t*nh*d,nkv=bb*t*nk*d,np=bb*nh*t*t;
      for (i=0;i<nq;i++) { aq[i]=sinf((float)i*0.37f)*1.7f; ady[i]=cosf((float)i*0.11f)*0.5f; }
      for (i=0;i<nkv;i++) { ak[i]=cosf((float)i*0.53f)*1.3f; av[i]=sinf((float)i*0.29f)-0.1f; }
      ref_attention(bb,t,nh,nk,d,aq,ak,av,p0,y0,expf);
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
    for (i=0;i<37;i++) dg0[i]=dg1[i]=0.5f*cosf((float)i);
    memset(inv,0,sizeof(inv));
    bcir_tensor_rms_backward(40,37,1e-5f,x,gamma,g,e0,dg0);
    bcir_tensor_rms_backward_pool(pool,40,37,1e-5f,x,gamma,g,e1,dg1,inv);
    assert(!memcmp(e0,e1,40*37*sizeof(float)) && !memcmp(dg0,dg1,sizeof(dg0)) && inv[39]!=0.0f);
    memcpy(f0,x,sizeof(f0)); bcir_tensor_rope(2,40,3,8,10000.0,f0,0);
    memcpy(f1,x,sizeof(f1)); bcir_tensor_rope_pool(pool,2,40,3,8,10000.0,f1,0);
    assert(!memcmp(f0,f1,2*40*3*8*sizeof(float)));
    bcir_tensor_rope(2,40,3,8,10000.0,f0,1); bcir_tensor_rope_pool(pool,2,40,3,8,10000.0,f1,1);
    assert(!memcmp(f0,f1,2*40*3*8*sizeof(float)));
    /* the state after three steps on this pool is the serial state, and the losses and norms
     * the steps report the serial ones, under every contract */
    for (ci=0;ci<nc;ci++) {
      bcir_decoder_state st=fixture_of(5,8,1,2); contract=contracts[ci]; st.pool=pool; st.contract=contract;
      memset(events,0,sizeof(events));
      assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
      assert(!memcmp(ref[contract][0],w,sizeof(w)) && !memcmp(ref[contract][1],m,sizeof(m)) &&
          !memcmp(ref[contract][2],v,sizeof(v)) && !memcmp(refev[contract],events,sizeof(events)));
    }
    bcir_tensor_pool_stop(pool); bcir_tensor_pool_stop(pool);
    assert(bcir_tensor_pool_threads(pool)==1);
  }
  bcir_tensor_pool_stop(NULL);
  printf("  PASS native pool kernels (the GEMM pair job included) and the training step bit-identical to the "
      "serial kernels and references at %u thread count(s) (rail %d), %u contracts, the reduce norms apart at %u of 18 steps\n",
      ran,rails,(unsigned)nc,apart);
}
/* NDT-REDUCE: the pooled reductions against serial references written apart from the pool --
 * the loss as the row-order chain of the rows' terms recomputed from the logits the forward
 * left, and the embedding's gradient (an untied head leaves it to the scatter alone) as the
 * row-order scatter of the final adjoint the backward leaves at the scratch arena's start; the
 * wide fixture (three row chunks, tokens the batch repeats) on no pool and on a pool of two. */
static void reduction_references(void) {
  static _Alignas(max_align_t) unsigned char pool_memory[4096]; static float ref[128];
  uint32_t ids[40],targets[40]; size_t i,j,n,d=8,vocab=16; unsigned pools=0,p;
  for (p=0;p<4;p++) {
    /* batch 4 (32 rows: the mean's division by a power of two is exact, so the loss carries the
     * chain's every bit) and batch 5 (40 rows: a ragged row chunk), on no pool and a pool of two */
    bcir_decoder_state s=fixture_of(p&1 ? 5 : 4,8,0,2); bcir_tensor_pool *pool=NULL; double loss,ce=0.0;
    const float *logits,*cur;
    n=s.plan.spec.batch*8; stream(ids,targets,n);
    if (p&2) {
      if (!bcir_tensor_pool_size(2)) continue;
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),2); assert(pool!=NULL);
    }
    s.pool=pool; pools++;
    assert(s.plan.parameters==1448);  /* the untied head's 128 after the tied fixture's 1,320 */
    assert(bcir_decoder_loss_backward(&s,ids,targets,n,0,&loss)==0);
    logits=act+s.plan.activations-n*vocab; cur=scratch;
    for (i=0;i<n;i++) {
      float maxv=logits[i*vocab]; double sum=0.0;
      for (j=1;j<vocab;j++) if (logits[i*vocab+j]>maxv) maxv=logits[i*vocab+j];
      for (j=0;j<vocab;j++) sum+=exp((double)logits[i*vocab+j]-maxv);
      ce+=log(sum)+(double)maxv-logits[i*vocab+targets[i]];
    }
    ce/=(double)n;
    assert(!memcmp(&loss,&ce,sizeof(double)));
    memset(ref,0,sizeof(ref));
    for (i=0;i<n;i++) for (j=0;j<d;j++) ref[(size_t)ids[i]*d+j]+=cur[i*d+j];
    assert(!memcmp(ref,grad,vocab*d*sizeof(float)));
    bcir_tensor_pool_stop(pool);
  }
  printf("  PASS native pooled reductions: the loss is the rows' chain and the embedding gradient "
      "the row-order scatter, %u runs over two shapes on no pool and a pool of two\n",pools);
}
/* NDT-POOL: the pool's threads claim their tasks from one counter. Every task of every job runs
 * exactly once -- 600 jobs of 1 to 1,024 tasks at two to four threads, back to back (the workers
 * spinning between generations), every hundredth after a pause longer than the spin (the workers
 * asleep), the last just before the pool stops (the workers joined while they spin) -- each task
 * counting into its own slot, so a skipped or a doubled claim shows in the counts. */
static void tally_task(void *ctx, size_t task) { ((unsigned *)ctx)[task]++; }
static void pool_claims(void) {
  static _Alignas(max_align_t) unsigned char memory[4096];
  static unsigned tally[1024]; unsigned threads,pools=0; size_t r,i;
  for (threads=2;threads<=4;threads++) {
    bcir_tensor_pool *pool; volatile unsigned long idle;
    if (!bcir_tensor_pool_size(threads)) continue;
    pool=bcir_tensor_pool_init(memory,sizeof(memory),threads); assert(pool!=NULL);
    memset(tally,0,sizeof(tally));
    for (r=0;r<600;r++) {
      bcir_tensor_pool_run(pool,tally_task,tally,1+(r*389)%1024);
      if (r%100==99) for (idle=0;idle<5000000ul;idle++) {}
    }
    for (i=0;i<1024;i++) {
      unsigned want=0;
      for (r=0;r<600;r++) want+=i<1+(r*389)%1024;
      assert(tally[i]==want);
    }
    bcir_tensor_pool_stop(pool); pools++;
  }
  printf("  PASS native pool claims: every task of 600 jobs exactly once on %u pool(s), back to back, "
      "after the workers sleep and at the stop\n",pools);
}
/* NDT-SCAN: the step's whole-buffer passes on the pool. A wider fixture -- width 32, feed-forward
 * 64, two layers: 19,104 parameters, one 16,384-element scan chunk and a 2,720-element tail -- on
 * no pool and on pools of two and three: a non-finite weight or accumulated gradient at either end
 * of either chunk is refused before the forward pass runs (the activations untouched); finite
 * weights whose final norm overflows are refused by the activations' scan; and a gradient buffer
 * holding stale values is zeroed whole, its tail included, so the step is the step from zeros, bit
 * for bit, the same at every pool. */
static void scan_references(void) {
  static float bw[19104],bg[19104],bm[19104],bv[19104],bact[65536],bscratch[65536],want[19104];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  bcir_decoder_spec spec={16,32,2,1,2,64,1,4,1,10000.0,1e-6};
  uint32_t x[]={1,2,3,4},y[]={2,3,4,5}; size_t i,k,n=0; unsigned p,runs=0; double loss,first=0.0;
  for (p=0;p<3;p++) {
    bcir_decoder_state s; bcir_tensor_pool *pool=NULL; double again;
    if (p) {
      if (!bcir_tensor_pool_size(p+1)) continue;
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),p+1); assert(pool!=NULL);
    }
    memset(&s,0,sizeof(s));
    assert(bcir_decoder_make_plan(&spec,&s.plan)==0);
    n=s.plan.parameters;
    assert(n==19104 && n>16384 && n%16384 && s.plan.activations<=65536 && s.plan.scratch<=65536);
    s.weights=bw; s.gradients=bg; s.moment1=bm; s.moment2=bv; s.activations=bact; s.scratch=bscratch;
    s.weight_capacity=s.gradient_capacity=s.moment1_capacity=s.moment2_capacity=n;
    s.activation_capacity=s.scratch_capacity=65536; s.beta1_power=s.beta2_power=1.0; s.pool=pool;
    for (i=0;i<n;i++) bw[i]=0.05f*sinf((float)(i+1)*0.37f);
    memset(bm,0,sizeof(bm)); memset(bv,0,sizeof(bv));
    /* stale gradients in every element: the zero fill reaches the tail, the step is the one from
     * zeros, and every pool's step is the serial one */
    memset(bg,0,sizeof(bg));
    assert(bcir_decoder_loss_backward(&s,x,y,4,0,&loss)==0);
    if (!runs) { memcpy(want,bg,sizeof(bg)); first=loss; }
    assert(!memcmp(want,bg,sizeof(bg)) && !memcmp(&first,&loss,sizeof(double)));
    for (i=0;i<n;i++) bg[i]=1000.0f;
    assert(bcir_decoder_loss_backward(&s,x,y,4,0,&again)==0);
    assert(!memcmp(want,bg,sizeof(bg)) && !memcmp(&first,&again,sizeof(double)));
    /* either end of either chunk: 0, 16,383 | 16,384, n-1 */
    for (k=0;k<4;k++) {
      size_t at=k==0 ? 0 : k==1 ? 16383 : k==2 ? 16384 : n-1; float keep=bw[at];
      bw[at]=k&1 ? NAN : INFINITY; bact[0]=777.0f;
      assert(bcir_decoder_forward(&s,x,4)==BCIR_DT_NUMERIC && bact[0]==777.0f);
      assert(bcir_decoder_loss_backward(&s,x,y,4,0,&loss)==BCIR_DT_NUMERIC && bact[0]==777.0f);
      bw[at]=keep; memset(bg,0,sizeof(bg)); bg[at]=k&1 ? -INFINITY : NAN;
      assert(bcir_decoder_loss_backward(&s,x,y,4,1,&loss)==BCIR_DT_NUMERIC && bact[0]==777.0f);
      bg[at]=0.0f;
    }
    /* finite weights, an overflowing final norm (its gamma, the last 32 parameters, at FLT_MAX):
     * the activations' scan refuses what the weights' scan admitted */
    for (i=n-32;i<n;i++) bw[i]=FLT_MAX;
    assert(bcir_decoder_forward(&s,x,4)==BCIR_DT_NUMERIC);
    assert(bcir_decoder_loss_backward(&s,x,y,4,0,&loss)==BCIR_DT_NUMERIC);
    bcir_tensor_pool_stop(pool); runs++;
  }
  printf("  PASS native whole-buffer passes on the pool: non-finite weights and gradients refused at "
      "either end of either scan chunk, an overflowing activation refused, stale gradients zeroed "
      "whole (%zu parameters, %u runs on no pool and pools of two and three)\n",n,runs);
}
/* The fused contract (v2, NDT-FMA): beta first, then one fmaf per term for k ascending -- checked
 * bit for bit against that loop over the same shapes, transposes and scalings as v1, on every
 * fused instance this host can run (the portable fmaf one always), serially and on a pool of
 * three. The two contracts differ: the fused result is not the v1 result on these inputs. */
static void gemm_fused_contract(void) {
  static float a[140*140],b[140*140],c[140*140],want[140*140],init[140*140],v1[140*140];
  static const size_t shapes[][3]={{1,1,1},{3,33,5},{33,31,65},{32,32,32},{7,65,40},{64,1,33},
                                   {5,129,130},{66,64,64},{9,63,128},{4,96,1},{130,7,3}};
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  size_t s,i,j,l,ta,tb,differ=0; int variant,ran=0,mask=bcir_tensor_mm_fused_variants();
  bcir_tensor_pool *pool=bcir_tensor_pool_size(3) ? bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3) : NULL;
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
      volatile float av=alpha*a[ta ? l*m+i : i*k+l];  /* the product alpha*A rounded as in v1 */
      want[i*n+j]=fmaf(av,b[tb ? j*k+l : l*n+j],want[i*n+j]);
    }
    memcpy(c,init,m*n*sizeof(float));
    bcir_tensor_mm_fused(NULL,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c);
    assert(!memcmp(c,want,m*n*sizeof(float)));
    memcpy(v1,init,m*n*sizeof(float));
    bcir_tensor_mm(NULL,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,v1);
    differ+=memcmp(v1,want,m*n*sizeof(float))!=0;
    if (pool) {
      memcpy(c,init,m*n*sizeof(float));
      bcir_tensor_mm_fused_pool(pool,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c);
      assert(!memcmp(c,want,m*n*sizeof(float)));
    }
    for (variant=1;variant<=4;variant*=2) {
      memcpy(c,init,m*n*sizeof(float));
      if (!(mask&variant)) {
        assert(bcir_tensor_mm_fused_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c)==0);
        assert(!memcmp(c,init,m*n*sizeof(float)));
        continue;
      }
      assert(bcir_tensor_mm_fused_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c)==variant);
      assert(!memcmp(c,want,m*n*sizeof(float)));
      ran++;
    }
  }
  assert(differ>0);
  assert(bcir_tensor_mm_fused_variant(3,0,0,1,1,1,1.0f,a,b,0.0f,c)==0);
  bcir_tensor_pool_stop(pool);
  printf("  PASS native GEMM fused contract (v2): bit-identical to the fmaf ascending-k loop, every "
      "transpose, %d variant run(s) (mask %d); %zu of 44 cases differ from v1\n",ran,mask,differ);
}
/* The blocked fused contract (v3, NDT-KBLOCK): beta first; the depth split by the one rule
 * (bcir_tensor_kblock_split: whole 448-blocks while two remain, then half the rest rounded up to
 * 16, then the rest); each block a fused chain from zero, added to c as c + alpha*P -- checked
 * bit for bit against that loop over the v2 shapes and five deeper ones that cross the rule's
 * three cases, on every instance, serially and on a pool of three; equal to v2 exactly where the
 * contract says (one block, beta 0, alpha 1) and apart from it elsewhere; the rule on ten depths. */
static void ref_kblock(size_t ta, size_t tb, size_t m, size_t n, size_t k, float alpha,
    const float *a, const float *b, float beta, const float *init, float *want) {
  size_t lens[64],nb=bcir_tensor_kblock_split(k,lens,64),i,j,l,bi,l0=0;
  assert(nb<=64);
  for (i=0;i<m*n;i++) want[i]=beta==0.0f ? 0.0f : beta*init[i];
  for (bi=0;bi<nb;bi++) {
    for (i=0;i<m;i++) for (j=0;j<n;j++) {
      float p=0.0f; volatile float ap;
      for (l=l0;l<l0+lens[bi];l++) p=fmaf(a[ta ? l*m+i : i*k+l],b[tb ? j*k+l : l*n+j],p);
      ap=alpha*p; want[i*n+j]=want[i*n+j]+ap;
    }
    l0+=lens[bi];
  }
  assert(l0==k);
}
static void gemm_kblock_contract(void) {
  static float a[140*140],b[140*140],c[140*140],want[140*140],init[140*140],v2[140*140];
  static const size_t shapes[][3]={{1,1,1},{3,33,5},{33,31,65},{32,32,32},{7,65,40},{64,1,33},
                                   {5,129,130},{66,64,64},{9,63,128},{4,96,1},{130,7,3},
                                   {17,9,1000},{3,9,2048},{33,7,449},{16,16,897},{9,5,448},
                                   {13,20,449},{25,50,300},{40,33,16}};  /* NDT-BLOCK: 4-row and single-row tails, ragged 16..48-wide blocks */
  static const size_t depths[]={0,1,447,448,449,897,1000,1024,2048,3000};
  static const size_t rule[][8]={{0},{1},{447},{448},{224,225},{448,224,225},{448,288,264},
                                 {448,288,288},{448,448,448,352,352},{448,448,448,448,448,384,376}};
  static const size_t counts[]={0,1,1,1,2,3,3,3,5,7};
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  size_t s,i,ta,tb,differ=0,equal=0,lens[8]; int variant,ran=0,mask=bcir_tensor_mm_kblock_variants();
  bcir_tensor_pool *pool=bcir_tensor_pool_size(3) ? bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3) : NULL;
  assert(mask==bcir_tensor_mm_fused_variants() && (mask&1) && !(mask&~7));
  for (s=0;s<sizeof(depths)/sizeof(depths[0]);s++) {
    size_t n=bcir_tensor_kblock_split(depths[s],lens,8),total=0;
    assert(n==counts[s]);
    for (i=0;i<n;i++) { assert(lens[i]==rule[s][i]); total+=lens[i]; }
    assert(total==depths[s] && bcir_tensor_kblock_split(depths[s],NULL,0)==n);
  }
  for (i=0;i<140*140;i++) {
    a[i]=sinf((float)i*0.71f)*(i%3 ? 4.0f : 0.125f); b[i]=cosf((float)i*1.37f)-0.25f;
  }
  for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (ta=0;ta<2;ta++) for (tb=0;tb<2;tb++) {
    size_t m=shapes[s][0],n=shapes[s][1],k=shapes[s][2];
    float alpha=s%2 ? 0.75f : 1.0f,beta=s%3==1 ? 1.0f : s%3==2 ? 0.5f : 0.0f;
    int one=k<=BCIR_MM_KBLOCK && beta==0.0f && alpha==1.0f;
    assert(m*k<=140*140 && k*n<=140*140 && m*n<=140*140);
    for (i=0;i<m*n;i++) init[i]=0.01f*(float)(i%17)-0.05f;
    ref_kblock(ta,tb,m,n,k,alpha,a,b,beta,init,want);
    memcpy(c,init,m*n*sizeof(float));
    bcir_tensor_mm_kblock(NULL,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c);
    assert(!memcmp(c,want,m*n*sizeof(float)));
    memcpy(v2,init,m*n*sizeof(float));
    bcir_tensor_mm_fused(NULL,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,v2);
    if (one) { assert(!memcmp(v2,want,m*n*sizeof(float))); equal++; }
    else differ+=memcmp(v2,want,m*n*sizeof(float))!=0;
    if (pool) {
      memcpy(c,init,m*n*sizeof(float));
      bcir_tensor_mm_kblock_pool(pool,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c);
      assert(!memcmp(c,want,m*n*sizeof(float)));
    }
    for (variant=1;variant<=4;variant*=2) {
      memcpy(c,init,m*n*sizeof(float));
      if (!(mask&variant)) {
        assert(bcir_tensor_mm_kblock_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c)==0);
        assert(!memcmp(c,init,m*n*sizeof(float)));
        continue;
      }
      assert(bcir_tensor_mm_kblock_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c)==variant);
      assert(!memcmp(c,want,m*n*sizeof(float)));
      ran++;
    }
  }
  assert(differ>0 && equal>0);
  assert(bcir_tensor_mm_kblock_variant(3,0,0,1,1,1,1.0f,a,b,0.0f,c)==0);
  bcir_tensor_pool_stop(pool);
  printf("  PASS native GEMM blocked contract (v3): bit-identical to the K-blocked fmaf chains, every "
      "transpose, %d variant run(s) (mask %d); %zu of 64 cases apart from v2, %zu equal where the "
      "contract says; the split rule on 10 depths\n",ran,mask,differ,equal);
}
/* The BCIR exponential and the v2 contracts that use it (NDT-EXP): every instance this host runs
 * agrees bit for bit over 2^24 inputs (every binary32 in [-104, 89] at stride 2^8), within 1 ulp
 * of exp in double rounded to binary32; the special values; SiLU v2 and the SwiGLU VJP v2 on a
 * pool against their scalar definitions; the attention softmax v2 against the reference loop with
 * the BCIR exponential; the v2 kernels are not the v1 kernels on these inputs. */
static float silu2_ref(float x) {
  float e=bcir_tensor_expf(x<0.0f ? x : -x),s=1.0f/(1.0f+e),sg=x>=0.0f ? s : e*s;
  return x*sg;
}
static void exp_contract(void) {
  static float x[20000],u[20000],g[20000],e0[20000],e1[20000],f0[20000],f1[20000];
  static float aq[2*70*8*48],ak[2*70*4*130],av[2*70*4*130],p0[2*8*70*70],p1[2*8*70*70],y0[2*70*8*48],y1[2*70*8*48];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const size_t att_shapes[][5]={{2,70,8,4,48},{1,33,3,1,65},{2,64,4,1,24},{1,1,1,1,2}};
  size_t mismatch=0,swept,refused,i,s; unsigned ulp=0,none; int variant,mask=bcir_tensor_mm_variants(),d1,d2,d3=0;
  bcir_tensor_pool *pool=bcir_tensor_pool_size(3) ? bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3) : NULL;
  swept=bcir_tensor_exp_sweep(256u,&mismatch,&ulp);
  assert(swept>8000000 && mismatch==0 && ulp<=1);
  assert(bcir_tensor_exp_sweep(3u,&refused,&none)==0 && refused==0 && none==0);  /* not a power of two */
  assert(bcir_tensor_expf(0.0f)==1.0f && bcir_tensor_expf(-INFINITY)==0.0f && isinf(bcir_tensor_expf(INFINITY)) &&
      isinf(bcir_tensor_expf(89.0f)) && bcir_tensor_expf(-104.0f)==0.0f && isnan(bcir_tensor_expf(NAN)) &&
      bcir_tensor_expf(-100.0f)>0.0f && bcir_tensor_expf(-100.0f)<1e-43f);
  for (i=0;i<20000;i++) { x[i]=sinf((float)i*0.37f)*12.0f; u[i]=cosf((float)i*0.53f)*1.3f; g[i]=sinf((float)i*0.29f)-0.1f; }
  for (variant=1;variant<=4;variant*=2) {
    if (!(mask&variant)) { assert(bcir_tensor_exp_variant(variant,20000,x,e1)==0); continue; }
    assert(bcir_tensor_exp_variant(variant,20000,x,e1)==variant);
    for (i=0;i<20000;i++) assert(e1[i]==bcir_tensor_expf(x[i]));
  }
  bcir_tensor_exp(20000,x,e1);
  for (i=0;i<20000;i++) assert(e1[i]==bcir_tensor_expf(x[i]));
  for (i=0;i<20000;i++) e0[i]=silu2_ref(x[i]);
  bcir_tensor_silu_v2_pool(pool,20000,x,e1); assert(!memcmp(e0,e1,sizeof(e0)));
  bcir_tensor_silu_v2_pool(NULL,20000,x,f1); assert(!memcmp(e0,f1,sizeof(e0)));
  bcir_tensor_silu(NULL,20000,x,f0); d1=memcmp(f0,e0,sizeof(e0))!=0;
  for (i=0;i<20000;i++) {
    float e=bcir_tensor_expf(x[i]<0.0f ? x[i] : -x[i]),sg=1.0f/(1.0f+e),sv=x[i]>=0.0f ? sg : e*sg;
    e0[i]=g[i]*u[i]*(sv+x[i]*sv*(1.0f-sv)); f0[i]=g[i]*x[i]*sv;
  }
  bcir_tensor_swiglu_backward_v2_pool(pool,20000,x,u,g,e1,f1);
  assert(!memcmp(e0,e1,sizeof(e0)) && !memcmp(f0,f1,sizeof(f0)));
  bcir_tensor_swiglu_backward(20000,x,u,g,e1,f1); d2=memcmp(e0,e1,sizeof(e0))!=0;
  for (s=0;s<sizeof(att_shapes)/sizeof(att_shapes[0]);s++) {
    size_t bb=att_shapes[s][0],t=att_shapes[s][1],nh=att_shapes[s][2],nk=att_shapes[s][3],d=att_shapes[s][4];
    size_t nq=bb*t*nh*d,nkv=bb*t*nk*d,np=bb*nh*t*t;
    for (i=0;i<nq;i++) aq[i]=sinf((float)i*0.37f)*1.7f;
    for (i=0;i<nkv;i++) { ak[i]=cosf((float)i*0.53f)*1.3f; av[i]=sinf((float)i*0.29f)-0.1f; }
    ref_attention(bb,t,nh,nk,d,aq,ak,av,p0,y0,bcir_tensor_expf);
    bcir_tensor_attention_v2_pool(pool,bb,t,nh,nk,d,aq,ak,av,p1,y1);
    assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y1,nq*sizeof(float)));
    bcir_tensor_attention_pool(pool,bb,t,nh,nk,d,aq,ak,av,p1,y1);
    d3|=memcmp(p0,p1,np*sizeof(float))!=0;
  }
  assert(d1 && d2 && d3);  /* each v2 kernel is its own, not v1 under another name */
  /* the trainer's own SiLU under each contract: a forward pass's layer-0 hidden activations are
   * silu(gate)*up by the contract's sigmoid -- v2's through the BCIR exponential, v1's libm's. The
   * fixture's cache (bcir_decoder_train.c, `activations`): the caches start after (layers+1)*nd =
   * 96 floats; within layer 0, gate sits after n1,q (nd each), k,v (nk each), p (b*h*t*t), ctx, x2,
   * n2 (nd each) = 224, up 64 later, hidden 64 after that. */
  {
    bcir_decoder_state st=fixture(); uint32_t ids2[4]={1,2,1,4}; size_t off=96+224; float want[64];
    st.contract=BCIR_DT_CONTRACT_EXP;
    assert(bcir_decoder_forward(&st,ids2,4)==0);
    for (i=0;i<64;i++) want[i]=silu2_ref(act[off+i])*act[off+64+i];
    assert(!memcmp(want,act+off+128,64*sizeof(float)));
    st=fixture(); st.contract=0;
    assert(bcir_decoder_forward(&st,ids2,4)==0);
    bcir_tensor_silu(NULL,64,act+off,want);
    for (i=0;i<64;i++) want[i]*=act[off+64+i];
    assert(!memcmp(want,act+off+128,64*sizeof(float)));
  }
  bcir_tensor_pool_stop(pool);
  printf("  PASS native exponential (v2): %zu inputs, every instance identical, %u ulp at most; SiLU, "
      "SwiGLU VJP and softmax v2 on a pool as defined, and apart from v1\n",swept,ulp);
}
int main(void) {
  gemm_contract(); attention_reference(); adamw_reference(); finite_differences(); refusals();
  learning(); pool_identity(); pool_claims(); reduction_references(); scan_references();
  gemm_fused_contract(); gemm_kblock_contract(); exp_contract(); return 0;
}
