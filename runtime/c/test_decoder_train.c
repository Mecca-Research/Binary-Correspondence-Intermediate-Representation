/* Complete decoder gradients, memory contracts, atomic updates and learning. */
#include "bcir_decoder_train.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
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
/* The bf16 contract's workspace for a fixture state (NDT-BF16): this buffer, the size the plan
 * asks for, under that contract; nothing otherwise. */
static _Alignas(64) unsigned char workspace[1<<18];
static void lend(bcir_decoder_state *s) {
  size_t need=bcir_decoder_workspace_bytes(&s->plan,s->contract);
  assert(need<=sizeof(workspace));
  s->workspace=need ? workspace : NULL; s->workspace_bytes=need;
}
/* ATT-CONTIG: the binary32 attention's regions for a plan's shapes on a pool of `threads` workers (0:
 * any pool) -- under a contract without BF16 the plan's whole need, under BF16 a part of it; 0 below
 * the copies' context threshold, so every figure below holds in a build that lowers it too */
static size_t att32_need(const bcir_decoder_plan *p,unsigned threads) {
  const bcir_decoder_spec *sp=&p->spec;
  return bcir_tensor_attention_pool_workspace(sp->batch,sp->time,sp->heads,sp->kvheads,
      sp->width/sp->heads,threads);
}
static size_t most_of(size_t a,size_t b) { return a>b ? a : b; }
/* WS-PLAN: the 9.84M decoder's GEMM groups' figure under the bf16 contract (trainer_bf16 derives it) */
static size_t gemm_984m(void) {
#ifndef BCIR_DT_NO_PANELS
  return 4522048u;
#else
  return 10747968u;
#endif
}
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
  bad=s; bad.contract=128;  /* a contract bit the ABI does not define */
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  /* the bf16 attention (BF16-ATTN) without the bf16 contract's workspace, and beside the fused one */
  bad=s; bad.contract=BCIR_DT_CONTRACT_ATTNBF16; lend(&bad); act[0]=123;
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID && act[0]==123);
  bad=s; bad.contract=BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTN|BCIR_DT_CONTRACT_ATTNBF16; lend(&bad);
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID && act[0]==123);
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
/* The same loops under the fused attention contract (ATTN-FMA): every multiply-add one fmaf over
 * the same terms in the same order from the same start -- each score's channels, each context
 * element's keys, each probability adjoint's channels, the dot's keys, the dq, dk and dv sums --
 * and the scaling, the softmax and the score adjoint as above. Written apart from the kernels. */
static void ref_attention_fused(size_t b, size_t t, size_t nh, size_t nk, size_t d, const float *q,
    const float *k, const float *v, float *p, float *y, float (*ex)(float)) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(y,0,b*t*nh*d*sizeof(float)); memset(p,0,b*nh*t*t*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t;
    float maxv=-INFINITY,sum=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float z=0.0f;
      for (c=0;c<d;c++) z=fmaf(q[qi+c],k[kj+c],z);
      p[pi+j]=z*scale; if (p[pi+j]>maxv) maxv=p[pi+j];
    }
    for (j=0;j<=i;j++) { p[pi+j]=ex(p[pi+j]-maxv); sum+=p[pi+j]; }
    for (j=0;j<=i;j++) {
      size_t vj=((r*t+j)*nk+kh)*d; p[pi+j]/=sum;
      for (c=0;c<d;c++) y[qi+c]=fmaf(p[pi+j],v[vj+c],y[qi+c]);
    }
  }
}
static void ref_attention_backward_fused(size_t b, size_t t, size_t nh, size_t nk, size_t d,
    const float *q, const float *k, const float *v, const float *p, const float *dy, float *dq,
    float *dk, float *dv) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(dq,0,b*t*nh*d*sizeof(float)); memset(dk,0,b*t*nk*d*sizeof(float));
  memset(dv,0,b*t*nk*d*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t; float dot=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f;
      for (c=0;c<d;c++) dp=fmaf(dy[qi+c],v[kj+c],dp);
      dot=fmaf(p[pi+j],dp,dot);
    }
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f,ds;
      for (c=0;c<d;c++) dp=fmaf(dy[qi+c],v[kj+c],dp);
      ds=p[pi+j]*(dp-dot)*scale;
      for (c=0;c<d;c++) {
        dq[qi+c]=fmaf(ds,k[kj+c],dq[qi+c]); dk[kj+c]=fmaf(ds,q[qi+c],dk[kj+c]);
        dv[kj+c]=fmaf(p[pi+j],dy[qi+c],dv[kj+c]);
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
      {1,40,1,1,1},{1,20,1,1,32},{1,9,2,1,33},{1,11,2,2,81}};  /* the context's and the transposed products' widths: 32, 33, 48+33 */
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
/* The attention under its contracts (ATTN-FMA), through bcir_tensor_attention_contract_scan_pool
 * and bcir_tensor_attention_backward_contract_pool: with FUSED, the forward (under libm's
 * exponential and under BCIR's) and the in-place backward against the fused reference loops on
 * attention_reference's shapes, finite and not, serially and on pools of two and four; without
 * it, each entry the v1 or v2 entry bit for bit; and the two contracts apart on some shape -- the
 * fused path ran (a FUSED bit the unit ignored would match the v1 loops, never these). */
static int all_finite(const float *x,size_t n);
static void attention_fused_reference(void) {
  static float q[2*70*8*48],k[2*70*4*130],v[2*70*4*130],dy[2*70*8*48],p0[2*330*330],
      p1[2*330*330],y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],
      dk0[2*70*4*130],dk1[2*70*4*130],dv0[2*70*4*130],dv1[2*70*4*130],y2[2*70*8*48],dq2[2*70*8*48];
  static const size_t shapes[][5]={{1,1,1,1,2},{1,2,1,1,1},{2,5,4,2,6},{1,37,2,1,48},
      {2,70,8,4,48},{1,33,3,1,65},{1,40,2,2,130},{2,64,4,1,24},{1,330,2,1,8},
      {1,7,1,1,3},{1,8,1,1,16},{1,9,2,1,17},{1,31,1,1,15},{1,32,2,2,47},{1,33,2,1,49},
      {1,40,1,1,1},{1,20,1,1,32},{1,9,2,1,33},{1,11,2,2,81}};  /* the context's and the transposed products' widths: 32, 33, 48+33 */
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  size_t s,i,pass,cases=0,apart=0; unsigned ex,threads; int bad;
  for (threads=1;threads<=4;threads+=threads==1 ? 1 : 2) {
    bcir_tensor_pool *pool=NULL;
    if (threads>1) {
      size_t need=bcir_tensor_pool_size(threads);
      if (!need) continue;
      assert(need<=sizeof(pool_memory));
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool);
    }
    for (pass=0;pass<2;pass++) for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (ex=0;ex<2;ex++) {
      size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
      size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
      unsigned plain=ex ? BCIR_TENSOR_ATTENTION_EXP : 0u;
      assert(nq<=sizeof(q)/sizeof(q[0]) && nkv<=sizeof(k)/sizeof(k[0]) && np<=sizeof(p0)/sizeof(p0[0]));
      for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
      for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
      if (pass && t>3) { v[(t-1)*nk*d]=INFINITY; dy[d]=NAN; }
      ref_attention_fused(b,t,nh,nk,d,q,k,v,p0,y0,ex ? bcir_tensor_expf : expf);
      memset(p1,0xff,np*sizeof(float)); memset(y1,0xff,nq*sizeof(float)); bad=0;
      bcir_tensor_attention_contract_scan_pool(pool,plain|BCIR_TENSOR_ATTENTION_FUSED,b,t,nh,nk,d,q,k,v,p1,y1,&bad);
      assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y1,nq*sizeof(float)));
      assert(bad==!(all_finite(y1,nq) && all_finite(k,nkv)));
      ref_attention_backward_fused(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
      memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float)); memset(dv1,0xff,nkv*sizeof(float));
      bcir_tensor_attention_backward_contract_pool(pool,plain|BCIR_TENSOR_ATTENTION_FUSED,b,t,nh,nk,d,q,k,v,p1,dy,
          dq1,dk1,dv1);
      assert(!memcmp(dq0,dq1,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
          !memcmp(dv0,dv1,nkv*sizeof(float)));
      /* without FUSED the contract entries are the v1 (and v2) entries, bit for bit */
      if (ex) bcir_tensor_attention_v2_pool(pool,b,t,nh,nk,d,q,k,v,p0,y0);
      else bcir_tensor_attention_pool(pool,b,t,nh,nk,d,q,k,v,p0,y0);
      memset(p1,0xff,np*sizeof(float)); memset(y2,0xff,nq*sizeof(float));
      bcir_tensor_attention_contract_scan_pool(pool,plain,b,t,nh,nk,d,q,k,v,p1,y2,NULL);
      assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y2,nq*sizeof(float)));
      memcpy(p1,p0,np*sizeof(float));
      bcir_tensor_attention_backward_inplace_pool(pool,b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
      memset(dq2,0xff,nq*sizeof(float));
      bcir_tensor_attention_backward_contract_pool(pool,plain,b,t,nh,nk,d,q,k,v,p1,dy,dq2,dk1,dv1);
      assert(!memcmp(dq0,dq2,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
          !memcmp(dv0,dv1,nkv*sizeof(float)));
      apart+=memcmp(y1,y2,nq*sizeof(float))!=0 || memcmp(dq1,dq2,nq*sizeof(float))!=0;
      cases++;
    }
    if (pool) bcir_tensor_pool_stop(pool);
  }
  assert(apart>0);
  printf("  PASS fused attention contract (ATTN-FMA): %zu cases against the fmaf reference loops "
      "(forward under both exponentials, the in-place backward, finite and not, serial and pooled), "
      "%zu of them apart from v1\n",cases,apart);
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
/* RoPE as the kernel computed it before NDT-ROPE: a pow, a cosine and a sine per pair and position */
static void ref_rope(size_t b, size_t t, size_t heads, size_t dim, double base, float *x, int inverse) {
  size_t r,pos,h,j;
  for (pos=0;pos<t;pos++) for (j=0;j<dim/2;j++) {
    double angle=(double)pos*pow(base,-2.0*(double)j/(double)dim);
    float c=(float)cos(angle),s=(float)sin(angle)*(inverse ? -1.0f : 1.0f);
    for (r=0;r<b;r++) for (h=0;h<heads;h++) {
      size_t off=((r*t+pos)*heads+h)*dim+j;
      float a=x[off],z=x[off+dim/2]; x[off]=a*c-z*s; x[off+dim/2]=z*c+a*s;
    }
  }
}
static void count_task(void *ctx, size_t task) { ((unsigned char *)ctx)[task]++; }
static void pool_identity(void) {
  static float a[14000],b[14000],c0[20000],c1[20000],init[20000];
  static float aq[2*70*8*48],ak[2*70*4*130],av[2*70*4*130],ady[2*70*8*48],p0[2*8*70*70],
      p1[2*8*70*70],y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],
      dk0[2*70*4*130],dk1[2*70*4*130],dv0[2*70*4*130],dv1[2*70*4*130];
  static float x[20000],u[20000],g[20000],e0[20000],e1[20000],f0[20000],f1[20000],gamma[389];
  static float dg0[389],dg1[389],inv[48];
  static float ref[64][3][4096]; static bcir_decoder_event refev[64][3];
  static unsigned char hits[300]; uint32_t contract,other;
  /* v1; fused; exp; fused+exp; kblock; kblock+exp; bf16; bf16+exp; each with the reduce bit;
   * and the fused attention (ATTN-FMA) alone, with fused, with exp, and with kblock+exp and
   * bf16+exp, the contracts the trainer runs fastest under, each of the last two with and without
   * the reduce bit (the reduce comparison below pairs a contract with its reduce-free one) */
  static const uint32_t contracts[]={0,1,2,3,4,6,8,9,10,11,12,14,16,18,24,26,32,33,34,38,46,50,58};
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
  for (i=0;i<389;i++) gamma[i]=1.0f+0.1f*sinf((float)i);
  stream(ids,targets,120);
  for (ci=0;ci<nc;ci++) {
    bcir_decoder_state st=fixture_of(5,8,1,2); contract=contracts[ci]; st.contract=contract;
    lend(&st);
    assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
    memcpy(ref[contract][0],w,sizeof(w)); memcpy(ref[contract][1],m,sizeof(m)); memcpy(ref[contract][2],v,sizeof(v));
    memcpy(refev[contract],events,sizeof(events));
  }
  /* each GEMM and exponential contract its own state; the reduce bit changes the norm's last
   * bits and nothing else, so its states are compared through the norms the steps report: a part
   * in 10^12 apart at most, the same loss at the first step (the state is the same until the
   * first update), and apart in some bit at some step -- a difference the float rounding of every
   * update absorbs over these three steps, so the reduce states coincide with their chain's here,
   * while over a long run they need not (bcir_decoder_train.h: a state is its contract's). Under
   * bf16 the exponential's last-bit differences are absorbed here too: the GEMMs read their
   * operands rounded to eight significant bits, and three steps of this fixture leave bf16 and
   * bf16+exp one state (trainer_bf16 holds the exp bit to the activations it changes) */
  for (ci=0;ci<nc;ci++) for (cj=0;cj<ci;cj++) {
    contract=contracts[ci]; other=contracts[cj];
    if ((contract^other)==BCIR_DT_CONTRACT_REDUCE) continue;
    if ((contract&other&BCIR_DT_CONTRACT_BF16) &&
        ((contract^other)&~(uint32_t)BCIR_DT_CONTRACT_REDUCE)==BCIR_DT_CONTRACT_EXP) continue;
    /* and so may the fused attention's (attention_fused_reference holds the bit to the products) */
    if ((contract&other&BCIR_DT_CONTRACT_BF16) &&
        ((contract^other)&~(uint32_t)(BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_EXP))==BCIR_DT_CONTRACT_ATTN) continue;
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
          /* the group job (NDT-SCHED-1): a chain of three into c -- this shape at its beta, then at
           * half the depth adding, then at a quarter's scale -- and two independent products,
           * against the pool entries one after another in that order */
          {
            bcir_tensor_mm_call ch[3],ind[2]; size_t q;
            ch[0]=x; ch[1]=x; ch[2]=x; ch[1].k=kk/2 ? kk/2 : 1; ch[1].alpha=0.5f; ch[1].beta=1.0f;
            ch[2].beta=0.25f; ind[0]=y; ind[1]=y; ind[1].c=e1; ind[1].beta=0.0f; ind[1].alpha=-1.5f;
            memcpy(c0,init,sizeof(c0)); memcpy(f0,init,sizeof(f0)); memcpy(e0,init,sizeof(e0));
            for (q=0;q<3;q++)
              (contract==2 ? bcir_tensor_mm_kblock_pool : contract ? bcir_tensor_mm_fused_pool : bcir_tensor_mm_pool)
                (pool,ch[q].ta,ch[q].tb,ch[q].m,ch[q].n,ch[q].k,ch[q].alpha,ch[q].a,ch[q].b,ch[q].beta,c0);
            for (q=0;q<2;q++)
              (contract==2 ? bcir_tensor_mm_kblock_pool : contract ? bcir_tensor_mm_fused_pool : bcir_tensor_mm_pool)
                (pool,ind[q].ta,ind[q].tb,ind[q].m,ind[q].n,ind[q].k,ind[q].alpha,ind[q].a,ind[q].b,ind[q].beta,q ? e0 : f0);
            memcpy(c1,init,sizeof(c1)); memcpy(f1,init,sizeof(f1)); memcpy(e1,init,sizeof(e1));
            bcir_tensor_mm_group_pool(pool,contract,ch,3,ind,2);
            assert(!memcmp(c0,c1,mm*n*sizeof(float)) && !memcmp(f0,f1,m2*n2*sizeof(float)) &&
                !memcmp(e0,e1,m2*n2*sizeof(float)));
            /* no chain (the forward's linears), and a chain alone */
            memcpy(f1,init,sizeof(f1)); memcpy(e1,init,sizeof(e1)); bcir_tensor_mm_group_pool(pool,contract,NULL,0,ind,2);
            assert(!memcmp(f0,f1,m2*n2*sizeof(float)) && !memcmp(e0,e1,m2*n2*sizeof(float)));
            memcpy(c1,init,sizeof(c1)); bcir_tensor_mm_group_pool(pool,contract,ch,3,NULL,0);
            assert(!memcmp(c0,c1,mm*n*sizeof(float)));
            /* the serial fallbacks: a chain whose calls differ in n (the tiles would not line up),
             * and more calls than BCIR_TENSOR_MM_GROUP, each the calls one after another */
            {
              bcir_tensor_mm_call cs[BCIR_TENSOR_MM_GROUP+1];
              ch[1].n=n/2 ? n/2 : 1;
              memcpy(c0,init,sizeof(c0));
              for (q=0;q<3;q++)
                (contract==2 ? bcir_tensor_mm_kblock_pool : contract ? bcir_tensor_mm_fused_pool : bcir_tensor_mm_pool)
                  (pool,ch[q].ta,ch[q].tb,ch[q].m,ch[q].n,ch[q].k,ch[q].alpha,ch[q].a,ch[q].b,ch[q].beta,c0);
              memcpy(c1,init,sizeof(c1)); bcir_tensor_mm_group_pool(pool,contract,ch,3,NULL,0);
              assert(!memcmp(c0,c1,mm*n*sizeof(float)));
              for (q=0;q<BCIR_TENSOR_MM_GROUP+1;q++) { cs[q]=x; cs[q].beta=1.0f; cs[q].alpha=(float)(q+1); }
              cs[0].beta=beta; memcpy(c0,init,sizeof(c0));
              for (q=0;q<BCIR_TENSOR_MM_GROUP+1;q++)
                (contract==2 ? bcir_tensor_mm_kblock_pool : contract ? bcir_tensor_mm_fused_pool : bcir_tensor_mm_pool)
                  (pool,cs[q].ta,cs[q].tb,cs[q].m,cs[q].n,cs[q].k,cs[q].alpha,cs[q].a,cs[q].b,cs[q].beta,c0);
              memcpy(c1,init,sizeof(c1)); bcir_tensor_mm_group_pool(pool,contract,cs,BCIR_TENSOR_MM_GROUP+1,NULL,0);
              assert(!memcmp(c0,c1,mm*n*sizeof(float)));
            }
          }
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
      /* ATTN-FMA: the fused products on this pool, the fused loops' bits */
      ref_attention_fused(bb,t,nh,nk,d,aq,ak,av,p0,y0,expf);
      memset(p1,0xff,np*sizeof(float)); memset(y1,0xff,nq*sizeof(float));
      bcir_tensor_attention_contract_scan_pool(pool,BCIR_TENSOR_ATTENTION_FUSED,bb,t,nh,nk,d,aq,ak,av,p1,y1,NULL);
      assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(y0,y1,nq*sizeof(float)));
      ref_attention_backward_fused(bb,t,nh,nk,d,aq,ak,av,p0,ady,dq0,dk0,dv0);
      memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float)); memset(dv1,0xff,nkv*sizeof(float));
      bcir_tensor_attention_backward_contract_pool(pool,BCIR_TENSOR_ATTENTION_FUSED,bb,t,nh,nk,d,aq,ak,av,p1,ady,
          dq1,dk1,dv1);
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
    /* RMSNorm over 40 x 37 (two whole row chunks and a part one) and 48 x 389 (three whole chunks,
     * 24 blocks of 16 columns and a ragged five): the whole chunks are AVX-512's 16-row lanes
     * (NDT-RMS), held with the rest to the serial kernels */
    for (s=0;s<2;s++) {
      size_t rows=s ? 48 : 40,width=s ? 389 : 37;
      bcir_tensor_rms(rows,width,1e-5f,x,gamma,e0); bcir_tensor_rms_pool(pool,rows,width,1e-5f,x,gamma,e1);
      assert(!memcmp(e0,e1,rows*width*sizeof(float)));
      for (i=0;i<width;i++) dg0[i]=dg1[i]=0.5f*cosf((float)i);
      memset(inv,0,sizeof(inv));
      bcir_tensor_rms_backward(rows,width,1e-5f,x,gamma,g,e0,dg0);
      bcir_tensor_rms_backward_pool(pool,rows,width,1e-5f,x,gamma,g,e1,dg1,inv);
      assert(!memcmp(e0,e1,rows*width*sizeof(float)) && !memcmp(dg0,dg1,width*sizeof(float)) &&
          inv[0]!=0.0f && inv[rows-1]!=0.0f);
    }
    /* RoPE against the per-pair reference, the serial kernel and the pool, forward and inverse:
     * 2 x 40 positions of 3 heads of 8, and 1 x 33 of 2 heads of 260 (130 pairs: three ranges of
     * frequencies) */
    for (s=0;s<2;s++) {
      size_t rb=s ? 1 : 2,rt=s ? 33 : 40,rh=s ? 2 : 3,rd=s ? 260 : 8,rn=rb*rt*rh*rd; int inv2;
      for (inv2=0;inv2<2;inv2++) {
        memcpy(e0,x,rn*sizeof(float)); ref_rope(rb,rt,rh,rd,10000.0,e0,inv2);
        memcpy(f0,x,rn*sizeof(float)); bcir_tensor_rope(rb,rt,rh,rd,10000.0,f0,inv2);
        memcpy(f1,x,rn*sizeof(float)); bcir_tensor_rope_pool(pool,rb,rt,rh,rd,10000.0,f1,inv2);
        assert(!memcmp(e0,f0,rn*sizeof(float)) && !memcmp(e0,f1,rn*sizeof(float)));
      }
    }
    /* the state after three steps on this pool is the serial state, and the losses and norms
     * the steps report the serial ones, under every contract */
    for (ci=0;ci<nc;ci++) {
      bcir_decoder_state st=fixture_of(5,8,1,2); contract=contracts[ci]; st.pool=pool; st.contract=contract;
      lend(&st);
      memset(events,0,sizeof(events));
      assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
      assert(!memcmp(ref[contract][0],w,sizeof(w)) && !memcmp(ref[contract][1],m,sizeof(m)) &&
          !memcmp(ref[contract][2],v,sizeof(v)) && !memcmp(refev[contract],events,sizeof(events)));
    }
    bcir_tensor_pool_stop(pool); bcir_tensor_pool_stop(pool);
    assert(bcir_tensor_pool_threads(pool)==1);
  }
  bcir_tensor_pool_stop(NULL);
  printf("  PASS native pool kernels (the GEMM pair and group jobs included) and the training step bit-identical to the "
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
  /* NDT-BETA: the kernels scale c by beta in their first K block. A depth of zero still scales
   * it (no K block runs), on every path; and under beta 0 a block adds its terms to +0.0, as the
   * zero fill and the add did, so a zero product at a negative alpha stores +0.0, never -0.0 */
  {
    static const float betas[]={0.0f,0.5f,1.0f,-2.0f};
    size_t q,m=9,n=50,k; int path;
    for (i=0;i<m*n;i++) init[i]=i%7==3 ? -0.0f : 0.01f*(float)(i%17)-0.05f;
    for (q=0;q<sizeof(betas)/sizeof(betas[0]);q++) for (path=-2;path<=4;path++) {
      float beta=betas[q];
      if (path>0 && (path==3 || !(mask&path))) continue;
      for (i=0;i<m*n;i++) want[i]=beta==0.0f ? 0.0f : beta*init[i];
      memcpy(c,init,m*n*sizeof(float));
      if (path==-2) bcir_tensor_mm_kblock(NULL,0,1,m,n,0,-1.0f,a,b,beta,c);
      else if (path==-1 || path==0) { if (!pool) continue; bcir_tensor_mm_kblock_pool(pool,path==-1,1,m,n,0,-1.0f,a,b,beta,c); }
      else assert(bcir_tensor_mm_kblock_variant(path,1,0,m,n,0,-1.0f,a,b,beta,c)==path);
      assert(!memcmp(c,want,m*n*sizeof(float)));
    }
    memset(v2,0,sizeof(v2));  /* an A of zeros: every product is +0.0, at alpha -1 a -0.0 term */
    n=20;  /* B is k x n: 897 x 20 within the operands */
    for (k=1;k<=897;k+=448) for (path=-2;path<=4;path++) {
      if (path>0 && (path==3 || !(mask&path))) continue;
      for (i=0;i<m*n;i++) want[i]=0.0f;
      memcpy(c,init,m*n*sizeof(float));
      if (path==-2) bcir_tensor_mm_kblock(NULL,0,0,m,n,k,-1.0f,v2,b,0.0f,c);
      else if (path==-1 || path==0) { if (!pool) continue; bcir_tensor_mm_kblock_pool(pool,path==-1,0,m,n,k,-1.0f,v2,b,0.0f,c); }
      else assert(bcir_tensor_mm_kblock_variant(path,0,0,m,n,k,-1.0f,v2,b,0.0f,c)==path);
      assert(!memcmp(c,want,m*n*sizeof(float)));
    }
  }
  bcir_tensor_pool_stop(pool);
  printf("  PASS native GEMM blocked contract (v3): bit-identical to the K-blocked fmaf chains, every "
      "transpose, %d variant run(s) (mask %d); %zu of 64 cases apart from v2, %zu equal where the "
      "contract says; the split rule on 10 depths; beta in the first K block (a depth of zero "
      "scaled, a zero product at a negative alpha +0.0)\n",ran,mask,differ,equal);
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
  {  /* EXP-LANES: each instance on the edges -- +-0, +-inf, NaN, the overflow and underflow
      * thresholds and their neighbours, the clamps -- in a block's lanes and in the tail, each value
      * exp1's (a NaN a NaN); and a call whose y partly overlaps x (y = x + sh) element by element in
      * order, as the element loop runs it: each element's input what the element sh before it wrote */
    static const float edge[]={0.0f,-0.0f,INFINITY,-INFINITY,NAN,88.72283172607421875f,88.7228394f,
        88.7228241f,-103.972084045410156f,-103.97209f,-103.97208f,89.0f,89.5f,-104.0f,-104.5f,
        -87.33654f,1e-30f,-1e-30f,3.4e38f,-3.4e38f};
    static const size_t shifts[]={1,9,17,60};  /* 60: past n bytes, inside n floats */
    static float ex[203],ey[203],ez[263],ref[263];
    size_t ne=sizeof(edge)/sizeof(edge[0]),sh,si;
    for (i=0;i<203;i++) ex[i]=edge[(i*7)%ne];
    for (variant=1;variant<=4;variant*=2) {
      if (!(mask&variant)) continue;
      assert(bcir_tensor_exp_variant(variant,203,ex,ey)==variant);
      for (i=0;i<203;i++) {
        float want=bcir_tensor_expf(ex[i]);
        assert(isnan(want) ? isnan(ey[i]) : !memcmp(&want,&ey[i],sizeof(float)));
      }
      for (si=0;si<sizeof(shifts)/sizeof(shifts[0]);si++) {
        sh=shifts[si];
        for (i=0;i<203+sh;i++) ez[i]=ref[i]=sinf((float)i*0.41f)*9.0f;
        for (i=0;i<203;i++) ref[i+sh]=bcir_tensor_expf(ref[i]);
        assert(bcir_tensor_exp_variant(variant,203,ez,ez+sh)==variant);
        assert(!memcmp(ez,ref,(203+sh)*sizeof(float)));
      }
    }
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
/* NDT-FUSE: the passes the step merged, each held to the passes it replaced, serially and on a
 * pool of three -- SiLU times up in one pass (v1 and v2, y in place of x or of u too) to the SiLU
 * then the product (a NaN matching a NaN: its sign is the compiler's operand order); the RMSNorm VJP adding into dx to the VJP into a buffer then the add, and its
 * gamma adjoint started at +0.0 to a zero fill (dg full of NaN, so a read shows; one column whose
 * terms are all -0.0, which the fill's +0.0 absorbs); a fresh gradient's first writers (NDT-ZERO)
 * to the fill, every element of a NaN-filled gradient written, tied and untied, under the v1,
 * fused and blocked GEMMs; and a run's opening and closing scans (NDT-FACTS-CALL): its first
 * micro-batch scans the weights before the arena is written, and a step whose gradient overflows
 * from finite activations reports the backward's refusal -- ahead of an invalid optimizer state,
 * and with two micro-batches at the first one's closing scan, the second never run. */
/* Bit for bit, except that a NaN matches a NaN: C leaves a NaN's sign and payload to the operand
 * order the compiler picks, and the one-pass and two-pass SiLU are compiled apart. */
static int same_floats(const float *a,const float *b,size_t n) {
  size_t i;
  for (i=0;i<n;i++) if (memcmp(a+i,b+i,sizeof(float)) && !(isnan(a[i]) && isnan(b[i]))) return 0;
  return 1;
}
static void overflowing(void) {
  size_t i,off=128;
  for (i=0;i<4096;i++) w[i]=0.1f*sinf((float)(i+1)*0.37f);
  for (i=0;i<8;i++) w[off+i]=1.0f;
  off+=8+64+32+32+64; for (i=0;i<8;i++) w[off+i]=1.0f;
  off+=8;  /* layer 0's up projection x1e18 (one sign), its down projection x1e-18 */
  for (i=0;i<128;i++) w[off+128+i]=fabsf(w[off+128+i])*1e18f;
  for (i=0;i<128;i++) w[off+256+i]*=1e-18f;
  off+=3*128; for (i=0;i<8;i++) w[off+i]=1.0f;
  off+=8+64+32+32+64; for (i=0;i<8;i++) w[off+i]=1.0f; off+=8+3*128;
  for (i=0;i<8;i++) w[off+i]=1e24f;  /* the final norm's gamma */
}
static void fused_passes(void) {
  static float x[20000],u[20000],dy[20000],e0[20000],e1[20000],t[20000],gam[389],dg0[389],dg1[389];
  static float inv[48],keepw[4096],keepm[4096],keepv[4096],want[4096];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const uint32_t contracts[]={0,1,4,6};
  uint32_t ids[80],targets[80]; bcir_decoder_event events[2]; bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,1.0};
  size_t i,p,s,c,done; unsigned pools=0; double loss,again;
  for (i=0;i<20000;i++) { x[i]=sinf((float)i*0.37f)*3.0f; u[i]=cosf((float)i*0.53f)*1.3f; dy[i]=sinf((float)i*0.29f)-0.1f; }
  x[0]=0.0f; x[1]=-0.0f; x[2]=INFINITY; x[3]=-INFINITY; x[4]=NAN; x[5]=1e30f; x[6]=-1e30f; x[7]=FLT_MIN/2;
  x[8]=-88.7f; x[9]=89.5f; u[10]=0.0f; u[11]=-0.0f; u[12]=INFINITY; u[13]=FLT_MAX;
  for (i=0;i<389;i++) gam[i]=1.0f+0.1f*sinf((float)i);
  stream(ids,targets,80);
  for (p=0;p<2;p++) {
    bcir_tensor_pool *pool=NULL;
    if (p) {
      if (!bcir_tensor_pool_size(3)) continue;
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3); assert(pool!=NULL);
    }
    /* SiLU times up: v1 (libm) and v2, against the SiLU then the product */
    for (s=0;s<2;s++) {
      if (s) bcir_tensor_silu_v2_pool(NULL,20000,x,e0); else bcir_tensor_silu(NULL,20000,x,e0);
      bcir_tensor_mul_pool(NULL,20000,u,e0);
      memset(e1,0xff,sizeof(e1));
      if (s) bcir_tensor_silu_mul_v2_pool(pool,20000,x,u,e1); else bcir_tensor_silu_mul_pool(pool,20000,x,u,e1);
      assert(same_floats(e0,e1,20000) && isnan(e1[4]) && !isnan(e1[5]));
      memcpy(t,u,sizeof(t));
      if (s) bcir_tensor_silu_mul_v2_pool(pool,20000,x,t,t); else bcir_tensor_silu_mul_pool(pool,20000,x,t,t);
      assert(same_floats(e0,t,20000));
      memcpy(t,x,sizeof(t));
      if (s) bcir_tensor_silu_mul_v2_pool(pool,20000,t,u,t); else bcir_tensor_silu_mul_pool(pool,20000,t,u,t);
      assert(same_floats(e0,t,20000));
    }
    /* the RMSNorm VJP's two writes over 40 x 37 and 48 x 389 (whole 16-row chunks and a part one) */
    for (s=0;s<2;s++) {
      size_t rows=s ? 48 : 40,width=s ? 389 : 37; unsigned how;
      float xs[48*389];
      memcpy(xs,x+16,rows*width*sizeof(float));
      for (i=0;i<rows;i++) { xs[i*width+3]=-0.0f; dy[i*width+3]=fabsf(dy[i*width+3])+0.5f; }
      for (how=0;how<4;how++) {
        for (i=0;i<width;i++) dg0[i]=how&BCIR_TENSOR_RMSB_FIRST ? 0.0f : 0.5f*cosf((float)i);
        for (i=0;i<width;i++) dg1[i]=how&BCIR_TENSOR_RMSB_FIRST ? NAN : 0.5f*cosf((float)i);
        memcpy(e0,u,rows*width*sizeof(float)); memcpy(e1,u,rows*width*sizeof(float));
        bcir_tensor_rms_backward_pool(NULL,rows,width,1e-5f,xs,gam,dy,t,dg0,inv);
        if (how&BCIR_TENSOR_RMSB_ADD) bcir_tensor_add_pool(NULL,rows*width,t,e0);
        else memcpy(e0,t,rows*width*sizeof(float));
        bcir_tensor_rms_backward_into_pool(pool,rows,width,1e-5f,xs,gam,dy,e1,dg1,inv,how);
        assert(!memcmp(e0,e1,rows*width*sizeof(float)) && !memcmp(dg0,dg1,width*sizeof(float)));
        if (how&BCIR_TENSOR_RMSB_FIRST) { uint32_t bits; memcpy(&bits,&dg1[3],4); assert(bits==0u); }
      }
    }
    /* NDT-ZERO: a NaN-filled gradient under accumulate=0 is the zero-filled one, element for element */
    for (s=0;s<2;s++) for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) {
      bcir_decoder_state st=fixture_of(5,8,s ? 0u : 1u,2); size_t n=st.plan.parameters;
      st.contract=contracts[c]; st.pool=pool;
      assert(bcir_decoder_loss_backward(&st,ids,targets,40,0,&loss)==0);
      memcpy(want,grad,n*sizeof(float));
      for (i=0;i<n;i++) grad[i]=NAN;
      assert(bcir_decoder_loss_backward(&st,ids,targets,40,0,&again)==0);
      assert(!memcmp(want,grad,n*sizeof(float)) && !memcmp(&loss,&again,sizeof(double)));
    }
    /* a run's first micro-batch scans the weights before the forward writes the arena */
    for (s=1;s<=2;s++) {
      bcir_decoder_state st=fixture_of(5,8,1,2); st.pool=pool; w[st.plan.parameters-1]=NAN; act[0]=777.0f;
      assert(bcir_decoder_run_accum(&st,&opt,ids,targets,40*s,NULL,1,s,events,1,&done)==BCIR_DT_NUMERIC);
      assert(done==0 && st.step==0 && act[0]==777.0f);
    }
    /* finite activations, an overflowing gradient: the backward refuses it alone and in a run */
    {
      bcir_decoder_state st=fixture_of(5,8,1,2); st.pool=pool; overflowing();
      assert(bcir_decoder_forward(&st,ids,40)==BCIR_DT_OK);
      assert(bcir_decoder_loss_backward(&st,ids,targets,40,0,&loss)==BCIR_DT_NUMERIC);
      memcpy(want,grad,st.plan.parameters*sizeof(float));
      memcpy(keepw,w,sizeof(w)); memcpy(keepm,m,sizeof(m)); memcpy(keepv,v,sizeof(v));
      assert(bcir_decoder_run(&st,&opt,ids,targets,40,NULL,1,events,1,&done)==BCIR_DT_NUMERIC && done==0);
      /* the backward's verdict before the update's own: the run's beta power is no power */
      st.beta1_power=2.0;
      assert(bcir_decoder_run(&st,&opt,ids,targets,40,NULL,1,events,1,&done)==BCIR_DT_NUMERIC && done==0);
      st.beta1_power=1.0;
      /* two micro-batches: the first one's closing scan refuses, the second never runs */
      for (i=0;i<st.plan.parameters;i++) grad[i]=0.25f;
      assert(bcir_decoder_run_accum(&st,&opt,ids,targets,80,NULL,1,2,events,1,&done)==BCIR_DT_NUMERIC);
      assert(done==0 && !memcmp(want,grad,st.plan.parameters*sizeof(float)));
      assert(st.step==0 && !memcmp(keepw,w,sizeof(w)) && !memcmp(keepm,m,sizeof(m)) && !memcmp(keepv,v,sizeof(v)));
      /* and a finite step on that invalid power reports the update's refusal */
      st=fixture_of(5,8,1,2); st.pool=pool; st.beta1_power=2.0;
      assert(bcir_decoder_run(&st,&opt,ids,targets,40,NULL,1,events,1,&done)==BCIR_DT_INVALID && done==0);
    }
    bcir_tensor_pool_stop(pool); pools++;
  }
  printf("  PASS native fused passes: SiLU times up, the RMSNorm VJP's added dx and +0.0 gamma start, a fresh "
      "gradient's first writers (tied, untied, four contracts) and a run's opening and deferred closing scans "
      "against the passes they replaced (%u run(s): no pool and a pool of three)\n",pools);
}
/* NDT-SCAN-W: the forward's writers answer for the activations' finiteness. Each scan entry
 * computes its plain entry's values bit for bit and sets *bad exactly when one of its regions holds
 * an element that is not finite after the call -- held to a scan of every region as the call left
 * it, and a verdict already set left set -- with NaN and both infinities at the ends of the entries'
 * task chunks in every region a caller writes, outputs that overflow from finite inputs, and the
 * key whose scores are all -inf, which only the keys' own scan sees. Through the trainer, with every
 * other activation finite: that key behind RoPE, refused; logits that overflow alone, refused; a
 * provider's SiLU that maps an infinite gate to 0, refused by the scans after it -- and each state
 * without its overflow admitted. */
static int all_finite(const float *x,size_t n) {
  size_t i;
  for (i=0;i<n;i++) if (!isfinite(x[i])) return 0;
  return 1;
}
static void rms_case(bcir_tensor_pool *pool,size_t rows,size_t width,const float *x,
    const float *g,float *y0,float *y1,int preset) {
  int bad=preset;
  bcir_tensor_rms_pool(pool,rows,width,1e-5f,x,g,y0);
  memset(y1,0xff,rows*width*sizeof(float));
  bcir_tensor_rms_scan_pool(pool,rows,width,1e-5f,x,g,y1,&bad);
  assert(!memcmp(y0,y1,rows*width*sizeof(float)));
  assert(bad==(preset || !all_finite(x,rows*width) || !all_finite(y1,rows*width)));
}
static void silu_mul_case(bcir_tensor_pool *pool,int v2,size_t n,const float *x,const float *u,
    float *y0,float *y1,int preset) {
  int bad=preset;
  if (v2) bcir_tensor_silu_mul_v2_pool(pool,n,x,u,y0); else bcir_tensor_silu_mul_pool(pool,n,x,u,y0);
  memset(y1,0xff,n*sizeof(float));
  if (v2) bcir_tensor_silu_mul_v2_scan_pool(pool,n,x,u,y1,&bad);
  else bcir_tensor_silu_mul_scan_pool(pool,n,x,u,y1,&bad);
  assert(!memcmp(y0,y1,n*sizeof(float)));
  assert(bad==(preset || !all_finite(x,n) || !all_finite(u,n) || !all_finite(y1,n)));
}
static void attention_case(bcir_tensor_pool *pool,int v2,const size_t *sh,const float *q,
    const float *k,const float *v,float *p0,float *p1,float *c0,float *c1,int preset,int hidden) {
  size_t b=sh[0],t=sh[1],nh=sh[2],nk=sh[3],d=sh[4],nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
  int bad=preset;
  if (v2) bcir_tensor_attention_v2_pool(pool,b,t,nh,nk,d,q,k,v,p0,c0);
  else bcir_tensor_attention_pool(pool,b,t,nh,nk,d,q,k,v,p0,c0);
  memset(p1,0xff,np*sizeof(float)); memset(c1,0xff,nq*sizeof(float));
  if (v2) bcir_tensor_attention_v2_scan_pool(pool,b,t,nh,nk,d,q,k,v,p1,c1,&bad);
  else bcir_tensor_attention_scan_pool(pool,b,t,nh,nk,d,q,k,v,p1,c1,&bad);
  assert(!memcmp(p0,p1,np*sizeof(float)) && !memcmp(c0,c1,nq*sizeof(float)));
  assert(bad==(preset || !all_finite(q,nq) || !all_finite(k,nkv) || !all_finite(v,nkv) ||
      !all_finite(p1,np) || !all_finite(c1,nq)));
  if (hidden) assert(all_finite(p1,np) && all_finite(c1,nq) && !all_finite(k,nkv) && bad);
}
/* The fixture with every weight zero but the norms, two embeddings, layer 0's queries' channel 1
 * (-1 on dimension 6, both heads) and its key's channel 1 (`key` on dimension 2): token 3 is
 * e2 + 0.1 e6 and token 5 is e6, so at position 1 (token 3) the key's channel 1 is 2.8 `key` and
 * every other key is zero there, every query's channel 1 negative. RoPE turns the pair of channels
 * 1 and 3 by 0.01 a position: at FLT_MAX both of the key's are +inf, both of every later query's
 * negative, so every score of key 1 is -inf, its probability 0, and every other activation is
 * finite (values zero, the residual stream the embeddings). */
static void hidden_key(float key) {
  size_t i,l;
  memset(w,0,sizeof(w));
  w[3*8+2]=1.0f; w[3*8+6]=0.1f; w[5*8+6]=1.0f;
  for (l=0;l<2;l++) for (i=0;i<8;i++) { w[128+l*592+i]=1.0f; w[128+l*592+200+i]=1.0f; }
  for (i=0;i<8;i++) w[1312+i]=1.0f;
  w[128+8+1*8+6]=-1.0f; w[128+8+5*8+6]=-1.0f;
  w[128+72+1*8+2]=key;
}
/* A provider's SiLU that answers 0 for an infinite or NaN gate: one that hides what it reads. */
static void hiding_silu(void *ctx,size_t n,const float *x,float *y) {
  size_t i; (void)ctx;
  for (i=0;i<n;i++) y[i]=isfinite(x[i]) ? x[i]/(1.0f+expf(-x[i])) : 0.0f;
}
static void writer_verdicts(void) {
  static float x[20000],u[20000],y0[20000],y1[20000],g[37],keepw[4096],keepg[4096];
  static float q[2*24*4*16],k[2*24*2*16],v[2*24*2*16],p0[2*4*24*24],p1[2*4*24*24],
      c0[2*24*4*16],c1[2*24*4*16];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const float bads[3]={NAN,INFINITY,-INFINITY};
  static const size_t rms_at[]={0,15*37+36,16*37,32*37,39*37+36},ew_at[]={0,8191,8192,16384,19999};
  static const size_t shapes[][5]={{2,24,4,2,16},{1,9,2,1,5},{1,1,1,1,2}};
  static const uint32_t contracts[]={0,6,BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTNBF16};
  uint32_t ids[4]={5,3,5,5},targets[4]={3,5,5,3};
  size_t i,j,p,s,e,a,r,c,cases=0; unsigned pools=0; double loss;
  for (p=0;p<2;p++) {
    bcir_tensor_pool *pool=NULL;
    if (p) {
      if (!bcir_tensor_pool_size(3)) continue;
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3); assert(pool!=NULL);
    }
    /* the RMSNorm over 40 x 37: two whole 16-row chunks and part of one */
    for (i=0;i<37;i++) g[i]=1.0f+0.1f*sinf((float)i);
    for (i=0;i<40*37;i++) x[i]=sinf((float)i*0.37f)*2.0f;
    rms_case(pool,40,37,x,g,y0,y1,0); rms_case(pool,40,37,x,g,y0,y1,1); cases+=2;
    for (a=0;a<sizeof(rms_at)/sizeof(rms_at[0]);a++) for (e=0;e<3;e++) {
      float keep=x[rms_at[a]]; x[rms_at[a]]=bads[e];
      rms_case(pool,40,37,x,g,y0,y1,0); x[rms_at[a]]=keep; cases++;
    }
    g[36]=FLT_MAX; rms_case(pool,40,37,x,g,y0,y1,0); assert(!all_finite(y1,40*37)); g[36]=1.0f; cases++;
    /* the SwiGLU product over 20,000: two whole 8,192-element chunks and part of one */
    for (s=0;s<2;s++) {
      for (i=0;i<20000;i++) { x[i]=sinf((float)i*0.37f)*3.0f; u[i]=cosf((float)i*0.53f)*1.3f; }
      silu_mul_case(pool,(int)s,20000,x,u,y0,y1,0); silu_mul_case(pool,(int)s,20000,x,u,y0,y1,1); cases+=2;
      for (a=0;a<sizeof(ew_at)/sizeof(ew_at[0]);a++) for (e=0;e<3;e++) {
        float keep=x[ew_at[a]]; x[ew_at[a]]=bads[e];
        silu_mul_case(pool,(int)s,20000,x,u,y0,y1,0); x[ew_at[a]]=keep;
        keep=u[ew_at[a]]; u[ew_at[a]]=bads[e];
        silu_mul_case(pool,(int)s,20000,x,u,y0,y1,0); u[ew_at[a]]=keep; cases+=2;
      }
      x[19999]=10.0f; u[19999]=FLT_MAX;
      silu_mul_case(pool,(int)s,20000,x,u,y0,y1,0); assert(!all_finite(y1,20000)); cases++;
    }
    /* the attention: two batches past a 16-row block with a group of two, ragged rows, one row */
    for (s=0;s<2;s++) for (j=0;j<sizeof(shapes)/sizeof(shapes[0]);j++) {
      const size_t *sh=shapes[j]; size_t b=sh[0],t=sh[1],nh=sh[2],nk=sh[3],d=sh[4];
      size_t nq=b*t*nh*d,nkv=b*t*nk*d,h;
      for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f)*1.7f;
      for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
      attention_case(pool,(int)s,sh,q,k,v,p0,p1,c0,c1,0,0);
      attention_case(pool,(int)s,sh,q,k,v,p0,p1,c0,c1,1,0); cases+=2;
      for (r=0;r<3;r++) for (e=0;e<3;e++) for (a=0;a<3;a++) {
        float *in=r==0 ? q : r==1 ? k : v; size_t len=r==0 ? nq : nkv;
        size_t at=a==0 ? 0 : a==1 ? len-1 : len/b*(b-1),keep_at=at; float keep=in[keep_at];
        in[at]=bads[e]; attention_case(pool,(int)s,sh,q,k,v,p0,p1,c0,c1,0,0); in[keep_at]=keep; cases++;
      }
      if (t>1) {  /* the last batch's key 1 infinite in channel 0, every query of its group negative there */
        size_t rb=b-1; float keep=k[((rb*t+1)*nk+0)*d];
        for (i=0;i<t;i++) for (h=0;h<nh/nk;h++) q[((rb*t+i)*nh+h)*d]=-fabsf(q[((rb*t+i)*nh+h)*d])-0.5f;
        k[((rb*t+1)*nk+0)*d]=INFINITY;
        attention_case(pool,(int)s,sh,q,k,v,p0,p1,c0,c1,0,1); k[((rb*t+1)*nk+0)*d]=keep; cases++;
      }
      /* finite operands, a score past FLT_MAX: a row of NaN probabilities */
      for (c=0;c<d;c++) { q[c]=1e30f; k[c]=1e30f; }
      attention_case(pool,(int)s,sh,q,k,v,p0,p1,c0,c1,0,0); assert(!all_finite(p1,b*nh*t*t)); cases++;
    }
    /* through the trainer: v1 and the exp + kblock contracts */
    for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) {
      bcir_decoder_state st=fixture(); st.pool=pool; st.contract=contracts[c]; lend(&st);
      assert(st.plan.activations==1024);
      hidden_key(1.0f);
      assert(bcir_decoder_forward(&st,ids,4)==BCIR_DT_OK);
      /* FLT_MAX is +inf as a bfloat16 operand, and its products with the other tokens' zeros NaN:
       * under the bf16 contract the key's weight is bfloat16's largest finite value, 2.8 times
       * which overflows as FLT_MAX's does */
      hidden_key((contracts[c]&BCIR_DT_CONTRACT_BF16) ? 0x1.fep127f : FLT_MAX);
      assert(bcir_decoder_forward(&st,ids,4)==BCIR_DT_NUMERIC);
      /* layer 0's keys (16 floats at 160) the only activations that are not finite */
      assert(all_finite(act,160) && !all_finite(act+160,16) && all_finite(act+176,1024-176));
      memcpy(keepg,grad,sizeof(grad));
      assert(bcir_decoder_loss_backward(&st,ids,targets,4,0,&loss)==BCIR_DT_NUMERIC);
      assert(!memcmp(keepg,grad,sizeof(grad)));
      /* an untied head at FLT_MAX: the logits (the last 64 floats) alone overflow */
      st=fixture_of(1,4,0,2); st.pool=pool; st.contract=contracts[c]; lend(&st);
      assert(bcir_decoder_forward(&st,ids,4)==BCIR_DT_OK);
      for (i=0;i<128;i++) w[1320+i]=FLT_MAX;
      assert(bcir_decoder_forward(&st,ids,4)==BCIR_DT_NUMERIC);
      assert(all_finite(act,st.plan.activations-64) && !all_finite(act+st.plan.activations-64,64));
      /* a provider's SiLU that hides an infinite gate: layer 0's MLP zero but its gate's row 0 on
       * dimension 0, which the second norm scales by 1e10 -- at 1 the gate is finite, at FLT_MAX
       * row 0 is infinite at every position and the provider's 0 times up's 0 hides it */
      st=fixture(); st.pool=pool; st.contract=contracts[c]; lend(&st); st.provider.silu=hiding_silu;
      memcpy(keepw,w,sizeof(w));
      for (i=128+208;i<128+592;i++) w[i]=0.0f;
      w[128+200]=1e10f; w[128+208]=1.0f;
      assert(bcir_decoder_forward(&st,ids,4)==BCIR_DT_OK);
      w[128+208]=FLT_MAX;
      assert(bcir_decoder_forward(&st,ids,4)==BCIR_DT_NUMERIC);
      /* the gate (64 floats at 320) the only activations that are not finite */
      assert(all_finite(act,320) && !all_finite(act+320,64) && all_finite(act+384,1024-384));
      memcpy(w,keepw,sizeof(w));
      cases+=6;
    }
    bcir_tensor_pool_stop(pool); pools++;
  }
  printf("  PASS native writers' verdicts: the RMSNorm, the SwiGLU product (v1, v2) and the attention "
      "(v1, v2) as their plain entries, bit for bit, each verdict its regions' scan (%zu cases: "
      "non-finite elements at every task chunk's ends, overflowing outputs, the key the softmax "
      "hides); through the trainer that key, overflowing logits and a hiding provider SiLU refused "
      "with every other activation finite, the bf16 attention's among them (%u run(s): no pool and a "
      "pool of three)\n",cases,pools);
}
/* NDT-UPDATE: the first pass's paired chunks and the commit's shared divisions, on two decoders
 * (vocab 16, width 32, four heads over two KV heads, two layers): feed-forward 64 and untied, 19,616
 * parameters, twenty of the reduce contract's 1,024-element chunks -- nine whole pairs, which take
 * the paired loop on a vector variant, and a pair whose second chunk is short, which takes the chunk
 * loop -- and feed-forward 48 and tied, 16,032 parameters, sixteen chunks, its last pair short too;
 * every buffer NaN past the parameters, which no pass may read. Against ref_update (true
 * divisions, each chunk's chain by itself), the state, the verdict and the norm bit for bit, under
 * the reduce contract and the chain, serially and on a pool of three, at a scale of 1 and of 1/3,
 * clipping on and off, the square sum through bcir_decoder_gradient_norm too; then at each of the
 * first sixteen chunks, at its first and last element, a refusal planted -- a NaN gradient, both
 * infinities, a gradient of 1e20 (or -1e20) whose square overflows the second moment with clipping
 * off, so the gradient maximum the bound reads must have seen it, a NaN weight, an infinite first
 * moment, a negative second moment -- each refused with the state untouched, and a -0.0 second
 * moment admitted as the reference admits it; and a -0.0 gradient over a -0.0 first moment and
 * weight, whose weight the reference makes +0.0: the first division keeps its zero's sign. */
#define UPDATE_N 24576
static float uw[UPDATE_N],ug[UPDATE_N],um1[UPDATE_N],um2[UPDATE_N];
static float rw[UPDATE_N],rg[UPDATE_N],rm1[UPDATE_N],rm2[UPDATE_N];
static float uw0[UPDATE_N],ug0[UPDATE_N],um10[UPDATE_N],um20[UPDATE_N];
static float uact[65536],uscratch[65536];
static const bcir_decoder_spec update_specs[2]={{16,32,4,2,2,64,2,9,0,10000.0,1e-6},
    {16,32,4,2,2,48,2,9,1,10000.0,1e-6}};
static size_t update_spec;
static bcir_decoder_state update_state(float *wt,float *gr,float *m1,float *m2,uint32_t contract,
    bcir_tensor_pool *pool) {
  bcir_decoder_state s; bcir_decoder_spec spec=update_specs[update_spec];
  memset(&s,0,sizeof(s));
  assert(bcir_decoder_make_plan(&spec,&s.plan)==0);
  assert(s.plan.parameters<=UPDATE_N && s.plan.activations<=65536 && s.plan.scratch<=65536);
  s.weights=wt; s.gradients=gr; s.moment1=m1; s.moment2=m2; s.activations=uact; s.scratch=uscratch;
  s.weight_capacity=s.gradient_capacity=s.moment1_capacity=s.moment2_capacity=UPDATE_N;
  s.activation_capacity=s.scratch_capacity=65536; s.beta1_power=s.beta2_power=1.0;
  s.contract=contract; s.pool=pool;
  return s;
}
static void update_fill(size_t n,unsigned seed) {
  size_t i;
  for (i=0;i<n;i++) {
    /* magnitudes over six decades, so each chunk's chain rounds its own way */
    float scale=i%13==0 ? 1e3f : i%7==0 ? 1e-3f : i%5==0 ? 30.0f : 1.0f;
    uw[i]=0.1f*sinf((float)(i+seed)*0.37f); ug[i]=0.05f*cosf((float)(3*i+seed)*0.71f)*scale;
    um1[i]=0.01f*sinf((float)(i+2*seed)*0.13f); um2[i]=1e-4f*(1.0f+sinf((float)(i+seed)*0.29f));
  }
  for (i=n;i<UPDATE_N;i++) uw[i]=ug[i]=um1[i]=um2[i]=NAN;
  memcpy(rw,uw,n*4); memcpy(rg,ug,n*4); memcpy(rm1,um1,n*4); memcpy(rm2,um2,n*4);
}
static int update_untouched(size_t n) {
  return !memcmp(uw,uw0,n*4) && !memcmp(ug,ug0,n*4) && !memcmp(um1,um10,n*4) && !memcmp(um2,um20,n*4);
}
/* One decoder's share: the reference's state, verdict and norm over every contract, pool and scale,
 * then the refusals at each of the first sixteen chunks. */
static void update_pairs_on(bcir_tensor_pool *pool,unsigned *refused,unsigned *admitted) {
  bcir_decoder_state s=update_state(uw,ug,um1,um2,BCIR_DT_CONTRACT_REDUCE,NULL),r;
  size_t n=s.plan.parameters,i,step,chunk,end; unsigned k;
  assert(bcir_decoder_reduce_chunk(n)==1024);
  assert(update_spec ? n==16032 && (n+1023)/1024==16 && n%1024 : n==19616 && (n+1023)/1024==20);
  for (k=0;k<8;k++) {
    uint32_t contract=k&1 ? 0u : BCIR_DT_CONTRACT_REDUCE; double gs=k&4 ? 1.0/3.0 : 1.0;
    update_fill(n,k);
    s=update_state(uw,ug,um1,um2,contract,k&2 ? pool : NULL); r=update_state(rw,rg,rm1,rm2,contract,NULL);
    for (step=0;step<4;step++) {
      bcir_decoder_adamw o={1e-3,0.9,0.95,1e-8,0.01,step%2 ? 0.0 : 1.0};
      double have=-1.0,want=ref_norm(ug,n,gs,contract!=0),na=-1.0,nb=-2.0;
      assert(bcir_decoder_gradient_norm(&s,gs,&have)==BCIR_DT_OK && !memcmp(&have,&want,sizeof(double)));
      assert(bcir_decoder_update(&s,&o,gs,&na)==BCIR_DT_OK && ref_update(&r,&o,gs,&nb)==BCIR_DT_OK);
      assert(!memcmp(&na,&nb,sizeof(double)) && s.step==r.step);
      assert(!memcmp(uw,rw,n*4) && !memcmp(um1,rm1,n*4) && !memcmp(um2,rm2,n*4));
      for (i=0;i<n;i++) ug[i]=rg[i]=0.05f*sinf((float)(i+step+k)*0.97f)*(i%11 ? 1.0f : 40.0f);
    }
  }
  /* each of the first sixteen chunks, at its first and last element (the short chunk's last
   * element where the chunk ends early) */
  for (chunk=0;chunk<16;chunk++) for (end=0;end<2;end++) {
    size_t at=chunk*1024+(end ? 1023 : 0); int c;
    if (at>=n) at=n-1;
    for (c=0;c<9;c++) {
      bcir_decoder_adamw o={1e-3,0.9,0.95,1e-8,0.01,c==3 || c==4 ? 0.0 : 1.0};
      double norm=-1.0; int want=c==8 ? BCIR_DT_OK : BCIR_DT_NUMERIC;
      update_fill(n,7); s=update_state(uw,ug,um1,um2,BCIR_DT_CONTRACT_REDUCE,pool);
      r=update_state(rw,rg,rm1,rm2,BCIR_DT_CONTRACT_REDUCE,NULL);
      switch (c) {
        case 0: ug[at]=rg[at]=NAN; break;
        case 1: ug[at]=rg[at]=INFINITY; break;
        case 2: ug[at]=rg[at]=-INFINITY; break;
        case 3: ug[at]=rg[at]=1e20f; break;
        case 4: ug[at]=rg[at]=-1e20f; break;
        case 5: uw[at]=rw[at]=NAN; break;
        case 6: um1[at]=rm1[at]=INFINITY; break;
        case 7: um2[at]=rm2[at]=-1e-30f; break;
        default: um2[at]=rm2[at]=-0.0f; break;
      }
      memcpy(uw0,uw,n*4); memcpy(ug0,ug,n*4); memcpy(um10,um1,n*4); memcpy(um20,um2,n*4);
      assert(ref_update(&r,&o,1.0,&norm)==want && bcir_decoder_update(&s,&o,1.0,&norm)==want);
      if (want==BCIR_DT_OK) {
        assert(!memcmp(uw,rw,n*4) && !memcmp(um1,rm1,n*4) && !memcmp(um2,rm2,n*4)); (*admitted)++;
      } else { assert(update_untouched(n) && s.step==0); (*refused)++; }
      /* the square sum refuses exactly the gradients that are not finite */
      if (c<3) assert(bcir_decoder_gradient_norm(&s,1.0,&norm)==BCIR_DT_NUMERIC);
    }
  }
}
static void update_pairs(void) {
  static _Alignas(max_align_t) unsigned char pool_memory[4096]; bcir_tensor_pool *pool=NULL;
  bcir_decoder_state s,r; size_t n,at=3*1024+517; unsigned refused=0,admitted=0; uint32_t bits;
  bcir_decoder_adamw o={1e-3,0.9,0.95,1e-8,0.0,0.0}; double na=-1.0,nb=-2.0;
  if (bcir_tensor_pool_size(3)) {
    pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3); assert(pool!=NULL);
  }
  for (update_spec=0;update_spec<2;update_spec++) update_pairs_on(pool,&refused,&admitted);
  /* the zero's sign: (-0.0)/(1-b1) is -0.0, so the candidate is -0.0 - (-0.0) = +0.0 */
  update_spec=0;
  s=update_state(uw,ug,um1,um2,BCIR_DT_CONTRACT_REDUCE,pool); n=s.plan.parameters;
  r=update_state(rw,rg,rm1,rm2,BCIR_DT_CONTRACT_REDUCE,NULL);
  update_fill(n,3);
  uw[at]=rw[at]=-0.0f; ug[at]=rg[at]=-0.0f; um1[at]=rm1[at]=-0.0f;
  assert(bcir_decoder_update(&s,&o,1.0,&na)==BCIR_DT_OK && ref_update(&r,&o,1.0,&nb)==BCIR_DT_OK);
  memcpy(&bits,&rw[at],4); assert(bits==0u);
  assert(!memcmp(uw,rw,n*4) && !memcmp(um1,rm1,n*4) && !memcmp(um2,rm2,n*4));
  bcir_tensor_pool_stop(pool);
  printf("  PASS native update pairs: the reference update's state, verdict and norm bit for bit "
      "on 19,616 and 16,032 parameters (whole pairs of chunks and short ones; both norm contracts, "
      "no pool and a pool of three, scales 1 and 1/3), %u refusals at each of sixteen chunks with "
      "the state untouched, %u -0.0 second moments admitted, and the first division's zero sign "
      "kept\n",refused,admitted);
}
#ifdef BCIR_DT_DIVISION_PROBE
void bcir_decoder_division_probe(size_t n,const float *x,double d,double *q);
/* NDT-UPDATE: the commit's shared division, as the commit dispatches it, against the division over
 * every float dividend, for the bias corrections the update takes -- one minus 0.9, 0.95 and 0.999
 * to the first, second, tenth and thousandth power, computed as the update computes them -- and
 * divisors at the edges of its range: 1, 2^-53, the largest double below 1, all-ones
 * significands. Every significand at one exponent with both signs, then the zeros, the subnormal
 * floats, FLT_MAX and a sweep of exponents: a float dividend's quotient scales by its power of two
 * while nothing under- or overflows, so a significand checked at one exponent is checked at all. */
static void division_probe(void) {
  static float x[65536]; static double q[65536]; double d[16]; size_t nd=0,i,b,k; unsigned long long checked=0;
  static const double betas[3]={0.9,0.95,0.999}; static const unsigned steps[4]={1,2,10,1000};
  for (i=0;i<3;i++) {
    double power=1.0; unsigned t,s=0;
    for (t=1;t<=1000;t++) { double bias=power*betas[i]; if (t==steps[s]) { d[nd++]=1-bias; s++; } power=bias; }
  }
  d[nd++]=1.0; d[nd++]=ldexp(1.0,-53); d[nd++]=nextafter(1.0,0.0); d[nd++]=ldexp(2.0-ldexp(1.0,-52),-2);
  assert(nd==16);
  for (k=0;k<nd;k++) {
    for (b=0;b<(1u<<24)/65536;b++) {
      for (i=0;i<65536;i++) {
        uint32_t bits=0x3f800000u|(uint32_t)((b*32768+i/2)&0x7fffffu); float f;
        memcpy(&f,&bits,4); x[i]=i&1 ? -f : f;
      }
      bcir_decoder_division_probe(65536,x,d[k],q);
      for (i=0;i<65536;i++) { double want=(double)x[i]/d[k]; assert(!memcmp(&q[i],&want,8)); }
      checked+=65536;
    }
    {
      static const float edges[]={0.0f,-0.0f,FLT_MAX,-FLT_MAX,FLT_MIN,-FLT_MIN,1.4e-45f,-1.4e-45f,1.0f};
      int e; size_t m=0;
      for (i=0;i<sizeof(edges)/sizeof(edges[0]);i++) x[m++]=edges[i];
      for (e=-149;e<=127;e++) { x[m++]=(float)ldexp(1.5,e); x[m++]=(float)ldexp(1.9999999,e); }
      bcir_decoder_division_probe(m,x,d[k],q);
      for (i=0;i<m;i++) { double want=(double)x[i]/d[k]; assert(!memcmp(&q[i],&want,8)); }
      checked+=m;
    }
  }
  printf("  PASS native shared division: the commit's corrected products are the division's quotient "
      "bit for bit over %llu dividends (every float significand, both signs, the zeros, the "
      "subnormals and the extremes) for %zu divisors\n",checked,nd);
}
#endif
/* --- the bf16 contract (NDT-BF16) ---------------------------------------------------------------
 * A reference written apart from the unit's: bf16 by choosing the nearer of the two 16-bit
 * truncations around x (an exact difference in double; a tie to the even one), a lane's step as
 * the 24-bit rounding of the sum with the exponent unbounded (frexp, rint, ldexp) compared with
 * 2^-126, and the GEMM as the contract's loop over the binary32 operands themselves. */
static uint16_t ref_bf(float x) {
  uint32_t u,lo,hi; float fl,fh; double dl,dh;
  memcpy(&u,&x,4);
  if (isnan(x)) return (uint16_t)((u>>16)|0x40u);
  if (fpclassify(x)==FP_SUBNORMAL || x==0.0f) return (uint16_t)((u>>16)&0x8000u);
  lo=u&0xffff0000u; hi=lo+0x10000u;
  if (lo==u) return (uint16_t)(lo>>16);
  memcpy(&fl,&lo,4); memcpy(&fh,&hi,4);
  /* past the largest finite bf16 the upper candidate is the infinity, 2^128 for nearness */
  dl=fabs((double)x-(double)fl); dh=fabs((isinf(fh) ? copysign(0x1p128,x) : (double)fh)-(double)x);
  return (uint16_t)((dl<dh || (dl==dh && !(lo&0x10000u)) ? lo : hi)>>16);
}
static float ref_bfv(float x) { uint32_t u=(uint32_t)ref_bf(x)<<16; float f; memcpy(&f,&u,4); return f; }
static float ref_flush(float x) { return fpclassify(x)==FP_SUBNORMAL ? (signbit(x) ? -0.0f : 0.0f) : x; }
static float ref_lane(float e,float a,float b) {
  double d=(double)e+(double)a*(double)b,f,r; int ex=0;
  if (d!=d || d==0.0 || isinf(d)) return (float)d;
  /* no less than 2^-126, its 24-bit rounding is no less either (2^-126 is one bit): a normal
   * result, binary32's own rounding -- the tininess test below decides only what lies under it */
  if (fabs(d)>=0x1p-126) return (float)d;
  f=frexp(fabs(d),&ex);  /* apart: an argument list's order of evaluation is unspecified */
  r=ldexp(rint(ldexp(f,24)),ex-24);  /* |d| on 24 bits, the exponent unbounded */
  if (r<0x1p-126) return d<0 ? -0.0f : 0.0f;
  return (float)d;
}
static void ref_bf16(size_t ta,size_t tb,size_t m,size_t n,size_t k,float alpha,const float *a,
    const float *b,float beta,const float *c0,float *c) {
  size_t i,j,l0,p,h;
  for (i=0;i<m;i++) for (j=0;j<n;j++) {
    float acc=beta==0.0f ? 0.0f : beta*c0[i*n+j];
    for (l0=0;l0<k;l0+=32) {
      float e=0.0f,o=0.0f;
      for (p=0;p<16;p++) for (h=0;h<2;h++) {
        size_t l=l0+2*p+h;
        float av=l<k ? ref_bfv(alpha*a[ta ? l*m+i : i*k+l]) : 0.0f;
        float bv=l<k ? ref_bfv(b[tb ? j*k+l : l*n+j]) : 0.0f;
        if (h) o=ref_lane(o,av,bv); else e=ref_lane(e,av,bv);
      }
      acc=ref_flush(ref_flush(acc)+ref_flush(e+o));
    }
    c[i*n+j]=acc;
  }
}
static int same_bits_or_nan(const float *x,const float *y,size_t n) {
  size_t i;
  for (i=0;i<n;i++) if (!(isnan(x[i]) && isnan(y[i])) && memcmp(x+i,y+i,4)) return 0;
  return 1;
}
/* operands: mixed magnitudes; with `edges` a share of zeros of both signs, subnormals, +-2^-126,
 * products near 2^-150, and the odd infinity */
static float bf16_operand(unsigned i,int edges) {
  unsigned h=(i*2654435761u)^((i>>7)*40503u);
  if (edges) switch (h%11) {
  case 0: return 0.0f;
  case 1: return -0.0f;
  case 2: { uint32_t u=(h*2246822519u)&0x807fffffu; float f; memcpy(&f,&u,4); return f; }
  case 3: return h&16 ? -0x1p-126f : 0x1p-126f;
  case 4: return ldexpf(1.0f+(float)((h>>8)&63)/128.0f,-75-(int)((h>>14)&1))*(h&32 ? -1.0f : 1.0f);
  case 5: return h%97==5 ? (h&64 ? INFINITY : -INFINITY) : 1.5f;
  default: break;
  }
  return ldexpf(1.0f+(float)((h>>9)&0xffff)/65536.0f,(int)(h%17)-8)*(h&1 ? -1.0f : 1.0f);
}
/* --- the attention on bfloat16 operands (BF16-ATTN) ---------------------------------------------
 * The reference, written apart from the unit's: each product's element the bf16 GEMM's (ref_bfv,
 * ref_lane and ref_flush above) over the depth tiles the causal mask reaches from its 32-row block --
 * a query row i's key tiles 0 .. i/32 for the context and dq, a key row j's query tiles from j/32
 * on for dv and dk, every channel tile for the scores and dp -- a tile's terms past the mask the
 * +0.0 the probabilities and the adjoints hold there; the scaling, the softmax, the dot and the
 * score adjoint the v1 loops'. The score adjoints live here; p is the probabilities throughout.
 * Each operand is rounded to bfloat16 once, before the products read it (ref_round): the rounding is
 * a function of the operand alone, so it is the rounding at every term the operand reaches, and the
 * reference spends its time in its lanes rather than re-rounding one value at each of the t or d
 * terms it enters (BF16-ATTN-COST: the reference was two thirds of the harness's run without the
 * x86 instances). ref16 takes the rounded operands. */
static void ref_round(const float *x,size_t n,float *out) {
  size_t i;
  for (i=0;i<n;i++) out[i]=ref_bfv(x[i]);
}
static float ref16(const float *a,size_t sa,const float *b,size_t sb,size_t depth,size_t q0,size_t q1,
    float acc) {
  size_t q,pp,h;
  for (q=q0;q<q1;q++) {
    float e=0.0f,o=0.0f;
    for (pp=0;pp<16;pp++) for (h=0;h<2;h++) {
      size_t l=q*32+2*pp+h; float av=l<depth ? a[l*sa] : 0.0f,bv=l<depth ? b[l*sb] : 0.0f;
      if (h) o=ref_lane(o,av,bv); else e=ref_lane(e,av,bv);
    }
    acc=ref_flush(ref_flush(acc)+ref_flush(e+o));
  }
  return acc;
}
static void ref_attention_bf16(size_t b,size_t t,size_t nh,size_t nk,size_t d,const float *q,
    const float *k,const float *v,float *p,float *y,float (*ex)(float)) {
  static float qr[53760],kr[26880],vr[26880],pr[330];
  size_t r,h,i,j,c,dt=(d+31)/32; float scale=1.0f/sqrtf((float)d);
  assert(b*t*nh*d<=sizeof(qr)/sizeof(qr[0]) && b*t*nk*d<=sizeof(kr)/sizeof(kr[0]) && t<=sizeof(pr)/sizeof(pr[0]));
  ref_round(q,b*t*nh*d,qr); ref_round(k,b*t*nk*d,kr); ref_round(v,b*t*nk*d,vr);
  for (r=0;r<b;r++) for (h=0;h<nh;h++) {
    size_t kh=h/(nh/nk); float *ph=p+(r*nh+h)*t*t;
    for (i=0;i<t;i++) {
      float maxv=-INFINITY,sum=0.0f;
      for (j=0;j<t;j++) ph[i*t+j]=0.0f;
      for (j=0;j<=i;j++) {
        ph[i*t+j]=ref16(qr+(r*t+i)*nh*d+h*d,1,kr+(r*t+j)*nk*d+kh*d,1,d,0,dt,0.0f)*scale;
        if (ph[i*t+j]>maxv) maxv=ph[i*t+j];
      }
      for (j=0;j<=i;j++) { ph[i*t+j]=ex(ph[i*t+j]-maxv); sum+=ph[i*t+j]; }
      for (j=0;j<=i;j++) ph[i*t+j]/=sum;
    }
    for (i=0;i<t;i++) {
      ref_round(ph+i*t,t,pr);  /* the row's probabilities, rounded */
      for (c=0;c<d;c++) y[(r*t+i)*nh*d+h*d+c]=ref16(pr,1,vr+r*t*nk*d+kh*d+c,nk*d,t,0,i/32+1,0.0f);
    }
  }
}
static void ref_attention_backward_bf16(size_t b,size_t t,size_t nh,size_t nk,size_t d,
    const float *q,const float *k,const float *v,const float *p,const float *dy,float *dq,float *dk,
    float *dv,float *ds) {
  static float qr[53760],kr[26880],vr[26880],dyr[53760],pr[108900],dsr[108900];
  size_t r,h,i,j,c,dt=(d+31)/32,nt=(t+31)/32,share=nh/nk; float scale=1.0f/sqrtf((float)d);
  assert(b*t*nh*d<=sizeof(qr)/sizeof(qr[0]) && b*t*nk*d<=sizeof(kr)/sizeof(kr[0]) &&
      t*t<=sizeof(pr)/sizeof(pr[0]));
  ref_round(q,b*t*nh*d,qr); ref_round(k,b*t*nk*d,kr); ref_round(v,b*t*nk*d,vr);
  ref_round(dy,b*t*nh*d,dyr);
  for (r=0;r<b;r++) for (h=0;h<nh;h++) {
    size_t kh=h/share; int first=h%share==0; const float *ph=p+(r*nh+h)*t*t;
    const float *dyh=dyr+r*t*nh*d+h*d,*qh=qr+r*t*nh*d+h*d,*kk=kr+r*t*nk*d+kh*d,*vk=vr+r*t*nk*d+kh*d;
    float *dvk=dv+r*t*nk*d+kh*d,*dkk=dk+r*t*nk*d+kh*d,*dqh=dq+r*t*nh*d+h*d;
    ref_round(ph,t*t,pr);  /* the head's probabilities, rounded */
    for (j=0;j<t;j++) for (c=0;c<d;c++)
      dvk[j*nk*d+c]=ref16(pr+j,t,dyh+c,nh*d,t,j/32,nt,first ? 0.0f : dvk[j*nk*d+c]);
    for (i=0;i<t;i++) {
      float dot=0.0f;
      for (j=0;j<t;j++) ds[i*t+j]=0.0f;
      for (j=0;j<=i;j++) {
        float dp=ref16(dyh+i*nh*d,1,vk+j*nk*d,1,d,0,dt,0.0f);
        volatile float prod=ph[i*t+j]*dp;
        ds[i*t+j]=dp; dot+=prod;
      }
      for (j=0;j<=i;j++) ds[i*t+j]=ph[i*t+j]*(ds[i*t+j]-dot)*scale;
    }
    ref_round(ds,t*t,dsr);  /* the head's score adjoints, rounded */
    for (i=0;i<t;i++) for (c=0;c<d;c++) dqh[i*nh*d+c]=ref16(dsr+i*t,1,kk+c,nk*d,t,0,i/32+1,0.0f);
    for (j=0;j<t;j++) for (c=0;c<d;c++)
      dkk[j*nk*d+c]=ref16(dsr+j,t,qh+c,nh*d,t,j/32,nt,first ? 0.0f : dkk[j*nk*d+c]);
  }
}
static int all_finite(const float *x,size_t n);
/* The bf16 attention (BF16-ATTN) through its entries against the reference above: shapes of one row
 * and channel to past the 32-row blocks and tiles (ragged contexts and widths, head groups of one to
 * four, the backward's 330-token context), finite and with an infinite value and a NaN adjoint,
 * under both exponentials, and with the key the softmax hides (each score of a key -inf, its
 * probabilities 0, every output finite: the verdict sees the key itself); every instance the host
 * admits (portable, AVX-512, AMX), serial, on pools of two and four and on one with more workers than
 * tasks (a region a task); the backward leaving p as it was; the refusals writing nothing; and the
 * contract apart from the binary32 attention. A task is a (batch, KV head) and its region the pool's
 * choice, functions of the shapes and the pool alone, so the pools of two and four run on the finite
 * pass under libm's exponential, and the other passes and the BCIR exponential serially and on the
 * pool with more workers than tasks (BF16-ATTN-COST: every pool on every pass was most of the
 * harness's run where the instances are the portable loops, the aarch64 job's and the CI runners').
 * Under ThreadSanitizer, whose question is the races, the finite pass under libm's exponential on
 * every pool and instance; the numbers are the plain builds'. */
#if defined(__SANITIZE_THREAD__)
#define DT_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define DT_TSAN 1
#endif
#endif
#ifndef DT_TSAN
#define DT_TSAN 0
#endif
static void attention_bf16_reference(void) {
  static float q[53760],k[26880],v[26880],dy[53760],p0[217800],p1[217800],y0[53760],y1[53760],
      dq0[53760],dq1[53760],dk0[26880],dk1[26880],dv0[26880],dv1[26880],ds[108900],keep[217800];
  static _Alignas(64) unsigned char ws[4<<20];
  static _Alignas(max_align_t) unsigned char pool_memory[3][16384];
  static const size_t shapes[][5]={{1,1,1,1,1},{1,2,1,1,2},{2,5,4,2,6},{1,37,2,1,48},{2,70,8,4,48},
      {1,33,3,1,65},{1,40,2,2,130},{2,64,4,1,24},{1,96,4,2,40},{1,31,1,1,15},{1,32,2,2,47},
      {1,128,2,1,48},{1,330,2,1,8}};
  static const unsigned counts[]={1,2,4,64}; static const int variants[]={1,4,8};
  bcir_tensor_pool *pools[4]={NULL,NULL,NULL,NULL};
  int mask=bcir_tensor_mm_bf16_variants(),pass,ex,vi,bad; size_t s,i,pi,cases=0,apart=0,refusals=0,hidden=0;
  for (pi=1;pi<4;pi++) {
    size_t need=bcir_tensor_pool_size(counts[pi]);
    if (need && need<=sizeof(pool_memory[0])) pools[pi]=bcir_tensor_pool_init(pool_memory[pi-1],sizeof(pool_memory[0]),counts[pi]);
  }
  for (pass=0;pass<(DT_TSAN ? 1 : 3);pass++) for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
    size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
    size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,need=bcir_tensor_attention_bf16_workspace(b,t,nh,nk,d);
    assert(nq<=sizeof(q)/sizeof(q[0]) && nkv<=sizeof(k)/sizeof(k[0]) && np<=sizeof(p0)/sizeof(p0[0]));
    assert(t*t<=sizeof(ds)/sizeof(ds[0]) && need>0 && need<=sizeof(ws));
    if (pass==2 && t<2) continue;
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    if (pass==1 && t>3) { v[(t-1)*nk*d]=INFINITY; dy[d]=NAN; }
    if (pass==2) {  /* the last batch's key 1 infinite in channel 0, its KV head's queries negative there */
      size_t rb=b-1,h;
      for (i=0;i<t;i++) for (h=0;h<nh/nk;h++) q[((rb*t+i)*nh+h)*d]=-fabsf(q[((rb*t+i)*nh+h)*d])-0.5f;
      k[((rb*t+1)*nk+0)*d]=INFINITY;
    }
    for (ex=0;ex<(DT_TSAN ? 1 : 2);ex++) {
      unsigned contract=ex ? BCIR_TENSOR_ATTENTION_EXP : 0u;
      ref_attention_bf16(b,t,nh,nk,d,q,k,v,p0,y0,ex ? bcir_tensor_expf : expf);
      memset(dq0,0xff,nq*sizeof(float)); memset(dk0,0xff,nkv*sizeof(float)); memset(dv0,0xff,nkv*sizeof(float));
      ref_attention_backward_bf16(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0,ds);
      for (vi=0;vi<3;vi++) for (pi=0;pi<4;pi++) {
        /* WS-PLAN: each pool at the regions it uses -- one a worker where the workers are no more
         * than the tasks -- the bytes past them never touched */
        size_t own=bcir_tensor_attention_bf16_pool_workspace(b,t,nh,nk,d,bcir_tensor_pool_threads(pools[pi]));
        if (!(mask&variants[vi]) || (pi && !pools[pi])) continue;
        if ((pass || ex) && (pi==1 || pi==2)) continue;  /* two and four: the finite pass, libm's exp */
        assert(own>0 && own<=need && own+64<=sizeof(ws));
        memset(ws+own,0xa5,64);
        memset(p1,0xff,np*sizeof(float)); memset(y1,0xff,nq*sizeof(float)); bad=0;
        assert(bcir_tensor_attention_bf16_scan_pool(pools[pi],variants[vi],contract,b,t,nh,nk,d,q,k,v,p1,
            y1,&bad,ws,own)==variants[vi]);
        assert(same_bits_or_nan(p0,p1,np) && same_bits_or_nan(y0,y1,nq));
        for (i=0;i<np;i++) if ((i%t)>(i/t)%t) assert(p1[i]==0.0f && !signbit(p1[i]));
        assert(bad==!(all_finite(y1,nq) && all_finite(k,nkv)));
        if (pass==2) { assert(all_finite(p1,np) && all_finite(y1,nq) && bad); hidden++; }
        memcpy(keep,p1,np*sizeof(float));
        memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float)); memset(dv1,0xff,nkv*sizeof(float));
        assert(bcir_tensor_attention_bf16_backward_pool(pools[pi],variants[vi],b,t,nh,nk,d,q,k,v,p1,dy,dq1,
            dk1,dv1,ws,own)==variants[vi]);
        assert(same_bits_or_nan(dq0,dq1,nq) && same_bits_or_nan(dk0,dk1,nkv) && same_bits_or_nan(dv0,dv1,nkv));
        for (i=0;i<64;i++) assert(ws[own+i]==0xa5);
        assert(!memcmp(keep,p1,np*sizeof(float)));  /* p read, never written */
        cases++;
      }
      /* the binary32 attention under v1's products: another result somewhere */
      if (!pass) {
        bcir_tensor_attention_contract_scan_pool(NULL,contract,b,t,nh,nk,d,q,k,v,p1,y1,NULL);
        apart+=memcmp(y0,y1,nq*sizeof(float))!=0;
      }
    }
    /* refused, nothing written: a workspace one byte short of the serial entry's one region or none,
     * an instance the unit does not have, kvheads not dividing heads */
    {
      size_t one=bcir_tensor_attention_bf16_pool_workspace(b,t,nh,nk,d,1);
      assert(one<=need && (b*nk>1 ? one<need : one==need));
      memset(y1,0x5a,nq*sizeof(float)); memcpy(keep,y1,nq*sizeof(float)); memset(dq1,0x5a,nq*sizeof(float));
      assert(bcir_tensor_attention_bf16_scan_pool(NULL,0,0u,b,t,nh,nk,d,q,k,v,p1,y1,NULL,ws,one-1)==0);
      assert(bcir_tensor_attention_bf16_scan_pool(NULL,0,0u,b,t,nh,nk,d,q,k,v,p1,y1,NULL,NULL,need)==0);
      assert(bcir_tensor_attention_bf16_scan_pool(NULL,2,0u,b,t,nh,nk,d,q,k,v,p1,y1,NULL,ws,need)==0);
      assert(bcir_tensor_attention_bf16_backward_pool(NULL,0,b,t,nh,nk,d,q,k,v,p1,dy,dq1,dk1,dv1,ws,one-1)==0);
    }
    if (nh>1) assert(bcir_tensor_attention_bf16_scan_pool(NULL,0,0u,b,t,nh,nh+1,d,q,k,v,p1,y1,NULL,ws,sizeof(ws))==0);
    assert(!memcmp(keep,y1,nq*sizeof(float)) && !memcmp(keep,dq1,nq*sizeof(float)));
    refusals++;
  }
  for (pi=1;pi<4;pi++) bcir_tensor_pool_stop(pools[pi]);
  assert(apart>0 && cases>=(DT_TSAN ? 13*2u : 13*2*2*2u) && (DT_TSAN || hidden>=12*2));
  printf("  PASS bf16 attention (BF16-ATTN): %zu cases against the reference (%s; %s, the backward "
      "leaving p, serial and pooled), %zu with the key the softmax hides refused, apart from v1 on %zu, "
      "every refusal writing nothing on %zu shapes\n",cases,
      mask&8 ? "portable, AVX-512 and AMX" : mask&4 ? "portable and AVX-512" : "portable",
      DT_TSAN ? "ThreadSanitizer: the finite pass under libm's exponential" :
      "forward under both exponentials, finite and not",hidden,apart,refusals);
}
#ifdef BCIR_DT_PANELS_PROBE
int bcir_tensor_bf16_amx_offered(void);
#endif
static void gemm_bf16_contract(void) {
  static float a[150*200],b[200*150],c0[150*150],want[150*150],got[150*150];
  static _Alignas(64) unsigned char ws[1<<20];
  static const size_t dims[][3]={{1,1,1},{2,3,5},{15,17,16},{16,16,32},{17,31,33},{33,47,63},
      {64,65,64},{65,64,100},{100,33,130},{130,130,31},{3,150,200},{150,2,96}};
  static const float alphas[]={1.0f,0.75f,-1.25f},betas[]={0.0f,1.0f,0.5f};
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  int mask=bcir_tensor_mm_bf16_variants(),variant; size_t s,i,ta,tb,cases=0,chunked=0,swept=0;
  bcir_tensor_pool *pool=NULL; unsigned threads=bcir_tensor_pool_rails() ? 4 : 1; uint32_t u;
  assert((mask&1) && !(mask&~13));
  /* the conversion on every 4099th binary32 and its edges */
  for (u=0;;u+=4099u) {
    float f; memcpy(&f,&u,4); assert(bcir_tensor_bf16_bits(f)==ref_bf(f)); swept++;
    if (u>0xffffffffu-4099u) break;
  }
  {
    static const uint32_t edge[]={0x00000000u,0x80000000u,0x00000001u,0x807fffffu,0x00800000u,
        0x3f808000u,0x3f818000u,0x3f807fffu,0x3f808001u,0x7f7f8000u,0x7f7fffffu,0xff7f7fffu,
        0x7f800000u,0xff800000u,0x7f800001u,0xffc00123u,0x7fbfffffu};
    for (i=0;i<sizeof(edge)/sizeof(edge[0]);i++) {
      float f; memcpy(&f,edge+i,4); assert(bcir_tensor_bf16_bits(f)==ref_bf(f)); swept++;
    }
  }
  if (threads>1 && bcir_tensor_pool_size(threads) && bcir_tensor_pool_size(threads)<=sizeof(pool_memory))
    pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads);
  for (s=0;s<sizeof(dims)/sizeof(dims[0]);s++) for (ta=0;ta<2;ta++) for (tb=0;tb<2;tb++) {
    size_t m=dims[s][0],n=dims[s][1],k=dims[s][2],full=bcir_tensor_mm_bf16_workspace(m,n,k);
    size_t least=2*64*((k+31)/32*32)+64,sizes[3],z;
    float alpha=alphas[(s+ta)%3],beta=betas[(s+tb)%3]; int edges=(int)(s%2);
    for (i=0;i<m*k;i++) a[i]=bf16_operand((unsigned)(i*3+s),edges);
    for (i=0;i<k*n;i++) b[i]=bf16_operand((unsigned)(i*5+s+7),edges);
    for (i=0;i<m*n;i++) c0[i]=bf16_operand((unsigned)(i*7+s+11),edges);
    if (beta==0.0f) c0[0]=NAN;  /* C is not read under beta 0 */
    ref_bf16(ta,tb,m,n,k,alpha,a,b,beta,c0,want);
    assert(full==2*((m+31)/32*32+(n+31)/32*32)*((k+31)/32*32)+64 && full<=sizeof(ws)-64);
    sizes[0]=full; sizes[1]=least; sizes[2]=(full+least)/2;
    for (variant=1;variant<=8;variant*=2) {
      if (!(mask&variant)) {
        memcpy(got,c0,m*n*4);
        assert(bcir_tensor_mm_bf16_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,got,ws,full)==0);
        assert(!memcmp(got,c0,m*n*4));
        continue;
      }
      for (z=0;z<3;z++) {
        memcpy(got,c0,m*n*4);
        assert(bcir_tensor_mm_bf16_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,got,ws+z,sizes[z])==variant);
        assert(same_bits_or_nan(got,want,m*n)); cases++; chunked+=sizes[z]<full;
      }
      /* below one strip of each: refused, nothing written */
      memcpy(got,c0,m*n*4);
      assert(bcir_tensor_mm_bf16_variant(variant,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,got,ws,least-1)==0);
      assert(!memcmp(got,c0,m*n*4));
    }
    memcpy(got,c0,m*n*4);
    assert(bcir_tensor_mm_bf16_pool(pool,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,got,ws,full)==1);
    assert(same_bits_or_nan(got,want,m*n));
    memcpy(got,c0,m*n*4);  /* no depth: beta alone, +0.0 under beta 0, and no workspace needed */
    assert(bcir_tensor_mm_bf16_pool(pool,(int)ta,(int)tb,m,n,0,alpha,a,b,beta,got,NULL,0)==1);
    for (i=0;i<m*n;i++) {
      float e=beta==0.0f ? 0.0f : beta*c0[i];
      assert(!memcmp(got+i,&e,4) || (isnan(e) && isnan(got[i])));
    }
  }
  assert(bcir_tensor_mm_bf16_variant(2,0,0,1,1,1,1.0f,a,b,0.0f,got,ws,sizeof(ws))==0);
  assert(bcir_tensor_mm_bf16_variant(3,0,0,1,1,1,1.0f,a,b,0.0f,got,ws,sizeof(ws))==0);
  /* no rows or no columns: every entry accepts the call and writes nothing -- alone, through
   * each instance, with and without workspace, and grouped, in a chain and side by side */
  {
    static const size_t empty[][3]={{0,5,7},{6,0,7},{0,0,7},{0,5,0},{6,0,0},{0,40,64}};
    for (s=0;s<sizeof(empty)/sizeof(empty[0]);s++) {
      size_t m=empty[s][0],n=empty[s][1],k=empty[s][2]; bcir_tensor_mm_call e[2];
      memcpy(got,c0,sizeof got);
      for (variant=1;variant<=8;variant*=2) if (mask&variant)
        assert(bcir_tensor_mm_bf16_variant(variant,0,1,m,n,k,1.0f,a,b,0.5f,got,ws,sizeof(ws))==variant);
      assert(bcir_tensor_mm_bf16_pool(pool,0,1,m,n,k,1.0f,a,b,0.5f,got,ws,sizeof(ws))==1);
      assert(bcir_tensor_mm_bf16_pool(pool,1,0,m,n,k,-2.0f,a,b,0.0f,got,NULL,0)==1);
      e[0].ta=0; e[0].tb=0; e[0].m=m; e[0].n=n; e[0].k=k; e[0].alpha=1.0f; e[0].a=a; e[0].b=b;
      e[0].beta=0.0f; e[0].c=got; e[1]=e[0]; e[1].beta=1.0f;
      assert(bcir_tensor_mm_bf16_group_pool(pool,e,2,NULL,0,ws,sizeof(ws))==1);
      assert(bcir_tensor_mm_bf16_group_pool(pool,NULL,0,e,2,ws,sizeof(ws))==1);
      assert(bcir_tensor_mm_bf16_group_pool(pool,e,1,e+1,1,NULL,0)==1);
      assert(!memcmp(got,c0,sizeof got));
    }
  }
  /* the contract's edges, each element's bits stated: (0,0) a lane at 2^-126 then -1.328*2^-151,
   * tiny though 2^-126 on the subnormal grid -> +0; (1,1) c = -0 and every product -0, the lanes
   * from +0 -> +0; (2,2) a subnormal c (DAZ) + 2^-126 -> 2^-126; (3,3) products of 2^-140 each
   * flushed in lanes from +0 -> c = 1 */
  {
    enum { M=4, N=4, K=40 }; float ea[M*K],eb[K*N],ec[M*N],eg[M*N];
    static const uint32_t diag[4]={0x00000000u,0x00000000u,0x00800000u,0x3f800000u}; size_t l;
    memset(ea,0,sizeof ea); memset(eb,0,sizeof eb); memset(ec,0,sizeof ec);
    for (l=0;l<K;l++) { ea[1*K+l]=1.0f; eb[l*N+1]=-0.0f; ea[3*K+l]=0x1p-70f; eb[l*N+3]=0x1p-70f; }
    ea[0]=1.0f; eb[0]=0x1p-126f; ea[2]=-0x1.4p-75f; eb[2*N]=0x1.1p-76f; ea[2*K]=1.0f; eb[2]=0x1p-126f;
    ec[1*N+1]=-0.0f; ec[2*N+2]=0x1p-149f; ec[3*N+3]=1.0f;
    for (variant=1;variant<=8;variant*=2) {
      if (!(mask&variant)) continue;
      memcpy(eg,ec,sizeof eg);
      assert(bcir_tensor_mm_bf16_variant(variant,0,0,M,N,K,1.0f,ea,eb,1.0f,eg,ws,sizeof(ws))==variant);
      for (i=0;i<4;i++) { uint32_t g; memcpy(&g,eg+i*N+i,4); assert(g==diag[i]); }
      ref_bf16(0,0,M,N,K,1.0f,ea,eb,1.0f,ec,want); assert(same_bits_or_nan(eg,want,M*N));
    }
  }
  /* the groups: q/k/v (A shared, packed once) and the backward's chain into one dx with the dW
   * calls (B shared); whole, with too little for the group (one call at a time), and chunked */
  {
    enum { R=70, D=40, KV=24 }; static float x[R*D],wq[3][D*D],y[3][R*D],yr[3][R*D],dy[3][R*D];
    static float dw[3][D*D],dwr[3][D*D],dx[R*D],dxr[R*D];
    const size_t outs[3]={D,KV,KV}; bcir_tensor_mm_call cf[3],cx[3],cw[3]; size_t q,gf,gb,unshared=0,shrink;
    for (i=0;i<R*D;i++) { x[i]=bf16_operand((unsigned)i,0); dx[i]=dxr[i]=bf16_operand((unsigned)i+9,0); }
    for (q=0;q<3;q++) {
      for (i=0;i<outs[q]*D;i++) { wq[q][i]=bf16_operand((unsigned)(i+q*977),0); dw[q][i]=dwr[q][i]=bf16_operand((unsigned)(i+q*31),0); }
      for (i=0;i<R*outs[q];i++) dy[q][i]=bf16_operand((unsigned)(i+q*7919),0);
      cf[q].ta=0; cf[q].tb=1; cf[q].m=R; cf[q].n=outs[q]; cf[q].k=D; cf[q].alpha=1.0f;
      cf[q].a=x; cf[q].b=wq[q]; cf[q].beta=0.0f; cf[q].c=y[q];
      cx[q].ta=0; cx[q].tb=0; cx[q].m=R; cx[q].n=D; cx[q].k=outs[q]; cx[q].alpha=1.0f;
      cx[q].a=dy[q]; cx[q].b=wq[q]; cx[q].beta=q ? 1.0f : 0.5f; cx[q].c=dx;
      cw[q].ta=1; cw[q].tb=0; cw[q].m=outs[q]; cw[q].n=D; cw[q].k=R; cw[q].alpha=1.0f;
      cw[q].a=dy[q]; cw[q].b=x; cw[q].beta=1.0f; cw[q].c=dw[q];
      ref_bf16(0,1,R,outs[q],D,1.0f,x,wq[q],0.0f,yr[q],yr[q]);
      unshared+=bcir_tensor_mm_bf16_workspace(R,outs[q],D)-64;
    }
    for (q=0;q<3;q++) ref_bf16(0,0,R,D,outs[q],1.0f,dy[q],wq[q],q ? 1.0f : 0.5f,dxr,dxr);
    for (q=0;q<3;q++) ref_bf16(1,0,outs[q],D,R,1.0f,dy[q],x,1.0f,dwr[q],dwr[q]);
    gf=bcir_tensor_mm_bf16_group_workspace(NULL,0,cf,3); gb=bcir_tensor_mm_bf16_group_workspace(cx,3,cw,3);
    /* A packed once for the three: the group less than the three calls' packs by two of x's */
    assert(gf+2*(bcir_tensor_mm_bf16_workspace(R,32,D)-64-2*32*((D+31)/32*32))==unshared+64 && gb<=sizeof(ws));
    for (shrink=0;shrink<3;shrink++) {
      size_t bytes=shrink==0 ? gf : shrink==1 ? gf-1 : 2*64*64+64;
      for (q=0;q<3;q++) memset(y[q],0xff,sizeof y[q]);
      assert(bcir_tensor_mm_bf16_group_pool(pool,NULL,0,cf,3,ws,bytes)==1);
      for (q=0;q<3;q++) assert(same_bits_or_nan(y[q],yr[q],R*outs[q]));
    }
    assert(bcir_tensor_mm_bf16_group_pool(pool,cx,3,cw,3,ws,gb)==1);
    assert(same_bits_or_nan(dx,dxr,R*D));
    for (q=0;q<3;q++) assert(same_bits_or_nan(dw[q],dwr[q],outs[q]*D));
  }
  bcir_tensor_pool_stop(pool);
#ifdef BCIR_DT_PANELS_PROBE
  /* the AMX instance admitted wherever this build and the host offer it: its admission differential
     passes here -- a wrong AMX kernel is withdrawn and never run, so no value above could show it */
  assert(!bcir_tensor_bf16_amx_offered() || (mask&8));
#endif
  printf("  PASS bf16 contract (NDT-BF16): instances %d, %zu GEMMs (%zu chunked) and the groups against the "
      "reference, the conversion on %zu inputs, the edges' bits, the empty shapes\n",mask,cases,chunked,swept);
}
/* The trainer under the bf16 contract: its refusals, the workspace the plan asks for (never less
 * than a step's GEMM groups pack -- the probe build's record), the state at every thread count. */
#ifdef BCIR_DT_WORKSPACE_PROBE
size_t bcir_decoder_workspace_probe(void);
#endif
static void trainer_bf16(void) {
  static float ref[3][4096]; static _Alignas(max_align_t) unsigned char pool_memory[4096];
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  uint32_t ids[120],targets[120],x[]={1,2,3,4}; bcir_decoder_event events[3]; size_t done,need;
  bcir_decoder_state s=fixture(),bad; unsigned threads; bcir_decoder_plan refused=s.plan;
  double loss;
  s.contract=BCIR_DT_CONTRACT_BF16; need=bcir_decoder_workspace_bytes(&s.plan,s.contract);
  assert(need>0 && need<=sizeof(workspace) && bcir_decoder_workspace_bytes(&s.plan,0)==att32_need(&s.plan,0) &&
      bcir_decoder_workspace_bytes(&s.plan,BCIR_DT_CONTRACT_KBLOCK)==att32_need(&s.plan,0) &&
      bcir_decoder_workspace_bytes(NULL,BCIR_DT_CONTRACT_BF16)==SIZE_MAX);
  refused.spec.heads=3; assert(bcir_decoder_workspace_bytes(&refused,BCIR_DT_CONTRACT_BF16)==SIZE_MAX);
  /* ATT-CONTIG: without BF16 a plan the planner refuses copies nothing -- 0, as no plan at all */
  assert(bcir_decoder_workspace_bytes(&refused,0)==0 && bcir_decoder_workspace_bytes(NULL,0)==0 &&
      bcir_decoder_workspace_need(&refused,BCIR_DT_CONTRACT_ATTN,4)==0 && bcir_decoder_workspace_need(NULL,0,4)==0);
  /* WS-PLAN: the plan's workspace is what the step's largest group packs. For the 9.84M decoder
   * (vocab 260, width 384, 8 heads over 4, 6 layers, FF 1024, 8 x 128 tokens) that is the
   * backward's q/k/v group: its three input adjoints' A (1024 rows by 384, by 192 twice) and B (the
   * weights, 384 by 384 and 384 by 192 twice), its three weight adjoints' A (384, 192 and 192 rows
   * by 1024) and the attention norm's output as their B, once -- 2 * (393,216 + 2 * 196,608 +
   * 147,456 + 2 * 73,728 + 393,216 + 2 * 196,608 + 393,216) = 4,521,984 bytes and 64 of
   * alignment. The gate/up group, every call packed whole, was the plan's figure (four calls of
   * 1024 x 384 x 1024, 11,534,592); it reads its adjoints as panels, and its panel groups pack
   * 2,359,360. The floor is a 32-row strip and a 32-column strip at the rows' depth (1024). */
  {
    bcir_decoder_spec big={260,384,8,4,6,1024,8,128,1,10000.0,1e-6}; bcir_decoder_plan bp;
    assert(bcir_decoder_make_plan(&big,&bp)==0);
    /* the GEMM groups' figure, and the binary32 attention's regions where the build copies there */
#ifndef BCIR_DT_NO_PANELS
    assert(gemm_984m()==4522048u);
    assert(bcir_decoder_workspace_bytes(&bp,BCIR_DT_CONTRACT_BF16)==most_of(gemm_984m(),att32_need(&bp,0)));
    assert(bcir_decoder_workspace_need(&bp,BCIR_DT_CONTRACT_BF16,4)==most_of(gemm_984m(),att32_need(&bp,4)));
    assert(bcir_decoder_workspace_panels(&bp,BCIR_DT_CONTRACT_BF16)==2359360u);
#else  /* the gate/up group packed: its four calls whole but the attention norm's B once, 786,432 less */
    assert(bcir_decoder_workspace_bytes(&bp,BCIR_DT_CONTRACT_BF16)==most_of(gemm_984m(),att32_need(&bp,0)) &&
        gemm_984m()==4*bcir_tensor_mm_bf16_workspace(1024,384,1024)-786432u-3*64u);
    assert(bcir_decoder_workspace_panels(&bp,BCIR_DT_CONTRACT_BF16)==0);
#endif
    assert(bcir_decoder_workspace_floor(&bp,BCIR_DT_CONTRACT_BF16,1)==bcir_tensor_mm_bf16_workspace(32,32,1024) &&
        bcir_tensor_mm_bf16_workspace(32,32,1024)==131136u);
    assert(bcir_decoder_workspace_panels(&bp,0)==0 && bcir_decoder_workspace_floor(&bp,0,4)==0 &&
        bcir_decoder_workspace_need(&bp,BCIR_DT_CONTRACT_KBLOCK,4)==att32_need(&bp,4));
    assert(bcir_decoder_workspace_panels(NULL,BCIR_DT_CONTRACT_BF16)==SIZE_MAX &&
        bcir_decoder_workspace_floor(NULL,BCIR_DT_CONTRACT_BF16,4)==SIZE_MAX &&
        bcir_decoder_workspace_need(NULL,BCIR_DT_CONTRACT_BF16,4)==SIZE_MAX);
  }
  bad=s; bad.contract|=BCIR_DT_CONTRACT_FUSED; lend(&bad); assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  bad=s; bad.contract|=BCIR_DT_CONTRACT_KBLOCK; lend(&bad); assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  bad=s; assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);  /* no workspace */
  /* WS-PLAN: admitted from the floor up (a byte short of the need runs, in chunks), refused below */
  bad=s; bad.workspace=workspace; bad.workspace_bytes=need-1; assert(bcir_decoder_forward(&bad,x,4)==0);
  bad=s; bad.workspace=workspace; bad.workspace_bytes=bcir_decoder_workspace_floor(&s.plan,s.contract,1)-1;
  assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_CAPACITY);
  bad=s; bad.workspace=w; bad.workspace_bytes=need; assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  bad=s; lend(&bad); assert(bcir_decoder_loss_backward(&bad,x,x,4,0,(double *)(void *)workspace)==BCIR_DT_INVALID);
  /* ATT-CONTIG: without the bf16 contract a lent workspace is the attention's -- any size, none the
   * default -- and held apart from the state and the arenas all the same */
  bad=fixture(); bad.workspace=w; bad.workspace_bytes=64; assert(bcir_decoder_forward(&bad,x,4)==BCIR_DT_INVALID);
  bad=fixture(); bad.workspace=workspace; bad.workspace_bytes=1; assert(bcir_decoder_forward(&bad,x,4)==0);
  bad=fixture(); bad.workspace=workspace; bad.workspace_bytes=0; assert(bcir_decoder_forward(&bad,x,4)==0);
  bad=fixture(); bad.workspace=workspace; bad.workspace_bytes=64;  /* a caller's buffer in it, refused */
  assert(bcir_decoder_loss_backward(&bad,x,x,4,0,(double *)(void *)workspace)==BCIR_DT_INVALID);
  bad=fixture(); bad.workspace=workspace; bad.workspace_bytes=64; assert(bcir_decoder_loss_backward(&bad,x,x,4,0,&loss)==0);
  stream(ids,targets,120);
  for (threads=1;threads<=4;threads++) {
    bcir_tensor_pool *pool=NULL; bcir_decoder_state st=fixture_of(5,8,1,2);
    if (threads>1) {
      if (!bcir_tensor_pool_size(threads) || bcir_tensor_pool_size(threads)>sizeof(pool_memory)) continue;
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads);
    }
    st.pool=pool; st.contract=BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE; lend(&st);
    assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
    if (threads==1) { memcpy(ref[0],w,sizeof w); memcpy(ref[1],m,sizeof m); memcpy(ref[2],v,sizeof v); }
    else assert(!memcmp(ref[0],w,sizeof w) && !memcmp(ref[1],m,sizeof m) && !memcmp(ref[2],v,sizeof v));
    bcir_tensor_pool_stop(pool);
  }
  /* the exp bit acts under bf16: one forward's activations apart, bf16 against bf16+exp (their
   * states can coincide, the exponential's last bits under the operands' rounding: pool_identity) */
  {
    static float a16[32768]; bcir_decoder_state st=fixture();
    st.contract=BCIR_DT_CONTRACT_BF16; lend(&st);
    assert(bcir_decoder_forward(&st,x,4)==0); memcpy(a16,act,sizeof a16);
    st=fixture(); st.contract=BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP; lend(&st);
    assert(bcir_decoder_forward(&st,x,4)==0); assert(memcmp(a16,act,sizeof a16)!=0);
  }
  /* it learns: forty steps under bf16 bring the loss down as v1's do */
  {
    bcir_decoder_state st=fixture(); double first=0.0,last=0.0; unsigned step; uint32_t y[]={2,3,4,5};
    bcir_decoder_adamw o={0.02,0.9,0.95,1e-8,0.0,1.0};
    st.contract=BCIR_DT_CONTRACT_BF16; lend(&st);
    for (step=0;step<40;step++) {
      double gn;
      assert(bcir_decoder_loss_backward(&st,x,y,4,0,&loss)==0);
      if (!step) first=loss;
      last=loss;
      assert(bcir_decoder_update(&st,&o,1.0,&gn)==0);
    }
    assert(last<0.5*first);
  }
#ifdef BCIR_DT_WORKSPACE_PROBE
  {
    bcir_decoder_state st=fixture_of(5,8,0,2); size_t seen;
    st.contract=BCIR_DT_CONTRACT_BF16; lend(&st);
    (void)bcir_decoder_workspace_probe();
    assert(bcir_decoder_run(&st,&opt,ids,targets,40,NULL,1,events,1,&done)==0 && done==1);
    seen=bcir_decoder_workspace_probe();
    {  /* what the GEMM groups pack is the plan's need -- or below a need the attention's copies set */
      size_t want=bcir_decoder_workspace_bytes(&st.plan,st.contract);
      assert(seen>0 && (seen==want || (want==att32_need(&st.plan,0) && seen<want)));
    }
    printf("  PASS bf16 workspace probe: the step's largest GEMM group packs the %zu bytes the plan asks\n",seen);
  }
#endif
  printf("  PASS bf16 trainer (NDT-BF16): refusals, %zu bytes of workspace, the state at every thread count, learning\n",need);
}
/* The trainer under the fused attention contract (ATTN-FMA): the bit admitted with each GEMM
 * contract and with none, its state its own and one at every thread count (pool_identity), and --
 * where the build has the tensor unit's probe -- every attention job of a run, forward and
 * backward, running the fused products under the bit and v1's without it: the contract each job
 * computed with, so a wrong argument at any of the trainer's calls is seen (the stored forwards
 * under BCIR_TENSOR_ATTENTION_LOWER too: P-LOWER). */
#ifdef BCIR_DT_ATTENTION_PROBE
unsigned bcir_tensor_attention_probe(int backward);
#endif
static void trainer_attn(void) {
  static const uint32_t gemm[]={0,BCIR_DT_CONTRACT_FUSED,BCIR_DT_CONTRACT_KBLOCK,BCIR_DT_CONTRACT_BF16};
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  uint32_t ids[40],targets[40]; bcir_decoder_event events[1]; size_t done,i; unsigned with;
#ifdef BCIR_DT_ATTENTION_PROBE
  {  /* the probe itself: no job reads as no contract, two contracts as none */
    float q[4]={0.5f,-0.25f,1.0f,0.75f},p[4],y[4];
    (void)bcir_tensor_attention_probe(0); (void)bcir_tensor_attention_probe(1);
    assert(bcir_tensor_attention_probe(0)==~0u && bcir_tensor_attention_probe(1)==~0u);
    bcir_tensor_attention_contract_scan_pool(NULL,BCIR_TENSOR_ATTENTION_FUSED,1,2,1,1,2,q,q,q,p,y,NULL);
    assert(bcir_tensor_attention_probe(0)==BCIR_TENSOR_ATTENTION_FUSED);
    bcir_tensor_attention_contract_scan_pool(NULL,BCIR_TENSOR_ATTENTION_FUSED,1,2,1,1,2,q,q,q,p,y,NULL);
    bcir_tensor_attention_contract_scan_pool(NULL,0,1,2,1,1,2,q,q,q,p,y,NULL);
    assert(bcir_tensor_attention_probe(0)==~0u);
  }
#endif
  stream(ids,targets,40);
  for (i=0;i<sizeof(gemm)/sizeof(gemm[0]);i++) for (with=0;with<2;with++) {
    bcir_decoder_state st=fixture_of(5,8,1,2);
    st.contract=gemm[i]|BCIR_DT_CONTRACT_EXP|(with ? BCIR_DT_CONTRACT_ATTN : 0u); lend(&st);
#ifdef BCIR_DT_ATTENTION_PROBE
    (void)bcir_tensor_attention_probe(0); (void)bcir_tensor_attention_probe(1);
#endif
    assert(bcir_decoder_run(&st,&opt,ids,targets,40,NULL,1,events,1,&done)==0 && done==1);
#ifdef BCIR_DT_ATTENTION_PROBE
    {
      unsigned fused=with ? BCIR_TENSOR_ATTENTION_FUSED : 0u;
      assert(bcir_tensor_attention_probe(0)==(BCIR_TENSOR_ATTENTION_EXP|fused|BCIR_TENSOR_ATTENTION_LOWER));
      assert(bcir_tensor_attention_probe(1)==fused);
    }
#endif
  }
  {
#ifdef BCIR_DT_ATTENTION_PROBE
    static const char probed[]=", every attention job of a run, forward and backward, on its contract";
#else
    static const char probed[]="";
#endif
    printf("  PASS fused attention trainer (ATTN-FMA): the bit with each GEMM contract and none%s\n",probed);
  }
}
/* The trainer under the bf16 attention (BF16-ATTN): one state at every thread count, under each of the
 * bf16 contract's companions; another state than the bf16 contract's without the bit; every attention
 * job of a run on the bf16 contract (the probe); the backward leaving the probabilities in the arena;
 * the workspace sized for the attention too. */
static void trainer_attnbf16(void) {
  static float ref[3][4096],probs[32768]; static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const uint32_t contracts[]={BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTNBF16,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTNBF16,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTNBF16};
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  uint32_t ids[120],targets[120]; bcir_decoder_event events[3]; size_t done,c; unsigned threads;
  stream(ids,targets,120);
  {  /* the workspace: at least the GEMMs' and the attention's, for the 9.84M decoder and the fixture */
    bcir_decoder_spec big={260,384,8,4,6,1024,8,128,1,10000.0,1e-6}; bcir_decoder_plan bp;
    bcir_decoder_state st=fixture_of(5,8,1,2); size_t att;
    assert(bcir_decoder_make_plan(&big,&bp)==0);
    att=bcir_tensor_attention_bf16_workspace(8,128,8,4,48);
    assert(bcir_decoder_workspace_bytes(&bp,BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTNBF16)==most_of(att,gemm_984m()));
    /* WS-PLAN: a pool of four uses four of the 32 (batch, KV head) tasks' regions -- 1,081,408
     * bytes, the floor; the step's GEMMs then ask more */
    assert(bcir_tensor_attention_bf16_pool_workspace(8,128,8,4,48,4)==1081408u &&
        bcir_tensor_attention_bf16_pool_workspace(8,128,8,4,48,0)==att && att==8650816u &&
        bcir_tensor_attention_bf16_pool_workspace(8,128,8,4,48,64)==att);
    assert(bcir_decoder_workspace_floor(&bp,BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTNBF16,4)==1081408u &&
        bcir_decoder_workspace_need(&bp,BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTNBF16,4)==gemm_984m());
    /* the plain bf16 contract's figure the GEMMs' where the binary32 attention copies nothing (ATT-CONTIG) */
    assert(bcir_decoder_workspace_bytes(&st.plan,contracts[0])>=bcir_tensor_attention_bf16_workspace(5,8,1,1,2) &&
        (att32_need(&st.plan,0) ||
         bcir_decoder_workspace_bytes(&st.plan,contracts[0])>=bcir_decoder_workspace_bytes(&st.plan,BCIR_DT_CONTRACT_BF16)));
  }
  for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) {
    for (threads=1;threads<=4;threads++) {
      bcir_tensor_pool *pool=NULL; bcir_decoder_state st=fixture_of(5,8,1,2);
      if (threads>1) {
        if (!bcir_tensor_pool_size(threads) || bcir_tensor_pool_size(threads)>sizeof(pool_memory)) continue;
        pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads);
      }
      st.pool=pool; st.contract=contracts[c]; lend(&st);
#ifdef BCIR_DT_ATTENTION_PROBE
      (void)bcir_tensor_attention_probe(0); (void)bcir_tensor_attention_probe(1);
#endif
      assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
#ifdef BCIR_DT_ATTENTION_PROBE
      assert(bcir_tensor_attention_probe(0)==(((contracts[c]&BCIR_DT_CONTRACT_EXP) ? BCIR_TENSOR_ATTENTION_EXP : 0u)|
          BCIR_TENSOR_ATTENTION_BF16));
      assert(bcir_tensor_attention_probe(1)==BCIR_TENSOR_ATTENTION_BF16);
#endif
      if (threads==1) { memcpy(ref[0],w,sizeof w); memcpy(ref[1],m,sizeof m); memcpy(ref[2],v,sizeof v); }
      else assert(!memcmp(ref[0],w,sizeof w) && !memcmp(ref[1],m,sizeof m) && !memcmp(ref[2],v,sizeof v));
      bcir_tensor_pool_stop(pool);
    }
    {  /* the bit changes the state: the same steps without it */
      bcir_decoder_state st=fixture_of(5,8,1,2);
      st.contract=contracts[c]&~(uint32_t)BCIR_DT_CONTRACT_ATTNBF16; lend(&st);
      assert(bcir_decoder_run(&st,&opt,ids,targets,120,NULL,3,events,3,&done)==0 && done==3);
      assert(memcmp(ref[0],w,sizeof w)!=0);
    }
  }
  {  /* the backward leaves each layer's probabilities as the forward wrote them; the binary32
      * attention's rewrites them with its score adjoints, so the regions are the probabilities */
    static const uint32_t pair[2]={BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTNBF16,
        BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP};
    size_t k2;
    for (k2=0;k2<2;k2++) {
      bcir_decoder_state st=fixture_of(5,8,1,2); double loss; const bcir_decoder_spec *sp=&st.plan.spec;
      size_t n=(size_t)sp->batch*sp->time,d=sp->width,hd=d/sp->heads,nd=n*d,nk=n*sp->kvheads*hd;
      size_t a=(size_t)sp->batch*sp->heads*sp->time*sp->time,nf=n*sp->ff,la=5*nd+2*nk+a+3*nf,l,kept=0;
      st.contract=pair[k2]; lend(&st);
      assert(a*sp->layers<=sizeof(probs)/sizeof(probs[0]));
      assert(bcir_decoder_forward(&st,ids,40)==0);
      for (l=0;l<sp->layers;l++) memcpy(probs+l*a,act+(sp->layers+1)*nd+l*la+2*nd+2*nk,a*sizeof(float));
      assert(bcir_decoder_loss_backward(&st,ids,targets,40,0,&loss)==0);
      for (l=0;l<sp->layers;l++)
        kept+=!memcmp(probs+l*a,act+(sp->layers+1)*nd+l*la+2*nd+2*nk,a*sizeof(float));
      assert(kept==(k2 ? 0 : sp->layers));
    }
  }
  printf("  PASS bf16 attention trainer (BF16-ATTN): three bf16 contracts with the bit, one state at one to "
      "four threads and another without it, every attention job of a run on its contract, the arena's "
      "probabilities left as the forward wrote them\n");
}
#ifdef BCIR_DT_PANELS_PROBE
unsigned bcir_decoder_panels_probe(void);
unsigned bcir_tensor_panels_probe(void);
#endif
/* the float whose top 16 bits are v's and whose low 16 are 0x8000: the midpoint between two
 * bfloat16 values, where the conversion turns on the last bits of what is converted */
static float bf16_midpoint(float v) {
  uint32_t b; memcpy(&b,&v,4); b=(b&0xffff0000u)|0x8000u; memcpy(&v,&b,4); return v;
}
/* BF16-PANELS: the layouts as the harness reads bcir_tensor.h -- A (rows x depth): tile
 * (i/16, l/32), row i%16, lane l%32; B (depth x columns): tile (j/16, l/32), pair (l%32)/2, lane
 * 2*(j%16)+l%2; A^T: A's place of the transpose's element -- each tile 512 elements, tiles of one
 * row (column) block in depth order */
static size_t ref_panel_a(size_t depth,size_t i,size_t l) {
  return (((i/16)*(depth/32))+l/32)*512+(i%16)*32+(l%32);
}
static size_t ref_panel_b(size_t depth,size_t l,size_t j) {
  return (((j/16)*(depth/32))+l/32)*512+((l%32)/2)*32+2*(j%16)+(l%2);
}
/* an r x c tensor's panel as A (into a) and as B (b) or A^T (b, at), by the conversion alone */
static void ref_panels(size_t r,size_t c,const float *x,unsigned short *a,unsigned short *b,int at) {
  size_t i,l;
  for (i=0;i<r;i++) for (l=0;l<c;l++) {
    unsigned short h=(unsigned short)bcir_tensor_bf16_bits(x[i*c+l]);
    a[ref_panel_a(c,i,l)]=h;
    if (at) b[ref_panel_a(r,l,i)]=h; else b[ref_panel_b(r,i,l)]=h;
  }
}
static int same_shorts(const unsigned short *a,const unsigned short *b,size_t n) {
  return !memcmp(a,b,n*sizeof(*a));
}
/* a writer's panel against the reference's, element by element: the same bits, and a quiet NaN
 * wherever the reference holds one. Which NaN an operation on two NaNs propagates is the compiled
 * code's (the operand order its instructions take), so a binary32 kernel's NaN -- the writers'
 * copy of it and the entries' -- is a quiet NaN, not a pattern: what the bf16 contract says of its
 * results, and what same_bits_or_nan holds the GEMMs to. The first difference is reported, with
 * its case. A quiet NaN: the exponent's ones and the significand's top bit (bf16_bits' 0x0040). */
static int bf16_quiet_nan(unsigned short h) { return (h&0x7fc0u)==0x7fc0u; }
static int same_panel(const char *what,size_t r,size_t c,int v2,unsigned threads,
    const unsigned short *got,const unsigned short *want) {
  size_t i,n=r*c;
  for (i=0;i<n;i++) if (bf16_quiet_nan(want[i]) ? !bf16_quiet_nan(got[i]) : got[i]!=want[i]) {
    fprintf(stderr,"panels: %s, %zu x %zu, v%d, %u thread(s): element %zu is 0x%04x, the reference 0x%04x\n",
        what,r,c,v2 ? 2 : 1,threads,i,(unsigned)got[i],(unsigned)want[i]);
    return 0;
  }
  return 1;
}
static void panels_contract(void) {
  enum { N=64*1056 };
  static float x[N],u[N],dy[N],y[N],dg[N],du[N],c0[N],want[N],got[N];
  static unsigned short pa[N],pb[N],pc[N],pd[N],ra[N],rb[N],rc[N],rd[N];
  static _Alignas(64) unsigned char ws[1<<19];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  /* the last two rows longer than a writer's segment of 1024 columns: a row in two calls */
  static const size_t shapes[][2]={{32,32},{64,96},{96,160},{32,128},{64,32},{32,1056},{64,1056}};
  static const size_t calls[][3]={{32,32,32},{64,96,32},{96,64,160},{32,128,64}};
  size_t s,i,cases=0,gemms=0,verdicts[2]={0,0}; unsigned threads,most=bcir_tensor_pool_rails() ? 4 : 1; int v2;
  for (threads=1;threads<=most;threads+=3) {
    bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads) : NULL;
    if (threads>1) assert(pool);
    for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (v2=0;v2<2;v2++) {
      size_t r=shapes[s][0],c=shapes[s][1],n=r*c; int bad,want_bad;
      /* odd shapes with the edges (zeros, subnormals, the odd infinity), even ones finite */
      for (i=0;i<n;i++) { x[i]=bf16_operand((unsigned)i,(int)(s%2)); u[i]=bf16_operand((unsigned)i+7u,(int)(s%2)); dy[i]=bf16_operand((unsigned)i+11u,0); }
      if (s==2) { x[n-1]=NAN; }                  /* a NaN input: a value not finite */
      if (s==4) { x[5]=1e30f; u[5]=1e30f; }       /* finite operands, an infinite value */
      if (s==6) { x[1056+1040]=NAN; }             /* a NaN in a row's second segment */
      want_bad=0;
      if (v2) bcir_tensor_silu_mul_v2_scan_pool(pool,n,x,u,y,&want_bad);
      else bcir_tensor_silu_mul_scan_pool(pool,n,x,u,y,&want_bad);
      if (s==0) assert(!want_bad);
      if (s==2 || s==4 || s==6) assert(want_bad);
      verdicts[want_bad!=0]++;
      ref_panels(r,c,y,ra,rb,0);
      bad=0; memset(pa,0xab,sizeof pa); memset(pb,0xab,sizeof pb);
      assert(bcir_tensor_silu_mul_panels_pool(pool,v2,r,c,x,u,pa,pb,&bad)==1);
      assert(bad==want_bad && same_panel("SiLU product as A",r,c,v2,threads,pa,ra) &&
          same_panel("SiLU product as B",r,c,v2,threads,pb,rb));
      assert(bcir_tensor_silu_mul_panels_pool(pool,v2,r,c,x,u,pa,pb,NULL)==1 &&
          same_panel("SiLU product as A, no verdict",r,c,v2,threads,pa,ra));
      if (v2) bcir_tensor_swiglu_backward_v2_pool(pool,n,x,u,dy,dg,du);
      else bcir_tensor_swiglu_backward_pool(pool,n,x,u,dy,dg,du);
      ref_panels(r,c,dg,ra,rb,1); ref_panels(r,c,du,rc,rd,1);
      memset(pa,0xab,sizeof pa); memset(pb,0xab,sizeof pb); memset(pc,0xab,sizeof pc); memset(pd,0xab,sizeof pd);
      assert(bcir_tensor_swiglu_backward_panels_pool(pool,v2,r,c,x,u,dy,pa,pb,pc,pd)==1);
      assert(same_panel("gate adjoint as A",r,c,v2,threads,pa,ra) &&
          same_panel("gate adjoint as A^T",r,c,v2,threads,pb,rb) &&
          same_panel("up adjoint as A",r,c,v2,threads,pc,rc) &&
          same_panel("up adjoint as A^T",r,c,v2,threads,pd,rd));
      cases++;
    }
    /* values at the conversion's edge: every binary32 value built within an ulp or two of a
     * bfloat16 midpoint, so a writer whose value slips by one ulp converts differently (elsewhere the
     * conversion hides most such slips). The SiLU product is s*u with s the entry's own SiLU, which
     * it returns at u = 1, so u = M/s puts the product next to M; the SwiGLU VJP's two adjoints are
     * dy*(u*...) and (dy*g)*s, read back at dy = u = 1, and dy = M/(that) puts the even elements'
     * dg and the odd elements' du next to theirs */
    for (v2=0;v2<2;v2++) {
      enum { R=64, C=96, NB=R*C };
      int bad=0;
      for (i=0;i<NB;i++) { x[i]=-4.0f+8.0f*(float)((i*37u)%NB)/(float)NB+1e-3f; u[i]=1.0f; dy[i]=1.0f; }
      if (v2) bcir_tensor_silu_mul_v2_scan_pool(pool,NB,x,u,y,&bad);
      else bcir_tensor_silu_mul_scan_pool(pool,NB,x,u,y,&bad);
      for (i=0;i<NB;i++) u[i]=bf16_midpoint(y[i])/y[i];
      if (v2) bcir_tensor_silu_mul_v2_scan_pool(pool,NB,x,u,y,&bad);
      else bcir_tensor_silu_mul_scan_pool(pool,NB,x,u,y,&bad);
      ref_panels(R,C,y,ra,rb,0);
      assert(bcir_tensor_silu_mul_panels_pool(pool,v2,R,C,x,u,pa,pb,NULL)==1 && same_shorts(pa,ra,NB) && same_shorts(pb,rb,NB));
      for (i=0;i<NB;i++) u[i]=1.0f;
      if (v2) bcir_tensor_swiglu_backward_v2_pool(pool,NB,x,u,dy,dg,du);
      else bcir_tensor_swiglu_backward_pool(pool,NB,x,u,dy,dg,du);
      for (i=0;i<NB;i++) dy[i]=i%2 ? bf16_midpoint(du[i])/du[i] : bf16_midpoint(dg[i])/dg[i];
      if (v2) bcir_tensor_swiglu_backward_v2_pool(pool,NB,x,u,dy,dg,du);
      else bcir_tensor_swiglu_backward_pool(pool,NB,x,u,dy,dg,du);
      ref_panels(R,C,dg,ra,rb,1); ref_panels(R,C,du,rc,rd,1);
      assert(bcir_tensor_swiglu_backward_panels_pool(pool,v2,R,C,x,u,dy,pa,pb,pc,pd)==1);
      assert(same_shorts(pa,ra,NB) && same_shorts(pb,rb,NB) && same_shorts(pc,rc,NB) && same_shorts(pd,rd,NB));
      cases++;
    }
#ifdef BCIR_DT_PANELS_PROBE
    {  /* the tensor unit's record: the values each job computed, from its own fields */
      (void)bcir_tensor_panels_probe();
      assert(bcir_tensor_panels_probe()==0u);
      assert(bcir_tensor_silu_mul_panels_pool(pool,0,32,32,x,u,pa,pb,NULL)==1 && bcir_tensor_panels_probe()==1u);
      assert(bcir_tensor_silu_mul_panels_pool(pool,1,32,32,x,u,pa,pb,NULL)==1);
      assert(bcir_tensor_swiglu_backward_panels_pool(pool,0,32,32,x,u,dy,pa,pb,pc,pd)==1 && bcir_tensor_panels_probe()==6u);
      assert(bcir_tensor_swiglu_backward_panels_pool(pool,1,32,32,x,u,dy,pa,pb,pc,pd)==1 && bcir_tensor_panels_probe()==8u);
      assert(bcir_tensor_silu_mul_panels_pool(pool,1,33,32,x,u,pa,pb,NULL)==0 && bcir_tensor_panels_probe()==0u);
    }
#endif
    /* refused: shapes off the 32 grid write nothing; empty writes nothing and succeeds */
    {
      int bad=0; memset(pa,0xab,sizeof pa); memset(ra,0xab,sizeof ra);
      assert(bcir_tensor_silu_mul_panels_pool(pool,1,33,32,x,u,pa,pb,&bad)==0 && !bad && same_shorts(pa,ra,N));
      assert(bcir_tensor_silu_mul_panels_pool(pool,0,32,48,x,u,pa,pb,&bad)==0 && same_shorts(pa,ra,N));
      assert(bcir_tensor_swiglu_backward_panels_pool(pool,1,40,32,x,u,dy,pa,pb,pc,pd)==0 && same_shorts(pa,ra,N));
      assert(bcir_tensor_silu_mul_panels_pool(pool,1,0,32,x,u,pa,pb,&bad)==1 && !bad && same_shorts(pa,ra,N));
    }
    /* the grouped entry reading panels: each call kind the trainer makes -- an input's A (x W^T), an
     * adjoint's A (dy W), an activation's B and an adjoint's A^T (dy^T x) -- and a chain of two
     * adjoints into one dx beside their weight adjoints, against the same calls packed */
    for (s=0;s<sizeof(calls)/sizeof(calls[0]);s++) {
      size_t m=calls[s][0],nn=calls[s][1],k=calls[s][2],kind;
      for (i=0;i<m*k || i<k*nn || i<m*nn;i++) {
        if (i<N) { x[i]=bf16_operand((unsigned)i*3u,1); u[i]=bf16_operand((unsigned)i*5u+1u,0); c0[i]=bf16_operand((unsigned)i+3u,0); }
      }
      for (kind=0;kind<4;kind++) {
        bcir_tensor_mm_call cl; bcir_tensor_bf16_panels pre; size_t q;
        /* kind 1 reads only B as a panel, so its A pack may scale by any alpha */
        memset(&cl,0,sizeof cl); cl.m=m; cl.n=nn; cl.k=k; cl.alpha=kind==1 ? 1.5f : 1.0f;
        cl.beta=kind==3 ? 1.0f : 0.0f; cl.ta=kind==2; cl.tb=kind==0; cl.a=x; cl.b=u; cl.c=want;
        memcpy(want,c0,m*nn*sizeof(float)); memcpy(got,c0,m*nn*sizeof(float));
        assert(bcir_tensor_mm_bf16_group_pool(pool,NULL,0,&cl,1,ws,sizeof ws)==1);
        /* the panels of the call's own operands: A (m x k, or k x m read transposed) and B */
        for (q=0;q<m*k;q++) { size_t i0=q/k,l=q%k; y[i0*k+l]=cl.ta ? x[l*m+i0] : x[i0*k+l]; }
        for (q=0;q<k*nn;q++) { size_t l=q/nn,j=q%nn; dg[l*nn+j]=cl.tb ? u[j*k+l] : u[l*nn+j]; }
        ref_panels(m,k,y,ra,rb,0); ref_panels(k,nn,dg,rc,rd,0);
        pre.a=kind==1 ? NULL : ra; pre.b=kind==0 ? NULL : rd;
        cl.a=pre.a ? NULL : x; cl.b=pre.b ? NULL : u; cl.c=got;
        assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,ws,sizeof ws)==1);
        assert(same_bits_or_nan(want,got,m*nn)); gemms++;
      }
      {  /* the backward's gate/up group: two adjoints chained into one dx, their weight adjoints apart */
        bcir_tensor_mm_call cx[2],cw[2]; bcir_tensor_bf16_panels pre[4]; size_t q;
        static float dx0[96*160],dx1[96*160],w0[96*160],w1[96*160],w2[96*160],w3[96*160];
        for (q=0;q<2;q++) {
          memset(&cx[q],0,sizeof cx[q]); memset(&cw[q],0,sizeof cw[q]);
          cx[q].m=m; cx[q].n=nn; cx[q].k=k; cx[q].alpha=1.0f; cx[q].beta=q ? 1.0f : 0.0f;
          cx[q].a=q ? dy : x; cx[q].b=u; cx[q].c=dx0;
          cw[q].ta=1; cw[q].m=k; cw[q].n=nn; cw[q].k=m; cw[q].alpha=1.0f; cw[q].beta=0.5f;
          cw[q].a=q ? dy : x; cw[q].b=c0; cw[q].c=q ? w1 : w0;
        }
        for (q=0;q<k*nn;q++) { w0[q]=w1[q]=w2[q]=w3[q]=c0[q]; }
        assert(bcir_tensor_mm_bf16_group_pool(pool,cx,2,cw,2,ws,sizeof ws)==1);
        ref_panels(m,k,x,ra,rb,1); ref_panels(m,k,dy,rc,rd,1);
        for (q=0;q<2;q++) { cx[q].a=NULL; cx[q].c=dx1; cw[q].a=NULL; cw[q].c=q ? w3 : w2; }
        pre[0].a=ra; pre[0].b=NULL; pre[1].a=rc; pre[1].b=NULL;
        pre[2].a=rb; pre[2].b=NULL; pre[3].a=rd; pre[3].b=NULL;
        assert(bcir_tensor_mm_bf16_group_panels_pool(pool,cx,2,cw,2,pre,ws,sizeof ws)==1);
        assert(same_bits_or_nan(dx0,dx1,m*nn) && same_bits_or_nan(w0,w2,k*nn) && same_bits_or_nan(w1,w3,k*nn));
        gemms++;
      }
    }
    {  /* a panel is its own call's: a later call reading the same operands packs them, and the
        * workspace the group needs counts those packs (a byte short of it is refused) */
      bcir_tensor_mm_call two[2]; bcir_tensor_bf16_panels pre[2]; size_t q,need;
      memset(two,0,sizeof two);
      for (q=0;q<2;q++) { two[q].m=64; two[q].n=32; two[q].k=32; two[q].alpha=1.0f; two[q].a=x; two[q].b=u; }
      two[0].c=dg;
      assert(bcir_tensor_mm_bf16_group_pool(pool,NULL,0,two,1,ws,sizeof ws)==1);  /* the call packed */
      two[0].c=got; two[1].c=y;
      pre[0].a=rc; pre[0].b=rd; pre[1].a=NULL; pre[1].b=NULL;  /* call 0's panels: another product's */
      need=bcir_tensor_mm_bf16_group_workspace(NULL,0,two,2);
      assert(need>1 && bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,two,2,pre,ws,need-1)==0);
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,two,2,pre,ws,need)==1);
      assert(same_bits_or_nan(y,dg,64*32));
      gemms++;
    }
    {  /* refused, nothing written: a panel's call off the 32 grid (its rows, its columns, its
        * depth), an A panel's at alpha 2, a workspace short of the operand it packs, and no
        * workspace for it */
      bcir_tensor_mm_call cl; bcir_tensor_bf16_panels pre;
      memset(&cl,0,sizeof cl); cl.m=33; cl.n=32; cl.k=32; cl.alpha=1.0f; cl.b=u; cl.c=got;
      for (i=0;i<64*64;i++) got[i]=c0[i];
      pre.a=ra; pre.b=NULL;
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,ws,sizeof ws)==0);
      cl.m=32; cl.n=48;
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,ws,sizeof ws)==0);
      cl.n=32; cl.k=40;
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,ws,sizeof ws)==0);
      cl.m=64; cl.n=64; cl.k=32; cl.alpha=2.0f;  /* an A panel is A alone: alpha 1 only */
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,ws,sizeof ws)==0);
      cl.alpha=1.0f;
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,ws,64)==0);
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,NULL,0)==0);
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,NULL,sizeof ws)==0);
      assert(same_bits_or_nan(got,c0,64*64));
      pre.b=rd; cl.b=NULL;  /* both operands panels: no workspace at all */
      assert(bcir_tensor_mm_bf16_group_panels_pool(pool,NULL,0,&cl,1,&pre,NULL,0)==1);
    }
    bcir_tensor_pool_stop(pool);
  }
  assert(verdicts[0]>0 && verdicts[1]>0);  /* both verdicts given, each the scan entry's */
  printf("  PASS bf16 panels (BF16-PANELS): %zu writer cases (SiLU*u as A and B, the SwiGLU VJP as A and A^T, both SiLUs) equal to the binary32 entries' values converted in place, verdicts, refusals; %zu grouped GEMMs reading panels equal to the packed calls, refusals\n",
      cases,gemms);
}
/* BF16-PANELS: a decoder whose shapes are multiples of 32 under each bf16 contract ends in one
 * state whether the SwiGLU block's tensors are panels (64-byte aligned slots), binary32 (the
 * activation arena a float off that alignment: the hidden activation binary32, the adjoints
 * panels) or the other way (the scratch arena off: the hidden activation a panel, the adjoints
 * binary32), at one thread and at four. A second model's width (48) is off the 32 grid while its
 * rows and hidden width are on it: every GEMM a panel would feed has the width as a dimension, so
 * it writes no panel at all. */
static void trainer_panels(void) {
  enum { P=40000, A=48000, S=24000 };
  static _Alignas(64) float pw[P],pg[P],pm[P],pv[P],pa[A+16],ps[S+16],ref[3][P];
  static _Alignas(64) unsigned char pws[1<<18];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const uint32_t contracts[]={BCIR_DT_CONTRACT_BF16,BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTN};
  static const bcir_decoder_spec specs[2]={{16,32,2,1,2,64,2,16,1,10000.0,1e-6},
      {16,48,2,1,2,64,2,16,1,10000.0,1e-6}};
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  uint32_t ids[96],targets[96]; bcir_decoder_event events[3]; size_t done,i,cs,sp; unsigned layout,threads;
  stream(ids,targets,96);
  for (sp=0;sp<2;sp++)
  for (cs=0;cs<sizeof(contracts)/sizeof(contracts[0]);cs++)
  for (threads=1;threads<=(bcir_tensor_pool_rails() ? 4u : 1u);threads+=3)
  for (layout=0;layout<3;layout++) {
    bcir_decoder_state st; bcir_tensor_pool *pool=NULL; bcir_decoder_spec spec=specs[sp];
    memset(&st,0,sizeof st);
    assert(bcir_decoder_make_plan(&spec,&st.plan)==0);
    assert(st.plan.parameters<=P && st.plan.activations<=A && st.plan.scratch<=S);
    st.weights=pw; st.gradients=pg; st.moment1=pm; st.moment2=pv;
    st.activations=pa+(layout==1); st.scratch=ps+(layout==2);  /* one float off 64-byte alignment */
    st.weight_capacity=st.gradient_capacity=st.moment1_capacity=st.moment2_capacity=P;
    st.activation_capacity=A; st.scratch_capacity=S; st.beta1_power=st.beta2_power=1.0;
    memset(pm,0,sizeof pm); memset(pv,0,sizeof pv); memset(pg,0,sizeof pg);
    for (i=0;i<P;i++) pw[i]=0.1f*sinf((float)(i+1)*0.37f);
    if (threads>1) { pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool); }
    st.pool=pool; st.contract=contracts[cs];
    st.workspace=pws; st.workspace_bytes=bcir_decoder_workspace_bytes(&st.plan,st.contract);
    assert(st.workspace_bytes>0 && st.workspace_bytes<=sizeof pws);
#ifdef BCIR_DT_PANELS_PROBE
    (void)bcir_decoder_panels_probe(); (void)bcir_tensor_panels_probe();
#endif
    assert(bcir_decoder_run(&st,&opt,ids,targets,96,NULL,3,events,3,&done)==0 && done==3);
#if defined(BCIR_DT_PANELS_PROBE) && defined(BCIR_DT_NO_PANELS)
    assert(bcir_decoder_panels_probe()==0u && bcir_tensor_panels_probe()==0u);
#elif defined(BCIR_DT_PANELS_PROBE)
    {  /* the trainer's paths, and the values the tensor unit's writers computed on them: v1's
        * SiLU and VJP without the exponential bit, v2's with it */
      unsigned v2=(contracts[cs]&BCIR_DT_CONTRACT_EXP)!=0,silu=v2 ? 2u : 1u,vjp=v2 ? 8u : 4u;
      unsigned paths=sp ? 0u : layout==0 ? 7u : layout==1 ? 4u : 3u;
      assert(bcir_decoder_panels_probe()==paths);
      assert(bcir_tensor_panels_probe()==((paths&1u ? silu : 0u)|(paths&4u ? vjp : 0u)));
    }
#endif
    if (threads==1 && layout==0) { memcpy(ref[0],pw,sizeof pw); memcpy(ref[1],pm,sizeof pm); memcpy(ref[2],pv,sizeof pv); }
    else assert(!memcmp(ref[0],pw,sizeof pw) && !memcmp(ref[1],pm,sizeof pm) && !memcmp(ref[2],pv,sizeof pv));
    bcir_tensor_pool_stop(pool);
  }
  {
#ifdef BCIR_DT_PANELS_PROBE
    static const char probed[]=", each run's panel paths the probe's";
#else
    static const char probed[]="";
#endif
    printf("  PASS bf16 panels trainer (BF16-PANELS): three bf16 contracts, one state with the SwiGLU block's panels, without them and mixed, at one thread and four; none where the width is off the grid%s\n",probed);
  }
}
#ifdef BCIR_DT_SCAN_PROBE
unsigned long bcir_decoder_scan_probe(void);
#endif
/* FACTS-RUN: inside one run call the weights after a step's update are that commit's, every one
 * finite, so the steps after the first skip the weights' opening scan. The state and the events
 * are those of the same steps run one call at a time -- serially and on a pool, micro-batches of
 * one and two, under four contracts -- and the scan probe (BCIR_DT_SCAN_PROBE) counts one scan for
 * the run against one a call for the calls. */
static void facts_run(void) {
  static float keepw[4096],keepm[4096],keepv[4096];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const uint32_t contracts[]={0,BCIR_DT_CONTRACT_FUSED,
      BCIR_DT_CONTRACT_KBLOCK|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTN};
  uint32_t ids[24],targets[24]; bcir_decoder_event one[3],each[3];
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,1.0};
  size_t c,a,k,p,done,n=4,cases=0;
  stream(ids,targets,24);
  for (p=0;p<2;p++) {
    bcir_tensor_pool *pool=NULL;
    if (p) {
      if (!bcir_tensor_pool_size(3)) continue;
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),3); assert(pool!=NULL);
    }
    for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) for (a=1;a<=2;a++) {
      bcir_decoder_state st=fixture(); size_t params=st.plan.parameters;
      st.contract=contracts[c]; st.pool=pool; lend(&st);
#ifdef BCIR_DT_SCAN_PROBE
      (void)bcir_decoder_scan_probe();
#endif
      assert(bcir_decoder_run_accum(&st,&opt,ids,targets,3*a*n,NULL,3,a,one,3,&done)==0 && done==3);
#ifdef BCIR_DT_SCAN_PROBE
      assert(bcir_decoder_scan_probe()==1u);
#endif
      memcpy(keepw,w,params*sizeof(float)); memcpy(keepm,m,params*sizeof(float));
      memcpy(keepv,v,params*sizeof(float));
      st=fixture(); st.contract=contracts[c]; st.pool=pool; lend(&st);
      for (k=0;k<3;k++)
        assert(bcir_decoder_run_accum(&st,&opt,ids+k*a*n,targets+k*a*n,a*n,NULL,1,a,each+k,1,&done)==0 &&
            done==1);
#ifdef BCIR_DT_SCAN_PROBE
      assert(bcir_decoder_scan_probe()==3u);
#endif
      assert(!memcmp(keepw,w,params*sizeof(float)) && !memcmp(keepm,m,params*sizeof(float)) &&
          !memcmp(keepv,v,params*sizeof(float)));
      for (k=0;k<3;k++)
        assert(!memcmp(&one[k].loss,&each[k].loss,sizeof(double)) &&
            !memcmp(&one[k].grad_norm,&each[k].grad_norm,sizeof(double)) && one[k].step==each[k].step);
      cases++;
    }
    bcir_tensor_pool_stop(pool);
  }
  assert(cases>=8);
  {
#ifdef BCIR_DT_SCAN_PROBE
    static const char probed[]=", one weight scan a run and one a call";
#else
    static const char probed[]="";
#endif
    printf("  PASS native run facts (FACTS-RUN): a run of three steps is three calls of one, state and events, under four contracts and two accumulations, serially and pooled%s\n",probed);
  }
}
#ifdef BCIR_DT_WORKSPACE_PROBE
void bcir_decoder_workspace_probe_admit(size_t bytes);
#endif
/* A state over the large arenas for a spec (the panels shape and the fixture's), weights the
 * harness's, moments zero, at a pool, a contract, a workspace and a recompute set. */
enum { WP_P=40000, WP_A=48000, WP_S=24000 };
static _Alignas(64) float wp_w[WP_P],wp_g[WP_P],wp_m[WP_P],wp_v[WP_P],wp_a[WP_A],wp_s[WP_S];
static _Alignas(64) unsigned char wp_ws[1<<18];
static bcir_decoder_state wide_state(const bcir_decoder_spec *spec,bcir_tensor_pool *pool,
    uint32_t contract,size_t workspace,uint64_t recompute) {
  bcir_decoder_state st; size_t i;
  memset(&st,0,sizeof st);
  assert(bcir_decoder_make_plan(spec,&st.plan)==0);
  assert(st.plan.parameters<=WP_P && st.plan.activations<=WP_A && st.plan.scratch<=WP_S);
  st.weights=wp_w; st.gradients=wp_g; st.moment1=wp_m; st.moment2=wp_v;
  st.activations=wp_a; st.scratch=wp_s;
  st.weight_capacity=st.gradient_capacity=st.moment1_capacity=st.moment2_capacity=WP_P;
  st.activation_capacity=WP_A; st.scratch_capacity=WP_S; st.beta1_power=st.beta2_power=1.0;
  memset(wp_m,0,sizeof wp_m); memset(wp_v,0,sizeof wp_v); memset(wp_g,0,sizeof wp_g);
  for (i=0;i<WP_P;i++) wp_w[i]=0.1f*sinf((float)(i+1)*0.37f);
  st.pool=pool; st.contract=contract; st.recompute=recompute;
  assert(workspace<=sizeof wp_ws);
  st.workspace=contract&BCIR_DT_CONTRACT_BF16 ? wp_ws : NULL; st.workspace_bytes=workspace;
  return st;
}
/* WS-PLAN: the bf16 workspace from its floor up. The plan's need is what the step packs -- the
 * probe's most, a group's panels counted as its entry counts them -- on the panels shape (32 rows,
 * width 32, FF 128) and on the fixture's wide shape (its width off the grid: no panels). Every
 * workspace from the floor up trains one state, at one thread and four, the SwiGLU block with its
 * panels exactly from the panel groups' need up; below the floor admission refuses, nothing
 * written; and where the harness's floor admits less (probe builds), the product that refuses --
 * in the forward, or in the backward alone where the whole forward runs -- fails its step with
 * BCIR_DT_CAPACITY, the weights as they were. */
static void trainer_workspace_plan(void) {
  static float ref[3][WP_P],start[WP_P];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const uint32_t contracts[]={
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTNBF16};
  /* the panels shape's hidden width four times its width, so its SwiGLU panel groups are the step's
   * largest and a count that packed their panels would be seen; the wide shape's rows the deepest
   * depth, so a backward product can refuse where the whole forward runs */
  static const bcir_decoder_spec specs[2]={{16,32,2,1,2,128,2,16,1,10000.0,1e-6},
      {16,8,2,1,2,16,5,8,1,10000.0,1e-6}};
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  uint32_t ids[120],targets[120]; bcir_decoder_event events[3]; size_t done,sp,cs,k,sizes=0;
  unsigned threads;
  stream(ids,targets,120);
  for (sp=0;sp<2;sp++)
  for (cs=0;cs<sizeof(contracts)/sizeof(contracts[0]);cs++)
  for (threads=1;threads<=(bcir_tensor_pool_rails() ? 4u : 1u);threads+=3) {
    bcir_tensor_pool *pool=NULL; bcir_decoder_plan plan; size_t need,gate,least,size[9],n,q;
    uint32_t c=contracts[cs];
    assert(bcir_decoder_make_plan(&specs[sp],&plan)==0);
    n=(size_t)specs[sp].batch*specs[sp].time;
    if (threads>1) { pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool); }
    need=bcir_decoder_workspace_need(&plan,c,threads); gate=bcir_decoder_workspace_panels(&plan,c);
    least=bcir_decoder_workspace_floor(&plan,c,threads);
#ifdef BCIR_DT_NO_PANELS
    int paneled=0;  /* a build without panels plans the block packed */
#else
    int paneled=!sp;
#endif
    assert(least>0 && least<=need && need<=sizeof wp_ws && (paneled ? gate>0 && gate<=need : gate==0));
    assert(bcir_decoder_workspace_need(&plan,c,0)>=need);  /* any pool: a region a task */
    /* the need and a byte short, the panel groups' need and a byte short (where the floor is below
     * it: the bf16 attention's regions can be above), between, the floor; then a byte short of it */
    size[0]=need; size[1]=need-1; size[2]=gate; size[3]=gate-1; size[4]=(gate+least)/2;
    size[5]=(need+least)/2; size[6]=least+1; size[7]=least; size[8]=least-1;
    for (k=0;k<9;k++) {
      bcir_decoder_state st;
      int rc;
      if (k<8 && (size[k]<least || size[k]>need)) continue;  /* a panel need of 0 less one wraps */
      for (q=1;q<k && k<8;q++) if (size[q]==size[k]) break;
      if (k<8 && q<k) continue;  /* the same size as an earlier one */
      st=wide_state(&specs[sp],pool,c,size[k],0);
      if (k==8) memcpy(start,wp_w,sizeof start);
      done=99;
#ifdef BCIR_DT_PANELS_PROBE
      (void)bcir_decoder_panels_probe();
#endif
#ifdef BCIR_DT_WORKSPACE_PROBE
      (void)bcir_decoder_workspace_probe();
#endif
      rc=bcir_decoder_run(&st,&opt,ids,targets,3*n,NULL,3,events,3,&done);
      if (k==8) {  /* below the floor: refused before anything is written, the count included */
        assert(rc==BCIR_DT_CAPACITY && done==99 && !memcmp(start,wp_w,sizeof start));
        continue;
      }
      assert(rc==0 && done==3); sizes++;
#ifdef BCIR_DT_PANELS_PROBE
      assert(bcir_decoder_panels_probe()==(gate && size[k]>=gate ? 7u : 0u));
#endif
#ifdef BCIR_DT_WORKSPACE_PROBE
      {  /* the step's GEMM groups pack what the plan says, no more; without panels, more -- the plan's
          * figure the most of theirs and the binary32 attention's regions (ATT-CONTIG), which the
          * shapes reach where a build lowers the copies' threshold */
        size_t seen=bcir_decoder_workspace_probe(),att=att32_need(&plan,threads),
            gemms=bcir_decoder_workspace_need(&plan,c&~(uint32_t)BCIR_DT_CONTRACT_ATTNBF16,threads);
        if (!k && (!gate || need>=gate) && !(c&BCIR_DT_CONTRACT_ATTNBF16)) assert(need==most_of(seen,att));
        if (att) {
          if (!gate || size[k]>=gate) assert(seen==gemms || (gemms==att && seen<att));
        }
        else if (!gate || size[k]>=gate) assert(seen==gemms);
        else assert(seen>gemms);
      }
#endif
      if (!k) { memcpy(ref[0],wp_w,sizeof wp_w); memcpy(ref[1],wp_m,sizeof wp_m); memcpy(ref[2],wp_v,sizeof wp_v); }
      else assert(!memcmp(ref[0],wp_w,sizeof wp_w) && !memcmp(ref[1],wp_m,sizeof wp_m) && !memcmp(ref[2],wp_v,sizeof wp_v));
    }
#ifdef BCIR_DT_WORKSPACE_PROBE
    {  /* the harness's floor admits half the plan's: the deepest product refuses, the step fails */
      bcir_decoder_state st=wide_state(&specs[sp],pool,c,least/2,0);
      memcpy(start,wp_w,sizeof start);
      bcir_decoder_workspace_probe_admit(least/2);
      assert(bcir_decoder_run(&st,&opt,ids,targets,3*n,NULL,3,events,3,&done)==BCIR_DT_CAPACITY &&
          done==0 && !memcmp(start,wp_w,sizeof start));
      assert(bcir_decoder_forward(&st,ids,n)==BCIR_DT_CAPACITY);
      bcir_decoder_workspace_probe_admit(SIZE_MAX);
    }
    if (sp==1 && !(c&BCIR_DT_CONTRACT_ATTNBF16)) {
      /* the forward's deepest strips and no more (its depths the width and the hidden width): the
       * forward runs whole, the backward's weight adjoints (the rows deep) refuse, and the step fails
       * with CAPACITY, nothing committed -- the backward's own verdict, and a group's, which the
       * forward's refusal above reaches first */
      size_t fwd=bcir_tensor_mm_bf16_workspace(32,32,specs[sp].ff>specs[sp].width ? specs[sp].ff :
          specs[sp].width);
      bcir_decoder_state st=wide_state(&specs[sp],pool,c,fwd,0);
      assert(fwd<least);
      memcpy(start,wp_w,sizeof start);
      bcir_decoder_workspace_probe_admit(fwd);
      assert(bcir_decoder_forward(&st,ids,n)==BCIR_DT_OK);
      done=99;
      assert(bcir_decoder_run(&st,&opt,ids,targets,3*n,NULL,3,events,3,&done)==BCIR_DT_CAPACITY &&
          done==0 && !memcmp(start,wp_w,sizeof start));
      bcir_decoder_workspace_probe_admit(SIZE_MAX);
    }
#endif
    bcir_tensor_pool_stop(pool);
  }
  printf("  PASS bf16 workspace plan (WS-PLAN): one state at %zu workspaces from the floor up, the panels from "
      "their groups' need, below the floor refused%s\n",sizes,
#ifdef BCIR_DT_WORKSPACE_PROBE
      "; the plan's need the probe's, a refused product the step's CAPACITY"
#else
      ""
#endif
      );
}
/* REMAT: every recompute set trains the state the all-kept one does, bit for bit, in the arena
 * bcir_decoder_activation_count names: the fixture's four layers under v1, the fused and blocked
 * contracts with the fused attention, bf16 with the fused attention and with the bf16 attention,
 * and the panels shape's two layers under bf16, at one thread and four, a run of single steps and
 * one of two accumulated micro-batches; the floats past the count untouched; the eval forward's
 * logits the arena's last; a bit past the last layer refused, an arena one float short refused.
 * Every set at one thread under v1 and under the bf16 attention (the slots' arithmetic is the
 * contract's alone); the rest, four threads and two micro-batches on three sets -- none, every
 * layer, every other -- so the portable build's bf16 stays within its budget. */
static void trainer_recompute(void) {
  static float ref[3][WP_P],logits[4096];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const uint32_t contracts[]={0,
      BCIR_DT_CONTRACT_FUSED|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_KBLOCK|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE|BCIR_DT_CONTRACT_ATTNBF16};
  static const bcir_decoder_spec specs[2]={{16,8,2,1,4,16,5,8,1,10000.0,1e-6},
      {16,32,2,1,2,64,2,16,1,10000.0,1e-6}};
  static const uint64_t sets[]={0,0xF,0x5,0xA,0x1,0x8,0x7,0xE,0x6};
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,0.05};
  uint32_t ids[240],targets[240]; bcir_decoder_event events[3]; size_t done,sp,cs,k,runs=0;
  unsigned threads,accum;
  stream(ids,targets,240);
  for (sp=0;sp<2;sp++)
  for (cs=sp ? 3 : 0;cs<sizeof(contracts)/sizeof(contracts[0]);cs++)
  for (threads=1;threads<=(bcir_tensor_pool_rails() ? 4u : 1u);threads+=3)
  for (accum=1;accum<=2;accum++) {
    int every=(cs==0 || cs==4) && threads==1 && accum==1;  /* every set, or none / all / every other */
    bcir_tensor_pool *pool=NULL; bcir_decoder_plan plan; const bcir_decoder_spec *spec=&specs[sp];
    size_t n=(size_t)spec->batch*spec->time,d=spec->width,hd=d/spec->heads,nd=n*d,nk=n*spec->kvheads*hd;
    size_t la=5*nd+2*nk+(size_t)spec->batch*spec->heads*spec->time*spec->time+3*n*spec->ff,nv=n*spec->vocab;
    uint32_t c=contracts[cs];
    if (accum==2 && cs!=0 && cs!=4) continue;  /* before the pool starts: nothing to stop */
    if (DT_TSAN && cs!=0 && cs!=4) continue;
    assert(bcir_decoder_make_plan(spec,&plan)==0 && nv<=sizeof(logits)/sizeof(logits[0]));
    if (threads>1) { pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool); }
    for (k=0;k<sizeof(sets)/sizeof(sets[0]);k++) {
      uint64_t set=sets[k]&(((uint64_t)1<<spec->layers)-1u);
      size_t kept=0,gone,count,i,l,j;
      bcir_decoder_state st;
      if (!every && k>2) break;
      for (j=0;j<k;j++) if ((sets[j]&(((uint64_t)1<<spec->layers)-1u))==set) break;
      if (j<k) continue;  /* a set an earlier one named on these layers */
      for (l=0;l<spec->layers;l++) kept+=!((set>>l)&1u);
      gone=spec->layers-kept;
      count=plan.activations-(gone ? gone-1 : 0)*la;
      assert(bcir_decoder_activation_count(&plan,set)==count);
      st=wide_state(spec,pool,c,bcir_decoder_workspace_need(&plan,c,threads),set);
      for (i=count;i<WP_A;i++) wp_a[i]=-7.25f;  /* past the arena: must stay */
      st.activation_capacity=count;
      if (accum==1) assert(bcir_decoder_run(&st,&opt,ids,targets,3*n,NULL,3,events,3,&done)==0 && done==3);
      else assert(bcir_decoder_run_accum(&st,&opt,ids,targets,6*n,NULL,3,2,events,3,&done)==0 && done==3);
      for (i=count;i<WP_A;i++) assert(wp_a[i]==-7.25f);
      if (!set) { memcpy(ref[0],wp_w,sizeof wp_w); memcpy(ref[1],wp_m,sizeof wp_m); memcpy(ref[2],wp_v,sizeof wp_v); }
      else assert(!memcmp(ref[0],wp_w,sizeof wp_w) && !memcmp(ref[1],wp_m,sizeof wp_m) && !memcmp(ref[2],wp_v,sizeof wp_v));
      /* the eval forward: the logits are the arena's last nv floats, the all-kept forward's */
      assert(bcir_decoder_forward(&st,ids,n)==0);
      if (!set) memcpy(logits,wp_a+count-nv,nv*sizeof(float));
      else assert(!memcmp(logits,wp_a+count-nv,nv*sizeof(float)));
      runs++;
      /* refusals: an arena one float short; a bit past the last layer */
      st.activation_capacity=count-1; assert(bcir_decoder_forward(&st,ids,n)==BCIR_DT_CAPACITY);
      st.activation_capacity=count; st.recompute=set|((uint64_t)1<<spec->layers);
      assert(bcir_decoder_forward(&st,ids,n)==BCIR_DT_INVALID);
      assert(bcir_decoder_activation_count(&plan,st.recompute)==SIZE_MAX);
    }
    bcir_tensor_pool_stop(pool);
  }
  assert(bcir_decoder_activation_count(NULL,0)==SIZE_MAX);
  printf("  PASS recompute (REMAT): %zu runs under up to nine recompute sets, five contracts, one thread and "
      "four, one micro-batch and two, each the all-kept state in the arena its count names, the logits last\n",runs);
}
/* PX-3b (DISPATCH): the pool's cap. A job runs on the caller and at most cap-1 workers, every task
 * once, and on the caller alone, in ascending order, at a cap of one; the last cap set is the one
 * in force, and 0 or a count at or past the pool's is all its threads. A task notes the address of a
 * local of its own frame: one thread, one address (its call chain is the same every task), so the
 * distinct addresses are the threads that ran tasks -- no atomics and no OS call. */
static uintptr_t cap_frame[512]; static unsigned cap_runs[512]; static size_t cap_seq[512],cap_at;
static void cap_task(void *ctx, size_t task) {
  volatile unsigned char here=0; volatile unsigned long spin;
  cap_frame[task]=(uintptr_t)&here; cap_runs[task]++;
  if (ctx) cap_seq[cap_at++]=task;  /* the serial run's order (one thread: no race) */
  for (spin=0;spin<2000ul;spin++) {}  /* work enough that every worker running takes tasks */
}
static size_t cap_threads(void) {
  size_t i,j,count=0;
  for (i=0;i<512;i++) {
    for (j=0;j<i && cap_frame[j]!=cap_frame[i];j++) {}
    count+=j==i;
  }
  return count;
}
static void pool_caps(void) {
  static _Alignas(max_align_t) unsigned char memory[4096];
  bcir_tensor_pool *pool; unsigned threads=4,k; size_t i,r;
  assert(bcir_tensor_pool_cap(NULL,3)==0 && bcir_tensor_pool_workers(NULL)==1);
  if (!bcir_tensor_pool_size(threads)) {
    printf("  PASS pool cap (PX-3b): a serial-only host, the cap of no pool 0\n");
    return;
  }
  pool=bcir_tensor_pool_init(memory,sizeof(memory),threads); assert(pool!=NULL);
  assert(bcir_tensor_pool_workers(pool)==4);
  assert(bcir_tensor_pool_cap(pool,2)==0 && bcir_tensor_pool_workers(pool)==2);
  assert(bcir_tensor_pool_cap(pool,9)==2 && bcir_tensor_pool_workers(pool)==4);
  assert(bcir_tensor_pool_cap(pool,4)==0 && bcir_tensor_pool_workers(pool)==4);
  assert(bcir_tensor_pool_cap(pool,1)==0 && bcir_tensor_pool_workers(pool)==1);
  assert(bcir_tensor_pool_cap(pool,0)==1 && bcir_tensor_pool_workers(pool)==4);
  for (k=1;k<=threads;k++) {
    bcir_tensor_pool_cap(pool,k);
    for (r=0;r<20;r++) {
      size_t seen;
      memset(cap_frame,0,sizeof(cap_frame)); memset(cap_runs,0,sizeof(cap_runs)); cap_at=0;
      bcir_tensor_pool_run(pool,cap_task,k==1 ? (void *)cap_seq : NULL,512);
      for (i=0;i<512;i++) assert(cap_runs[i]==1 && cap_frame[i]);
      seen=cap_threads(); assert(seen>=1 && seen<=k);
      if (k==1) { assert(cap_at==512); for (i=0;i<512;i++) assert(cap_seq[i]==i); }
    }
  }
  bcir_tensor_pool_cap(pool,0); bcir_tensor_pool_stop(pool);
  /* how many threads claimed tasks is the scheduler's, never printed: a verdict's text is the same
   * on every run (the BCIR Make gate holds a section's output to its shell run's, byte for byte) */
  printf("  PASS pool cap (PX-3b): 20 jobs of 512 tasks at each cap of a four-thread pool, every task "
      "once, on at most cap threads, one in order on the caller\n");
}
/* PX-3b: the GEMM and attention entries with the instance named. Every instance a family admits is
 * the plain entry's bits, at caps one, two and all of a four-thread pool, each returns the one it
 * ran, and the kernels that ran were that instance's -- the tensor unit's probe, which this harness's
 * build defines, sees exactly it (the widest for 0) and nothing for a refusal; anything else -- a bit
 * the family does not admit, two bits, a contract outside 0..3, panels outside the bf16 contract --
 * returns 0 with nothing written. */
#ifdef BCIR_TENSOR_INSTANCE_PROBE
unsigned bcir_tensor_instance_probe(int family);
#define DISPATCH_CLEAR(f) ((void)bcir_tensor_instance_probe(f))
#define DISPATCH_RAN(f,want) assert(bcir_tensor_instance_probe(f)==(unsigned)(want))
#else
#define DISPATCH_CLEAR(f) ((void)(f))
#define DISPATCH_RAN(f,want) ((void)(f),(void)(want))
#endif
static int dispatch_same(const float *x, const float *y, size_t n) { return !memcmp(x,y,n*sizeof(float)); }
static int dispatch_widest(int mask) { return mask&8 ? 8 : mask&4 ? 4 : mask&2 ? 2 : 1; }
static void dispatch_entries(void) {
  static float a[67*130],b[130*45],c[67*45],want[67*45],got[67*45],b2[130*45],c2[67*45],o[67*45];
  static float q[2*33*4*16],kk[2*33*2*16],vv[2*33*2*16],dy[2*33*4*16],p0[2*4*33*33],p1[2*4*33*33];
  static float y0[2*33*4*16],y1[2*33*4*16],dq0[2*33*4*16],dq1[2*33*4*16],dk0[2*33*2*16],dk1[2*33*2*16],
      dv0[2*33*2*16],dv1[2*33*2*16];
  static _Alignas(64) unsigned char ws[1<<20];
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  static const unsigned att_contracts[]={0,BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED,BCIR_TENSOR_ATTENTION_EXP};
  static const int bad[]={3,6,12,16,-1,-4};
  const size_t m=67,n=45,k=130; size_t i,cases=0,refused=0,ai,bi; int contract,v;
  unsigned cap,threads=bcir_tensor_pool_rails() ? 4u : 1u; bcir_tensor_pool *pool=NULL;
  if (threads>1 && bcir_tensor_pool_size(threads)) pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads);
  if (!pool) threads=1;
  for (i=0;i<m*k;i++) a[i]=sinf((float)i*0.71f)*(i%3 ? 1.5f : 0.125f);
  for (i=0;i<k*n;i++) { b[i]=cosf((float)i*1.37f)-0.25f; b2[i]=sinf((float)i*0.11f); }
  for (i=0;i<m*n;i++) c[i]=sinf((float)i*0.3f);
  for (contract=0;contract<4;contract++) {
    int mask=bcir_tensor_mm_contract_variants(contract),f=contract==3 ? 1 : 0;
    bcir_tensor_mm_call x,y,chain[2];
    assert(mask&1);
    x.ta=0; x.tb=0; x.m=m; x.n=n; x.k=k; x.alpha=1.0f; x.a=a; x.b=b; x.beta=0.5f; x.c=NULL;
    y=x; y.b=b2; y.beta=0.0f;
    for (cap=1;cap<=threads;cap=cap==1 ? 2u : cap*2u) {
      bcir_tensor_pool_cap(pool,cap);
      /* one call: the plain entry's (variant 0) result, the widest instance, then every instance */
      memcpy(want,c,sizeof(c)); DISPATCH_CLEAR(f);
      assert(bcir_tensor_mm_contract_variant_pool(pool,contract,0,0,0,m,n,k,1.0f,a,b,0.5f,want,ws,
          sizeof(ws))==dispatch_widest(mask));
      DISPATCH_RAN(f,dispatch_widest(mask));
      for (v=1;v<=8;v*=2) {
        memcpy(got,c,sizeof(c)); DISPATCH_CLEAR(f);
        if (!(mask&v)) {
          assert(bcir_tensor_mm_contract_variant_pool(pool,contract,v,0,0,m,n,k,1.0f,a,b,0.5f,got,ws,sizeof(ws))==0);
          DISPATCH_RAN(f,0); assert(dispatch_same(got,c,m*n)); refused++; continue;
        }
        assert(bcir_tensor_mm_contract_variant_pool(pool,contract,v,0,0,m,n,k,1.0f,a,b,0.5f,got,ws,sizeof(ws))==v);
        DISPATCH_RAN(f,v); assert(dispatch_same(got,want,m*n)); cases++;
        /* a chain of two into one C as one job, against the default instance's */
        memcpy(got,c,sizeof(c)); memcpy(c2,c,sizeof(c));
        chain[0]=x; chain[0].c=c2; chain[1]=y; chain[1].beta=1.0f; chain[1].c=c2;
        assert(bcir_tensor_mm_group_variant_pool(pool,contract,0,chain,2,NULL,0,NULL,ws,sizeof(ws)));
        chain[0].c=got; chain[1].c=got; DISPATCH_CLEAR(f);
        assert(bcir_tensor_mm_group_variant_pool(pool,contract,v,chain,2,NULL,0,NULL,ws,sizeof(ws))==v);
        DISPATCH_RAN(f,v); assert(dispatch_same(got,c2,m*n)); cases++;
        if (contract<3) {  /* the pair (NDT-SCHED-0): its first call is the one above */
          bcir_tensor_mm_call px=x,py=y;
          memcpy(got,c,sizeof(c)); memset(o,0,sizeof(o)); px.c=got; py.c=o; DISPATCH_CLEAR(f);
          assert(bcir_tensor_mm_pair_variant_pool(pool,contract,v,&px,&py)==v);
          DISPATCH_RAN(f,v); assert(dispatch_same(got,want,m*n)); cases++;
        }
      }
      for (i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        memcpy(got,c,sizeof(c)); DISPATCH_CLEAR(f);
        assert(bcir_tensor_mm_contract_variant_pool(pool,contract,bad[i],0,0,m,n,k,1.0f,a,b,0.5f,got,ws,sizeof(ws))==0);
        DISPATCH_RAN(f,0); assert(dispatch_same(got,c,m*n)); refused++;
      }
    }
    if (contract<3) {  /* panels belong to the bf16 contract */
      bcir_tensor_bf16_panels pre[1]={{NULL,NULL}}; bcir_tensor_mm_call one=x;
      memcpy(got,c,sizeof(c)); one.c=got;
      assert(bcir_tensor_mm_group_variant_pool(pool,contract,0,NULL,0,&one,1,pre,ws,sizeof(ws))==0);
      assert(dispatch_same(got,c,m*n)); refused++;
    }
  }
  for (contract=-1;contract<=4;contract+=5) {
    memcpy(got,c,sizeof(c));
    assert(bcir_tensor_mm_contract_variant_pool(pool,contract,0,0,0,m,n,k,1.0f,a,b,0.5f,got,ws,sizeof(ws))==0);
    assert(bcir_tensor_mm_contract_variants(contract)==0 && dispatch_same(got,c,m*n)); refused++;
  }
  /* the binary32 attention: its products' family, every instance the default's bits */
  for (i=0;i<sizeof(q)/sizeof(q[0]);i++) { q[i]=sinf((float)i*0.37f); dy[i]=cosf((float)i*0.23f); }
  for (i=0;i<sizeof(kk)/sizeof(kk[0]);i++) { kk[i]=cosf((float)i*0.41f); vv[i]=sinf((float)i*0.19f)-0.1f; }
  for (ai=0;ai<sizeof(att_contracts)/sizeof(att_contracts[0]);ai++) {
    unsigned ct=att_contracts[ai];
    int mask=(ct&BCIR_TENSOR_ATTENTION_FUSED) ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
    bcir_tensor_pool_cap(pool,0); DISPATCH_CLEAR(2);
    assert(bcir_tensor_attention_contract_variant_scan_pool(pool,0,ct,2,33,4,2,16,q,kk,vv,p0,y0,NULL)==
        dispatch_widest(mask));
    memcpy(p1,p0,sizeof(p0));
    assert(bcir_tensor_attention_backward_contract_variant_pool(pool,0,ct,2,33,4,2,16,q,kk,vv,p1,dy,dq0,
        dk0,dv0)==dispatch_widest(mask));
    DISPATCH_RAN(2,dispatch_widest(mask));
    for (bi=0;bi<4;bi++) {
      v=1<<bi;
      for (cap=1;cap<=threads;cap=cap==1 ? 2u : cap*2u) {
        bcir_tensor_pool_cap(pool,cap);
        memset(y1,0,sizeof(y1)); memset(p1,0,sizeof(p1)); DISPATCH_CLEAR(2);
        if (!(mask&v)) {
          assert(bcir_tensor_attention_contract_variant_scan_pool(pool,v,ct,2,33,4,2,16,q,kk,vv,p1,y1,NULL)==0);
          DISPATCH_RAN(2,0);
          for (i=0;i<sizeof(y1)/sizeof(y1[0]);i++) assert(y1[i]==0.0f);
          refused++; continue;
        }
        assert(bcir_tensor_attention_contract_variant_scan_pool(pool,v,ct,2,33,4,2,16,q,kk,vv,p1,y1,NULL)==v);
        assert(dispatch_same(y1,y0,sizeof(y0)/sizeof(y0[0])) && dispatch_same(p1,p0,sizeof(p0)/sizeof(p0[0])));
        assert(bcir_tensor_attention_backward_contract_variant_pool(pool,v,ct,2,33,4,2,16,q,kk,vv,p1,dy,dq1,
            dk1,dv1)==v);
        DISPATCH_RAN(2,v);
        assert(dispatch_same(dq1,dq0,sizeof(dq0)/sizeof(dq0[0])) && dispatch_same(dk1,dk0,sizeof(dk0)/sizeof(dk0[0])) &&
            dispatch_same(dv1,dv0,sizeof(dv0)/sizeof(dv0[0])));
        cases++;
      }
    }
    DISPATCH_CLEAR(2);
    assert(bcir_tensor_attention_contract_variant_scan_pool(pool,3,ct,2,33,4,2,16,q,kk,vv,p1,y1,NULL)==0);
    DISPATCH_RAN(2,0); refused++;
  }
  bcir_tensor_pool_cap(pool,0); bcir_tensor_pool_stop(pool);
  printf("  PASS instances (PX-3b): %zu runs of every admitted instance at caps one, two and all, each "
      "the plain entry's bits, naming what ran, the kernels that ran its own; %zu requests outside the "
      "families refused, nothing written\n",cases,refused);
}
/* INFER-ATTN: the forward under BCIR_TENSOR_ATTENTION_SCRATCH against the stored forward, bit for
 * bit -- the context and the verdict -- and nothing written past bcir_tensor_attention_scratch_floats
 * floats of p, which is smaller than the stored probabilities wherever a context is longer than its
 * strip: the strips ran, not the stored forward. The short contexts (1, 5, 33, 64 and 40 rows of 130
 * channels, past the scores' 128-channel block: one strip) run every instance the products admit
 * under each contract (v1, EXP, FUSED, both), serially and on pools of two and four, finite and not.
 * A row's values are its own, so a long context needs only the boundaries its strips add, and runs
 * fewer arms -- v1 and EXP+FUSED, one thread and four, finite: two strips with the v1 kernel's rows
 * in the second (330), strips that start inside a 32-column block and end in those rows (700 and
 * 703: strips of 80), and, at one thread on the widest instances, the 32-row strips and slots of a
 * context past 2,048 (2100). The bf16 attention likewise, on each of its instances. */
/* ATT-CONTIG: the binary32 attention through its workspace entries against its entries without one,
 * bit for bit -- the context, the verdict, the stored probabilities (and the score adjoints the
 * backward leaves in their place), dq, dk and dv, and the scratch forward's context -- lent the
 * pool's need (its tasks copy their heads' rows: a context at the threshold, a row of more than one
 * head) from a 64-byte boundary and from the address after one (the regions from the first
 * boundary, inside the bytes lent), a byte short of it (the rows in place, the workspace untouched)
 * and none; every byte outside the regions untouched. The sizes' rule: one region a worker where
 * the workers are no more than the forward's tasks, else one a task, of seven t x d copies to the
 * next 64-byte boundary each, and the slack to the workspace's first; 0 below the threshold, for
 * one head a row, or kvheads not dividing heads; SIZE_MAX where it overflows. Three passes: finite;
 * a value and a query non-finite; a key no context reaches (every context finite, the verdict the
 * keys' scan). The short contexts copy in a build that lowers the threshold (the C harness's
 * -contig build) on every instance and contract, finite and not, serially -- and on pools of two,
 * four and 64 (more workers than tasks: a region a task) finite on the widest instance under v1 and
 * EXP+FUSED; the threshold's own context copies in every build, so at one thread. */
static void attention_contig(void) {
  static const size_t shapes[][5]={{1,1,2,1,4},{1,5,2,1,3},{2,33,4,2,17},{1,64,2,2,8},{1,40,2,1,130},
      {2,70,8,4,48},{1,64,4,1,24},{1,31,1,1,15},{1,330,2,1,8},{1,1024,2,1,4}};
  static const unsigned contracts[]={0u,BCIR_TENSOR_ATTENTION_EXP,BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED};
  static const unsigned counts[]={1,2,4,64};
  static _Alignas(max_align_t) unsigned char pool_memory[3][16384];
  bcir_tensor_pool *pools[4]={NULL,NULL,NULL,NULL};
  size_t s,c,i,pi,cases=0,copied=0;
  for (pi=1;pi<4;pi++) {
    size_t need=bcir_tensor_pool_size(counts[pi]);
    if (need && need<=sizeof(pool_memory[0]))
      pools[pi]=bcir_tensor_pool_init(pool_memory[pi-1],sizeof(pool_memory[0]),counts[pi]);
  }
  {  /* the sizes: a region the backward's seven copies and its kept tiles (BWD-KEEP: 32 rows of 64
      * columns a tile, enough tiles for the context), past the workspace's first 64-byte boundary */
    size_t f=(1024*4+15)/16*16,per=(7*f+1024/64*32*64)*4,f2=(1057*4+15)/16*16;
    assert(bcir_tensor_attention_workspace(1,1024,2,1,4)==2*per+63);
    assert(bcir_tensor_attention_pool_workspace(1,1024,2,1,4,1)==per+63);
    assert(bcir_tensor_attention_pool_workspace(1,1024,2,1,4,2)==2*per+63);
    assert(bcir_tensor_attention_pool_workspace(1,1057,2,1,4,1)==(7*f2+(1057/64+1)*32*64)*4+63);
    assert(bcir_tensor_attention_pool_workspace(1,1024,2,1,4,64)==bcir_tensor_attention_workspace(1,1024,2,1,4));
    assert(bcir_tensor_attention_pool_workspace(1,1024,2,1,4,0)==bcir_tensor_attention_workspace(1,1024,2,1,4));
    assert(bcir_tensor_attention_workspace(1,1024,1,1,4)==0 && bcir_tensor_attention_workspace(1,1024,3,2,4)==0);
    assert(bcir_tensor_attention_workspace(0,1024,2,1,4)==0 && bcir_tensor_attention_workspace(1,0,2,1,4)==0 &&
        bcir_tensor_attention_workspace(1,1024,2,0,4)==0 && bcir_tensor_attention_workspace(1,1024,2,1,0)==0);
    assert(bcir_tensor_attention_workspace(SIZE_MAX/2,4096,2,1,64)==SIZE_MAX);
#ifndef BCIR_ATT_CONTIG_FROM
    assert(bcir_tensor_attention_workspace(1,1023,2,1,4)==0);  /* below the threshold: the rows in place */
#endif
  }
  for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
    size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
    size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,most=bcir_tensor_attention_workspace(b,t,nh,nk,d);
    int longest=t>=1024;
    float *q=malloc(nq*sizeof(float)),*k=malloc(nkv*sizeof(float)),*v=malloc(nkv*sizeof(float));
    float *dy=malloc(nq*sizeof(float)),*y[2],*p[2],*dq[2],*dk[2],*dv[2];
    /* the entries' results without a workspace: the stored forward's p and context, the backward's p
     * (its score adjoints), the scratch forward's context */
    float *pf=malloc(np*sizeof(float)),*pb=malloc(np*sizeof(float)),*yf=malloc(nq*sizeof(float)),
        *ys=malloc(nq*sizeof(float));
    unsigned char *raw=malloc(most+128),*ws;
    int arm,badf=0,bads=0;
    if (longest && DT_TSAN) { free(q); free(k); free(v); free(dy); free(pf); free(pb); free(yf); free(ys); free(raw); continue; }
    assert(pf && pb && yf && ys);
    for (arm=0;arm<2;arm++) {
      y[arm]=malloc(nq*sizeof(float)); p[arm]=malloc(np*sizeof(float)); dq[arm]=malloc(nq*sizeof(float));
      dk[arm]=malloc(nkv*sizeof(float)); dv[arm]=malloc(nkv*sizeof(float));
      assert(y[arm] && p[arm] && dq[arm] && dk[arm] && dv[arm]);
    }
    assert(q && k && v && dy && raw);
    ws=raw+(64-(uintptr_t)raw%64)%64;  /* a 64-byte boundary: the slack unused */
    for (pi=0;pi<4;pi++) {
      size_t own=bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,bcir_tensor_pool_threads(pools[pi]));
      unsigned pass;
      if ((pi && !pools[pi]) || (longest && pi)) continue;
      assert(own<=most);
      for (pass=0;pass<3;pass++) for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) {
        unsigned ct=contracts[c];
        int fmask=(ct&BCIR_TENSOR_ATTENTION_FUSED) ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
        int variant;
        if (pass==2 && t<2) continue;
        /* the long context, and the pools (the regions' choice: a worker's or a task's), the finite
         * pass on the widest instance under v1 and EXP+FUSED (the copies are no instance's) */
        if ((longest || pi) && (pass || c==1 || c==2)) continue;
        if (DT_TSAN && (pass || c)) continue;
        for (variant=1;variant<=4;variant*=2) {
          int bad[2],lend;
          if (!(fmask&variant) || ((longest || pi) && variant*2<=(fmask&7))) continue;
          for (lend=0;lend<4;lend++) {  /* none (the entries without one), the need, a byte short, the
                                          * need from an odd address (the regions from its first 64-byte
                                          * boundary, inside the bytes lent) */
            size_t bytes=lend==1 || lend==3 ? own : own ? own-1 : 0; int a=lend ? 1 : 0;
            unsigned char *at=lend==3 ? ws+1 : ws;
            for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
            for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
            if (pass==1 && t>3) { v[(t-1)*nk*d]=INFINITY; q[d]=NAN; }
            if (pass==2) {  /* the last batch's key 1 infinite in channel 0, its KV head's queries
                             * negative there: every context finite, the verdict the keys' scan alone */
              size_t rb=b-1,hh;
              for (i=0;i<t;i++) for (hh=0;hh<nh/nk;hh++) q[((rb*t+i)*nh+hh)*d]=-fabsf(q[((rb*t+i)*nh+hh)*d])-0.5f;
              k[((rb*t+1)*nk+0)*d]=INFINITY;
            }
            memset(ws,0xa5,most+64);
            memset(p[a],0xff,np*sizeof(float)); memset(y[a],0xff,nq*sizeof(float)); bad[a]=0;
            if (!lend) assert(bcir_tensor_attention_contract_variant_scan_pool(pools[pi],variant,ct,b,t,nh,nk,d,
                q,k,v,p[a],y[a],&bad[a])==variant);
            else assert(bcir_tensor_attention_contract_variant_scan_ws_pool(pools[pi],variant,ct,b,t,nh,nk,d,
                q,k,v,p[a],y[a],&bad[a],at,bytes)==variant);
            if (!lend) { memcpy(pf,p[a],np*sizeof(float)); memcpy(yf,y[a],nq*sizeof(float)); badf=bad[a]; }
            else assert(!memcmp(yf,y[a],nq*sizeof(float)) && badf==bad[a] && !memcmp(pf,p[a],np*sizeof(float)));
            memset(dq[a],0xff,nq*sizeof(float)); memset(dk[a],0xff,nkv*sizeof(float)); memset(dv[a],0xff,nkv*sizeof(float));
            if (!lend) assert(bcir_tensor_attention_backward_contract_variant_pool(pools[pi],variant,ct,b,t,nh,nk,d,
                q,k,v,p[a],dy,dq[a],dk[a],dv[a])==variant);
            else assert(bcir_tensor_attention_backward_contract_variant_ws_pool(pools[pi],variant,ct,b,t,nh,nk,d,
                q,k,v,p[a],dy,dq[a],dk[a],dv[a],at,bytes)==variant);
            if (!lend) memcpy(pb,p[a],np*sizeof(float));
            else assert(!memcmp(dq[0],dq[1],nq*sizeof(float)) && !memcmp(dk[0],dk[1],nkv*sizeof(float)) &&
                !memcmp(dv[0],dv[1],nkv*sizeof(float)) && !memcmp(pb,p[a],np*sizeof(float)));
            /* the scratch forward's context and verdict: the stored forward's (the slots are p's start) */
            memset(y[a],0xff,nq*sizeof(float)); bad[a]=0;
            if (!lend) assert(bcir_tensor_attention_contract_variant_scan_pool(pools[pi],variant,
                ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p[a],y[a],&bad[a])==variant);
            else assert(bcir_tensor_attention_contract_variant_scan_ws_pool(pools[pi],variant,
                ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p[a],y[a],&bad[a],at,bytes)==variant);
            if (!lend) { memcpy(ys,y[a],nq*sizeof(float)); bads=bad[a]; }
            else assert(!memcmp(ys,y[a],nq*sizeof(float)) && bads==bad[a]);
            assert(bad[a]==(pass==1 ? t>3 : pass==2) && badf==bads && !memcmp(yf,ys,nq*sizeof(float)));
            if (pass==2) assert(all_finite(ys,nq));
            /* nothing written outside the regions -- [0, own-63) lent from a 64-byte boundary, [64, own+1)
             * from the address after it -- and a byte short of the need, nothing written at all */
            for (i=0;i<most+64;i++)
              if (!own || !((lend==1 && i<own-63) || (lend==3 && i>=64 && i<own+1))) assert(ws[i]==0xa5);
            if (lend) cases++;
            if ((lend==1 || lend==3) && own) copied++;
          }
        }
      }
    }
    for (arm=0;arm<2;arm++) { free(y[arm]); free(p[arm]); free(dq[arm]); free(dk[arm]); free(dv[arm]); }
    free(q); free(k); free(v); free(dy); free(pf); free(pb); free(yf); free(ys); free(raw);
  }
  for (pi=1;pi<4;pi++) if (pools[pi]) bcir_tensor_pool_stop(pools[pi]);
  assert(cases>0 && copied>0);
  printf("  PASS ATT-CONTIG: %zu cases of the attention's workspace entries their entries' without one bit for bit,"
      " %zu of them with the tasks on copies of their heads' rows\n",cases,copied);
}
/* FWD-STRIP: from BCIR_ATT_STRIP_FROM positions the stored forward runs each head a strip of rows at a
 * time, in place -- against the reference loops (v1's and the fused contract's, under libm's
 * exponential and BCIR's) at the threshold's own context, below it and past it (a context no multiple
 * of 8), p NaN-filled first: p (every future column +0.0), y and the verdict bit for bit, finite and
 * with an infinite key, serially and on a pool of four, without a workspace and lent the pool's need
 * (ATT-CONTIG's copies); and, where the build has the unit's probe, the path the values cannot show:
 * every head of a stored forward at or past the threshold ran its strips, below it none, and the
 * scratch forward none. P-LOWER: each forward once more under BCIR_TENSOR_ATTENTION_LOWER, p filled
 * with a NaN canary first -- y, the verdict and every probability j <= i the same, every future
 * column +0.0 or the canary, and past the threshold the canary still in the last column of every row
 * whose strip ends before it (a strip's rows: bcir_tensor_attention_scratch_floats' slot), so the
 * fill did not run -- and, on the pool with the workspace lent, the backward from those probabilities,
 * canaries and all, the backward's from the reference's, bit for bit (no column past a row's diagonal
 * is read). Below the threshold one arm: LOWER leaves nothing there, the head writing every column. */
#ifdef BCIR_DT_ATTENTION_PROBE
size_t bcir_tensor_attention_strips_probe(void);
#endif
static void attention_strips(void) {
#ifdef BCIR_ATT_STRIP_FROM
  const size_t from=BCIR_ATT_STRIP_FROM;
#else
  const size_t from=1024;
#endif
  const size_t ts[3]={from>1 ? from-1 : from,from,from+7},b=1,nh=2,nk=1,d=4;
  static const unsigned contracts[]={0u,BCIR_TENSOR_ATTENTION_EXP,BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED};
  static _Alignas(max_align_t) unsigned char pool_memory[16384];
  const uint32_t canary=0x7fc0beefu;
  bcir_tensor_pool *pools[2]={NULL,NULL};
  size_t si,c,i,pi,hr,row,col,cases=0,striped=0,lowered=0,kept=0,backs=0;
  {
    size_t need=bcir_tensor_pool_size(4);
    if (need && need<=sizeof(pool_memory)) pools[1]=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),4);
  }
  for (si=0;si<3;si++) {
    size_t t=ts[si],nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,most=bcir_tensor_attention_workspace(b,t,nh,nk,d);
    size_t rows=bcir_tensor_attention_scratch_floats(1,t,1)/t,last=(t-1)/rows*rows;  /* the last strip's first row */
    float *q,*k,*v,*dy,*p0,*p1,*pr,*y0,*y1,*dq[2],*dk[2],*dv[2];
    unsigned char *raw,*ws;
    unsigned inf;
    if (si && ts[si]==ts[si-1]) continue;
    if (DT_TSAN && t>=1024) continue;
    q=malloc(nq*sizeof(float)); k=malloc(nkv*sizeof(float)); v=malloc(nkv*sizeof(float));
    dy=malloc(nq*sizeof(float)); p0=malloc(np*sizeof(float)); p1=malloc(np*sizeof(float));
    pr=malloc(np*sizeof(float)); y0=malloc(nq*sizeof(float)); y1=malloc(nq*sizeof(float));
    raw=malloc(most+128);
    assert(q && k && v && dy && p0 && p1 && pr && y0 && y1 && raw);
    for (i=0;i<2;i++) {
      dq[i]=malloc(nq*sizeof(float)); dk[i]=malloc(nkv*sizeof(float)); dv[i]=malloc(nkv*sizeof(float));
      assert(dq[i] && dk[i] && dv[i]);
    }
    ws=raw+(64-(uintptr_t)raw%64)%64;
    for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) for (inf=0;inf<2;inf++) {
      unsigned ct=contracts[c]; int ex=(ct&BCIR_TENSOR_ATTENTION_EXP)!=0,lend,low;
      if (inf && c!=0 && c!=3) continue;
      for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
      for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
      if (inf) k[(t/2)*nk*d]=INFINITY;
      if (ct&BCIR_TENSOR_ATTENTION_FUSED) ref_attention_fused(b,t,nh,nk,d,q,k,v,p0,y0,ex ? bcir_tensor_expf : expf);
      else ref_attention(b,t,nh,nk,d,q,k,v,p0,y0,ex ? bcir_tensor_expf : expf);
      /* the backward from the reference's probabilities (every value the same at every thread count
       * and with a workspace or none: dispatch_entries, attention_contig) */
      memcpy(pr,p0,np*sizeof(float));
      memset(dq[0],0xff,nq*sizeof(float)); memset(dk[0],0xff,nkv*sizeof(float)); memset(dv[0],0xff,nkv*sizeof(float));
      assert(bcir_tensor_attention_backward_contract_variant_pool(NULL,0,ct,b,t,nh,nk,d,q,k,v,pr,dy,dq[0],dk[0],dv[0]));
      for (pi=0;pi<2;pi++) for (lend=0;lend<2;lend++) for (low=0;low<2;low++) {
        size_t own=bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,bcir_tensor_pool_threads(pools[pi]));
        unsigned cl=ct|(low ? BCIR_TENSOR_ATTENTION_LOWER : 0u);
        int bad=0;
        if (pi && !pools[pi]) continue;
        if (low && t<from && (pi || lend)) continue;  /* below the threshold LOWER writes every column: one arm */
        if (low) for (i=0;i<np;i++) memcpy(p1+i,&canary,sizeof canary);
        else memset(p1,0xff,np*sizeof(float));
        memset(y1,0xff,nq*sizeof(float));
#ifdef BCIR_DT_ATTENTION_PROBE
        (void)bcir_tensor_attention_strips_probe();
#endif
        if (lend) assert(bcir_tensor_attention_contract_variant_scan_ws_pool(pools[pi],0,cl,b,t,nh,nk,d,q,k,v,p1,y1,&bad,
            ws,own));
        else assert(bcir_tensor_attention_contract_variant_scan_pool(pools[pi],0,cl,b,t,nh,nk,d,q,k,v,p1,y1,&bad));
        assert(!memcmp(y0,y1,nq*sizeof(float)));
        for (hr=0;hr<b*nh;hr++) for (row=0;row<t;row++) {
          const float *r0=p0+(hr*t+row)*t,*r1=p1+(hr*t+row)*t; uint32_t bits;
          assert(!memcmp(r0,r1,(row+1)*sizeof(float)));  /* j <= i */
          for (col=row+1;col<t;col++) {
            memcpy(&bits,r1+col,sizeof bits);
            assert(bits==0u || (low && bits==canary));  /* +0.0, or under LOWER as p held it */
          }
          /* at or past the threshold, a row whose strip ends before the last column never writes it */
          if (low && t>=from && row<last) { memcpy(&bits,r1+t-1,sizeof bits); assert(bits==canary); kept++; }
        }
        assert(bad==!(all_finite(y1,nq) && all_finite(k,nkv)) && bad==(int)inf);
#ifdef BCIR_DT_ATTENTION_PROBE
        assert(bcir_tensor_attention_strips_probe()==(t>=from ? b*nh : 0));
        if (!low) {  /* the scratch forward runs its own strips, never the stored forward's */
          bad=0;
          assert(bcir_tensor_attention_contract_variant_scan_pool(pools[pi],0,ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,
              d,q,k,v,p1,y1,&bad));
          assert(bcir_tensor_attention_strips_probe()==0 && !memcmp(y0,y1,nq*sizeof(float)));
        }
#endif
        /* the backward from the canaries' probabilities, LOWER in its word too: on the pool, lent (the
         * trainer's own path past the threshold; attention_lower and trainer_lower take the rest) */
        if (low && lend && pi==(pools[1] ? 1u : 0u)) {
          unsigned cb=ct|BCIR_TENSOR_ATTENTION_LOWER;
          memset(dq[1],0xff,nq*sizeof(float)); memset(dk[1],0xff,nkv*sizeof(float)); memset(dv[1],0xff,nkv*sizeof(float));
          if (lend) assert(bcir_tensor_attention_backward_contract_variant_ws_pool(pools[pi],0,cb,b,t,nh,nk,d,q,k,v,p1,
              dy,dq[1],dk[1],dv[1],ws,own));
          else assert(bcir_tensor_attention_backward_contract_variant_pool(pools[pi],0,cb,b,t,nh,nk,d,q,k,v,p1,dy,dq[1],
              dk[1],dv[1]));
          assert(!memcmp(dq[0],dq[1],nq*sizeof(float)) && !memcmp(dk[0],dk[1],nkv*sizeof(float)) &&
              !memcmp(dv[0],dv[1],nkv*sizeof(float)));
          backs++;
        }
        cases++;
        if (t>=from) striped++;
        if (low) lowered++;
      }
    }
    free(q); free(k); free(v); free(dy); free(p0); free(p1); free(pr); free(y0); free(y1); free(raw);
    for (i=0;i<2;i++) { free(dq[i]); free(dk[i]); free(dv[i]); }
  }
  if (pools[1]) bcir_tensor_pool_stop(pools[1]);
  assert(cases>0 && striped>0 && lowered>0 && backs>0);
  if (from>=1024) assert(kept>0);  /* the measuring host's threshold: every context past it strips */
  printf("  PASS FWD-STRIP: %zu stored forwards bit for bit the reference loops' at the threshold's context %zu and "
      "either side of it, %zu of them a strip of rows at a time; %zu under LOWER (P-LOWER: %zu last columns as p "
      "held them, %zu backwards from them the reference's)\n",cases,from,striped,lowered,kept,backs);
}
static void attention_scratch(void) {
  static const size_t shapes[][5]={{1,1,1,1,2},{1,5,2,1,3},{2,33,4,2,17},{1,64,2,2,8},
      {1,40,2,1,130},{1,330,2,1,8},{1,700,2,1,8},{1,703,1,1,12},{1,2100,1,1,4}};
  static const unsigned contracts[]={0u,BCIR_TENSOR_ATTENTION_EXP,BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED};
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  const uint32_t canary=0x7fc0deadu;
  size_t s,c,i,cases=0,bf16_cases=0,compact=0; unsigned threads,pass; int variant;
  for (threads=1;threads<=4;threads+=threads==1 ? 1 : 2) {
    bcir_tensor_pool *pool=NULL;
    if (threads>1) {
      size_t need=bcir_tensor_pool_size(threads);
      if (!need) continue;
      assert(need<=sizeof(pool_memory));
      pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool);
    }
    for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) {
      size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
      int longer=t>64,longest=t>=2048;
      if ((longer && threads==2) || (longest && threads>1)) continue;
      size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,sf=bcir_tensor_attention_scratch_floats(b,t,nh);
      size_t ws=bcir_tensor_attention_bf16_pool_workspace(b,t,nh,nk,d,threads);
      float *q=malloc(nq*sizeof(float)),*k=malloc(nkv*sizeof(float)),*v=malloc(nkv*sizeof(float));
      float *y0=malloc(nq*sizeof(float)),*y1=malloc(nq*sizeof(float)),*p0=malloc(np*sizeof(float));
      float *p1=malloc((sf+4096)*sizeof(float)); unsigned char *raw=malloc(ws+64),*work;
      assert(q && k && v && y0 && y1 && p0 && p1 && raw && sf<=np);
      work=raw+(64-(uintptr_t)raw%64)%64;  /* a 64-byte boundary: the packs' own, no slack taken */
      /* a slot a (batch, query head): whole rows, at least 32 of them or all */
      assert(sf%(b*nh*t)==0 && (sf/(b*nh*t)>=32 || sf/(b*nh*t)==t) && (t<2048 || sf==b*nh*32*t));
      compact+=sf<np;
      for (pass=0;pass<2;pass++) for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) {
        unsigned ct=contracts[c];
        int fmask=(ct&BCIR_TENSOR_ATTENTION_FUSED) ? bcir_tensor_mm_fused_variants() : bcir_tensor_mm_variants();
        int bmask=bcir_tensor_mm_bf16_variants();
        if (longer && (pass || c==1 || c==2)) continue;
        for (variant=1;variant<=8;variant*=2) {
          int fp32=variant<8 && (fmask&variant) && (!longest || variant*2>(fmask&7));
          /* the bf16 attention has no fused contract: its products are the bf16 GEMM's */
          int bf16=!(ct&BCIR_TENSOR_ATTENTION_FUSED) && variant!=2 && (bmask&variant) &&
              (!longest || variant*2>bmask);
          int bad0,bad1,form;
          for (form=0;form<2;form++) {
            if (!(form ? bf16 : fp32)) continue;
            for (i=0;i<nq;i++) q[i]=sinf((float)i*0.37f)*1.7f;
            for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
            if (pass && t>3) { v[(t-1)*nk*d]=INFINITY; q[d]=NAN; }
            memset(p0,0xff,np*sizeof(float)); memset(y0,0xff,nq*sizeof(float));
            memset(y1,0xff,nq*sizeof(float));
            for (i=0;i<sf+4096;i++) memcpy(p1+i,&canary,sizeof canary);
            bad0=bad1=0;
            if (form) {
              assert(bcir_tensor_attention_bf16_scan_pool(pool,variant,ct,b,t,nh,nk,d,q,k,v,p0,y0,
                  &bad0,work,ws)==variant);
              assert(bcir_tensor_attention_bf16_scan_pool(pool,variant,ct|BCIR_TENSOR_ATTENTION_SCRATCH,
                  b,t,nh,nk,d,q,k,v,p1,y1,&bad1,work,ws)==variant);
              bf16_cases++;
            } else {
              assert(bcir_tensor_attention_contract_variant_scan_pool(pool,variant,ct,b,t,nh,nk,d,q,k,v,
                  p0,y0,&bad0)==variant);
              assert(bcir_tensor_attention_contract_variant_scan_pool(pool,variant,
                  ct|BCIR_TENSOR_ATTENTION_SCRATCH,b,t,nh,nk,d,q,k,v,p1,y1,&bad1)==variant);
              cases++;
            }
            assert(!memcmp(y0,y1,nq*sizeof(float)) && bad0==bad1 && bad0==(pass && t>3));
            for (i=sf;i<sf+4096;i++) assert(!memcmp(p1+i,&canary,sizeof canary));
          }
        }
      }
      free(q); free(k); free(v); free(y0); free(y1); free(p0); free(p1); free(raw);
    }
    if (pool) bcir_tensor_pool_stop(pool);
  }
  assert(cases>0 && bf16_cases>0 && compact>0);
  printf("  PASS attention without its probabilities (INFER-ATTN): the context and the verdict the "
      "stored forward's on %zu binary32 and %zu bf16 cases, every write inside the scratch slots\n",
      cases,bf16_cases);
}
/* INFER-ATTN: bcir_decoder_forward's attention keeps no probabilities, and its logits are the
 * backward's forward's bit for bit: under the v1, EXP and ATTN-FMA attention, a GEMM contract and
 * the bf16 attention, at one and four threads, over a context of one strip and of nine (700 rows);
 * with the probe, every forward job ran under SCRATCH and the backward's forward never did. */
static void trainer_infer(void) {
  static const uint32_t contracts[]={0,BCIR_DT_CONTRACT_EXP,BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_KBLOCK|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_ATTNBF16,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTNBF16};
  static const uint32_t times[]={8,700};
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  size_t c,ti,i,runs=0; unsigned threads;
  for (ti=0;ti<sizeof(times)/sizeof(times[0]);ti++) for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++)
    for (threads=1;threads<=4;threads+=3) {
      bcir_decoder_spec spec={16,8,2,1,1,16,1,4,1,10000.0,1e-6};
      bcir_decoder_state st; bcir_tensor_pool *pool=NULL; size_t n,need,off=128;
      uint32_t *ids,*targets; float *logits; double loss; unsigned char *raw;
      if (threads>1) {
        if (!bcir_tensor_pool_size(threads) || bcir_tensor_pool_size(threads)>sizeof(pool_memory)) continue;
        pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool);
      }
      spec.time=times[ti]; n=spec.batch*spec.time;
      memset(&st,0,sizeof st);
      assert(bcir_decoder_make_plan(&spec,&st.plan)==0);
      st.weights=calloc(st.plan.parameters,sizeof(float)); st.gradients=calloc(st.plan.parameters,sizeof(float));
      st.moment1=calloc(st.plan.parameters,sizeof(float)); st.moment2=calloc(st.plan.parameters,sizeof(float));
      st.activations=calloc(st.plan.activations,sizeof(float)); st.scratch=calloc(st.plan.scratch,sizeof(float));
      assert(st.weights && st.gradients && st.moment1 && st.moment2 && st.activations && st.scratch);
      st.weight_capacity=st.gradient_capacity=st.moment1_capacity=st.moment2_capacity=st.plan.parameters;
      st.activation_capacity=st.plan.activations; st.scratch_capacity=st.plan.scratch;
      st.beta1_power=st.beta2_power=1.0; st.pool=pool; st.contract=contracts[c];
      for (i=0;i<st.plan.parameters;i++) st.weights[i]=0.1f*sinf((float)(i+1)*0.37f);
      for (i=0;i<8;i++) st.weights[off+i]=1.0f;  /* the layer's two norms and the final one */
      off+=8+64+32+32+64; for (i=0;i<8;i++) st.weights[off+i]=1.0f;
      off+=8+3*128; for (i=0;i<8;i++) st.weights[off+i]=1.0f;
      need=bcir_decoder_workspace_bytes(&st.plan,st.contract);
      raw=malloc(need+64); assert(raw);
      st.workspace=need ? raw+(64-(uintptr_t)raw%64)%64 : NULL; st.workspace_bytes=need;
      ids=malloc(n*sizeof(uint32_t)); targets=malloc(n*sizeof(uint32_t)); logits=malloc(n*16*sizeof(float));
      assert(ids && targets && logits);
      stream(ids,targets,n);
#ifdef BCIR_DT_ATTENTION_PROBE
      (void)bcir_tensor_attention_probe(0); (void)bcir_tensor_attention_probe(1);
#endif
      assert(bcir_decoder_forward(&st,ids,n)==BCIR_DT_OK);
#ifdef BCIR_DT_ATTENTION_PROBE
      {
        unsigned want=((st.contract&BCIR_DT_CONTRACT_EXP) ? BCIR_TENSOR_ATTENTION_EXP : 0u)|
            ((st.contract&BCIR_DT_CONTRACT_ATTNBF16) ? BCIR_TENSOR_ATTENTION_BF16 :
             (st.contract&BCIR_DT_CONTRACT_ATTN) ? BCIR_TENSOR_ATTENTION_FUSED : 0u);
        assert(bcir_tensor_attention_probe(0)==(want|BCIR_TENSOR_ATTENTION_SCRATCH));
        memcpy(logits,st.activations+st.plan.activations-n*16,n*16*sizeof(float));
        assert(bcir_decoder_loss_backward(&st,ids,targets,n,0,&loss)==BCIR_DT_OK);
        /* the backward's forward a stored one: under P-LOWER where its attention is binary32 */
        assert(bcir_tensor_attention_probe(0)==(want|((st.contract&BCIR_DT_CONTRACT_ATTNBF16) ? 0u :
            BCIR_TENSOR_ATTENTION_LOWER)));
      }
#else
      memcpy(logits,st.activations+st.plan.activations-n*16,n*16*sizeof(float));
      assert(bcir_decoder_loss_backward(&st,ids,targets,n,0,&loss)==BCIR_DT_OK);
#endif
      assert(!memcmp(logits,st.activations+st.plan.activations-n*16,n*16*sizeof(float)));
      runs++;
      free(ids); free(targets); free(logits); free(raw);
      free(st.weights); free(st.gradients); free(st.moment1); free(st.moment2);
      free(st.activations); free(st.scratch);
      if (pool) bcir_tensor_pool_stop(pool);
    }
  assert(runs>0);
  printf("  PASS the forward without its probabilities (INFER-ATTN): %zu forwards, their logits the "
      "backward's forward's bit for bit\n",runs);
}
/* P-LOWER: the stored forward under BCIR_TENSOR_ATTENTION_LOWER on attention_reference's shapes, under
 * v1's and the fused contract's products with libm's exponential and BCIR's, finite and with an
 * infinite value and a NaN adjoint, serially and on a pool of four, without a workspace and lent the
 * pool's need, p filled with a NaN canary first: y, the verdict and every probability j <= i the
 * reference loops' bit for bit, every future column +0.0 or the canary; then the backward from those
 * probabilities, canaries and all (LOWER in its word or not), dq, dk and dv the reference backward's
 * from the reference's probabilities, bit for bit -- no column past a row's diagonal is read. Where
 * the build runs these contexts' forwards a strip at a time (BCIR_ATT_STRIP_FROM at most the longest,
 * 330: the harness's -long and -contig builds) the canary is kept in some column; below it in none,
 * the head-at-a-time forward writing every column. */
static void attention_lower(void) {
#ifdef BCIR_ATT_STRIP_FROM
  const size_t from=BCIR_ATT_STRIP_FROM;
#else
  const size_t from=1024;
#endif
  static float q[2*70*8*48],k[2*70*4*130],v[2*70*4*130],dy[2*70*8*48],p0[2*330*330],p1[2*330*330],
      y0[2*70*8*48],y1[2*70*8*48],dq0[2*70*8*48],dq1[2*70*8*48],dk0[2*70*4*130],dk1[2*70*4*130],
      dv0[2*70*4*130],dv1[2*70*4*130];
  static const size_t shapes[][5]={{1,1,1,1,2},{1,2,1,1,1},{2,5,4,2,6},{1,37,2,1,48},
      {2,70,8,4,48},{1,33,3,1,65},{1,40,2,2,130},{2,64,4,1,24},{1,330,2,1,8},
      {1,7,1,1,3},{1,8,1,1,16},{1,9,2,1,17},{1,31,1,1,15},{1,32,2,2,47},{1,33,2,1,49},
      {1,40,1,1,1},{1,20,1,1,32},{1,9,2,1,33},{1,11,2,2,81}};
  static const unsigned contracts[]={0u,BCIR_TENSOR_ATTENTION_EXP,BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP|BCIR_TENSOR_ATTENTION_FUSED};
  static _Alignas(max_align_t) unsigned char pool_memory[16384];
  const uint32_t canary=0x7fc0beefu;
  bcir_tensor_pool *pools[2]={NULL,NULL};
  size_t s,c,i,pi,hr,row,col,cases=0,kept=0,longest=0; unsigned pass;
  {
    size_t need=bcir_tensor_pool_size(4);
    if (need && need<=sizeof(pool_memory)) pools[1]=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),4);
  }
  for (pass=0;pass<2;pass++) for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (c=0;c<4;c++) {
    size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4];
    size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,most=bcir_tensor_attention_workspace(b,t,nh,nk,d);
    unsigned ct=contracts[c]; int ex=(ct&BCIR_TENSOR_ATTENTION_EXP)!=0,lend;
    unsigned char *raw=malloc(most+128),*ws;
    assert(raw && nq<=sizeof(q)/sizeof(q[0]) && nkv<=sizeof(k)/sizeof(k[0]) && np<=sizeof(p0)/sizeof(p0[0]));
    ws=raw+(64-(uintptr_t)raw%64)%64;
    if (t>longest) longest=t;
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    if (pass && t>3) { v[(t-1)*nk*d]=INFINITY; dy[d]=NAN; }
    if (ct&BCIR_TENSOR_ATTENTION_FUSED) {
      ref_attention_fused(b,t,nh,nk,d,q,k,v,p0,y0,ex ? bcir_tensor_expf : expf);
      ref_attention_backward_fused(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
    } else {
      ref_attention(b,t,nh,nk,d,q,k,v,p0,y0,ex ? bcir_tensor_expf : expf);
      ref_attention_backward(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
    }
    for (pi=0;pi<2;pi++) for (lend=0;lend<2;lend++) {
      size_t own=bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,bcir_tensor_pool_threads(pools[pi]));
      unsigned cl=ct|BCIR_TENSOR_ATTENTION_LOWER,cb=ct|(lend ? BCIR_TENSOR_ATTENTION_LOWER : 0u);
      int bad=0;
      if (pi && !pools[pi]) continue;
      for (i=0;i<np;i++) memcpy(p1+i,&canary,sizeof canary);
      memset(y1,0xff,nq*sizeof(float));
      if (lend) assert(bcir_tensor_attention_contract_variant_scan_ws_pool(pools[pi],0,cl,b,t,nh,nk,d,q,k,v,p1,y1,&bad,
          ws,own));
      else assert(bcir_tensor_attention_contract_variant_scan_pool(pools[pi],0,cl,b,t,nh,nk,d,q,k,v,p1,y1,&bad));
      assert(!memcmp(y0,y1,nq*sizeof(float)) && bad==!(all_finite(y1,nq) && all_finite(k,nkv)));
      for (hr=0;hr<b*nh;hr++) for (row=0;row<t;row++) {
        const float *r0=p0+(hr*t+row)*t,*r1=p1+(hr*t+row)*t;
        assert(!memcmp(r0,r1,(row+1)*sizeof(float)));  /* j <= i */
        for (col=row+1;col<t;col++) {
          uint32_t bits;
          memcpy(&bits,r1+col,sizeof bits);
          assert(bits==0u || bits==canary);
          kept+=bits==canary;
        }
      }
      memset(dq1,0xff,nq*sizeof(float)); memset(dk1,0xff,nkv*sizeof(float)); memset(dv1,0xff,nkv*sizeof(float));
      if (lend) assert(bcir_tensor_attention_backward_contract_variant_ws_pool(pools[pi],0,cb,b,t,nh,nk,d,q,k,v,p1,dy,
          dq1,dk1,dv1,ws,own));
      else assert(bcir_tensor_attention_backward_contract_variant_pool(pools[pi],0,cb,b,t,nh,nk,d,q,k,v,p1,dy,dq1,dk1,
          dv1));
      assert(!memcmp(dq0,dq1,nq*sizeof(float)) && !memcmp(dk0,dk1,nkv*sizeof(float)) &&
          !memcmp(dv0,dv1,nkv*sizeof(float)));
      cases++;
    }
    free(raw);
  }
  if (pools[1]) bcir_tensor_pool_stop(pools[1]);
  assert(cases>0 && (from<=longest ? kept>0 : kept==0));
  printf("  PASS P-LOWER: %zu stored forwards under LOWER the reference loops' bit for bit (y, the verdict, every "
      "probability j <= i), every future column +0.0 or as p held it (%zu kept), the backward from them the "
      "reference's\n",cases,kept);
}
/* P-LOWER: a training step reads nothing of its arenas it did not write -- the probabilities past a
 * row's diagonal among them, which its stored binary32 forward leaves as the region held them from
 * BCIR_ATT_STRIP_FROM: from activations, scratch and workspace whose every float is a NaN canary
 * first, the step's loss, gradients and logits are the step's from zeroed arenas bit for bit, under
 * the v1, EXP and ATTN-FMA attention, a GEMM contract and the bf16 attention, at one and four threads,
 * over a context of 300 rows (two strips a head in the harness's -long builds, which strip at every
 * context) and of 1031 (past the threshold). And the canary is kept in the activations exactly where a
 * binary32 forward ran a strip at a time: there its strips leave the future columns; every other float
 * of the arena the step writes. */
static void trainer_lower(void) {
#ifdef BCIR_ATT_STRIP_FROM
  const size_t from=BCIR_ATT_STRIP_FROM;
#else
  const size_t from=1024;
#endif
  static const uint32_t contracts[]={0,BCIR_DT_CONTRACT_EXP,BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTN,
      BCIR_DT_CONTRACT_KBLOCK|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_REDUCE,
      BCIR_DT_CONTRACT_BF16|BCIR_DT_CONTRACT_EXP|BCIR_DT_CONTRACT_ATTNBF16};
  static const uint32_t times[]={300,1031};
  static _Alignas(max_align_t) unsigned char pool_memory[4096];
  const uint32_t canary=0x7fc0beefu;
  size_t c,ti,i,runs=0,lowered=0; unsigned threads;
  for (ti=0;ti<sizeof(times)/sizeof(times[0]);ti++) for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++)
    for (threads=1;threads<=4;threads+=3) {
      bcir_decoder_spec spec={16,8,2,1,1,16,1,4,1,10000.0,1e-6};
      bcir_decoder_state st[2]; bcir_tensor_pool *pool=NULL; size_t n,need,arm,canaries=0;
      uint32_t *ids,*targets; double loss[2]; unsigned char *raw[2];
      int binary32=!(contracts[c]&BCIR_DT_CONTRACT_ATTNBF16);
      if (DT_TSAN && times[ti]>=1024) continue;
      if (threads>1) {
        if (!bcir_tensor_pool_size(threads) || bcir_tensor_pool_size(threads)>sizeof(pool_memory)) continue;
        pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),threads); assert(pool);
      }
      spec.time=times[ti]; n=spec.batch*spec.time;
      ids=malloc(n*sizeof(uint32_t)); targets=malloc(n*sizeof(uint32_t)); assert(ids && targets);
      stream(ids,targets,n);
      for (arm=0;arm<2;arm++) {  /* arm 0 zeroed, arm 1 the canary in every float of its arenas */
        bcir_decoder_state *x=&st[arm]; size_t off=128;
        memset(x,0,sizeof *x);
        assert(bcir_decoder_make_plan(&spec,&x->plan)==0);
        x->weights=calloc(x->plan.parameters,sizeof(float)); x->gradients=calloc(x->plan.parameters,sizeof(float));
        x->moment1=calloc(x->plan.parameters,sizeof(float)); x->moment2=calloc(x->plan.parameters,sizeof(float));
        x->activations=calloc(x->plan.activations,sizeof(float)); x->scratch=calloc(x->plan.scratch,sizeof(float));
        assert(x->weights && x->gradients && x->moment1 && x->moment2 && x->activations && x->scratch);
        x->weight_capacity=x->gradient_capacity=x->moment1_capacity=x->moment2_capacity=x->plan.parameters;
        x->activation_capacity=x->plan.activations; x->scratch_capacity=x->plan.scratch;
        x->beta1_power=x->beta2_power=1.0; x->pool=pool; x->contract=contracts[c];
        for (i=0;i<x->plan.parameters;i++) x->weights[i]=0.1f*sinf((float)(i+1)*0.37f);
        for (i=0;i<8;i++) x->weights[off+i]=1.0f;  /* the layer's two norms and the final one */
        off+=8+64+32+32+64; for (i=0;i<8;i++) x->weights[off+i]=1.0f;
        off+=8+3*128; for (i=0;i<8;i++) x->weights[off+i]=1.0f;
        need=bcir_decoder_workspace_bytes(&x->plan,x->contract);
        raw[arm]=calloc(need+64,1); assert(raw[arm]);
        x->workspace=need ? raw[arm]+(64-(uintptr_t)raw[arm]%64)%64 : NULL; x->workspace_bytes=need;
        if (arm) {
          for (i=0;i<x->plan.activations;i++) memcpy(x->activations+i,&canary,sizeof canary);
          for (i=0;i<x->plan.scratch;i++) memcpy(x->scratch+i,&canary,sizeof canary);
          for (i=0;i+4<=need;i+=4) memcpy((unsigned char *)x->workspace+i,&canary,sizeof canary);
        }
        assert(bcir_decoder_loss_backward(x,ids,targets,n,0,&loss[arm])==BCIR_DT_OK);
      }
      assert(!memcmp(&loss[0],&loss[1],sizeof loss[0]));
      assert(!memcmp(st[0].gradients,st[1].gradients,st[0].plan.parameters*sizeof(float)));
      assert(!memcmp(st[0].activations+st[0].plan.activations-n*16,st[1].activations+st[1].plan.activations-n*16,
          n*16*sizeof(float)));
      for (i=0;i<st[1].plan.activations;i++) { uint32_t bits; memcpy(&bits,st[1].activations+i,4); canaries+=bits==canary; }
      assert((canaries>0)==(binary32 && spec.time>=from));
      lowered+=canaries>0;
      runs++;
      for (arm=0;arm<2;arm++) {
        free(raw[arm]); free(st[arm].weights); free(st[arm].gradients); free(st[arm].moment1); free(st[arm].moment2);
        free(st[arm].activations); free(st[arm].scratch);
      }
      free(ids); free(targets);
      if (pool) bcir_tensor_pool_stop(pool);
    }
  assert(runs>0 && (DT_TSAN || lowered>0));
  printf("  PASS a training step from NaN-filled arenas (P-LOWER): %zu steps the zeroed arenas' bit for bit, %zu of "
      "them leaving the canary where their stored forward's strips left p's future columns\n",runs,lowered);
}
/* BWD-KEEP: with a region (ATT-CONTIG's: from BCIR_ATT_CONTIG_FROM positions, where a row holds more than
 * one head) a backward task keeps every probability-adjoint tile of a row block and evaluates each once;
 * without one -- no workspace, or a byte short of the pool's need -- the tiles past the first
 * BCIR_ATT_KEEP (256) columns are evaluated twice, as before. dq, dk and dv are the same bits all three
 * ways, under v1's and the fused contract's products, serially and on a pool of four, at the threshold's
 * context and past it (33 rows on: neither a whole row block nor a whole tile); and where the build has
 * the unit's probe, the tiles each backward evaluated are its path's count -- the path the values cannot
 * show. The attention references and ATT-CONTIG's cases hold the kept path to the reference loops. */
#ifdef BCIR_DT_ATTENTION_PROBE
size_t bcir_tensor_attention_tiles_probe(void);
#endif
/* one head's tiles: row blocks of 32, each reaching tiles of 64 columns to its last row's; past the
 * first 256 columns each evaluated again where the row block's tiles are not kept */
static size_t keep_tiles(size_t t,int kept) {
  size_t i0,n=0;
  for (i0=0;i0<t;i0+=32) {
    size_t end=i0+32<t ? i0+32 : t,tiles=(end+63)/64;
    n+=tiles+(kept || tiles<=256/64 ? 0 : tiles-256/64);
  }
  return n;
}
static void attention_keep(void) {
#ifdef BCIR_ATT_CONTIG_FROM
  const size_t from=BCIR_ATT_CONTIG_FROM;
#else
  const size_t from=1024;
#endif
  const size_t ts[2]={from,from+33},b=1,nh=2,nk=1,d=4;
  static const unsigned contracts[]={0u,BCIR_TENSOR_ATTENTION_FUSED};
  static _Alignas(max_align_t) unsigned char pool_memory[16384];
  bcir_tensor_pool *pools[2]={NULL,NULL};
  size_t si,c,i,pi,arm,cases=0,apart=0;
  {
    size_t need=bcir_tensor_pool_size(4);
    if (need && need<=sizeof(pool_memory)) pools[1]=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),4);
  }
  for (si=0;si<2;si++) {
    size_t t=ts[si],nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t,most=bcir_tensor_attention_workspace(b,t,nh,nk,d);
    float *q,*k,*v,*dy,*p0,*pw,*y,*dq[3],*dk[3],*dv[3];
    unsigned char *raw,*ws;
    if (DT_TSAN && t>=1024) continue;
    q=malloc(nq*sizeof(float)); k=malloc(nkv*sizeof(float)); v=malloc(nkv*sizeof(float));
    dy=malloc(nq*sizeof(float)); y=malloc(nq*sizeof(float)); p0=malloc(np*sizeof(float)); pw=malloc(np*sizeof(float));
    raw=malloc(most+128);
    assert(q && k && v && dy && y && p0 && pw && raw);
    for (arm=0;arm<3;arm++) {
      dq[arm]=malloc(nq*sizeof(float)); dk[arm]=malloc(nkv*sizeof(float)); dv[arm]=malloc(nkv*sizeof(float));
      assert(dq[arm] && dk[arm] && dv[arm]);
    }
    ws=raw+(64-(uintptr_t)raw%64)%64;
    for (i=0;i<nq;i++) { q[i]=sinf((float)i*0.37f)*1.7f; dy[i]=cosf((float)i*0.11f)*0.5f; }
    for (i=0;i<nkv;i++) { k[i]=cosf((float)i*0.53f)*1.3f; v[i]=sinf((float)i*0.29f)-0.1f; }
    for (c=0;c<sizeof(contracts)/sizeof(contracts[0]);c++) {
      bcir_tensor_attention_contract_scan_pool(NULL,contracts[c],b,t,nh,nk,d,q,k,v,p0,y,NULL);
      for (pi=0;pi<2;pi++) {
        size_t own=bcir_tensor_attention_pool_workspace(b,t,nh,nk,d,bcir_tensor_pool_threads(pools[pi]));
        if (pi && !pools[pi]) continue;
        assert(own>0 && own<=most+64);
        for (arm=0;arm<3;arm++) {  /* lent none, a byte short of the pool's need, the need */
          size_t bytes=arm==2 ? own : arm==1 ? own-1 : 0;
          memcpy(pw,p0,np*sizeof(float));
          memset(dq[arm],0xff,nq*sizeof(float)); memset(dk[arm],0xff,nkv*sizeof(float));
          memset(dv[arm],0xff,nkv*sizeof(float));
#ifdef BCIR_DT_ATTENTION_PROBE
          (void)bcir_tensor_attention_tiles_probe();
#endif
          assert(bcir_tensor_attention_backward_contract_variant_ws_pool(pools[pi],0,contracts[c],b,t,nh,nk,d,q,k,v,pw,
              dy,dq[arm],dk[arm],dv[arm],arm ? ws : NULL,bytes));
#ifdef BCIR_DT_ATTENTION_PROBE
          assert(bcir_tensor_attention_tiles_probe()==b*nh*keep_tiles(t,arm==2));
#endif
          if (arm) assert(!memcmp(dq[0],dq[arm],nq*sizeof(float)) && !memcmp(dk[0],dk[arm],nkv*sizeof(float)) &&
              !memcmp(dv[0],dv[arm],nkv*sizeof(float)));
        }
        cases++;
        apart+=keep_tiles(t,1)!=keep_tiles(t,0);
      }
    }
    free(q); free(k); free(v); free(dy); free(y); free(p0); free(pw); free(raw);
    for (arm=0;arm<3;arm++) { free(dq[arm]); free(dk[arm]); free(dv[arm]); }
  }
  if (pools[1]) bcir_tensor_pool_stop(pools[1]);
  assert(cases>0 && (from>256+32 ? apart>0 : 1));
  printf("  PASS BWD-KEEP: %zu backwards at the copies' threshold %zu and past it the same bits lent none, a byte "
      "short and the need, each path's tiles counted (%zu of them at a context where keeping saves a second "
      "evaluation)\n",cases,from,apart);
}
int main(void) {
  gemm_contract(); attention_reference(); attention_fused_reference(); adamw_reference();
  finite_differences(); refusals();
  learning(); pool_identity(); pool_claims(); reduction_references(); scan_references();
  gemm_fused_contract(); gemm_kblock_contract(); exp_contract(); fused_passes();
  writer_verdicts(); update_pairs(); gemm_bf16_contract(); trainer_bf16(); trainer_attn();
  panels_contract(); trainer_panels(); facts_run(); attention_bf16_reference(); trainer_attnbf16();
  trainer_workspace_plan(); trainer_recompute(); pool_caps(); dispatch_entries();
  attention_scratch(); attention_contig(); attention_strips(); attention_lower(); attention_keep();
  trainer_infer();
  trainer_lower();
#ifdef BCIR_DT_DIVISION_PROBE
  division_probe();
#endif
  return 0;
}
