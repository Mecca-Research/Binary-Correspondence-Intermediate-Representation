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
 * bit against the naive loop over every transpose pair and the 32-wide tiles' edges. */
static void gemm_contract(void) {
  static float a[65*65],b[65*65],c[65*65],want[65*65];
  static const size_t shapes[][3]={{1,1,1},{3,33,5},{33,31,65},{32,32,32},{7,65,40},{64,1,33}};
  size_t s,i,j,l,ta,tb;
  for (i=0;i<65*65;i++) {
    a[i]=sinf((float)i*0.71f)*(i%3 ? 4.0f : 0.125f); b[i]=cosf((float)i*1.37f)-0.25f;
  }
  for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (ta=0;ta<2;ta++) for (tb=0;tb<2;tb++) {
    size_t m=shapes[s][0],n=shapes[s][1],k=shapes[s][2];
    float alpha=s%2 ? 0.75f : 1.0f,beta=s%3==1 ? 1.0f : s%3==2 ? 0.5f : 0.0f;
    for (i=0;i<m*n;i++) want[i]=c[i]=0.01f*(float)(i%17)-0.05f;
    for (i=0;i<m*n;i++) want[i]=beta==0.0f ? 0.0f : beta*want[i];
    for (i=0;i<m;i++) for (j=0;j<n;j++) for (l=0;l<k;l++) {
      float av=alpha*a[ta ? l*m+i : i*k+l];
      want[i*n+j]+=av*b[tb ? j*k+l : l*n+j];
    }
    bcir_tensor_mm(NULL,(int)ta,(int)tb,m,n,k,alpha,a,b,beta,c);
    assert(!memcmp(c,want,m*n*sizeof(float)));
  }
  puts("  PASS native GEMM contract: bit-identical to the ascending-k loop, every transpose");
}
int main(void) { gemm_contract(); finite_differences(); refusals(); learning(); return 0; }
