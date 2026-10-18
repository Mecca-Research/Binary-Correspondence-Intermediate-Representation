/* Old-against-new differential for NDT-SCAN-W: the forward's verdict on its activations, taken by
 * their writers, against the one scan of the whole arena after the forward. For three specs (the C
 * harness's tied fixture, the same untied, and a wider one: 32 wide, 4 query heads over 2 KV heads,
 * two sequences of nine), the twelve contracts and pools of one, two and four threads, each of a
 * set of crafted states runs the public forward, the public backward and a one-step run, and every
 * byte the API can observe is hashed (FNV-1a): return codes, the completed count and event, the
 * loss, and the activations, gradients, weights, moments and step state after each call. The states:
 * the finite control; layer 0's key at +inf for one position with every score of it -inf (the
 * values zero: only the keys are not finite); an untied head (or a tied embedding) at FLT_MAX; the
 * final gamma at FLT_MAX; layer 0's up, layer 1's o and layer 0's v projections at FLT_MAX; layer
 * 0's q and k projections at 1e30 (scores past FLT_MAX); and a provider's SiLU that answers 0 for an
 * infinite or NaN gate, finite and with layer 0's gate overflowing. The tree before the slice and
 * the tree with it must print the same verdict table and the same digest.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I TREE verdicts.c TREE/bcir_decoder_train.c TREE/bcir_tensor.c -lm
 *   ./a.out
 */
#include "bcir_decoder_train.h"
#include "bcir_tensor.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t digest=1469598103934665603ull;
static void put(const void *p, size_t n) {
  const unsigned char *b=(const unsigned char *)p; size_t i;
  for (i=0;i<n;i++) { digest^=b[i]; digest*=1099511628211ull; }
}
static void put_i(long long v) { put(&v,sizeof(v)); }
static void hiding_silu(void *ctx, size_t n, const float *x, float *y) {
  size_t i; (void)ctx;
  for (i=0;i<n;i++) y[i]=isfinite(x[i]) ? x[i]/(1.0f+expf(-x[i])) : 0.0f;
}
typedef struct layout { size_t d,k,f,v,l,lp,n; } layout;
static layout layout_of(const bcir_decoder_spec *sp) {
  layout o; o.d=sp->width; o.k=(size_t)sp->kvheads*(sp->width/sp->heads); o.f=sp->ff; o.v=sp->vocab;
  o.l=sp->layers; o.lp=2*o.d+2*o.d*o.d+2*o.d*o.k+3*o.d*o.f; o.n=(size_t)sp->batch*sp->time; return o;
}
/* offsets within layer l: norm1 0, q d, k d+d*d, v +k*d, o +k*d, norm2 +d*d, gate +d, up +f*d, down +f*d */
static size_t layer_at(const layout *o, size_t l) { return o->v*o->d+l*o->lp; }
static size_t q_at(const layout *o, size_t l) { return layer_at(o,l)+o->d; }
static size_t k_at(const layout *o, size_t l) { return q_at(o,l)+o->d*o->d; }
static size_t v_at(const layout *o, size_t l) { return k_at(o,l)+o->k*o->d; }
static size_t o_at(const layout *o, size_t l) { return v_at(o,l)+o->k*o->d; }
static size_t norm2_at(const layout *o, size_t l) { return o_at(o,l)+o->d*o->d; }
static size_t gate_at(const layout *o, size_t l) { return norm2_at(o,l)+o->d; }
static size_t up_at(const layout *o, size_t l) { return gate_at(o,l)+o->f*o->d; }
static size_t final_at(const layout *o) { return o->v*o->d+o->l*o->lp; }
static void fill(float *w, size_t n, const layout *o) {
  size_t i,l;
  for (i=0;i<n;i++) w[i]=0.1f*sinf((float)(i+1)*0.37f);
  for (l=0;l<o->l;l++) for (i=0;i<o->d;i++) { w[layer_at(o,l)+i]=1.0f; w[norm2_at(o,l)+i]=1.0f; }
  for (i=0;i<o->d;i++) w[final_at(o)+i]=1.0f;
}
/* every weight zero but the norms; token 3 = e2 + 0.1 e6, token 5 = e6; layer 0's queries' channel
 * hd/2-1 (the slowest RoPE pair) -1 on dimension 6 for every head of KV head 0, and its key's channel
 * hd/2-1 `key` on dimension 2: at FLT_MAX the key of every token-3 position is +inf in that channel
 * and its pair's, and every score of it -inf */
static void hidden_key(float *w, size_t n, const layout *o, const bcir_decoder_spec *sp, float key) {
  size_t i,l,h,hd=sp->width/sp->heads,c=hd/2-1,group=sp->heads/sp->kvheads;
  memset(w,0,n*sizeof(float));
  w[3*o->d+2]=1.0f; w[3*o->d+6]=0.1f; w[5*o->d+6]=1.0f;
  for (l=0;l<o->l;l++) for (i=0;i<o->d;i++) { w[layer_at(o,l)+i]=1.0f; w[norm2_at(o,l)+i]=1.0f; }
  for (i=0;i<o->d;i++) w[final_at(o)+i]=1.0f;
  for (h=0;h<group;h++) w[q_at(o,0)+(h*hd+c)*o->d+6]=-1.0f;
  w[k_at(o,0)+c*o->d+2]=key;
}
enum { C_CONTROL, C_HIDDEN_KEY, C_HEAD, C_FINAL, C_UP, C_O, C_V, C_SCORES, C_PROVIDER_FINITE,
  C_PROVIDER_GATE, C_COUNT };
static const char *names[C_COUNT]={"control","hidden-key","head","final-gamma","up","o","v","scores",
  "provider-finite","provider-gate"};
int main(void) {
  static const bcir_decoder_spec specs[3]={
    {16,8,2,1,2,16,1,4,1,10000.0,1e-6},{16,8,2,1,2,16,1,4,0,10000.0,1e-6},
    {16,32,4,2,2,64,2,9,0,10000.0,1e-6}};
  static _Alignas(max_align_t) unsigned char pool_memory[8192];
  static const unsigned pools[3]={1,2,4};
  bcir_decoder_adamw opt={0.01,0.9,0.95,1e-8,0.01,1.0};
  unsigned table[3][C_COUNT][3]; size_t sp,c,p,i,contract;
  memset(table,0,sizeof(table));
  for (sp=0;sp<3;sp++) for (contract=0;contract<16;contract++) {
    bcir_decoder_state s; layout o=layout_of(&specs[sp]); size_t n,na,ntok;
    float *w,*g,*m,*v,*a,*sc; uint32_t ids[18],tg[18];
    if ((contract&BCIR_DT_CONTRACT_FUSED) && (contract&BCIR_DT_CONTRACT_KBLOCK)) continue;  /* 12 of 16 */
    memset(&s,0,sizeof(s));
    if (bcir_decoder_make_plan(&specs[sp],&s.plan)) { fprintf(stderr,"plan refused\n"); return 2; }
    n=s.plan.parameters; na=s.plan.activations; ntok=o.n;
    w=calloc(n,4); g=calloc(n,4); m=calloc(n,4); v=calloc(n,4); a=calloc(na,4); sc=calloc(s.plan.scratch,4);
    if (!w || !g || !m || !v || !a || !sc) { perror("calloc"); return 2; }
    for (i=0;i<ntok;i++) { ids[i]=(i%specs[sp].time)==1 ? 3u : 5u; tg[i]=(uint32_t)((i*7+3)%16); }
    for (p=0;p<3;p++) {
      bcir_tensor_pool *pool=NULL;
      if (pools[p]>1) {
        if (!bcir_tensor_pool_size(pools[p])) continue;
        pool=bcir_tensor_pool_init(pool_memory,sizeof(pool_memory),pools[p]);
        if (!pool) { fprintf(stderr,"pool refused\n"); return 2; }
      }
      for (c=0;c<C_COUNT;c++) {
        int fwd,bwd,run; double loss=0.0; bcir_decoder_event ev; size_t done=0;
        s.weights=w; s.gradients=g; s.moment1=m; s.moment2=v; s.activations=a; s.scratch=sc;
        s.weight_capacity=s.gradient_capacity=s.moment1_capacity=s.moment2_capacity=n;
        s.activation_capacity=na; s.scratch_capacity=s.plan.scratch; s.pool=pool;
        s.contract=(uint32_t)contract; s.step=0; s.beta1_power=s.beta2_power=1.0;
        s.provider.ctx=NULL; s.provider.mm=NULL; s.provider.silu=NULL;
        memset(g,0,n*4); memset(m,0,n*4); memset(v,0,n*4); memset(a,0,na*4); memset(&ev,0,sizeof(ev));
        fill(w,n,&o);
        switch (c) {
          case C_HIDDEN_KEY: hidden_key(w,n,&o,&specs[sp],FLT_MAX); break;
          case C_HEAD:
            if (specs[sp].tied) for (i=0;i<o.v*o.d;i++) w[i]=FLT_MAX;
            else for (i=0;i<o.v*o.d;i++) w[final_at(&o)+o.d+i]=FLT_MAX;
            break;
          case C_FINAL: for (i=0;i<o.d;i++) w[final_at(&o)+i]=FLT_MAX; break;
          case C_UP: for (i=0;i<o.f*o.d;i++) w[up_at(&o,0)+i]=FLT_MAX; break;
          case C_O: for (i=0;i<o.d*o.d;i++) w[o_at(&o,1)+i]=FLT_MAX; break;
          case C_V: for (i=0;i<o.k*o.d;i++) w[v_at(&o,0)+i]=FLT_MAX; break;
          case C_SCORES:
            for (i=0;i<o.d*o.d;i++) w[q_at(&o,0)+i]=1e30f;
            for (i=0;i<o.k*o.d;i++) w[k_at(&o,0)+i]=1e30f;
            break;
          case C_PROVIDER_FINITE: s.provider.silu=hiding_silu; break;
          case C_PROVIDER_GATE:
            s.provider.silu=hiding_silu;
            for (i=gate_at(&o,0);i<gate_at(&o,0)+3*o.f*o.d;i++) w[i]=0.0f;
            w[norm2_at(&o,0)]=1e10f; w[gate_at(&o,0)]=FLT_MAX;
            break;
          default: break;
        }
        fwd=bcir_decoder_forward(&s,ids,ntok); put_i(fwd); put(a,na*4);
        bwd=bcir_decoder_loss_backward(&s,ids,tg,ntok,0,&loss); put_i(bwd); put(&loss,sizeof(loss));
        put(g,n*4); put(a,na*4);
        run=bcir_decoder_run(&s,&opt,ids,tg,ntok,NULL,1,&ev,1,&done);
        put_i(run); put_i((long long)done); put(&ev,sizeof(ev)); put(w,n*4); put(m,n*4); put(v,n*4);
        put(g,n*4); put(&s.step,sizeof(s.step)); put(&s.beta1_power,8); put(&s.beta2_power,8);
        table[sp][c][fwd==BCIR_DT_NUMERIC ? 1 : fwd==BCIR_DT_OK ? 0 : 2]++;
        /* the backward runs the same forward: its verdict on the activations is the forward's */
        if (fwd==BCIR_DT_NUMERIC && bwd!=BCIR_DT_NUMERIC) { fprintf(stderr,"the backward admitted what the forward refused\n"); return 3; }
      }
      bcir_tensor_pool_stop(pool);
    }
    free(w); free(g); free(m); free(v); free(a); free(sc);
  }
  for (sp=0;sp<3;sp++) {
    printf("spec %zu:",sp);
    for (c=0;c<C_COUNT;c++) printf(" %s ok %u numeric %u other %u;",names[c],table[sp][c][0],table[sp][c][1],table[sp][c][2]);
    printf("\n");
  }
  printf("digest %016llx\n",(unsigned long long)digest);
  return 0;
}
