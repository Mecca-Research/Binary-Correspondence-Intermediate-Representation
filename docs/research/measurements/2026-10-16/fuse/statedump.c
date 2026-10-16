/* Old-against-new differential for NDT-FUSE and NDT-ATTN-6: one build of the trainer runs a fixed
 * schedule and writes every byte the API can observe -- return codes, completed counts, events,
 * losses and norms, and the weights, moments, gradients and step state after each call -- and the
 * tree before the slices and the tree with them must write the same bytes. For three small specs
 * (tied, untied, a wider one), the twelve contracts and pools of one, two and four threads: a run
 * of every step in one call; the same steps one call at a time; the public backward over a
 * gradient full of NaN, -0.0 and stale values, accumulating and not, and the public update; a run
 * of two micro-batches a step; and the refusals -- a NaN weight, an invalid beta power, and a
 * crafted state whose activations are finite and whose gradient overflows (layer 0's up projection
 * x1e18, its down projection x1e-18, the final gamma 1e24), alone, with an invalid power and with
 * two micro-batches. `lm` adds the 9.84M spec on a pool of four, hashed (FNV-1a) instead of written.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I TREE statedump.c TREE/bcir_decoder_train.c TREE/bcir_tensor.c -lm
 *   ./a.out OUT [lm]
 */
#include "bcir_decoder_train.h"
#include "bcir_tensor.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *out; static int hashing; static uint64_t digest=1469598103934665603ull;
static void put(const void *p, size_t n) {
  if (hashing) { const unsigned char *b=(const unsigned char *)p; size_t i; for (i=0;i<n;i++) { digest^=b[i]; digest*=1099511628211ull; } return; }
  if (fwrite(p,1,n,out)!=n) { perror("fwrite"); exit(2); }
}
static void put_i(long long v) { put(&v,sizeof(v)); }

typedef struct buf { float *w,*g,*m,*v,*a,*s; size_t n; } buf;
static buf alloc_state(bcir_decoder_state *s, const bcir_decoder_spec *spec) {
  buf b; memset(s,0,sizeof(*s));
  if (bcir_decoder_make_plan(spec,&s->plan)) { fprintf(stderr,"plan refused\n"); exit(2); }
  b.n=s->plan.parameters;
  b.w=calloc(b.n,4); b.g=calloc(b.n,4); b.m=calloc(b.n,4); b.v=calloc(b.n,4);
  b.a=calloc(s->plan.activations,4); b.s=calloc(s->plan.scratch,4);
  if (!b.w || !b.g || !b.m || !b.v || !b.a || !b.s) { perror("calloc"); exit(2); }
  s->weights=b.w; s->gradients=b.g; s->moment1=b.m; s->moment2=b.v; s->activations=b.a; s->scratch=b.s;
  s->weight_capacity=s->gradient_capacity=s->moment1_capacity=s->moment2_capacity=b.n;
  s->activation_capacity=s->plan.activations; s->scratch_capacity=s->plan.scratch;
  s->beta1_power=s->beta2_power=1.0;
  return b;
}
static void free_state(buf *b) { free(b->w); free(b->g); free(b->m); free(b->v); free(b->a); free(b->s); }
static void init_weights(buf *b, const bcir_decoder_spec *spec) {
  size_t i; for (i=0;i<b->n;i++) b->w[i]=0.05f*sinf((float)(i+1)*0.37f);
  (void)spec;
}
static void dump_state(const bcir_decoder_state *s, const buf *b) {
  put(b->w,b->n*4); put(b->g,b->n*4); put(b->m,b->n*4); put(b->v,b->n*4);
  put(&s->step,sizeof(s->step)); put(&s->beta1_power,8); put(&s->beta2_power,8);
}
static void stream(uint32_t *x, uint32_t *y, size_t n, uint32_t vocab, unsigned seed) {
  size_t i; for (i=0;i<n;i++) { x[i]=(uint32_t)((i*37+11+seed)%vocab); y[i]=(uint32_t)((i*29+5+3*seed)%vocab); }
}

static const uint32_t CONTRACTS[]={0,1,2,3,4,6,8,9,10,11,12,14};
/* The small tied fixture's weights with layer 0's up projection scaled by 1e18 (one sign), its down
 * projection by 1e-18 and the final norm's gamma at 1e24: the activations finite, a gradient not. */
static void overflowing(buf *b) {
  size_t i,off=128;
  for (i=0;i<b->n;i++) b->w[i]=0.1f*sinf((float)(i+1)*0.37f);
  for (i=0;i<8;i++) b->w[off+i]=1.0f;
  off+=8+64+32+32+64; for (i=0;i<8;i++) b->w[off+i]=1.0f;
  off+=8;
  for (i=0;i<128;i++) b->w[off+128+i]=fabsf(b->w[off+128+i])*1e18f;
  for (i=0;i<128;i++) b->w[off+256+i]*=1e-18f;
  off+=3*128; for (i=0;i<8;i++) b->w[off+i]=1.0f;
  off+=8+64+32+32+64; for (i=0;i<8;i++) b->w[off+i]=1.0f; off+=8+3*128; for (i=0;i<8;i++) b->w[off+i]=1e24f;
}

/* One schedule on one spec: for each contract and pool size, a run of `steps` in one call, the same
 * steps one call at a time (fresh state), a run with accumulation 2, the public backward and update
 * with stale gradients, and the refusals (non-finite weight, non-finite gradient by overflow if one
 * is found, invalid beta powers alone and with it). */
static void schedule(const bcir_decoder_spec *spec, unsigned steps, const unsigned *threads, size_t nthreads) {
  size_t ci,ti; unsigned i; static _Alignas(max_align_t) unsigned char mem[1<<16];
  size_t tokens=(size_t)spec->batch*spec->time;
  uint32_t *x=malloc(tokens*4*steps*2),*y=malloc(tokens*4*steps*2);
  bcir_decoder_event *ev=malloc(sizeof(*ev)*steps*2); size_t done;
  bcir_decoder_adamw opt={2e-3,0.9,0.95,1e-8,0.01,1.0};
  stream(x,y,tokens*steps*2,spec->vocab,0);
  for (ci=0;ci<sizeof(CONTRACTS)/sizeof(CONTRACTS[0]);ci++) for (ti=0;ti<nthreads;ti++) {
    bcir_decoder_state s; buf b; bcir_tensor_pool *pool=NULL; int rc; double loss,norm;
    if (threads[ti]>1) {
      size_t need=bcir_tensor_pool_size(threads[ti]);
      if (!need || need>sizeof(mem)) continue;
      pool=bcir_tensor_pool_init(mem,sizeof(mem),threads[ti]);
    }
    put_i((long long)CONTRACTS[ci]); put_i(threads[ti]);
    /* a run of every step in one call */
    b=alloc_state(&s,spec); init_weights(&b,spec); s.contract=CONTRACTS[ci]; s.pool=pool;
    memset(ev,0,sizeof(*ev)*steps*2);
    rc=bcir_decoder_run(&s,&opt,x,y,tokens*steps,NULL,steps,ev,steps,&done);
    put_i(rc); put_i((long long)done); put(ev,sizeof(*ev)*steps); dump_state(&s,&b);
    /* the same steps one call at a time, then stale NaN gradients under the public backward */
    free_state(&b); b=alloc_state(&s,spec); init_weights(&b,spec); s.contract=CONTRACTS[ci]; s.pool=pool;
    for (i=0;i<steps;i++) {
      memset(ev,0,sizeof(*ev));
      rc=bcir_decoder_run(&s,&opt,x+i*tokens,y+i*tokens,tokens,NULL,1,ev,1,&done);
      put_i(rc); put_i((long long)done); put(ev,sizeof(*ev));
    }
    dump_state(&s,&b);
    { size_t k; for (k=0;k<b.n;k++) b.g[k]=k%3 ? NAN : (k%2 ? -0.0f : 1000.0f); }
    rc=bcir_decoder_loss_backward(&s,x,y,tokens,0,&loss); put_i(rc); put(&loss,8); dump_state(&s,&b);
    rc=bcir_decoder_loss_backward(&s,x+tokens,y+tokens,tokens,1,&loss); put_i(rc); put(&loss,8); dump_state(&s,&b);
    rc=bcir_decoder_update(&s,&opt,0.5,&norm); put_i(rc); put(&norm,8); dump_state(&s,&b);
    /* accumulation 2 in one call */
    free_state(&b); b=alloc_state(&s,spec); init_weights(&b,spec); s.contract=CONTRACTS[ci]; s.pool=pool;
    memset(ev,0,sizeof(*ev)*steps*2);
    rc=bcir_decoder_run_accum(&s,&opt,x,y,tokens*steps*2,NULL,steps,2,ev,steps,&done);
    put_i(rc); put_i((long long)done); put(ev,sizeof(*ev)*steps); dump_state(&s,&b);
    /* refusals: a NaN weight; invalid beta powers; both kinds of gradient trouble searched */
    free_state(&b); b=alloc_state(&s,spec); init_weights(&b,spec); s.contract=CONTRACTS[ci]; s.pool=pool;
    b.w[b.n/2]=NAN; rc=bcir_decoder_run(&s,&opt,x,y,tokens*2,NULL,2,ev,2,&done);
    put_i(rc); put_i((long long)done); dump_state(&s,&b);
    init_weights(&b,spec); s.beta1_power=2.0;
    rc=bcir_decoder_run(&s,&opt,x,y,tokens*2,NULL,2,ev,2,&done); put_i(rc); put_i((long long)done); dump_state(&s,&b);
    {
      /* the final norm's gamma (the last `width` parameters when tied) scaled up until a backward
       * refuses: the first scale at which it does, and what run and run-with-bad-powers report */
      int e,found=0; size_t k,wd=spec->width;
      for (e=4;e<=38 && !found;e++) {
        init_weights(&b,spec); s.beta1_power=s.beta2_power=1.0; s.step=0;
        for (k=b.n-wd;k<b.n;k++) b.w[k]=powf(10.0f,(float)e)*(k%2 ? 1.0f : -1.0f);
        rc=bcir_decoder_loss_backward(&s,x,y,tokens,0,&loss);
        if (rc==BCIR_DT_NUMERIC && bcir_decoder_forward(&s,x,tokens)==BCIR_DT_OK) {
          found=1; put_i(e);
          init_weights(&b,spec); for (k=b.n-wd;k<b.n;k++) b.w[k]=powf(10.0f,(float)e)*(k%2 ? 1.0f : -1.0f);
          rc=bcir_decoder_run(&s,&opt,x,y,tokens,NULL,1,ev,1,&done); put_i(rc); put_i((long long)done); dump_state(&s,&b);
          s.beta1_power=2.0;
          rc=bcir_decoder_run(&s,&opt,x,y,tokens,NULL,1,ev,1,&done); put_i(rc); put_i((long long)done); dump_state(&s,&b);
          s.beta1_power=1.0;
        }
      }
      if (!found) put_i(-1);
      fprintf(stderr,"contract %u threads %u: overflow scale %s\n",CONTRACTS[ci],threads[ti],found ? "found" : "none");
    }
    if (spec->vocab==16 && spec->tied && spec->width==8) {
      /* the crafted overflow: forward finite, backward refused; run, run with an invalid power, and
       * accumulation 2 report the backward's verdict and leave the state */
      int fwd;
      overflowing(&b); s.beta1_power=s.beta2_power=1.0; s.step=0;
      fwd=bcir_decoder_forward(&s,x,tokens); put_i(fwd);
      rc=bcir_decoder_loss_backward(&s,x,y,tokens,0,&loss); put_i(rc);
      if (fwd!=BCIR_DT_OK || rc!=BCIR_DT_NUMERIC) { fprintf(stderr,"the crafted overflow state does not overflow\n"); exit(3); }
      overflowing(&b); rc=bcir_decoder_run(&s,&opt,x,y,tokens,NULL,1,ev,1,&done); put_i(rc); put_i((long long)done); dump_state(&s,&b);
      overflowing(&b); s.beta1_power=2.0;
      rc=bcir_decoder_run(&s,&opt,x,y,tokens,NULL,1,ev,1,&done); put_i(rc); put_i((long long)done); dump_state(&s,&b);
      s.beta1_power=1.0; overflowing(&b);
      rc=bcir_decoder_run_accum(&s,&opt,x,y,tokens*2,NULL,1,2,ev,1,&done); put_i(rc); put_i((long long)done); dump_state(&s,&b);
      fprintf(stderr,"contract %u threads %u: crafted overflow checked\n",CONTRACTS[ci],threads[ti]);
    }
    free_state(&b);
    bcir_tensor_pool_stop(pool);
  }
  free(x); free(y); free(ev);
}

int main(int argc, char **argv) {
  static const unsigned th[]={1,2,4};
  bcir_decoder_spec small={16,8,2,1,2,16,5,8,1,10000.0,1e-6},untied=small,wide={260,64,4,2,2,96,2,16,1,10000.0,1e-5};
  if (argc<2) { fprintf(stderr,"usage: %s OUT [lm]\n",argv[0]); return 2; }
  out=fopen(argv[1],"wb"); if (!out) { perror("fopen"); return 2; }
  untied.tied=0;
  schedule(&small,3,th,3);
  schedule(&untied,3,th,3);
  schedule(&wide,2,th,3);
  if (argc>2 && !strcmp(argv[2],"lm")) {
    bcir_decoder_spec lm={260,384,8,4,6,1024,8,128,1,10000.0,1e-5}; static const unsigned t4[]={4};
    hashing=1; schedule(&lm,2,t4,1); hashing=0;
    fprintf(out,"lm digest %016llx\n",(unsigned long long)digest);
  }
  fclose(out);
  return 0;
}
