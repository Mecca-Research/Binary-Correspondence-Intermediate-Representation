/* EXP-SCALEF's exhaustive witness: a tensor unit's AVX-512 exponential and v2 SiLU forms against their
 * scalar definitions (bcir_tensor_expf, the harness's silu2_ref and SwiGLU VJP loops) over every binary32
 * input, bit for bit -- in place, apart, an output aliasing an input, and one overlapping an input by one
 * float (the element loop's order). Links against the unit under test.
 * The exponential's reference is the unit's portable instance (exp1 an element, bcir_tensor_exp_variant 1, held to
 * bcir_tensor_expf by the harness); SiLU's is scalar here, over every input; SiLU times u and the SwiGLU VJP (the same
 * sigmoid, then their products) every 16th block of a part, the overlapping outputs every 64th.
 * usage: exp_exhaustive PART PARTS [STRIDE]  -- blocks of 2^16 inputs, block b when b % PARTS == PART (every
 *        STRIDE-th such block: a smoke test); prints one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float sg_ref(float x) { float e=bcir_tensor_expf(x<0.0f ? x : -x),s=1.0f/(1.0f+e); return x>=0.0f ? s : e*s; }
static uint64_t payloads;  /* the SiLU forms at a NaN input: both NaN, payloads apart (the compiler's operand order) */
static int same(const float *a, const float *b, size_t n, uint64_t *bad, const float *xs, const char *what) {
  size_t i; int any=0, exact=!strncmp(what,"exp",3);
  for (i=0;i<n;i++) if (memcmp(a+i,b+i,4)) {
    if (!exact && xs[i]!=xs[i] && a[i]!=a[i] && b[i]!=b[i]) { payloads++; continue; }
    if (*bad<4) fprintf(stderr,"%s mismatch at x=%a: want %a got %a\n",what,xs[i],a[i],b[i]);
    (*bad)++; any=1;
  }
  return any;
}
int main(int argc, char **argv) {
  enum { N=1<<16 };
  static float xs[N+1],u[N],dy[N],want[N],got[N],w2[N],g2[N],buf[N+1];
  uint64_t part,parts,stride,blk,swept=0,bad_exp=0,bad_silu=0,bad_mul=0,bad_swiglu=0,bad_alias=0,products=0,overlaps=0;
  size_t i;
  if (argc<3) { fprintf(stderr,"usage: exp_exhaustive PART PARTS [STRIDE]\n"); return 2; }
  part=strtoull(argv[1],NULL,10); parts=strtoull(argv[2],NULL,10); stride=argc>3 ? strtoull(argv[3],NULL,10) : 1;
  if (!parts || part>=parts || !stride) return 2;
  if (!(bcir_tensor_mm_variants()&4)) { printf("{\"skipped\":\"no AVX-512 instance\"}\n"); return 0; }
  for (i=0;i<N;i++) { u[i]=1.25f+(float)(i%7)*0.5f-(float)(i%3); dy[i]=0.75f-(float)(i%5)*0.375f; }
  for (blk=part;blk<(1ull<<32)/N;blk+=parts*stride) {
    for (i=0;i<N;i++) { uint32_t b=(uint32_t)(blk*N+i); memcpy(xs+i,&b,4); }
    /* the exponential: apart and in place */
    if (bcir_tensor_exp_variant(1,N,xs,want)!=1 || bcir_tensor_exp_variant(4,N,xs,got)!=4) return 3;
    same(want,got,N,&bad_exp,xs,"exp");
    memcpy(buf,xs,sizeof(float)*N); bcir_tensor_exp_variant(4,N,buf,buf);
    same(want,buf,N,&bad_exp,xs,"exp in place");
    /* SiLU v2 */
    for (i=0;i<N;i++) want[i]=xs[i]*sg_ref(xs[i]);
    bcir_tensor_silu_v2_pool(NULL,N,xs,got); same(want,got,N,&bad_silu,xs,"silu");
    memcpy(buf,xs,sizeof(float)*N); bcir_tensor_silu_v2_pool(NULL,N,buf,buf); same(want,buf,N,&bad_silu,xs,"silu in place");
    if ((blk/parts)%16) { swept+=N; continue; }
    products+=N;
    /* SiLU times u: apart, y the gate, y the up */
    for (i=0;i<N;i++) { float s=xs[i]*sg_ref(xs[i]); want[i]=s*u[i]; }
    bcir_tensor_silu_mul_v2_pool(NULL,N,xs,u,got); same(want,got,N,&bad_mul,xs,"silu_mul");
    memcpy(buf,xs,sizeof(float)*N); bcir_tensor_silu_mul_v2_pool(NULL,N,buf,u,buf); same(want,buf,N,&bad_mul,xs,"silu_mul y=x");
    memcpy(buf,u,sizeof(float)*N); bcir_tensor_silu_mul_v2_pool(NULL,N,xs,buf,buf); same(want,buf,N,&bad_mul,xs,"silu_mul y=u");
    /* the SwiGLU VJP v2: g the gate */
    for (i=0;i<N;i++) {
      float s=sg_ref(xs[i]);
      w2[i]=dy[i]*u[i]*(s+xs[i]*s*(1.0f-s)); want[i]=dy[i]*xs[i]*s;
    }
    bcir_tensor_swiglu_backward_v2_pool(NULL,N,xs,u,dy,g2,got);
    same(w2,g2,N,&bad_swiglu,xs,"swiglu dg"); same(want,got,N,&bad_swiglu,xs,"swiglu du");
    /* an output one float past its input: the element loop's order, each element reading the one before's result */
    if ((blk/parts)%64==0) {
      overlaps+=N-1;
      memcpy(buf,xs,sizeof(float)*N);
      for (i=0;i<N;i++) want[i]=buf[i];
      for (i=0;i+1<N;i++) want[i+1]=bcir_tensor_expf(want[i]);  /* y[i] = exp(x[i]) with y = x+1 */
      bcir_tensor_exp_variant(4,N-1,buf,buf+1);
      same(want+1,buf+1,N-1,&bad_alias,xs,"exp overlapping");
      memcpy(buf,xs,sizeof(float)*N);
      for (i=0;i<N;i++) want[i]=buf[i];
      for (i=0;i+1<N;i++) want[i+1]=want[i]*sg_ref(want[i]);
      bcir_tensor_silu_v2_pool(NULL,N-1,buf,buf+1);
      same(want+1,buf+1,N-1,&bad_alias,xs,"silu overlapping");
    }
    swept+=N;
  }
  printf("{\"part\":%llu,\"parts\":%llu,\"stride\":%llu,\"inputs\":%llu,\"product_inputs\":%llu,\"overlap_inputs\":%llu,"
      "\"exp\":%llu,\"silu\":%llu,\"silu_mul\":%llu,\"swiglu\":%llu,\"overlapping\":%llu,\"silu_nan_payloads_apart\":%llu}\n",(unsigned long long)part,
      (unsigned long long)parts,(unsigned long long)stride,(unsigned long long)swept,(unsigned long long)products,
      (unsigned long long)overlaps,(unsigned long long)bad_exp,(unsigned long long)bad_silu,
      (unsigned long long)bad_mul,(unsigned long long)bad_swiglu,(unsigned long long)bad_alias,(unsigned long long)payloads);
  return (bad_exp||bad_silu||bad_mul||bad_swiglu||bad_alias) ? 1 : 0;
}
