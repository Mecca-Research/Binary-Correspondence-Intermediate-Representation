/* ATTN-RATE (EXP-LANES): the elementwise kernels' outputs over lengths that end mid-vector and
 * mid-block and inputs that reach the exponential's edges (0, +-0, subnormal results, overflow,
 * +-inf, NaN), each instance and the pool split; two FNV-1a digests a line -- `v`, every NaN read as
 * one quiet NaN (the runtime's rule: which NaN an operation on NaNs gives is the compiled code's, not
 * the contract's), and `raw`, the bytes as written: two builds are one arithmetic when every v
 * matches. */
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fnv(const void *b,size_t n,uint64_t h) { const unsigned char *c=b; while (n--) { h^=*c++; h*=1099511628211ull; } return h; }
static int RAW;  /* 0: every NaN hashed as 0x7fc00000 */
static uint64_t fnvf(const float *f,size_t n,uint64_t h) {
  size_t i; const float q=NAN;
  for (i=0;i<n;i++) h=fnv(!RAW && isnan(f[i]) ? &q : &f[i],4,h);
  return h;
}
int main(int argc,char **argv) {
  RAW=argc>1 && !strcmp(argv[1],"raw");
  static const size_t len[]={1,7,15,16,17,31,33,63,64,65,127,129,255,257,1000,4097,20000,65599};
  static _Alignas(64) unsigned char pool_memory[1<<16];
  const float edge[]={0.0f,-0.0f,1.0f,-1.0f,88.7f,89.0f,-87.3f,-100.0f,-104.0f,-150.0f,1e-30f,-1e-30f,30.0f,-30.0f,
    INFINITY,-INFINITY,NAN,3.4e38f,-3.4e38f};
  size_t li,i,ne=sizeof edge/sizeof edge[0]; int variant;
  bcir_tensor_pool *pool=bcir_tensor_pool_size(3) ? bcir_tensor_pool_init(pool_memory,sizeof pool_memory,3) : NULL;
  printf("variants %d\n",bcir_tensor_mm_variants());
  for (li=0;li<sizeof len/sizeof len[0];li++) {
    size_t n=len[li]; float *x=malloc(n*4),*u=malloc(n*4),*dy=malloc(n*4),*y=calloc(n,4),*z=calloc(n,4),*w=calloc(n,4);
    for (i=0;i<n;i++) { x[i]=sinf((float)i*0.37f)*14.0f; u[i]=cosf((float)i*0.53f)*1.3f; dy[i]=sinf((float)i*0.29f)-0.1f; }
    for (i=0;i<n && i<ne;i++) x[(i*7919u)%n]=edge[i];
    bcir_tensor_exp(n,x,y); printf("n%zu exp %016llx\n",n,(unsigned long long)fnvf(y,n,1469598103934665603ull));
    for (variant=1;variant<=4;variant*=2) {
      int ran;
      memset(y,0,n*4);
      ran=bcir_tensor_exp_variant(variant,n,x,y);  /* run before the digest reads y: arguments are unsequenced */
      printf("n%zu exp_variant%d ran%d %016llx\n",n,variant,ran,(unsigned long long)fnvf(y,n,1469598103934665603ull));
    }
    bcir_tensor_silu_v2_pool(NULL,n,x,y); printf("n%zu silu_v2 %016llx\n",n,(unsigned long long)fnvf(y,n,1469598103934665603ull));
    bcir_tensor_silu_v2_pool(pool,n,x,z); printf("n%zu silu_v2_pool %016llx\n",n,(unsigned long long)fnvf(z,n,1469598103934665603ull));
    bcir_tensor_silu_mul_v2_pool(NULL,n,x,u,y); printf("n%zu silu_mul_v2 %016llx\n",n,(unsigned long long)fnvf(y,n,1469598103934665603ull));
    bcir_tensor_silu_mul_v2_pool(pool,n,x,u,z); printf("n%zu silu_mul_v2_pool %016llx\n",n,(unsigned long long)fnvf(z,n,1469598103934665603ull));
    bcir_tensor_swiglu_backward_v2_pool(NULL,n,x,u,dy,y,z);
    printf("n%zu swiglu_vjp_v2 %016llx\n",n,(unsigned long long)fnvf(z,n,fnvf(y,n,1469598103934665603ull)));
    bcir_tensor_swiglu_backward_v2_pool(pool,n,x,u,dy,w,z);
    printf("n%zu swiglu_vjp_v2_pool %016llx\n",n,(unsigned long long)fnvf(z,n,fnvf(w,n,1469598103934665603ull)));
    free(x); free(u); free(dy); free(y); free(z); free(w);
  }
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
