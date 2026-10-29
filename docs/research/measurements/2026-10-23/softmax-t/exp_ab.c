/* SOFTMAX-T's before/after witness: two tensor units (shared libraries, RTLD_LOCAL) on every binary32 input, their
 * exponential (each instance the host has), v2 SiLU, SiLU times u and SwiGLU VJP compared byte for byte -- the
 * claim "every value as it was" itself, independent of any reference's own NaN propagation. At a NaN input the
 * two outputs must both be NaN or both not ("a NaN a NaN", the harness's rule); a NaN's payload is counted apart
 * (the SiLU forms' payloads are the compiler's: its order of a product's operands).
 * usage: exp_ab BEFORE.so AFTER.so PART PARTS [STRIDE]  -- prints one JSON line */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*exp_fn)(int, size_t, const float *, float *);
typedef void (*silu_fn)(void *, size_t, const float *, float *);
typedef void (*mul_fn)(void *, size_t, const float *, const float *, float *);
typedef void (*swi_fn)(void *, size_t, const float *, const float *, const float *, float *, float *);
typedef int (*var_fn)(void);
struct unit { exp_fn ex; silu_fn si; mul_fn mu; swi_fn sw; var_fn vars; };
static int load(const char *path, struct unit *u) {
  void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if (!h) { fprintf(stderr,"%s\n",dlerror()); return 0; }
  u->ex=(exp_fn)dlsym(h,"bcir_tensor_exp_variant"); u->si=(silu_fn)dlsym(h,"bcir_tensor_silu_v2_pool");
  u->mu=(mul_fn)dlsym(h,"bcir_tensor_silu_mul_v2_pool"); u->sw=(swi_fn)dlsym(h,"bcir_tensor_swiglu_backward_v2_pool");
  u->vars=(var_fn)dlsym(h,"bcir_tensor_mm_variants");
  return u->ex && u->si && u->mu && u->sw && u->vars;
}
static uint64_t payloads;
static void cmp(int exact, const float *a, const float *b, const float *xs, size_t n, uint64_t *bad, uint64_t *nanbad) {
  size_t i;
  for (i=0;i<n;i++) if (memcmp(a+i,b+i,4)) {
    if (!exact && xs[i]!=xs[i] && a[i]!=a[i] && b[i]!=b[i]) payloads++;  /* both NaN: the payload alone */
    else if (xs[i]!=xs[i]) (*nanbad)++;
    else (*bad)++;
  }
}
int main(int argc, char **argv) {
  enum { N=1<<16 };
  static float xs[N],u[N],dy[N],a0[N],a1[N],b0[N],b1[N];
  struct unit A,B; uint64_t part,parts,stride,blk,swept=0,bad[4]={0},nanbad[4]={0}; int v,mask; size_t i;
  if (argc<5 || !load(argv[1],&A) || !load(argv[2],&B)) { fprintf(stderr,"usage: exp_ab BEFORE.so AFTER.so PART PARTS [STRIDE]\n"); return 2; }
  part=strtoull(argv[3],NULL,10); parts=strtoull(argv[4],NULL,10); stride=argc>5 ? strtoull(argv[5],NULL,10) : 1;
  if (!parts || part>=parts || !stride) return 2;
  mask=A.vars()&B.vars();
  for (i=0;i<N;i++) { u[i]=1.25f+(float)(i%7)*0.5f-(float)(i%3); dy[i]=0.75f-(float)(i%5)*0.375f; }
  for (blk=part;blk<(1ull<<32)/N;blk+=parts*stride) {
    for (i=0;i<N;i++) { uint32_t b=(uint32_t)(blk*N+i); memcpy(xs+i,&b,4); }
    for (v=1;v<=4;v*=2) {
      if (!(mask&v)) continue;
      A.ex(v,N,xs,a0); B.ex(v,N,xs,b0); cmp(1,a0,b0,xs,N,&bad[0],&nanbad[0]);  /* the exponential returns its NaN */
    }
    A.si(NULL,N,xs,a0); B.si(NULL,N,xs,b0); cmp(0,a0,b0,xs,N,&bad[1],&nanbad[1]);
    A.mu(NULL,N,xs,u,a0); B.mu(NULL,N,xs,u,b0); cmp(0,a0,b0,xs,N,&bad[2],&nanbad[2]);
    A.sw(NULL,N,xs,u,dy,a0,a1); B.sw(NULL,N,xs,u,dy,b0,b1); cmp(0,a0,b0,xs,N,&bad[3],&nanbad[3]); cmp(0,a1,b1,xs,N,&bad[3],&nanbad[3]);
    swept+=N;
  }
  printf("{\"part\":%llu,\"parts\":%llu,\"stride\":%llu,\"inputs\":%llu,\"instances\":%d,\"exp\":%llu,\"silu\":%llu,"
      "\"silu_mul\":%llu,\"swiglu\":%llu,\"at_nan\":{\"exp\":%llu,\"silu\":%llu,\"silu_mul\":%llu,\"swiglu\":%llu},"
      "\"nan_payloads_apart\":%llu}\n",
      (unsigned long long)part,(unsigned long long)parts,(unsigned long long)stride,(unsigned long long)swept,mask,
      (unsigned long long)bad[0],(unsigned long long)bad[1],(unsigned long long)bad[2],(unsigned long long)bad[3],
      (unsigned long long)nanbad[0],(unsigned long long)nanbad[1],(unsigned long long)nanbad[2],(unsigned long long)nanbad[3],
      (unsigned long long)payloads);
  return (bad[0]||bad[1]||bad[2]||bad[3]||nanbad[0]||nanbad[1]||nanbad[2]||nanbad[3]) ? 1 : 0;
}
