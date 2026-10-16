/* NDT-ATTN-7's differential: the attention forward (v1 and v2 softmax) and its in-place backward
 * over twelve shapes and four input sets -- two finite, two with NaN (two payloads), +-Inf, -0.0,
 * subnormal and overflowing operands among them -- printing per shape, set and softmax the FNV-1a
 * hash of every output's bytes, the same hash with every NaN read as one pattern, and, on the v1
 * softmax, how many elements of the context, dq, dk and dv differ from the scalar reference loops
 * of runtime/c/test_decoder_train.c (copied here verbatim) and how many of those are NaN in both.
 * Built against two trees, the lines must agree but for the raw hashes of sets with two NaN payloads.
 *
 *   cc -O3 -ffp-contract=off -fno-math-errno -pthread -I TREE nan_payloads.c TREE/bcir_tensor.c -lm
 *   ./a.out THREADS
 */
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
  const unsigned char *b=(const unsigned char *)p; size_t i;
  for (i=0;i<n;i++) { h^=b[i]; h*=1099511628211ull; }
  return h;
}
static uint64_t fnv_values(uint64_t h, const float *f, size_t n) {  /* every NaN as 0x7fc00000 */
  size_t i;
  for (i=0;i<n;i++) { uint32_t u; memcpy(&u,f+i,4); if (f[i]!=f[i]) u=0x7fc00000u; h=fnv(h,&u,4); }
  return h;
}
static size_t differing(const float *a, const float *b, size_t n, size_t *nan_pairs) {
  size_t i,c=0;
  for (i=0;i<n;i++) if (memcmp(a+i,b+i,4)) { c++; if (a[i]!=a[i] && b[i]!=b[i]) (*nan_pairs)++; }
  return c;
}
static uint32_t rs;
static uint32_t rnd(void) { rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }
static float val(int special) {
  float u=(float)(rnd()%2000001u)/1e6f-1.0f;
  if (special) {
    uint32_t c=rnd()%997u;
    if (c==0) return NAN;
    if (c==1) return INFINITY;
    if (c==2) return -INFINITY;
    if (c==3) return -0.0f;
    if (c==4) return 1e-40f;
    if (c==5) return 3e37f;
    if (c==6) return -3e37f;
    if (c==8) { uint32_t x=0x7fc12345u; float f; memcpy(&f,&x,4); return f; }  /* a second payload */
  }
  return u;
}
int main(int argc, char **argv) {
  static const size_t shapes[][5]={{2,128,8,4,48},{1,100,4,2,48},{1,37,2,1,48},{2,64,4,4,32},{1,129,2,2,64},
    {1,8,1,1,48},{1,7,1,1,16},{1,200,2,1,40},{1,300,2,1,48},{1,16,4,2,96},{3,24,2,2,48},{1,250,1,1,64}};
  unsigned threads=argc>1 ? (unsigned)atoi(argv[1]) : 1; size_t s; int set;
  static _Alignas(64) unsigned char mem[1<<16];
  bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(mem,sizeof(mem),threads) : NULL;
  for (s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++) for (set=0;set<4;set++) {
    size_t b=shapes[s][0],t=shapes[s][1],nh=shapes[s][2],nk=shapes[s][3],d=shapes[s][4],i; int v2;
    size_t nq=b*t*nh*d,nkv=b*t*nk*d,np=b*nh*t*t;
    float *q=malloc(nq*4),*k=malloc(nkv*4),*v=malloc(nkv*4),*p=malloc(np*4),*y=malloc(nq*4),*dy=malloc(nq*4),
        *dq=malloc(nq*4),*dk=malloc(nkv*4),*dv=malloc(nkv*4),*p0=malloc(np*4),*y0=malloc(nq*4),
        *dq0=malloc(nq*4),*dk0=malloc(nkv*4),*dv0=malloc(nkv*4);
    int special=set>=2; float sc=set==1 ? 8.0f : 1.0f;
    if (!q || !k || !v || !p || !y || !dy || !dq || !dk || !dv || !p0 || !y0 || !dq0 || !dk0 || !dv0) return 2;
    rs=(uint32_t)(1000u*s+(unsigned)set+1u);
    for (i=0;i<nq;i++) { q[i]=sc*val(special); dy[i]=val(special); }
    for (i=0;i<nkv;i++) { k[i]=sc*val(special); v[i]=val(special); }
    for (v2=0;v2<2;v2++) {
      uint64_t raw=1469598103934665603ull,value=raw; size_t dif=0,pairs=0;
      for (i=0;i<np;i++) p[i]=(float)(rnd()%7u);  /* stale arena values the kernels must not read */
      for (i=0;i<nq;i++) { y[i]=NAN; dq[i]=NAN; }
      for (i=0;i<nkv;i++) { dk[i]=NAN; dv[i]=NAN; }
      if (v2) bcir_tensor_attention_v2_pool(pool,b,t,nh,nk,d,q,k,v,p,y);
      else bcir_tensor_attention_pool(pool,b,t,nh,nk,d,q,k,v,p,y);
      raw=fnv(raw,p,np*4); raw=fnv(raw,y,nq*4); value=fnv_values(value,p,np); value=fnv_values(value,y,nq);
      if (!v2) {
        ref_attention(b,t,nh,nk,d,q,k,v,p0,y0,expf);
        dif+=differing(y0,y,nq,&pairs);
      }
      bcir_tensor_attention_backward_inplace_pool(pool,b,t,nh,nk,d,q,k,v,p,dy,dq,dk,dv);
      raw=fnv(raw,p,np*4); raw=fnv(raw,dq,nq*4); raw=fnv(raw,dk,nkv*4); raw=fnv(raw,dv,nkv*4);
      value=fnv_values(value,p,np); value=fnv_values(value,dq,nq); value=fnv_values(value,dk,nkv);
      value=fnv_values(value,dv,nkv);
      if (!v2) {
        ref_attention_backward(b,t,nh,nk,d,q,k,v,p0,dy,dq0,dk0,dv0);
        dif+=differing(dq0,dq,nq,&pairs)+differing(dk0,dk,nkv,&pairs)+differing(dv0,dv,nkv,&pairs);
        printf("shape %zu set %d v1 raw %016llx values %016llx vs-reference %zu (%zu NaN pairs)\n",s,set,
            (unsigned long long)raw,(unsigned long long)value,dif,pairs);
      } else printf("shape %zu set %d v2 raw %016llx values %016llx\n",s,set,(unsigned long long)raw,
          (unsigned long long)value);
    }
    free(q); free(k); free(v); free(p); free(y); free(dy); free(dq); free(dk); free(dv);
    free(p0); free(y0); free(dq0); free(dk0); free(dv0);
  }
  bcir_tensor_pool_stop(pool); return 0;
}
