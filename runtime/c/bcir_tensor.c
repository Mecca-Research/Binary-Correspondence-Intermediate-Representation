#include "bcir_tensor.h"
#include <math.h>
#include <string.h>
#include <limits.h>

void bcir_tensor_cblas_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  bcir_cblas_sgemm_fn fn=ctx ? *(bcir_cblas_sgemm_fn *)ctx : NULL;
  if (!fn || m>INT_MAX || n>INT_MAX || k>INT_MAX) {
    bcir_tensor_mm(NULL,ta,tb,m,n,k,alpha,a,b,beta,c); return;
  }
  fn(101,ta ? 112 : 111,tb ? 112 : 111,(int)m,(int)n,(int)k,alpha,a,
      (int)(ta ? m : k),b,(int)(tb ? k : n),beta,c,(int)n);
}
/* The GEMM contract every path keeps: c[i][j] starts at its beta-scaled value and then adds
 * (alpha*A(i,l))*B(l,j) for l ascending -- one rounded multiply and one rounded add per term,
 * since contraction is off by the build contract. The kernel below changes only the loop nest:
 * a KCxNR tile of B is packed once per (column block, depth block) and reused by every row, and
 * an MRxNR block of C stays in registers across the tile's depth instead of making a round trip
 * through memory per term (the old nest was bound by that store-to-load latency, and its B
 * (tb=0) nest also by alias checks around a 32-iteration loop). A ragged right edge is padded
 * with zero columns that are computed and never stored. Each element sees the same terms in the
 * same order, so every variant is bit-identical to the ascending-k loop on every input, finite
 * or not; test_decoder_train.c holds each variant the host can run to that loop bit for bit. */
#define BCIR_MM_KC 64
#define BCIR_MM_MRMAX 4
#define BCIR_MM_NRMAX 64
#if defined(__GNUC__) || defined(__clang__)
#define BCIR_MM_INLINE static inline __attribute__((always_inline))
#else
#define BCIR_MM_INLINE static inline
#endif
/* The causal products attention issues reuse the kernel with a mask over the matrices' own
 * indices: MM_LOWER stores only C[i][j] with j <= i; MM_KLE adds term l to row i only when
 * l <= i; MM_KGE only when l >= i. A masked term is skipped, never added as a zero, so an
 * operand outside the mask -- the zeros above a causal diagonal, or a non-finite value -- cannot
 * reach a result. Strides are explicit (lda, ldb, ldc) so a head's rows can be read in place. */
enum { MM_FULL, MM_LOWER, MM_KLE, MM_KGE };
typedef struct mm_args {
  int ta,tb; size_t m,n,k; float alpha;
  const float *a; size_t lda; const float *b; size_t ldb; float *c; size_t ldc; int mode;
} mm_args;
/* Rows [i, i+mr) of C over columns [jb, jb+nj) and depths [lb, lb+nl); bt is the packed tile. */
BCIR_MM_INLINE void mm_block(size_t mr, size_t nr, const mm_args *x, const float *bt, size_t i,
    size_t jb, size_t nj, size_t lb, size_t nl, int mode) {
  float acc[BCIR_MM_MRMAX][BCIR_MM_NRMAX];
  size_t r,j,l;
  for (r=0;r<mr;r++) {
    const float *ci=x->c+(i+r)*x->ldc+jb;
    if (nj==nr) { for (j=0;j<nr;j++) acc[r][j]=ci[j]; }
    else for (j=0;j<nr;j++) acc[r][j]=j<nj ? ci[j] : 0.0f;
  }
  for (l=0;l<nl;l++) {
    const float *bl=bt+l*nr; float av[BCIR_MM_MRMAX];
    for (r=0;r<mr;r++) av[r]=x->alpha*x->a[x->ta ? (lb+l)*x->lda+i+r : (i+r)*x->lda+lb+l];
    for (r=0;r<mr;r++) {
      if ((mode==MM_KLE && lb+l>i+r) || (mode==MM_KGE && lb+l<i+r)) continue;
      for (j=0;j<nr;j++) acc[r][j]+=av[r]*bl[j];
    }
  }
  for (r=0;r<mr;r++) {
    float *ci=x->c+(i+r)*x->ldc+jb; size_t end=nj;
    if (mode==MM_LOWER) end=i+r<jb ? 0 : i+r-jb+1<nj ? i+r-jb+1 : nj;
    if (end==nr) { for (j=0;j<nr;j++) ci[j]=acc[r][j]; }
    else for (j=0;j<end;j++) ci[j]=acc[r][j];
  }
}
/* Whether rows [i, i+mr) of this (column, depth) block are outside the mask's reach entirely. */
BCIR_MM_INLINE int mm_unmasked(int mode, size_t i, size_t mr, size_t jb, size_t nj, size_t lb,
    size_t nl) {
  return mode==MM_FULL || (mode==MM_LOWER && i>=jb+nj-1) || (mode==MM_KLE && i>=lb+nl-1) ||
    (mode==MM_KGE && i+mr-1<=lb);
}
BCIR_MM_INLINE void mm_kernel(size_t mr, size_t nr, const mm_args *x) {
  float bt[BCIR_MM_KC*BCIR_MM_NRMAX];
  size_t i,j,l,jb,lb;
  for (jb=0;jb<x->n;jb+=nr) {
    size_t nj=x->n-jb<nr ? x->n-jb : nr;
    for (lb=0;lb<x->k;lb+=BCIR_MM_KC) {
      size_t nl=x->k-lb<BCIR_MM_KC ? x->k-lb : BCIR_MM_KC,i0=0,i1=x->m;
      /* the rows this block reaches at all: under MM_LOWER none above jb stores here, under
       * MM_KLE none above lb takes a term, under MM_KGE none past lb+nl-1 does */
      if (x->mode==MM_LOWER) i0=jb;
      if (x->mode==MM_KLE) i0=lb;
      if (x->mode==MM_KGE && lb+nl<i1) i1=lb+nl;
      if (i0>=i1) continue;
      for (l=0;l<nl;l++) for (j=0;j<nr;j++)
        bt[l*nr+j]=j>=nj ? 0.0f : x->tb ? x->b[(jb+j)*x->ldb+lb+l] : x->b[(lb+l)*x->ldb+jb+j];
      for (i=i0;i+mr<=i1;i+=mr) {
        if (mm_unmasked(x->mode,i,mr,jb,nj,lb,nl)) mm_block(mr,nr,x,bt,i,jb,nj,lb,nl,MM_FULL);
        else mm_block(mr,nr,x,bt,i,jb,nj,lb,nl,x->mode);
      }
      for (;i<i1;i++) {
        if (mm_unmasked(x->mode,i,1,jb,nj,lb,nl)) mm_block(1,nr,x,bt,i,jb,nj,lb,nl,MM_FULL);
        else mm_block(1,nr,x,bt,i,jb,nj,lb,nl,x->mode);
      }
    }
  }
}
static void mm_generic(const mm_args *x) { mm_kernel(2,32,x); }
/* x86-64 Linux with GCC or Clang also carries AVX2 and AVX-512 variants of the same kernel,
 * chosen per call from what the CPU and OS support; anything else runs mm_generic.
 * BCIR_TENSOR_MAX_VARIANT caps the choice (1, 2 or 4) so a build can run a narrower one. */
#ifndef BCIR_TENSOR_MAX_VARIANT
#define BCIR_TENSOR_MAX_VARIANT 4
#endif
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(BCIR_TENSOR_NO_DISPATCH) && BCIR_TENSOR_MAX_VARIANT>1
#define BCIR_MM_DISPATCH 1
__attribute__((target("avx2"))) static void mm_avx2(const mm_args *x) { mm_kernel(2,32,x); }
__attribute__((target("avx512f"))) static void mm_avx512(const mm_args *x) { mm_kernel(4,64,x); }
#endif
int bcir_tensor_mm_variants(void) {
  int v=1;
#ifdef BCIR_MM_DISPATCH
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) v|=2;
  if (BCIR_TENSOR_MAX_VARIANT>=4 && __builtin_cpu_supports("avx512f")) v|=4;
#endif
  return v;
}
static void mm_run(int variant, const mm_args *x) {
  if (!x->m || !x->n || !x->k) return;
#ifdef BCIR_MM_DISPATCH
  if (variant==4) { mm_avx512(x); return; }
  if (variant==2) { mm_avx2(x); return; }
#endif
  (void)variant;
  mm_generic(x);
}
static int mm_widest(void) {
  int v=bcir_tensor_mm_variants();
  return v&4 ? 4 : v&2 ? 2 : 1;
}
int bcir_tensor_mm_variant(int variant, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  mm_args x; size_t i;
  if ((variant!=1 && variant!=2 && variant!=4) || !(bcir_tensor_mm_variants()&variant)) return 0;
  for (i=0;i<m*n;i++) c[i]=beta==0.0f ? 0.0f : beta*c[i];
  x.ta=ta; x.tb=tb; x.m=m; x.n=n; x.k=k; x.alpha=alpha; x.a=a; x.lda=ta ? m : k;
  x.b=b; x.ldb=tb ? k : n; x.c=c; x.ldc=n; x.mode=MM_FULL;
  mm_run(variant,&x);
  return variant;
}
void bcir_tensor_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  (void)ctx;
  bcir_tensor_mm_variant(mm_widest(),ta,tb,m,n,k,alpha,a,b,beta,c);
}
static bcir_tensor_mm_fn mm(const bcir_tensor_provider *p) {
  return p && p->mm ? p->mm : bcir_tensor_mm;
}
void bcir_tensor_linear(const bcir_tensor_provider *p, size_t r, size_t in,
    size_t out, const float *x, const float *w, float *y) {
  mm(p)(p ? p->ctx : NULL,0,1,r,out,in,1.0f,x,w,0.0f,y);
}
void bcir_tensor_linear_backward(const bcir_tensor_provider *p, size_t r,
    size_t in, size_t out, const float *x, const float *w, const float *dy,
    float *dx, float *dw, float beta) {
  mm(p)(p ? p->ctx : NULL,0,0,r,in,out,1.0f,dy,w,beta,dx);
  mm(p)(p ? p->ctx : NULL,1,0,out,in,r,1.0f,dy,x,1.0f,dw);
}
static float sigmoid(float x) {
  if (x>=0.0f) return 1.0f/(1.0f+expf(-x));
  { float e=expf(x); return e/(1.0f+e); }
}
void bcir_tensor_silu(void *ctx, size_t n, const float *x, float *y) {
  size_t i; (void)ctx;
  for (i=0;i<n;i++) y[i]=x[i]*sigmoid(x[i]);
}
void bcir_tensor_swiglu_backward(size_t n, const float *g, const float *u,
    const float *dy, float *dg, float *du) {
  size_t i;
  for (i=0;i<n;i++) {
    float s=sigmoid(g[i]);
    dg[i]=dy[i]*u[i]*(s+g[i]*s*(1.0f-s)); du[i]=dy[i]*g[i]*s;
  }
}
void bcir_tensor_rms(size_t rows, size_t width, float eps, const float *x,
    const float *g, float *y) {
  size_t r,j;
  for (r=0;r<rows;r++) {
    float sum=0.0f,inv;
    for (j=0;j<width;j++) sum+=x[r*width+j]*x[r*width+j];
    inv=1.0f/sqrtf(sum/(float)width+eps);
    for (j=0;j<width;j++) y[r*width+j]=x[r*width+j]*inv*g[j];
  }
}
void bcir_tensor_rms_backward(size_t rows, size_t width, float eps,
    const float *x, const float *g, const float *dy, float *dx, float *dg) {
  size_t r,j;
  for (r=0;r<rows;r++) {
    float sum=0.0f,dot=0.0f,inv,coeff;
    for (j=0;j<width;j++) {
      sum+=x[r*width+j]*x[r*width+j]; dot+=dy[r*width+j]*g[j]*x[r*width+j];
    }
    inv=1.0f/sqrtf(sum/(float)width+eps); coeff=dot*inv*inv*inv/(float)width;
    for (j=0;j<width;j++) {
      dx[r*width+j]=dy[r*width+j]*g[j]*inv-x[r*width+j]*coeff;
      dg[j]+=dy[r*width+j]*x[r*width+j]*inv;
    }
  }
}
void bcir_tensor_rope(size_t b, size_t t, size_t heads, size_t dim,
    double base, float *x, int inverse) {
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
/* Causal attention as two masked products per (batch, head) through the GEMM kernel, then the
 * reference loop's softmax: each score is the dot product accumulated from zero over the channels
 * in order and then scaled, each context row adds p_ij v_j for j ascending -- the operations the
 * reference loop (test_decoder_train.c keeps it) performs, in its order, so the result is
 * bit-identical; future scores stay exactly zero. */
void bcir_tensor_attention(size_t b, size_t t, size_t nh, size_t nk, size_t d,
    const float *q, const float *k, const float *v, float *p, float *y) {
  size_t r,h,i,j; float scale=1.0f/sqrtf((float)d); int variant=mm_widest();
  memset(y,0,b*t*nh*d*sizeof(float)); memset(p,0,b*nh*t*t*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) {
    size_t kh=h/(nh/nk); float *ph=p+(r*nh+h)*t*t; mm_args x;
    x.ta=0; x.tb=1; x.m=t; x.n=t; x.k=d; x.alpha=1.0f; x.a=q+(r*t*nh+h)*d; x.lda=nh*d;
    x.b=k+(r*t*nk+kh)*d; x.ldb=nk*d; x.c=ph; x.ldc=t; x.mode=MM_LOWER;
    mm_run(variant,&x);
    for (i=0;i<t;i++) {
      float *pi=ph+i*t,maxv=-INFINITY,sum=0.0f;
      for (j=0;j<=i;j++) { pi[j]*=scale; if (pi[j]>maxv) maxv=pi[j]; }
      for (j=0;j<=i;j++) { pi[j]=expf(pi[j]-maxv); sum+=pi[j]; }
      for (j=0;j<=i;j++) pi[j]/=sum;
    }
    x.tb=0; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=v+(r*t*nk+kh)*d; x.ldb=nk*d;
    x.c=y+(r*t*nh+h)*d; x.ldc=nh*d; x.mode=MM_KLE;
    mm_run(variant,&x);
  }
}
void bcir_tensor_attention_backward(size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, const float *p,
    const float *dy, float *dq, float *dk, float *dv) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(dq,0,b*t*nh*d*sizeof(float)); memset(dk,0,b*t*nk*d*sizeof(float));
  memset(dv,0,b*t*nk*d*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t;
    float dot=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f;
      for (c=0;c<d;c++) dp+=dy[qi+c]*v[kj+c];
      dot+=p[pi+j]*dp;
    }
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float dp=0.0f,ds;
      for (c=0;c<d;c++) dp+=dy[qi+c]*v[kj+c];
      ds=p[pi+j]*(dp-dot)*scale;
      for (c=0;c<d;c++) {
        dq[qi+c]+=ds*k[kj+c]; dk[kj+c]+=ds*q[qi+c]; dv[kj+c]+=p[pi+j]*dy[qi+c];
      }
    }
  }
}
/* The reference backward above recomputes each probability adjoint dp_ij = dy_i . v_j twice, a
 * scalar dot product at a time. This one computes the same values through the GEMM kernel and
 * rewrites p in place with the score adjoint ds_ij = p_ij (dp_ij - dot_i) scale, which the two
 * remaining products then read; nothing beyond p and the outputs is written. Per (batch, head):
 * dv_j += p_ij dy_i (i >= j, before p is rewritten); then, a chunk of rows at a time, dot_i =
 * sum p_ij dp_ij in j order and ds_ij from a second evaluation of the same dp tiles, as the
 * reference does; then dq_i += ds_ij k_j (j <= i) and dk_j += ds_ij q_i (i >= j). Every
 * accumulator receives the reference's terms in the reference's order -- heads in order for the
 * shared GQA dk/dv, i ascending within a head -- so dq, dk and dv are bit-identical to it. */
#define BCIR_ATT_ROWS 32
#define BCIR_ATT_COLS 64
void bcir_tensor_attention_backward_inplace(size_t b, size_t t, size_t nh, size_t nk,
    size_t d, const float *q, const float *k, const float *v, float *p, const float *dy,
    float *dq, float *dk, float *dv) {
  size_t r,h,i,j,i0; float scale=1.0f/sqrtf((float)d); int variant=mm_widest();
  memset(dq,0,b*t*nh*d*sizeof(float)); memset(dk,0,b*t*nk*d*sizeof(float));
  memset(dv,0,b*t*nk*d*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) {
    size_t kh=h/(nh/nk),ldh=nh*d,ldk=nk*d; float *ph=p+(r*nh+h)*t*t;
    const float *qh=q+(r*t*nh+h)*d,*dyh=dy+(r*t*nh+h)*d;
    const float *kk=k+(r*t*nk+kh)*d,*vk=v+(r*t*nk+kh)*d;
    mm_args x;
    x.ta=1; x.tb=0; x.m=t; x.n=d; x.k=t; x.alpha=1.0f; x.a=ph; x.lda=t; x.b=dyh; x.ldb=ldh;
    x.c=dv+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    mm_run(variant,&x);
    for (i0=0;i0<t;i0+=BCIR_ATT_ROWS) {
      size_t rows=t-i0<BCIR_ATT_ROWS ? t-i0 : BCIR_ATT_ROWS,pass,jt;
      float dot[BCIR_ATT_ROWS],tile[BCIR_ATT_ROWS*BCIR_ATT_COLS];
      for (i=0;i<rows;i++) dot[i]=0.0f;
      for (pass=0;pass<2;pass++) for (jt=0;jt<i0+rows;jt+=BCIR_ATT_COLS) {
        size_t cols=i0+rows-jt<BCIR_ATT_COLS ? i0+rows-jt : BCIR_ATT_COLS;
        memset(tile,0,rows*BCIR_ATT_COLS*sizeof(float));
        x.ta=0; x.tb=1; x.m=rows; x.n=cols; x.k=d; x.a=dyh+i0*ldh; x.lda=ldh; x.b=vk+jt*ldk;
        x.ldb=ldk; x.c=tile; x.ldc=BCIR_ATT_COLS; x.mode=MM_FULL;
        mm_run(variant,&x);
        for (i=0;i<rows;i++) {
          float *pi=ph+(i0+i)*t; size_t end=i0+i+1<jt+cols ? i0+i+1 : jt+cols;
          for (j=jt;j<end;j++) {
            float dp=tile[i*BCIR_ATT_COLS+j-jt];
            if (!pass) dot[i]+=pi[j]*dp;
            else pi[j]=pi[j]*(dp-dot[i])*scale;
          }
        }
      }
    }
    x.ta=0; x.tb=0; x.m=t; x.n=d; x.k=t; x.a=ph; x.lda=t; x.b=kk; x.ldb=ldk;
    x.c=dq+(r*t*nh+h)*d; x.ldc=ldh; x.mode=MM_KLE;
    mm_run(variant,&x);
    x.ta=1; x.b=qh; x.ldb=ldh; x.c=dk+(r*t*nk+kh)*d; x.ldc=ldk; x.mode=MM_KGE;
    mm_run(variant,&x);
  }
}
