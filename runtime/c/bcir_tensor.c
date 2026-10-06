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
void bcir_tensor_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c) {
  size_t i,j,l,ib,jb,lb;
  (void)ctx;
  for (i=0;i<m*n;i++) c[i]=beta==0.0f ? 0.0f : beta*c[i];
  /* Bounded loops, not scalar source expansion. Ascending k accumulation is retained
   * across blocks; contraction is disabled by the build contract. */
  for (ib=0;ib<m;ib+=32) for (lb=0;lb<k;lb+=32) for (jb=0;jb<n;jb+=32)
    for (i=ib;i<m && i<ib+32;i++) for (l=lb;l<k && l<lb+32;l++) {
      float av=alpha*a[ta ? l*m+i : i*k+l];
      for (j=jb;j<n && j<jb+32;j++) c[i*n+j]+=av*b[tb ? j*k+l : l*n+j];
    }
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
void bcir_tensor_attention(size_t b, size_t t, size_t nh, size_t nk, size_t d,
    const float *q, const float *k, const float *v, float *p, float *y) {
  size_t r,h,i,j,c; float scale=1.0f/sqrtf((float)d);
  memset(y,0,b*t*nh*d*sizeof(float)); memset(p,0,b*nh*t*t*sizeof(float));
  for (r=0;r<b;r++) for (h=0;h<nh;h++) for (i=0;i<t;i++) {
    size_t kh=h/(nh/nk),qi=((r*t+i)*nh+h)*d,pi=((r*nh+h)*t+i)*t;
    float maxv=-INFINITY,sum=0.0f;
    for (j=0;j<=i;j++) {
      size_t kj=((r*t+j)*nk+kh)*d; float z=0.0f;
      for (c=0;c<d;c++) z+=q[qi+c]*k[kj+c];
      p[pi+j]=z*scale; if (p[pi+j]>maxv) maxv=p[pi+j];
    }
    for (j=0;j<=i;j++) { p[pi+j]=expf(p[pi+j]-maxv); sum+=p[pi+j]; }
    for (j=0;j<=i;j++) {
      size_t vj=((r*t+j)*nk+kh)*d; p[pi+j]/=sum;
      for (c=0;c<d;c++) y[qi+c]+=p[pi+j]*v[vj+c];
    }
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
