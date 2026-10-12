/* The update's passes on the 9.84M decoder's parameter count: the first pass as written (four
 * streams, the double chain, the maxima), the gradient stream alone with the chain, and the
 * whole bcir_decoder_update on a pool of 1 and 4 -- medians of 7 timed rounds after one. */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bcir_decoder_train.h"
#include "bcir_tensor.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return x<y ? -1 : x>y; }
static double median(double *v,int n) { qsort(v,n,sizeof(double),cmp); return v[n/2]; }
static volatile double sink;
static double first_pass(const float *gr,const float *wt,const float *m1,const float *m2,size_t n,double gs) {
  double norm=0.0,grmax=0,wmax=0,mmax=0,vmax=0; int bad=0; size_t i;
  for (i=0;i<n;i++) {
    double g=(double)gr[i]*gs,ag=fabsf(gr[i]),aw=fabsf(wt[i]),am=fabsf(m1[i]),av=m2[i];
    bad|=!(fabs(g)<=DBL_MAX) | !(aw<=FLT_MAX) | !(am<=FLT_MAX) | !(av>=0 && av<=FLT_MAX);
    norm+=g*g;
    grmax=ag>grmax ? ag : grmax; wmax=aw>wmax ? aw : wmax; mmax=am>mmax ? am : mmax; vmax=av>vmax ? av : vmax;
  }
  sink=grmax+wmax+mmax+vmax+bad; return norm;
}
static double grad_only(const float *gr,size_t n,double gs) {
  double norm=0.0,grmax=0; int bad=0; size_t i;
  for (i=0;i<n;i++) { double g=(double)gr[i]*gs,ag=fabsf(gr[i]); bad|=!(fabs(g)<=DBL_MAX); norm+=g*g; grmax=ag>grmax ? ag : grmax; }
  sink=grmax+bad; return norm;
}
int main(void) {
  bcir_decoder_spec spec={260,384,8,4,6,1024,8,128,1,10000.0,1e-5}; bcir_decoder_plan plan;
  bcir_decoder_state s; bcir_decoder_adamw opt={5e-4,0.9,0.95,1e-8,0.1,1.0}; double norm,t[8];
  size_t n,i; int r; unsigned threads;
  if (bcir_decoder_make_plan(&spec,&plan)) { puts("plan refused"); return 1; }
  n=plan.parameters; printf("parameters %zu (%.1f MB per array)\n",n,n*4/1e6);
  float *w=malloc(n*4),*g=malloc(n*4),*m1=malloc(n*4),*m2=malloc(n*4),*act=malloc(plan.activations*4),*scr=malloc(plan.scratch*4);
  for (i=0;i<n;i++) { w[i]=0.02f*sinf((float)i); g[i]=1e-3f*cosf((float)i*0.7f); m1[i]=1e-4f*sinf((float)i*0.3f); m2[i]=1e-8f*(1.0f+cosf((float)i*0.1f)); }
  for (r=0;r<8;r++) { double t0=now(); norm=first_pass(g,w,m1,m2,n,1.0); t[r]=now()-t0; } sink=norm;
  printf("first pass as written (four streams, chain, maxima): %.1f ms\n",1000*median(t+1,7));
  for (r=0;r<8;r++) { double t0=now(); norm=grad_only(g,n,1.0); t[r]=now()-t0; } sink=norm;
  printf("gradient stream alone (chain + its maximum):          %.1f ms\n",1000*median(t+1,7));
  for (threads=1;threads<=4;threads*=4) {
    static _Alignas(max_align_t) unsigned char mem[8192]; bcir_tensor_pool *pool=threads>1 ? bcir_tensor_pool_init(mem,sizeof(mem),threads) : NULL;
    memset(&s,0,sizeof(s)); s.plan=plan; s.weights=w; s.gradients=g; s.moment1=m1; s.moment2=m2; s.activations=act; s.scratch=scr;
    s.weight_capacity=n; s.gradient_capacity=n; s.moment1_capacity=n; s.moment2_capacity=n; s.activation_capacity=plan.activations; s.scratch_capacity=plan.scratch;
    s.beta1_power=1.0; s.beta2_power=1.0; s.pool=pool;
    for (r=0;r<8;r++) { double t0=now(); int rc=bcir_decoder_update(&s,&opt,1.0,&norm); t[r]=now()-t0; if (rc) { printf("update rc %d\n",rc); return 1; } }
    printf("bcir_decoder_update, pool of %u:                      %.1f ms (norm %.6f)\n",threads,1000*median(t+1,7),norm);
    bcir_tensor_pool_stop(pool);
  }
  return 0;
}
