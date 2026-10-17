/* NDT-UPDATE's shared division against the division: the trainer's own div_shared and
 * shared_reciprocal (its unit compiled in), for each divisor d in a list, every float x
 * (every significand of [1,2) at one exponent, both signs, plus zeros, the extremes and a sweep of
 * exponents) has div_shared(x,d,shared_reciprocal(d)) == x/d bit for bit.  The dividend domain of the AdamW
 * commit's two constant divisions is exactly the finite floats (m and v converted to double), and
 * for a fixed d both sides scale exactly by powers of two while nothing under- or overflows, so a
 * significand sweep at one exponent covers every finite float x for that d.
 * The divisors: the update's bias corrections 1 - beta^t as it computes them (power = power*beta)
 * for ten betas and the first STEPS steps, divisors at the edges of [2^-53, 1] (powers of two,
 * 1 - 2^-k, significands of all ones and of one, the largest double below 1) and RANDOM random
 * ones over the range. The form with one correction alone is counted beside it (not used).
 *
 *   cc -O2 -ffp-contract=off -fno-math-errno -mfma -pthread -I runtime/c division_check.c runtime/c/bcir_tensor.c -lm
 *   ./a.out [STEPS [RANDOM]] */
#include "bcir_decoder_train.c"
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the trainer's own division and reciprocal (its unit compiled in) */
static long long one_miss_total;
static inline double div_one(double x, double d, double y) {  /* one correction only: counted, not used */
  double q0=x*y, r0=fma(-q0,d,x);
  return copysign(fma(r0,y,q0),x);
}
static int same(double a, double b) { uint64_t u,v; memcpy(&u,&a,8); memcpy(&v,&b,8); return u==v; }

static double *divisors; static size_t ndiv; static int nthreads=2;
static long long bad_total;
static pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER;

static long long check_d(double d) {
  double y=shared_reciprocal(d); long long bad=0,miss=0; uint32_t s; int e;
  /* every significand at exponent 0, both signs */
  for (s=0;s<(1u<<23);s++) {
    uint32_t bits=0x3f800000u|s; float f; double x;
    memcpy(&f,&bits,4); x=(double)f;
    if (!same(div_shared(x,d,y),x/d)) bad++;
    if (!same(div_shared(-x,d,y),-x/d)) bad++;
    if (!same(div_one(x,d,y),x/d)) miss++;
  }
  /* a sweep of exponents for a few significands, the zeros, subnormal floats and FLT_MAX */
  for (e=-149;e<=127;e++) {
    static const uint32_t sig[]={0,1,2,3,0x7fffff,0x7ffffe,0x400000,0x555555,0x2aaaaa,0x123457};
    size_t k;
    for (k=0;k<sizeof(sig)/sizeof(sig[0]);k++) {
      double x=ldexp(1.0+(double)sig[k]/8388608.0,e);
      float f=(float)x; x=(double)f;            /* a float (subnormal ones rounded to their grid) */
      if (!same(div_shared(x,d,y),x/d)) bad++;
      if (!same(div_shared(-x,d,y),-x/d)) bad++;
    }
  }
  {
    double xs[]={0.0,-0.0,(double)FLT_MAX,-(double)FLT_MAX,(double)FLT_MIN,(double)FLT_TRUE_MIN,
      -(double)FLT_TRUE_MIN,1.0,-1.0};
    size_t k;
    for (k=0;k<sizeof(xs)/sizeof(xs[0]);k++) if (!same(div_shared(xs[k],d,y),xs[k]/d)) bad++;
  }
  pthread_mutex_lock(&mu); one_miss_total+=miss; pthread_mutex_unlock(&mu);
  return bad;
}
static void *worker(void *arg) {
  size_t t=(size_t)arg,i; long long bad=0;
  for (i=t;i<ndiv;i+=(size_t)nthreads) {
    long long b=check_d(divisors[i]);
    if (b) { pthread_mutex_lock(&mu); printf("MISMATCH d=%a: %lld\n",divisors[i],b); pthread_mutex_unlock(&mu); }
    bad+=b;
  }
  pthread_mutex_lock(&mu); bad_total+=bad; pthread_mutex_unlock(&mu);
  return NULL;
}
static void add_divisor(double d, size_t *n, size_t cap) { if (*n<cap && d>0 && d<=1) divisors[(*n)++]=d; }
int main(int argc, char **argv) {
  size_t cap=200000,n=0,t; int k; uint64_t seed=0x9e3779b97f4a7c15ull;
  int steps=argc>1 ? atoi(argv[1]) : 2000, randoms=argc>2 ? atoi(argv[2]) : 4000;
  static const double betas[]={0.9,0.95,0.98,0.99,0.995,0.999,0.9999,0.5,0.0,0.1};
  pthread_t th[8];
  divisors=malloc(cap*sizeof(double));
  /* the trainer's own sequence: b = power*beta, d = 1-b, power = b after the step */
  for (k=0;k<(int)(sizeof(betas)/sizeof(betas[0]));k++) {
    double power=1.0; int s;
    for (s=0;s<steps;s++) { double b=power*betas[k]; add_divisor(1-b,&n,cap); power=b; }
  }
  /* adversarial: powers of two, 1-2^-k, significands all ones, the largest double below 1 */
  for (k=0;k<=60;k++) { add_divisor(ldexp(1.0,-k),&n,cap); add_divisor(1-ldexp(1.0,-k),&n,cap);
    add_divisor(ldexp(2.0-ldexp(1.0,-52),-k-1),&n,cap); add_divisor(ldexp(1.0+ldexp(1.0,-52),-k-1),&n,cap); }
  add_divisor(nextafter(1.0,0.0),&n,cap); add_divisor(ldexp(1.0,-53),&n,cap); add_divisor(1.5,&n,cap);
  /* random divisors in (0,1] with random significands and exponents down to 2^-53 */
  for (k=0;k<randoms;k++) {
    uint64_t m; int e; double d;
    seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; m=seed>>12;
    seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; e=-(int)(seed%54);
    d=ldexp(1.0+(double)m/4503599627370496.0,e-1); add_divisor(d,&n,cap);
  }
  ndiv=n;
  for (t=0;t<(size_t)nthreads;t++) pthread_create(&th[t],NULL,worker,(void *)t);
  for (t=0;t<(size_t)nthreads;t++) pthread_join(th[t],NULL);
  printf("divisors %zu, dividends per divisor %u: div_shared against the division, %lld mismatches; "
      "one correction alone (not the trainer's form) misses %lld of %zu\n",
      ndiv,(1u<<24)+277u*20u+9u,bad_total,one_miss_total,ndiv*(size_t)(1u<<23));
  return bad_total!=0;
}
