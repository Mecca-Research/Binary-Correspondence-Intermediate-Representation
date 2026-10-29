/* The core's AVX-512 FMA rate with every operand in registers: W threads started together, each 24
 * independent accumulators (vfmadd231ps), ITERS iterations; GFLOP/s a thread and the TSC-cycle rate.
 * usage: fma_peak W ITERS */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
typedef struct { long iters; double secs; float sink; pthread_barrier_t *bar; } arm;
__attribute__((target("avx512f"))) static void *run(void *p) {
  arm *a=(arm *)p; long i;
  __m512 x=_mm512_set1_ps(1.0000001f),y=_mm512_set1_ps(0.9999999f);
#define D(q) __m512 a##q=_mm512_set1_ps((float)q);
  D(0) D(1) D(2) D(3) D(4) D(5) D(6) D(7) D(8) D(9) D(10) D(11) D(12) D(13) D(14) D(15) D(16) D(17) D(18) D(19) D(20) D(21) D(22) D(23)
  pthread_barrier_wait(a->bar);
  double t0=now();
  for (i=0;i<a->iters;i++) {
#define F(q) a##q=_mm512_fmadd_ps(a##q,x,y);
    F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15) F(16) F(17) F(18) F(19) F(20) F(21) F(22) F(23)
  }
  a->secs=now()-t0;
#define S(q) s=_mm512_add_ps(s,a##q);
  { __m512 s=a0; S(1) S(2) S(3) S(4) S(5) S(6) S(7) S(8) S(9) S(10) S(11) S(12) S(13) S(14) S(15) S(16) S(17) S(18) S(19) S(20) S(21) S(22) S(23)
    a->sink=_mm512_reduce_add_ps(s); }
  return NULL;
}
int main(int argc,char **argv) {
  if (argc!=3) return 2;
  unsigned w=(unsigned)atoi(argv[1]),q; long iters=atol(argv[2]); arm a[16]; pthread_t th[16]; pthread_barrier_t bar;
  pthread_barrier_init(&bar,NULL,w);
  for (q=0;q<w;q++) { a[q].iters=iters; a[q].bar=&bar; pthread_create(&th[q],NULL,run,&a[q]); }
  double gf=0; for (q=0;q<w;q++) { pthread_join(th[q],NULL); gf+=24.0*16*2*iters/a[q].secs/1e9; }
  printf("{\"threads\": %u, \"gflops_a_thread\": %.1f, \"ghz_equiv\": %.2f}\n",w,gf/w,gf/w/64.0);
  return 0;
}
