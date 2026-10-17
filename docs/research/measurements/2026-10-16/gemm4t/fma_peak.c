/* The AVX-512 FMA rate this host gives one thread, with one to four running at once (NDT-GEMM-4T).
 * Twelve independent accumulators per thread, each a chain of vfmadd231ps (four-cycle latency, two
 * ports), so the probe is bound by the ports, not the chains; prints one JSON line per thread count.
 *   cc -O2 -pthread fma_peak.c -o fma_peak && ./fma_peak */
#define _POSIX_C_SOURCE 199309L
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static volatile float sink; static const long iters=20000000;
#define F(v) v=_mm512_fmadd_ps(v,b,c)
__attribute__((target("avx512f"))) static void *run(void *arg) {
  __m512 b=_mm512_set1_ps(1.0000001f),c=_mm512_set1_ps(1e-7f);
  __m512 a0=_mm512_set1_ps(0),a1=_mm512_set1_ps(1),a2=_mm512_set1_ps(2),a3=_mm512_set1_ps(3),a4=_mm512_set1_ps(4),
         a5=_mm512_set1_ps(5),a6=_mm512_set1_ps(6),a7=_mm512_set1_ps(7),a8=_mm512_set1_ps(8),a9=_mm512_set1_ps(9),
         a10=_mm512_set1_ps(10),a11=_mm512_set1_ps(11);
  long i; (void)arg;
  for (i=0;i<iters;i++) {
    F(a0);F(a1);F(a2);F(a3);F(a4);F(a5);F(a6);F(a7);F(a8);F(a9);F(a10);F(a11);
    __asm__ volatile("" : "+v"(a0),"+v"(a1),"+v"(a2),"+v"(a3),"+v"(a4),"+v"(a5));
  }
  a0=_mm512_add_ps(a0,a1); a2=_mm512_add_ps(a2,a3); a4=_mm512_add_ps(a4,a5); a6=_mm512_add_ps(a6,a7);
  a8=_mm512_add_ps(a8,a9); a10=_mm512_add_ps(a10,a11);
  a0=_mm512_add_ps(_mm512_add_ps(a0,a2),_mm512_add_ps(_mm512_add_ps(a4,a6),_mm512_add_ps(a8,a10)));
  sink=_mm512_reduce_add_ps(a0); return NULL;
}
int main(void) {
  int t;
  for (t=1;t<=4;t++) {
    pthread_t th[4]; int q; double t0=now(),dt;
    for (q=0;q<t;q++) pthread_create(&th[q],NULL,run,NULL);
    for (q=0;q<t;q++) pthread_join(th[q],NULL);
    dt=now()-t0;
    printf("{\"threads\": %d, \"gflops_per_thread\": %.1f, \"gflops\": %.1f}\n",t,iters*12.0*32/dt/1e9,t*iters*12.0*32/dt/1e9);
  }
  return 0;
}
