/* ATT-BOUND: the host's AVX-512 rates, measured: 16 independent accumulators of _mm512_fmadd_ps (and of mul+add, v1's form) a thread,
 * timed by CLOCK_MONOTONIC and by TSC; usage: fma_peak THREADS ITERS */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
static long iters; static volatile float sink;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
__attribute__((target("avx512f"))) static void *fma_run(void *arg) {
  __m512 a[16], b = _mm512_set1_ps(0.999f), c = _mm512_set1_ps(1e-6f); long i; int k; (void)arg;
  for (k = 0; k < 16; k++) a[k] = _mm512_set1_ps((float)k);
  for (i = 0; i < iters; i++) for (k = 0; k < 16; k++) a[k] = _mm512_fmadd_ps(a[k], b, c);
  __m512 s = a[0]; for (k = 1; k < 16; k++) s = _mm512_add_ps(s, a[k]);
  sink = _mm512_reduce_add_ps(s); return NULL;
}
__attribute__((target("avx512f"))) static void *muladd_run(void *arg) {
  __m512 a[16], b = _mm512_set1_ps(0.999f), c = _mm512_set1_ps(1e-6f); long i; int k; (void)arg;
  for (k = 0; k < 16; k++) a[k] = _mm512_set1_ps((float)k);
  for (i = 0; i < iters; i++) for (k = 0; k < 16; k++) a[k] = _mm512_add_ps(_mm512_mul_ps(a[k], b), c);
  __m512 s = a[0]; for (k = 1; k < 16; k++) s = _mm512_add_ps(s, a[k]);
  sink = _mm512_reduce_add_ps(s); return NULL;
}
int main(int argc, char **argv) {
  int th = argc > 1 ? atoi(argv[1]) : 1, t, m; pthread_t p[64]; iters = argc > 2 ? atol(argv[2]) : 50000000L;
  for (m = 0; m < 2; m++) {
    void *(*fn)(void *) = m ? muladd_run : fma_run;
    double t0 = now(); unsigned long long c0 = __rdtsc();
    for (t = 0; t < th; t++) pthread_create(&p[t], NULL, fn, NULL);
    for (t = 0; t < th; t++) pthread_join(p[t], NULL);
    double dt = now() - t0; unsigned long long dc = __rdtsc() - c0;
    double flops = (double)th * iters * 16 * 16 * 2;  /* 16 accumulators x 16 lanes x (multiply + add) */
    printf("%s %d thread(s): %.1f GFLOP/s total, %.1f a thread; %.2f TSC GHz; %.2f flops a TSC cycle a thread\n",
           m ? "mul+add" : "fma", th, flops / dt / 1e9, flops / dt / 1e9 / th, dc / dt / 1e9, flops / th / dc);
  }
  return 0;
}
