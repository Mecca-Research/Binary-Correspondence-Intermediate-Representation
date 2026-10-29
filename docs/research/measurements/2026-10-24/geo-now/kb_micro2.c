/* KBK-0b: one 448-term contract block of the k-blocked kernel over 128 rows (16 blocks of 8 x 48), the packed tile
 * 448 x 48 (86 KiB, past a core's 48 KiB of L1d): "base" runs each row block's whole block of terms (the tree's
 * order); "sub S" runs the terms S at a time across every row block, each block's 24 accumulators stored after a
 * sub-range and loaded before the next -- the same fused chain from zero for every element, so the same bits.
 * usage: kb_micro2 S REPS TRIES */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define INL static inline __attribute__((always_inline))
enum { M = 128, K = 448, N = 48 };
__attribute__((target("avx512f"))) INL void chain(const float *a, size_t lda, const float *bt, size_t l0, size_t l1,
    __m512 *acc0, __m512 *acc1, __m512 *acc2) {
  size_t l, r;
  for (l = l0; l < l1; l++) {
    __m512 b0 = _mm512_load_ps(bt + l * 48), b1 = _mm512_load_ps(bt + l * 48 + 16), b2 = _mm512_load_ps(bt + l * 48 + 32);
    for (r = 0; r < 8; r++) {
      __m512 av = _mm512_set1_ps(a[r * lda + l]);
      acc0[r] = _mm512_fmadd_ps(av, b0, acc0[r]);
      acc1[r] = _mm512_fmadd_ps(av, b1, acc1[r]);
      acc2[r] = _mm512_fmadd_ps(av, b2, acc2[r]);
    }
  }
}
__attribute__((target("avx512f"))) INL void fold(float *c, const __m512 *acc0, const __m512 *acc1, const __m512 *acc2) {
  size_t r;
  for (r = 0; r < 8; r++) {  /* c = c + 1*P: the rounded product, then the rounded sum */
    _mm512_storeu_ps(c + r * N, _mm512_add_ps(_mm512_loadu_ps(c + r * N), _mm512_mul_ps(_mm512_set1_ps(1.0f), acc0[r])));
    _mm512_storeu_ps(c + r * N + 16, _mm512_add_ps(_mm512_loadu_ps(c + r * N + 16), _mm512_mul_ps(_mm512_set1_ps(1.0f), acc1[r])));
    _mm512_storeu_ps(c + r * N + 32, _mm512_add_ps(_mm512_loadu_ps(c + r * N + 32), _mm512_mul_ps(_mm512_set1_ps(1.0f), acc2[r])));
  }
}
__attribute__((target("avx512f"), noinline)) static void base(const float *a, const float *bt, float *c) {
  size_t i, r;
  for (i = 0; i < M; i += 8) {
    __m512 acc0[8], acc1[8], acc2[8];
    for (r = 0; r < 8; r++) { acc0[r] = _mm512_setzero_ps(); acc1[r] = _mm512_setzero_ps(); acc2[r] = _mm512_setzero_ps(); }
    chain(a + i * K, K, bt, 0, K, acc0, acc1, acc2);
    fold(c + i * N, acc0, acc1, acc2);
  }
}
__attribute__((target("avx512f"), noinline)) static void sub(const float *a, const float *bt, float *c, float *p, size_t s) {
  size_t l0, i, r;
  for (l0 = 0; l0 < K; l0 += s) {
    size_t l1 = l0 + s < K ? l0 + s : K;
    for (i = 0; i < M; i += 8) {
      __m512 acc0[8], acc1[8], acc2[8]; float *pi = p + i * N;
      for (r = 0; r < 8; r++) {
        if (l0) { acc0[r] = _mm512_load_ps(pi + r * N); acc1[r] = _mm512_load_ps(pi + r * N + 16); acc2[r] = _mm512_load_ps(pi + r * N + 32); }
        else { acc0[r] = _mm512_setzero_ps(); acc1[r] = _mm512_setzero_ps(); acc2[r] = _mm512_setzero_ps(); }
      }
      chain(a + i * K, K, bt, l0, l1, acc0, acc1, acc2);
      if (l1 < K) for (r = 0; r < 8; r++) { _mm512_store_ps(pi + r * N, acc0[r]); _mm512_store_ps(pi + r * N + 16, acc1[r]); _mm512_store_ps(pi + r * N + 32, acc2[r]); }
      else fold(c + i * N, acc0, acc1, acc2);
    }
  }
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
  size_t s = argc > 1 ? strtoul(argv[1], 0, 10) : 224, i; long reps = argc > 2 ? atol(argv[2]) : 2000, q; int tries = argc > 3 ? atoi(argv[3]) : 5, t;
  float *a = aligned_alloc(64, M * K * 4), *bt = aligned_alloc(64, K * 48 * 4), *c0 = aligned_alloc(64, M * N * 4),
        *c1 = aligned_alloc(64, M * N * 4), *p = aligned_alloc(64, M * N * 4);
  double tb = 1e30, ts = 1e30;
  for (i = 0; i < M * K; i++) a[i] = (float)((i * 37) % 101) * 0.01f - 0.5f;
  for (i = 0; i < K * 48; i++) bt[i] = (float)((i * 13) % 97) * 0.02f - 0.9f;
  memset(c0, 0, M * N * 4); memset(c1, 0, M * N * 4);
  base(a, bt, c0); sub(a, bt, c1, p, s);
  printf("same bits: %s\n", memcmp(c0, c1, M * N * 4) ? "NO" : "yes");
  for (t = 0; t < tries; t++) {
    double x = now(); for (q = 0; q < reps; q++) base(a, bt, c0); x = now() - x; if (x < tb) tb = x;
    x = now(); for (q = 0; q < reps; q++) sub(a, bt, c1, p, s); x = now() - x; if (x < ts) ts = x;
  }
  printf("M=%d K=%d N=%d: base %.1f GFLOP/s, sub %zu %.1f GFLOP/s (x%.3f of base's time)\n", M, K, N,
      2.0 * M * N * K * reps / tb * 1e-9, s, 2.0 * M * N * K * reps / ts * 1e-9, ts / tb);
  return 0;
}
