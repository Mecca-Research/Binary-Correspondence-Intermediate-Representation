/* Survey (scratch): the host's read bandwidth from memory at 1-4 threads, each thread summing its own contiguous
 * part of a 1 GiB buffer (or, strided, each thread a column range of every 5,632-byte row, as DECODE-THREADS reads). */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint8_t *buf; static size_t total = (size_t)1 << 30, rowlen = 5632; static int strided;
typedef struct { int t, n; double sum; } job;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
__attribute__((target("avx2"))) static void *work(void *p) {
  job *j = p; __m256i acc = _mm256_setzero_si256();
  if (!strided) {
    size_t part = total / j->n, a = part * j->t, b = a + part;
    for (size_t i = a; i < b; i += 32) acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_load_si256((const __m256i *)(buf + i)), _mm256_setzero_si256()));
  } else {
    size_t rows = total / rowlen, cols = rowlen / j->n, c0 = cols * j->t, c1 = c0 + cols;
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = c0; c + 32 <= c1; c += 32) acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(buf + r * rowlen + c)), _mm256_setzero_si256()));
  }
  uint64_t v[4]; memcpy(v, &acc, 32); j->sum = (double)(v[0] + v[1] + v[2] + v[3]); return NULL;
}
int main(void) {
  buf = aligned_alloc(64, total); memset(buf, 1, total);
  for (strided = 0; strided < 2; ++strided)
    for (int n = 1; n <= 4; n *= 2) {
      double best = 1e9;
      for (int rep = 0; rep < 5; ++rep) {
        pthread_t th[4]; job jb[4]; double t0 = now();
        for (int t = 0; t < n; ++t) { jb[t].t = t; jb[t].n = n; pthread_create(&th[t], NULL, work, &jb[t]); }
        for (int t = 0; t < n; ++t) pthread_join(th[t], NULL);
        double t1 = now() - t0; if (t1 < best) best = t1;
      }
      printf("%s %d thread(s): %.2f GB/s\n", strided ? "column ranges of 5,632-byte rows" : "contiguous parts", n, total / best / 1e9);
    }
  return 0;
}
