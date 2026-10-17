/* BF16-1 (NDT-BF16): the step's GEMM shapes at THREADS threads, the bf16 contract (the pooled entry:
 * the widest admitted instance; or VARIANT alone, on the calling thread) against the blocked FP32
 * kernel (kblock+...), the median of REPS calls of each, alternated call by call (README.json).
 *
 *   gcc -std=gnu11 -O3 -ffp-contract=off -fno-math-errno -I runtime/c gemm.c runtime/c/bcir_tensor.c \
 *       -lm -pthread
 *   ./a.out [threads] [reps] [variant] */
#include "bcir_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 4, reps = argc > 2 ? atoi(argv[2]) : 15, variant = argc > 3 ? atoi(argv[3]) : 0;
  static const struct { const char *name; int ta, tb; size_t m, n, k; } S[] = {
    {"fwd q   x.Wq^T", 0, 1, 1024, 384, 384}, {"fwd k   x.Wk^T", 0, 1, 1024, 192, 384},
    {"fwd gate", 0, 1, 1024, 1024, 384},      {"fwd down", 0, 1, 1024, 384, 1024},
    {"fwd head", 0, 1, 1024, 260, 384},
    {"dx q    dy.Wq", 0, 0, 1024, 384, 384},  {"dx k    dy.Wk", 0, 0, 1024, 384, 192},
    {"dx gate", 0, 0, 1024, 384, 1024},       {"dx down", 0, 0, 1024, 1024, 384},
    {"dW q    dy^T.x", 1, 0, 384, 384, 1024}, {"dW k", 1, 0, 192, 384, 1024},
    {"dW gate", 1, 0, 1024, 384, 1024},       {"dW down", 1, 0, 384, 1024, 1024},
  };
  size_t pb = bcir_tensor_pool_size((unsigned)threads); void *pm = aligned_alloc(64, (pb + 63) / 64 * 64);
  bcir_tensor_pool *pool = bcir_tensor_pool_init(pm, pb, (unsigned)threads);
  double tot_b = 0, tot_f = 0;
  printf("bf16 variants %d, threads %d\n", bcir_tensor_mm_bf16_variants(), threads);
  for (size_t s = 0; s < sizeof S / sizeof S[0]; s++) {
    size_t m = S[s].m, n = S[s].n, k = S[s].k, ws = bcir_tensor_mm_bf16_workspace(m, n, k);
    float *a = malloc(m * k * 4), *b = malloc(k * n * 4), *c = malloc(m * n * 4);
    void *w = aligned_alloc(64, (ws + 63) / 64 * 64);
    for (size_t i = 0; i < m * k; i++) a[i] = (float)((i * 2654435761u % 1000) - 500) * 1e-3f;
    for (size_t i = 0; i < k * n; i++) b[i] = (float)((i * 40503u % 1000) - 500) * 1e-3f;
    double tb[64], tf[64];
    for (int r = 0; r < reps; r++) {
      memset(c, 0, m * n * 4);
      double t0 = now();
      if (variant) bcir_tensor_mm_bf16_variant(variant, S[s].ta, S[s].tb, m, n, k, 1.0f, a, b, 0.0f, c, w, ws);
      else bcir_tensor_mm_bf16_pool(pool, S[s].ta, S[s].tb, m, n, k, 1.0f, a, b, 0.0f, c, w, ws);
      tb[r] = now() - t0;
      t0 = now();
      bcir_tensor_mm_kblock_pool(pool, S[s].ta, S[s].tb, m, n, k, 1.0f, a, b, 0.0f, c);
      tf[r] = now() - t0;
    }
    qsort(tb, (size_t)reps, sizeof tb[0], cmpd); qsort(tf, (size_t)reps, sizeof tf[0], cmpd);
    double fl = 2.0 * m * n * k;
    printf("%-16s %5zu %5zu %5zu  bf16 %7.3f ms %6.0f GF/s | fp32 %7.3f ms %6.0f GF/s | x%.2f\n", S[s].name, m, n, k,
           tb[reps / 2] * 1e3, fl / tb[reps / 2] / 1e9, tf[reps / 2] * 1e3, fl / tf[reps / 2] / 1e9, tf[reps / 2] / tb[reps / 2]);
    tot_b += tb[reps / 2]; tot_f += tf[reps / 2];
    free(a); free(b); free(c); free(w);
  }
  printf("sum of medians: bf16 %.3f ms, fp32 %.3f ms, x%.2f\n", tot_b * 1e3, tot_f * 1e3, tot_f / tot_b);
  bcir_tensor_pool_stop(pool); free(pm);
  return 0;
}
