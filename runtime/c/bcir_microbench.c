/* BCIR bare-metal cost microbench (the measurement half of the L1 calibration
 * loop). Times four access regimes over a cache-defeating buffer -- streaming,
 * strided, random (gather), pure compute -- and prints a frozen Q8 cost table as
 * JSON matching bcir.kbcir.microbench.CalibratedProfile. Unlike the Python
 * stdlib harness (which measures through the interpreter, so cache effects
 * collapse), this runs on bare metal, so the gather ratio reflects real latency.
 *
 * Deterministic access orders (a seeded LCG permutation; only the timing varies).
 * This is offline (L2/L3) tooling -- never the hot path; the planner consumes
 * only the frozen table it prints.
 *
 *   cc -O2 -std=c11 bcir_microbench.c -o mb && ./mb [n] [repeats] [cal_gen]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Q8 256

static uint64_t g_state = 0xBC12u;
static uint64_t lcg(void) {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return g_state;
}

static long now_ns(void) {
  /* ISO C11 timespec_get (no POSIX feature macros needed; portable under -std=c11). */
  struct timespec t;
  timespec_get(&t, TIME_UTC);
  return (long)t.tv_sec * 1000000000L + (long)t.tv_nsec;
}

static int cmp_long(const void *a, const void *b) {
  long x = *(const long *)a, y = *(const long *)b;
  return (x > y) - (x < y);
}

static long median(long *xs, int k) {
  qsort(xs, (size_t)k, sizeof(long), cmp_long);
  return xs[k / 2];
}

/* one pass touching n indices; volatile sink keeps the loop live under -O2. */
static volatile double g_sink = 0.0;

static long pass_stream(const double *buf, size_t n) {
  long t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[k];
  long dt = now_ns() - t0;
  g_sink += s;
  return dt < 1 ? 1 : dt;
}

static long pass_strided(const double *buf, size_t n, size_t stride) {
  long t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[(k * stride) % n];
  long dt = now_ns() - t0;
  g_sink += s;
  return dt < 1 ? 1 : dt;
}

static long pass_random(const double *buf, const size_t *perm, size_t n) {
  long t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[perm[k]];
  long dt = now_ns() - t0;
  g_sink += s;
  return dt < 1 ? 1 : dt;
}

static long pass_compute(size_t n) {
  long t0 = now_ns();
  double s = 1.0;
  for (size_t k = 0; k < n; ++k) s = s * 1.0000001 + 0.5;
  long dt = now_ns() - t0;
  g_sink += s;
  return dt < 1 ? 1 : dt;
}

/* q8(ns) = max(256, ns*256/base): a regime is never cheaper than streaming. */
static long q8(long ns, long base) {
  long v = base > 0 ? (ns * Q8) / base : Q8;
  return v < Q8 ? Q8 : v;
}

int main(int argc, char **argv) {
  size_t n = (argc > 1) ? (size_t)strtoull(argv[1], NULL, 10) : (1u << 22);
  int repeats = (argc > 2) ? atoi(argv[2]) : 5;
  long cal_gen = (argc > 3) ? atol(argv[3]) : 1;
  if (repeats < 1) repeats = 1;
  if (n < 2) n = 2;   /* need >=2 elements: the Fisher-Yates loop (n-1 down to 1) underflows and pass_strided does % n */

  double *buf = malloc(n * sizeof *buf);
  size_t *perm = malloc(n * sizeof *perm);
  long *tmp = malloc((size_t)repeats * sizeof *tmp);
  if (!buf || !perm || !tmp) { fprintf(stderr, "oom\n"); return 2; }

  for (size_t i = 0; i < n; ++i) { buf[i] = (double)(i % 97); perm[i] = i; }
  for (size_t i = n - 1; i > 0; --i) {           /* seeded Fisher-Yates */
    size_t j = (size_t)(lcg() % (i + 1));
    size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }

  for (int r = 0; r < repeats; ++r) tmp[r] = pass_stream(buf, n);
  long t_stream = median(tmp, repeats);
  for (int r = 0; r < repeats; ++r) tmp[r] = pass_strided(buf, n, 16);
  long t_strided = median(tmp, repeats);
  for (int r = 0; r < repeats; ++r) tmp[r] = pass_random(buf, perm, n);
  long t_random = median(tmp, repeats);
  for (int r = 0; r < repeats; ++r) tmp[r] = pass_compute(n);
  long t_compute = median(tmp, repeats);

  printf("{\n");
  printf("  \"name\": \"native\",\n");
  printf("  \"cal_gen\": %ld,\n", cal_gen);
  printf("  \"samples\": %d,\n", repeats);
  printf("  \"provenance\": \"native microbench (bare-metal) n=%zu repeats=%d\",\n", n, repeats);
  printf("  \"stream_q8\": %d,\n", Q8);
  printf("  \"strided_q8\": %ld,\n", q8(t_strided, t_stream));
  printf("  \"random_q8\": %ld,\n", q8(t_random, t_stream));
  printf("  \"compute_q8\": %ld\n", q8(t_compute, t_stream));
  printf("}\n");

  free(buf); free(perm); free(tmp);
  return 0;
}
