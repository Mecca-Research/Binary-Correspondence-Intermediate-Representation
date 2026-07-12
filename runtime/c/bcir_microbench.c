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
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
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

static uint64_t now_ns(void) {
  /* ISO C11 timespec_get (no POSIX feature macros needed; portable under -std=c11). */
  struct timespec t = {0, 0};
  if (timespec_get(&t, TIME_UTC) != TIME_UTC || t.tv_sec < 0 || t.tv_nsec < 0)
    return 0;
  uint64_t sec = (uint64_t)t.tv_sec, ns = (uint64_t)t.tv_nsec;
  if (sec > (UINT64_MAX - ns) / UINT64_C(1000000000)) return UINT64_MAX;
  return sec * UINT64_C(1000000000) + ns;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

static uint64_t median(uint64_t *xs, int k) {
  qsort(xs, (size_t)k, sizeof(uint64_t), cmp_u64);
  return xs[k / 2];
}

/* one pass touching n indices; volatile sink keeps the loop live under -O2. */
static volatile double g_sink = 0.0;

static uint64_t elapsed_since(uint64_t start) {
  uint64_t end = now_ns();
  return end > start ? end - start : 1;
}

static uint64_t pass_stream(const double *buf, size_t n) {
  uint64_t t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[k];
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

static uint64_t pass_strided(const double *buf, size_t n, size_t stride) {
  uint64_t t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[(k * stride) % n];
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

static uint64_t pass_random(const double *buf, const size_t *perm, size_t n) {
  uint64_t t0 = now_ns();
  double s = 0.0;
  for (size_t k = 0; k < n; ++k) s += buf[perm[k]];
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

static uint64_t pass_compute(size_t n) {
  uint64_t t0 = now_ns();
  double s = 1.0;
  for (size_t k = 0; k < n; ++k) s = s * 1.0000001 + 0.5;
  uint64_t dt = elapsed_since(t0);
  g_sink += s;
  return dt;
}

/* q8(ns) = max(256, ns*256/base): a regime is never cheaper than streaming. */
static uint64_t q8(uint64_t ns, uint64_t base) {
  uint64_t v = !base ? Q8 : ns > UINT64_MAX / Q8 ? UINT64_MAX : (ns * Q8) / base;
  return v < Q8 ? Q8 : v;
}

static int parse_u64(const char *s, uint64_t *out) {
  char *end; unsigned long long value;
  if(!s||!*s||*s=='-')return -1;
  errno=0;value=strtoull(s,&end,10);
  if(errno||*end)return -1;*out=(uint64_t)value;return 0;
}

static int parse_i64(const char *s, int64_t *out) {
  char *end; long long value;
  if(!s||!*s)return -1;
  errno=0;value=strtoll(s,&end,10);
  if(errno||*end)return -1;*out=(int64_t)value;return 0;
}

int main(int argc, char **argv) {
  uint64_t parsed_n=1u<<22,parsed_repeats=5;int64_t cal_gen=1;
  if((argc>1&&parse_u64(argv[1],&parsed_n))||(argc>2&&parse_u64(argv[2],&parsed_repeats))||
     (argc>3&&parse_i64(argv[3],&cal_gen))||argc>4||parsed_n>SIZE_MAX||
     parsed_repeats<1||parsed_repeats>INT_MAX){fprintf(stderr,"invalid arguments\n");return 2;}
  size_t n=(size_t)parsed_n;int repeats=(int)parsed_repeats;
  if (n < 2) n = 2;   /* need >=2 elements: the Fisher-Yates loop (n-1 down to 1) underflows and pass_strided does % n */

  if(n>SIZE_MAX/sizeof(double)||n>SIZE_MAX/sizeof(size_t)||
     (size_t)repeats>SIZE_MAX/sizeof(uint64_t)){fprintf(stderr,"requested benchmark is too large\n");return 2;}

  double *buf = malloc(n * sizeof *buf);
  size_t *perm = malloc(n * sizeof *perm);
  uint64_t *tmp = malloc((size_t)repeats * sizeof *tmp);
  if (!buf || !perm || !tmp) { fprintf(stderr, "oom\n"); free(buf); free(perm); free(tmp); return 2; }

  for (size_t i = 0; i < n; ++i) { buf[i] = (double)(i % 97); perm[i] = i; }
  for (size_t i = n - 1; i > 0; --i) {           /* seeded Fisher-Yates */
    size_t j = (size_t)(lcg() % (i + 1));
    size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }

  for (int r = 0; r < repeats; ++r) tmp[r] = pass_stream(buf, n);
  uint64_t t_stream = median(tmp, repeats);
  for (int r = 0; r < repeats; ++r) tmp[r] = pass_strided(buf, n, 16);
  uint64_t t_strided = median(tmp, repeats);
  for (int r = 0; r < repeats; ++r) tmp[r] = pass_random(buf, perm, n);
  uint64_t t_random = median(tmp, repeats);
  for (int r = 0; r < repeats; ++r) tmp[r] = pass_compute(n);
  uint64_t t_compute = median(tmp, repeats);

  printf("{\n");
  printf("  \"name\": \"native\",\n");
  printf("  \"cal_gen\": %" PRId64 ",\n", cal_gen);
  printf("  \"samples\": %d,\n", repeats);
  printf("  \"provenance\": \"native microbench (bare-metal) n=%zu repeats=%d\",\n", n, repeats);
  printf("  \"stream_q8\": %d,\n", Q8);
  printf("  \"strided_q8\": %" PRIu64 ",\n", q8(t_strided, t_stream));
  printf("  \"random_q8\": %" PRIu64 ",\n", q8(t_random, t_stream));
  printf("  \"compute_q8\": %" PRIu64 "\n", q8(t_compute, t_stream));
  printf("}\n");

  free(buf); free(perm); free(tmp);
  return 0;
}
