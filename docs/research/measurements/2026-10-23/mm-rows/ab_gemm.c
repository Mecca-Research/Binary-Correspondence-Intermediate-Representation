/* MM-ROWS's A/B: the k-blocked GEMM entries of two builds of bcir_tensor.c (BASE, PROTO) in one process, arms base,
 * proto and base again (A/A), on one set of buffers, each arm with its own pool made outside the clock; per case a
 * warm round, then ROUNDS rounds in rotated arm order, each arm's time the mean of REPS calls; ratios over base's
 * time in the same round, the geometric mean and its 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)).
 * A case is  g:TA,TB,M,N,K  (one GEMM), p:CALL:CALL  (a pair job: bcir_tensor_mm_pair_pool), or
 * cN:CALL:QxCALL  (a group job: N chained copies of the first call into one C, Q independent copies of the second,
 * each into its own C) -- every call at alpha 1, the chain's first at beta 0 and the rest at beta 1.
 * usage: ab_gemm BASE PROTO ROUNDS THREADS REPS CASE...   (THREADS 1: the serial entry, no pool)  -> one JSON line a case */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct opaque_pool pool_t;
typedef struct call_t { int ta, tb; size_t m, n, k; float alpha; const float *a, *b; float beta; float *c; } call_t;  /* bcir_tensor_mm_call */
typedef void (*kb_fn)(pool_t *, int, int, size_t, size_t, size_t, float, const float *, const float *, float, float *);
typedef void (*pair_fn)(pool_t *, int, const call_t *, const call_t *);
typedef void (*group_fn)(pool_t *, int, const call_t *, size_t, const call_t *, size_t);
typedef size_t (*psize_fn)(unsigned);
typedef pool_t *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(pool_t *);
enum { ARMS = 3, MAXR = 200, MAXQ = 4 };
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void *alloc64(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 64)) return NULL; memset(p, 0, n ? n : 64); return p; }
static void *sym(void *h, const char *n) { void *f = dlsym(h, n); if (!f) { fprintf(stderr, "missing %s\n", n); exit(1); } return f; }
static void interval(const double *r, int n, double out[3]) {
  static const double t95[] = {0, 0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
      2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069};
  double m = 0, sd = 0, h; int i;
  for (i = 0; i < n; i++) m += log(r[i]);
  m /= n;
  for (i = 0; i < n; i++) sd += (log(r[i]) - m) * (log(r[i]) - m);
  sd = sqrt(sd / (n - 1)); h = (n <= 24 ? t95[n] : 1.96 + 2.4 / (n - 1)) * sd / sqrt((double)n);
  out[0] = exp(m); out[1] = exp(m - h); out[2] = exp(m + h);
}
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(const double *x, int n) { double s[MAXR]; memcpy(s, x, n * sizeof(double)); qsort(s, n, sizeof(double), cmp); return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }
static int parse_call(const char *s, call_t *c) {
  int ta, tb; unsigned long m, n, k;
  if (sscanf(s, "%d,%d,%lu,%lu,%lu", &ta, &tb, &m, &n, &k) != 5 || (ta & ~1) || (tb & ~1) || !m || !n) return 0;
  c->ta = ta; c->tb = tb; c->m = m; c->n = n; c->k = k; c->alpha = 1.0f; c->beta = 0.0f; return 1;
}
static float *fill(size_t n, float f) { float *p = alloc64(n * 4); size_t i; if (!p) exit(1); for (i = 0; i < n; i++) p[i] = sinf((float)i * f) * 0.5f; return p; }
int main(int argc, char **argv) {
  void *h[2]; kb_fn kb[2]; pair_fn pr[2]; group_fn gr[2]; psize_fn psize; pinit_fn pinit; pstop_fn pstop;
  int rounds, reps, cs, r, j, arm, q, nchain, ny; unsigned threads; pool_t *pool[2] = {NULL, NULL};
  static double ms[ARMS][MAXR];
  if (argc < 7) { fprintf(stderr, "usage: ab_gemm BASE PROTO ROUNDS THREADS REPS CASE...\n"); return 2; }
  for (j = 0; j < 2; j++) {
    h[j] = dlopen(argv[1 + j], RTLD_NOW | RTLD_LOCAL); if (!h[j]) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    *(void **)&kb[j] = sym(h[j], "bcir_tensor_mm_kblock_pool");
    *(void **)&pr[j] = sym(h[j], "bcir_tensor_mm_pair_pool");
    *(void **)&gr[j] = sym(h[j], "bcir_tensor_mm_group_pool");
  }
  rounds = atoi(argv[3]); threads = (unsigned)atoi(argv[4]); reps = atoi(argv[5]);
  if (rounds < 2 || rounds > MAXR || !threads || reps < 1) return 2;
  if (threads > 1) for (j = 0; j < 2; j++) {
    size_t need; void *mem;
    *(void **)&psize = sym(h[j], "bcir_tensor_pool_size"); *(void **)&pinit = sym(h[j], "bcir_tensor_pool_init");
    need = psize(threads); mem = alloc64(need); pool[j] = pinit(mem, need, threads);
    if (!pool[j]) { fprintf(stderr, "pool refused\n"); return 1; }
  }
  for (cs = 6; cs < argc; cs++) {
    const char *spec = argv[cs]; call_t x, y, chain[MAXQ], ind[MAXQ]; char kind = spec[0]; const char *p1, *p2;
    float *a, *b, *a2 = NULL, *b2 = NULL, *c, *cy[MAXQ] = {0};
    nchain = 0; ny = 0;
    if (kind == 'g') { if (!parse_call(spec + 2, &x)) return 2; }
    else if (kind == 'p') { p1 = spec + 2; p2 = strchr(p1, ':'); if (!p2 || !parse_call(p1, &x) || !parse_call(p2 + 1, &y)) return 2; }
    else if (kind == 'c') {
      nchain = atoi(spec + 1); p1 = strchr(spec, ':'); p2 = p1 ? strchr(p1 + 1, ':') : NULL;
      if (!p1 || !p2 || nchain < 1 || nchain > MAXQ || !parse_call(p1 + 1, &x)) return 2;
      ny = atoi(p2 + 1); p2 = strchr(p2, 'x'); if (!p2 || ny < 1 || ny > MAXQ || !parse_call(p2 + 1, &y)) return 2;
    } else return 2;
    a = fill(x.m * (x.k ? x.k : 1), 0.37f); b = fill((x.k ? x.k : 1) * x.n, 0.53f); c = fill(x.m * x.n, 0.11f);
    x.a = a; x.b = b; x.c = c;
    if (kind != 'g') {
      a2 = fill(y.m * (y.k ? y.k : 1), 0.29f); b2 = fill((y.k ? y.k : 1) * y.n, 0.71f); y.a = a2; y.b = b2;
      for (q = 0; q < (ny ? ny : 1); q++) cy[q] = fill(y.m * y.n, 0.13f);
      y.c = cy[0];
    }
    for (q = 0; q < nchain; q++) { chain[q] = x; chain[q].beta = q ? 1.0f : 0.0f; }
    for (q = 0; q < ny; q++) { ind[q] = y; ind[q].c = cy[q]; }
    for (r = -1; r < rounds; r++) for (j = 0; j < ARMS; j++) {
      int lib, it; double t0;
      arm = (j + (r < 0 ? 0 : r)) % ARMS; lib = arm == 1 ? 1 : 0;
      t0 = now();
      for (it = 0; it < reps; it++) {
        if (kind == 'g') kb[lib](pool[lib], x.ta, x.tb, x.m, x.n, x.k, 1.0f, x.a, x.b, 0.0f, x.c);
        else if (kind == 'p') pr[lib](pool[lib], 2, &x, &y);
        else gr[lib](pool[lib], 2, chain, (size_t)nchain, ind, (size_t)ny);
      }
      if (r >= 0) ms[arm][r] = (now() - t0) * 1e3 / reps;
    }
    {
      double ratio[MAXR], ci[3], gf; size_t fl = 2 * x.m * x.n * x.k * (size_t)(nchain ? nchain : 1);
      if (kind != 'g') fl += 2 * y.m * y.n * y.k * (size_t)(ny ? ny : 1);
      gf = fl / (median(ms[0], rounds) * 1e-3) / 1e9;
      printf("{\"case\":\"%s\",\"threads\":%u,\"rounds\":%d,\"reps\":%d,\"base_ms\":%.4f,\"base_gflops\":%.1f", spec, threads, rounds, reps, median(ms[0], rounds), gf);
      for (arm = 1; arm < ARMS; arm++) {
        for (r = 0; r < rounds; r++) ratio[r] = ms[arm][r] / ms[0][r];
        interval(ratio, rounds, ci);
        printf(",\"%s\":{\"ms\":%.4f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", arm == 1 ? "proto" : "A/A", median(ms[arm], rounds), ci[0], ci[1], ci[2]);
      }
      printf(",\"raw_ms\":{\"base\":[");
      for (arm = 0; arm < ARMS; arm++) {
        if (arm) printf("],\"%s\":[", arm == 1 ? "proto" : "A/A");
        for (r = 0; r < rounds; r++) printf(r ? ",%.5f" : "%.5f", ms[arm][r]);
      }
      printf("]}}\n"); fflush(stdout);
    }
    free(a); free(b); free(c); free(a2); free(b2); for (q = 0; q < MAXQ; q++) free(cy[q]);
  }
  for (j = 0; j < 2; j++) if (pool[j]) { *(void **)&pstop = sym(h[j], "bcir_tensor_pool_stop"); pstop(pool[j]); }
  return 0;
}
