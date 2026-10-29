/* THP-0: the attention (the stored forward, then the backward) on one tensor unit with two sets of buffers --
 * "4k": posix_memalign'd and advised MADV_NOHUGEPAGE; "huge": 2 MiB-aligned and advised MADV_HUGEPAGE before first
 * touch -- arms 4k, huge and 4k again (A/A), rotated rounds after a warm one, each arm's pool made outside the clock;
 * ratios over 4k's time in the same round, the geometric mean and its 95% interval (Student t). The process's
 * AnonHugePages before and after the huge set is touched is printed (the witness that the advice took).
 * usage: ab_thp LIB ROUNDS B T H KH D THREADS CONTRACT */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
typedef struct opaque_pool pool_t;
typedef int (*fwdw_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *, void *, size_t);
typedef int (*bwdw_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, const float *, float *, float *, float *, void *, size_t);
typedef size_t (*need_fn)(size_t, size_t, size_t, size_t, size_t, unsigned);
typedef size_t (*psize_fn)(unsigned);
typedef pool_t *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(pool_t *);
enum { ARMS = 3, MAXR = 64 };
static const char *const NAMES[ARMS] = {"4k", "huge", "A/A"};
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void *alloc_pages(size_t n, int huge) {
  size_t al = huge ? (size_t)2 << 20 : 64, sz = (n + al - 1) / al * al; void *p = NULL;
  if (posix_memalign(&p, al, sz ? sz : al)) return NULL;
  if (madvise(p, sz, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE)) { /* 4k's tail may not be page-aligned: advise its pages */
    if (!huge) { char *a = (char *)(((size_t)p + 4095) & ~(size_t)4095); if (sz > 8192) madvise(a, (sz - 8192) & ~(size_t)4095, MADV_NOHUGEPAGE); }
  }
  memset(p, 0, sz ? sz : al);
  return p;
}
static long anon_huge_kb(void) {
  FILE *f = fopen("/proc/self/smaps_rollup", "r"); char line[256]; long kb = -1;
  if (!f) return -1;
  while (fgets(line, sizeof line, f)) if (!strncmp(line, "AnonHugePages:", 14)) kb = strtol(line + 14, NULL, 10);
  fclose(f); return kb;
}
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
typedef struct { float *q, *k, *v, *p, *y, *dy, *dq, *dk, *dv; void *ws; } bufs;
static int make(bufs *s, size_t nq, size_t nkv, size_t np, size_t wsn, int huge) {
  size_t i;
  s->q = alloc_pages(nq * 4, huge); s->k = alloc_pages(nkv * 4, huge); s->v = alloc_pages(nkv * 4, huge);
  s->y = alloc_pages(nq * 4, huge); s->p = alloc_pages(np * 4, huge); s->dy = alloc_pages(nq * 4, huge);
  s->dq = alloc_pages(nq * 4, huge); s->dk = alloc_pages(nkv * 4, huge); s->dv = alloc_pages(nkv * 4, huge);
  s->ws = alloc_pages(wsn, huge);
  if (!s->q || !s->k || !s->v || !s->y || !s->p || !s->dy || !s->dq || !s->dk || !s->dv || !s->ws) return 0;
  for (i = 0; i < nq; i++) { s->q[i] = sinf((float)i * 0.37f) * 1.7f; s->dy[i] = cosf((float)i * 0.11f); }
  for (i = 0; i < nkv; i++) { s->k[i] = cosf((float)i * 0.53f) * 1.3f; s->v[i] = sinf((float)i * 0.29f) - 0.1f; }
  return 1;
}
int main(int argc, char **argv) {
  void *h; fwdw_fn fw; bwdw_fn bw; need_fn need; psize_fn psize; pinit_fn pinit; pstop_fn pstop;
  int rounds, r, j, arm, bad = 0, ph; size_t b, t, nh, nk, d, nq, nkv, np, wsn; unsigned threads, ct;
  static double ms[3][ARMS][MAXR]; bufs set[2]; long kb0, kb1; size_t i; int same = 1;
  if (argc != 10) { fprintf(stderr, "usage: ab_thp LIB ROUNDS B T H KH D THREADS CONTRACT\n"); return 2; }
  h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  *(void **)&fw = sym(h, "bcir_tensor_attention_contract_variant_scan_ws_pool");
  *(void **)&bw = sym(h, "bcir_tensor_attention_backward_contract_variant_ws_pool");
  *(void **)&need = sym(h, "bcir_tensor_attention_pool_workspace");
  *(void **)&psize = sym(h, "bcir_tensor_pool_size"); *(void **)&pinit = sym(h, "bcir_tensor_pool_init");
  *(void **)&pstop = sym(h, "bcir_tensor_pool_stop");
  rounds = atoi(argv[2]); b = strtoul(argv[3], 0, 10); t = strtoul(argv[4], 0, 10); nh = strtoul(argv[5], 0, 10);
  nk = strtoul(argv[6], 0, 10); d = strtoul(argv[7], 0, 10); threads = (unsigned)atoi(argv[8]); ct = (unsigned)atoi(argv[9]);
  if (rounds < 2 || rounds > MAXR || !b || !t || !nh || !nk || !d) return 2;
  nq = b * t * nh * d; nkv = b * t * nk * d; np = b * nh * t * t;
  wsn = need(b, t, nh, nk, d, threads ? threads : 1);
  kb0 = anon_huge_kb();
  if (!make(&set[0], nq, nkv, np, wsn, 0)) return 1;
  kb1 = anon_huge_kb();
  if (!make(&set[1], nq, nkv, np, wsn, 1)) return 1;
  fprintf(stderr, "AnonHugePages: %ld kB before, %ld kB after the 4k set, %ld kB after the huge set\n", kb0, kb1, anon_huge_kb());
  for (r = -1; r < rounds; r++) for (j = 0; j < ARMS; j++) {
    pool_t *pool = NULL; void *mem = NULL; double t0, t1, t2; bufs *s;
    arm = r < 0 ? j : (j + r) % ARMS; s = &set[arm == 1];
    if (threads > 1) { size_t ps = psize(threads); mem = malloc(ps); pool = pinit(mem, ps, threads); }
    t0 = now();
    if (!fw(pool, 0, ct, b, t, nh, nk, d, s->q, s->k, s->v, s->p, s->y, &bad, s->ws, wsn)) return 3;
    t1 = now();
    if (!bw(pool, 0, ct, b, t, nh, nk, d, s->q, s->k, s->v, s->p, s->dy, s->dq, s->dk, s->dv, s->ws, wsn)) return 4;
    t2 = now();
    if (r >= 0) { ms[0][arm][r] = 1000 * (t1 - t0); ms[1][arm][r] = 1000 * (t2 - t1); ms[2][arm][r] = 1000 * (t2 - t0); }
    if (pool) pstop(pool);
    free(mem);
  }
  for (i = 0; i < nq; i++) same &= !memcmp(&set[0].dq[i], &set[1].dq[i], 4) && !memcmp(&set[0].y[i], &set[1].y[i], 4);
  for (i = 0; i < nkv; i++) same &= !memcmp(&set[0].dk[i], &set[1].dk[i], 4) && !memcmp(&set[0].dv[i], &set[1].dv[i], 4);
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"rounds\":%d,\"workspace\":%zu,\"same_bits\":%s,"
      "\"anon_huge_kb\":[%ld,%ld,%ld],\"arms\":{", b, t, nh, nk, d, threads, ct, rounds, wsn, same ? "true" : "false", kb0, kb1, anon_huge_kb());
  for (j = 0; j < ARMS; j++) {
    printf("%s\"%s\":{", j ? "," : "", NAMES[j]);
    for (ph = 0; ph < 3; ph++) {
      double ratio[MAXR], ci[3];
      printf("%s\"%s\":{\"median_ms\":%.4f,\"ms\":[", ph ? "," : "", ph == 0 ? "forward" : ph == 1 ? "backward" : "total",
          median(ms[ph][j], rounds));
      for (r = 0; r < rounds; r++) printf("%s%.4f", r ? "," : "", ms[ph][j][r]);
      printf("]");
      if (j) {
        for (r = 0; r < rounds; r++) ratio[r] = ms[ph][j][r] / ms[ph][0][r];
        interval(ratio, rounds, ci);
        printf(",\"ratio\":%.4f,\"interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
      }
      printf("}");
    }
    printf("}");
  }
  printf("}}\n");
  return 0;
}
