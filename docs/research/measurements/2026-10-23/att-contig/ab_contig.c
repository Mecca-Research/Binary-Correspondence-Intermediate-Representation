/* ATT-CONTIG's A/B: the binary32 attention's forward and backward, arms in one process on one set of
 * buffers, rounds in rotated order (a warm-up round first), each arm's own pool made outside the clock:
 *   before   the tree before ATT-CONTIG, its entries (the trainer's layout: B x H heads a row)
 *   after    the tree after it, its workspace entries lent the pool's need (bcir_tensor_attention_pool_workspace)
 *   in-place the tree after it, its entries without a workspace (the rows in place: the path below the need)
 *   bound    the tree after it on the contiguous layout (B*H batches of one head, nothing to copy): the most
 *            any layout change could take
 *   A/A      `before` again
 * Each ratio is an arm's time over `before`'s in the same round; the interval the geometric mean's 95%
 * (Student t over the rounds' log ratios).
 * usage: ab_contig BEFORE_LIB AFTER_LIB ROUNDS B T H D THREADS CONTRACT  -> one JSON object on stdout */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct opaque_pool pool_t;
typedef int (*fwd_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *);
typedef int (*bwd_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, const float *, float *, float *, float *);
typedef int (*fwdw_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, float *, int *, void *, size_t);
typedef int (*bwdw_fn)(pool_t *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
    const float *, const float *, float *, const float *, float *, float *, float *, void *, size_t);
typedef size_t (*need_fn)(size_t, size_t, size_t, size_t, size_t, unsigned);
typedef size_t (*psize_fn)(unsigned);
typedef pool_t *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(pool_t *);
enum { ARMS = 5, MAXR = 64 };
static const char *const NAMES[ARMS] = {"before", "after", "in-place", "bound", "A/A"};
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void *alloc64(size_t bytes) { void *p = NULL; if (posix_memalign(&p, 64, bytes ? bytes : 64)) return NULL; memset(p, 0, bytes ? bytes : 64); return p; }
static void *sym(void *h, const char *n) { void *f = dlsym(h, n); if (!f) { fprintf(stderr, "missing %s\n", n); exit(1); } return f; }
static void interval(const double *r, int n, double out[3]) {
  /* Student t's two-sided 95% quantile for n observations (n-1 degrees of freedom): the table to n = 24,
   * then 1.96 + 2.4/(n-1), within 0.004 of the quantile from 25 to 64 observations */
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
int main(int argc, char **argv) {
  void *ha, *hb; fwd_fn fwd[2]; bwd_fn bwd[2]; fwdw_fn fwdw; bwdw_fn bwdw; need_fn need; psize_fn psize; pinit_fn pinit; pstop_fn pstop;
  int rounds, r, j, arm, bad = 0, ph; size_t b, t, nh, d, n, np, i, wsn = 0; unsigned threads, ct; unsigned char *wsb = NULL;
  static double ms[3][ARMS][MAXR]; float *q, *k, *v, *p, *y, *dy, *dq, *dk, *dv;
  if (argc != 10) { fprintf(stderr, "usage: ab_contig BEFORE_LIB AFTER_LIB ROUNDS B T H D THREADS CONTRACT\n"); return 2; }
  ha = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL); hb = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!ha || !hb) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  rounds = atoi(argv[3]); b = strtoul(argv[4], 0, 10); t = strtoul(argv[5], 0, 10); nh = strtoul(argv[6], 0, 10);
  d = strtoul(argv[7], 0, 10); threads = (unsigned)atoi(argv[8]); ct = (unsigned)atoi(argv[9]);
  if (rounds < 2 || rounds > MAXR || !b || !t || !nh || !d) return 2;
  *(void **)&fwd[0] = sym(ha, "bcir_tensor_attention_contract_variant_scan_pool");
  *(void **)&bwd[0] = sym(ha, "bcir_tensor_attention_backward_contract_variant_pool");
  *(void **)&fwd[1] = sym(hb, "bcir_tensor_attention_contract_variant_scan_pool");
  *(void **)&bwd[1] = sym(hb, "bcir_tensor_attention_backward_contract_variant_pool");
  *(void **)&fwdw = sym(hb, "bcir_tensor_attention_contract_variant_scan_ws_pool");
  *(void **)&bwdw = sym(hb, "bcir_tensor_attention_backward_contract_variant_ws_pool");
  *(void **)&need = sym(hb, "bcir_tensor_attention_pool_workspace");
  *(void **)&psize = sym(ha, "bcir_tensor_pool_size"); *(void **)&pinit = sym(ha, "bcir_tensor_pool_init");
  *(void **)&pstop = sym(ha, "bcir_tensor_pool_stop");
  n = b * t * nh * d; np = b * nh * t * t;
  q = alloc64(n * 4); k = alloc64(n * 4); v = alloc64(n * 4); y = alloc64(n * 4); p = alloc64(np * 4); dy = alloc64(n * 4);
  dq = alloc64(n * 4); dk = alloc64(n * 4); dv = alloc64(n * 4);
  if (!q || !k || !v || !y || !p || !dy || !dq || !dk || !dv) return 1;
  for (i = 0; i < n; i++) { q[i] = sinf((float)i * 0.37f) * 1.7f; dy[i] = cosf((float)i * 0.11f); k[i] = cosf((float)i * 0.53f) * 1.3f; v[i] = sinf((float)i * 0.29f) - 0.1f; }
  wsn = need(b, t, nh, nh, d, threads ? threads : 1);
  wsb = alloc64(wsn);
  for (r = -1; r < rounds; r++) for (j = 0; j < ARMS; j++) {
    pool_t *pool = NULL; void *mem = NULL; double t0, t1, t2;
    arm = r < 0 ? j : (j + r) % ARMS;
    if (threads > 1) { size_t ps = psize(threads); mem = alloc64(ps); pool = pinit(mem, ps, threads); }
    t0 = now();
    if (arm == 1) fwdw(pool, 0, ct, b, t, nh, nh, d, q, k, v, p, y, &bad, wsb, wsn);
    else if (arm == 3) fwd[1](pool, 0, ct, b * nh, t, 1, 1, d, q, k, v, p, y, &bad);
    else fwd[arm == 2](pool, 0, ct, b, t, nh, nh, d, q, k, v, p, y, &bad);
    t1 = now();
    if (arm == 1) bwdw(pool, 0, ct, b, t, nh, nh, d, q, k, v, p, dy, dq, dk, dv, wsb, wsn);
    else if (arm == 3) bwd[1](pool, 0, ct, b * nh, t, 1, 1, d, q, k, v, p, dy, dq, dk, dv);
    else bwd[arm == 2](pool, 0, ct, b, t, nh, nh, d, q, k, v, p, dy, dq, dk, dv);
    t2 = now();
    if (r >= 0) { ms[0][arm][r] = 1000 * (t1 - t0); ms[1][arm][r] = 1000 * (t2 - t1); ms[2][arm][r] = 1000 * (t2 - t0); }
    if (pool) pstop(pool);
    free(mem);
  }
  printf("{\"shape\":[%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"rounds\":%d,\"workspace\":%zu,\"arms\":{", b, t, nh, d,
      threads, ct, rounds, wsn);
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
