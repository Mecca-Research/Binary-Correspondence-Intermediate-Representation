/* DECODE-ROW's kernel survey: gqa_row's committed form against row-major and register-blocked forms of its
 * scores and context, each chain the committed order; byte for byte over random rows, then timed at the 860K
 * decoder's shapes (4 heads of 32, 2 kv heads, contexts 1..T).  usage: gqa_bench T REPS */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void row_fn(const double *, const double *, const double *, int, int, int, int, int, double, double *,
                    double *);

static void softmax(double *s, int t_len) {
  double m = s[0];
  for (int j = 1; j < t_len; j++) if (s[j] > m) m = s[j];
  double sum = 0.0;
  for (int j = 0; j < t_len; j++) { s[j] = exp(s[j] - m); sum += s[j]; }
  if (sum == 0.0) sum = 1.0;
  for (int j = 0; j < t_len; j++) s[j] /= sum;
}

static void row_old(const double *qh, const double *k_rows, const double *v_rows, int t_len, int keep,
                    int n_kv_heads, int g, int d_k, double sc, double *s, double *o) {
  for (int j = 0; j < t_len; j++) {
    if (j >= keep) { s[j] = -INFINITY; continue; }
    const double *kj = k_rows + ((size_t)j * n_kv_heads + g) * d_k;
    double acc = 0.0;
    for (int c = 0; c < d_k; c++) acc += qh[c] * kj[c];
    s[j] = acc * sc;
  }
  softmax(s, t_len);
  for (int c = 0; c < d_k; c++) {
    double acc = 0.0;
    for (int j = 0; j < t_len; j++) acc += s[j] * v_rows[((size_t)j * n_kv_heads + g) * d_k + c];
    o[c] = acc;
  }
}

static void scores1(const double *qh, const double *kg, size_t stride, int t_len, int keep, int d_k, double sc,
                    double *s) {
  for (int j = 0; j < t_len; j++) {
    if (j >= keep) { s[j] = -INFINITY; continue; }
    const double *kj = kg + (size_t)j * stride;
    double acc = 0.0;
    for (int c = 0; c < d_k; c++) acc += qh[c] * kj[c];
    s[j] = acc * sc;
  }
}

static void scores4(const double *qh, const double *kg, size_t stride, int t_len, int keep, int d_k, double sc,
                    double *s) {
  int scored = keep < t_len ? keep : t_len, j = 0;
  for (; scored - j >= 4; j += 4) {
    const double *k0 = kg + (size_t)j * stride, *k1 = k0 + stride, *k2 = k1 + stride, *k3 = k2 + stride;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (int c = 0; c < d_k; c++) {
      double qc = qh[c];
      a0 += qc * k0[c];
      a1 += qc * k1[c];
      a2 += qc * k2[c];
      a3 += qc * k3[c];
    }
    s[j] = a0 * sc; s[j + 1] = a1 * sc; s[j + 2] = a2 * sc; s[j + 3] = a3 * sc;
  }
  for (; j < scored; j++) {
    const double *kj = kg + (size_t)j * stride;
    double acc = 0.0;
    for (int c = 0; c < d_k; c++) acc += qh[c] * kj[c];
    s[j] = acc * sc;
  }
  for (; j < t_len; j++) s[j] = -INFINITY;
}

static void scores8(const double *qh, const double *kg, size_t stride, int t_len, int keep, int d_k, double sc,
                    double *s) {
  int scored = keep < t_len ? keep : t_len, j = 0;
  for (; scored - j >= 8; j += 8) {
    const double *k0 = kg + (size_t)j * stride;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;
    for (int c = 0; c < d_k; c++) {
      double qc = qh[c];
      a0 += qc * k0[c];
      a1 += qc * k0[stride + c];
      a2 += qc * k0[2 * stride + c];
      a3 += qc * k0[3 * stride + c];
      a4 += qc * k0[4 * stride + c];
      a5 += qc * k0[5 * stride + c];
      a6 += qc * k0[6 * stride + c];
      a7 += qc * k0[7 * stride + c];
    }
    s[j] = a0 * sc; s[j + 1] = a1 * sc; s[j + 2] = a2 * sc; s[j + 3] = a3 * sc;
    s[j + 4] = a4 * sc; s[j + 5] = a5 * sc; s[j + 6] = a6 * sc; s[j + 7] = a7 * sc;
  }
  scores4(qh, kg + (size_t)j * stride, stride, t_len - j, keep - j, d_k, sc, s + j);
}

static void ctx_rowmajor(const double *s, const double *vg, size_t stride, int t_len, int d_k, double *o) {
  for (int c = 0; c < d_k; c++) o[c] = 0.0;
  for (int j = 0; j < t_len; j++) {
    const double *vj = vg + (size_t)j * stride;
    double sj = s[j];
    for (int c = 0; c < d_k; c++) o[c] += sj * vj[c];
  }
}

static void ctx_blk8(const double *s, const double *vg, size_t stride, int t_len, int d_k, double *o) {
  int c = 0;
  for (; d_k - c >= 8; c += 8) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;
    const double *vj = vg + c;
    for (int j = 0; j < t_len; j++, vj += stride) {
      double sj = s[j];
      a0 += sj * vj[0];
      a1 += sj * vj[1];
      a2 += sj * vj[2];
      a3 += sj * vj[3];
      a4 += sj * vj[4];
      a5 += sj * vj[5];
      a6 += sj * vj[6];
      a7 += sj * vj[7];
    }
    o[c] = a0; o[c + 1] = a1; o[c + 2] = a2; o[c + 3] = a3;
    o[c + 4] = a4; o[c + 5] = a5; o[c + 6] = a6; o[c + 7] = a7;
  }
  for (; c < d_k; c++) {
    double acc = 0.0;
    for (int j = 0; j < t_len; j++) acc += s[j] * vg[(size_t)j * stride + c];
    o[c] = acc;
  }
}

static void ctx_blk4(const double *s, const double *vg, size_t stride, int t_len, int d_k, double *o) {
  int c = 0;
  for (; d_k - c >= 4; c += 4) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    const double *vj = vg + c;
    for (int j = 0; j < t_len; j++, vj += stride) {
      double sj = s[j];
      a0 += sj * vj[0];
      a1 += sj * vj[1];
      a2 += sj * vj[2];
      a3 += sj * vj[3];
    }
    o[c] = a0; o[c + 1] = a1; o[c + 2] = a2; o[c + 3] = a3;
  }
  for (; c < d_k; c++) {
    double acc = 0.0;
    for (int j = 0; j < t_len; j++) acc += s[j] * vg[(size_t)j * stride + c];
    o[c] = acc;
  }
}

#define VARIANT(name, SC, CX)                                                                                      \
  static void name(const double *qh, const double *k_rows, const double *v_rows, int t_len, int keep,           \
                   int n_kv_heads, int g, int d_k, double sc, double *s, double *o) {                            \
    size_t stride = (size_t)n_kv_heads * (size_t)d_k;                                                           \
    SC(qh, k_rows + (size_t)g * d_k, stride, t_len, keep, d_k, sc, s);                                           \
    softmax(s, t_len);                                                                                           \
    CX(s, v_rows + (size_t)g * d_k, stride, t_len, d_k, o);                                                      \
  }
VARIANT(row_s1_rowmajor, scores1, ctx_rowmajor)
VARIANT(row_s1_blk8, scores1, ctx_blk8)
VARIANT(row_s4_blk8, scores4, ctx_blk8)
VARIANT(row_s8_blk8, scores8, ctx_blk8)
VARIANT(row_s4_blk4, scores4, ctx_blk4)
VARIANT(row_s8_blk4, scores8, ctx_blk4)

static uint64_t rng = 0x9E3779B97F4A7C15u;
static double rnd(void) {
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return ((double)(rng >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
  int T = argc > 1 ? atoi(argv[1]) : 205, reps = argc > 2 ? atoi(argv[2]) : 15;
  struct { const char *name; row_fn *f; } v[] = {{"old", row_old}, {"s1_rowmajor", row_s1_rowmajor},
    {"s1_blk8", row_s1_blk8}, {"s4_blk8", row_s4_blk8}, {"s8_blk8", row_s8_blk8}, {"s4_blk4", row_s4_blk4},
    {"s8_blk4", row_s8_blk4}};
  int nv = (int)(sizeof v / sizeof v[0]);
  /* bytes: random shapes, keep in [1, t], every variant against old */
  long compared = 0;
  for (int trial = 0; trial < 3000; trial++) {
    int dks[] = {1, 2, 3, 4, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 48, 64, 65, 128};
    int dk = dks[trial % 20], nkv = 1 + (trial / 20) % 3, t = 1 + (trial * 7919) % 300;
    int g = trial % nkv, keep = 1 + (trial * 104729) % t;
    double scale = trial % 5 == 0 ? 1e-3 : trial % 5 == 1 ? 30.0 : 1.0;
    size_t kv = (size_t)t * nkv * dk;
    double *q = malloc(dk * sizeof *q), *k = malloc(kv * sizeof *k), *vv = malloc(kv * sizeof *vv);
    double *s0 = malloc(t * sizeof *s0), *s1 = malloc(t * sizeof *s1), *o0 = malloc(dk * sizeof *o0),
           *o1 = malloc(dk * sizeof *o1);
    for (int i = 0; i < dk; i++) q[i] = rnd() * scale;
    for (size_t i = 0; i < kv; i++) { k[i] = rnd() * scale; vv[i] = rnd() * (i % 11 == 0 ? 1e300 : 1.0); }
    if (trial % 17 == 0) vv[kv / 2] = 0.0, k[kv / 3] = -0.0;
    double sc = 1.0 / sqrt((double)dk);
    row_old(q, k, vv, t, keep, nkv, g, dk, sc, s0, o0);
    for (int i = 1; i < nv; i++) {
      memset(o1, 0x5A, dk * sizeof *o1);
      v[i].f(q, k, vv, t, keep, nkv, g, dk, sc, s1, o1);
      if (memcmp(o0, o1, dk * sizeof *o0) || memcmp(s0, s1, t * sizeof *s0)) {
        printf("DIFF %s trial %d dk %d nkv %d t %d keep %d\n", v[i].name, trial, dk, nkv, t, keep);
        return 1;
      }
      compared++;
    }
    free(q); free(k); free(vv); free(s0); free(s1); free(o0); free(o1);
  }
  /* timing: 4 heads of 32 over 2 kv heads, every context 1..T (a request's decode steps), 4 layers */
  int nh = 4, nkv = 2, dk = 32, layers = 4;
  size_t kv = (size_t)T * nkv * dk;
  double *q = malloc(nh * dk * sizeof *q), *k = malloc(layers * kv * sizeof *k),
         *vv = malloc(layers * kv * sizeof *vv), *s = malloc(T * sizeof *s), *o = malloc(nh * dk * sizeof *o);
  for (int i = 0; i < nh * dk; i++) q[i] = rnd();
  for (size_t i = 0; i < layers * kv; i++) { k[i] = rnd(); vv[i] = rnd(); }
  double sc = 1.0 / sqrt((double)dk), ms[16][64];
  volatile double sink = 0.0;
  for (int r = -1; r < reps; r++)
    for (int i = 0; i < nv; i++) {
      int at = (i + (r < 0 ? 0 : r)) % nv;
      double t0 = now();
      for (int t = 1; t <= T; t++)
        for (int l = 0; l < layers; l++)
          for (int h = 0; h < nh; h++)
            v[at].f(q + h * dk, k + l * kv, vv + l * kv, t, t, nkv, h / (nh / nkv), dk, sc, s, o + h * dk);
      if (r >= 0) ms[at][r] = (now() - t0) * 1e3;
      sink += o[0];
    }
  printf("{\"compared\":%ld,\"T\":%d,\"reps\":%d", compared, T, reps);
  for (int i = 0; i < nv; i++) {
    qsort(ms[i], (size_t)reps, sizeof(double), cmp);
    printf(",\"%s_ms\":%.4f", v[i].name, ms[i][reps / 2]);
  }
  printf("}\n");
  return sink == 12345.0;
}
