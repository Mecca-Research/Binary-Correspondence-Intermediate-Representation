/*===- test_decode.c - rung-5 decode-kernel differential harness --------------------------===
 * stdin: "rmsnorm rows dim" + rows*dim x + dim gamma  |  "rope rows dim base off" + rows*dim x
 *      | "embedding vocab dim n" + vocab*dim table + n ids
 *      | "gqa seq nh nkv dk" + q + k + v (pre-roped)  |  "gqa_row t nh nkv dk" + q_row + k + v
 *      | "order seed" (DECODE-ROW: the kernels against the one-chain loops, byte for byte)
 * stdout: the outputs, %.17g, space-separated (one line). cc test_decode.c bcir_decode.c -lm  */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_decode.h"

static double *rd(int n) {
  double *v = malloc((size_t)n * sizeof *v);
  for (int i = 0; i < n; i++) if (scanf("%lf", &v[i]) != 1) exit(2);
  return v;
}

/* The loops the kernels are held to, as bcir_decode.c wrote them before DECODE-ROW: one score row's
 * chain at a time, one context coordinate's chain at a time down every key row, and a rotation
 * computing its angles for every row it turns. DECODE-ROW's blocks and table must give these bytes. */
static void one_chain_row(const double *qh, const double *k_rows, const double *v_rows, int t_len,
                          int keep, int n_kv_heads, int g, int d_k, double sc, double *s, double *o) {
  for (int j = 0; j < t_len; j++) {
    if (j >= keep) { s[j] = -INFINITY; continue; }
    const double *kj = k_rows + ((size_t)j * n_kv_heads + g) * d_k;
    double acc = 0.0;
    for (int c = 0; c < d_k; c++) acc += qh[c] * kj[c];
    s[j] = acc * sc;
  }
  double m = s[0];
  for (int j = 1; j < t_len; j++) if (s[j] > m) m = s[j];
  double sum = 0.0;
  for (int j = 0; j < t_len; j++) { s[j] = exp(s[j] - m); sum += s[j]; }
  if (sum == 0.0) sum = 1.0;
  for (int j = 0; j < t_len; j++) s[j] /= sum;
  for (int c = 0; c < d_k; c++) {
    double acc = 0.0;
    for (int j = 0; j < t_len; j++) acc += s[j] * v_rows[((size_t)j * n_kv_heads + g) * d_k + c];
    o[c] = acc;
  }
}

static void one_chain_rope(const double *x, int rows, int dim, double base, int pos_offset, double *out) {
  for (int r = 0; r < rows; r++) {
    const double *row = x + (size_t)r * dim;
    double *o = out + (size_t)r * dim;
    for (int k = 0; k < dim / 2; k++) {
      double th = ((double)pos_offset + (double)r) * pow(base, -(2.0 * k) / dim);
      double c = cos(th), s = sin(th);
      double x0 = row[2 * k], x1 = row[2 * k + 1];
      o[2 * k] = x0 * c - x1 * s;
      o[2 * k + 1] = x0 * s + x1 * c;
    }
  }
}

static uint64_t order_state;
static double order_value(int range) { /* xorshift64: uniform in (-1, 1), scaled by the case's range */
  order_state ^= order_state << 13;
  order_state ^= order_state >> 7;
  order_state ^= order_state << 17;
  double u = (double)(order_state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
  switch (range) {
  case 1: return u * 1e-3;                           /* flat probabilities */
  case 2: return u * 40.0;                           /* scores past exp's range: zeros and ones */
  case 3: return (order_state & 7u) == 0 ? 0.0 : (order_state & 7u) == 1 ? -0.0 : u; /* signed zeros */
  case 4: return (order_state & 63u) == 0 ? INFINITY : u; /* the odd infinity: default NaNs */
  case 5: return ldexp(u, (int)(order_state % 2000u) - 1000); /* every exponent: subnormal products */
  default: return u;
  }
}

static int order_differs(const char *what, const void *got, const void *want, size_t bytes, int a, int b,
                         int c, int d, int range) {
  if (!memcmp(got, want, bytes)) return 0;
  printf("order: %s differs at %d %d %d %d range %d\n", what, a, b, c, d, range);
  return 1;
}

/* Every shape around the kernels' blocks (8 and 4 score rows, 8 context coordinates), every head grouping
 * and value range, the full causal entry point and the cached row; the rotation at many positions and
 * dimensions, whole and through the table a position. Exit 1 at the first byte that differs. */
static int order(unsigned seed) {
  static const int dks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 40, 64, 65, 128};
  static const int lens[] = {1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 24, 31, 33, 64, 100, 205, 257};
  static const int groups[][2] = {{1, 1}, {2, 1}, {4, 2}, {4, 4}, {6, 3}, {8, 2}};
  long compared = 0;
  order_state = 0x9E3779B97F4A7C15u ^ seed;
  for (int di = 0; di < (int)(sizeof dks / sizeof dks[0]); di++)
    for (int li = 0; li < (int)(sizeof lens / sizeof lens[0]); li++) {
      int dk = dks[di], t = lens[li], nh = groups[(di + li) % 6][0], nkv = groups[(di + li) % 6][1];
      int range = (di * 7 + li) % 6;
      size_t qn = (size_t)t * nh * dk, kvn = (size_t)t * nkv * dk;
      double *q = malloc(qn * sizeof *q), *k = malloc(kvn * sizeof *k), *v = malloc(kvn * sizeof *v);
      double *s = malloc((size_t)t * sizeof *s), *s1 = malloc((size_t)t * sizeof *s1);
      double *o = malloc(qn * sizeof *o), *o1 = malloc(qn * sizeof *o1);
      if (!q || !k || !v || !s || !s1 || !o || !o1) return 2;
      for (size_t i = 0; i < qn; i++) q[i] = order_value(range);
      for (size_t i = 0; i < kvn; i++) k[i] = order_value(range), v[i] = order_value(range == 2 ? 0 : range);
      double sc = 1.0 / sqrt((double)dk);
      /* the cached row: the last position's query, every head */
      memset(o, 0x5A, (size_t)nh * dk * sizeof *o);
      if (bcir_gqa_attention_row(q + (size_t)(t - 1) * nh * dk, k, v, t, nh, nkv, dk, s, o)) return 3;
      for (int h = 0; h < nh; h++)
        one_chain_row(q + ((size_t)(t - 1) * nh + h) * dk, k, v, t, t, nkv, h / (nh / nkv), dk, sc, s1,
                      o1 + (size_t)h * dk);
      if (order_differs("gqa_row", o, o1, (size_t)nh * dk * sizeof *o, dk, t, nh, nkv, range) ||
          order_differs("gqa_row scores", s, s1, (size_t)t * sizeof *s, dk, t, nh, nkv, range)) return 1;
      /* the full causal attention: row i keeps positions 0..i */
      memset(o, 0x5A, qn * sizeof *o);
      if (bcir_gqa_attention(q, k, v, t, nh, nkv, dk, s, o)) return 3;
      for (int h = 0; h < nh; h++)
        for (int i = 0; i < t; i++)
          one_chain_row(q + ((size_t)i * nh + h) * dk, k, v, t, i + 1, nkv, h / (nh / nkv), dk, sc, s1,
                        o1 + ((size_t)i * nh + h) * dk);
      if (order_differs("gqa", o, o1, qn * sizeof *o, dk, t, nh, nkv, range)) return 1;
      compared += 2;
      free(q), free(k), free(v), free(s), free(s1), free(o), free(o1);
    }
  /* the rotation: whole rows, and one row through the table of its position */
  static const int dims[] = {2, 4, 6, 8, 16, 30, 32, 64, 128, 256};
  static const double bases[] = {10000.0, 500000.0, 1.5, 1e6};
  static const int offsets[] = {0, 1, 2, 7, 31, 100, 4095, 65537, 1 << 20, INT_MAX - 300};
  for (int di = 0; di < (int)(sizeof dims / sizeof dims[0]); di++)
    for (int bi = 0; bi < 4; bi++)
      for (int pi = 0; pi < (int)(sizeof offsets / sizeof offsets[0]); pi++) {
        int dim = dims[di], rows = 1 + (di + pi) % 5, off = offsets[pi], range = (di + bi + pi) % 6;
        size_t n = (size_t)rows * dim;
        double *x = malloc(n * sizeof *x), *o = malloc(n * sizeof *o), *o1 = malloc(n * sizeof *o1);
        double *cs = malloc((size_t)dim * sizeof *cs);
        if (!x || !o || !o1 || !cs) return 2;
        for (size_t i = 0; i < n; i++) x[i] = order_value(range == 4 ? 0 : range);
        one_chain_rope(x, rows, dim, bases[bi], off, o1);
        memset(o, 0x5A, n * sizeof *o);
        bcir_rope(x, rows, dim, bases[bi], off, o);
        if (order_differs("rope", o, o1, n * sizeof *o, dim, bi, off, rows, range)) return 1;
        for (int r = 0; r < rows; r++) {
          memset(cs, 0x5A, (size_t)dim * sizeof *cs);
          bcir_rope_table(dim, bases[bi], (double)(off + r), cs);
          memset(o, 0x5A, (size_t)dim * sizeof *o);
          bcir_rope_apply(x + (size_t)r * dim, dim, cs, o);
          if (order_differs("rope table", o, o1 + (size_t)r * dim, (size_t)dim * sizeof *o, dim, bi, off, r,
                            range)) return 1;
        }
        compared += 1 + rows;
        free(x), free(o), free(o1), free(cs);
      }
  printf("order: %ld comparisons, every byte the one-chain loops'\n", compared);
  return 0;
}

int main(void) {
  char kind[16];
  if (scanf("%15s", kind) != 1) return 2;
  if (!strcmp(kind, "order")) {
    unsigned seed;
    if (scanf("%u", &seed) != 1) return 2;
    return order(seed);
  }
  if (!strcmp(kind, "rmsnorm")) {
    int rows, dim; if (scanf("%d %d", &rows, &dim) != 2) return 2;
    double *x = rd(rows * dim), *g = rd(dim), *o = malloc((size_t)rows * dim * sizeof *o);
    bcir_rmsnorm(x, rows, dim, g, 1e-6, o);
    for (int i = 0; i < rows * dim; i++) printf("%s%.17g", i ? " " : "", o[i]);
  } else if (!strcmp(kind, "rope")) {
    int rows, dim, off; double base;
    if (scanf("%d %d %lf %d", &rows, &dim, &base, &off) != 4) return 2;
    double *x = rd(rows * dim), *o = malloc((size_t)rows * dim * sizeof *o);
    bcir_rope(x, rows, dim, base, off, o);
    for (int i = 0; i < rows * dim; i++) printf("%s%.17g", i ? " " : "", o[i]);
  } else if (!strcmp(kind, "embedding")) {
    int vocab, dim, n; if (scanf("%d %d %d", &vocab, &dim, &n) != 3) return 2;
    double *t = rd(vocab * dim);
    int *ids = malloc((size_t)n * sizeof *ids);
    for (int i = 0; i < n; i++) if (scanf("%d", &ids[i]) != 1) return 2;
    double *o = malloc((size_t)n * dim * sizeof *o);
    if (bcir_embedding(t, vocab, dim, ids, n, o)) return 3;   /* out-of-range id refused */
    for (int i = 0; i < n * dim; i++) printf("%s%.17g", i ? " " : "", o[i]);
  } else if (!strcmp(kind, "gqa")) {
    int seq, nh, nkv, dk; if (scanf("%d %d %d %d", &seq, &nh, &nkv, &dk) != 4) return 2;
    double *q = rd(seq * nh * dk), *k = rd(seq * nkv * dk), *v = rd(seq * nkv * dk);
    double *s = malloc((size_t)seq * sizeof *s);
    double *o = malloc((size_t)seq * nh * dk * sizeof *o);
    if (bcir_gqa_attention(q, k, v, seq, nh, nkv, dk, s, o)) return 3;   /* ragged groups */
    for (int i = 0; i < seq * nh * dk; i++) printf("%s%.17g", i ? " " : "", o[i]);
  } else if (!strcmp(kind, "gqa_row")) {
    int t, nh, nkv, dk; if (scanf("%d %d %d %d", &t, &nh, &nkv, &dk) != 4) return 2;
    double *q = rd(nh * dk), *k = rd(t * nkv * dk), *v = rd(t * nkv * dk);
    double *s = malloc((size_t)t * sizeof *s);
    double *o = malloc((size_t)nh * dk * sizeof *o);
    if (bcir_gqa_attention_row(q, k, v, t, nh, nkv, dk, s, o)) return 3;
    for (int i = 0; i < nh * dk; i++) printf("%s%.17g", i ? " " : "", o[i]);
  } else return 2;
  printf("\n");
  return 0;
}
