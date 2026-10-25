/*===- bcir_decode.c - the rung-5 LLM decode stage kernels (see bcir_decode.h) ------------===*/
#include "bcir_decode.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int doubles_fit2(size_t a, size_t b) {
  return !b || (a <= SIZE_MAX / b && a * b <= (size_t)PTRDIFF_MAX / sizeof(double));
}

static int doubles_fit3(size_t a, size_t b, size_t c) {
  size_t ab;
  if (b && a > SIZE_MAX / b) return 0;
  ab = a * b;
  return doubles_fit2(ab, c);
}

void bcir_rmsnorm(const double *x, int rows, int dim, const double *gamma, double eps,
                  double *out) {
  if (!x || !gamma || !out || rows <= 0 || dim <= 0 || !isfinite(eps) || eps < 0.0) return;
  if (!doubles_fit2((size_t)rows, (size_t)dim)) return;
  for (int r = 0; r < rows; r++) {
    const double *row = x + (size_t)r * dim;
    double acc = 0.0;                                /* ascending-index sum, the oracle order */
    for (int c = 0; c < dim; c++) acc += row[c] * row[c];
    double rms = sqrt(acc / dim + eps);
    for (int c = 0; c < dim; c++) out[(size_t)r * dim + c] = row[c] / rms * gamma[c];
  }
}

/* the rotation's angle for pair k at `position`, and the pair rotated by its cosine and sine: one definition
 * each, so bcir_rope and a table filled once a position (bcir_rope_table / bcir_rope_apply) agree bit for bit */
static double rope_angle(double position, int k, int dim, double base) {
  return position * pow(base, -(2.0 * k) / dim);
}
static void rope_pair(const double *x, double c, double s, double *o) {
  double x0 = x[0], x1 = x[1];
  o[0] = x0 * c - x1 * s;
  o[1] = x0 * s + x1 * c;
}

void bcir_rope(const double *x, int rows, int dim, double base, int pos_offset, double *out) {
  if (!x || !out || rows <= 0 || dim <= 0 || (dim & 1) || !isfinite(base) || base <= 0.0) return;
  if (!doubles_fit2((size_t)rows, (size_t)dim)) return;
  for (int r = 0; r < rows; r++) {
    const double *row = x + (size_t)r * dim;
    double *o = out + (size_t)r * dim;
    for (int k = 0; k < dim / 2; k++) {
      double th = rope_angle((double)pos_offset + (double)r, k, dim, base);
      rope_pair(row + 2 * k, cos(th), sin(th), o + 2 * k);
    }
  }
}

void bcir_rope_table(int dim, double base, double position, double *cs) {
  if (!cs || dim <= 0 || (dim & 1) || !isfinite(base) || base <= 0.0) return;
  for (int k = 0; k < dim / 2; k++) {
    double th = rope_angle(position, k, dim, base);
    cs[2 * k] = cos(th);
    cs[2 * k + 1] = sin(th);
  }
}

void bcir_rope_apply(const double *x, int dim, const double *cs, double *out) {
  if (!x || !cs || !out || dim <= 0 || (dim & 1)) return;
  for (int k = 0; k < dim / 2; k++) rope_pair(x + 2 * k, cs[2 * k], cs[2 * k + 1], out + 2 * k);
}

int bcir_embedding(const double *table, int vocab, int dim, const int *ids, int n_ids,
                   double *out) {
  if (vocab < 1 || dim < 1 || n_ids < 0 || (n_ids && (!table || !ids || !out))) return -1;
  if (!doubles_fit2((size_t)vocab, (size_t)dim) ||
      !doubles_fit2((size_t)n_ids, (size_t)dim)) return -1;
  /* Validate every gather index before writing row zero: a bad late id must not
   * leave a partially materialized embedding tensor. */
  for (int t = 0; t < n_ids; t++) if (ids[t] < 0 || ids[t] >= vocab) return -1;
  for (int t = 0; t < n_ids; t++) {
    const double *src = table + (size_t)ids[t] * dim;
    for (int d = 0; d < dim; d++) out[(size_t)t * dim + d] = src[d];
  }
  return 0;
}

/* The scores of key rows 0 .. n-1 (kg: kv head g's row 0, `stride` doubles a row): each row's dot product
 * from +0.0 in ascending channel order, then times sc -- one row's chain, term for term, whichever block
 * computes it. Eight rows and then four run their chains side by side (DECODE-ROW), the rest one at a
 * time: the latency of one chain no longer paces the row. */
static void gqa_scores(const double *qh, const double *kg, size_t stride, int n, int d_k, double sc,
                       double *s) {
  int j = 0;
  for (; n - j >= 8; j += 8) {
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
  for (; n - j >= 4; j += 4) {
    const double *k0 = kg + (size_t)j * stride;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (int c = 0; c < d_k; c++) {
      double qc = qh[c];
      a0 += qc * k0[c];
      a1 += qc * k0[stride + c];
      a2 += qc * k0[2 * stride + c];
      a3 += qc * k0[3 * stride + c];
    }
    s[j] = a0 * sc; s[j + 1] = a1 * sc; s[j + 2] = a2 * sc; s[j + 3] = a3 * sc;
  }
  for (; j < n; j++) {
    const double *kj = kg + (size_t)j * stride;
    double acc = 0.0;                                /* ascending-index, the oracle order */
    for (int c = 0; c < d_k; c++) acc += qh[c] * kj[c];
    s[j] = acc * sc;
  }
}

/* A @ V_g: each coordinate's sum from +0.0 over every key row in ascending order (matmul's order),
 * term for term. Eight coordinates run their chains side by side down the rows (DECODE-ROW), each row's
 * eight read together, the rest one at a time. */
static void gqa_context(const double *s, const double *vg, size_t stride, int t_len, int d_k,
                        double *o) {
  int c = 0;
  for (; d_k - c >= 8; c += 8) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;
    for (int j = 0; j < t_len; j++) {
      const double *vj = vg + (size_t)j * stride + c;  /* row j's eight: no pointer past the rows */
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

/* the shared per-query-row GQA arithmetic: scores (masked to the first `keep` positions) ->
 * stable softmax -> A @ V_g -- ONE code path for the full and the cached entry points, so
 * the naive-vs-cached kernel invariant is bitwise by construction. */
static void gqa_row(const double *qh, const double *k_rows, const double *v_rows, int t_len,
                    int keep, int n_kv_heads, int g, int d_k, double sc, double *s,
                    double *o) {
  size_t stride = (size_t)n_kv_heads * (size_t)d_k;  /* one key (value) row to the next */
  int scored = keep < t_len ? keep : t_len;
  if (scored < 0) scored = 0;
  gqa_scores(qh, k_rows + (size_t)g * d_k, stride, scored, d_k, sc, s);
  for (int j = scored; j < t_len; j++) s[j] = -INFINITY;  /* the causal mask: exp(-inf) == 0 */
  double m = s[0];                                   /* stable softmax over the row */
  for (int j = 1; j < t_len; j++) if (s[j] > m) m = s[j];
  double sum = 0.0;
  for (int j = 0; j < t_len; j++) { s[j] = exp(s[j] - m); sum += s[j]; }
  if (sum == 0.0) sum = 1.0;                         /* mirrors the oracle's `or 1.0` */
  for (int j = 0; j < t_len; j++) s[j] /= sum;
  gqa_context(s, v_rows + (size_t)g * d_k, stride, t_len, d_k, o);  /* every row: masked ones add 0 */
}

int bcir_gqa_attention(const double *q, const double *k, const double *v, int seq,
                       int n_heads, int n_kv_heads, int d_k, double *scores, double *out) {
  if (seq < 0 || n_heads < 1 || d_k < 1 || n_kv_heads < 1 ||
      n_kv_heads > n_heads || n_heads % n_kv_heads) return -1;
  if (!seq) return 0;
  if (!q || !k || !v || !scores || !out) return -1;
  if (!doubles_fit3((size_t)seq, (size_t)n_heads, (size_t)d_k) ||
      !doubles_fit3((size_t)seq, (size_t)n_kv_heads, (size_t)d_k)) return -1;
  int group = n_heads / n_kv_heads;
  double sc = 1.0 / sqrt((double)d_k);
  for (int h = 0; h < n_heads; h++) {
    int g = h / group;                               /* the shared kv head */
    for (int i = 0; i < seq; i++)
      gqa_row(q + ((size_t)i * n_heads + h) * d_k, k, v, seq, i + 1, n_kv_heads, g, d_k, sc,
              scores, out + ((size_t)i * n_heads + h) * d_k);
  }
  return 0;
}

int bcir_gqa_attention_row(const double *q_row, const double *k_rows, const double *v_rows,
                           int t_len, int n_heads, int n_kv_heads, int d_k,
                           double *scores, double *out) {
  if (t_len < 1 || n_heads < 1 || d_k < 1 || n_kv_heads < 1 ||
      n_kv_heads > n_heads || n_heads % n_kv_heads ||
      !q_row || !k_rows || !v_rows || !scores || !out) return -1;
  if (!doubles_fit2((size_t)n_heads, (size_t)d_k) ||
      !doubles_fit3((size_t)t_len, (size_t)n_kv_heads, (size_t)d_k)) return -1;
  int group = n_heads / n_kv_heads;
  double sc = 1.0 / sqrt((double)d_k);
  for (int h = 0; h < n_heads; h++)                  /* the i == t_len-1 row, incrementally */
    gqa_row(q_row + (size_t)h * d_k, k_rows, v_rows, t_len, t_len, n_kv_heads, h / group,
            d_k, sc, scores, out + (size_t)h * d_k);
  return 0;
}
