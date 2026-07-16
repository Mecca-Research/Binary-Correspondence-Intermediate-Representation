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

void bcir_rope(const double *x, int rows, int dim, double base, int pos_offset, double *out) {
  if (!x || !out || rows <= 0 || dim <= 0 || (dim & 1) || !isfinite(base) || base <= 0.0) return;
  if (!doubles_fit2((size_t)rows, (size_t)dim)) return;
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

/* the shared per-query-row GQA arithmetic: scores (masked to the first `keep` positions) ->
 * stable softmax -> A @ V_g -- ONE code path for the full and the cached entry points, so
 * the naive-vs-cached kernel invariant is bitwise by construction. */
static void gqa_row(const double *qh, const double *k_rows, const double *v_rows, int t_len,
                    int keep, int n_kv_heads, int g, int d_k, double sc, double *s,
                    double *o) {
  for (int j = 0; j < t_len; j++) {
    if (j >= keep) { s[j] = -INFINITY; continue; }   /* the causal mask: exp(-inf) == 0 */
    const double *kj = k_rows + ((size_t)j * n_kv_heads + g) * d_k;
    double acc = 0.0;                                /* ascending-index, the oracle order */
    for (int c = 0; c < d_k; c++) acc += qh[c] * kj[c];
    s[j] = acc * sc;
  }
  double m = s[0];                                   /* stable softmax over the row */
  for (int j = 1; j < t_len; j++) if (s[j] > m) m = s[j];
  double sum = 0.0;
  for (int j = 0; j < t_len; j++) { s[j] = exp(s[j] - m); sum += s[j]; }
  if (sum == 0.0) sum = 1.0;                         /* mirrors the oracle's `or 1.0` */
  for (int j = 0; j < t_len; j++) s[j] /= sum;
  for (int c = 0; c < d_k; c++) {                    /* A @ V_g, ascending j like matmul */
    double acc = 0.0;
    for (int j = 0; j < t_len; j++) acc += s[j] * v_rows[((size_t)j * n_kv_heads + g) * d_k + c];
    o[c] = acc;
  }
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
