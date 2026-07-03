/*===- bcir_decode.c - the rung-5 LLM decode stage kernels (see bcir_decode.h) ------------===*/
#include "bcir_decode.h"

#include <math.h>
#include <stddef.h>

void bcir_rmsnorm(const double *x, int rows, int dim, const double *gamma, double eps,
                  double *out) {
  for (int r = 0; r < rows; r++) {
    const double *row = x + (size_t)r * dim;
    double acc = 0.0;                                /* ascending-index sum, the oracle order */
    for (int c = 0; c < dim; c++) acc += row[c] * row[c];
    double rms = sqrt(acc / dim + eps);
    for (int c = 0; c < dim; c++) out[(size_t)r * dim + c] = row[c] / rms * gamma[c];
  }
}

void bcir_rope(const double *x, int rows, int dim, double base, int pos_offset, double *out) {
  for (int r = 0; r < rows; r++) {
    const double *row = x + (size_t)r * dim;
    double *o = out + (size_t)r * dim;
    for (int k = 0; k < dim / 2; k++) {
      double th = (double)(pos_offset + r) * pow(base, -(2.0 * k) / dim);
      double c = cos(th), s = sin(th);
      double x0 = row[2 * k], x1 = row[2 * k + 1];
      o[2 * k] = x0 * c - x1 * s;
      o[2 * k + 1] = x0 * s + x1 * c;
    }
  }
}

int bcir_embedding(const double *table, int vocab, int dim, const int *ids, int n_ids,
                   double *out) {
  for (int t = 0; t < n_ids; t++) {
    if (ids[t] < 0 || ids[t] >= vocab) return -1;    /* the oracle raises; the twin refuses */
    const double *src = table + (size_t)ids[t] * dim;
    for (int d = 0; d < dim; d++) out[(size_t)t * dim + d] = src[d];
  }
  return 0;
}
