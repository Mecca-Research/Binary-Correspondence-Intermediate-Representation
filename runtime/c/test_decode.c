/*===- test_decode.c - rung-5 decode-kernel differential harness --------------------------===
 * stdin: "rmsnorm rows dim" + rows*dim x + dim gamma  |  "rope rows dim base off" + rows*dim x
 *      | "embedding vocab dim n" + vocab*dim table + n ids
 * stdout: the outputs, %.17g, space-separated (one line). cc test_decode.c bcir_decode.c -lm  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_decode.h"

static double *rd(int n) {
  double *v = malloc((size_t)n * sizeof *v);
  for (int i = 0; i < n; i++) if (scanf("%lf", &v[i]) != 1) exit(2);
  return v;
}

int main(void) {
  char kind[16];
  if (scanf("%15s", kind) != 1) return 2;
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
  } else return 2;
  printf("\n");
  return 0;
}
