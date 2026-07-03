/*===- test_train.c - D1 step 4 harness: REAL training in C over the binary pack ----------===
 *
 * Reads the binary train-step StreamPack (argv[1], oracle-encoded) and a dataset file
 * (argv[2]), then runs the SAME training loop as the oracle's train_planned: for every
 * epoch, for every full batch (deterministic order, no shuffle), load the batch into the
 * state and dispatch the step through the deterministic executor with the numeric stage
 * kernels (bcir_train.c) behind the per-claim callback. No Python anywhere in the loop.
 *
 * Dataset file (text): "nf b n epochs lr" then n rows of nf doubles (X), n doubles (y),
 * nf+1 doubles (w0). Prints the first step's dispatch order, one "epoch <mean-loss>" line
 * per epoch, and the final weights -- %.17g so the Python gate compares round-trip exact.
 *
 *   cc test_train.c bcir_train.c bcir_exec.c bcir_runtime.c -I . -lm -o t
 *   ./t step.pack data.txt
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_train.h"

static uint64_t g_order[64];
static size_t g_norder;

/* First-step wrapper: record the dispatch order once, then compute (order proves the
 * executor -- not the harness -- sequences the six stages). */
static int record_and_compute(const bcir_exec_item *item, void *ctx) {
  if (g_norder < sizeof g_order / sizeof g_order[0]) g_order[g_norder++] = item->claim_id;
  return bcir_train_kernel(item, ctx);
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <pack-file> <data-file>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("pack"); return 2; }
  static uint8_t pack[1 << 20];
  size_t plen = fread(pack, 1, sizeof pack, fp);
  fclose(fp);

  FILE *df = fopen(argv[2], "r");
  if (!df) { perror("data"); return 2; }
  int nf, b, n, epochs;
  double lr;
  if (fscanf(df, "%d %d %d %d %lf", &nf, &b, &n, &epochs, &lr) != 5 ||
      nf < 1 || b < 1 || n < b || epochs < 1) {
    fprintf(stderr, "bad data header\n"); fclose(df); return 2;
  }
  double *X = malloc((size_t)n * nf * sizeof(double));
  double *y = malloc((size_t)n * sizeof(double));
  double *w = malloc((size_t)(nf + 1) * sizeof(double));
  double *bx = malloc((size_t)b * nf * sizeof(double));
  double *by = malloc((size_t)b * sizeof(double));
  double *z = malloc((size_t)b * sizeof(double));
  double *act = malloc((size_t)b * sizeof(double));
  double *lossv = malloc((size_t)b * sizeof(double));
  double *grad = malloc((size_t)(nf + 1) * sizeof(double));
  if (!X || !y || !w || !bx || !by || !z || !act || !lossv || !grad) { fclose(df); return 2; }
  for (int i = 0; i < n * nf; i++) if (fscanf(df, "%lf", &X[i]) != 1) { fclose(df); return 2; }
  for (int i = 0; i < n; i++) if (fscanf(df, "%lf", &y[i]) != 1) { fclose(df); return 2; }
  for (int j = 0; j <= nf; j++) if (fscanf(df, "%lf", &w[j]) != 1) { fclose(df); return 2; }
  fclose(df);

  bcir_train_state st = {nf, b, lr, bx, by, w, z, act, lossv, 0.0, grad};
  static bcir_exec_item scratch[64];
  static bcir_phase_stat phases[16];
  bcir_exec_result res;

  for (int e = 0; e < epochs; e++) {
    double esum = 0.0;
    int nsteps = 0;
    for (int lo = 0; lo + b <= n; lo += b) {          /* full batches only, oracle order */
      for (int i = 0; i < b; i++) {
        for (int j = 0; j < nf; j++) bx[i * nf + j] = X[(lo + i) * nf + j];
        by[i] = y[lo + i];
      }
      bcir_exec_fn fn = (e == 0 && lo == 0) ? record_and_compute : bcir_train_kernel;
      bcir_status s = bcir_sp_execute(pack, plen, scratch, 64, phases, 16, fn, &st, &res);
      if (s != BCIR_OK) { printf("ERR %d\n", (int)s); return 1; }
      esum += st.loss;
      nsteps++;
    }
    /* the oracle epoch loss: sum(step losses) / n_steps, ascending accumulation */
    printf("epoch %.17g\n", esum / nsteps);
  }
  printf("order:");
  for (size_t i = 0; i < g_norder; i++) printf(" %llu", (unsigned long long)g_order[i]);
  printf("\nweights:");
  for (int j = 0; j <= nf; j++) printf(" %.17g", w[j]);
  printf("\n");
  free(X); free(y); free(w); free(bx); free(by); free(z); free(act); free(lossv); free(grad);
  return 0;
}
