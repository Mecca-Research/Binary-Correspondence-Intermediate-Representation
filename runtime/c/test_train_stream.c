/*===- test_train_stream.c - D1 step 7 harness: the STREAMED step trains in C -------------===
 *
 * Reads the binary STREAMED-step StreamPack (argv[1], oracle-encoded from
 * train_stream_module) and a dataset file (argv[2], the test_train.c format plus a
 * `streams` field), then runs the SAME loop as the oracle's train_streamed: per full
 * batch, stream s takes rows [s*mb, (s+1)*mb) of the batch (the oracle's split), and the
 * deterministic executor dispatches the per-stream stage kernels + the gradient combine +
 * the SINGLE weight update (bcir_stream_kernel) over the pack. No Python in the loop.
 *
 * Dataset file (text): "nf b n epochs streams lr" then n rows of nf doubles (X), n doubles
 * (y), nf+1 doubles (w0). Prints the first step's dispatch order, one "epoch <mean-loss>"
 * line per epoch (the mean over steps of the mean of stream means -- the oracle's number),
 * and the final weights, %.17g.
 *
 *   cc test_train_stream.c bcir_train.c bcir_exec.c bcir_runtime.c -I . -lm -o ts
 *   ./ts stream.pack data.txt
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_train.h"

static uint64_t g_order[128];
static size_t g_norder;

static int record_and_compute(const bcir_exec_item *item, void *ctx) {
  if (g_norder < sizeof g_order / sizeof g_order[0]) g_order[g_norder++] = item->claim_id;
  return bcir_stream_kernel(item, ctx);
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
  int nf, b, n, epochs, streams;
  double lr;
  if (fscanf(df, "%d %d %d %d %d %lf", &nf, &b, &n, &epochs, &streams, &lr) != 6 ||
      nf < 1 || b < 1 || n < b || epochs < 1 || streams < 1 || b % streams != 0) {
    fprintf(stderr, "bad data header\n"); fclose(df); return 2;
  }
  int mb = b / streams;
  double *X = malloc((size_t)n * nf * sizeof(double));
  double *y = malloc((size_t)n * sizeof(double));
  double *w = malloc((size_t)(nf + 1) * sizeof(double));
  double *sx = malloc((size_t)streams * mb * nf * sizeof(double));
  double *sy = malloc((size_t)streams * mb * sizeof(double));
  double *z = malloc((size_t)streams * mb * sizeof(double));
  double *act = malloc((size_t)streams * mb * sizeof(double));
  double *lossv = malloc((size_t)streams * mb * sizeof(double));
  double *loss = malloc((size_t)streams * sizeof(double));
  double *grad = malloc((size_t)streams * (nf + 1) * sizeof(double));
  double *gradc = malloc((size_t)(nf + 1) * sizeof(double));
  if (!X || !y || !w || !sx || !sy || !z || !act || !lossv || !loss || !grad || !gradc) {
    fclose(df); return 2;
  }
  for (int i = 0; i < n * nf; i++) if (fscanf(df, "%lf", &X[i]) != 1) { fclose(df); return 2; }
  for (int i = 0; i < n; i++) if (fscanf(df, "%lf", &y[i]) != 1) { fclose(df); return 2; }
  for (int j = 0; j <= nf; j++) if (fscanf(df, "%lf", &w[j]) != 1) { fclose(df); return 2; }
  fclose(df);

  bcir_stream_state st = {nf, mb, streams, lr, sx, sy, w, z, act, lossv, loss, grad, gradc};
  static bcir_exec_item scratch[128];
  static bcir_phase_stat phases[64];
  bcir_exec_result res;

  for (int e = 0; e < epochs; e++) {
    double esum = 0.0;
    int nsteps = 0;
    for (int lo = 0; lo + b <= n; lo += b) {          /* full batches, the oracle order */
      for (int s = 0; s < streams; s++)               /* stream s: rows [s*mb, (s+1)*mb) */
        for (int i = 0; i < mb; i++) {
          for (int j = 0; j < nf; j++)
            sx[((size_t)s * mb + i) * nf + j] = X[(size_t)(lo + s * mb + i) * nf + j];
          sy[(size_t)s * mb + i] = y[lo + s * mb + i];
        }
      bcir_exec_fn fn = (e == 0 && lo == 0) ? record_and_compute : bcir_stream_kernel;
      bcir_status stt = bcir_sp_execute(pack, plen, scratch, 128, phases, 64, fn, &st, &res);
      if (stt != BCIR_OK) { printf("ERR %d\n", (int)stt); return 1; }
      double lsum = 0.0;                              /* the mean of the stream means */
      for (int s = 0; s < streams; s++) lsum += loss[s];
      esum += lsum / streams;
      nsteps++;
    }
    printf("epoch %.17g\n", esum / nsteps);
  }
  printf("order:");
  for (size_t i = 0; i < g_norder; i++) printf(" %llu", (unsigned long long)g_order[i]);
  printf("\nweights:");
  for (int j = 0; j <= nf; j++) printf(" %.17g", w[j]);
  printf("\n");
  free(X); free(y); free(w); free(sx); free(sy); free(z); free(act); free(lossv);
  free(loss); free(grad); free(gradc);
  return 0;
}
