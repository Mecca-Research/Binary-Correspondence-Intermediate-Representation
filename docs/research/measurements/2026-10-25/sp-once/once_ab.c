/* SP-ONCE's A/B, one process: GEM as the tree before ran it (BEFORE: its bcir_runtime.c, bcir_exec.c,
 * bcir_decoder_gem.c and the units they link, every global renamed before_*) against the tree after's, and BEFORE
 * again (the A/A control), on the qualification's native pack and its lab model. Cases: bcir_sp_execute with a kernel
 * that does nothing (the trust boundary a GEM run crosses before its first kernel); bcir_dgem_load then bcir_dgem_free
 * (GEM's load of the program); bcir_dgem_run (a whole request: the card's prompt and 48 new tokens). Per case a warm
 * round, then ROUNDS rounds in rotated arm order, an arm's time the mean of REPS calls; each arm's ratio over BEFORE's
 * in the same round, the geometric mean and its 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)). Every
 * run's tokens and final logits are BEFORE's bytes, and every execute's result BEFORE's (the run exits otherwise).
 * usage: once_ab ROUNDS MODEL.bcirq8 PACK PROMPT_IDS  -> one JSON line a case */
#define _POSIX_C_SOURCE 200809L
#include "bcir_decoder_gem.h"
#include "bcir_exec.h"
#include "bcir_q8_model.h"
#include "bcir_runtime.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void before_bcir_q8_model_init(bcir_q8_model *model);
int before_bcir_q8_model_load(const char *path, bcir_q8_model *out, char *error, size_t error_capacity);
void before_bcir_q8_model_free(bcir_q8_model *model);
bcir_status before_bcir_sp_execute(const uint8_t *data, size_t len, bcir_exec_item *scratch, size_t scratch_cap,
                                   bcir_phase_stat *phases, size_t phases_cap, bcir_exec_fn fn, void *ctx,
                                   bcir_exec_result *out);
int before_bcir_dgem_load(bcir_dgem *g, const bcir_q8_model *model, const uint8_t *pack, size_t len,
                          const bcir_host_allocator *allocator);
int before_bcir_dgem_run(bcir_dgem *g, const int32_t *prompt, size_t prompt_len, int32_t *generated,
                         size_t max_new, double *final_logits);
void before_bcir_dgem_free(bcir_dgem *g);

enum { MAXR = 200 };
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void interval(const double *r, int n, double out[3]) {
  static const double t95[] = {0, 0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
                               2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069};
  double m = 0, sd = 0, h;
  int i;
  for (i = 0; i < n; i++) m += log(r[i]);
  m /= n;
  for (i = 0; i < n; i++) sd += (log(r[i]) - m) * (log(r[i]) - m);
  sd = sqrt(sd / (n - 1));
  h = (n <= 24 ? t95[n] : 1.96 + 2.4 / (n - 1)) * sd / sqrt((double)n);
  out[0] = exp(m);
  out[1] = exp(m - h);
  out[2] = exp(m + h);
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double median(const double *x, int n) {
  double s[MAXR];
  memcpy(s, x, (size_t)n * sizeof(double));
  qsort(s, (size_t)n, sizeof(double), cmp);
  return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}
static void report(const char *what, int rounds, int reps, double ms[][MAXR]) {
  static const char *names[3] = {"before", "after", "A/A"};
  printf("{\"case\":\"%s\",\"rounds\":%d,\"reps\":%d,\"before_ms\":%.5f", what, rounds, reps, median(ms[0], rounds));
  for (int a = 1; a < 3; a++) {
    double ratio[MAXR], ci[3];
    for (int r = 0; r < rounds; r++) ratio[r] = ms[a][r] / ms[0][r];
    interval(ratio, rounds, ci);
    printf(",\"%s\":{\"ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", names[a], median(ms[a], rounds), ci[0],
           ci[1], ci[2]);
  }
  printf("}\n");
  fflush(stdout);
}
static uint8_t *read_all(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  long n;
  uint8_t *p;
  if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) <= 0 || fseek(f, 0, SEEK_SET)) exit(2);
  p = malloc((size_t)n);
  if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
  fclose(f);
  *len = (size_t)n;
  return p;
}

int main(int argc, char **argv) {
  static double ms[3][MAXR];
  bcir_q8_model model[2];
  char err[256];
  size_t pack_len, n_prompt = 0;
  int32_t prompt[256];
  int rounds;
  if (argc != 5 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR) {
    fprintf(stderr, "usage: once_ab ROUNDS MODEL.bcirq8 PACK PROMPT_IDS\n");
    return 2;
  }
  for (char *t = strtok(argv[4], ","); t && n_prompt < 256; t = strtok(NULL, ",")) prompt[n_prompt++] = atoi(t);
  uint8_t *pack = read_all(argv[3], &pack_len);
  before_bcir_q8_model_init(&model[0]);
  bcir_q8_model_init(&model[1]);
  if (before_bcir_q8_model_load(argv[2], &model[0], err, sizeof err) ||
      bcir_q8_model_load(argv[2], &model[1], err, sizeof err)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  bcir_streampack_header hdr;
  if (bcir_sp_validate(pack, pack_len, &hdr) != BCIR_OK) return 1;
  size_t n = hdr.n_segments;
  bcir_exec_item *scratch = malloc(n * sizeof *scratch);
  bcir_phase_stat *phases = malloc(n * sizeof *phases);
  if (!scratch || !phases) return 1;

  /* the trust boundary a run crosses: bcir_sp_execute with no kernel */
  {
    bcir_exec_result want, got;
    if (before_bcir_sp_execute(pack, pack_len, scratch, n, phases, n, NULL, NULL, &want) != BCIR_OK) return 1;
    const int reps = 20;
    for (int r = -1; r < rounds; r++)
      for (int a = 0; a < 3; a++) {
        int arm = (a + (r < 0 ? 0 : r)) % 3;
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          bcir_status st = arm == 1 ? bcir_sp_execute(pack, pack_len, scratch, n, phases, n, NULL, NULL, &got)
                                    : before_bcir_sp_execute(pack, pack_len, scratch, n, phases, n, NULL, NULL, &got);
          if (st != BCIR_OK || got.executed != want.executed || got.n_phases != want.n_phases ||
              got.n_segments != want.n_segments) {
            fprintf(stderr, "execute differs\n");
            return 1;
          }
        }
        if (r >= 0) ms[arm][r] = (now() - t0) * 1e3 / reps;
      }
    report("bcir_sp_execute, no kernel", rounds, reps, ms);
  }
  /* GEM's load of the program */
  {
    const int reps = 10;
    for (int r = -1; r < rounds; r++)
      for (int a = 0; a < 3; a++) {
        int arm = (a + (r < 0 ? 0 : r)) % 3;
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          bcir_dgem g;
          int rc = arm == 1 ? bcir_dgem_load(&g, &model[1], pack, pack_len, NULL)
                            : before_bcir_dgem_load(&g, &model[0], pack, pack_len, NULL);
          if (rc) {
            fprintf(stderr, "load refused: %d\n", rc);
            return 1;
          }
          if (arm == 1) bcir_dgem_free(&g);
          else before_bcir_dgem_free(&g);
        }
        if (r >= 0) ms[arm][r] = (now() - t0) * 1e3 / reps;
      }
    report("bcir_dgem_load + bcir_dgem_free", rounds, reps, ms);
  }
  /* a whole request through GEM */
  {
    bcir_dgem g[3];
    int32_t gen[3][512], want_gen[512];
    double *logits[3], *want_logits;
    size_t max_new, vocab = model[1].vocab_size;
    for (int a = 0; a < 3; a++) {
      int rc = a == 1 ? bcir_dgem_load(&g[a], &model[1], pack, pack_len, NULL)
                      : before_bcir_dgem_load(&g[a], &model[0], pack, pack_len, NULL);
      logits[a] = malloc(vocab * sizeof(double));
      if (rc || !logits[a]) return 1;
    }
    want_logits = malloc(vocab * sizeof(double));
    max_new = bcir_dgem_capacity(&g[1]) - bcir_dgem_prompt_len(&g[1]);
    if (!want_logits || max_new > 512 || bcir_dgem_prompt_len(&g[1]) != n_prompt) return 1;
    if (before_bcir_dgem_run(&g[0], prompt, n_prompt, want_gen, max_new, want_logits)) return 1;
    const int reps = 3;
    for (int r = -1; r < rounds; r++)
      for (int a = 0; a < 3; a++) {
        int arm = (a + (r < 0 ? 0 : r)) % 3;
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          int rc = arm == 1 ? bcir_dgem_run(&g[arm], prompt, n_prompt, gen[arm], max_new, logits[arm])
                            : before_bcir_dgem_run(&g[arm], prompt, n_prompt, gen[arm], max_new, logits[arm]);
          if (rc || memcmp(gen[arm], want_gen, max_new * sizeof(int32_t)) ||
              memcmp(logits[arm], want_logits, vocab * sizeof(double))) {
            fprintf(stderr, "a run's tokens or logits differ\n");
            return 1;
          }
        }
        if (r >= 0) ms[arm][r] = (now() - t0) * 1e3 / reps;
      }
    report("bcir_dgem_run, the whole request", rounds, reps, ms);
    for (int a = 0; a < 3; a++) {
      if (a == 1) bcir_dgem_free(&g[a]);
      else before_bcir_dgem_free(&g[a]);
      free(logits[a]);
    }
    free(want_logits);
  }
  before_bcir_q8_model_free(&model[0]);
  bcir_q8_model_free(&model[1]);
  free(pack);
  free(scratch);
  free(phases);
  return 0;
}
