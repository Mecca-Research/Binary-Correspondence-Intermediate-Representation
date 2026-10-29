/* Q8-HOIST's decode A/B, one process: the C decoder as the tree before ran it (BEFORE: its bcir_ai_kernels.c,
 * bcir_q8_model.c, bcir_decode.c and bcir_llama.c, every global renamed before_*) against the tree after's; BEFORE
 * again (the A/A control: the same code on the same weights); and a second copy of the tree before's units, every
 * global renamed before2_*, on a copy of the model of its own (the placement control: what the after arm differs
 * from BEFORE in besides the patch -- its code at addresses of its own, its weights a buffer of their own). A round
 * runs the four arms in rotated order: REPS greedy generations of 8 and of 200 new tokens from one 5-token prompt,
 * an arm's per-token time (t200 - t8) / 192; each arm's ratio over BEFORE's in the same round, and the after arm's
 * over the placement control's, the geometric mean and its 95% interval (Student t to 24 rounds, then
 * 1.96 + 2.4/(n-1)), after a warm round. Every generation's tokens and final logits are BEFORE's bytes.
 * usage: hoist_ab ROUNDS REPS MODEL.bcirq8  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_llama.h"
#include "bcir_q8_model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void before_bcir_q8_model_init(bcir_q8_model *model);
int before_bcir_q8_model_load(const char *path, bcir_q8_model *out, char *error, size_t error_capacity);
void before_bcir_q8_model_free(bcir_q8_model *model);
int before_bcir_llama_generate_greedy(const bcir_q8_model *model, const int32_t *prompt_ids, size_t prompt_count,
                                      size_t max_new_tokens, int32_t *generated_ids, double *final_logits);
void before2_bcir_q8_model_init(bcir_q8_model *model);
int before2_bcir_q8_model_load(const char *path, bcir_q8_model *out, char *error, size_t error_capacity);
void before2_bcir_q8_model_free(bcir_q8_model *model);
int before2_bcir_llama_generate_greedy(const bcir_q8_model *model, const int32_t *prompt_ids, size_t prompt_count,
                                       size_t max_new_tokens, int32_t *generated_ids, double *final_logits);

enum { MAXR = 200, NEW = 200, ARMS = 4 };
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
/* arm 0 BEFORE and 2 the A/A control on model 0, arm 1 the tree after's on model 1, arm 3 the placement control on
 * model 2 */
static int generate(int arm, const bcir_q8_model *model, const int32_t *prompt, size_t n, int32_t *got,
                    double *logits) {
  if (arm == 1) return bcir_llama_generate_greedy(&model[1], prompt, 5, n, got, logits);
  if (arm == 3) return before2_bcir_llama_generate_greedy(&model[2], prompt, 5, n, got, logits);
  return before_bcir_llama_generate_greedy(&model[0], prompt, 5, n, got, logits);
}
static void print_ratio(const char *name, const double *num, const double *den, int rounds) {
  double ratio[MAXR], ci[3];
  for (int r = 0; r < rounds; r++) ratio[r] = num[r] / den[r];
  interval(ratio, rounds, ci);
  printf(",\"%s\":{\"per_token_ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", name, median(num, rounds), ci[0],
         ci[1], ci[2]);
}

int main(int argc, char **argv) {
  static double per_token[ARMS][MAXR];
  static int32_t want[NEW], got[NEW];
  bcir_q8_model model[3];
  char err[256];
  const int32_t prompt[5] = {1, 100, 200, 300, 400};
  int rounds, reps;
  if (argc != 4 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (reps = atoi(argv[2])) < 1) {
    fprintf(stderr, "usage: hoist_ab ROUNDS REPS MODEL.bcirq8\n");
    return 2;
  }
  before_bcir_q8_model_init(&model[0]);
  bcir_q8_model_init(&model[1]);
  before2_bcir_q8_model_init(&model[2]);
  if (before_bcir_q8_model_load(argv[3], &model[0], err, sizeof err) ||
      bcir_q8_model_load(argv[3], &model[1], err, sizeof err) ||
      before2_bcir_q8_model_load(argv[3], &model[2], err, sizeof err)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  size_t vocab = model[1].vocab_size;
  double *want_logits = malloc(vocab * sizeof(double)), *logits = malloc(vocab * sizeof(double));
  if (!want_logits || !logits) return 1;
  if (before_bcir_llama_generate_greedy(&model[0], prompt, 5, NEW, want, want_logits)) return 1;
  for (int r = -1; r < rounds; r++)
    for (int a = 0; a < ARMS; a++) {
      int arm = (a + (r < 0 ? 0 : r)) % ARMS;
      double t[2];
      for (int len = 0; len < 2; len++) {
        size_t n = len ? NEW : 8u;
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          int rc = generate(arm, model, prompt, n, got, logits);
          if (rc || memcmp(got, want, n * sizeof(int32_t)) || (len && memcmp(logits, want_logits, vocab * sizeof(double)))) {
            fprintf(stderr, "arm %d: a generation's tokens or logits differ\n", arm);
            return 1;
          }
        }
        t[len] = (now() - t0) / reps;
      }
      if (r >= 0) per_token[arm][r] = (t[1] - t[0]) * 1e3 / (NEW - 8);
    }
  printf("{\"rounds\":%d,\"reps\":%d,\"before_per_token_ms\":%.5f", rounds, reps, median(per_token[0], rounds));
  print_ratio("after", per_token[1], per_token[0], rounds);
  print_ratio("A/A", per_token[2], per_token[0], rounds);
  print_ratio("placement", per_token[3], per_token[0], rounds);
  print_ratio("after_over_placement", per_token[1], per_token[3], rounds);
  printf(",\"every_generation_the_same_bytes\":true}\n");
  before_bcir_q8_model_free(&model[0]);
  bcir_q8_model_free(&model[1]);
  before2_bcir_q8_model_free(&model[2]);
  free(want_logits);
  free(logits);
  return 0;
}
