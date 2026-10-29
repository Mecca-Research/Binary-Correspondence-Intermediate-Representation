/* Q8-BOUND's decode A/B in one binary: the tree after's C decoder with bound_switch.py's kernel unit, the once-a-call
 * choice switched at run time (bcir_ai_q8_bound_off), so every arm runs the same code at the same addresses on the
 * same weights and differs only in the path each matvec takes. Three arms: OFF (every call the terms as before
 * Q8-BOUND), ON (the unit as committed) and OFF again (the A/A control). A round runs the arms in rotated order: REPS
 * greedy generations of 8 and of 200 new tokens from one 5-token prompt, an arm's per-token time (t200 - t8) / 192;
 * each arm's ratio over OFF's in the same round, the geometric mean and its 95% interval (Student t to 24 rounds,
 * then 1.96 + 2.4/(n-1)), after a warm round. Every generation's tokens and final logits are OFF's bytes, and every
 * matvec of an arm took that arm's path (bcir_ai_q8_bound_calls: ON's every call hoisted, OFF's none) -- the run
 * exits otherwise; a 200-token generation's matvec calls are reported.
 * usage: bound_ab ROUNDS REPS MODEL.bcirq8  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_llama.h"
#include "bcir_q8_model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int bcir_ai_q8_bound_off;
extern unsigned long bcir_ai_q8_bound_calls[2];

enum { MAXR = 200, NEW = 200, ARMS = 3 };
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

int main(int argc, char **argv) {
  static double per_token[ARMS][MAXR];
  static int32_t want[NEW], got[NEW];
  bcir_q8_model model;
  char err[256];
  const int32_t prompt[5] = {1, 100, 200, 300, 400};
  int rounds, reps;
  unsigned long calls = 0u;
  if (argc != 4 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (reps = atoi(argv[2])) < 1) {
    fprintf(stderr, "usage: bound_ab ROUNDS REPS MODEL.bcirq8\n");
    return 2;
  }
  bcir_q8_model_init(&model);
  if (bcir_q8_model_load(argv[3], &model, err, sizeof err)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  size_t vocab = model.vocab_size;
  double *want_logits = malloc(vocab * sizeof(double)), *logits = malloc(vocab * sizeof(double));
  if (!want_logits || !logits) return 1;
  bcir_ai_q8_bound_off = 1;
  if (bcir_llama_generate_greedy(&model, prompt, 5, NEW, want, want_logits)) return 1;
  for (int r = -1; r < rounds; r++)
    for (int a = 0; a < ARMS; a++) {
      int arm = (a + (r < 0 ? 0 : r)) % ARMS;
      double t[2];
      bcir_ai_q8_bound_off = arm != 1;
      for (int len = 0; len < 2; len++) {
        size_t n = len ? NEW : 8u;
        unsigned long hoisted = bcir_ai_q8_bound_calls[0], again = bcir_ai_q8_bound_calls[1];
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          int rc = bcir_llama_generate_greedy(&model, prompt, 5, n, got, logits);
          if (rc || memcmp(got, want, n * sizeof(int32_t)) || (len && memcmp(logits, want_logits, vocab * sizeof(double)))) {
            fprintf(stderr, "arm %d: a generation's tokens or logits differ\n", arm);
            return 1;
          }
        }
        t[len] = (now() - t0) / reps;
        hoisted = bcir_ai_q8_bound_calls[0] - hoisted;
        again = bcir_ai_q8_bound_calls[1] - again;
        if ((arm == 1 ? again : hoisted) != 0u || (arm == 1 ? hoisted : again) == 0u) {
          fprintf(stderr, "arm %d: %lu matvecs hoisted, %lu not\n", arm, hoisted, again);
          return 1;
        }
        if (len) calls = (hoisted + again) / (unsigned long)reps;
      }
      if (r >= 0) per_token[arm][r] = (t[1] - t[0]) * 1e3 / (NEW - 8);
    }
  printf("{\"rounds\":%d,\"reps\":%d,\"matvecs_a_generation\":%lu,\"off_per_token_ms\":%.5f", rounds, reps, calls,
         median(per_token[0], rounds));
  static const char *names[ARMS] = {"off", "on", "A/A"};
  for (int a = 1; a < ARMS; a++) {
    double ratio[MAXR], ci[3];
    for (int r = 0; r < rounds; r++) ratio[r] = per_token[a][r] / per_token[0][r];
    interval(ratio, rounds, ci);
    printf(",\"%s\":{\"per_token_ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", names[a], median(per_token[a], rounds),
           ci[0], ci[1], ci[2]);
  }
  printf(",\"every_generation_the_same_bytes\":true,\"every_matvec_on_its_arms_path\":true}\n");
  bcir_q8_model_free(&model);
  free(want_logits);
  free(logits);
  return 0;
}
