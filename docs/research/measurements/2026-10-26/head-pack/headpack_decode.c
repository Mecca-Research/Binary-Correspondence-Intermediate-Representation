/* HEAD-PACK's decode A/B in one binary: the tree after's C decoder, built with the instance probe, on one loaded model.
 * Three arms: UNPACKED (a copy of the model with no pack, so the head reads its rows as they are, as the tree before's
 * decoder did), PACKED (the model as loaded, its head packed) and UNPACKED again (the A/A control) -- the same code at
 * the same addresses on the same weights, differing only in the head's path. A round runs the arms in rotated order:
 * REPS greedy generations of SHORT and of LONG new tokens from one 5-token prompt, an arm's per-token time
 * (t_long - t_short) / (LONG - SHORT); each arm's ratio over UNPACKED's in the same round, the geometric mean and its
 * 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)), after a warm round. Every generation's tokens and final
 * logits are UNPACKED's bytes, and every generation's last head call ran its arm's path (bcir_ai_q8_pack_probe: the
 * packed arm's on the pack, its passes and blocks the model's rows', the unpacked arms' not) -- the run exits
 * otherwise. The model loads with a file-size cap of MAX_MIB.
 * usage: headpack_decode ROUNDS REPS MODEL.bcirq8 SHORT LONG MAX_MIB  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include "bcir_llama.h"
#include "bcir_q8_model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int bcir_ai_q8_pack_probe(size_t *passes, size_t *blocks);

enum { MAXR = 200, MAXNEW = 1000, ARMS = 3 };
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
  static int32_t want[MAXNEW], got[MAXNEW];
  bcir_q8_model model, unpacked;
  char err[256];
  const int32_t prompt[5] = {1, 100, 200, 300, 400};
  int rounds, reps, shortn, longn, cap, vector;
  if (argc != 7 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (reps = atoi(argv[2])) < 1 ||
      (shortn = atoi(argv[4])) < 1 || (longn = atoi(argv[5])) <= shortn || longn > MAXNEW || (cap = atoi(argv[6])) < 1) {
    fprintf(stderr, "usage: headpack_decode ROUNDS REPS MODEL.bcirq8 SHORT LONG MAX_MIB\n");
    return 2;
  }
  bcir_q8_model_init(&model);
  if (bcir_q8_model_load_limited(argv[3], &model, err, sizeof err, (uint64_t)cap << 20)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  if (model.head_pack == NULL) {
    fprintf(stderr, "headpack_decode: the loader packed no head (d_model %u, groups of %u)\n", model.d_model,
            (unsigned)model.group_size);
    return 1;
  }
  unpacked = model; /* the same storage and tensors, no pack: never freed */
  unpacked.head_pack = NULL;
  unpacked.head_pack_size = 0u;
  vector = (bcir_ai_q8_instances() & 6) != 0;
  size_t vocab = model.vocab_size, want_passes = vocab / 32u, want_blocks = (vocab % 32u + 7u) / 8u;
  double *want_logits = malloc(vocab * sizeof(double)), *logits = malloc(vocab * sizeof(double));
  if (!want_logits || !logits) return 1;
  if (bcir_llama_generate_greedy(&unpacked, prompt, 5, (size_t)longn, want, want_logits)) return 1;
  for (int r = -1; r < rounds; r++)
    for (int a = 0; a < ARMS; a++) {
      int arm = (a + (r < 0 ? 0 : r)) % ARMS;
      const bcir_q8_model *m = arm == 1 ? &model : &unpacked;
      double t[2];
      for (int len = 0; len < 2; len++) {
        size_t n = (size_t)(len ? longn : shortn);
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          size_t passes = 99u, blocks = 99u;
          int rc = bcir_llama_generate_greedy(m, prompt, 5, n, got, logits), path;
          if (rc || memcmp(got, want, n * sizeof(int32_t)) || (len && memcmp(logits, want_logits, vocab * sizeof(double)))) {
            fprintf(stderr, "arm %d: a generation's tokens or logits differ\n", arm);
            return 1;
          }
          path = bcir_ai_q8_pack_probe(&passes, &blocks);
          if (arm == 1 ? path != (vector ? 1 : 2) || passes != (vector ? want_passes : 0u) ||
                             blocks != (vector ? want_blocks : 0u)
                       : path != 0) {
            fprintf(stderr, "arm %d: the head's last call off its arm's path (%d, %zu, %zu)\n", arm, path, passes,
                    blocks);
            return 1;
          }
        }
        t[len] = (now() - t0) / reps;
      }
      if (r >= 0) per_token[arm][r] = (t[1] - t[0]) * 1e3 / (longn - shortn);
    }
  printf("{\"rounds\":%d,\"reps\":%d,\"short\":%d,\"long\":%d,\"vocab\":%zu,\"d_model\":%u,\"head_pack_bytes\":%zu,"
         "\"unpacked_per_token_ms\":%.5f",
         rounds, reps, shortn, longn, vocab, model.d_model, model.head_pack_size, median(per_token[0], rounds));
  static const char *names[ARMS] = {"unpacked", "packed", "A/A"};
  for (int a = 1; a < ARMS; a++) {
    double ratio[MAXR], ci[3];
    for (int r = 0; r < rounds; r++) ratio[r] = per_token[a][r] / per_token[0][r];
    interval(ratio, rounds, ci);
    printf(",\"%s\":{\"per_token_ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", names[a], median(per_token[a], rounds),
           ci[0], ci[1], ci[2]);
  }
  printf(",\"every_generation_the_same_bytes\":true,\"every_head_on_its_arms_path\":true}\n");
  bcir_q8_model_free(&model);
  free(want_logits);
  free(logits);
  return 0;
}
