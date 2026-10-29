/* KERNEL-ALIGN's decode placement A/B: copies of builds of the C decoder's units (bcir_ai_kernels.c,
 * bcir_q8_model.c, bcir_llama.c, bcir_decode.c, partially linked, every global renamed LABEL_) in one binary, each
 * copy's text where build_decode.py placed it; one model loaded once (the first copy's loader) and decoded by every
 * copy, lent no runner. Arms: arms.h's copies and the first again (the A/A control). A round runs the arms in rotated
 * order: REPS greedy generations of SHORT and of LONG new tokens from one 5-token prompt, an arm's per-token time
 * (t_long - t_short) / (LONG - SHORT); each arm's ratio over the first's in the same round, the geometric mean and its
 * 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)), after a warm round. Every generation's tokens and
 * final logits are the first copy's bytes -- the run exits otherwise.
 * usage: decode_place ROUNDS REPS MODEL.bcirq8 SHORT LONG MAX_MIB  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_llama.h"
#include "bcir_q8_model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*generate_fn)(const bcir_q8_model *, const int32_t *, size_t, size_t, int32_t *, double *,
                           const bcir_host_allocator *, const bcir_llama_runner *);
#define ARM(p)                                                                                                    \
  int p##_bcir_llama_generate_greedy_with_runner(const bcir_q8_model *, const int32_t *, size_t, size_t,        \
                                                 int32_t *, double *, const bcir_host_allocator *,              \
                                                 const bcir_llama_runner *);                                      \
  void p##_bcir_q8_model_init(bcir_q8_model *);                                                                 \
  int p##_bcir_q8_model_load_limited(const char *, bcir_q8_model *, char *, size_t, uint64_t);                  \
  void p##_bcir_q8_model_free(bcir_q8_model *);
#include "arms.h"
#undef ARM
#define ARM(p) p##_bcir_llama_generate_greedy_with_runner,
static const generate_fn GENERATE[] = {
#include "arms.h"
};
#undef ARM
#define ARM(p) #p,
static const char *const NAMES[] = {
#include "arms.h"
};
#undef ARM
#define ARM(p) {p##_bcir_q8_model_init, p##_bcir_q8_model_load_limited, p##_bcir_q8_model_free},
static const struct {
  void (*init)(bcir_q8_model *);
  int (*load)(const char *, bcir_q8_model *, char *, size_t, uint64_t);
  void (*release)(bcir_q8_model *);
} LOADER[] = {
#include "arms.h"
};
#undef ARM
enum { COPIES = (int)(sizeof GENERATE / sizeof *GENERATE), ARMS = COPIES + 1, MAXR = 200, MAXNEW = 1000 };

static const double T95[] = {0,     0,     12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
                             2.306, 2.262, 2.228,  2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
                             2.110, 2.101, 2.093,  2.086, 2.080, 2.074, 2.069};
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static int cmp(const void *x, const void *y) {
  double p = *(const double *)x, q = *(const double *)y;
  return (p > q) - (p < q);
}
static double median(const double *x, int n) {
  double s[MAXR];
  memcpy(s, x, (size_t)n * sizeof(double));
  qsort(s, (size_t)n, sizeof(double), cmp);
  return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}
static void interval(const double *r, int n, double out[3]) {
  double m = 0.0, v = 0.0, h;
  int i;
  for (i = 0; i < n; ++i) m += log(r[i]);
  m /= n;
  for (i = 0; i < n; ++i) v += (log(r[i]) - m) * (log(r[i]) - m);
  v /= n - 1;
  h = (n <= 24 ? T95[n] : 1.96 + 2.4 / (n - 1)) * sqrt(v / n);
  out[0] = exp(m), out[1] = exp(m - h), out[2] = exp(m + h);
}

int main(int argc, char **argv) {
  static double per_token[ARMS][MAXR];
  static int32_t want[MAXNEW], got[MAXNEW];
  bcir_host_allocator allocator = bcir_host_allocator_default();
  bcir_q8_model model;
  char err[256];
  const int32_t prompt[5] = {1, 100, 200, 300, 400};
  int rounds, reps, shortn, longn, cap, a, r, q, i;
  if (argc != 7 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (reps = atoi(argv[2])) < 1 ||
      (shortn = atoi(argv[4])) < 1 || (longn = atoi(argv[5])) <= shortn || longn > MAXNEW ||
      (cap = atoi(argv[6])) < 1) {
    fprintf(stderr, "usage: decode_place ROUNDS REPS MODEL.bcirq8 SHORT LONG MAX_MIB\n");
    return 2;
  }
  LOADER[0].init(&model);
  if (LOADER[0].load(argv[3], &model, err, sizeof err, (uint64_t)cap << 20)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  size_t vocab = model.vocab_size;
  double *want_logits = malloc(vocab * sizeof(double)), *logits = malloc(vocab * sizeof(double));
  if (!want_logits || !logits) return 1;
  if (GENERATE[0](&model, prompt, 5, (size_t)longn, want, want_logits, &allocator, NULL)) return 1;
  for (r = -1; r < rounds; r++)
    for (q = 0; q < ARMS; q++) {
      double t[2] = {0.0, 0.0};
      a = (q + (r > 0 ? r : 0)) % ARMS;
      for (i = 0; i < 2; i++) {
        size_t n = (size_t)(i ? longn : shortn);
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          int rc = GENERATE[a % COPIES](&model, prompt, 5, n, got, logits, &allocator, NULL);
          if (rc || memcmp(got, want, n * sizeof *got) != 0 ||
              (n == (size_t)longn && memcmp(logits, want_logits, vocab * sizeof *logits) != 0)) {
            fprintf(stderr, "%s's generation of %zu differs from %s's\n", NAMES[a % COPIES], n, NAMES[0]);
            return 1;
          }
        }
        t[i] = (now() - t0) / reps;
      }
      if (r >= 0) per_token[a][r] = (t[1] - t[0]) / (double)(longn - shortn) * 1e3;
    }
  printf("{\"model\":\"%s\",\"vocab\":%zu,\"rounds\":%d,\"reps\":%d,\"short\":%d,\"long\":%d,"
         "\"every_generation_the_first_copys_bytes\":true,\"first\":\"%s\",\"first_per_token_ms\":%.5f,\"arms\":{",
         argv[3], vocab, rounds, reps, shortn, longn, NAMES[0], median(per_token[0], rounds));
  for (a = 1; a < ARMS; a++) {
    double ratio[MAXR], iv[3];
    for (r = 0; r < rounds; r++) ratio[r] = per_token[a][r] / per_token[0][r];
    interval(ratio, rounds, iv);
    printf("%s\"%s\":{\"per_token_ms\":%.5f,\"ratio\":[%.4f,%.4f,%.4f]}", a > 1 ? "," : "",
           a == COPIES ? "A/A" : NAMES[a], median(per_token[a], rounds), iv[0], iv[1], iv[2]);
  }
  printf("}}\n");
  LOADER[0].release(&model);
  free(want_logits), free(logits);
  return 0;
}
