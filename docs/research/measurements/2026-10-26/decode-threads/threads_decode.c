/* DECODE-THREADS' decode A/B in one binary: the tree after's C decoder on one loaded model, lent no runner, the
 * trainer's deterministic pool at 2, 3 and 4 threads (the default thresholds, bcir_llama.h), and no runner again (the
 * A/A control) -- five arms, the same code on the same weights. A round runs the arms in rotated order: REPS greedy
 * generations of SHORT and of LONG new tokens from one 5-token prompt, an arm's per-token time
 * (t_long - t_short) / (LONG - SHORT); each arm's ratio over the no-runner arm's in the same round, the geometric
 * mean and its 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)), after a warm round. Every generation's
 * tokens and final logits are the no-runner arm's bytes, and the runner counts the operations it split and their
 * parts, the same every generation of an arm's length -- the run exits otherwise. The model loads with a file-size
 * cap of MAX_MIB.
 * usage: threads_decode ROUNDS REPS MODEL.bcirq8 SHORT LONG MAX_MIB  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include "bcir_llama.h"
#include "bcir_q8_model.h"
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAXR = 200, MAXNEW = 1000, ARMS = 5 };
static const unsigned threads_of[ARMS] = {1u, 2u, 3u, 4u, 1u};
static const char *names[ARMS] = {"none", "threads_2", "threads_3", "threads_4", "A/A"};

/* the pool lent through a counter: each call an operation split, each task a part */
typedef struct lent { bcir_tensor_pool *pool; size_t calls, parts; } lent;
static void lent_run(void *self, bcir_llama_task_fn task, void *ctx, size_t tasks) {
  lent *l = (lent *)self;
  l->calls++;
  l->parts += tasks;
  bcir_tensor_pool_run(l->pool, task, ctx, tasks);
}
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
  static lent lents[ARMS];
  static size_t splits[ARMS][2], parts[ARMS][2]; /* an arm's operations split and parts a generation: SHORT, LONG */
  bcir_llama_runner runners[ARMS];
  bcir_host_allocator allocator = bcir_host_allocator_default();
  bcir_q8_model model;
  char err[256];
  const int32_t prompt[5] = {1, 100, 200, 300, 400};
  int rounds, reps, shortn, longn, cap, a;
  if (argc != 7 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (reps = atoi(argv[2])) < 1 ||
      (shortn = atoi(argv[4])) < 1 || (longn = atoi(argv[5])) <= shortn || longn > MAXNEW || (cap = atoi(argv[6])) < 1) {
    fprintf(stderr, "usage: threads_decode ROUNDS REPS MODEL.bcirq8 SHORT LONG MAX_MIB\n");
    return 2;
  }
  bcir_q8_model_init(&model);
  if (bcir_q8_model_load_limited(argv[3], &model, err, sizeof err, (uint64_t)cap << 20)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  for (a = 0; a < ARMS; a++) {
    memset(&runners[a], 0, sizeof runners[a]);
    if (threads_of[a] > 1u) {
      size_t bytes = bcir_tensor_pool_size(threads_of[a]);
      void *memory = bytes ? malloc(bytes) : NULL;
      lents[a].pool = memory ? bcir_tensor_pool_init(memory, bytes, threads_of[a]) : NULL;
      if (lents[a].pool == NULL) {
        fprintf(stderr, "threads_decode: no pool of %u threads here\n", threads_of[a]);
        return 1;
      }
      runners[a].run = lent_run;
      runners[a].self = &lents[a];
      runners[a].workers = threads_of[a];
      runners[a].split_min = BCIR_LLAMA_SPLIT_MIN;
      runners[a].head_split_min = BCIR_LLAMA_HEAD_SPLIT_MIN;
    }
  }
  size_t vocab = model.vocab_size;
  double *want_logits = malloc(vocab * sizeof(double)), *logits = malloc(vocab * sizeof(double));
  if (!want_logits || !logits) return 1;
  if (bcir_llama_generate_greedy_with_runner(&model, prompt, 5, (size_t)longn, want, want_logits, &allocator, NULL))
    return 1;
  for (a = 0; a < ARMS; a++) splits[a][0] = splits[a][1] = parts[a][0] = parts[a][1] = SIZE_MAX;
  for (int r = -1; r < rounds; r++)
    for (int q = 0; q < ARMS; q++) {
      int arm = (q + (r < 0 ? 0 : r)) % ARMS;
      const bcir_llama_runner *runner = threads_of[arm] > 1u ? &runners[arm] : NULL;
      double t[2];
      for (int len = 0; len < 2; len++) {
        size_t n = (size_t)(len ? longn : shortn);
        double t0 = now();
        for (int k = 0; k < reps; k++) {
          int rc;
          lents[arm].calls = lents[arm].parts = 0u;
          rc = bcir_llama_generate_greedy_with_runner(&model, prompt, 5, n, got, logits, &allocator, runner);
          if (rc || memcmp(got, want, n * sizeof(int32_t)) || (len && memcmp(logits, want_logits, vocab * sizeof(double)))) {
            fprintf(stderr, "arm %s: a generation's tokens or logits differ\n", names[arm]);
            return 1;
          }
          if (splits[arm][len] == SIZE_MAX) splits[arm][len] = lents[arm].calls, parts[arm][len] = lents[arm].parts;
          if (lents[arm].calls != splits[arm][len] || lents[arm].parts != parts[arm][len]) {
            fprintf(stderr, "arm %s: a generation split differently\n", names[arm]);
            return 1;
          }
        }
        t[len] = (now() - t0) / reps;
      }
      if (r >= 0) per_token[arm][r] = (t[1] - t[0]) * 1e3 / (longn - shortn);
    }
  printf("{\"rounds\":%d,\"reps\":%d,\"short\":%d,\"long\":%d,\"vocab\":%zu,\"d_model\":%u,\"d_ff\":%u,\"layers\":%u,"
         "\"instances\":%d,\"split_min\":%zu,\"head_split_min\":%zu,\"none_per_token_ms\":%.5f",
         rounds, reps, shortn, longn, vocab, model.d_model, model.d_ff, model.n_layers, bcir_ai_q8_instances(),
         (size_t)BCIR_LLAMA_SPLIT_MIN, (size_t)BCIR_LLAMA_HEAD_SPLIT_MIN, median(per_token[0], rounds));
  for (a = 1; a < ARMS; a++) {
    double ratio[MAXR], ci[3];
    for (int r = 0; r < rounds; r++) ratio[r] = per_token[a][r] / per_token[0][r];
    interval(ratio, rounds, ci);
    printf(",\"%s\":{\"per_token_ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f],\"splits_a_token\":%.4f,"
           "\"parts_a_token\":%.4f}",
           names[a], median(per_token[a], rounds), ci[0], ci[1], ci[2],
           (double)(splits[a][1] - splits[a][0]) / (longn - shortn), (double)(parts[a][1] - parts[a][0]) / (longn - shortn));
  }
  printf(",\"every_generation_the_same_bytes\":true,\"every_generation_of_an_arm_split_alike\":true}\n");
  for (a = 0; a < ARMS; a++) bcir_tensor_pool_stop(lents[a].pool);
  bcir_q8_model_free(&model);
  free(want_logits);
  free(logits);
  return 0;
}
