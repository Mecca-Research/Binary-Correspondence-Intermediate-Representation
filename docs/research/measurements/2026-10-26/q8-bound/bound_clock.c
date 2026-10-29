/* Q8-BOUND's clock probe (NEXT-9's clock_probe.c on bound_switch.py's unit): the 860K decoder's 28 matvecs (four
 * layers of q, k, v, o, gate, up and down, groups of 32) through the unit's prevalidated entry
 * (bcir_ai_q8_matvec_f64_prevalidated, as the decoder calls it), Q8-BOUND's choice off or on, each pass followed by
 * a dependent integer add chain of CHAIN steps timed alone. A step of the chain is one cycle (each add waits for the
 * last), so its nanoseconds a step are the core's cycle time while the matvecs run around it; an arm with no
 * matvecs (idle) reads the core's cycle time with only the chain running. A round runs the arms in rotated order
 * (off, on, off again as the A/A control, idle), each PASSES passes of the matvecs and a chain after each; an arm's
 * ns a step and its matvecs' time a pass are its medians over the rounds, and each arm's ratio over off's in the same
 * round the geometric mean and its 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)), after a warm round.
 * In mode `pass` the chain follows each pass of 28 (the matvecs back to back, as in a kernel bench); in mode
 * `matvec` a chain of CHAIN/28 steps follows each matvec, as a decode runs its other operations between them. Every
 * matvec of an arm took that arm's path (bcir_ai_q8_bound_calls) -- the run exits otherwise.
 * usage: bound_clock ROUNDS PASSES CHAIN pass|matvec  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int bcir_ai_q8_bound_off;
extern unsigned long bcir_ai_q8_bound_calls[2];

enum { MAXR = 200, ARMS = 4, MATS = 28 };
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
/* n dependent adds: each waits for the one before it */
static uint64_t chain(uint64_t n) {
  uint64_t x = 1u;
  for (uint64_t i = 0u; i < n; ++i) __asm__ volatile("add %1, %0" : "+r"(x) : "r"(i));
  return x;
}
static uint64_t rng = UINT64_C(0x9E3779B97F4A7C15);
static uint64_t next(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

int main(int argc, char **argv) {
  static const size_t layer[7][2] = {{128, 128}, {128, 64}, {128, 64}, {128, 128}, {128, 344}, {128, 344}, {344, 128}};
  static double ns_step[ARMS][MAXR], pass_us[ARMS][MAXR];
  uint8_t *codes[MATS], *exps[MATS];
  double *x[MATS], y[344];
  size_t ins[MATS], outs[MATS], groups[MATS];
  int rounds, passes;
  long chain_steps;
  int sparse;
  volatile uint64_t sink = 0u;
  if (argc != 5 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (passes = atoi(argv[2])) < 1 ||
      (chain_steps = atol(argv[3])) < 1000 || (strcmp(argv[4], "pass") && strcmp(argv[4], "matvec"))) {
    fprintf(stderr, "usage: bound_clock ROUNDS PASSES CHAIN pass|matvec\n");
    return 2;
  }
  sparse = !strcmp(argv[4], "matvec");
  for (int m = 0; m < MATS; m++) {
    size_t n;
    ins[m] = layer[m % 7][0];
    outs[m] = layer[m % 7][1];
    n = ins[m] * outs[m];
    groups[m] = (n + 31u) / 32u;
    codes[m] = malloc(n);
    exps[m] = malloc(2u * groups[m]);
    x[m] = malloc(ins[m] * sizeof(double));
    if (!codes[m] || !exps[m] || !x[m]) return 1;
    for (size_t k = 0; k < n; k++) codes[m][k] = (uint8_t)(int8_t)((int)(next() % 255u) - 127); /* never 0x80 */
    for (size_t g = 0; g < groups[m]; g++) {
      uint16_t raw = (uint16_t)(int16_t)(-10 + (int)(next() % 7u));
      exps[m][2u * g] = (uint8_t)raw;
      exps[m][2u * g + 1u] = (uint8_t)(raw >> 8);
    }
    for (size_t i = 0; i < ins[m]; i++) x[m][i] = (double)(next() >> 11) * 0x1p-52 - 1.0;
  }
  for (int r = -1; r < rounds; r++)
    for (int a = 0; a < ARMS; a++) {
      int arm = (a + (r < 0 ? 0 : r)) % ARMS;
      double chain_s = 0.0, mv_s = 0.0;
      unsigned long hoisted = bcir_ai_q8_bound_calls[0], again = bcir_ai_q8_bound_calls[1];
      bcir_ai_q8_bound_off = arm != 1;
      for (int p = 0; p < passes; p++) {
        if (sparse) {
          for (int m = 0; m < MATS; m++) {
            double t0 = now(), t1;
            if (arm != 3 && bcir_ai_q8_matvec_f64_prevalidated(x[m], ins[m], codes[m], ins[m] * outs[m], exps[m],
                                                               groups[m], outs[m], 32u, y) != BCIR_AI_OK)
              return 1;
            t1 = now();
            sink += chain((uint64_t)(chain_steps / MATS));
            chain_s += now() - t1;
            mv_s += t1 - t0;
          }
          continue;
        }
        double t0 = now(), t1;
        if (arm != 3)
          for (int m = 0; m < MATS; m++)
            if (bcir_ai_q8_matvec_f64_prevalidated(x[m], ins[m], codes[m], ins[m] * outs[m], exps[m], groups[m], outs[m],
                                                   32u, y) != BCIR_AI_OK)
              return 1;
        t1 = now();
        sink += chain((uint64_t)chain_steps);
        chain_s += now() - t1;
        mv_s += t1 - t0;
      }
      hoisted = bcir_ai_q8_bound_calls[0] - hoisted;
      again = bcir_ai_q8_bound_calls[1] - again;
      if (hoisted != (arm == 1 ? (unsigned long)passes * MATS : 0u) ||
          again != (arm == 0 || arm == 2 ? (unsigned long)passes * MATS : 0u)) {
        fprintf(stderr, "bound_clock: arm %d: %lu matvecs hoisted, %lu not\n", arm, hoisted, again);
        return 1;
      }
      if (r >= 0) {
        ns_step[arm][r] = chain_s * 1e9 / ((double)passes * (double)(sparse ? chain_steps / MATS * MATS : chain_steps));
        pass_us[arm][r] = mv_s * 1e6 / passes;
      }
    }
  printf("{\"mode\":\"%s\",\"rounds\":%d,\"passes\":%d,\"chain\":%ld", argv[4], rounds, passes, chain_steps);
  static const char *names[ARMS] = {"off", "on", "A/A", "idle"};
  for (int a = 0; a < ARMS; a++) {
    double ratio[MAXR], ci[3];
    printf(",\"%s\":{\"chain_ns_a_step\":%.4f", names[a], median(ns_step[a], rounds));
    if (a != 3) printf(",\"matvecs_us_a_pass\":%.2f", median(pass_us[a], rounds));
    if (a != 0) {
      for (int r = 0; r < rounds; r++) ratio[r] = ns_step[a][r] / ns_step[0][r];
      interval(ratio, rounds, ci);
      printf(",\"chain_over_off\":%.4f,\"interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
    }
    if (a == 1 || a == 2) {
      for (int r = 0; r < rounds; r++) ratio[r] = pass_us[a][r] / pass_us[0][r];
      interval(ratio, rounds, ci);
      printf(",\"matvecs_over_off\":%.4f,\"matvecs_interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
    }
    printf("}");
  }
  printf(",\"every_matvec_on_its_arms_path\":true,\"sink\":%llu}\n", (unsigned long long)(sink & 1u));
  for (int m = 0; m < MATS; m++) {
    free(codes[m]);
    free(exps[m]);
    free(x[m]);
  }
  return 0;
}
