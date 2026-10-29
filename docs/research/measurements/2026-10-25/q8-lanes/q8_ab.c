/* Q8-LANES' kernel A/B: runtime/c/bcir_ai_kernels.c's BCIRQ8 matvec and row dots before Q8-LANES (BEFORE: the tree
 * before's unit, built with its entry points renamed before_*) and after it on each instance -- 1 (the portable
 * loops), 2 (AVX2), 4 (AVX-512) -- and BEFORE again (the A/A control), one process, one set of buffers: per case a
 * warm round, then ROUNDS rounds in rotated arm order, an arm's time the mean of REPS calls; each arm's ratio over
 * BEFORE's in the same round, the geometric mean and its 95% interval (Student t to 24 rounds, then
 * 1.96 + 2.4/(n-1)); every arm's outputs BEFORE's bytes, every call (the run exits otherwise). BEFORE runs its
 * prevalidated entries, the after arms their _instance entries (the prevalidated ones are instance 0's). A case is
 * m:IN,OUT,GROUP (the matvec of IN coordinates into OUT outputs) or r:IN,ROWS,GROUP (ROWS row dots of IN
 * coordinates); inputs uniform on a 1.37e-4 grid over [-1.37, 1.37], codes over [-127, 127], exponents over
 * [-30, 10]. An instance the host does not admit is null.
 * usage: q8_ab ROUNDS REPS CASE...  -> one JSON line a case */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int before_q8_matvec_f64_prevalidated(const double *, size_t, const uint8_t *, size_t, const uint8_t *, size_t,
                                      size_t, size_t, double *);
int before_q8_rows_dot_f64_prevalidated(const double *, size_t, const uint8_t *, size_t, const uint8_t *, size_t,
                                        size_t, size_t, double *);

enum { ARMS = 5, MAXR = 200 };
static const int arm_instance[ARMS] = {0, 1, 2, 4, 0}; /* 0: BEFORE */
static const char *arm_name[ARMS] = {"before", "portable", "avx2", "avx512", "A/A"};
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static uint64_t state = UINT64_C(0x9E3779B97F4A7C15);
static uint32_t next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (uint32_t)(state >> 11);
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
static int call(char kind, int inst, const double *x, size_t in, const uint8_t *codes, size_t n,
                const uint8_t *exps, size_t groups, size_t out, size_t g, double *y) {
  if (kind == 'm')
    return inst == 0 ? before_q8_matvec_f64_prevalidated(x, in, codes, n, exps, groups, out, g, y)
                     : bcir_ai_q8_matvec_f64_instance(inst, x, in, codes, n, exps, groups, out, g, y);
  return inst == 0 ? before_q8_rows_dot_f64_prevalidated(x, in, codes, n, exps, groups, out, g, y)
                   : bcir_ai_q8_rows_dot_f64_instance(inst, x, in, codes, n, exps, groups, out, g, y);
}
int main(int argc, char **argv) {
  int rounds, reps, cs, admitted = bcir_ai_q8_instances();
  if (argc < 4) {
    fprintf(stderr, "usage: q8_ab ROUNDS REPS CASE...\n");
    return 2;
  }
  rounds = atoi(argv[1]);
  reps = atoi(argv[2]);
  if (rounds < 2 || rounds > MAXR || reps < 1) return 2;
  for (cs = 3; cs < argc; cs++) {
    char kind = argv[cs][0];
    unsigned long in, out, g;
    size_t n, groups, i;
    double *x, *ref, *got, ms[ARMS][MAXR];
    uint8_t *codes, *exps;
    int r, a, arm;
    if (sscanf(argv[cs] + 2, "%lu,%lu,%lu", &in, &out, &g) != 3 || (kind != 'm' && kind != 'r') || !in || !out ||
        !g)
      return 2;
    n = in * out;
    groups = 1 + (n - 1) / g;
    x = malloc(in * 8);
    ref = malloc(out * 8);
    got = malloc(out * 8);
    codes = malloc(n);
    exps = malloc(2 * groups);
    if (!x || !ref || !got || !codes || !exps) return 1;
    for (i = 0; i < in; i++) x[i] = ((int)(next() % 20001u) - 10000) * 1.37e-4;
    for (i = 0; i < n; i++) {
      uint8_t c = (uint8_t)next();
      codes[i] = c == 0x80 ? 0x81 : c;
    }
    for (i = 0; i < groups; i++) {
      uint16_t u = (uint16_t)(int16_t)((int)(next() % 41u) - 30);
      exps[2 * i] = (uint8_t)u;
      exps[2 * i + 1] = (uint8_t)(u >> 8);
    }
    if (call(kind, 0, x, in, codes, n, exps, groups, out, g, ref) != BCIR_AI_OK) return 1;
    for (r = -1; r < rounds; r++)
      for (a = 0; a < ARMS; a++) {
        int k, inst;
        double t0;
        arm = (a + (r < 0 ? 0 : r)) % ARMS;
        inst = arm_instance[arm];
        if (inst && !(admitted & inst)) {
          if (r >= 0) ms[arm][r] = 0.0;
          continue;
        }
        t0 = now();
        for (k = 0; k < reps; k++) {
          if (call(kind, inst, x, in, codes, n, exps, groups, out, g, got) != BCIR_AI_OK) return 1;
        }
        if (r >= 0) ms[arm][r] = (now() - t0) * 1e3 / reps;
        if (memcmp(ref, got, out * 8)) {
          fprintf(stderr, "%s differs on %s\n", arm_name[arm], argv[cs]);
          return 1;
        }
      }
    printf("{\"case\":\"%s\",\"kernel\":\"%s\",\"rounds\":%d,\"reps\":%d,\"before_us\":%.4f", argv[cs],
           kind == 'm' ? "matvec" : "rows_dot", rounds, reps, median(ms[0], rounds) * 1e3);
    for (arm = 1; arm < ARMS; arm++) {
      double ratio[MAXR], ci[3];
      if (arm_instance[arm] && !(admitted & arm_instance[arm])) {
        printf(",\"%s\":null", arm_name[arm]);
        continue;
      }
      for (r = 0; r < rounds; r++) ratio[r] = ms[arm][r] / ms[0][r];
      interval(ratio, rounds, ci);
      printf(",\"%s\":{\"us\":%.4f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", arm_name[arm],
             median(ms[arm], rounds) * 1e3, ci[0], ci[1], ci[2]);
    }
    printf(",\"bytes_equal\":true}\n");
    fflush(stdout);
    free(x);
    free(ref);
    free(got);
    free(codes);
    free(exps);
  }
  return 0;
}
