/* Q8-BOUND's kernel A/B in one binary: bound_switch.py's kernel unit, each vector instance's matvec with the
 * once-a-call choice switched off (every call the committed kernels) and on (the unit as committed), and the widest
 * instance off again (the A/A control); one set of buffers. Per case a warm round, then ROUNDS rounds in rotated arm
 * order, an arm's time the mean of REPS calls; each arm's ratio over its instance's off arm in the same round (the
 * A/A's over AVX-512 off), the geometric mean and its 95% interval (Student t to 24 rounds, then
 * 1.96 + 2.4/(n-1)); every call's outputs the portable loops' bytes, and every call of an arm on that arm's path
 * (bcir_ai_q8_bound_calls) -- the run exits otherwise. A case is IN,OUT,GROUP (the matvec of IN coordinates into OUT
 * outputs); inputs uniform on a 1.37e-4 grid over [-1.37, 1.37], codes over [-127, 127], exponents over [-30, 10]
 * (the ranges Q8-LANES' q8_ab.c draws from: every x * s proved exact). An instance the host does not admit is null.
 * usage: bound_kernels ROUNDS REPS CASE...  -> one JSON line a case */
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

enum { MAXR = 64, ARMS = 5 };
static const int arm_instance[ARMS] = {4, 4, 2, 2, 4};
static const int arm_off[ARMS] = {1, 0, 1, 0, 1};
static const char *arm_name[ARMS] = {"avx512_off", "avx512_on", "avx2_off", "avx2_on", "A/A"};
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
static uint64_t rng = UINT64_C(0x9E3779B97F4A7C15);
static uint32_t next(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (uint32_t)(rng >> 11);
}

int main(int argc, char **argv) {
  int rounds, reps, instances = bcir_ai_q8_instances();
  if (argc < 4 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (reps = atoi(argv[2])) < 1) {
    fprintf(stderr, "usage: bound_kernels ROUNDS REPS IN,OUT,GROUP...\n");
    return 2;
  }
  for (int a = 3; a < argc; a++) {
    size_t in, out, g, n, groups;
    static double t[ARMS][MAXR];
    if (sscanf(argv[a], "%zu,%zu,%zu", &in, &out, &g) != 3 || !in || !out || !g) {
      fprintf(stderr, "bound_kernels: case %s\n", argv[a]);
      return 2;
    }
    n = in * out;
    groups = 1u + (n - 1u) / g;
    double *x = malloc(in * sizeof(double)), *ref = malloc(out * sizeof(double)), *got = malloc(out * sizeof(double));
    uint8_t *codes = malloc(n), *exps = malloc(2u * groups);
    if (!x || !ref || !got || !codes || !exps) return 1;
    for (size_t i = 0; i < in; i++) x[i] = ((int)(next() % 20001u) - 10000) * 1.37e-4;
    for (size_t i = 0; i < n; i++) codes[i] = (uint8_t)(int8_t)((int)(next() % 255u) - 127);
    for (size_t i = 0; i < groups; i++) {
      uint16_t u = (uint16_t)(int16_t)((int)(next() % 41u) - 30);
      exps[2u * i] = (uint8_t)u;
      exps[2u * i + 1u] = (uint8_t)(u >> 8);
    }
    if (bcir_ai_q8_matvec_f64_instance(1, x, in, codes, n, exps, groups, out, g, ref) != BCIR_AI_OK) return 1;
    for (int r = -1; r < rounds; r++)
      for (int k = 0; k < ARMS; k++) {
        int arm = (k + (r < 0 ? 0 : r)) % ARMS, inst = arm_instance[arm];
        unsigned long hoisted, again;
        if (!(instances & inst)) continue;
        bcir_ai_q8_bound_off = arm_off[arm];
        hoisted = bcir_ai_q8_bound_calls[0];
        again = bcir_ai_q8_bound_calls[1];
        double t0 = now();
        for (int rep = 0; rep < reps; rep++) {
          if (bcir_ai_q8_matvec_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) != BCIR_AI_OK ||
              memcmp(ref, got, out * sizeof(double))) {
            fprintf(stderr, "bound_kernels: %s, case %s: an output differs from the portable loops'\n",
                    arm_name[arm], argv[a]);
            return 1;
          }
        }
        if (r >= 0) t[arm][r] = (now() - t0) / reps * 1e6;
        hoisted = bcir_ai_q8_bound_calls[0] - hoisted;
        again = bcir_ai_q8_bound_calls[1] - again;
        if (arm_off[arm] ? hoisted != 0u || again != (unsigned long)reps : hoisted != (unsigned long)reps || again) {
          fprintf(stderr, "bound_kernels: %s, case %s: %lu calls hoisted, %lu not, of %d\n", arm_name[arm], argv[a],
                  hoisted, again, reps);
          return 1;
        }
      }
    printf("{\"case\":\"%s\",\"rounds\":%d,\"reps\":%d", argv[a], rounds, reps);
    for (int arm = 0; arm < ARMS; arm++) {
      if (!(instances & arm_instance[arm])) {
        printf(",\"%s\":null", arm_name[arm]);
        continue;
      }
      printf(",\"%s\":{\"us\":%.4f", arm_name[arm], median(t[arm], rounds));
      if (arm != 0 && arm != 2) {
        int base = arm == 3 ? 2 : 0;
        double ratio[MAXR], ci[3];
        for (int r = 0; r < rounds; r++) ratio[r] = t[arm][r] / t[base][r];
        interval(ratio, rounds, ci);
        printf(",\"ratio\":%.4f,\"interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
      }
      printf("}");
    }
    printf(",\"every_output_the_portable_loops_bytes\":true,\"every_call_on_its_arms_path\":true}\n");
    free(x);
    free(ref);
    free(got);
    free(codes);
    free(exps);
  }
  return 0;
}
