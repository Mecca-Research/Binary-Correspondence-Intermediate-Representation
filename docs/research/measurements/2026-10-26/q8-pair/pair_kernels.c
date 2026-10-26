/* Q8-PAIR's kernel A/B in one binary: pair_switch.py's kernel unit, the choice switched at run time
 * (bcir_ai_q8_pair_off), so every arm runs the same code at the same addresses and differs only in the kernel the
 * pair's calls take. For each vector instance (4 AVX-512, 2 AVX2) three arms: OFF (the tree before's kernel: the
 * AVX-512 block, the AVX2 runs), ON (the pair, as committed) and OFF again (the A/A control) -- six arms a round in
 * rotated order, after a warm round. A case is IN,OUT,GROUP,COUNT,REPS: COUNT matrices of that shape called in turn,
 * REPS times each a pass (COUNT 1: one matrix again and again, in L2 where it fits; COUNT x IN x OUT past L2 or past
 * L3: each matrix from L3 or DRAM at its call); an arm's time a pass. Each arm's ratio over its instance's OFF in the
 * same round, the geometric mean and its 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)). Every output of
 * every call is the portable loops' bytes, and every call of an arm took its arm's kernel (bcir_ai_q8_pair_calls) --
 * the run exits otherwise.
 * usage: pair_kernels ROUNDS CASE...  -> one JSON line a case */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int bcir_ai_q8_pair_off;
extern unsigned long bcir_ai_q8_pair_calls[2];

enum { MAXR = 64, ARMS = 6 };
static const char *arm_name[ARMS] = {"avx512_off", "avx512_on", "avx512_A/A", "avx2_off", "avx2_on", "avx2_A/A"};
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void interval(const double *r, int n, double out[3]) {
  static const double t95[] = {0, 0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
                               2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069};
  double m = 0, sd = 0, h;
  for (int i = 0; i < n; i++) m += log(r[i]);
  m /= n;
  for (int i = 0; i < n; i++) sd += (log(r[i]) - m) * (log(r[i]) - m);
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
static uint64_t next64(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

int main(int argc, char **argv) {
  int rounds;
  if (argc < 3 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR) {
    fprintf(stderr, "usage: pair_kernels ROUNDS IN,OUT,GROUP,COUNT,REPS...\n");
    return 2;
  }
  if ((bcir_ai_q8_instances() & 6) != 6) {
    fprintf(stderr, "pair_kernels: this host admits no AVX2 and AVX-512 instances\n");
    return 2;
  }
  for (int a = 2; a < argc; a++) {
    size_t in, out, g, count, reps, n, groups;
    static double t[ARMS][MAXR];
    if (sscanf(argv[a], "%zu,%zu,%zu,%zu,%zu", &in, &out, &g, &count, &reps) != 5 || !in || !out || !g || !count ||
        !reps || out % 8u || g < 8u || (g & (g - 1u)))
      return 2; /* the pair's shapes only */
    n = in * out;
    groups = 1u + (n - 1u) / g;
    double *x = malloc(in * sizeof(double)), *ref = malloc(count * out * sizeof(double)),
           *got = malloc(out * sizeof(double));
    uint8_t *codes = malloc(n * count), *exps = malloc(2u * groups * count);
    if (!x || !ref || !got || !codes || !exps) return 1;
    for (size_t i = 0; i < in; i++) x[i] = ((int)(next64() % 20001u) - 10000) * 1.37e-4;
    for (size_t i = 0; i < n * count; i += 8) { /* codes in [-127, 127], eight a draw */
      uint64_t w = next64();
      for (size_t b = 0; b < 8 && i + b < n * count; b++)
        codes[i + b] = (uint8_t)(int8_t)((int)((w >> (8 * b)) % 255u) - 127);
    }
    for (size_t i = 0; i < groups * count; i++) { /* the format's exponents: every x * s exact (Q8-BOUND's hoist) */
      uint16_t u = (uint16_t)(int16_t)((int)(next64() % 41u) - 30);
      exps[2u * i] = (uint8_t)u;
      exps[2u * i + 1u] = (uint8_t)(u >> 8);
    }
    for (size_t m = 0; m < count; m++)
      if (bcir_ai_q8_matvec_f64_instance(1, x, in, codes + m * n, n, exps + 2u * groups * m, groups, out, g,
                                         ref + m * out) != BCIR_AI_OK)
        return 1;
    for (int r = -1; r < rounds; r++)
      for (int k = 0; k < ARMS; k++) {
        int arm = (k + (r < 0 ? 0 : r)) % ARMS, inst = arm < 3 ? 4 : 2, on = arm % 3 == 1;
        unsigned long pair0 = bcir_ai_q8_pair_calls[0], before0 = bcir_ai_q8_pair_calls[1];
        bcir_ai_q8_pair_off = !on;
        double t0 = now();
        for (size_t m = 0; m < count; m++)
          for (size_t q = 0; q < reps; q++)
            if (bcir_ai_q8_matvec_f64_instance(inst, x, in, codes + m * n, n, exps + 2u * groups * m, groups, out, g,
                                               got) != BCIR_AI_OK ||
                memcmp(ref + m * out, got, out * sizeof(double))) {
              fprintf(stderr, "pair_kernels: %s, case %s: an output differs from the portable loops'\n",
                      arm_name[arm], argv[a]);
              return 1;
            }
        if (r >= 0) t[arm][r] = (now() - t0) * 1e6;
        if (bcir_ai_q8_pair_calls[on ? 0 : 1] - (on ? pair0 : before0) != count * reps ||
            bcir_ai_q8_pair_calls[on ? 1 : 0] != (on ? before0 : pair0)) {
          fprintf(stderr, "pair_kernels: %s, case %s: a call off its arm's kernel\n", arm_name[arm], argv[a]);
          return 1;
        }
      }
    printf("{\"case\":\"%s\",\"rounds\":%d,\"code_bytes\":%zu", argv[a], rounds, n * count);
    for (int arm = 0; arm < ARMS; arm++) {
      double ratio[MAXR], ci[3];
      printf(",\"%s\":{\"us_a_pass\":%.2f", arm_name[arm], median(t[arm], rounds));
      if (arm % 3) {
        for (int r = 0; r < rounds; r++) ratio[r] = t[arm][r] / t[arm - arm % 3][r];
        interval(ratio, rounds, ci);
        printf(",\"ratio\":%.4f,\"interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
      }
      printf("}");
    }
    printf(",\"every_output_the_portable_loops_bytes\":true,\"every_call_on_its_arms_kernel\":true}\n");
    fflush(stdout);
    free(x);
    free(ref);
    free(got);
    free(codes);
    free(exps);
  }
  return 0;
}
