/* NEXT-12's clock probe: NEXT-9's instrument (measurements/2026-10-26/next9/clock_probe.c), run with the register-
 * blocked prototype as an arm, so the question "does the blocked kernel lose in the decoder through the core's
 * clock?" is asked on the host that measured the loss rather than answered from another host's reading.
 *
 * The 860K decoder's 28 matvecs (four layers of q, k, v, o, gate, up and down, groups of 32), each followed -- mode
 * `matvec` -- by a dependent integer add chain of CHAIN/28 steps timed alone, as a decode runs its other operations
 * between its matvecs (mode `pass`: one chain of CHAIN steps after the 28 back to back). A step of the chain is one
 * cycle (each add waits for the last), so its nanoseconds a step are the core's cycle time while the matvecs run
 * around it. This is a cycle LENGTH read against CLOCK_MONOTONIC, with no frequency in hertz claimed and no PMU: on a
 * VM it is the one instrument that sees the core slow down. Arms: the committed kernel's AVX-512 instance (C512, the
 * reference), the blocked AVX-512 prototype (B512), the committed kernel again (A/A), and no matvecs at all (idle).
 * A round runs the arms in an order rotated round by round, after one warm round; an arm's chain time and matvec
 * time a pass are medians over the rounds, and each arm's ratio over C512's in the same round the geometric mean and
 * its 95% interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)). Before timing, the blocked kernel's output of
 * every matrix is compared with the committed kernel's, byte for byte (status 2 on a difference).
 * usage: block_clock ROUNDS PASSES CHAIN pass|matvec  -> one JSON line */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double median(const double *x, int n) {
  double s[MAXR];
  memcpy(s, x, (size_t)n * sizeof(double));
  qsort(s, (size_t)n, sizeof(double), cmpd);
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

/* the prototype, as block_ab.c holds it (AVX-512, group size 32, the calls Q8-BOUND proves) */
static double scale_of(const uint8_t *e, size_t g) {
  int v = (int)(int16_t)((uint16_t)e[2u * g] | ((uint16_t)e[2u * g + 1u] << 8));
  uint64_t bits = (uint64_t)(v + 1023) << 52;
  double s;
  memcpy(&s, &bits, sizeof s);
  return s;
}
__attribute__((target("avx512f"))) static inline __m512d cvt8(const uint8_t *c) {
  int64_t w;
  memcpy(&w, c, sizeof w);
  return _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w)));
}
__attribute__((target("avx512f"))) static void blk512_xs(const double *x, size_t in, const uint8_t *codes,
                                                          const uint8_t *exps, size_t out, double *y) {
  size_t k0, i;
  for (k0 = 0; k0 < out; k0 += 32u) {
    size_t nv = (out - k0) / 8u < 4u ? (out - k0) / 8u : 4u;
    __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (i = 0; i < in; ++i) {
      size_t e = i * out + k0, g = e >> 5, b = ((g + 1u) << 5) - e;
      double xi = x[i];
      __m512d s0 = _mm512_set1_pd(xi * scale_of(exps, g));
      __m512d s1 = b < 8u * nv ? _mm512_set1_pd(xi * scale_of(exps, g + 1u)) : s0;
      const uint8_t *c = codes + e;
      a0 = _mm512_add_pd(a0, _mm512_mul_pd(cvt8(c), s0));
      if (nv > 1) a1 = _mm512_add_pd(a1, _mm512_mul_pd(cvt8(c + 8), b > 8u ? s0 : s1));
      if (nv > 2) a2 = _mm512_add_pd(a2, _mm512_mul_pd(cvt8(c + 16), b > 16u ? s0 : s1));
      if (nv > 3) a3 = _mm512_add_pd(a3, _mm512_mul_pd(cvt8(c + 24), b > 24u ? s0 : s1));
    }
    _mm512_storeu_pd(y + k0, a0);
    if (nv > 1) _mm512_storeu_pd(y + k0 + 8, a1);
    if (nv > 2) _mm512_storeu_pd(y + k0 + 16, a2);
    if (nv > 3) _mm512_storeu_pd(y + k0 + 24, a3);
  }
}

int main(int argc, char **argv) {
  static const size_t layer[7][2] = {{128, 128}, {128, 64}, {128, 64}, {128, 128}, {128, 344}, {128, 344}, {344, 128}};
  static double ns_step[ARMS][MAXR], pass_us[ARMS][MAXR];
  static const char *names[ARMS] = {"C512", "B512", "A/A", "idle"};
  uint8_t *codes[MATS], *exps[MATS];
  double *x[MATS], y[344], ref[344];
  size_t ins[MATS], outs[MATS], groups[MATS];
  int rounds, passes, sparse;
  long chain_steps;
  volatile uint64_t sink = 0u;
  if (argc != 5 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR || (passes = atoi(argv[2])) < 1 ||
      (chain_steps = atol(argv[3])) < 1000 || (strcmp(argv[4], "pass") && strcmp(argv[4], "matvec"))) {
    fprintf(stderr, "usage: block_clock ROUNDS PASSES CHAIN pass|matvec\n");
    return 64;
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
  for (int m = 0; m < MATS; m++) {
    if (bcir_ai_q8_matvec_f64_instance(4, x[m], ins[m], codes[m], ins[m] * outs[m], exps[m], 2u * groups[m], outs[m],
                                       32u, ref) != BCIR_AI_OK)
      return 1;
    blk512_xs(x[m], ins[m], codes[m], exps[m], outs[m], y);
    if (memcmp(ref, y, outs[m] * sizeof(double)) != 0) {
      printf("{\"error\":\"matrix %d: the blocked kernel's output differs from the committed kernel's\"}\n", m);
      return 2;
    }
  }
  for (int r = -1; r < rounds; r++)
    for (int a = 0; a < ARMS; a++) {
      int arm = (a + (r < 0 ? 0 : r)) % ARMS;
      double chain_s = 0.0, mv_s = 0.0;
      for (int p = 0; p < passes; p++) {
        if (sparse) {
          for (int m = 0; m < MATS; m++) {
            double t0 = now(), t1;
            if (arm == 1)
              blk512_xs(x[m], ins[m], codes[m], exps[m], outs[m], y);
            else if (arm != 3 && bcir_ai_q8_matvec_f64_instance(4, x[m], ins[m], codes[m], ins[m] * outs[m], exps[m],
                                                                2u * groups[m], outs[m], 32u, y) != BCIR_AI_OK)
              return 1;
            t1 = now();
            sink += chain((uint64_t)(chain_steps / MATS));
            chain_s += now() - t1;
            mv_s += t1 - t0;
          }
          continue;
        }
        double t0 = now(), t1;
        for (int m = 0; m < MATS && arm != 3; m++) {
          if (arm == 1)
            blk512_xs(x[m], ins[m], codes[m], exps[m], outs[m], y);
          else if (bcir_ai_q8_matvec_f64_instance(4, x[m], ins[m], codes[m], ins[m] * outs[m], exps[m], 2u * groups[m],
                                                  outs[m], 32u, y) != BCIR_AI_OK)
            return 1;
        }
        t1 = now();
        sink += chain((uint64_t)chain_steps);
        chain_s += now() - t1;
        mv_s += t1 - t0;
      }
      if (r >= 0) {
        ns_step[arm][r] = chain_s * 1e9 / ((double)passes * (double)(sparse ? chain_steps / MATS * MATS : chain_steps));
        pass_us[arm][r] = mv_s * 1e6 / passes;
      }
    }
  printf("{\"mode\":\"%s\",\"rounds\":%d,\"passes\":%d,\"chain\":%ld", argv[4], rounds, passes, chain_steps);
  for (int a = 0; a < ARMS; a++) {
    double ratio[MAXR], ci[3];
    printf(",\"%s\":{\"chain_ns_a_step\":%.4f", names[a], median(ns_step[a], rounds));
    if (a != 3) printf(",\"matvecs_us_a_pass\":%.2f", median(pass_us[a], rounds));
    if (a != 0) {
      for (int r = 0; r < rounds; r++) ratio[r] = ns_step[a][r] / ns_step[0][r];
      interval(ratio, rounds, ci);
      printf(",\"chain_over_C512\":%.4f,\"interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
    }
    if (a == 1 || a == 2) {
      for (int r = 0; r < rounds; r++) ratio[r] = pass_us[a][r] / pass_us[0][r];
      interval(ratio, rounds, ci);
      printf(",\"matvecs_over_C512\":%.4f,\"matvecs_interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
    }
    printf("}");
  }
  printf(",\"sink\":%llu}\n", (unsigned long long)(sink & 1u));
  for (int m = 0; m < MATS; m++) {
    free(codes[m]);
    free(exps[m]);
    free(x[m]);
  }
  return 0;
}
