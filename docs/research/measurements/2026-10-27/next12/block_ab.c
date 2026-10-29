/* NEXT-12's register-blocked matvec, classified rather than discarded: the committed vector matvec
 * (bcir_ai_q8_matvec_f64_instance, Q8-PAIR's kernel where Q8-BOUND proves the call) against a register-blocked
 * prototype -- four 8-column (AVX-512) or 4-column (AVX2) accumulators held in registers down every input row, each
 * output's sum the same terms, code * (x * s), added in the same ascending-row order -- in three regimes:
 *
 *   hot     one matrix of each of the 860K decoder's seven shapes, called again and again (its codes in L1/L2);
 *   token   a token of the 860K decoder: its 28 matrices (four layers of q, k, v, o, gate, up, down, each its own
 *           codes) called in turn, as a decode calls them (about 1 MiB of codes, L2/L3);
 *   memory  four layers of TinyLlama-1.1B's widths (2048 wide, 256-wide k and v, 5632-wide gate, up and down) in
 *           turn: about 180 MiB of codes, five times this host's L3, so the codes stream from memory.
 *
 * Arms, each against the committed AVX-512 kernel's time in the same round (C512): the blocked AVX-512 kernel (B512),
 * the committed kernel again (A/A), the committed AVX2 instance (C256) and the blocked AVX2 kernel against it (B256).
 * Before any timing every arm's output of every matrix is compared with its committed kernel's, byte for byte; a
 * difference ends the run with status 2. A round runs every arm once, in an order rotated round by round, after one
 * warm round; an arm's ratio is the geometric mean of its rounds' ratios with a 95% interval (Student t to 24 rounds,
 * then 1.96 + 2.4/(n-1)), the rule NEXT-9's clock probe used.
 * usage: block_ab ROUNDS hot|token|memory  -> one JSON line per arm (and, for hot, per shape) */
#define _POSIX_C_SOURCE 200809L
#include "bcir_ai_kernels.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAXR = 401, ARMS = 5, MAXM = 28 };
static const char *const ARM_NAME[ARMS] = {"C512", "B512", "A/A", "C256", "B256"};

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static uint64_t rng = UINT64_C(0x9E3779B97F4A7C15);
static uint64_t next(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
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
  static double s[MAXR];
  memcpy(s, x, (size_t)n * sizeof(double));
  qsort(s, (size_t)n, sizeof(double), cmpd);
  return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/* a group's scale, 2**e built from its bits (the loader admits e in [-300, 300]) */
static double scale_of(const uint8_t *e, size_t g) {
  int v = (int)(int16_t)((uint16_t)e[2u * g] | ((uint16_t)e[2u * g + 1u] << 8));
  uint64_t bits = (uint64_t)(v + 1023) << 52;
  double s;
  memcpy(&s, &bits, sizeof s);
  return s;
}

/* The prototype, AVX-512: out % 8 == 0, group size 32, every x * s exact (the calls Q8-BOUND proves). Each block of
 * 32 columns keeps four 8-column accumulators in registers down every input row; a vector's eight columns lie in one
 * group or straddle two (b, the columns left in the row's first group, says where), so a vector takes s0 or s1. */
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
/* The prototype, AVX2: the same blocks of 16 columns, four 4-column accumulators */
__attribute__((target("avx2"))) static inline __m256d cvt4(const uint8_t *c) {
  int32_t w;
  memcpy(&w, c, sizeof w);
  return _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(w)));
}
__attribute__((target("avx2"))) static void blk256_xs(const double *x, size_t in, const uint8_t *codes,
                                                       const uint8_t *exps, size_t out, double *y) {
  size_t k0, i;
  for (k0 = 0; k0 < out; k0 += 16u) {
    size_t nv = (out - k0) / 4u < 4u ? (out - k0) / 4u : 4u;
    __m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (i = 0; i < in; ++i) {
      size_t e = i * out + k0, g = e >> 5, b = ((g + 1u) << 5) - e;
      double xi = x[i];
      __m256d s0 = _mm256_set1_pd(xi * scale_of(exps, g));
      __m256d s1 = b < 4u * nv ? _mm256_set1_pd(xi * scale_of(exps, g + 1u)) : s0;
      const uint8_t *c = codes + e;
      a0 = _mm256_add_pd(a0, _mm256_mul_pd(cvt4(c), s0));
      if (nv > 1) a1 = _mm256_add_pd(a1, _mm256_mul_pd(cvt4(c + 4), b > 4u ? s0 : s1));
      if (nv > 2) a2 = _mm256_add_pd(a2, _mm256_mul_pd(cvt4(c + 8), b > 8u ? s0 : s1));
      if (nv > 3) a3 = _mm256_add_pd(a3, _mm256_mul_pd(cvt4(c + 12), b > 12u ? s0 : s1));
    }
    _mm256_storeu_pd(y + k0, a0);
    if (nv > 1) _mm256_storeu_pd(y + k0 + 4, a1);
    if (nv > 2) _mm256_storeu_pd(y + k0 + 8, a2);
    if (nv > 3) _mm256_storeu_pd(y + k0 + 12, a3);
  }
}

static uint8_t *codes[MAXM], *exps[MAXM];
static size_t ins[MAXM], outs[MAXM], ns[MAXM], gs[MAXM];
static double *x, *y;
static int nm;

static void run(int arm, int m) {
  switch (arm) {
  case 0:
  case 2:
    (void)bcir_ai_q8_matvec_f64_instance(4, x, ins[m], codes[m], ns[m], exps[m], 2u * gs[m], outs[m], 32u, y);
    break;
  case 1: blk512_xs(x, ins[m], codes[m], exps[m], outs[m], y); break;
  case 3:
    (void)bcir_ai_q8_matvec_f64_instance(2, x, ins[m], codes[m], ns[m], exps[m], 2u * gs[m], outs[m], 32u, y);
    break;
  default: blk256_xs(x, ins[m], codes[m], exps[m], outs[m], y); break;
  }
}

/* every arm's output of matrix m against its committed kernel's, byte for byte */
static int same_bits(int m) {
  static double ref[6000], got[6000];
  int arm;
  for (arm = 1; arm < ARMS; ++arm) {
    int base = 0; /* every instance of the committed kernel is the portable loops' bits (Q8-LANES), so all arms are held to C512 */
    run(base, m);
    memcpy(ref, y, outs[m] * sizeof(double));
    run(arm, m);
    memcpy(got, y, outs[m] * sizeof(double));
    if (memcmp(ref, got, outs[m] * sizeof(double)) != 0) return 0;
  }
  return 1;
}

static void make(int m, size_t in, size_t out) {
  size_t i;
  ins[m] = in, outs[m] = out, ns[m] = in * out, gs[m] = (ns[m] + 31u) / 32u;
  codes[m] = malloc(ns[m]), exps[m] = malloc(2u * gs[m]);
  if (!codes[m] || !exps[m]) exit(3);
  for (i = 0; i < ns[m]; ++i) codes[m][i] = (uint8_t)(int8_t)((int)(next() % 255u) - 127);
  for (i = 0; i < gs[m]; ++i) {
    int16_t e = (int16_t)(-9 - (int)(next() % 4u));
    memcpy(exps[m] + 2u * i, &e, 2u);
  }
}

/* one timed measurement: `reps` passes over matrices [m0, m1) on `arm`, microseconds a pass */
static double timed(int arm, int m0, int m1, int reps) {
  int rep, m;
  double t = now();
  for (rep = 0; rep < reps; ++rep)
    for (m = m0; m < m1; ++m) run(arm, m);
  return (now() - t) / reps * 1e6;
}

static void report(const char *regime, const char *what, int rounds, double us[ARMS][MAXR]) {
  static double r[MAXR];
  double ci[3];
  int arm, k;
  for (arm = 0; arm < ARMS; ++arm) {
    int base = arm >= 3 && arm != 3 ? 3 : 0;
    printf("{\"regime\":\"%s\",\"what\":\"%s\",\"arm\":\"%s\",\"median_us\":%.4f", regime, what, ARM_NAME[arm],
           median(us[arm], rounds));
    if (arm != 0 && arm != 3) {
      for (k = 0; k < rounds; ++k) r[k] = us[arm][k] / us[base][k];
      interval(r, rounds, ci);
      printf(",\"against\":\"%s\",\"ratio\":%.4f,\"lo\":%.4f,\"hi\":%.4f", ARM_NAME[base], ci[0], ci[1], ci[2]);
    }
    printf(",\"rounds\":%d}\n", rounds);
  }
}

int main(int argc, char **argv) {
  static const size_t s860[7][2] = {{128, 128}, {128, 64}, {128, 64}, {128, 128}, {128, 344}, {128, 344}, {344, 128}};
  static const size_t stiny[7][2] = {{2048, 2048}, {2048, 256}, {2048, 256}, {2048, 2048},
                                     {2048, 5632}, {2048, 5632}, {5632, 2048}};
  static double us[ARMS][MAXR];
  int rounds, k, arm, m, sh;
  const char *regime;
  if (argc != 3 || (rounds = atoi(argv[1])) < 5 || rounds > MAXR - 1 ||
      (strcmp(argv[2], "hot") && strcmp(argv[2], "token") && strcmp(argv[2], "memory"))) {
    fprintf(stderr, "usage: block_ab ROUNDS hot|token|memory\n");
    return 64;
  }
  regime = argv[2];
  x = aligned_alloc(64, 6144 * sizeof(double));
  y = aligned_alloc(64, 6144 * sizeof(double));
  if (!x || !y) return 3;
  for (k = 0; k < 6144; ++k) x[k] = ((double)(next() % 2001u) - 1000.5) / 997.0;
  if (!strcmp(regime, "memory")) {
    for (m = 0; m < 28; ++m) make(m, stiny[m % 7][0], stiny[m % 7][1]);
    nm = 28;
  } else {
    for (m = 0; m < 28; ++m) make(m, s860[m % 7][0], s860[m % 7][1]);
    nm = 28;
  }
  for (m = 0; m < nm; ++m)
    if (!same_bits(m)) {
      printf("{\"regime\":\"%s\",\"error\":\"matrix %d: an arm's output differs from its committed kernel's\"}\n",
             regime, m);
      return 2;
    }
  printf("{\"regime\":\"%s\",\"bits\":\"every arm's output of every matrix the committed kernel's, byte for byte\","
         "\"matrices\":%d}\n", regime, nm);
  if (!strcmp(regime, "hot")) {
    for (sh = 0; sh < 7; ++sh) {
      char what[64];
      for (k = -1; k < rounds; ++k)
        for (arm = 0; arm < ARMS; ++arm) {
          int a = (arm + (k > 0 ? k : 0)) % ARMS;
          double t = timed(a, sh, sh + 1, 400);
          if (k >= 0) us[a][k] = t;
        }
      snprintf(what, sizeof what, "%zux%zu", ins[sh], outs[sh]);
      report(regime, what, rounds, us);
    }
    return 0;
  }
  for (k = -1; k < rounds; ++k)
    for (arm = 0; arm < ARMS; ++arm) {
      int a = (arm + (k > 0 ? k : 0)) % ARMS;
      double t = timed(a, 0, nm, !strcmp(regime, "token") ? 20 : 1);
      if (k >= 0) us[a][k] = t;
    }
  report(regime, !strcmp(regime, "token") ? "a token's 28 matvecs" : "four layers' 28 matvecs", rounds, us);
  return 0;
}
