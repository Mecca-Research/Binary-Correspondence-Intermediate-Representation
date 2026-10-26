/* NEXT-9's matvec bench: the committed AVX-512 block matvec (copied byte for byte) against one whose x times
 * the group's scale is formed once a row and group -- code*(x*s) for x*(code*s), equal bit for bit wherever x*s
 * is exact (code*s always is) -- with no check (the hoist's ceiling), the same in a loop over an array of
 * accumulators, and Q8-HOIST's prototype (q8_hoist.patch's block kernel, copied byte for byte): the hoist with
 * every x*s checked, a call that formed an inexact one computed again by the committed kernel. Byte equality over
 * random shapes and value ranges: the ceiling and the loop form where every x*s is exact (every nonzero |x| in
 * [2^-722, 2^723), exponents in [-300, 300]), the prototype on every case -- among them inputs below that range
 * whose every x*s is still exact (hoisted) and inputs to 2^1023 and subnormal (computed again). Then the four timed in rotated order at
 * the decoders' shapes (every x*s exact there, so the prototype never falls back), a ratio over the committed
 * kernel's a round, its median and range -- each shape one matrix called again and again, its codes in L1 -- and
 * the 860K decoder's 28 matvecs in its order, each once a pass, its codes in L2 as in a decode. One JSON line a
 * result.  usage: hoist_bench ROUNDS (3-64) */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int exponent_le(const uint8_t *bytes, size_t group) {
  const uint8_t *value = bytes + 2u * group;
  uint16_t raw = (uint16_t)value[0] | (uint16_t)((uint16_t)value[1] << 8);
  return raw >= UINT16_C(0x8000) ? (int)raw - 65536 : (int)raw;
}
static double q8_scale(const uint8_t *exponents_le, size_t group) {
  int e = exponent_le(exponents_le, group);
  if (e >= -1022 && e <= 1023) {
    uint64_t bits = (uint64_t)(e + 1023) << 52;
    double scale;
    memcpy(&scale, &bits, sizeof scale);
    return scale;
  }
  return ldexp(1.0, e);
}
__attribute__((target("avx512f"))) static inline __m512d q8_terms512(const uint8_t *c, __m512d vx, double s) {
  int64_t w;
  memcpy(&w, c, sizeof w);
  return _mm512_mul_pd(vx, _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w))),
                                         _mm512_set1_pd(s)));
}
/* the committed kernel, byte for byte */
__attribute__((target("avx512f"))) static void ref512(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,
    double *output) {
  unsigned shift = 0u;
  size_t j0, i;
  while (((size_t)1 << shift) < group_size) ++shift;
  for (j0 = 0u; j0 < output_count; j0 += 64u) {
    size_t vectors = output_count - j0 >= 64u ? 8u : (output_count - j0) / 8u;
    __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (i = 0u; i < input_count; ++i) {
      size_t e0 = i * output_count + j0, group = e0 >> shift;
      const uint8_t *c = codes + e0;
      __m512d vx = _mm512_set1_pd(input[i]);
      double scale = q8_scale(exponents_le, group);
#define Q8_BLOCK_VECTOR(a, v)                                                                            \
  if (vectors > (v)) {                                                                                   \
    if (((e0 + 8u * (v)) >> shift) != group) {                                                          \
      group = (e0 + 8u * (v)) >> shift;                                                                  \
      scale = q8_scale(exponents_le, group);                                                             \
    }                                                                                                    \
    a = _mm512_add_pd(a, q8_terms512(c + 8u * (v), vx, scale));                                          \
  }
      Q8_BLOCK_VECTOR(a0, 0u); Q8_BLOCK_VECTOR(a1, 1u); Q8_BLOCK_VECTOR(a2, 2u); Q8_BLOCK_VECTOR(a3, 3u);
      Q8_BLOCK_VECTOR(a4, 4u); Q8_BLOCK_VECTOR(a5, 5u); Q8_BLOCK_VECTOR(a6, 6u); Q8_BLOCK_VECTOR(a7, 7u);
#undef Q8_BLOCK_VECTOR
    }
    if (vectors > 0u) _mm512_storeu_pd(output + j0, a0);
    if (vectors > 1u) _mm512_storeu_pd(output + j0 + 8u, a1);
    if (vectors > 2u) _mm512_storeu_pd(output + j0 + 16u, a2);
    if (vectors > 3u) _mm512_storeu_pd(output + j0 + 24u, a3);
    if (vectors > 4u) _mm512_storeu_pd(output + j0 + 32u, a4);
    if (vectors > 5u) _mm512_storeu_pd(output + j0 + 40u, a5);
    if (vectors > 6u) _mm512_storeu_pd(output + j0 + 48u, a6);
    if (vectors > 7u) _mm512_storeu_pd(output + j0 + 56u, a7);
  }
}
/* eight columns' terms with x*s formed once: code times the broadcast xs */
__attribute__((target("avx512f"))) static inline __m512d q8_terms512_xs(const uint8_t *c, __m512d vxs) {
  int64_t w;
  memcpy(&w, c, sizeof w);
  return _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w))), vxs);
}
__attribute__((target("avx512f"))) static void hoist512(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,
    double *output) {
  unsigned shift = 0u;
  size_t j0, i;
  while (((size_t)1 << shift) < group_size) ++shift;
  for (j0 = 0u; j0 < output_count; j0 += 64u) {
    size_t vectors = output_count - j0 >= 64u ? 8u : (output_count - j0) / 8u;
    __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (i = 0u; i < input_count; ++i) {
      size_t e0 = i * output_count + j0, group = e0 >> shift;
      const uint8_t *c = codes + e0;
      double x = input[i];
      __m512d vxs = _mm512_set1_pd(x * q8_scale(exponents_le, group));
#define Q8_BLOCK_VECTOR(a, v)                                                                            \
  if (vectors > (v)) {                                                                                   \
    if (((e0 + 8u * (v)) >> shift) != group) {                                                          \
      group = (e0 + 8u * (v)) >> shift;                                                                  \
      vxs = _mm512_set1_pd(x * q8_scale(exponents_le, group));                                           \
    }                                                                                                    \
    a = _mm512_add_pd(a, q8_terms512_xs(c + 8u * (v), vxs));                                             \
  }
      Q8_BLOCK_VECTOR(a0, 0u); Q8_BLOCK_VECTOR(a1, 1u); Q8_BLOCK_VECTOR(a2, 2u); Q8_BLOCK_VECTOR(a3, 3u);
      Q8_BLOCK_VECTOR(a4, 4u); Q8_BLOCK_VECTOR(a5, 5u); Q8_BLOCK_VECTOR(a6, 6u); Q8_BLOCK_VECTOR(a7, 7u);
#undef Q8_BLOCK_VECTOR
    }
    if (vectors > 0u) _mm512_storeu_pd(output + j0, a0);
    if (vectors > 1u) _mm512_storeu_pd(output + j0 + 8u, a1);
    if (vectors > 2u) _mm512_storeu_pd(output + j0 + 16u, a2);
    if (vectors > 3u) _mm512_storeu_pd(output + j0 + 24u, a3);
    if (vectors > 4u) _mm512_storeu_pd(output + j0 + 32u, a4);
    if (vectors > 5u) _mm512_storeu_pd(output + j0 + 40u, a5);
    if (vectors > 6u) _mm512_storeu_pd(output + j0 + 48u, a6);
    if (vectors > 7u) _mm512_storeu_pd(output + j0 + 56u, a7);
  }
}

/* Q8-HOIST's prototype, as q8_hoist.patch adds it to runtime/c/bcir_ai_kernels.c */
/* Q8-HOIST: is x * s, formed as xs, exact -- finite, and normal or x zero (+0.0 or -0.0, its sign kept)? */
static int q8_xs_exact(double x, double xs) {
  double magnitude = fabs(xs);
  return magnitude <= 0x1.fffffffffffffp1023 && (x == 0.0 || magnitude >= 0x1p-1022);
}
/* The block with x * s formed once a row and group (Q8-HOIST): 0, and the output to be computed again, when
 * one was not exact (each 64 columns' rows checked before their sums are stored). */
__attribute__((target("avx512f"))) static int q8_matvec_block512_xs(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,
    double *output) {
  unsigned shift = 0u;
  size_t j0, i;
  int exact = 1;
  while (((size_t)1 << shift) < group_size) ++shift;
  for (j0 = 0u; j0 < output_count; j0 += 64u) {
    size_t vectors = output_count - j0 >= 64u ? 8u : (output_count - j0) / 8u;
    __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (i = 0u; i < input_count; ++i) {
      size_t e0 = i * output_count + j0, group = e0 >> shift;
      const uint8_t *c = codes + e0;
      double x = input[i], xs = x * q8_scale(exponents_le, group);
      __m512d vxs = _mm512_set1_pd(xs);
      exact &= q8_xs_exact(x, xs);
#define Q8_HOIST_VECTOR(a, v)                                                                            \
  if (vectors > (v)) {                                                                                   \
    if (((e0 + 8u * (v)) >> shift) != group) {                                                          \
      group = (e0 + 8u * (v)) >> shift;                                                                  \
      xs = x * q8_scale(exponents_le, group);                                                            \
      exact &= q8_xs_exact(x, xs);                                                                       \
      vxs = _mm512_set1_pd(xs);                                                                          \
    }                                                                                                    \
    a = _mm512_add_pd(a, q8_terms512_xs(c + 8u * (v), vxs));                                             \
  }
      Q8_HOIST_VECTOR(a0, 0u); Q8_HOIST_VECTOR(a1, 1u); Q8_HOIST_VECTOR(a2, 2u); Q8_HOIST_VECTOR(a3, 3u);
      Q8_HOIST_VECTOR(a4, 4u); Q8_HOIST_VECTOR(a5, 5u); Q8_HOIST_VECTOR(a6, 6u); Q8_HOIST_VECTOR(a7, 7u);
#undef Q8_HOIST_VECTOR
    }
    if (!exact) return 0;
    if (vectors > 0u) _mm512_storeu_pd(output + j0, a0);
    if (vectors > 1u) _mm512_storeu_pd(output + j0 + 8u, a1);
    if (vectors > 2u) _mm512_storeu_pd(output + j0 + 16u, a2);
    if (vectors > 3u) _mm512_storeu_pd(output + j0 + 24u, a3);
    if (vectors > 4u) _mm512_storeu_pd(output + j0 + 32u, a4);
    if (vectors > 5u) _mm512_storeu_pd(output + j0 + 40u, a5);
    if (vectors > 6u) _mm512_storeu_pd(output + j0 + 48u, a6);
    if (vectors > 7u) _mm512_storeu_pd(output + j0 + 56u, a7);
  }
  return 1;
}

/* hoisted, two input rows a step: each accumulator adds row i's term, then row i+1's, as before */
__attribute__((target("avx512f"))) static void hoist2_512(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,
    double *output) {
  unsigned shift = 0u;
  size_t j0, i;
  while (((size_t)1 << shift) < group_size) ++shift;
  for (j0 = 0u; j0 < output_count; j0 += 64u) {
    size_t vectors = output_count - j0 >= 64u ? 8u : (output_count - j0) / 8u;
    __m512d a[8];
    for (int v = 0; v < 8; v++) a[v] = _mm512_setzero_pd();
    for (i = 0u; i < input_count; ++i) {
      size_t e0 = i * output_count + j0;
      const uint8_t *c = codes + e0;
      double x = input[i];
      size_t group = e0 >> shift;
      __m512d vxs = _mm512_set1_pd(x * q8_scale(exponents_le, group));
      for (size_t v = 0; v < vectors; v++) {
        size_t gv = (e0 + 8u * v) >> shift;
        if (gv != group) { group = gv; vxs = _mm512_set1_pd(x * q8_scale(exponents_le, group)); }
        a[v] = _mm512_add_pd(a[v], q8_terms512_xs(c + 8u * v, vxs));
      }
    }
    for (size_t v = 0; v < vectors; v++) _mm512_storeu_pd(output + j0 + 8u * v, a[v]);
  }
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double unit(void) { return (double)(next() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0; }

static void fill(uint8_t *codes, size_t n, uint8_t *exps, size_t groups, int emin, int emax) {
  for (size_t k = 0; k < n; k++) {
    int c = (int)(next() % 255u) - 127;  /* never -128 (0x80, refused at load) */
    codes[k] = (uint8_t)(int8_t)c;
  }
  for (size_t g = 0; g < groups; g++) {
    int e = emin + (int)(next() % (uint64_t)(emax - emin + 1));
    uint16_t raw = (uint16_t)(int16_t)e;
    exps[2 * g] = (uint8_t)raw; exps[2 * g + 1] = (uint8_t)(raw >> 8);
  }
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
  int rounds = argc > 1 ? atoi(argv[1]) : 15;
  if (argc > 2 || rounds < 3 || rounds > 64) {
    fprintf(stderr, "usage: hoist_bench ROUNDS (3-64)\n");
    return 2;
  }
  /* 1. byte equality: the ceiling and the loop form where the guard holds, the prototype on every case */
  size_t checks = 0, cases = 0, proto_outputs = 0, proto_hoisted = 0, proto_again = 0;
  static const size_t shapes[][3] = {{128,128,32},{128,64,32},{128,344,32},{344,128,32},{17,8,8},{5,72,16},{33,200,64},{64,1000,8},{1,8,8},{9,16,128}};
  for (int range = 0; range < 8; range++) {
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
      size_t in = shapes[s][0], out = shapes[s][1], g = shapes[s][2], n = in * out, groups = (n + g - 1) / g;
      uint8_t *codes = malloc(n), *exps = malloc(2 * groups);
      double *x = malloc(in * sizeof(double)), *y0 = malloc(out * sizeof(double)), *y1 = malloc(out * sizeof(double));
      int emin = range == 4 || range == 7 ? -300 : range == 6 ? 0 : -12;
      int emax = range == 4 || range == 7 ? 300 : range == 6 ? 40 : 2;
      fill(codes, n, exps, groups, emin, emax);
      for (size_t i = 0; i < in; i++) {
        double v = unit();
        if (range == 1) v *= 1e6;
        if (range == 2) v = ldexp(v, (int)(next() % 1300) - 650);   /* 2^-651 .. 2^649 */
        if (range == 3) v = (next() % 4 == 0) ? (next() % 2 ? 0.0 : -0.0) : v;
        if (range == 4) v = ldexp(v, (int)(next() % 800) - 400);    /* with e in [-300, 300] */
        if (range == 5) v = (next() % 3 == 0) ? 0.0 : v;
        if (range == 6) v = ldexp(v, -723 - (int)(next() % 250)); /* below the guard, every x*s still normal */
        if (range == 7) v = ldexp(v, (int)(next() % 2098) - 1074); /* to 2^1023 and subnormal: some x*s inexact */
        x[i] = v;
      }
      /* the guard: every nonzero |x| in [2^-722, 2^723]: x*s normal and finite for every e in [-300, 300] */
      int guarded = 1;
      for (size_t i = 0; i < in; i++)
        if (x[i] != 0.0 && !(fabs(x[i]) >= 0x1p-722 && fabs(x[i]) < 0x1p723)) guarded = 0;
      ref512(x, in, codes, exps, out, g, y0);
      hoist512(x, in, codes, exps, out, g, y1);
      if (guarded) { double *y2 = malloc(out * sizeof(double)); hoist2_512(x, in, codes, exps, out, g, y2); if (memcmp(y1, y2, out * sizeof(double))) { printf("loop form differs\n"); return 1; } free(y2); }
      {
        double *y3 = malloc(out * sizeof(double));
        memset(y3, 0x5A, out * sizeof(double));
        int took = q8_matvec_block512_xs(x, in, codes, exps, out, g, y3);
        if (!took) ref512(x, in, codes, exps, out, g, y3); /* the prototype's fallback */
        if (guarded && !took) { printf("the prototype fell back where every x*s is exact\n"); return 1; }
        if (memcmp(y0, y3, out * sizeof(double)) != 0) {
          printf("PROTOTYPE DIFFERS range %d shape %zu,%zu,%zu\n", range, in, out, g);
          return 1;
        }
        proto_outputs += out;
        proto_hoisted += took != 0;
        proto_again += took == 0;
        free(y3);
      }
      cases++;
      if (guarded) {
        checks += out;
        if (memcmp(y0, y1, out * sizeof(double)) != 0) {
          for (size_t j = 0; j < out; j++) if (memcmp(&y0[j], &y1[j], 8)) { printf("DIFFERS range %d shape %zu,%zu,%zu at %zu: %a %a\n", range, in, out, g, j, y0[j], y1[j]); break; }
          return 1;
        }
      }
      free(codes); free(exps); free(x); free(y0); free(y1);
    }
  }
  if (!proto_hoisted || !proto_again) { printf("the prototype took one path on every case\n"); return 1; }
  printf("{\"equality\":{\"cases\":%zu,\"outputs_compared_where_every_x_s_is_exact\":%zu,\"prototype\":{\"outputs_compared\":%zu,\"calls_hoisted\":%zu,\"calls_computed_again\":%zu},\"every_byte_the_committed_kernels\":true}}\n", cases, checks, proto_outputs, proto_hoisted, proto_again);
  /* 2. time, interleaved, at the decoders' shapes */
  static const size_t tshapes[][3] = {{128,128,32},{128,64,32},{128,344,32},{344,128,32},{2048,2048,32},{2048,256,32},{2048,5632,32},{5632,2048,32}};
  for (size_t s = 0; s < sizeof tshapes / sizeof tshapes[0]; s++) {
    size_t in = tshapes[s][0], out = tshapes[s][1], g = tshapes[s][2], n = in * out, groups = (n + g - 1) / g;
    uint8_t *codes = malloc(n), *exps = malloc(2 * groups);
    double *x = malloc(in * sizeof(double)), *y = malloc(out * sizeof(double));
    fill(codes, n, exps, groups, -10, -4);
    for (size_t i = 0; i < in; i++) x[i] = unit();
    int reps = (int)(4000000 / n) + 1;
    double tr[64], th[64], t2[64], t3[64], ratio[64], ratio2[64], ratio3[64];
    for (int r = -1; r < rounds; r++) {
      for (int arm = 0; arm < 4; arm++) {
        int a = (arm + (r < 0 ? 0 : r)) % 4;
        double t0 = now();
        for (int k = 0; k < reps; k++) { if (a == 0) ref512(x, in, codes, exps, out, g, y); else if (a == 1) hoist512(x, in, codes, exps, out, g, y); else if (a == 2) hoist2_512(x, in, codes, exps, out, g, y); else if (!q8_matvec_block512_xs(x, in, codes, exps, out, g, y)) ref512(x, in, codes, exps, out, g, y); }
        double t = (now() - t0) / reps * 1e6;
        if (r >= 0) { if (a == 0) tr[r] = t; else if (a == 1) th[r] = t; else if (a == 2) t2[r] = t; else t3[r] = t; }
      }
      if (r >= 0) { ratio[r] = th[r] / tr[r]; ratio2[r] = t2[r] / tr[r]; ratio3[r] = t3[r] / tr[r]; }
    }
    qsort(tr, rounds, sizeof(double), cmpd); qsort(th, rounds, sizeof(double), cmpd); qsort(t2, rounds, sizeof(double), cmpd); qsort(ratio, rounds, sizeof(double), cmpd); qsort(ratio2, rounds, sizeof(double), cmpd); qsort(t3, rounds, sizeof(double), cmpd); qsort(ratio3, rounds, sizeof(double), cmpd);
    printf("{\"matvec\":[%zu,%zu,%zu],\"rounds\":%d,\"committed_us\":%.3f,\"hoisted\":{\"us\":%.3f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]},\"hoisted_loop_form\":{\"us\":%.3f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]},\"prototype\":{\"us\":%.3f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]}}\n", in, out, g, rounds, tr[rounds/2], th[rounds/2], ratio[rounds/2], ratio[0], ratio[rounds-1], t2[rounds/2], ratio2[rounds/2], ratio2[0], ratio2[rounds-1], t3[rounds/2], ratio3[rounds/2], ratio3[0], ratio3[rounds-1]);
    free(codes); free(exps); free(x); free(y);
  }
  /* 3. the 860K decoder's matvecs in its order, each matrix once a pass -- four layers of q, k, v, o, gate, up and
   * down, 0.86 MB of codes, so each matrix's codes have left L1 by its next use and come from L2, as in a decode */
  {
    static const size_t layer[7][2] = {{128, 128}, {128, 64}, {128, 64}, {128, 128}, {128, 344}, {128, 344}, {344, 128}};
    enum { MATS = 4 * 7 };
    uint8_t *codes[MATS], *exps[MATS];
    double *x[MATS], *y = malloc(344 * sizeof(double));
    size_t ins[MATS], outs[MATS], bytes = 0;
    for (int m = 0; m < MATS; m++) {
      size_t n, groups;
      ins[m] = layer[m % 7][0];
      outs[m] = layer[m % 7][1];
      n = ins[m] * outs[m];
      groups = (n + 31) / 32;
      codes[m] = malloc(n);
      exps[m] = malloc(2 * groups);
      x[m] = malloc(ins[m] * sizeof(double));
      fill(codes[m], n, exps[m], groups, -10, -4);
      for (size_t i = 0; i < ins[m]; i++) x[m][i] = unit();
      bytes += n;
    }
    int reps = 20;
    double tr[64], th[64], t2[64], t3[64], ratio[64], ratio2[64], ratio3[64];
    for (int r = -1; r < rounds; r++) {
      for (int arm = 0; arm < 4; arm++) {
        int a = (arm + (r < 0 ? 0 : r)) % 4;
        double t0 = now();
        for (int k = 0; k < reps; k++)
          for (int m = 0; m < MATS; m++) {
            if (a == 0) ref512(x[m], ins[m], codes[m], exps[m], outs[m], 32, y);
            else if (a == 1) hoist512(x[m], ins[m], codes[m], exps[m], outs[m], 32, y);
            else if (a == 2) hoist2_512(x[m], ins[m], codes[m], exps[m], outs[m], 32, y);
            else if (!q8_matvec_block512_xs(x[m], ins[m], codes[m], exps[m], outs[m], 32, y))
              ref512(x[m], ins[m], codes[m], exps[m], outs[m], 32, y);
          }
        double t = (now() - t0) / reps * 1e6;
        if (r >= 0) { if (a == 0) tr[r] = t; else if (a == 1) th[r] = t; else if (a == 2) t2[r] = t; else t3[r] = t; }
      }
      if (r >= 0) { ratio[r] = th[r] / tr[r]; ratio2[r] = t2[r] / tr[r]; ratio3[r] = t3[r] / tr[r]; }
    }
    qsort(tr, rounds, sizeof(double), cmpd); qsort(th, rounds, sizeof(double), cmpd); qsort(t2, rounds, sizeof(double), cmpd); qsort(t3, rounds, sizeof(double), cmpd);
    qsort(ratio, rounds, sizeof(double), cmpd); qsort(ratio2, rounds, sizeof(double), cmpd); qsort(ratio3, rounds, sizeof(double), cmpd);
    printf("{\"sequence\":\"the 860K decoder's matvecs, each once a pass\",\"matvecs\":%d,\"code_bytes\":%zu,\"rounds\":%d,\"committed_us\":%.3f,\"hoisted\":{\"us\":%.3f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]},\"hoisted_loop_form\":{\"us\":%.3f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]},\"prototype\":{\"us\":%.3f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]}}\n",
           (int)MATS, bytes, rounds, tr[rounds/2], th[rounds/2], ratio[rounds/2], ratio[0], ratio[rounds-1], t2[rounds/2], ratio2[rounds/2], ratio2[0], ratio2[rounds-1], t3[rounds/2], ratio3[rounds/2], ratio3[0], ratio3[rounds-1]);
    for (int m = 0; m < MATS; m++) { free(codes[m]); free(exps[m]); free(x[m]); }
    free(y);
  }
  return 0;
}
