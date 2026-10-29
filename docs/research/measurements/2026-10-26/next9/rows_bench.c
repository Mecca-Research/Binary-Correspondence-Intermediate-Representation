/* NEXT-9's head bench: the committed AVX-512 row-dot block (8 rows, a row's chain a lane, one accumulator;
 * copied byte for byte with its helpers) against two blocks interleaved (16 rows, two accumulators: two
 * independent chains a coordinate step), every row's chain the same terms in the same order. Byte equality over
 * shapes and value ranges, then the two timed in rotated order at the heads' shapes. One JSON line a result.
 * usage: rows_bench ROUNDS (3-64) */
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
/* 8 rows x 8 coordinates of codes, the rows `ld` apart, transposed: four registers of two coordinates'
 * eight codes each, coordinate 2q in the low 8 bytes of out[q] and 2q + 1 in the high */
__attribute__((target("avx2"))) static inline void q8_transpose8(const uint8_t *src, size_t ld,
                                                                 __m128i out[4]) {
  __m128i r[8], a[4], b[4];
  int q;
  for (q = 0; q < 8; ++q) {
    int64_t w;
    memcpy(&w, src + (size_t)q * ld, sizeof w);
    r[q] = _mm_cvtsi64_si128(w);
  }
  for (q = 0; q < 4; ++q) a[q] = _mm_unpacklo_epi8(r[2 * q], r[2 * q + 1]);
  b[0] = _mm_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm_unpackhi_epi16(a[0], a[1]);
  b[2] = _mm_unpacklo_epi16(a[2], a[3]);
  b[3] = _mm_unpackhi_epi16(a[2], a[3]);
  out[0] = _mm_unpacklo_epi32(b[0], b[2]);
  out[1] = _mm_unpackhi_epi32(b[0], b[2]);
  out[2] = _mm_unpacklo_epi32(b[1], b[3]);
  out[3] = _mm_unpackhi_epi32(b[1], b[3]);
}
/* the coordinates from c on that every row of the block keeps in its present group (one at least) */
static size_t q8_block_run(size_t input_count, size_t group_size, size_t row0, size_t rows, size_t c) {
  size_t n = input_count - c, r;
  for (r = 0u; r < rows; ++r) {
    size_t index = (row0 + r) * input_count + c, left = group_size - index % group_size;
    if (left < n) n = left;
  }
  return n;
}
/* row r of the block's scale at coordinate c (+0.0 for a short block's missing rows) */
static double q8_block_scale(const uint8_t *exponents_le, size_t input_count, size_t group_size, size_t row0,
                             size_t rows, size_t r, size_t c) {
  return r < rows ? q8_scale(exponents_le, ((row0 + r) * input_count + c) / group_size) : 0.0;
}
/* the codes of coordinate c of the block's rows (a short block's missing rows +0), as 8 bytes */
static int64_t q8_block_column(const uint8_t *codes, size_t input_count, size_t row0, size_t rows,
                               size_t c) {
  uint8_t bytes[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  int64_t w;
  size_t r;
  for (r = 0u; r < rows; ++r) bytes[r] = codes[(row0 + r) * input_count + c];
  memcpy(&w, bytes, sizeof w);
  return w;
}
/* The same block, eight lanes a register. */
__attribute__((target("avx512f,avx2"))) static void q8_rows_avx512(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t group_size, size_t row0, size_t rows,
    double *output) {
  __m512d t = _mm512_setzero_pd();
  size_t c = 0u, k;
  while (c < input_count) {
    size_t n = q8_block_run(input_count, group_size, row0, rows, c);
    __m512d vs = _mm512_setr_pd(q8_block_scale(exponents_le, input_count, group_size, row0, rows, 0u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 1u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 2u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 3u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 4u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 5u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 6u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 7u, c));
    for (k = 0u; rows == 8u && k + 8u <= n; k += 8u) {
      __m128i col[4];
      int q;
      q8_transpose8(codes + row0 * input_count + c + k, input_count, col);
      for (q = 0; q < 4; ++q) {
        __m512d c0 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(col[q]));
        __m512d c1 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_srli_si128(col[q], 8)));
        __m512d xa = _mm512_set1_pd(input[c + k + 2u * (size_t)q]);
        __m512d xb = _mm512_set1_pd(input[c + k + 2u * (size_t)q + 1u]);
        t = _mm512_add_pd(t, _mm512_mul_pd(xa, _mm512_mul_pd(c0, vs)));
        t = _mm512_add_pd(t, _mm512_mul_pd(xb, _mm512_mul_pd(c1, vs)));
      }
    }
    for (; k < n; ++k) {
      __m128i w = _mm_cvtsi64_si128(q8_block_column(codes, input_count, row0, rows, c + k));
      t = _mm512_add_pd(t, _mm512_mul_pd(_mm512_set1_pd(input[c + k]),
                                         _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(w)), vs)));
    }
    c += n;
  }
  _mm512_mask_storeu_pd(output, (__mmask8)((1u << rows) - 1u), t);
}

/* Two blocks of eight rows, rows row0 .. row0+15, their chains side by side: tA rows 0-7, tB rows 8-15. */
__attribute__((target("avx512f,avx2"))) static void q8_rows2_avx512(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t group_size, size_t row0, double *output) {
  __m512d tA = _mm512_setzero_pd(), tB = _mm512_setzero_pd();
  size_t c = 0u, k;
  while (c < input_count) {
    size_t n = q8_block_run(input_count, group_size, row0, 16u, c);
    __m512d sA = _mm512_setr_pd(q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 0u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 1u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 2u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 3u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 4u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 5u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 6u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, 8u, 7u, c));
    __m512d sB = _mm512_setr_pd(q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 0u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 1u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 2u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 3u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 4u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 5u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 6u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0 + 8u, 8u, 7u, c));
    for (k = 0u; k + 8u <= n; k += 8u) {
      __m128i colA[4], colB[4];
      int q;
      q8_transpose8(codes + row0 * input_count + c + k, input_count, colA);
      q8_transpose8(codes + (row0 + 8u) * input_count + c + k, input_count, colB);
      for (q = 0; q < 4; ++q) {
        __m512d a0 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(colA[q]));
        __m512d a1 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_srli_si128(colA[q], 8)));
        __m512d b0 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(colB[q]));
        __m512d b1 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_srli_si128(colB[q], 8)));
        __m512d xa = _mm512_set1_pd(input[c + k + 2u * (size_t)q]);
        __m512d xb = _mm512_set1_pd(input[c + k + 2u * (size_t)q + 1u]);
        tA = _mm512_add_pd(tA, _mm512_mul_pd(xa, _mm512_mul_pd(a0, sA)));
        tB = _mm512_add_pd(tB, _mm512_mul_pd(xa, _mm512_mul_pd(b0, sB)));
        tA = _mm512_add_pd(tA, _mm512_mul_pd(xb, _mm512_mul_pd(a1, sA)));
        tB = _mm512_add_pd(tB, _mm512_mul_pd(xb, _mm512_mul_pd(b1, sB)));
      }
    }
    for (; k < n; ++k) {
      __m128i wA = _mm_cvtsi64_si128(q8_block_column(codes, input_count, row0, 8u, c + k));
      __m128i wB = _mm_cvtsi64_si128(q8_block_column(codes, input_count, row0 + 8u, 8u, c + k));
      __m512d xv = _mm512_set1_pd(input[c + k]);
      tA = _mm512_add_pd(tA, _mm512_mul_pd(xv, _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(wA)), sA)));
      tB = _mm512_add_pd(tB, _mm512_mul_pd(xv, _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(wB)), sB)));
    }
    c += n;
  }
  _mm512_storeu_pd(output, tA);
  _mm512_storeu_pd(output + 8, tB);
}
static void ref_rows(const double *x, size_t in, const uint8_t *codes, const uint8_t *exps, size_t out, size_t g, double *y) {
  for (size_t row = 0; row < out; row += 8) {
    size_t rows = out - row < 8 ? out - row : 8;
    q8_rows_avx512(x, in, codes, exps, g, row, rows, y + row);
  }
}
static void two_rows(const double *x, size_t in, const uint8_t *codes, const uint8_t *exps, size_t out, size_t g, double *y) {
  size_t row = 0;
  for (; out - row >= 16; row += 16) q8_rows2_avx512(x, in, codes, exps, g, row, y + row);
  for (; row < out; row += 8) {
    size_t rows = out - row < 8 ? out - row : 8;
    q8_rows_avx512(x, in, codes, exps, g, row, rows, y + row);
  }
}
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double unit(void) { return (double)(next() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0; }
static void fill(uint8_t *codes, size_t n, uint8_t *exps, size_t groups, int emin, int emax) {
  for (size_t k = 0; k < n; k++) codes[k] = (uint8_t)(int8_t)((int)(next() % 255u) - 127);
  for (size_t gi = 0; gi < groups; gi++) {
    int e = emin + (int)(next() % (uint64_t)(emax - emin + 1));
    uint16_t raw = (uint16_t)(int16_t)e;
    exps[2 * gi] = (uint8_t)raw; exps[2 * gi + 1] = (uint8_t)(raw >> 8);
  }
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
  int rounds = argc > 1 ? atoi(argv[1]) : 15;
  if (argc > 2 || rounds < 3 || rounds > 64) {
    fprintf(stderr, "usage: rows_bench ROUNDS (3-64)\n");
    return 2;
  }
  static const size_t shapes[][3] = {{128,1024,32},{128,1000,32},{37,48,8},{64,24,16},{9,33,64},{200,17,32},{256,40,128},{8,16,8}};
  size_t cases = 0, outputs = 0;
  for (int range = 0; range < 4; range++)
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
      size_t in = shapes[s][0], out = shapes[s][1], g = shapes[s][2], n = in * out, groups = (n + g - 1) / g;
      uint8_t *codes = malloc(n), *exps = malloc(2 * groups);
      double *x = malloc(in * 8), *y0 = malloc(out * 8), *y1 = malloc(out * 8);
      fill(codes, n, exps, groups, range == 3 ? -300 : -12, range == 3 ? 300 : 2);
      for (size_t i = 0; i < in; i++) {
        double v = unit();
        if (range == 1) v = ldexp(v, (int)(next() % 2000) - 1000);
        if (range == 2) v = (next() % 3 == 0) ? (next() % 2 ? 0.0 : -0.0) : v;
        if (range == 3) v = ldexp(v, (int)(next() % 800) - 400);
        x[i] = v;
      }
      ref_rows(x, in, codes, exps, out, g, y0);
      two_rows(x, in, codes, exps, out, g, y1);
      cases++; outputs += out;
      if (memcmp(y0, y1, out * 8)) { printf("DIFFERS range %d shape %zu,%zu,%zu\n", range, in, out, g); return 1; }
      free(codes); free(exps); free(x); free(y0); free(y1);
    }
  printf("{\"equality\":{\"cases\":%zu,\"outputs\":%zu,\"every_byte_the_committed_kernels\":true}}\n", cases, outputs);
  static const size_t tshapes[][3] = {{128,1024,32},{2048,4096,32},{2048,32000,32}};
  for (size_t s = 0; s < sizeof tshapes / sizeof tshapes[0]; s++) {
    size_t in = tshapes[s][0], out = tshapes[s][1], g = tshapes[s][2], n = in * out, groups = (n + g - 1) / g;
    uint8_t *codes = malloc(n), *exps = malloc(2 * groups);
    double *x = malloc(in * 8), *y = malloc(out * 8);
    fill(codes, n, exps, groups, -10, -4);
    for (size_t i = 0; i < in; i++) x[i] = unit();
    int reps = (int)(4000000 / n) + 1;
    double tr[64], th[64], ratio[64];
    for (int r = -1; r < rounds; r++) {
      for (int arm = 0; arm < 2; arm++) {
        int a = (arm + (r < 0 ? 0 : r)) % 2;
        double t0 = now();
        for (int k = 0; k < reps; k++) { if (a == 0) ref_rows(x, in, codes, exps, out, g, y); else two_rows(x, in, codes, exps, out, g, y); }
        double t = (now() - t0) / reps * 1e6;
        if (r >= 0) { if (a == 0) tr[r] = t; else th[r] = t; }
      }
      if (r >= 0) ratio[r] = th[r] / tr[r];
    }
    qsort(tr, rounds, 8, cmpd); qsort(th, rounds, 8, cmpd); qsort(ratio, rounds, 8, cmpd);
    printf("{\"rows_dot\":[%zu,%zu,%zu],\"rounds\":%d,\"committed_us\":%.2f,\"two_blocks\":{\"us\":%.2f,\"ratio_median\":%.4f,\"range\":[%.4f,%.4f]}}\n", in, out, g, rounds, tr[rounds/2], th[rounds/2], ratio[rounds/2], ratio[0], ratio[rounds-1]);
    free(codes); free(exps); free(x); free(y);
  }
  return 0;
}
