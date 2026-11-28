#include "bcir_ai_kernels.h"

#include <math.h>
#include <string.h>

#define BCIR_AI_EXP_MIN (-300)
#define BCIR_AI_EXP_MAX 300

/* Q8-LANES: the BCIRQ8 matrix kernels' instances -- 1 the portable loops, 2 AVX2, 4 AVX-512 -- each the
 * portable loop's value for every output, bit for bit. An output's chain is the loop's own in every
 * instance: from +0.0, one term an input coordinate in increasing order, the term the input times the
 * code times its group's scale (a code times a power of two: exact), rounded as a product and then as a
 * sum, never fused -- every build of this unit passes -ffp-contract=off, and the vector instances spell
 * the multiplies and the add as separate operations. The vector instances only run several outputs'
 * chains side by side: the matvec a row's columns in the lanes, the row dots eight rows a vector over
 * their codes transposed eight by eight. A group's scale is ldexp(1.0, e): built from its bits where 2^e
 * is a normal double, ldexp's own elsewhere.
 *
 * Q8-BOUND: where a call's ranges prove every product x * s exact -- every exponent of its groups in [-300, 300], the
 * range bcir_ai_q8_values_validate admits, and every nonzero |x| in [2^-722, 2^724), so that x * s is x scaled by
 * a power of two from 2^-300 to 2^300 with neither overflow nor a subnormal result -- the vector matvecs form x * s
 * once a row and group and each term as the code times it. x * (code * s) and code * (x * s) are then the one product
 * of the same three factors rounded once (code * s is exact: a code at most eight bits, s a power of two), the
 * loop's bits. The ranges are read once a call, in vectors; a call outside them takes the terms as before.
 *
 * Q8-PAIR: where eight columns' terms lie in one group -- the columns a multiple of 8, the group a power of two of at
 * least 8 -- the vector matvecs take two input rows a pass: eight outputs are loaded (one AVX-512 vector, two AVX2
 * ones), row i's terms and then row i + 1's are added to them, and they are stored. Every output's chain keeps its
 * rows in increasing order, the portable loop's, and each row's codes are read in order, so the caches' prefetchers
 * follow a pass from L3 or memory, where the block it replaces walked a strip of 64 columns down every row. Elsewhere
 * the runs, a row at a time.
 *
 * HEAD-PACK: the row dots over a copy of their codes and exponents packed once (bcir_ai_q8_rows_pack, the layout in
 * bcir_ai_kernels.h): rows in blocks of eight, a block's groups one after another, each group's segment its eight
 * rows' exponents and then its coordinates' codes, eight bytes a coordinate. A block is then one stream read in the
 * order its terms are taken: a coordinate's eight codes one load, a group's eight scales one load and three integer
 * operations (2^e is (e + 1023) << 52 for every exponent a pack admits, |e| <= 300: q8_scale's own value), and no
 * division a group. The vector instances take four blocks a pass, 32 rows in four streams, then a block at a time.
 * Every row's chain is its own lane's, over the coordinates in order, each term x * (code * s): every output the
 * portable loop's.
 *
 * DECODE-THREADS: every matvec kernel runs over a range of columns [k0, k1), the whole call being [0, output_count),
 * so that a caller may compute a matvec's columns in parts on threads of its own (bcir_ai_q8_matvec_f64_columns).
 * An output's chain does not depend on the range it is computed in: its terms are its column's, every input row in
 * order. A range's call reads Q8-BOUND's ranges over the groups its columns touch, row by row, and takes Q8-PAIR's
 * two rows a pass where it starts and ends on a vector's edge; either path gives every output the loop's bits. Each
 * vector kernel has two instances, the whole call's -- its range the constant [0, output_count) -- and a range's, and
 * a call takes the whole call's where its range is the whole: Q8-PAIR's are one text in two, so the whole call's is
 * the kernel before the ranges instruction for instruction (clang 18 and GCC 13). */
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(BCIR_AI_NO_DISPATCH)
#define BCIR_AI_DISPATCH 1
#include <immintrin.h>
#endif
#ifndef BCIR_AI_MAX_INSTANCE
#define BCIR_AI_MAX_INSTANCE 4
#endif

uint32_t bcir_ai_abi_version(void) { return BCIR_AI_ABI_VERSION; }

static int valid_quantize_arguments(const double *values, size_t count,
                                    size_t group_size, const int16_t *exponents,
                                    size_t exponent_capacity, size_t code_capacity) {
  size_t groups;
  if (group_size == 0u || group_size > UINT16_MAX ||
      count > (size_t)BCIR_AI_MAX_ELEMENTS) return 0;
  groups = count == 0u ? 0u : 1u + (count - 1u) / group_size;
  if ((count != 0u && values == NULL) ||
      (groups != 0u && exponents == NULL) ||
      exponent_capacity < groups || code_capacity < count) return 0;
  return 1;
}

static int finite_values(const double *values, size_t count) {
  size_t index;
  for (index = 0u; index < count; ++index)
    if (!isfinite(values[index])) return 0;
  return 1;
}

static int group_exponent(const double *values, size_t count, int code_max) {
  size_t index;
  double maximum = 0.0;
  int binary_exponent, code_power = 0, exponent;
  for (index = 0u; index < count; ++index) {
    double magnitude = fabs(values[index]);
    if (magnitude > maximum) maximum = magnitude;
  }
  if (maximum == 0.0) return 0;
  (void)frexp(maximum, &binary_exponent);
  while ((1 << (code_power + 1)) <= code_max) ++code_power;
  exponent = binary_exponent - 1 - code_power;
  if (exponent < BCIR_AI_EXP_MIN) exponent = BCIR_AI_EXP_MIN;
  if (exponent > BCIR_AI_EXP_MAX) exponent = BCIR_AI_EXP_MAX;
  while (exponent < BCIR_AI_EXP_MAX &&
         (double)code_max * ldexp(1.0, exponent) < maximum) ++exponent;
  while (exponent > BCIR_AI_EXP_MIN &&
         (double)code_max * ldexp(1.0, exponent - 1) >= maximum) --exponent;
  return exponent;
}

static int quantize_code(double value, int code_max, int exponent) {
  double scaled = value / ldexp(1.0, exponent);
  double rounded;
  if (scaled >= (double)code_max) return code_max;
  if (scaled <= (double)-code_max) return -code_max;
  rounded = scaled >= 0.0 ? floor(scaled + 0.5) : -floor(-scaled + 0.5);
  if (rounded > (double)code_max) return code_max;
  if (rounded < (double)-code_max) return -code_max;
  return (int)rounded;
}

int bcir_ai_quantize_q8_f64(const double *values, size_t element_count,
                            size_t group_size, int16_t *exponents,
                            size_t exponent_capacity, int8_t *codes,
                            size_t code_capacity) {
  size_t group, groups;
  if (!valid_quantize_arguments(values, element_count, group_size, exponents,
                                exponent_capacity, code_capacity) ||
      (element_count != 0u && codes == NULL)) return BCIR_AI_INVALID_ARGUMENT;
  if (!finite_values(values, element_count)) return BCIR_AI_INVALID_NUMBER;
  groups = element_count == 0u ? 0u : 1u + (element_count - 1u) / group_size;
  for (group = 0u; group < groups; ++group) {
    size_t begin = group * group_size;
    size_t end = begin + group_size;
    size_t index;
    int exponent;
    if (end > element_count) end = element_count;
    exponent = group_exponent(values + begin, end - begin, 127);
    exponents[group] = (int16_t)exponent;
    for (index = begin; index < end; ++index)
      codes[index] = (int8_t)quantize_code(values[index], 127, exponent);
  }
  return BCIR_AI_OK;
}

int bcir_ai_quantize_q4_f64(const double *values, size_t element_count,
                            size_t group_size, int16_t *exponents,
                            size_t exponent_capacity, uint8_t *packed_codes,
                            size_t packed_capacity) {
  size_t group, groups, packed_bytes;
  if (element_count > SIZE_MAX - 1u) return BCIR_AI_INVALID_ARGUMENT;
  packed_bytes = (element_count + 1u) / 2u;
  if (!valid_quantize_arguments(values, element_count, group_size, exponents,
                                exponent_capacity, element_count) ||
      packed_capacity < packed_bytes ||
      (packed_bytes != 0u && packed_codes == NULL)) return BCIR_AI_INVALID_ARGUMENT;
  if (!finite_values(values, element_count)) return BCIR_AI_INVALID_NUMBER;
  if (packed_bytes != 0u) memset(packed_codes, 0, packed_bytes);
  groups = element_count == 0u ? 0u : 1u + (element_count - 1u) / group_size;
  for (group = 0u; group < groups; ++group) {
    size_t begin = group * group_size;
    size_t end = begin + group_size;
    size_t index;
    int exponent;
    if (end > element_count) end = element_count;
    exponent = group_exponent(values + begin, end - begin, 7);
    exponents[group] = (int16_t)exponent;
    for (index = begin; index < end; ++index) {
      uint8_t nibble = (uint8_t)(quantize_code(values[index], 7, exponent) & 0x0f);
      if ((index & 1u) != 0u)
        packed_codes[index / 2u] = (uint8_t)(packed_codes[index / 2u] | (uint8_t)(nibble << 4));
      else
        packed_codes[index / 2u] = nibble;
    }
  }
  return BCIR_AI_OK;
}

static int exponent_le(const uint8_t *bytes, size_t group) {
  const uint8_t *value = bytes + 2u * group;
  uint16_t raw = (uint16_t)value[0] | (uint16_t)((uint16_t)value[1] << 8);
  return raw >= UINT16_C(0x8000) ? (int)raw - 65536 : (int)raw;
}

static int q8_shape(size_t input_count, size_t output_count, size_t group_size,
                    size_t *elements, size_t *groups) {
  if (input_count == 0u || output_count == 0u || group_size == 0u ||
      group_size > UINT16_MAX || input_count > SIZE_MAX / output_count)
    return 0;
  *elements = input_count * output_count;
  if (*elements > (size_t)BCIR_AI_MAX_ELEMENTS) return 0;
  *groups = 1u + (*elements - 1u) / group_size;
  return 1;
}

int bcir_ai_q8_values_validate(const uint8_t *codes, size_t code_count,
                               const uint8_t *exponents_le,
                               size_t exponent_count, size_t element_count,
                               size_t group_size) {
  size_t groups, index;
  if (element_count == 0u || element_count > (size_t)BCIR_AI_MAX_ELEMENTS ||
      group_size == 0u || group_size > UINT16_MAX) return BCIR_AI_INVALID_ARGUMENT;
  groups = 1u + (element_count - 1u) / group_size;
  if (codes == NULL || exponents_le == NULL || code_count < element_count ||
      exponent_count < groups) return BCIR_AI_INVALID_ARGUMENT;
  for (index = 0u; index < element_count; ++index)
    if (codes[index] == UINT8_C(0x80)) return BCIR_AI_INVALID_NUMBER;
  for (index = 0u; index < groups; ++index) {
    int exponent = exponent_le(exponents_le, index);
    if (exponent < BCIR_AI_EXP_MIN || exponent > BCIR_AI_EXP_MAX)
      return BCIR_AI_INVALID_NUMBER;
  }
  return BCIR_AI_OK;
}

static int q8_input_validate(const double *input, size_t input_count) {
  size_t index;
  if (input == NULL) return 0;
  for (index = 0u; index < input_count; ++index)
    if (!isfinite(input[index])) return 0;
  return 1;
}

int bcir_ai_q8_instances(void) {
  int instances = 1;
#ifdef BCIR_AI_DISPATCH
  __builtin_cpu_init();
  if (BCIR_AI_MAX_INSTANCE >= 2 && __builtin_cpu_supports("avx2")) instances |= 2;
  if (BCIR_AI_MAX_INSTANCE >= 4 && __builtin_cpu_supports("avx512f")) instances |= 4;
#endif
  return instances;
}

/* 0 names the widest instance the host admits; 1, 2 or 4 itself where admitted; anything else 0 (refused) */
static int q8_instance(int instance) {
  int admitted = bcir_ai_q8_instances();
  if (instance == 0) return admitted & 4 ? 4 : admitted & 2 ? 2 : 1;
  return (instance == 1 || instance == 2 || instance == 4) && (admitted & instance) ? instance : 0;
}

#ifdef BCIR_AI_INSTANCE_PROBE
/* The instance the last matrix kernel ran (0 before any, and after a read): the path its values cannot
 * show. A build defines BCIR_AI_INSTANCE_PROBE to have the symbol (the harness's builds do). */
static int q8_instance_seen;
#define Q8_SAW(instance) (q8_instance_seen = (instance))
int bcir_ai_q8_instance_probe(void);
int bcir_ai_q8_instance_probe(void) {
  int seen = q8_instance_seen;
  q8_instance_seen = 0;
  return seen;
}
/* Q8-BOUND: the path the last vector matvec took -- 1 x * s once a row and group, 2 the terms as before (the
 * call's ranges not proved), 0 before any and after a read */
static int q8_bound_seen;
#define Q8_BOUND_SAW(path) (q8_bound_seen = (path))
int bcir_ai_q8_bound_probe(void);
int bcir_ai_q8_bound_probe(void) {
  int seen = q8_bound_seen;
  q8_bound_seen = 0;
  return seen;
}
/* Q8-PAIR: the kernel the last vector matvec ran -- 1 two rows a pass, 2 the runs (eight columns' terms not in one
 * group), 0 before any and after a read */
static int q8_pair_seen;
#define Q8_PAIR_SAW(path) (q8_pair_seen = (path))
int bcir_ai_q8_pair_probe(void);
int bcir_ai_q8_pair_probe(void) {
  int seen = q8_pair_seen;
  q8_pair_seen = 0;
  return seen;
}
/* DECODE-THREADS: the instance of its kernel the last vector matvec ran -- 1 the whole call's, its range the constant
 * [0, output_count), 2 a range's, 0 before any and after a read */
static int q8_columns_seen;
#define Q8_COLUMNS_SAW(path) (q8_columns_seen = (path))
int bcir_ai_q8_columns_probe(void);
int bcir_ai_q8_columns_probe(void) {
  int seen = q8_columns_seen;
  q8_columns_seen = 0;
  return seen;
}
/* HEAD-PACK: the last packed row-dot call -- 1 the vector kernels, 2 the portable loop over the pack, 0 before any
 * and after a read -- with the vector kernels' passes of 32 rows and blocks of at most 8 */
static int q8_pack_seen;
static size_t q8_pack_passes, q8_pack_blocks;
#define Q8_PACK_SAW(path, passes, blocks) \
  (q8_pack_seen = (path), q8_pack_passes = (passes), q8_pack_blocks = (blocks))
int bcir_ai_q8_pack_probe(size_t *passes, size_t *blocks);
int bcir_ai_q8_pack_probe(size_t *passes, size_t *blocks) {
  int seen = q8_pack_seen;
  if (passes != NULL) *passes = q8_pack_passes;
  if (blocks != NULL) *blocks = q8_pack_blocks;
  q8_pack_seen = 0;
  q8_pack_passes = q8_pack_blocks = 0u;
  return seen;
}
#else
#define Q8_SAW(instance) ((void)0)
#define Q8_BOUND_SAW(path) ((void)0)
#define Q8_PAIR_SAW(path) ((void)0)
#define Q8_COLUMNS_SAW(path) ((void)0)
#define Q8_PACK_SAW(path, passes, blocks) ((void)(passes), (void)(blocks))
#endif

#ifdef BCIR_AI_DISPATCH
/* ldexp(1.0, e) for group `group`'s exponent: its bits where 2^e is a normal double, ldexp elsewhere */
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

/* n columns of one group, one input row: y[k] += xi * (code[k] * s), a vector of columns at a time */
__attribute__((target("avx2"))) static void q8_run_avx2(double *y, double xi, const uint8_t *c, size_t n,
                                                        double s) {
  __m256d vx = _mm256_set1_pd(xi), vs = _mm256_set1_pd(s);
  size_t k = 0u;
  for (; k + 4u <= n; k += 4u) {
    int32_t w;
    __m256d code;
    memcpy(&w, c + k, sizeof w);
    code = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(w)));
    _mm256_storeu_pd(y + k,
                     _mm256_add_pd(_mm256_loadu_pd(y + k), _mm256_mul_pd(vx, _mm256_mul_pd(code, vs))));
  }
  for (; k < n; ++k) {
    int code = c[k] >= UINT8_C(0x80) ? (int)c[k] - 256 : (int)c[k];
    double term = xi * ((double)code * s);
    y[k] = y[k] + term;
  }
}
__attribute__((target("avx512f"))) static void q8_run_avx512(double *y, double xi, const uint8_t *c,
                                                             size_t n, double s) {
  __m512d vx = _mm512_set1_pd(xi), vs = _mm512_set1_pd(s);
  size_t k = 0u;
  for (; k + 8u <= n; k += 8u) {
    int64_t w;
    __m512d code;
    memcpy(&w, c + k, sizeof w);
    code = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w)));
    _mm512_storeu_pd(y + k,
                     _mm512_add_pd(_mm512_loadu_pd(y + k), _mm512_mul_pd(vx, _mm512_mul_pd(code, vs))));
  }
  for (; k < n; ++k) {
    int code = c[k] >= UINT8_C(0x80) ? (int)c[k] - 256 : (int)c[k];
    double term = xi * ((double)code * s);
    y[k] = y[k] + term;
  }
}
/* Q8-BOUND's runs: n columns of one group, one input row, xs that row's input times the group's scale (exact):
 * y[k] += code[k] * xs */
__attribute__((target("avx2"))) static void q8_run_avx2_xs(double *y, double xs, const uint8_t *c, size_t n) {
  __m256d vxs = _mm256_set1_pd(xs);
  size_t k = 0u;
  for (; k + 4u <= n; k += 4u) {
    int32_t w;
    __m256d code;
    memcpy(&w, c + k, sizeof w);
    code = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(w)));
    _mm256_storeu_pd(y + k, _mm256_add_pd(_mm256_loadu_pd(y + k), _mm256_mul_pd(code, vxs)));
  }
  for (; k < n; ++k) {
    int code = c[k] >= UINT8_C(0x80) ? (int)c[k] - 256 : (int)c[k];
    double term = (double)code * xs;
    y[k] = y[k] + term;
  }
}
__attribute__((target("avx512f"))) static void q8_run_avx512_xs(double *y, double xs, const uint8_t *c,
                                                                size_t n) {
  __m512d vxs = _mm512_set1_pd(xs);
  size_t k = 0u;
  for (; k + 8u <= n; k += 8u) {
    int64_t w;
    __m512d code;
    memcpy(&w, c + k, sizeof w);
    code = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w)));
    _mm512_storeu_pd(y + k, _mm512_add_pd(_mm512_loadu_pd(y + k), _mm512_mul_pd(code, vxs)));
  }
  for (; k < n; ++k) {
    int code = c[k] >= UINT8_C(0x80) ? (int)c[k] - 256 : (int)c[k];
    double term = (double)code * xs;
    y[k] = y[k] + term;
  }
}
/* The matvec a row at a time over columns [k0, k1) -- the whole call is [0, output_count) -- each row's columns in
 * runs of one group (any shape). Each run kernel is one body always inlined into two instances, the whole call's (its
 * range the constant [0, output_count)) and a range's, as Q8-PAIR's are one text in two: with the range a variable
 * in every call, the whole calls ran the range's code, under clang in up to x1.14 of their time (DECODE-THREADS). */
#define Q8_INLINE static inline __attribute__((always_inline))
Q8_INLINE void q8_matvec_runs_cols(int instance, const double *input, size_t input_count, const uint8_t *codes,
                                   const uint8_t *exponents_le, size_t output_count, size_t group_size, size_t k0,
                                   size_t k1, double *output) {
  size_t input_index, output_index, index;
  for (output_index = k0; output_index < k1; ++output_index) output[output_index] = 0.0;
  for (input_index = 0u; input_index < input_count; ++input_index) {
    index = input_index * output_count + k0;
    for (output_index = k0; output_index < k1;) {
      size_t group = index / group_size, left = (group + 1u) * group_size - index;
      size_t n = k1 - output_index < left ? k1 - output_index : left;
      double scale = q8_scale(exponents_le, group);
      if (instance == 4) q8_run_avx512(output + output_index, input[input_index], codes + index, n, scale);
      else q8_run_avx2(output + output_index, input[input_index], codes + index, n, scale);
      output_index += n;
      index += n;
    }
  }
}
/* Q8-BOUND's runs: the same runs with x * s formed once a run */
Q8_INLINE void q8_matvec_runs_xs_cols(int instance, const double *input, size_t input_count, const uint8_t *codes,
                                      const uint8_t *exponents_le, size_t output_count, size_t group_size, size_t k0,
                                      size_t k1, double *output) {
  size_t input_index, output_index, index;
  for (output_index = k0; output_index < k1; ++output_index) output[output_index] = 0.0;
  for (input_index = 0u; input_index < input_count; ++input_index) {
    double x = input[input_index];
    index = input_index * output_count + k0;
    for (output_index = k0; output_index < k1;) {
      size_t group = index / group_size, left = (group + 1u) * group_size - index;
      size_t n = k1 - output_index < left ? k1 - output_index : left;
      double xs = x * q8_scale(exponents_le, group);
      if (instance == 4) q8_run_avx512_xs(output + output_index, xs, codes + index, n);
      else q8_run_avx2_xs(output + output_index, xs, codes + index, n);
      output_index += n;
      index += n;
    }
  }
}
static void q8_matvec_runs(int instance, const double *input, size_t input_count, const uint8_t *codes,
                           const uint8_t *exponents_le, size_t output_count, size_t group_size, double *output) {
  q8_matvec_runs_cols(instance, input, input_count, codes, exponents_le, output_count, group_size, 0u, output_count,
                      output);
}
static void q8_matvec_runs_range(int instance, const double *input, size_t input_count, const uint8_t *codes,
                                 const uint8_t *exponents_le, size_t output_count, size_t group_size, size_t k0,
                                 size_t k1, double *output) {
  q8_matvec_runs_cols(instance, input, input_count, codes, exponents_le, output_count, group_size, k0, k1, output);
}
static void q8_matvec_runs_xs(int instance, const double *input, size_t input_count, const uint8_t *codes,
                              const uint8_t *exponents_le, size_t output_count, size_t group_size, double *output) {
  q8_matvec_runs_xs_cols(instance, input, input_count, codes, exponents_le, output_count, group_size, 0u,
                         output_count, output);
}
static void q8_matvec_runs_xs_range(int instance, const double *input, size_t input_count, const uint8_t *codes,
                                    const uint8_t *exponents_le, size_t output_count, size_t group_size, size_t k0,
                                    size_t k1, double *output) {
  q8_matvec_runs_xs_cols(instance, input, input_count, codes, exponents_le, output_count, group_size, k0, k1,
                         output);
}
/* one vector of eight columns' terms: x times each code times their group's scale s */
__attribute__((target("avx512f"))) static inline __m512d q8_terms512(const uint8_t *c, __m512d vx, double s) {
  int64_t w;
  memcpy(&w, c, sizeof w);
  return _mm512_mul_pd(vx, _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w))),
                                         _mm512_set1_pd(s)));
}
/* Q8-BOUND's terms: eight columns', xs the row's input times their group's scale -- each code times xs */
__attribute__((target("avx512f"))) static inline __m512d q8_terms512_xs(const uint8_t *c, __m512d vxs) {
  int64_t w;
  memcpy(&w, c, sizeof w);
  return _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(w))), vxs);
}
/* the same, four columns a vector */
__attribute__((target("avx2"))) static inline __m256d q8_terms256(const uint8_t *c, __m256d vx, double s) {
  int32_t w;
  memcpy(&w, c, sizeof w);
  return _mm256_mul_pd(vx, _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(w))),
                                         _mm256_set1_pd(s)));
}
__attribute__((target("avx2"))) static inline __m256d q8_terms256_xs(const uint8_t *c, __m256d vxs) {
  int32_t w;
  memcpy(&w, c, sizeof w);
  return _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(w))), vxs);
}
/* eight columns as two AVX2 vectors of four: the AVX2 pair's step is the AVX-512 pair's */
typedef struct q8_x8 {
  __m256d lo, hi;
} q8_x8;
__attribute__((target("avx2"))) static inline q8_x8 q8_x8_loadu(const double *p) {
  q8_x8 v;
  v.lo = _mm256_loadu_pd(p);
  v.hi = _mm256_loadu_pd(p + 4);
  return v;
}
__attribute__((target("avx2"))) static inline void q8_x8_storeu(double *p, q8_x8 v) {
  _mm256_storeu_pd(p, v.lo);
  _mm256_storeu_pd(p + 4, v.hi);
}
__attribute__((target("avx2"))) static inline q8_x8 q8_x8_add(q8_x8 a, q8_x8 b) {
  a.lo = _mm256_add_pd(a.lo, b.lo);
  a.hi = _mm256_add_pd(a.hi, b.hi);
  return a;
}
__attribute__((target("avx2"))) static inline q8_x8 q8_x8_set1(double x) {
  q8_x8 v;
  v.lo = _mm256_set1_pd(x);
  v.hi = v.lo;
  return v;
}
/* eight columns' terms: x * (code * s) as before, code * (x * s) where Q8-BOUND proves the call */
__attribute__((target("avx2"))) static inline q8_x8 q8_terms_x8(const uint8_t *c, q8_x8 vx, double s) {
  q8_x8 t;
  t.lo = q8_terms256(c, vx.lo, s);
  t.hi = q8_terms256(c + 4, vx.hi, s);
  return t;
}
__attribute__((target("avx2"))) static inline q8_x8 q8_terms_x8_xs(const uint8_t *c, q8_x8 vxs) {
  q8_x8 t;
  t.lo = q8_terms256_xs(c, vxs.lo);
  t.hi = q8_terms256_xs(c + 4, vxs.hi);
  return t;
}
/* Q8-PAIR's kernels: the matvec two input rows a pass where eight columns' terms lie in one group, over columns
 * [k0, k1), both multiples of eight (DECODE-THREADS; the whole call is [0, output_count)). Each row's scale is read
 * again at the step where its group changes; an odd last row runs alone. A row's state: its codes, its first
 * element, the column where its group ends (from k0 on), its input, its group's scale and the multiplier its terms
 * take (x * s where Q8-BOUND proves the call, x as before elsewhere). */
#define Q8_PAIR_ROW(r, row, VT, SET1, XS, K0)                                                            \
  const uint8_t *c##r = codes + (row) * output_count;                                                    \
  size_t e##r = (row) * output_count, n##r = ((((e##r + K0) >> shift) + 1u) << shift) - e##r;            \
  double x##r = input[row], s##r = q8_scale(exponents_le, (e##r + K0) >> shift);                         \
  VT m##r = SET1((XS) ? x##r * s##r : x##r)
#define Q8_PAIR_TURN(r, SET1, XS)                                                                        \
  if (k == n##r) {                                                                                       \
    s##r = q8_scale(exponents_le, (e##r + k) >> shift);                                                  \
    if (XS) m##r = SET1(x##r * s##r);                                                                    \
    n##r += group_size;                                                                                  \
  }
#define Q8_PAIR_BODY(VT, LANES, LOADU, STOREU, ADD, SET1, TERM, XS, K0, K1)                               \
    unsigned shift = 0u;                                                                                 \
    size_t i, k;                                                                                         \
    while (((size_t)1 << shift) < group_size) ++shift;                                                  \
    for (k = K0; k < K1; ++k) output[k] = 0.0;                                                           \
    for (i = 0u; i + 1u < input_count; i += 2u) {                                                        \
      Q8_PAIR_ROW(0, i, VT, SET1, XS, K0);                                                               \
      Q8_PAIR_ROW(1, i + 1u, VT, SET1, XS, K0);                                                          \
      for (k = K0; k < K1; k += (LANES)) {                                                               \
        Q8_PAIR_TURN(0, SET1, XS)                                                                        \
        Q8_PAIR_TURN(1, SET1, XS)                                                                        \
        STOREU(output + k, ADD(ADD(LOADU(output + k), TERM(0)), TERM(1)));                               \
      }                                                                                                  \
    }                                                                                                    \
    if (i < input_count) {                                                                               \
      Q8_PAIR_ROW(0, i, VT, SET1, XS, K0);                                                               \
      for (k = K0; k < K1; k += (LANES)) {                                                               \
        Q8_PAIR_TURN(0, SET1, XS)                                                                        \
        STOREU(output + k, ADD(LOADU(output + k), TERM(0)));                                             \
      }                                                                                                  \
    }
/* each pair kernel's two instances of its one text: the whole call's, its range [0, output_count), and a range's */
#define Q8_PAIR_KERNEL(NAME, TARGET, VT, LANES, LOADU, STOREU, ADD, SET1, TERM, XS)                       \
  __attribute__((target(TARGET))) static void NAME(const double *input, size_t input_count,              \
      const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,        \
      double *output) {                                                                                  \
    Q8_PAIR_BODY(VT, LANES, LOADU, STOREU, ADD, SET1, TERM, XS, 0u, output_count)                        \
  }                                                                                                      \
  __attribute__((target(TARGET))) static void NAME##_range(const double *input, size_t input_count,      \
      const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,        \
      size_t k0, size_t k1, double *output) {                                                            \
    Q8_PAIR_BODY(VT, LANES, LOADU, STOREU, ADD, SET1, TERM, XS, k0, k1)                                  \
  }
/* row r's term at column k: x * (code * s) as before, code * (x * s) where Q8-BOUND proves the call */
#define Q8_TERM512(r) q8_terms512(c##r + k, m##r, s##r)
#define Q8_TERM512_XS(r) q8_terms512_xs(c##r + k, m##r)
#define Q8_TERM256(r) q8_terms_x8(c##r + k, m##r, s##r)
#define Q8_TERM256_XS(r) q8_terms_x8_xs(c##r + k, m##r)
Q8_PAIR_KERNEL(q8_matvec_pair512, "avx512f", __m512d, 8u, _mm512_loadu_pd, _mm512_storeu_pd, _mm512_add_pd,
               _mm512_set1_pd, Q8_TERM512, 0)
Q8_PAIR_KERNEL(q8_matvec_pair512_xs, "avx512f", __m512d, 8u, _mm512_loadu_pd, _mm512_storeu_pd, _mm512_add_pd,
               _mm512_set1_pd, Q8_TERM512_XS, 1)
Q8_PAIR_KERNEL(q8_matvec_pair256, "avx2", q8_x8, 8u, q8_x8_loadu, q8_x8_storeu, q8_x8_add, q8_x8_set1, Q8_TERM256, 0)
Q8_PAIR_KERNEL(q8_matvec_pair256_xs, "avx2", q8_x8, 8u, q8_x8_loadu, q8_x8_storeu, q8_x8_add, q8_x8_set1,
               Q8_TERM256_XS, 1)
#undef Q8_TERM512
#undef Q8_TERM512_XS
#undef Q8_TERM256
#undef Q8_TERM256_XS
#undef Q8_PAIR_KERNEL
#undef Q8_PAIR_BODY
#undef Q8_PAIR_TURN
#undef Q8_PAIR_ROW
/* Q8-BOUND's ranges: every exponent of groups [g0, g1) in [-300, 300] -- (e + 300) as an unsigned 16-bit value at
 * most 600 exactly there */
__attribute__((target("avx2"))) static inline int q8_exponents_exact(const uint8_t *exponents_le, size_t g0,
                                                                     size_t g1) {
  const __m256i bias = _mm256_set1_epi16((short)-BCIR_AI_EXP_MIN);
  __m256i top = _mm256_setzero_si256();
  uint16_t lane[16];
  size_t g = g0;
  int k;
  for (; g + 16u <= g1; g += 16u) {
    __m256i e = _mm256_loadu_si256((const __m256i *)(const void *)(exponents_le + 2u * g));
    top = _mm256_max_epu16(top, _mm256_add_epi16(e, bias));
  }
  _mm256_storeu_si256((__m256i *)(void *)lane, top);
  for (k = 0; k < 16; ++k)
    if (lane[k] > BCIR_AI_EXP_MAX - BCIR_AI_EXP_MIN) return 0;
  for (; g < g1; ++g) {
    int e = exponent_le(exponents_le, g);
    if (e < BCIR_AI_EXP_MIN || e > BCIR_AI_EXP_MAX) return 0;
  }
  return 1;
}
/* and every nonzero |x| in [2^-722, 2^724). The inputs are finite (every entry checks them before a kernel runs). */
__attribute__((target("avx2"))) static inline int q8_input_exact(const double *input, size_t input_count) {
  const __m256d magnitude = _mm256_set1_pd(-0.0), low = _mm256_set1_pd(0x1p-722), high = _mm256_set1_pd(0x1p724);
  const __m256d zero = _mm256_setzero_pd();
  __m256d outside = zero;
  size_t i = 0u;
  for (; i + 4u <= input_count; i += 4u) {
    __m256d a = _mm256_andnot_pd(magnitude, _mm256_loadu_pd(input + i));
    __m256d tiny = _mm256_and_pd(_mm256_cmp_pd(a, low, _CMP_LT_OQ), _mm256_cmp_pd(a, zero, _CMP_NEQ_OQ));
    outside = _mm256_or_pd(outside, _mm256_or_pd(tiny, _mm256_cmp_pd(a, high, _CMP_GE_OQ)));
  }
  if (_mm256_movemask_pd(outside)) return 0;
  for (; i < input_count; ++i) {
    double a = fabs(input[i]);
    if (a != 0.0 && (a < 0x1p-722 || a >= 0x1p724)) return 0;
  }
  return 1;
}
/* Q8-BOUND's ranges, read once a call: 1 where every exponent of the call's groups and every input lie in them. A
 * call over columns [k0, k1) of the rows (DECODE-THREADS) reads the groups those columns touch, row by row. */
__attribute__((target("avx2"))) static int q8_call_exact(const double *input, size_t input_count,
                                                          const uint8_t *exponents_le, size_t output_count,
                                                          size_t group_size, size_t k0, size_t k1) {
  size_t i;
  if (k0 == 0u && k1 == output_count)
    return q8_exponents_exact(exponents_le, 0u, 1u + (input_count * output_count - 1u) / group_size) &&
           q8_input_exact(input, input_count);
  for (i = 0u; i < input_count; ++i)
    if (!q8_exponents_exact(exponents_le, (i * output_count + k0) / group_size,
                            (i * output_count + k1 - 1u) / group_size + 1u))
      return 0;
  return q8_input_exact(input, input_count);
}
/* Q8-BOUND where the call's ranges prove every x * s exact, the terms as before elsewhere; Q8-PAIR where eight columns'
 * terms lie in one group and the columns [k0, k1) start and end on a vector's edge, the runs elsewhere -- each
 * kernel's whole-call instance where the range is the whole, its range's elsewhere */
#define Q8_LANES_RUN(KERNEL, ...)                                                                        \
  do {                                                                                                   \
    Q8_COLUMNS_SAW(whole ? 1 : 2);                                                                       \
    if (whole) KERNEL(__VA_ARGS__, output);                                                              \
    else KERNEL##_range(__VA_ARGS__, k0, k1, output);                                                    \
  } while (0)
Q8_INLINE void q8_matvec_lanes(int instance, const double *input, size_t input_count, const uint8_t *codes,
                               const uint8_t *exponents_le, size_t output_count, size_t group_size, size_t k0,
                               size_t k1, double *output) {
  int pair = output_count % 8u == 0u && group_size >= 8u && !(group_size & (group_size - 1u)) && k0 % 8u == 0u &&
             k1 % 8u == 0u;
  int whole = k0 == 0u && k1 == output_count;
  Q8_PAIR_SAW(pair ? 1 : 2);
  if (q8_call_exact(input, input_count, exponents_le, output_count, group_size, k0, k1)) {
    Q8_BOUND_SAW(1);
    if (!pair)
      Q8_LANES_RUN(q8_matvec_runs_xs, instance, input, input_count, codes, exponents_le, output_count, group_size);
    else if (instance == 4)
      Q8_LANES_RUN(q8_matvec_pair512_xs, input, input_count, codes, exponents_le, output_count, group_size);
    else
      Q8_LANES_RUN(q8_matvec_pair256_xs, input, input_count, codes, exponents_le, output_count, group_size);
    return;
  }
  Q8_BOUND_SAW(2);
  if (!pair)
    Q8_LANES_RUN(q8_matvec_runs, instance, input, input_count, codes, exponents_le, output_count, group_size);
  else if (instance == 4)
    Q8_LANES_RUN(q8_matvec_pair512, input, input_count, codes, exponents_le, output_count, group_size);
  else
    Q8_LANES_RUN(q8_matvec_pair256, input, input_count, codes, exponents_le, output_count, group_size);
}
#undef Q8_LANES_RUN
#endif

/* Instance 1, the portable loops: the tree before Q8-LANES' prevalidated entries, byte for byte, as functions of
 * their own that are never inlined into the dispatching entries -- there the vector instances' code took their
 * registers, and clang's matvec reloaded three of its bases from the stack a term (x1.2 of its time). */
#if defined(__GNUC__) || defined(__clang__)
#define Q8_PORTABLE static __attribute__((noinline))
#elif defined(_MSC_VER)
#define Q8_PORTABLE static __declspec(noinline)
#else
#define Q8_PORTABLE static
#endif

Q8_PORTABLE int q8_matvec_portable(
    const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output) {
  size_t elements, groups, index, input_index, output_index, next_group;
  double scale;
  if (!q8_shape(input_count, output_count, group_size, &elements, &groups) ||
      !q8_input_validate(input, input_count) || codes == NULL ||
      exponents_le == NULL || output == NULL || code_count < elements ||
      exponent_count < groups) return BCIR_AI_INVALID_ARGUMENT;
  for (output_index = 0u; output_index < output_count; ++output_index)
    output[output_index] = 0.0;
  index = 0u;
  next_group = group_size;
  if (next_group > elements) next_group = elements;
  scale = ldexp(1.0, exponent_le(exponents_le, 0u));
  for (input_index = 0u; input_index < input_count; ++input_index) {
    for (output_index = 0u; output_index < output_count; ++output_index, ++index) {
      int code = codes[index] >= UINT8_C(0x80) ? (int)codes[index] - 256 : (int)codes[index];
      output[output_index] += input[input_index] * ((double)code * scale);
      if (index + 1u == next_group && index + 1u < elements) {
        size_t group = (index + 1u) / group_size;
        next_group = index + 1u + group_size;
        if (next_group > elements) next_group = elements;
        scale = ldexp(1.0, exponent_le(exponents_le, group));
      }
    }
  }
  for (output_index = 0u; output_index < output_count; ++output_index) {
    if (!isfinite(output[output_index])) {
      memset(output, 0, output_count * sizeof *output);
      return BCIR_AI_INVALID_NUMBER;
    }
  }
  return BCIR_AI_OK;
}

int bcir_ai_q8_matvec_f64_instance(
    int instance, const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output) {
  instance = q8_instance(instance);
  if (instance == 1) {
    int status = q8_matvec_portable(input, input_count, codes, code_count, exponents_le, exponent_count,
                                    output_count, group_size, output);
    if (status != BCIR_AI_INVALID_ARGUMENT) Q8_SAW(1);
    return status;
  }
#ifdef BCIR_AI_DISPATCH
  {
    size_t elements, groups, output_index;
    if (!q8_shape(input_count, output_count, group_size, &elements, &groups) ||
        !q8_input_validate(input, input_count) || codes == NULL ||
        exponents_le == NULL || output == NULL || code_count < elements ||
        exponent_count < groups || !instance) return BCIR_AI_INVALID_ARGUMENT;
    Q8_SAW(instance);
    q8_matvec_lanes(instance, input, input_count, codes, exponents_le, output_count, group_size, 0u, output_count,
                    output);
    for (output_index = 0u; output_index < output_count; ++output_index) {
      if (!isfinite(output[output_index])) {
        memset(output, 0, output_count * sizeof *output);
        return BCIR_AI_INVALID_NUMBER;
      }
    }
    return BCIR_AI_OK;
  }
#else
  return BCIR_AI_INVALID_ARGUMENT; /* without the dispatch only instance 1 is admitted */
#endif
}

int bcir_ai_q8_matvec_f64_prevalidated(
    const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output) {
  return bcir_ai_q8_matvec_f64_instance(0, input, input_count, codes, code_count, exponents_le,
                                        exponent_count, output_count, group_size, output);
}

int bcir_ai_q8_matvec_f64(const double *input, size_t input_count,
                          const uint8_t *codes, size_t code_count,
                          const uint8_t *exponents_le, size_t exponent_count,
                          size_t output_count, size_t group_size, double *output) {
  size_t elements, groups;
  int status;
  if (!q8_shape(input_count, output_count, group_size, &elements, &groups))
    return BCIR_AI_INVALID_ARGUMENT;
  status = bcir_ai_q8_values_validate(codes, code_count, exponents_le,
                                      exponent_count, elements, group_size);
  if (status != BCIR_AI_OK) return status;
  return bcir_ai_q8_matvec_f64_prevalidated(
      input, input_count, codes, code_count, exponents_le, exponent_count,
      output_count, group_size, output);
}

/* DECODE-THREADS: the portable loop over columns [k0, k1) alone -- each of those outputs' terms the loop's own, in
 * its order, each element's scale its group's */
Q8_PORTABLE void q8_matvec_columns_portable(const double *input, size_t input_count, const uint8_t *codes,
                                            const uint8_t *exponents_le, size_t output_count, size_t group_size,
                                            size_t k0, size_t k1, double *output) {
  size_t input_index, output_index;
  for (output_index = k0; output_index < k1; ++output_index) output[output_index] = 0.0;
  for (input_index = 0u; input_index < input_count; ++input_index) {
    size_t index = input_index * output_count + k0, next_group = (index / group_size + 1u) * group_size;
    double scale = ldexp(1.0, exponent_le(exponents_le, index / group_size));
    for (output_index = k0; output_index < k1; ++output_index, ++index) {
      int code;
      if (index == next_group) {
        scale = ldexp(1.0, exponent_le(exponents_le, index / group_size));
        next_group += group_size;
      }
      code = codes[index] >= UINT8_C(0x80) ? (int)codes[index] - 256 : (int)codes[index];
      output[output_index] += input[input_index] * ((double)code * scale);
    }
  }
}

int bcir_ai_q8_matvec_f64_columns_instance(
    int instance, const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, size_t column_begin, size_t column_end, double *output) {
  size_t elements, groups, output_index;
  instance = q8_instance(instance);
  if (!q8_shape(input_count, output_count, group_size, &elements, &groups) ||
      !q8_input_validate(input, input_count) || codes == NULL || exponents_le == NULL || output == NULL ||
      code_count < elements || exponent_count < groups || column_begin >= column_end ||
      column_end > output_count || !instance)
    return BCIR_AI_INVALID_ARGUMENT;
  Q8_SAW(instance);
  if (instance == 1)
    q8_matvec_columns_portable(input, input_count, codes, exponents_le, output_count, group_size, column_begin,
                               column_end, output);
#ifdef BCIR_AI_DISPATCH
  else
    q8_matvec_lanes(instance, input, input_count, codes, exponents_le, output_count, group_size, column_begin,
                    column_end, output);
#endif
  for (output_index = column_begin; output_index < column_end; ++output_index) {
    if (!isfinite(output[output_index])) {
      memset(output + column_begin, 0, (column_end - column_begin) * sizeof *output);
      return BCIR_AI_INVALID_NUMBER;
    }
  }
  return BCIR_AI_OK;
}

int bcir_ai_q8_matvec_f64_columns(
    const double *input, size_t input_count, const uint8_t *codes, size_t code_count,
    const uint8_t *exponents_le, size_t exponent_count, size_t output_count, size_t group_size,
    size_t column_begin, size_t column_end, double *output) {
  return bcir_ai_q8_matvec_f64_columns_instance(0, input, input_count, codes, code_count, exponents_le,
                                                exponent_count, output_count, group_size, column_begin,
                                                column_end, output);
}

#ifdef BCIR_AI_DISPATCH
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
/* Rows row0 .. row0+rows-1 (at most 8) of the row dots, a row's chain a lane: four lanes a register. */
__attribute__((target("avx2"))) static void q8_rows_avx2(const double *input, size_t input_count,
    const uint8_t *codes, const uint8_t *exponents_le, size_t group_size, size_t row0, size_t rows,
    double *output) {
  __m256d t0 = _mm256_setzero_pd(), t1 = _mm256_setzero_pd();
  size_t c = 0u, k;
  while (c < input_count) {
    size_t n = q8_block_run(input_count, group_size, row0, rows, c);
    __m256d s0 = _mm256_setr_pd(q8_block_scale(exponents_le, input_count, group_size, row0, rows, 0u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 1u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 2u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 3u, c));
    __m256d s1 = _mm256_setr_pd(q8_block_scale(exponents_le, input_count, group_size, row0, rows, 4u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 5u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 6u, c),
                                q8_block_scale(exponents_le, input_count, group_size, row0, rows, 7u, c));
    for (k = 0u; rows == 8u && k + 8u <= n; k += 8u) {
      __m128i col[4];
      int q;
      q8_transpose8(codes + row0 * input_count + c + k, input_count, col);
      for (q = 0; q < 4; ++q) {
        __m128i lo = col[q], hi = _mm_srli_si128(col[q], 8);
        __m256d xa = _mm256_set1_pd(input[c + k + 2u * (size_t)q]);
        __m256d xb = _mm256_set1_pd(input[c + k + 2u * (size_t)q + 1u]);
        __m256d ca0 = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(lo));
        __m256d ca1 = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(lo, 4)));
        __m256d cb0 = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(hi));
        __m256d cb1 = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(hi, 4)));
        t0 = _mm256_add_pd(t0, _mm256_mul_pd(xa, _mm256_mul_pd(ca0, s0)));
        t1 = _mm256_add_pd(t1, _mm256_mul_pd(xa, _mm256_mul_pd(ca1, s1)));
        t0 = _mm256_add_pd(t0, _mm256_mul_pd(xb, _mm256_mul_pd(cb0, s0)));
        t1 = _mm256_add_pd(t1, _mm256_mul_pd(xb, _mm256_mul_pd(cb1, s1)));
      }
    }
    for (; k < n; ++k) {  /* a run's last coordinates, or every one of a block short of eight rows */
      __m128i w = _mm_cvtsi64_si128(q8_block_column(codes, input_count, row0, rows, c + k));
      __m256d xv = _mm256_set1_pd(input[c + k]);
      t0 = _mm256_add_pd(t0, _mm256_mul_pd(xv, _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepi8_epi32(w)), s0)));
      t1 = _mm256_add_pd(t1, _mm256_mul_pd(xv, _mm256_mul_pd(
                                 _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(w, 4))), s1)));
    }
    c += n;
  }
  if (rows == 8u) {
    _mm256_storeu_pd(output, t0);
    _mm256_storeu_pd(output + 4, t1);
  } else {
    __m256i lanes = _mm256_setr_epi64x(0, 1, 2, 3);
    _mm256_maskstore_pd(output, _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)rows), lanes), t0);
    _mm256_maskstore_pd(output + 4, _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)rows - 4), lanes), t1);
  }
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
/* every row eight at a time, each block's totals checked in row order: a non-finite one zeroes the
 * output, as the portable loop leaves it */
static int q8_rows_lanes(int instance, const double *input, size_t input_count, const uint8_t *codes,
                         const uint8_t *exponents_le, size_t output_count, size_t group_size,
                         double *output) {
  size_t row, r;
  for (row = 0u; row < output_count; row += 8u) {
    size_t rows = output_count - row < 8u ? output_count - row : 8u;
    if (instance == 4)
      q8_rows_avx512(input, input_count, codes, exponents_le, group_size, row, rows, output + row);
    else
      q8_rows_avx2(input, input_count, codes, exponents_le, group_size, row, rows, output + row);
    for (r = 0u; r < rows; ++r) {
      if (!isfinite(output[row + r])) {
        memset(output, 0, output_count * sizeof *output);
        return BCIR_AI_INVALID_NUMBER;
      }
    }
  }
  return BCIR_AI_OK;
}

/* HEAD-PACK: a group's eight scales from its segment's eight exponents, 2^e as (e + 1023) << 52 (|e| <= 300, the
 * range a pack admits: q8_scale's own bits) */
#define Q8_PACK_SCALES512(P)                                                                                    \
  _mm512_castsi512_pd(_mm512_slli_epi64(                                                                        \
      _mm512_add_epi64(_mm512_cvtepi16_epi64(_mm_loadu_si128((const __m128i *)(P))), _mm512_set1_epi64(1023)), \
      52))
/* one coordinate's eight terms, x * (code * s), added to the eight rows' totals: the codes the eight bytes at P */
#define Q8_PACK_TERMS512(T, S, P)                                                                  \
  T = _mm512_add_pd(T, _mm512_mul_pd(x, _mm512_mul_pd(_mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(    \
                                                          _mm_loadl_epi64((const __m128i *)(P)))), \
                                                      S)))
#define Q8_PACK_SCALES256(L, H, P)                                                                               \
  do {                                                                                                           \
    __m128i e_ = _mm_loadu_si128((const __m128i *)(P));                                                          \
    __m256i bias_ = _mm256_set1_epi64x(1023);                                                                    \
    L = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi16_epi64(e_), bias_), 52));         \
    H = _mm256_castsi256_pd(                                                                                     \
        _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi16_epi64(_mm_srli_si128(e_, 8)), bias_), 52));           \
  } while (0)
#define Q8_PACK_TERMS256(TL, TH, SL, SH, P)                                                                       \
  do {                                                                                                            \
    __m128i w_ = _mm_loadl_epi64((const __m128i *)(P));                                                           \
    TL = _mm256_add_pd(TL, _mm256_mul_pd(x, _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepi8_epi32(w_)), SL)));     \
    TH = _mm256_add_pd(                                                                                           \
        TH, _mm256_mul_pd(x, _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(w_, 4))), SH))); \
  } while (0)
/* HEAD-PACK: rows row0 .. row0 + 31, the pack's blocks row0 / 8 to row0 / 8 + 3, each block's eight rows a register */
__attribute__((target("avx512f,avx2"))) static void q8_rows_pack512x4(const double *input, size_t input_count,
    const uint8_t *packed, size_t group_size, size_t row0, double *output) {
  size_t runs = input_count / group_size, step = 16u + 8u * group_size, span = runs * step, j, k;
  const uint8_t *p0 = packed + row0 / 8u * span, *p1 = p0 + span, *p2 = p1 + span, *p3 = p2 + span;
  __m512d t0 = _mm512_setzero_pd(), t1 = _mm512_setzero_pd(), t2 = _mm512_setzero_pd(), t3 = _mm512_setzero_pd();
  for (j = 0u; j < runs; ++j, p0 += step, p1 += step, p2 += step, p3 += step) {
    const double *xr = input + j * group_size;
    __m512d s0 = Q8_PACK_SCALES512(p0), s1 = Q8_PACK_SCALES512(p1);
    __m512d s2 = Q8_PACK_SCALES512(p2), s3 = Q8_PACK_SCALES512(p3);
    for (k = 0u; k < group_size; ++k) {
      __m512d x = _mm512_set1_pd(xr[k]);
      Q8_PACK_TERMS512(t0, s0, p0 + 16u + 8u * k);
      Q8_PACK_TERMS512(t1, s1, p1 + 16u + 8u * k);
      Q8_PACK_TERMS512(t2, s2, p2 + 16u + 8u * k);
      Q8_PACK_TERMS512(t3, s3, p3 + 16u + 8u * k);
    }
  }
  _mm512_storeu_pd(output, t0);
  _mm512_storeu_pd(output + 8, t1);
  _mm512_storeu_pd(output + 16, t2);
  _mm512_storeu_pd(output + 24, t3);
}
/* one block: rows row0 .. row0 + rows - 1 (row0 a multiple of 8, rows at most 8; a short block's missing rows,
 * codes 0 in the pack, never stored) */
__attribute__((target("avx512f,avx2"))) static void q8_rows_pack512(const double *input, size_t input_count,
    const uint8_t *packed, size_t group_size, size_t row0, size_t rows, double *output) {
  size_t runs = input_count / group_size, step = 16u + 8u * group_size, j, k;
  const uint8_t *p0 = packed + row0 / 8u * runs * step;
  __m512d t0 = _mm512_setzero_pd();
  for (j = 0u; j < runs; ++j, p0 += step) {
    const double *xr = input + j * group_size;
    __m512d s0 = Q8_PACK_SCALES512(p0);
    for (k = 0u; k < group_size; ++k) {
      __m512d x = _mm512_set1_pd(xr[k]);
      Q8_PACK_TERMS512(t0, s0, p0 + 16u + 8u * k);
    }
  }
  _mm512_mask_storeu_pd(output, (__mmask8)((1u << rows) - 1u), t0);
}
/* the same pass at AVX2, each block's eight rows two registers */
__attribute__((target("avx2"))) static void q8_rows_pack256x4(const double *input, size_t input_count,
    const uint8_t *packed, size_t group_size, size_t row0, double *output) {
  size_t runs = input_count / group_size, step = 16u + 8u * group_size, span = runs * step, j, k;
  const uint8_t *p0 = packed + row0 / 8u * span, *p1 = p0 + span, *p2 = p1 + span, *p3 = p2 + span;
  __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd(), b0 = _mm256_setzero_pd(), b1 = _mm256_setzero_pd();
  __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd(), d0 = _mm256_setzero_pd(), d1 = _mm256_setzero_pd();
  for (j = 0u; j < runs; ++j, p0 += step, p1 += step, p2 += step, p3 += step) {
    const double *xr = input + j * group_size;
    __m256d sa0, sa1, sb0, sb1, sc0, sc1, sd0, sd1;
    Q8_PACK_SCALES256(sa0, sa1, p0);
    Q8_PACK_SCALES256(sb0, sb1, p1);
    Q8_PACK_SCALES256(sc0, sc1, p2);
    Q8_PACK_SCALES256(sd0, sd1, p3);
    for (k = 0u; k < group_size; ++k) {
      __m256d x = _mm256_set1_pd(xr[k]);
      Q8_PACK_TERMS256(a0, a1, sa0, sa1, p0 + 16u + 8u * k);
      Q8_PACK_TERMS256(b0, b1, sb0, sb1, p1 + 16u + 8u * k);
      Q8_PACK_TERMS256(c0, c1, sc0, sc1, p2 + 16u + 8u * k);
      Q8_PACK_TERMS256(d0, d1, sd0, sd1, p3 + 16u + 8u * k);
    }
  }
  _mm256_storeu_pd(output, a0);
  _mm256_storeu_pd(output + 4, a1);
  _mm256_storeu_pd(output + 8, b0);
  _mm256_storeu_pd(output + 12, b1);
  _mm256_storeu_pd(output + 16, c0);
  _mm256_storeu_pd(output + 20, c1);
  _mm256_storeu_pd(output + 24, d0);
  _mm256_storeu_pd(output + 28, d1);
}
__attribute__((target("avx2"))) static void q8_rows_pack256(const double *input, size_t input_count,
    const uint8_t *packed, size_t group_size, size_t row0, size_t rows, double *output) {
  size_t runs = input_count / group_size, step = 16u + 8u * group_size, j, k;
  const uint8_t *p0 = packed + row0 / 8u * runs * step;
  __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
  __m256i lanes = _mm256_setr_epi64x(0, 1, 2, 3);
  for (j = 0u; j < runs; ++j, p0 += step) {
    const double *xr = input + j * group_size;
    __m256d sa0, sa1;
    Q8_PACK_SCALES256(sa0, sa1, p0);
    for (k = 0u; k < group_size; ++k) {
      __m256d x = _mm256_set1_pd(xr[k]);
      Q8_PACK_TERMS256(a0, a1, sa0, sa1, p0 + 16u + 8u * k);
    }
  }
  _mm256_maskstore_pd(output, _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)rows), lanes), a0);
  _mm256_maskstore_pd(output + 4, _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)rows - 4), lanes), a1);
}
/* HEAD-PACK's rows: four blocks a pass while 32 rows remain, then a block at a time; a pass's or a block's totals
 * checked in row order, a non-finite one zeroing every output as the portable loop leaves them */
static int q8_rows_pack_lanes(int instance, const double *input, size_t input_count, const uint8_t *packed,
                              size_t output_count, size_t group_size, double *output) {
  size_t row, rows, r, passes = 0u, blocks = 0u;
  int status = BCIR_AI_OK;
  for (row = 0u; row < output_count && status == BCIR_AI_OK; row += rows) {
    if (output_count - row >= 32u) {
      rows = 32u;
      ++passes;
      if (instance == 4)
        q8_rows_pack512x4(input, input_count, packed, group_size, row, output + row);
      else
        q8_rows_pack256x4(input, input_count, packed, group_size, row, output + row);
    } else {
      rows = output_count - row < 8u ? output_count - row : 8u;
      ++blocks;
      if (instance == 4)
        q8_rows_pack512(input, input_count, packed, group_size, row, rows, output + row);
      else
        q8_rows_pack256(input, input_count, packed, group_size, row, rows, output + row);
    }
    for (r = 0u; r < rows; ++r)
      if (!isfinite(output[row + r])) {
        memset(output, 0, output_count * sizeof *output);
        status = BCIR_AI_INVALID_NUMBER;
        break;
      }
  }
  Q8_PACK_SAW(1, passes, blocks);
  return status;
}
#endif

Q8_PORTABLE int q8_rows_dot_portable(
    const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output) {
  size_t elements, groups, row, coordinate, next_group = 0u;
  double scale = 0.0;
  if (!q8_shape(input_count, output_count, group_size, &elements, &groups) ||
      !q8_input_validate(input, input_count) || codes == NULL ||
      exponents_le == NULL || output == NULL || code_count < elements ||
      exponent_count < groups) return BCIR_AI_INVALID_ARGUMENT;
  for (row = 0u; row < output_count; ++row) {
    size_t base = row * input_count;
    double total = 0.0;
    for (coordinate = 0u; coordinate < input_count; ++coordinate) {
      size_t index = base + coordinate;
      int code = codes[index] >= UINT8_C(0x80) ? (int)codes[index] - 256 : (int)codes[index];
      if (index == next_group) {
        scale = ldexp(1.0, exponent_le(exponents_le, index / group_size));
        next_group = index + group_size;
        if (next_group > elements) next_group = elements;
      }
      total += input[coordinate] * ((double)code * scale);
    }
    if (!isfinite(total)) {
      memset(output, 0, output_count * sizeof *output);
      return BCIR_AI_INVALID_NUMBER;
    }
    output[row] = total;
  }
  return BCIR_AI_OK;
}

int bcir_ai_q8_rows_dot_f64_instance(
    int instance, const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output) {
  instance = q8_instance(instance);
  if (instance == 1) {
    int status = q8_rows_dot_portable(input, input_count, codes, code_count, exponents_le, exponent_count,
                                      output_count, group_size, output);
    if (status != BCIR_AI_INVALID_ARGUMENT) Q8_SAW(1);
    return status;
  }
#ifdef BCIR_AI_DISPATCH
  {
    size_t elements, groups;
    if (!q8_shape(input_count, output_count, group_size, &elements, &groups) ||
        !q8_input_validate(input, input_count) || codes == NULL ||
        exponents_le == NULL || output == NULL || code_count < elements ||
        exponent_count < groups || !instance) return BCIR_AI_INVALID_ARGUMENT;
    Q8_SAW(instance);
    return q8_rows_lanes(instance, input, input_count, codes, exponents_le, output_count, group_size,
                         output);
  }
#else
  return BCIR_AI_INVALID_ARGUMENT; /* without the dispatch only instance 1 is admitted */
#endif
}

int bcir_ai_q8_rows_dot_f64_prevalidated(
    const double *input, size_t input_count, const uint8_t *codes,
    size_t code_count, const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output) {
  return bcir_ai_q8_rows_dot_f64_instance(0, input, input_count, codes, code_count, exponents_le,
                                          exponent_count, output_count, group_size, output);
}

int bcir_ai_q8_rows_dot_f64(const double *input, size_t input_count,
                            const uint8_t *codes, size_t code_count,
                            const uint8_t *exponents_le, size_t exponent_count,
                            size_t output_count, size_t group_size, double *output) {
  size_t elements, groups;
  int status;
  if (!q8_shape(input_count, output_count, group_size, &elements, &groups))
    return BCIR_AI_INVALID_ARGUMENT;
  status = bcir_ai_q8_values_validate(codes, code_count, exponents_le,
                                      exponent_count, elements, group_size);
  if (status != BCIR_AI_OK) return status;
  return bcir_ai_q8_rows_dot_f64_prevalidated(
      input, input_count, codes, code_count, exponents_le, exponent_count,
      output_count, group_size, output);
}

/* HEAD-PACK: ceil(output_count / 8) blocks of input_count / group_size segments, 16 + 8 * group_size bytes each */
int bcir_ai_q8_rows_pack_size(size_t input_count, size_t output_count, size_t group_size, size_t *packed_size) {
  size_t elements, groups, blocks, runs, step;
  if (packed_size == NULL || !q8_shape(input_count, output_count, group_size, &elements, &groups) ||
      input_count % group_size != 0u)
    return BCIR_AI_INVALID_ARGUMENT;
  blocks = output_count / 8u + (output_count % 8u != 0u);
  runs = input_count / group_size;
  step = 16u + 8u * group_size; /* group_size <= UINT16_MAX: step at most 524,296 */
  if (runs > SIZE_MAX / step || blocks > SIZE_MAX / (runs * step)) return BCIR_AI_INVALID_ARGUMENT;
  *packed_size = blocks * runs * step;
  return BCIR_AI_OK;
}

int bcir_ai_q8_rows_pack(const uint8_t *codes, size_t code_count, const uint8_t *exponents_le,
                         size_t exponent_count, size_t input_count, size_t output_count, size_t group_size,
                         uint8_t *packed, size_t packed_capacity) {
  size_t size, runs, step, row, j, k;
  int status;
  if (bcir_ai_q8_rows_pack_size(input_count, output_count, group_size, &size) != BCIR_AI_OK || packed == NULL ||
      packed_capacity < size)
    return BCIR_AI_INVALID_ARGUMENT;
  status = bcir_ai_q8_values_validate(codes, code_count, exponents_le, exponent_count, input_count * output_count,
                                      group_size);
  if (status != BCIR_AI_OK) return status;
  memset(packed, 0, size);
  runs = input_count / group_size;
  step = 16u + 8u * group_size;
  for (row = 0u; row < output_count; ++row) {
    uint8_t *segment = packed + row / 8u * runs * step;
    size_t lane = row % 8u;
    for (j = 0u; j < runs; ++j, segment += step) {
      const uint8_t *exponent = exponents_le + 2u * (row * runs + j);
      const uint8_t *code = codes + row * input_count + j * group_size;
      segment[2u * lane] = exponent[0];
      segment[2u * lane + 1u] = exponent[1];
      for (k = 0u; k < group_size; ++k) segment[16u + 8u * k + lane] = code[k];
    }
  }
  return BCIR_AI_OK;
}

/* HEAD-PACK's portable loop: the row dots' own, a row at a time and its coordinates in order, over the pack */
Q8_PORTABLE int q8_rows_dot_packed_portable(const double *input, size_t input_count, const uint8_t *packed,
                                            size_t output_count, size_t group_size, double *output) {
  size_t runs = input_count / group_size, step = 16u + 8u * group_size, row, j, k;
  for (row = 0u; row < output_count; ++row) {
    const uint8_t *segment = packed + row / 8u * runs * step;
    size_t lane = row % 8u;
    double total = 0.0;
    for (j = 0u; j < runs; ++j, segment += step) {
      double scale = ldexp(1.0, exponent_le(segment, lane));
      for (k = 0u; k < group_size; ++k) {
        uint8_t byte = segment[16u + 8u * k + lane];
        int code = byte >= UINT8_C(0x80) ? (int)byte - 256 : (int)byte;
        total += input[j * group_size + k] * ((double)code * scale);
      }
    }
    if (!isfinite(total)) {
      memset(output, 0, output_count * sizeof *output);
      return BCIR_AI_INVALID_NUMBER;
    }
    output[row] = total;
  }
  return BCIR_AI_OK;
}

int bcir_ai_q8_rows_dot_f64_packed_instance(int instance, const double *input, size_t input_count,
                                            const uint8_t *packed, size_t packed_size, size_t output_count,
                                            size_t group_size, double *output) {
  size_t size;
  instance = q8_instance(instance);
  if (bcir_ai_q8_rows_pack_size(input_count, output_count, group_size, &size) != BCIR_AI_OK || packed == NULL ||
      packed_size < size || output == NULL || !q8_input_validate(input, input_count) || !instance)
    return BCIR_AI_INVALID_ARGUMENT;
  Q8_SAW(instance);
  if (instance == 1) {
    Q8_PACK_SAW(2, (size_t)0, (size_t)0);
    return q8_rows_dot_packed_portable(input, input_count, packed, output_count, group_size, output);
  }
#ifdef BCIR_AI_DISPATCH
  return q8_rows_pack_lanes(instance, input, input_count, packed, output_count, group_size, output);
#else
  return BCIR_AI_INVALID_ARGUMENT; /* without the dispatch only instance 1 is admitted */
#endif
}

int bcir_ai_q8_rows_dot_f64_packed(const double *input, size_t input_count, const uint8_t *packed,
                                   size_t packed_size, size_t output_count, size_t group_size, double *output) {
  return bcir_ai_q8_rows_dot_f64_packed_instance(0, input, input_count, packed, packed_size, output_count,
                                                 group_size, output);
}

static int match_less(const bcir_ai_match *left, const bcir_ai_match *right) {
  return left->squared_distance < right->squared_distance ||
         (left->squared_distance == right->squared_distance && left->index < right->index);
}

int bcir_ai_q15_topk(const int16_t *query, const int16_t *patterns,
                     const uint8_t *eligible, size_t pattern_count,
                     size_t dimension, size_t top_k, bcir_ai_match *matches,
                     size_t match_capacity, size_t *match_count) {
  size_t pattern, count = 0u;
  if (match_count == NULL) return BCIR_AI_INVALID_ARGUMENT;
  *match_count = 0u;
  if (query == NULL || patterns == NULL || matches == NULL ||
      pattern_count == 0u || pattern_count > (size_t)BCIR_AI_MAX_PATTERNS ||
      dimension == 0u || dimension > (size_t)BCIR_AI_MAX_DIMENSION ||
      top_k == 0u || top_k > (size_t)BCIR_AI_MAX_TOP_K ||
      top_k > pattern_count || match_capacity < top_k ||
      pattern_count > SIZE_MAX / dimension)
    return BCIR_AI_INVALID_ARGUMENT;
  if (eligible != NULL) {
    for (pattern = 0u; pattern < pattern_count; ++pattern)
      if (eligible[pattern] > 1u) return BCIR_AI_INVALID_ARGUMENT;
  }
  for (pattern = 0u; pattern < pattern_count; ++pattern) {
    bcir_ai_match candidate;
    size_t coordinate, position;
    uint64_t distance = 0u;
    if (eligible != NULL && eligible[pattern] == 0u) continue;
    for (coordinate = 0u; coordinate < dimension; ++coordinate) {
      int32_t difference = (int32_t)query[coordinate] -
                           (int32_t)patterns[pattern * dimension + coordinate];
      uint64_t magnitude = difference < 0 ? (uint64_t)(-(int64_t)difference) :
                                            (uint64_t)difference;
      distance += magnitude * magnitude;
    }
    candidate.index = (uint32_t)pattern;
    candidate.squared_distance = distance;
    if (count < top_k) {
      position = count++;
      matches[position] = candidate;
    } else {
      if (!match_less(&candidate, &matches[count - 1u])) continue;
      position = count - 1u;
      matches[position] = candidate;
    }
    while (position > 0u && match_less(&matches[position], &matches[position - 1u])) {
      bcir_ai_match temporary = matches[position - 1u];
      matches[position - 1u] = matches[position];
      matches[position] = temporary;
      --position;
    }
  }
  *match_count = count;
  return BCIR_AI_OK;
}

/* ============================================================================
 * LUT4-LANES (COMP-1): deterministic packed-1-bit factorized matvec & mb4
 *   portable (1), AVX2 (2), AVX-512 (4) -- 0 bit drift, -ffp-contract=off,
 *   zero heap allocations, DECODE-THREADS column slicing.
 * ============================================================================ */

#ifdef BCIR_AI_INSTANCE_PROBE
static int lut4_instance_seen;
#define LUT4_SAW(inst) (lut4_instance_seen = (inst))
int bcir_ai_lut4_instance_probe(void);
int bcir_ai_lut4_instance_probe(void) {
  int seen = lut4_instance_seen;
  lut4_instance_seen = 0;
  return seen;
}
#else
#define LUT4_SAW(inst) ((void)0)
#endif

static uint16_t lut4_rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}

static int lut4_fp16_is_finite(uint16_t raw) {
  return (raw & UINT16_C(0x7C00)) != UINT16_C(0x7C00);
}

static float lut4_fp16_to_f32_bits(uint16_t raw) {
  uint32_t sign = (uint32_t)(raw >> 15) & 1u;
  uint32_t exp = (uint32_t)(raw >> 10) & 0x1Fu;
  uint32_t mant = (uint32_t)(raw & 0x3FFu);
  if (exp == 0u) {
    if (mant == 0u) {
      uint32_t z = sign << 31;
      float out;
      memcpy(&out, &z, sizeof out);
      return out;
    }
    {
      float mag = ldexpf((float)mant, -24);
      return sign ? -mag : mag;
    }
  }
  if (exp == 31u) {
    uint32_t bits = (sign << 31) | UINT32_C(0x7F800000) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
  }
  {
    uint32_t bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
  }
}

static float lut4_fp16_f32(const uint8_t *le, size_t idx) {
  return lut4_fp16_to_f32_bits(lut4_rd16(le + 2u * idx));
}

static double lut4_fp16_f64(const uint8_t *le, size_t idx) {
  return (double)lut4_fp16_to_f32_bits(lut4_rd16(le + 2u * idx));
}

static int lut4_finite_f32(const float *values, size_t count) {
  size_t idx;
  if (values == NULL) return 0;
  for (idx = 0u; idx < count; ++idx)
    if (!isfinite(values[idx])) return 0;
  return 1;
}

int bcir_ai_lut4_instances(void) { return bcir_ai_q8_instances(); }

int bcir_ai_lut4_packed_bytes(size_t rows, size_t cols, size_t *out_bytes) {
  size_t rows32, bpg, groups4;
  if (out_bytes == NULL || rows == 0u || cols == 0u ||
      rows > (size_t)BCIR_AI_MAX_ELEMENTS || cols > (size_t)BCIR_AI_MAX_ELEMENTS ||
      rows > (size_t)BCIR_AI_MAX_ELEMENTS / cols ||
      rows > SIZE_MAX - 31u || cols > SIZE_MAX - 3u)
    return BCIR_AI_INVALID_ARGUMENT;
  rows32 = (rows + 31u) & ~(size_t)31u;
  bpg = rows32 >> 1u;
  groups4 = (cols + 3u) >> 2u;
  if (groups4 > SIZE_MAX / bpg) return BCIR_AI_INVALID_ARGUMENT;
  *out_bytes = groups4 * bpg;
  return BCIR_AI_OK;
}

int bcir_ai_lut4_pack_signs(const int8_t *signs, size_t rows, size_t cols,
                            uint8_t *packed, size_t packed_capacity) {
  size_t required, r, c, bpg, total;
  if (bcir_ai_lut4_packed_bytes(rows, cols, &required) != BCIR_AI_OK ||
      signs == NULL || packed == NULL || packed_capacity < required)
    return BCIR_AI_INVALID_ARGUMENT;
  total = rows * cols;
  for (r = 0u; r < total; ++r)
    if (signs[r] != 1 && signs[r] != -1) return BCIR_AI_INVALID_NUMBER;
  memset(packed, 0, required);
  bpg = ((rows + 31u) & ~(size_t)31u) >> 1u;
  for (r = 0u; r < rows; ++r) {
    size_t blk32 = r >> 5u;
    size_t lane = r & 15u;
    size_t is_hi = (r >> 4u) & 1u;
    const int8_t *row_ptr = signs + r * cols;
    for (c = 0u; c < cols; ++c) {
      if (row_ptr[c] < 0) {
        uint8_t bit = (uint8_t)(1u << (c & 3u));
        size_t off = (c >> 2u) * bpg + (blk32 << 4u) + lane;
        packed[off] = (uint8_t)(packed[off] | (uint8_t)(is_hi ? (bit << 4u) : bit));
      }
    }
  }
  return BCIR_AI_OK;
}

int bcir_ai_lut4_validate_packed(const uint8_t *packed, size_t packed_size,
                                 size_t rows, size_t cols) {
  size_t required, rows32, bpg, groups4, g, r, tail;
  if (bcir_ai_lut4_packed_bytes(rows, cols, &required) != BCIR_AI_OK ||
      packed == NULL || packed_size != required)
    return BCIR_AI_INVALID_ARGUMENT;
  rows32 = (rows + 31u) & ~(size_t)31u;
  bpg = rows32 >> 1u;
  groups4 = (cols + 3u) >> 2u;
  /* Padding rows [rows, rows32) must have zero nibbles in every group. */
  if (rows < rows32) {
    for (g = 0u; g < groups4; ++g) {
      const uint8_t *gp = packed + g * bpg;
      for (r = rows; r < rows32; ++r) {
        size_t blk32 = r >> 5u;
        size_t lane = r & 15u;
        uint8_t byte = gp[(blk32 << 4u) + lane];
        uint8_t nib = (uint8_t)(((r >> 4u) & 1u) ? (byte >> 4u) : (byte & 0x0Fu));
        if (nib != 0u) return BCIR_AI_INVALID_NUMBER;
      }
    }
  }
  /* Padding columns in the trailing 4-column group must have zero bits. */
  tail = cols & 3u;
  if (tail != 0u) {
    uint8_t pad_mask = (uint8_t)(0x0Fu & ~((1u << tail) - 1u));
    const uint8_t *last = packed + (groups4 - 1u) * bpg;
    size_t idx;
    for (idx = 0u; idx < bpg; ++idx) {
      uint8_t byte = last[idx];
      if (((byte & 0x0Fu) & pad_mask) != 0u || (((byte >> 4u) & 0x0Fu) & pad_mask) != 0u)
        return BCIR_AI_INVALID_NUMBER;
    }
  }
  return BCIR_AI_OK;
}

static int lut4_shape_basic(size_t input_count, size_t output_count,
                            const bcir_ai_lut4_path *paths, size_t path_count) {
  size_t p, min_dim;
  if (input_count == 0u || output_count == 0u ||
      input_count > (size_t)BCIR_AI_MAX_ELEMENTS / output_count ||
      paths == NULL || path_count == 0u || path_count > (size_t)BCIR_AI_LUT4_MAX_PATHS)
    return 0;
  min_dim = input_count < output_count ? input_count : output_count;
  for (p = 0u; p < path_count; ++p) {
    size_t req_v = 0u, req_u = 0u;
    if (paths[p].rank == 0u || paths[p].rank > (size_t)BCIR_AI_LUT4_MAX_RANK ||
        paths[p].rank > min_dim ||
        paths[p].v_packed == NULL || paths[p].u_packed == NULL ||
        paths[p].g_fp16_le == NULL || paths[p].l_fp16_le == NULL ||
        paths[p].h_fp16_le == NULL)
      return 0;
    if (bcir_ai_lut4_packed_bytes(paths[p].rank, input_count, &req_v) != BCIR_AI_OK ||
        bcir_ai_lut4_packed_bytes(output_count, paths[p].rank, &req_u) != BCIR_AI_OK ||
        paths[p].v_packed_bytes != req_v || paths[p].u_packed_bytes != req_u)
      return 0;
  }
  return 1;
}

int bcir_ai_lut4_paths_validate(size_t input_count, size_t output_count,
                                const bcir_ai_lut4_path *paths, size_t path_count) {
  size_t p, idx;
  if (!lut4_shape_basic(input_count, output_count, paths, path_count))
    return BCIR_AI_INVALID_ARGUMENT;
  for (p = 0u; p < path_count; ++p) {
    int st = bcir_ai_lut4_validate_packed(paths[p].v_packed, paths[p].v_packed_bytes,
                                          paths[p].rank, input_count);
    if (st != BCIR_AI_OK) return st;
    st = bcir_ai_lut4_validate_packed(paths[p].u_packed, paths[p].u_packed_bytes,
                                      output_count, paths[p].rank);
    if (st != BCIR_AI_OK) return st;
    for (idx = 0u; idx < input_count; ++idx)
      if (!lut4_fp16_is_finite(lut4_rd16(paths[p].g_fp16_le + 2u * idx)))
        return BCIR_AI_INVALID_NUMBER;
    for (idx = 0u; idx < paths[p].rank; ++idx)
      if (!lut4_fp16_is_finite(lut4_rd16(paths[p].l_fp16_le + 2u * idx)))
        return BCIR_AI_INVALID_NUMBER;
    for (idx = 0u; idx < output_count; ++idx)
      if (!lut4_fp16_is_finite(lut4_rd16(paths[p].h_fp16_le + 2u * idx)))
        return BCIR_AI_INVALID_NUMBER;
  }
  return BCIR_AI_OK;
}

/* Build the 16-entry 4-weight binary sign LUT for one group: identical in portable, AVX2, AVX-512. */
static void lut4_build_f64(double x0, double x1, double x2, double x3, double out[16]) {
  double s01_pp = x0 + x1, s01_mp = -x0 + x1, s01_pm = x0 - x1, s01_mm = -x0 - x1;
  double s23_pp = x2 + x3, s23_mp = -x2 + x3, s23_pm = x2 - x3, s23_mm = -x2 - x3;
  out[0]  = s01_pp + s23_pp; out[1]  = s01_mp + s23_pp; out[2]  = s01_pm + s23_pp; out[3]  = s01_mm + s23_pp;
  out[4]  = s01_pp + s23_mp; out[5]  = s01_mp + s23_mp; out[6]  = s01_pm + s23_mp; out[7]  = s01_mm + s23_mp;
  out[8]  = s01_pp + s23_pm; out[9]  = s01_mp + s23_pm; out[10] = s01_pm + s23_pm; out[11] = s01_mm + s23_pm;
  out[12] = s01_pp + s23_mm; out[13] = s01_mp + s23_mm; out[14] = s01_pm + s23_mm; out[15] = s01_mm + s23_mm;
}

static void lut4_build_f32(float x0, float x1, float x2, float x3, float out[16]) {
  float s01_pp = x0 + x1, s01_mp = -x0 + x1, s01_pm = x0 - x1, s01_mm = -x0 - x1;
  float s23_pp = x2 + x3, s23_mp = -x2 + x3, s23_pm = x2 - x3, s23_mm = -x2 - x3;
  out[0]  = s01_pp + s23_pp; out[1]  = s01_mp + s23_pp; out[2]  = s01_pm + s23_pp; out[3]  = s01_mm + s23_pp;
  out[4]  = s01_pp + s23_mp; out[5]  = s01_mp + s23_mp; out[6]  = s01_pm + s23_mp; out[7]  = s01_mm + s23_mp;
  out[8]  = s01_pp + s23_pm; out[9]  = s01_mp + s23_pm; out[10] = s01_pm + s23_pm; out[11] = s01_mm + s23_pm;
  out[12] = s01_pp + s23_mm; out[13] = s01_mp + s23_mm; out[14] = s01_pm + s23_mm; out[15] = s01_mm + s23_mm;
}

/* 32-row block accumulators for gn consecutive 4-col groups: portable, AVX2, AVX-512. */
static void lut4_accum32_f64_portable(const uint8_t *base, size_t bpg, const double *luts,
                                      size_t gn, double acc[32]) {
  size_t g, lane;
  for (g = 0u; g < gn; ++g) {
    const uint8_t *p = base + g * bpg;
    const double *lt = luts + g * 16u;
    for (lane = 0u; lane < 16u; ++lane) {
      uint8_t b = p[lane];
      acc[lane] = acc[lane] + lt[b & 0x0Fu];
      acc[lane + 16u] = acc[lane + 16u] + lt[b >> 4u];
    }
  }
}

static void lut4_accum32_f32_portable(const uint8_t *base, size_t bpg, const float *luts,
                                      size_t gn, float acc[32]) {
  size_t g, lane;
  for (g = 0u; g < gn; ++g) {
    const uint8_t *p = base + g * bpg;
    const float *lt = luts + g * 16u;
    for (lane = 0u; lane < 16u; ++lane) {
      uint8_t b = p[lane];
      acc[lane] = acc[lane] + lt[b & 0x0Fu];
      acc[lane + 16u] = acc[lane + 16u] + lt[b >> 4u];
    }
  }
}

static void lut4_accum32_mb4_f64_portable(const uint8_t *base, size_t bpg,
                                          const double *l0, const double *l1,
                                          const double *l2, const double *l3,
                                          size_t gn, double acc0[32], double acc1[32],
                                          double acc2[32], double acc3[32]) {
  size_t g, lane;
  for (g = 0u; g < gn; ++g) {
    const uint8_t *p = base + g * bpg;
    const double *t0 = l0 + g * 16u, *t1 = l1 + g * 16u;
    const double *t2 = l2 + g * 16u, *t3 = l3 + g * 16u;
    for (lane = 0u; lane < 16u; ++lane) {
      uint8_t b = p[lane];
      unsigned lo = (unsigned)(b & 0x0Fu);
      unsigned hi = (unsigned)(b >> 4u);
      acc0[lane] = acc0[lane] + t0[lo]; acc0[lane + 16u] = acc0[lane + 16u] + t0[hi];
      acc1[lane] = acc1[lane] + t1[lo]; acc1[lane + 16u] = acc1[lane + 16u] + t1[hi];
      acc2[lane] = acc2[lane] + t2[lo]; acc2[lane + 16u] = acc2[lane + 16u] + t2[hi];
      acc3[lane] = acc3[lane] + t3[lo]; acc3[lane + 16u] = acc3[lane + 16u] + t3[hi];
    }
  }
}

static void lut4_accum32_mb4_f32_portable(const uint8_t *base, size_t bpg,
                                          const float *l0, const float *l1,
                                          const float *l2, const float *l3,
                                          size_t gn, float acc0[32], float acc1[32],
                                          float acc2[32], float acc3[32]) {
  size_t g, lane;
  for (g = 0u; g < gn; ++g) {
    const uint8_t *p = base + g * bpg;
    const float *t0 = l0 + g * 16u, *t1 = l1 + g * 16u;
    const float *t2 = l2 + g * 16u, *t3 = l3 + g * 16u;
    for (lane = 0u; lane < 16u; ++lane) {
      uint8_t b = p[lane];
      unsigned lo = (unsigned)(b & 0x0Fu);
      unsigned hi = (unsigned)(b >> 4u);
      acc0[lane] = acc0[lane] + t0[lo]; acc0[lane + 16u] = acc0[lane + 16u] + t0[hi];
      acc1[lane] = acc1[lane] + t1[lo]; acc1[lane + 16u] = acc1[lane + 16u] + t1[hi];
      acc2[lane] = acc2[lane] + t2[lo]; acc2[lane + 16u] = acc2[lane + 16u] + t2[hi];
      acc3[lane] = acc3[lane] + t3[lo]; acc3[lane + 16u] = acc3[lane + 16u] + t3[hi];
    }
  }
}

#ifdef BCIR_AI_DISPATCH
__attribute__((target("avx2"))) static inline __m256 lut4_lookup8_f32_avx2(
    __m256 lut_lo, __m256 lut_hi, __m256i idx) {
  __m256 v_lo = _mm256_permutevar8x32_ps(lut_lo, idx);
  __m256 v_hi = _mm256_permutevar8x32_ps(lut_hi, idx);
  __m256 mask = _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28));
  return _mm256_blendv_ps(v_lo, v_hi, mask);
}

__attribute__((target("avx2"))) static void lut4_accum32_f32_avx2(
    const uint8_t *base, size_t bpg, const float *luts, size_t gn, float acc[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m256 a0 = _mm256_loadu_ps(acc);
  __m256 a1 = _mm256_loadu_ps(acc + 8);
  __m256 a2 = _mm256_loadu_ps(acc + 16);
  __m256 a3 = _mm256_loadu_ps(acc + 24);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    const float *lt = luts + g * 16u;
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m128i lo_b = _mm_and_si128(b, mask_lo);
    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(b, 4), mask_lo);
    __m256 lut_lo = _mm256_loadu_ps(lt);
    __m256 lut_hi = _mm256_loadu_ps(lt + 8);
    a0 = _mm256_add_ps(a0, lut4_lookup8_f32_avx2(lut_lo, lut_hi, _mm256_cvtepu8_epi32(lo_b)));
    a1 = _mm256_add_ps(a1, lut4_lookup8_f32_avx2(lut_lo, lut_hi,
                                                 _mm256_cvtepu8_epi32(_mm_srli_si128(lo_b, 8))));
    a2 = _mm256_add_ps(a2, lut4_lookup8_f32_avx2(lut_lo, lut_hi, _mm256_cvtepu8_epi32(hi_b)));
    a3 = _mm256_add_ps(a3, lut4_lookup8_f32_avx2(lut_lo, lut_hi,
                                                 _mm256_cvtepu8_epi32(_mm_srli_si128(hi_b, 8))));
  }
  _mm256_storeu_ps(acc, a0);
  _mm256_storeu_ps(acc + 8, a1);
  _mm256_storeu_ps(acc + 16, a2);
  _mm256_storeu_ps(acc + 24, a3);
}

__attribute__((target("avx2"))) static void lut4_accum32_mb4_f32_avx2(
    const uint8_t *base, size_t bpg,
    const float *l0, const float *l1, const float *l2, const float *l3,
    size_t gn, float acc0[32], float acc1[32], float acc2[32], float acc3[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m256 t0a = _mm256_loadu_ps(acc0),      t0b = _mm256_loadu_ps(acc0 + 8);
  __m256 t0c = _mm256_loadu_ps(acc0 + 16), t0d = _mm256_loadu_ps(acc0 + 24);
  __m256 t1a = _mm256_loadu_ps(acc1),      t1b = _mm256_loadu_ps(acc1 + 8);
  __m256 t1c = _mm256_loadu_ps(acc1 + 16), t1d = _mm256_loadu_ps(acc1 + 24);
  __m256 t2a = _mm256_loadu_ps(acc2),      t2b = _mm256_loadu_ps(acc2 + 8);
  __m256 t2c = _mm256_loadu_ps(acc2 + 16), t2d = _mm256_loadu_ps(acc2 + 24);
  __m256 t3a = _mm256_loadu_ps(acc3),      t3b = _mm256_loadu_ps(acc3 + 8);
  __m256 t3c = _mm256_loadu_ps(acc3 + 16), t3d = _mm256_loadu_ps(acc3 + 24);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m128i lo_b = _mm_and_si128(b, mask_lo);
    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(b, 4), mask_lo);
    __m256i i0 = _mm256_cvtepu8_epi32(lo_b);
    __m256i i1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_b, 8));
    __m256i i2 = _mm256_cvtepu8_epi32(hi_b);
    __m256i i3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_b, 8));
    __m256 v_lo, v_hi;
    v_lo = _mm256_loadu_ps(l0 + g * 16u); v_hi = _mm256_loadu_ps(l0 + g * 16u + 8u);
    t0a = _mm256_add_ps(t0a, lut4_lookup8_f32_avx2(v_lo, v_hi, i0));
    t0b = _mm256_add_ps(t0b, lut4_lookup8_f32_avx2(v_lo, v_hi, i1));
    t0c = _mm256_add_ps(t0c, lut4_lookup8_f32_avx2(v_lo, v_hi, i2));
    t0d = _mm256_add_ps(t0d, lut4_lookup8_f32_avx2(v_lo, v_hi, i3));
    v_lo = _mm256_loadu_ps(l1 + g * 16u); v_hi = _mm256_loadu_ps(l1 + g * 16u + 8u);
    t1a = _mm256_add_ps(t1a, lut4_lookup8_f32_avx2(v_lo, v_hi, i0));
    t1b = _mm256_add_ps(t1b, lut4_lookup8_f32_avx2(v_lo, v_hi, i1));
    t1c = _mm256_add_ps(t1c, lut4_lookup8_f32_avx2(v_lo, v_hi, i2));
    t1d = _mm256_add_ps(t1d, lut4_lookup8_f32_avx2(v_lo, v_hi, i3));
    v_lo = _mm256_loadu_ps(l2 + g * 16u); v_hi = _mm256_loadu_ps(l2 + g * 16u + 8u);
    t2a = _mm256_add_ps(t2a, lut4_lookup8_f32_avx2(v_lo, v_hi, i0));
    t2b = _mm256_add_ps(t2b, lut4_lookup8_f32_avx2(v_lo, v_hi, i1));
    t2c = _mm256_add_ps(t2c, lut4_lookup8_f32_avx2(v_lo, v_hi, i2));
    t2d = _mm256_add_ps(t2d, lut4_lookup8_f32_avx2(v_lo, v_hi, i3));
    v_lo = _mm256_loadu_ps(l3 + g * 16u); v_hi = _mm256_loadu_ps(l3 + g * 16u + 8u);
    t3a = _mm256_add_ps(t3a, lut4_lookup8_f32_avx2(v_lo, v_hi, i0));
    t3b = _mm256_add_ps(t3b, lut4_lookup8_f32_avx2(v_lo, v_hi, i1));
    t3c = _mm256_add_ps(t3c, lut4_lookup8_f32_avx2(v_lo, v_hi, i2));
    t3d = _mm256_add_ps(t3d, lut4_lookup8_f32_avx2(v_lo, v_hi, i3));
  }
  _mm256_storeu_ps(acc0, t0a); _mm256_storeu_ps(acc0 + 8, t0b);
  _mm256_storeu_ps(acc0 + 16, t0c); _mm256_storeu_ps(acc0 + 24, t0d);
  _mm256_storeu_ps(acc1, t1a); _mm256_storeu_ps(acc1 + 8, t1b);
  _mm256_storeu_ps(acc1 + 16, t1c); _mm256_storeu_ps(acc1 + 24, t1d);
  _mm256_storeu_ps(acc2, t2a); _mm256_storeu_ps(acc2 + 8, t2b);
  _mm256_storeu_ps(acc2 + 16, t2c); _mm256_storeu_ps(acc2 + 24, t2d);
  _mm256_storeu_ps(acc3, t3a); _mm256_storeu_ps(acc3 + 8, t3b);
  _mm256_storeu_ps(acc3 + 16, t3c); _mm256_storeu_ps(acc3 + 24, t3d);
}

__attribute__((target("avx2"))) static void lut4_accum32_f64_avx2(
    const uint8_t *base, size_t bpg, const double *luts, size_t gn, double acc[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m256d a0 = _mm256_loadu_pd(acc),      a1 = _mm256_loadu_pd(acc + 4);
  __m256d a2 = _mm256_loadu_pd(acc + 8),  a3 = _mm256_loadu_pd(acc + 12);
  __m256d a4 = _mm256_loadu_pd(acc + 16), a5 = _mm256_loadu_pd(acc + 20);
  __m256d a6 = _mm256_loadu_pd(acc + 24), a7 = _mm256_loadu_pd(acc + 28);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    const double *lt = luts + g * 16u;
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m128i lo_b = _mm_and_si128(b, mask_lo);
    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(b, 4), mask_lo);
    a0 = _mm256_add_pd(a0, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(lo_b), 8));
    a1 = _mm256_add_pd(a1, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(_mm_srli_si128(lo_b, 4)), 8));
    a2 = _mm256_add_pd(a2, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(_mm_srli_si128(lo_b, 8)), 8));
    a3 = _mm256_add_pd(a3, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(_mm_srli_si128(lo_b, 12)), 8));
    a4 = _mm256_add_pd(a4, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(hi_b), 8));
    a5 = _mm256_add_pd(a5, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(_mm_srli_si128(hi_b, 4)), 8));
    a6 = _mm256_add_pd(a6, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(_mm_srli_si128(hi_b, 8)), 8));
    a7 = _mm256_add_pd(a7, _mm256_i32gather_pd(lt, _mm_cvtepu8_epi32(_mm_srli_si128(hi_b, 12)), 8));
  }
  _mm256_storeu_pd(acc, a0);      _mm256_storeu_pd(acc + 4, a1);
  _mm256_storeu_pd(acc + 8, a2);  _mm256_storeu_pd(acc + 12, a3);
  _mm256_storeu_pd(acc + 16, a4); _mm256_storeu_pd(acc + 20, a5);
  _mm256_storeu_pd(acc + 24, a6); _mm256_storeu_pd(acc + 28, a7);
}

__attribute__((target("avx2"))) static void lut4_accum32_mb4_f64_avx2(
    const uint8_t *base, size_t bpg,
    const double *l0, const double *l1, const double *l2, const double *l3,
    size_t gn, double acc0[32], double acc1[32], double acc2[32], double acc3[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m128i lo_b = _mm_and_si128(b, mask_lo);
    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(b, 4), mask_lo);
    __m128i idx[8];
    const double *lts[4];
    double *accs[4];
    int tok, blk;
    idx[0] = _mm_cvtepu8_epi32(lo_b);
    idx[1] = _mm_cvtepu8_epi32(_mm_srli_si128(lo_b, 4));
    idx[2] = _mm_cvtepu8_epi32(_mm_srli_si128(lo_b, 8));
    idx[3] = _mm_cvtepu8_epi32(_mm_srli_si128(lo_b, 12));
    idx[4] = _mm_cvtepu8_epi32(hi_b);
    idx[5] = _mm_cvtepu8_epi32(_mm_srli_si128(hi_b, 4));
    idx[6] = _mm_cvtepu8_epi32(_mm_srli_si128(hi_b, 8));
    idx[7] = _mm_cvtepu8_epi32(_mm_srli_si128(hi_b, 12));
    lts[0] = l0 + g * 16u; lts[1] = l1 + g * 16u;
    lts[2] = l2 + g * 16u; lts[3] = l3 + g * 16u;
    accs[0] = acc0; accs[1] = acc1; accs[2] = acc2; accs[3] = acc3;
    for (tok = 0; tok < 4; ++tok) {
      for (blk = 0; blk < 8; ++blk) {
        double *dst = accs[tok] + 4 * blk;
        _mm256_storeu_pd(dst, _mm256_add_pd(_mm256_loadu_pd(dst),
                                            _mm256_i32gather_pd(lts[tok], idx[blk], 8)));
      }
    }
  }
}

__attribute__((target("avx512f"))) static void lut4_accum32_f32_avx512(
    const uint8_t *base, size_t bpg, const float *luts, size_t gn, float acc[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m512 a0 = _mm512_loadu_ps(acc);
  __m512 a1 = _mm512_loadu_ps(acc + 16);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m512i lo = _mm512_cvtepu8_epi32(_mm_and_si128(b, mask_lo));
    __m512i hi = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(b, 4), mask_lo));
    __m512 vlut = _mm512_loadu_ps(luts + g * 16u);
    a0 = _mm512_add_ps(a0, _mm512_permutexvar_ps(lo, vlut));
    a1 = _mm512_add_ps(a1, _mm512_permutexvar_ps(hi, vlut));
  }
  _mm512_storeu_ps(acc, a0);
  _mm512_storeu_ps(acc + 16, a1);
}

__attribute__((target("avx512f"))) static void lut4_accum32_mb4_f32_avx512(
    const uint8_t *base, size_t bpg,
    const float *l0, const float *l1, const float *l2, const float *l3,
    size_t gn, float acc0[32], float acc1[32], float acc2[32], float acc3[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m512 lo0 = _mm512_loadu_ps(acc0), hi0 = _mm512_loadu_ps(acc0 + 16);
  __m512 lo1 = _mm512_loadu_ps(acc1), hi1 = _mm512_loadu_ps(acc1 + 16);
  __m512 lo2 = _mm512_loadu_ps(acc2), hi2 = _mm512_loadu_ps(acc2 + 16);
  __m512 lo3 = _mm512_loadu_ps(acc3), hi3 = _mm512_loadu_ps(acc3 + 16);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m512i lo = _mm512_cvtepu8_epi32(_mm_and_si128(b, mask_lo));
    __m512i hi = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(b, 4), mask_lo));
    __m512 v0 = _mm512_loadu_ps(l0 + g * 16u);
    __m512 v1 = _mm512_loadu_ps(l1 + g * 16u);
    __m512 v2 = _mm512_loadu_ps(l2 + g * 16u);
    __m512 v3 = _mm512_loadu_ps(l3 + g * 16u);
    lo0 = _mm512_add_ps(lo0, _mm512_permutexvar_ps(lo, v0));
    hi0 = _mm512_add_ps(hi0, _mm512_permutexvar_ps(hi, v0));
    lo1 = _mm512_add_ps(lo1, _mm512_permutexvar_ps(lo, v1));
    hi1 = _mm512_add_ps(hi1, _mm512_permutexvar_ps(hi, v1));
    lo2 = _mm512_add_ps(lo2, _mm512_permutexvar_ps(lo, v2));
    hi2 = _mm512_add_ps(hi2, _mm512_permutexvar_ps(hi, v2));
    lo3 = _mm512_add_ps(lo3, _mm512_permutexvar_ps(lo, v3));
    hi3 = _mm512_add_ps(hi3, _mm512_permutexvar_ps(hi, v3));
  }
  _mm512_storeu_ps(acc0, lo0); _mm512_storeu_ps(acc0 + 16, hi0);
  _mm512_storeu_ps(acc1, lo1); _mm512_storeu_ps(acc1 + 16, hi1);
  _mm512_storeu_ps(acc2, lo2); _mm512_storeu_ps(acc2 + 16, hi2);
  _mm512_storeu_ps(acc3, lo3); _mm512_storeu_ps(acc3 + 16, hi3);
}

__attribute__((target("avx512f"))) static void lut4_accum32_f64_avx512(
    const uint8_t *base, size_t bpg, const double *luts, size_t gn, double acc[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m512d a0 = _mm512_loadu_pd(acc);
  __m512d a1 = _mm512_loadu_pd(acc + 8);
  __m512d a2 = _mm512_loadu_pd(acc + 16);
  __m512d a3 = _mm512_loadu_pd(acc + 24);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    const double *lt = luts + g * 16u;
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m128i lo_b = _mm_and_si128(b, mask_lo);
    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(b, 4), mask_lo);
    __m512d lut_lo = _mm512_loadu_pd(lt);
    __m512d lut_hi = _mm512_loadu_pd(lt + 8);
    a0 = _mm512_add_pd(a0, _mm512_permutex2var_pd(lut_lo, _mm512_cvtepu8_epi64(lo_b), lut_hi));
    a1 = _mm512_add_pd(a1, _mm512_permutex2var_pd(lut_lo, _mm512_cvtepu8_epi64(_mm_srli_si128(lo_b, 8)), lut_hi));
    a2 = _mm512_add_pd(a2, _mm512_permutex2var_pd(lut_lo, _mm512_cvtepu8_epi64(hi_b), lut_hi));
    a3 = _mm512_add_pd(a3, _mm512_permutex2var_pd(lut_lo, _mm512_cvtepu8_epi64(_mm_srli_si128(hi_b, 8)), lut_hi));
  }
  _mm512_storeu_pd(acc, a0);
  _mm512_storeu_pd(acc + 8, a1);
  _mm512_storeu_pd(acc + 16, a2);
  _mm512_storeu_pd(acc + 24, a3);
}

__attribute__((target("avx512f"))) static void lut4_accum32_mb4_f64_avx512(
    const uint8_t *base, size_t bpg,
    const double *l0, const double *l1, const double *l2, const double *l3,
    size_t gn, double acc0[32], double acc1[32], double acc2[32], double acc3[32]) {
  const __m128i mask_lo = _mm_set1_epi8(0x0F);
  __m512d t0a = _mm512_loadu_pd(acc0),      t0b = _mm512_loadu_pd(acc0 + 8);
  __m512d t0c = _mm512_loadu_pd(acc0 + 16), t0d = _mm512_loadu_pd(acc0 + 24);
  __m512d t1a = _mm512_loadu_pd(acc1),      t1b = _mm512_loadu_pd(acc1 + 8);
  __m512d t1c = _mm512_loadu_pd(acc1 + 16), t1d = _mm512_loadu_pd(acc1 + 24);
  __m512d t2a = _mm512_loadu_pd(acc2),      t2b = _mm512_loadu_pd(acc2 + 8);
  __m512d t2c = _mm512_loadu_pd(acc2 + 16), t2d = _mm512_loadu_pd(acc2 + 24);
  __m512d t3a = _mm512_loadu_pd(acc3),      t3b = _mm512_loadu_pd(acc3 + 8);
  __m512d t3c = _mm512_loadu_pd(acc3 + 16), t3d = _mm512_loadu_pd(acc3 + 24);
  size_t g;
  for (g = 0u; g < gn; ++g) {
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(base + g * bpg));
    __m128i lo_b = _mm_and_si128(b, mask_lo);
    __m128i hi_b = _mm_and_si128(_mm_srli_epi16(b, 4), mask_lo);
    __m512i i0 = _mm512_cvtepu8_epi64(lo_b);
    __m512i i1 = _mm512_cvtepu8_epi64(_mm_srli_si128(lo_b, 8));
    __m512i i2 = _mm512_cvtepu8_epi64(hi_b);
    __m512i i3 = _mm512_cvtepu8_epi64(_mm_srli_si128(hi_b, 8));
    __m512d vlo, vhi;
    vlo = _mm512_loadu_pd(l0 + g * 16u); vhi = _mm512_loadu_pd(l0 + g * 16u + 8u);
    t0a = _mm512_add_pd(t0a, _mm512_permutex2var_pd(vlo, i0, vhi));
    t0b = _mm512_add_pd(t0b, _mm512_permutex2var_pd(vlo, i1, vhi));
    t0c = _mm512_add_pd(t0c, _mm512_permutex2var_pd(vlo, i2, vhi));
    t0d = _mm512_add_pd(t0d, _mm512_permutex2var_pd(vlo, i3, vhi));
    vlo = _mm512_loadu_pd(l1 + g * 16u); vhi = _mm512_loadu_pd(l1 + g * 16u + 8u);
    t1a = _mm512_add_pd(t1a, _mm512_permutex2var_pd(vlo, i0, vhi));
    t1b = _mm512_add_pd(t1b, _mm512_permutex2var_pd(vlo, i1, vhi));
    t1c = _mm512_add_pd(t1c, _mm512_permutex2var_pd(vlo, i2, vhi));
    t1d = _mm512_add_pd(t1d, _mm512_permutex2var_pd(vlo, i3, vhi));
    vlo = _mm512_loadu_pd(l2 + g * 16u); vhi = _mm512_loadu_pd(l2 + g * 16u + 8u);
    t2a = _mm512_add_pd(t2a, _mm512_permutex2var_pd(vlo, i0, vhi));
    t2b = _mm512_add_pd(t2b, _mm512_permutex2var_pd(vlo, i1, vhi));
    t2c = _mm512_add_pd(t2c, _mm512_permutex2var_pd(vlo, i2, vhi));
    t2d = _mm512_add_pd(t2d, _mm512_permutex2var_pd(vlo, i3, vhi));
    vlo = _mm512_loadu_pd(l3 + g * 16u); vhi = _mm512_loadu_pd(l3 + g * 16u + 8u);
    t3a = _mm512_add_pd(t3a, _mm512_permutex2var_pd(vlo, i0, vhi));
    t3b = _mm512_add_pd(t3b, _mm512_permutex2var_pd(vlo, i1, vhi));
    t3c = _mm512_add_pd(t3c, _mm512_permutex2var_pd(vlo, i2, vhi));
    t3d = _mm512_add_pd(t3d, _mm512_permutex2var_pd(vlo, i3, vhi));
  }
  _mm512_storeu_pd(acc0, t0a); _mm512_storeu_pd(acc0 + 8, t0b);
  _mm512_storeu_pd(acc0 + 16, t0c); _mm512_storeu_pd(acc0 + 24, t0d);
  _mm512_storeu_pd(acc1, t1a); _mm512_storeu_pd(acc1 + 8, t1b);
  _mm512_storeu_pd(acc1 + 16, t1c); _mm512_storeu_pd(acc1 + 24, t1d);
  _mm512_storeu_pd(acc2, t2a); _mm512_storeu_pd(acc2 + 8, t2b);
  _mm512_storeu_pd(acc2 + 16, t2c); _mm512_storeu_pd(acc2 + 24, t2d);
  _mm512_storeu_pd(acc3, t3a); _mm512_storeu_pd(acc3 + 8, t3b);
  _mm512_storeu_pd(acc3 + 16, t3c); _mm512_storeu_pd(acc3 + 24, t3d);
}
#endif

static void lut4_accum32_f64(int inst, const uint8_t *base, size_t bpg,
                             const double *luts, size_t gn, double acc[32]) {
#ifdef BCIR_AI_DISPATCH
  if (inst == 4) { lut4_accum32_f64_avx512(base, bpg, luts, gn, acc); return; }
  if (inst == 2) { lut4_accum32_f64_avx2(base, bpg, luts, gn, acc); return; }
#else
  (void)inst;
#endif
  lut4_accum32_f64_portable(base, bpg, luts, gn, acc);
}

static void lut4_accum32_f32(int inst, const uint8_t *base, size_t bpg,
                             const float *luts, size_t gn, float acc[32]) {
#ifdef BCIR_AI_DISPATCH
  if (inst == 4) { lut4_accum32_f32_avx512(base, bpg, luts, gn, acc); return; }
  if (inst == 2) { lut4_accum32_f32_avx2(base, bpg, luts, gn, acc); return; }
#else
  (void)inst;
#endif
  lut4_accum32_f32_portable(base, bpg, luts, gn, acc);
}

static void lut4_accum32_mb4_f64(int inst, const uint8_t *base, size_t bpg,
                                 const double *l0, const double *l1,
                                 const double *l2, const double *l3,
                                 size_t gn, double acc0[32], double acc1[32],
                                 double acc2[32], double acc3[32]) {
#ifdef BCIR_AI_DISPATCH
  if (inst == 4) {
    lut4_accum32_mb4_f64_avx512(base, bpg, l0, l1, l2, l3, gn, acc0, acc1, acc2, acc3);
    return;
  }
  if (inst == 2) {
    lut4_accum32_mb4_f64_avx2(base, bpg, l0, l1, l2, l3, gn, acc0, acc1, acc2, acc3);
    return;
  }
#else
  (void)inst;
#endif
  lut4_accum32_mb4_f64_portable(base, bpg, l0, l1, l2, l3, gn, acc0, acc1, acc2, acc3);
}

static void lut4_accum32_mb4_f32(int inst, const uint8_t *base, size_t bpg,
                                 const float *l0, const float *l1,
                                 const float *l2, const float *l3,
                                 size_t gn, float acc0[32], float acc1[32],
                                 float acc2[32], float acc3[32]) {
#ifdef BCIR_AI_DISPATCH
  if (inst == 4) {
    lut4_accum32_mb4_f32_avx512(base, bpg, l0, l1, l2, l3, gn, acc0, acc1, acc2, acc3);
    return;
  }
  if (inst == 2) {
    lut4_accum32_mb4_f32_avx2(base, bpg, l0, l1, l2, l3, gn, acc0, acc1, acc2, acc3);
    return;
  }
#else
  (void)inst;
#endif
  lut4_accum32_mb4_f32_portable(base, bpg, l0, l1, l2, l3, gn, acc0, acc1, acc2, acc3);
}

enum { LUT4_V_GTILE = 32u, LUT4_MB4_MAX_RANK = 256u, LUT4_MB4_V_GTILE = 16u };

static int lut4_matvec_f64_core(int inst, const double *input, size_t input_count,
                                const bcir_ai_lut4_path *paths, size_t path_count,
                                size_t output_count, size_t col_begin, size_t col_end,
                                double *output) {
  double z_buf[BCIR_AI_LUT4_MAX_RANK];
  double lut_v[LUT4_V_GTILE * 16u];
  double lut_u[(BCIR_AI_LUT4_MAX_RANK / 4u) * 16u];
  size_t p, j;
  for (j = col_begin; j < col_end; ++j) output[j] = 0.0;
  for (p = 0u; p < path_count; ++p) {
    const bcir_ai_lut4_path *pt = &paths[p];
    size_t rank = pt->rank;
    size_t gv_total = (input_count + 3u) >> 2u;
    size_t r32_v = (rank + 31u) & ~(size_t)31u;
    size_t bpg_v = r32_v >> 1u;
    size_t gu_total = (rank + 3u) >> 2u;
    size_t r32_u = (output_count + 31u) & ~(size_t)31u;
    size_t bpg_u = r32_u >> 1u;
    size_t g0, r0, k;
    memset(z_buf, 0, r32_v * sizeof(double));
    for (g0 = 0u; g0 < gv_total; g0 += LUT4_V_GTILE) {
      size_t gn = gv_total - g0 < LUT4_V_GTILE ? gv_total - g0 : LUT4_V_GTILE;
      size_t t;
      for (t = 0u; t < gn; ++t) {
        size_t c0 = (g0 + t) << 2u;
        double x0 = c0 < input_count ? input[c0] * lut4_fp16_f64(pt->g_fp16_le, c0) : 0.0;
        double x1 = c0 + 1u < input_count ? input[c0 + 1u] * lut4_fp16_f64(pt->g_fp16_le, c0 + 1u) : 0.0;
        double x2 = c0 + 2u < input_count ? input[c0 + 2u] * lut4_fp16_f64(pt->g_fp16_le, c0 + 2u) : 0.0;
        double x3 = c0 + 3u < input_count ? input[c0 + 3u] * lut4_fp16_f64(pt->g_fp16_le, c0 + 3u) : 0.0;
        lut4_build_f64(x0, x1, x2, x3, lut_v + t * 16u);
      }
      for (r0 = 0u; r0 < r32_v; r0 += 32u)
        lut4_accum32_f64(inst, pt->v_packed + g0 * bpg_v + (r0 >> 1u), bpg_v, lut_v, gn, z_buf + r0);
    }
    for (k = 0u; k < rank; ++k) z_buf[k] = z_buf[k] * lut4_fp16_f64(pt->l_fp16_le, k);
    for (g0 = 0u; g0 < gu_total; ++g0) {
      size_t c0 = g0 << 2u;
      double x0 = c0 < rank ? z_buf[c0] : 0.0;
      double x1 = c0 + 1u < rank ? z_buf[c0 + 1u] : 0.0;
      double x2 = c0 + 2u < rank ? z_buf[c0 + 2u] : 0.0;
      double x3 = c0 + 3u < rank ? z_buf[c0 + 3u] : 0.0;
      lut4_build_f64(x0, x1, x2, x3, lut_u + g0 * 16u);
    }
    {
      size_t blk_begin = col_begin & ~(size_t)31u;
      size_t blk_end = (col_end + 31u) & ~(size_t)31u;
      for (r0 = blk_begin; r0 < blk_end; r0 += 32u) {
        double acc_blk[32];
        size_t lo = r0 > col_begin ? r0 : col_begin;
        size_t hi = r0 + 32u < col_end ? r0 + 32u : col_end;
        memset(acc_blk, 0, sizeof acc_blk);
        lut4_accum32_f64(inst, pt->u_packed + (r0 >> 1u), bpg_u, lut_u, gu_total, acc_blk);
        for (j = lo; j < hi; ++j)
          output[j] = output[j] + acc_blk[j - r0] * lut4_fp16_f64(pt->h_fp16_le, j);
      }
    }
  }
  for (j = col_begin; j < col_end; ++j) {
    if (!isfinite(output[j])) {
      memset(output + col_begin, 0, (col_end - col_begin) * sizeof *output);
      return BCIR_AI_INVALID_NUMBER;
    }
  }
  return BCIR_AI_OK;
}

static int lut4_matvec_f32_core(int inst, const float *input, size_t input_count,
                                const bcir_ai_lut4_path *paths, size_t path_count,
                                size_t output_count, size_t col_begin, size_t col_end,
                                float *output) {
  float z_buf[BCIR_AI_LUT4_MAX_RANK];
  float lut_v[LUT4_V_GTILE * 16u];
  float lut_u[(BCIR_AI_LUT4_MAX_RANK / 4u) * 16u];
  size_t p, j;
  for (j = col_begin; j < col_end; ++j) output[j] = 0.0f;
  for (p = 0u; p < path_count; ++p) {
    const bcir_ai_lut4_path *pt = &paths[p];
    size_t rank = pt->rank;
    size_t gv_total = (input_count + 3u) >> 2u;
    size_t r32_v = (rank + 31u) & ~(size_t)31u;
    size_t bpg_v = r32_v >> 1u;
    size_t gu_total = (rank + 3u) >> 2u;
    size_t r32_u = (output_count + 31u) & ~(size_t)31u;
    size_t bpg_u = r32_u >> 1u;
    size_t g0, r0, k;
    memset(z_buf, 0, r32_v * sizeof(float));
    for (g0 = 0u; g0 < gv_total; g0 += LUT4_V_GTILE) {
      size_t gn = gv_total - g0 < LUT4_V_GTILE ? gv_total - g0 : LUT4_V_GTILE;
      size_t t;
      for (t = 0u; t < gn; ++t) {
        size_t c0 = (g0 + t) << 2u;
        float x0 = c0 < input_count ? input[c0] * lut4_fp16_f32(pt->g_fp16_le, c0) : 0.0f;
        float x1 = c0 + 1u < input_count ? input[c0 + 1u] * lut4_fp16_f32(pt->g_fp16_le, c0 + 1u) : 0.0f;
        float x2 = c0 + 2u < input_count ? input[c0 + 2u] * lut4_fp16_f32(pt->g_fp16_le, c0 + 2u) : 0.0f;
        float x3 = c0 + 3u < input_count ? input[c0 + 3u] * lut4_fp16_f32(pt->g_fp16_le, c0 + 3u) : 0.0f;
        lut4_build_f32(x0, x1, x2, x3, lut_v + t * 16u);
      }
      for (r0 = 0u; r0 < r32_v; r0 += 32u)
        lut4_accum32_f32(inst, pt->v_packed + g0 * bpg_v + (r0 >> 1u), bpg_v, lut_v, gn, z_buf + r0);
    }
    for (k = 0u; k < rank; ++k) z_buf[k] = z_buf[k] * lut4_fp16_f32(pt->l_fp16_le, k);
    for (g0 = 0u; g0 < gu_total; ++g0) {
      size_t c0 = g0 << 2u;
      float x0 = c0 < rank ? z_buf[c0] : 0.0f;
      float x1 = c0 + 1u < rank ? z_buf[c0 + 1u] : 0.0f;
      float x2 = c0 + 2u < rank ? z_buf[c0 + 2u] : 0.0f;
      float x3 = c0 + 3u < rank ? z_buf[c0 + 3u] : 0.0f;
      lut4_build_f32(x0, x1, x2, x3, lut_u + g0 * 16u);
    }
    {
      size_t blk_begin = col_begin & ~(size_t)31u;
      size_t blk_end = (col_end + 31u) & ~(size_t)31u;
      for (r0 = blk_begin; r0 < blk_end; r0 += 32u) {
        float acc_blk[32];
        size_t lo = r0 > col_begin ? r0 : col_begin;
        size_t hi = r0 + 32u < col_end ? r0 + 32u : col_end;
        memset(acc_blk, 0, sizeof acc_blk);
        lut4_accum32_f32(inst, pt->u_packed + (r0 >> 1u), bpg_u, lut_u, gu_total, acc_blk);
        for (j = lo; j < hi; ++j)
          output[j] = output[j] + acc_blk[j - r0] * lut4_fp16_f32(pt->h_fp16_le, j);
      }
    }
  }
  for (j = col_begin; j < col_end; ++j) {
    if (!isfinite(output[j])) {
      memset(output + col_begin, 0, (col_end - col_begin) * sizeof *output);
      return BCIR_AI_INVALID_NUMBER;
    }
  }
  return BCIR_AI_OK;
}

int bcir_ai_lut4_matvec_f64_columns_instance(
    int instance, const double *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, size_t column_begin, size_t column_end,
    double *output) {
  instance = q8_instance(instance);
  if (!instance || !lut4_shape_basic(input_count, output_count, paths, path_count) ||
      !q8_input_validate(input, input_count) || output == NULL ||
      column_begin >= column_end || column_end > output_count)
    return BCIR_AI_INVALID_ARGUMENT;
  LUT4_SAW(instance);
  return lut4_matvec_f64_core(instance, input, input_count, paths, path_count,
                              output_count, column_begin, column_end, output);
}

int bcir_ai_lut4_matvec_f64_columns(
    const double *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, size_t column_begin, size_t column_end,
    double *output) {
  return bcir_ai_lut4_matvec_f64_columns_instance(
      0, input, input_count, paths, path_count, output_count, column_begin, column_end, output);
}

int bcir_ai_lut4_matvec_f64_instance(
    int instance, const double *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, double *output) {
  return bcir_ai_lut4_matvec_f64_columns_instance(
      instance, input, input_count, paths, path_count, output_count, 0u, output_count, output);
}

int bcir_ai_lut4_matvec_f64_prevalidated(
    const double *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, double *output) {
  return bcir_ai_lut4_matvec_f64_columns_instance(
      0, input, input_count, paths, path_count, output_count, 0u, output_count, output);
}

int bcir_ai_lut4_matvec_f64(
    const double *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, double *output) {
  int status = bcir_ai_lut4_paths_validate(input_count, output_count, paths, path_count);
  if (status != BCIR_AI_OK) return status;
  return bcir_ai_lut4_matvec_f64_prevalidated(input, input_count, paths, path_count,
                                              output_count, output);
}

int bcir_ai_lut4_matvec_f32_columns_instance(
    int instance, const float *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, size_t column_begin, size_t column_end,
    float *output) {
  instance = q8_instance(instance);
  if (!instance || !lut4_shape_basic(input_count, output_count, paths, path_count) ||
      !lut4_finite_f32(input, input_count) || output == NULL ||
      column_begin >= column_end || column_end > output_count)
    return BCIR_AI_INVALID_ARGUMENT;
  LUT4_SAW(instance);
  return lut4_matvec_f32_core(instance, input, input_count, paths, path_count,
                              output_count, column_begin, column_end, output);
}

int bcir_ai_lut4_matvec_f32_instance(
    int instance, const float *input, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, float *output) {
  return bcir_ai_lut4_matvec_f32_columns_instance(
      instance, input, input_count, paths, path_count, output_count, 0u, output_count, output);
}

int bcir_ai_lut4_matmul_mb4_f64_instance(
    int instance, const double *inputs, size_t token_count, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, double *outputs) {
  size_t tok0 = 0u, p, max_rank = 0u;
  instance = q8_instance(instance);
  if (!instance || !lut4_shape_basic(input_count, output_count, paths, path_count) ||
      token_count == 0u || token_count > (size_t)BCIR_AI_MAX_ELEMENTS / output_count ||
      token_count > (size_t)BCIR_AI_MAX_ELEMENTS / input_count ||
      !q8_input_validate(inputs, token_count * input_count) || outputs == NULL)
    return BCIR_AI_INVALID_ARGUMENT;
  LUT4_SAW(instance);
  for (p = 0u; p < path_count; ++p)
    if (paths[p].rank > max_rank) max_rank = paths[p].rank;
  if (max_rank <= LUT4_MB4_MAX_RANK) {
    double z4[4][LUT4_MB4_MAX_RANK];
    double lut_v[4][LUT4_MB4_V_GTILE * 16u];
    double lut_u[4][(LUT4_MB4_MAX_RANK / 4u) * 16u];
    for (; tok0 + 4u <= token_count; tok0 += 4u) {
      const double *x0 = inputs + (tok0 + 0u) * input_count;
      const double *x1 = inputs + (tok0 + 1u) * input_count;
      const double *x2 = inputs + (tok0 + 2u) * input_count;
      const double *x3 = inputs + (tok0 + 3u) * input_count;
      double *y0 = outputs + (tok0 + 0u) * output_count;
      double *y1 = outputs + (tok0 + 1u) * output_count;
      double *y2 = outputs + (tok0 + 2u) * output_count;
      double *y3 = outputs + (tok0 + 3u) * output_count;
      size_t j;
      for (j = 0u; j < output_count; ++j) y0[j] = y1[j] = y2[j] = y3[j] = 0.0;
      for (p = 0u; p < path_count; ++p) {
        const bcir_ai_lut4_path *pt = &paths[p];
        size_t rank = pt->rank;
        size_t gv_total = (input_count + 3u) >> 2u;
        size_t r32_v = (rank + 31u) & ~(size_t)31u;
        size_t bpg_v = r32_v >> 1u;
        size_t gu_total = (rank + 3u) >> 2u;
        size_t r32_u = (output_count + 31u) & ~(size_t)31u;
        size_t bpg_u = r32_u >> 1u;
        size_t g0, r0, k, tok;
        const double *xs[4];
        xs[0] = x0; xs[1] = x1; xs[2] = x2; xs[3] = x3;
        for (tok = 0u; tok < 4u; ++tok) memset(z4[tok], 0, r32_v * sizeof(double));
        for (g0 = 0u; g0 < gv_total; g0 += LUT4_MB4_V_GTILE) {
          size_t gn = gv_total - g0 < LUT4_MB4_V_GTILE ? gv_total - g0 : LUT4_MB4_V_GTILE;
          size_t t;
          for (t = 0u; t < gn; ++t) {
            size_t c0 = (g0 + t) << 2u;
            double s0 = c0 < input_count ? lut4_fp16_f64(pt->g_fp16_le, c0) : 0.0;
            double s1 = c0 + 1u < input_count ? lut4_fp16_f64(pt->g_fp16_le, c0 + 1u) : 0.0;
            double s2 = c0 + 2u < input_count ? lut4_fp16_f64(pt->g_fp16_le, c0 + 2u) : 0.0;
            double s3 = c0 + 3u < input_count ? lut4_fp16_f64(pt->g_fp16_le, c0 + 3u) : 0.0;
            for (tok = 0u; tok < 4u; ++tok) {
              double a0 = c0 < input_count ? xs[tok][c0] * s0 : 0.0;
              double a1 = c0 + 1u < input_count ? xs[tok][c0 + 1u] * s1 : 0.0;
              double a2 = c0 + 2u < input_count ? xs[tok][c0 + 2u] * s2 : 0.0;
              double a3 = c0 + 3u < input_count ? xs[tok][c0 + 3u] * s3 : 0.0;
              lut4_build_f64(a0, a1, a2, a3, lut_v[tok] + t * 16u);
            }
          }
          for (r0 = 0u; r0 < r32_v; r0 += 32u)
            lut4_accum32_mb4_f64(instance, pt->v_packed + g0 * bpg_v + (r0 >> 1u), bpg_v,
                                 lut_v[0], lut_v[1], lut_v[2], lut_v[3], gn,
                                 z4[0] + r0, z4[1] + r0, z4[2] + r0, z4[3] + r0);
        }
        for (k = 0u; k < rank; ++k) {
          double lk = lut4_fp16_f64(pt->l_fp16_le, k);
          z4[0][k] = z4[0][k] * lk; z4[1][k] = z4[1][k] * lk;
          z4[2][k] = z4[2][k] * lk; z4[3][k] = z4[3][k] * lk;
        }
        for (g0 = 0u; g0 < gu_total; ++g0) {
          size_t c0 = g0 << 2u;
          for (tok = 0u; tok < 4u; ++tok) {
            double a0 = c0 < rank ? z4[tok][c0] : 0.0;
            double a1 = c0 + 1u < rank ? z4[tok][c0 + 1u] : 0.0;
            double a2 = c0 + 2u < rank ? z4[tok][c0 + 2u] : 0.0;
            double a3 = c0 + 3u < rank ? z4[tok][c0 + 3u] : 0.0;
            lut4_build_f64(a0, a1, a2, a3, lut_u[tok] + g0 * 16u);
          }
        }
        for (r0 = 0u; r0 < r32_u; r0 += 32u) {
          double acc0[32], acc1[32], acc2[32], acc3[32];
          size_t hi = r0 + 32u < output_count ? r0 + 32u : output_count;
          memset(acc0, 0, sizeof acc0); memset(acc1, 0, sizeof acc1);
          memset(acc2, 0, sizeof acc2); memset(acc3, 0, sizeof acc3);
          lut4_accum32_mb4_f64(instance, pt->u_packed + (r0 >> 1u), bpg_u,
                               lut_u[0], lut_u[1], lut_u[2], lut_u[3], gu_total,
                               acc0, acc1, acc2, acc3);
          for (j = r0; j < hi; ++j) {
            double hj = lut4_fp16_f64(pt->h_fp16_le, j);
            y0[j] = y0[j] + acc0[j - r0] * hj;
            y1[j] = y1[j] + acc1[j - r0] * hj;
            y2[j] = y2[j] + acc2[j - r0] * hj;
            y3[j] = y3[j] + acc3[j - r0] * hj;
          }
        }
      }
      for (j = 0u; j < output_count; ++j) {
        if (!isfinite(y0[j]) || !isfinite(y1[j]) || !isfinite(y2[j]) || !isfinite(y3[j])) {
          memset(outputs, 0, token_count * output_count * sizeof *outputs);
          return BCIR_AI_INVALID_NUMBER;
        }
      }
    }
  }
  for (; tok0 < token_count; ++tok0) {
    int st = lut4_matvec_f64_core(instance, inputs + tok0 * input_count, input_count,
                                  paths, path_count, output_count, 0u, output_count,
                                  outputs + tok0 * output_count);
    if (st != BCIR_AI_OK) {
      memset(outputs, 0, token_count * output_count * sizeof *outputs);
      return st;
    }
  }
  return BCIR_AI_OK;
}

int bcir_ai_lut4_matmul_mb4_f32_instance(
    int instance, const float *inputs, size_t token_count, size_t input_count,
    const bcir_ai_lut4_path *paths, size_t path_count,
    size_t output_count, float *outputs) {
  size_t tok0 = 0u, p, max_rank = 0u;
  instance = q8_instance(instance);
  if (!instance || !lut4_shape_basic(input_count, output_count, paths, path_count) ||
      token_count == 0u || token_count > (size_t)BCIR_AI_MAX_ELEMENTS / output_count ||
      token_count > (size_t)BCIR_AI_MAX_ELEMENTS / input_count ||
      !lut4_finite_f32(inputs, token_count * input_count) || outputs == NULL)
    return BCIR_AI_INVALID_ARGUMENT;
  LUT4_SAW(instance);
  for (p = 0u; p < path_count; ++p)
    if (paths[p].rank > max_rank) max_rank = paths[p].rank;
  if (max_rank <= LUT4_MB4_MAX_RANK) {
    float z4[4][LUT4_MB4_MAX_RANK];
    float lut_v[4][LUT4_MB4_V_GTILE * 16u];
    float lut_u[4][(LUT4_MB4_MAX_RANK / 4u) * 16u];
    for (; tok0 + 4u <= token_count; tok0 += 4u) {
      const float *x0 = inputs + (tok0 + 0u) * input_count;
      const float *x1 = inputs + (tok0 + 1u) * input_count;
      const float *x2 = inputs + (tok0 + 2u) * input_count;
      const float *x3 = inputs + (tok0 + 3u) * input_count;
      float *y0 = outputs + (tok0 + 0u) * output_count;
      float *y1 = outputs + (tok0 + 1u) * output_count;
      float *y2 = outputs + (tok0 + 2u) * output_count;
      float *y3 = outputs + (tok0 + 3u) * output_count;
      size_t j;
      for (j = 0u; j < output_count; ++j) y0[j] = y1[j] = y2[j] = y3[j] = 0.0f;
      for (p = 0u; p < path_count; ++p) {
        const bcir_ai_lut4_path *pt = &paths[p];
        size_t rank = pt->rank;
        size_t gv_total = (input_count + 3u) >> 2u;
        size_t r32_v = (rank + 31u) & ~(size_t)31u;
        size_t bpg_v = r32_v >> 1u;
        size_t gu_total = (rank + 3u) >> 2u;
        size_t r32_u = (output_count + 31u) & ~(size_t)31u;
        size_t bpg_u = r32_u >> 1u;
        size_t g0, r0, k, tok;
        const float *xs[4];
        xs[0] = x0; xs[1] = x1; xs[2] = x2; xs[3] = x3;
        for (tok = 0u; tok < 4u; ++tok) memset(z4[tok], 0, r32_v * sizeof(float));
        for (g0 = 0u; g0 < gv_total; g0 += LUT4_MB4_V_GTILE) {
          size_t gn = gv_total - g0 < LUT4_MB4_V_GTILE ? gv_total - g0 : LUT4_MB4_V_GTILE;
          size_t t;
          for (t = 0u; t < gn; ++t) {
            size_t c0 = (g0 + t) << 2u;
            float s0 = c0 < input_count ? lut4_fp16_f32(pt->g_fp16_le, c0) : 0.0f;
            float s1 = c0 + 1u < input_count ? lut4_fp16_f32(pt->g_fp16_le, c0 + 1u) : 0.0f;
            float s2 = c0 + 2u < input_count ? lut4_fp16_f32(pt->g_fp16_le, c0 + 2u) : 0.0f;
            float s3 = c0 + 3u < input_count ? lut4_fp16_f32(pt->g_fp16_le, c0 + 3u) : 0.0f;
            for (tok = 0u; tok < 4u; ++tok) {
              float a0 = c0 < input_count ? xs[tok][c0] * s0 : 0.0f;
              float a1 = c0 + 1u < input_count ? xs[tok][c0 + 1u] * s1 : 0.0f;
              float a2 = c0 + 2u < input_count ? xs[tok][c0 + 2u] * s2 : 0.0f;
              float a3 = c0 + 3u < input_count ? xs[tok][c0 + 3u] * s3 : 0.0f;
              lut4_build_f32(a0, a1, a2, a3, lut_v[tok] + t * 16u);
            }
          }
          for (r0 = 0u; r0 < r32_v; r0 += 32u)
            lut4_accum32_mb4_f32(instance, pt->v_packed + g0 * bpg_v + (r0 >> 1u), bpg_v,
                                 lut_v[0], lut_v[1], lut_v[2], lut_v[3], gn,
                                 z4[0] + r0, z4[1] + r0, z4[2] + r0, z4[3] + r0);
        }
        for (k = 0u; k < rank; ++k) {
          float lk = lut4_fp16_f32(pt->l_fp16_le, k);
          z4[0][k] = z4[0][k] * lk; z4[1][k] = z4[1][k] * lk;
          z4[2][k] = z4[2][k] * lk; z4[3][k] = z4[3][k] * lk;
        }
        for (g0 = 0u; g0 < gu_total; ++g0) {
          size_t c0 = g0 << 2u;
          for (tok = 0u; tok < 4u; ++tok) {
            float a0 = c0 < rank ? z4[tok][c0] : 0.0f;
            float a1 = c0 + 1u < rank ? z4[tok][c0 + 1u] : 0.0f;
            float a2 = c0 + 2u < rank ? z4[tok][c0 + 2u] : 0.0f;
            float a3 = c0 + 3u < rank ? z4[tok][c0 + 3u] : 0.0f;
            lut4_build_f32(a0, a1, a2, a3, lut_u[tok] + g0 * 16u);
          }
        }
        for (r0 = 0u; r0 < r32_u; r0 += 32u) {
          float acc0[32], acc1[32], acc2[32], acc3[32];
          size_t hi = r0 + 32u < output_count ? r0 + 32u : output_count;
          memset(acc0, 0, sizeof acc0); memset(acc1, 0, sizeof acc1);
          memset(acc2, 0, sizeof acc2); memset(acc3, 0, sizeof acc3);
          lut4_accum32_mb4_f32(instance, pt->u_packed + (r0 >> 1u), bpg_u,
                               lut_u[0], lut_u[1], lut_u[2], lut_u[3], gu_total,
                               acc0, acc1, acc2, acc3);
          for (j = r0; j < hi; ++j) {
            float hj = lut4_fp16_f32(pt->h_fp16_le, j);
            y0[j] = y0[j] + acc0[j - r0] * hj;
            y1[j] = y1[j] + acc1[j - r0] * hj;
            y2[j] = y2[j] + acc2[j - r0] * hj;
            y3[j] = y3[j] + acc3[j - r0] * hj;
          }
        }
      }
      for (j = 0u; j < output_count; ++j) {
        if (!isfinite(y0[j]) || !isfinite(y1[j]) || !isfinite(y2[j]) || !isfinite(y3[j])) {
          memset(outputs, 0, token_count * output_count * sizeof *outputs);
          return BCIR_AI_INVALID_NUMBER;
        }
      }
    }
  }
  for (; tok0 < token_count; ++tok0) {
    int st = lut4_matvec_f32_core(instance, inputs + tok0 * input_count, input_count,
                                  paths, path_count, output_count, 0u, output_count,
                                  outputs + tok0 * output_count);
    if (st != BCIR_AI_OK) {
      memset(outputs, 0, token_count * output_count * sizeof *outputs);
      return st;
    }
  }
  return BCIR_AI_OK;
}
