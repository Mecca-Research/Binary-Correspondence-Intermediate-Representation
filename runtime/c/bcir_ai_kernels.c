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
 * portable loop's. */
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
/* The matvec a row at a time, each row's columns in runs of one group (any shape). */
static void q8_matvec_runs(int instance, const double *input, size_t input_count, const uint8_t *codes,
                           const uint8_t *exponents_le, size_t output_count, size_t group_size,
                           double *output) {
  size_t input_index, output_index, index = 0u;
  for (output_index = 0u; output_index < output_count; ++output_index) output[output_index] = 0.0;
  for (input_index = 0u; input_index < input_count; ++input_index) {
    for (output_index = 0u; output_index < output_count;) {
      size_t group = index / group_size, left = (group + 1u) * group_size - index;
      size_t n = output_count - output_index < left ? output_count - output_index : left;
      double scale = q8_scale(exponents_le, group);
      if (instance == 4) q8_run_avx512(output + output_index, input[input_index], codes + index, n, scale);
      else q8_run_avx2(output + output_index, input[input_index], codes + index, n, scale);
      output_index += n;
      index += n;
    }
  }
}
/* Q8-BOUND's runs: the same runs with x * s formed once a run */
static void q8_matvec_runs_xs(int instance, const double *input, size_t input_count, const uint8_t *codes,
                              const uint8_t *exponents_le, size_t output_count, size_t group_size,
                              double *output) {
  size_t input_index, output_index, index = 0u;
  for (output_index = 0u; output_index < output_count; ++output_index) output[output_index] = 0.0;
  for (input_index = 0u; input_index < input_count; ++input_index) {
    double x = input[input_index];
    for (output_index = 0u; output_index < output_count;) {
      size_t group = index / group_size, left = (group + 1u) * group_size - index;
      size_t n = output_count - output_index < left ? output_count - output_index : left;
      double xs = x * q8_scale(exponents_le, group);
      if (instance == 4) q8_run_avx512_xs(output + output_index, xs, codes + index, n);
      else q8_run_avx2_xs(output + output_index, xs, codes + index, n);
      output_index += n;
      index += n;
    }
  }
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
/* Q8-PAIR's kernels: the matvec two input rows a pass where eight columns' terms lie in one group. Each row's scale
 * is read again at the step where its group changes; an odd last row runs alone. A row's state: its codes, its first
 * element, the column where its group ends, its input, its group's scale and the multiplier its terms take (x * s
 * where Q8-BOUND proves the call, x as before elsewhere). */
#define Q8_PAIR_ROW(r, row, VT, SET1, XS)                                                                \
  const uint8_t *c##r = codes + (row) * output_count;                                                    \
  size_t e##r = (row) * output_count, n##r = ((((e##r) >> shift) + 1u) << shift) - e##r;                 \
  double x##r = input[row], s##r = q8_scale(exponents_le, e##r >> shift);                                \
  VT m##r = SET1((XS) ? x##r * s##r : x##r)
#define Q8_PAIR_TURN(r, SET1, XS)                                                                        \
  if (k == n##r) {                                                                                       \
    s##r = q8_scale(exponents_le, (e##r + k) >> shift);                                                  \
    if (XS) m##r = SET1(x##r * s##r);                                                                    \
    n##r += group_size;                                                                                  \
  }
#define Q8_PAIR_KERNEL(NAME, TARGET, VT, LANES, LOADU, STOREU, ADD, SET1, TERM, XS)                       \
  __attribute__((target(TARGET))) static void NAME(const double *input, size_t input_count,              \
      const uint8_t *codes, const uint8_t *exponents_le, size_t output_count, size_t group_size,        \
      double *output) {                                                                                  \
    unsigned shift = 0u;                                                                                 \
    size_t i, k;                                                                                         \
    while (((size_t)1 << shift) < group_size) ++shift;                                                  \
    for (k = 0u; k < output_count; ++k) output[k] = 0.0;                                                 \
    for (i = 0u; i + 1u < input_count; i += 2u) {                                                        \
      Q8_PAIR_ROW(0, i, VT, SET1, XS);                                                                   \
      Q8_PAIR_ROW(1, i + 1u, VT, SET1, XS);                                                              \
      for (k = 0u; k < output_count; k += (LANES)) {                                                     \
        Q8_PAIR_TURN(0, SET1, XS)                                                                        \
        Q8_PAIR_TURN(1, SET1, XS)                                                                        \
        STOREU(output + k, ADD(ADD(LOADU(output + k), TERM(0)), TERM(1)));                               \
      }                                                                                                  \
    }                                                                                                    \
    if (i < input_count) {                                                                               \
      Q8_PAIR_ROW(0, i, VT, SET1, XS);                                                                   \
      for (k = 0u; k < output_count; k += (LANES)) {                                                     \
        Q8_PAIR_TURN(0, SET1, XS)                                                                        \
        STOREU(output + k, ADD(LOADU(output + k), TERM(0)));                                             \
      }                                                                                                  \
    }                                                                                                    \
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
#undef Q8_PAIR_TURN
#undef Q8_PAIR_ROW
/* Q8-BOUND's ranges, read once a call: every exponent of its groups in [-300, 300] -- (e + 300) as an unsigned
 * 16-bit value at most 600 exactly there -- and every nonzero |x| in [2^-722, 2^724). 1 where both hold. The inputs
 * are finite (every entry checks them before a kernel runs). */
__attribute__((target("avx2"))) static int q8_call_exact(const double *input, size_t input_count,
                                                          const uint8_t *exponents_le, size_t groups) {
  const __m256i bias = _mm256_set1_epi16((short)-BCIR_AI_EXP_MIN);
  const __m256d magnitude = _mm256_set1_pd(-0.0), low = _mm256_set1_pd(0x1p-722), high = _mm256_set1_pd(0x1p724);
  const __m256d zero = _mm256_setzero_pd();
  __m256i top = _mm256_setzero_si256();
  __m256d outside = zero;
  uint16_t lane[16];
  size_t g = 0u, i = 0u;
  int k;
  for (; g + 16u <= groups; g += 16u) {
    __m256i e = _mm256_loadu_si256((const __m256i *)(const void *)(exponents_le + 2u * g));
    top = _mm256_max_epu16(top, _mm256_add_epi16(e, bias));
  }
  _mm256_storeu_si256((__m256i *)(void *)lane, top);
  for (k = 0; k < 16; ++k)
    if (lane[k] > BCIR_AI_EXP_MAX - BCIR_AI_EXP_MIN) return 0;
  for (; g < groups; ++g) {
    int e = exponent_le(exponents_le, g);
    if (e < BCIR_AI_EXP_MIN || e > BCIR_AI_EXP_MAX) return 0;
  }
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
/* Q8-BOUND where the call's ranges prove every x * s exact, the terms as before elsewhere; Q8-PAIR where eight columns'
 * terms lie in one group, the runs elsewhere */
static void q8_matvec_lanes(int instance, const double *input, size_t input_count, const uint8_t *codes,
                            const uint8_t *exponents_le, size_t output_count, size_t group_size,
                            double *output) {
  int pair = output_count % 8u == 0u && group_size >= 8u && !(group_size & (group_size - 1u));
  Q8_PAIR_SAW(pair ? 1 : 2);
  if (q8_call_exact(input, input_count, exponents_le, 1u + (input_count * output_count - 1u) / group_size)) {
    Q8_BOUND_SAW(1);
    if (!pair)
      q8_matvec_runs_xs(instance, input, input_count, codes, exponents_le, output_count, group_size, output);
    else if (instance == 4)
      q8_matvec_pair512_xs(input, input_count, codes, exponents_le, output_count, group_size, output);
    else
      q8_matvec_pair256_xs(input, input_count, codes, exponents_le, output_count, group_size, output);
    return;
  }
  Q8_BOUND_SAW(2);
  if (!pair)
    q8_matvec_runs(instance, input, input_count, codes, exponents_le, output_count, group_size, output);
  else if (instance == 4)
    q8_matvec_pair512(input, input_count, codes, exponents_le, output_count, group_size, output);
  else
    q8_matvec_pair256(input, input_count, codes, exponents_le, output_count, group_size, output);
}
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
    q8_matvec_lanes(instance, input, input_count, codes, exponents_le, output_count, group_size, output);
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
