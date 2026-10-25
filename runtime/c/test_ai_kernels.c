/* Deterministic malformed-input and arithmetic regressions for native AI kernels. */
#include "bcir_ai_kernels.h"
#include "bcir_q4_kernel.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "AI-kernel check failed at %s:%d: %s\n", \
          __FILE__, __LINE__, #condition); \
  return 1; \
} } while (0)

static int quantization_test(void) {
  const double values[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
  const int8_t expected_q8[5] = {-64, -32, 0, 32, 64};
  const uint8_t expected_q4[3] = {UINT8_C(0xec), UINT8_C(0x20), UINT8_C(0x04)};
  int16_t exponents[2] = {111, 222};
  int8_t q8[5] = {9, 9, 9, 9, 9};
  uint8_t q4[3] = {9, 9, 9};
  double invalid[5];
  memcpy(invalid, values, sizeof invalid);
  invalid[3] = NAN;

  CHECK(bcir_ai_quantize_q8_f64(values, 5u, 3u, exponents, 2u, q8, 5u) ==
        BCIR_AI_OK);
  CHECK(exponents[0] == -6 && exponents[1] == -6);
  CHECK(memcmp(q8, expected_q8, sizeof q8) == 0);
  CHECK(bcir_ai_quantize_q4_f64(values, 5u, 3u, exponents, 2u, q4, 3u) ==
        BCIR_AI_OK);
  CHECK(exponents[0] == -2 && exponents[1] == -2);
  CHECK(memcmp(q4, expected_q4, sizeof q4) == 0);

  exponents[0] = 111; exponents[1] = 222;
  memset(q8, 9, sizeof q8);
  CHECK(bcir_ai_quantize_q8_f64(invalid, 5u, 3u, exponents, 2u, q8, 5u) ==
        BCIR_AI_INVALID_NUMBER);
  CHECK(exponents[0] == 111 && exponents[1] == 222 && q8[0] == 9 && q8[4] == 9);
  memset(q4, 9, sizeof q4);
  CHECK(bcir_ai_quantize_q4_f64(values, 5u, 3u, exponents, 1u, q4, 3u) ==
        BCIR_AI_INVALID_ARGUMENT);
  CHECK(q4[0] == 9 && q4[2] == 9);
  CHECK(bcir_ai_quantize_q8_f64(NULL, 0u, 32u, NULL, 0u, NULL, 0u) == BCIR_AI_OK);
  return 0;
}

static int q8_matrix_test(void) {
  const double input[2] = {1.0, 2.0};
  uint8_t codes[6] = {1, 2, 3, 4, 5, 6};
  uint8_t exponents[4] = {0, 0, 1, 0};
  double output[3] = {-1.0, -1.0, -1.0};
  double invalid_input[2] = {1.0, NAN};

  CHECK(bcir_ai_q8_matvec_f64(input, 2u, codes, 6u, exponents, 2u,
                              3u, 4u, output) == BCIR_AI_OK);
  CHECK(output[0] == 9.0 && output[1] == 22.0 && output[2] == 27.0);
  CHECK(bcir_ai_q8_rows_dot_f64(input, 2u, codes, 6u, exponents, 2u,
                                3u, 4u, output) == BCIR_AI_OK);
  CHECK(output[0] == 5.0 && output[1] == 11.0 && output[2] == 34.0);

  output[0] = 7.0; output[1] = 8.0; output[2] = 9.0;
  codes[4] = UINT8_C(0x80);
  CHECK(bcir_ai_q8_matvec_f64(input, 2u, codes, 6u, exponents, 2u,
                              3u, 4u, output) == BCIR_AI_INVALID_NUMBER);
  CHECK(output[0] == 7.0 && output[1] == 8.0 && output[2] == 9.0);
  codes[4] = 5;
  CHECK(bcir_ai_q8_rows_dot_f64(invalid_input, 2u, codes, 6u, exponents, 2u,
                                3u, 4u, output) == BCIR_AI_INVALID_ARGUMENT);
  CHECK(output[0] == 7.0 && output[1] == 8.0 && output[2] == 9.0);

  {
    const double huge[1] = {DBL_MAX};
    const uint8_t huge_code[1] = {127};
    const uint8_t huge_exp[2] = {44, 1}; /* +300 */
    double failed = 3.0;
    CHECK(bcir_ai_q8_matvec_f64(huge, 1u, huge_code, 1u, huge_exp, 1u,
                                1u, 1u, &failed) == BCIR_AI_INVALID_NUMBER);
    CHECK(failed == 0.0);
  }
  return 0;
}

/* Q8-LANES: every instance the host admits against the portable loops, bit for bit -- the decoder's shapes,
 * the AVX-512 block's shapes (columns a multiple of 8, a power-of-two group of at least 8) and their tail
 * blocks, ragged ones (groups straddling rows and vectors, a group not a power of two, groups under 8),
 * exponents across the format's range and past a normal double's (ldexp's own values), the code 0x80 --
 * and the refusals: a non-finite output zeroes every output whichever row or column it is, an instance
 * the host does not admit and a non-finite input write nothing. */
#ifdef BCIR_AI_INSTANCE_PROBE
int bcir_ai_q8_instance_probe(void);
#endif
static uint64_t lanes_state = UINT64_C(0x9E3779B97F4A7C15);
static uint32_t lanes_next(void) {
  lanes_state ^= lanes_state << 13;
  lanes_state ^= lanes_state >> 7;
  lanes_state ^= lanes_state << 17;
  return (uint32_t)(lanes_state >> 11);
}
static void lanes_fill(double *x, size_t in, uint8_t *codes, size_t elements, uint8_t *exps,
                       size_t groups, int wide) {
  size_t i;
  for (i = 0u; i < in; ++i)
    x[i] = ((int)(lanes_next() % 20001u) - 10000) * 1.37e-4 * ((lanes_next() & 7u) ? 1.0 : 1e-3);
  for (i = 0u; i < elements; ++i) codes[i] = (uint8_t)lanes_next();
  for (i = 0u; i < groups; ++i) {
    /* the format's own range and a narrow one; past a normal double's (ldexp's subnormal scales, every
     * output finite); and past its largest (an infinite scale: the outputs it reaches non-finite) */
    int e = wide == 3   ? (int)(lanes_next() % 201u) + 900
            : wide == 2 ? -(int)(lanes_next() % 201u) - 900
            : wide      ? (int)(lanes_next() % 601u) - 300
                        : (int)(lanes_next() % 41u) - 30;
    uint16_t u = (uint16_t)(int16_t)e;
    exps[2u * i] = (uint8_t)u;
    exps[2u * i + 1u] = (uint8_t)(u >> 8);
  }
}
/* every byte of an output buffer still the 0x5A the test wrote before the call */
static int lanes_untouched(const double *buffer, size_t count) {
  const unsigned char *bytes = (const unsigned char *)buffer;
  size_t i;
  for (i = 0u; i < count * sizeof *buffer; ++i)
    if (bytes[i] != 0x5Au) return 0;
  return 1;
}
static int q8_lanes_test(void) {
  static const size_t shapes[][3] = {
      {128, 128, 32}, {128, 64, 32}, {128, 344, 32}, {344, 128, 32}, {128, 1024, 32}, {64, 72, 16},
      {3, 200, 8},    {5, 8, 8},     {2, 136, 64},   {17, 33, 5},    {100, 37, 64},   {64, 64, 48},
      {9, 40, 4},     {1, 1, 32},    {3, 5, 7},      {257, 129, 32}, {7, 9, 1},       {33, 17, 7},
      {16, 8, 16},    {24, 40, 8},   {200, 31, 48},  {128, 1023, 32}};
  double x[400], ref[1100], got[1100];
  static uint8_t codes[400 * 1100], exps[2 * 400 * 1100];
  int instances = bcir_ai_q8_instances(), wide, inst, checked = 0;
  size_t q, i;
  CHECK((instances & 1) && !(instances & ~7));
  for (wide = 0; wide < 4; ++wide) {
    for (q = 0u; q < sizeof shapes / sizeof *shapes; ++q) {
      size_t in = shapes[q][0], out = shapes[q][1], g = shapes[q][2], n = in * out;
      size_t groups = 1u + (n - 1u) / g;
      int sm, sr;
      lanes_fill(x, in, codes, n, exps, groups, wide);
      /* the matvec: in x out codes; the row dots: out rows of in coordinates (the same bytes) */
      memset(ref, 0xA5, sizeof ref);
      sm = bcir_ai_q8_matvec_f64_instance(1, x, in, codes, n, exps, groups, out, g, ref);
      CHECK(sm == BCIR_AI_OK || sm == BCIR_AI_INVALID_NUMBER);
      for (inst = 2; inst <= 4; inst *= 2) {
        if (!(instances & inst)) continue;
        memset(got, 0x5A, sizeof got);
        CHECK(bcir_ai_q8_matvec_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) == sm);
        CHECK(memcmp(ref, got, out * sizeof *ref) == 0);
        ++checked;
      }
      memset(ref, 0xA5, sizeof ref);
      sr = bcir_ai_q8_rows_dot_f64_instance(1, x, in, codes, n, exps, groups, out, g, ref);
      CHECK(sr == BCIR_AI_OK || sr == BCIR_AI_INVALID_NUMBER);
      for (inst = 2; inst <= 4; inst *= 2) {
        if (!(instances & inst)) continue;
        memset(got, 0x5A, sizeof got);
        CHECK(bcir_ai_q8_rows_dot_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) == sr);
        CHECK(memcmp(ref, got, out * sizeof *ref) == 0);
        ++checked;
      }
      CHECK(bcir_ai_q8_matvec_f64_prevalidated(x, in, codes, n, exps, groups, out, g, got) == sm);
#ifdef BCIR_AI_INSTANCE_PROBE
      /* the prevalidated entries run the widest instance the host admits */
      CHECK(bcir_ai_q8_instance_probe() == (instances & 4 ? 4 : instances & 2 ? 2 : 1));
#endif
      CHECK(bcir_ai_q8_rows_dot_f64_prevalidated(x, in, codes, n, exps, groups, out, g, got) == sr);
#ifdef BCIR_AI_INSTANCE_PROBE
      CHECK(bcir_ai_q8_instance_probe() == (instances & 4 ? 4 : instances & 2 ? 2 : 1));
      for (inst = 1; inst <= 4; inst *= 2) {
        if (!(instances & inst)) continue;
        CHECK(bcir_ai_q8_matvec_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) == sm);
        CHECK(bcir_ai_q8_instance_probe() == inst);
        CHECK(bcir_ai_q8_rows_dot_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) == sr);
        CHECK(bcir_ai_q8_instance_probe() == inst);
      }
#endif
    }
  }
  /* one output past a double's range, in the last block's last lane: every output zeroed */
  {
    size_t in = 3u, out = 72u, g = 8u, n = in * out, groups = n / g;
    for (i = 0u; i < in; ++i) x[i] = 0.5;
    memset(codes, 1, n);
    memset(exps, 0, 2u * groups);
    codes[2u * out + out - 1u] = 127;
    exps[2u * ((2u * out + out - 1u) / g)] = (uint8_t)(1023 & 0xff);
    exps[2u * ((2u * out + out - 1u) / g) + 1u] = (uint8_t)(1023 >> 8);
    x[2] = 1e300;
    for (inst = 1; inst <= 4; inst *= 2) {
      if (!(instances & inst)) continue;
      memset(got, 0x5A, sizeof got);
      CHECK(bcir_ai_q8_matvec_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) ==
            BCIR_AI_INVALID_NUMBER);
      for (i = 0u; i < out; ++i) CHECK(got[i] == 0.0 && !signbit(got[i]));
      /* as row dots: row 71 of 3 coordinates holds the code, its last group the scale */
      memset(got, 0x5A, sizeof got);
      CHECK(bcir_ai_q8_rows_dot_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) ==
            BCIR_AI_INVALID_NUMBER);
      for (i = 0u; i < out; ++i) CHECK(got[i] == 0.0 && !signbit(got[i]));
    }
  }
  /* refusals write nothing: instances the host does not admit, and a non-finite input */
  {
    /* never an instance, then the vector instances this host does not admit (none on an AVX-512 host) */
    static const int refused[] = {-1, 3, 5, 6, 7, 8, 16, 2, 4};
    size_t in = 4u, out = 16u, g = 8u, n = in * out, groups = n / g, r;
    lanes_fill(x, in, codes, n, exps, groups, 0);
    for (r = 0u; r < sizeof refused / sizeof *refused; ++r) {
      int bad = refused[r];
      if ((bad == 2 || bad == 4) && (instances & bad)) continue;
#ifdef BCIR_AI_INSTANCE_PROBE
      (void)bcir_ai_q8_instance_probe();  /* clear the probe: a refusal must leave it unset */
#endif
      memset(got, 0x5A, sizeof got);
      CHECK(bcir_ai_q8_matvec_f64_instance(bad, x, in, codes, n, exps, groups, out, g, got) ==
            BCIR_AI_INVALID_ARGUMENT);
      CHECK(bcir_ai_q8_rows_dot_f64_instance(bad, x, in, codes, n, exps, groups, out, g, got) ==
            BCIR_AI_INVALID_ARGUMENT);
      CHECK(lanes_untouched(got, sizeof got / sizeof *got));
#ifdef BCIR_AI_INSTANCE_PROBE
      CHECK(bcir_ai_q8_instance_probe() == 0);  /* a refusal runs no instance */
#endif
    }
    x[3] = NAN;
    for (inst = 0; inst <= 4; inst = inst ? inst * 2 : 1) {
      if (inst && !(instances & inst)) continue;
      memset(got, 0x5A, sizeof got);
      CHECK(bcir_ai_q8_matvec_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) ==
            BCIR_AI_INVALID_ARGUMENT);
      CHECK(bcir_ai_q8_rows_dot_f64_instance(inst, x, in, codes, n, exps, groups, out, g, got) ==
            BCIR_AI_INVALID_ARGUMENT);
      CHECK(lanes_untouched(got, sizeof got / sizeof *got));
    }
  }
  printf("Q8-LANES: instances %d, %d vector comparisons equal to the portable loops\n", instances, checked);
  CHECK(checked > 0 || instances == 1);
  return 0;
}

static int q15_test(void) {
  const int16_t query[2] = {0, 0};
  const int16_t patterns[6] = {1, 0, -1, 0, 5, 5};
  uint8_t eligible[3] = {1, 1, 1};
  bcir_ai_match matches[2];
  size_t count = 99u;
  CHECK(bcir_ai_q15_topk(query, patterns, eligible, 3u, 2u, 2u,
                         matches, 2u, &count) == BCIR_AI_OK);
  CHECK(count == 2u && matches[0].index == 0u && matches[0].squared_distance == 1u);
  CHECK(matches[1].index == 1u && matches[1].squared_distance == 1u);
  eligible[0] = 0u;
  CHECK(bcir_ai_q15_topk(query, patterns, eligible, 3u, 2u, 2u,
                         matches, 2u, &count) == BCIR_AI_OK);
  CHECK(count == 2u && matches[0].index == 1u && matches[1].index == 2u &&
        matches[1].squared_distance == 50u);
  eligible[2] = 2u; count = 77u;
  CHECK(bcir_ai_q15_topk(query, patterns, eligible, 3u, 2u, 2u,
                         matches, 2u, &count) == BCIR_AI_INVALID_ARGUMENT);
  CHECK(count == 0u);

  /* Fewer admitted rows than top_k asks for. Every case above admits at least
     two of three patterns, so `count < top_k` -- the loop's short return -- had
     never executed: a shipping branch with no test behind it. The shorter result
     is the answer, not a truncation, and the entries past it must be left
     exactly as the caller wrote them rather than filled with a stale match. */
  eligible[0] = 0u; eligible[1] = 1u; eligible[2] = 0u;
  matches[0].index = 0xEEEEEEEEu; matches[0].squared_distance = 0xEEEEu;
  matches[1].index = 0xDEADBEEFu; matches[1].squared_distance = 0xD15EA5Eu;
  count = 77u;
  CHECK(bcir_ai_q15_topk(query, patterns, eligible, 3u, 2u, 2u,
                         matches, 2u, &count) == BCIR_AI_OK);
  CHECK(count == 1u);
  CHECK(matches[0].index == 1u && matches[0].squared_distance == 1u);
  CHECK(matches[1].index == 0xDEADBEEFu && matches[1].squared_distance == 0xD15EA5Eu);

  /* And a mask that admits nothing returns nothing -- not everything, which is
     what a filter whose empty case was never tested tends to do. */
  eligible[0] = 0u; eligible[1] = 0u; eligible[2] = 0u;
  matches[0].index = 0xDEADBEEFu; matches[0].squared_distance = 0xD15EA5Eu;
  count = 77u;
  CHECK(bcir_ai_q15_topk(query, patterns, eligible, 3u, 2u, 2u,
                         matches, 2u, &count) == BCIR_AI_OK);
  CHECK(count == 0u);
  CHECK(matches[0].index == 0xDEADBEEFu && matches[0].squared_distance == 0xD15EA5Eu);
  return 0;
}

static int q4_group_test(void) {
  uint8_t packed[17];
  int8_t q8[33];
  uint8_t q4_exponents[4] = {0, 0, 1, 0};
  uint8_t q8_exponents[4] = {0, 0, 255, 255};
  double result = -7.0;
  memset(packed, UINT8_C(0x11), sizeof packed);
  packed[16] = UINT8_C(0x01);
  memset(q8, 2, sizeof q8);
  CHECK(bcir_q4_q8_group32_dot(packed, q4_exponents, q8, q8_exponents,
                               33u, 2u, &result) == 0);
  CHECK(result == 66.0);
  packed[0] = UINT8_C(0x18); result = -7.0;
  CHECK(bcir_q4_q8_group32_dot(packed, q4_exponents, q8, q8_exponents,
                               33u, 2u, &result) != 0);
  CHECK(result == -7.0);
  packed[0] = UINT8_C(0x11); q8[5] = INT8_MIN;
  CHECK(bcir_q4_q8_group32_dot(packed, q4_exponents, q8, q8_exponents,
                               33u, 2u, &result) != 0);
  CHECK(result == -7.0);
  return 0;
}

int main(void) {
  CHECK(bcir_ai_abi_version() == BCIR_AI_ABI_VERSION);
  CHECK(quantization_test() == 0);
  CHECK(q8_matrix_test() == 0);
  CHECK(q8_lanes_test() == 0);
  CHECK(q15_test() == 0);
  CHECK(q4_group_test() == 0);
  puts("native AI kernels: all gates passed");
  return 0;
}
