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
  CHECK(q15_test() == 0);
  CHECK(q4_group_test() == 0);
  puts("native AI kernels: all gates passed");
  return 0;
}
