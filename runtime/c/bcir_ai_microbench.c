/* Bounded native model-plan calibration. This is a hosted measurement tool,
 * not a legality oracle: it times deterministic dense and BCIRQ8 group-32
 * prefill/matvec loops and emits a timestamp-free JSON record. */
#include "bcir_ai_kernels.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BCIR_AI_BENCH_MAX_DIMENSION 64u
#define BCIR_AI_BENCH_MAX_REPEATS 15u

typedef struct sample_summary {
  uint64_t lower_ns;
  uint64_t median_ns;
  uint64_t upper_ns;
  double checksum;
} sample_summary;

static volatile double g_sink = 0.0;

static uint64_t now_ns(void) {
  struct timespec value = {0, 0};
  uint64_t seconds, nanos;
  if (timespec_get(&value, TIME_UTC) != TIME_UTC || value.tv_sec < 0 || value.tv_nsec < 0)
    return 0u;
  seconds = (uint64_t)value.tv_sec;
  nanos = (uint64_t)value.tv_nsec;
  if (seconds > (UINT64_MAX - nanos) / UINT64_C(1000000000)) return UINT64_MAX;
  return seconds * UINT64_C(1000000000) + nanos;
}

static uint64_t elapsed(uint64_t start) {
  uint64_t end = now_ns();
  return end > start ? end - start : 1u;
}

static int compare_u64(const void *left, const void *right) {
  uint64_t a = *(const uint64_t *)left;
  uint64_t b = *(const uint64_t *)right;
  return (a > b) - (a < b);
}

static int parse_size(const char *text, size_t minimum, size_t maximum, size_t *out) {
  char *end = NULL;
  unsigned long long value;
  if (text == NULL || *text == '\0' || *text == '-') return -1;
  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno || end == NULL || *end != '\0' || value < minimum || value > maximum ||
      value > SIZE_MAX) return -1;
  *out = (size_t)value;
  return 0;
}

static void dense_matvec(const double *input, const double *weights,
                         size_t n, double *output) {
  size_t input_index, output_index;
  for (output_index = 0u; output_index < n; ++output_index) output[output_index] = 0.0;
  for (input_index = 0u; input_index < n; ++input_index)
    for (output_index = 0u; output_index < n; ++output_index)
      output[output_index] += input[input_index] * weights[input_index * n + output_index];
}

static int run_pass(const double *activation, const double *dense_weights,
                    const uint8_t *q8_codes, const uint8_t *q8_exponents,
                    size_t q8_groups, size_t n, int quantized, int prefill,
                    double *workspace, uint64_t *duration, double *checksum) {
  size_t row, rows = prefill ? n : 1u;
  uint64_t start = now_ns();
  double total = 0.0;
  for (row = 0u; row < rows; ++row) {
    const double *input = activation + row * n;
    size_t output_index;
    if (quantized) {
      if (bcir_ai_q8_matvec_f64_prevalidated(
              input, n, q8_codes, n * n, q8_exponents, q8_groups,
              n, 32u, workspace) != BCIR_AI_OK) return -1;
    } else {
      dense_matvec(input, dense_weights, n, workspace);
    }
    for (output_index = 0u; output_index < n; ++output_index)
      total += workspace[output_index];
  }
  *duration = elapsed(start);
  *checksum = total;
  g_sink += total;
  return isfinite(total) ? 0 : -1;
}

static int measure(const double *activation, const double *dense_weights,
                   const uint8_t *q8_codes, const uint8_t *q8_exponents,
                   size_t q8_groups, size_t n, int quantized, int prefill,
                   size_t repeats, double *workspace, sample_summary *out) {
  uint64_t samples[BCIR_AI_BENCH_MAX_REPEATS];
  size_t repeat;
  double checksum = 0.0, value;
  if (run_pass(activation, dense_weights, q8_codes, q8_exponents, q8_groups,
               n, quantized, prefill, workspace, &samples[0], &value)) return -1;
  for (repeat = 0u; repeat < repeats; ++repeat) {
    if (run_pass(activation, dense_weights, q8_codes, q8_exponents, q8_groups,
                 n, quantized, prefill, workspace, &samples[repeat], &value)) return -1;
    checksum += value;
  }
  qsort(samples, repeats, sizeof samples[0], compare_u64);
  out->lower_ns = samples[0];
  out->median_ns = samples[repeats / 2u];
  out->upper_ns = samples[repeats - 1u];
  out->checksum = checksum;
  return 0;
}

int main(int argc, char **argv) {
  size_t n, repeats, elements, groups, index;
  int quantized;
  double activation[BCIR_AI_BENCH_MAX_DIMENSION * BCIR_AI_BENCH_MAX_DIMENSION];
  double weights[BCIR_AI_BENCH_MAX_DIMENSION * BCIR_AI_BENCH_MAX_DIMENSION];
  double workspace[BCIR_AI_BENCH_MAX_DIMENSION];
  int16_t native_exponents[(BCIR_AI_BENCH_MAX_DIMENSION *
                            BCIR_AI_BENCH_MAX_DIMENSION + 31u) / 32u];
  uint8_t exponent_le[2u * ((BCIR_AI_BENCH_MAX_DIMENSION *
                            BCIR_AI_BENCH_MAX_DIMENSION + 31u) / 32u)];
  uint8_t codes[BCIR_AI_BENCH_MAX_DIMENSION * BCIR_AI_BENCH_MAX_DIMENSION];
  sample_summary prefill, decode;
  if (argc != 4 || parse_size(argv[1], 4u, BCIR_AI_BENCH_MAX_DIMENSION, &n) ||
      parse_size(argv[2], 3u, BCIR_AI_BENCH_MAX_REPEATS, &repeats) ||
      (strcmp(argv[3], "source") != 0 && strcmp(argv[3], "bcirq8-group32") != 0)) {
    fprintf(stderr, "usage: bcir-ai-microbench dimension[4..64] repeats[3..15] source|bcirq8-group32\n");
    return 2;
  }
  quantized = strcmp(argv[3], "bcirq8-group32") == 0;
  elements = n * n;
  groups = 1u + (elements - 1u) / 32u;
  for (index = 0u; index < elements; ++index) {
    activation[index] = (double)((int)((index * 17u) % 31u) - 15) / 16.0;
    weights[index] = (double)((int)((index * 13u) % 29u) - 14) / 16.0;
  }
  if (bcir_ai_quantize_q8_f64(weights, elements, 32u, native_exponents,
                              groups, (int8_t *)(void *)codes, elements) != BCIR_AI_OK) {
    fprintf(stderr, "native model microbench quantization failed\n");
    return 2;
  }
  for (index = 0u; index < groups; ++index) {
    uint16_t value = (uint16_t)native_exponents[index];
    exponent_le[2u * index] = (uint8_t)(value & UINT16_C(0x00ff));
    exponent_le[2u * index + 1u] = (uint8_t)(value >> 8);
  }
  if (measure(activation, weights, codes, exponent_le, groups, n, quantized, 1,
              repeats, workspace, &prefill) ||
      measure(activation, weights, codes, exponent_le, groups, n, quantized, 0,
              repeats, workspace, &decode)) {
    fprintf(stderr, "native model microbench produced invalid output\n");
    return 2;
  }
  printf("{\"schema\":\"bcir.native_model_microbench.v1\","
         "\"dimension\":%zu,\"repeats\":%zu,\"weight_format\":\"%s\","
         "\"prefill\":{\"lower_ns\":%" PRIu64 ",\"median_ns\":%" PRIu64
         ",\"upper_ns\":%" PRIu64 ",\"checksum\":%.17g},"
         "\"decode\":{\"lower_ns\":%" PRIu64 ",\"median_ns\":%" PRIu64
         ",\"upper_ns\":%" PRIu64 ",\"checksum\":%.17g}}\n",
         n, repeats, argv[3], prefill.lower_ns, prefill.median_ns, prefill.upper_ns,
         prefill.checksum, decode.lower_ns, decode.median_ns, decode.upper_ns,
         decode.checksum);
  return 0;
}
