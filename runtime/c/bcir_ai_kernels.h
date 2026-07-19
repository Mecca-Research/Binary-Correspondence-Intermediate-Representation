#ifndef BCIR_AI_KERNELS_H
#define BCIR_AI_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(BCIR_AI_BUILD_SHARED)
#define BCIR_AI_API __declspec(dllexport)
#elif defined(_WIN32) && defined(BCIR_AI_USE_SHARED)
#define BCIR_AI_API __declspec(dllimport)
#else
#define BCIR_AI_API
#endif

#define BCIR_AI_MAX_ELEMENTS UINT32_C(67108864)
#define BCIR_AI_MAX_PATTERNS UINT32_C(65536)
#define BCIR_AI_MAX_DIMENSION UINT32_C(4096)
#define BCIR_AI_MAX_TOP_K UINT32_C(1024)
#define BCIR_AI_ABI_VERSION UINT32_C(1)

#define BCIR_AI_OK 0
#define BCIR_AI_INVALID_ARGUMENT (-1)
#define BCIR_AI_INVALID_NUMBER (-2)

typedef struct bcir_ai_match {
  uint32_t index;
  uint64_t squared_distance;
} bcir_ai_match;

/* Runtime handshake for explicitly loaded shared libraries. */
BCIR_AI_API uint32_t bcir_ai_abi_version(void);

/* Quantize finite f64 values with the same symmetric, power-of-two bridge used
 * by bcir.kbcir.quantize.  Q8 codes are signed [-127, 127].  Q4 codes are
 * signed [-7, 7], packed low nibble first with zero odd-element padding.
 *
 * Every pointer is borrowed and input/output spans must not overlap. Capacities
 * are measured in elements except for packed_capacity, which is bytes. The
 * functions validate the complete input before mutating either output, and
 * allocate no memory. */
BCIR_AI_API int bcir_ai_quantize_q8_f64(
    const double *values, size_t element_count, size_t group_size,
    int16_t *exponents, size_t exponent_capacity,
    int8_t *codes, size_t code_capacity);

BCIR_AI_API int bcir_ai_quantize_q4_f64(
    const double *values, size_t element_count, size_t group_size,
    int16_t *exponents, size_t exponent_capacity,
    uint8_t *packed_codes, size_t packed_capacity);

/* Cache-friendly row-major [input_count, output_count] BCIRQ8 matrix-vector
 * kernel.  Exponents are little-endian signed int16 values, one per group.
 * Input, weights, and output must not overlap. The output is zeroed if
 * arithmetic produces a non-finite value. */
BCIR_AI_API int bcir_ai_q8_matvec_f64(
    const double *input, size_t input_count,
    const uint8_t *codes, size_t code_count,
    const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output);

/* Validate one symmetric BCIRQ8 code/exponent inventory once, then reuse the
 * prevalidated entry points inside a decoder.  The prevalidated functions still
 * check dimensions, finite activations, capacities, and finite outputs; they
 * only omit rescanning immutable weight bytes on every token. */
BCIR_AI_API int bcir_ai_q8_values_validate(
    const uint8_t *codes, size_t code_count,
    const uint8_t *exponents_le, size_t exponent_count,
    size_t element_count, size_t group_size);

BCIR_AI_API int bcir_ai_q8_matvec_f64_prevalidated(
    const double *input, size_t input_count,
    const uint8_t *codes, size_t code_count,
    const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output);

/* Output-major [output_count, input_count] counterpart used by embedding/head
 * projections.  Each output dot preserves increasing-input accumulation order. */
BCIR_AI_API int bcir_ai_q8_rows_dot_f64(
    const double *input, size_t input_count,
    const uint8_t *codes, size_t code_count,
    const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output);

BCIR_AI_API int bcir_ai_q8_rows_dot_f64_prevalidated(
    const double *input, size_t input_count,
    const uint8_t *codes, size_t code_count,
    const uint8_t *exponents_le, size_t exponent_count,
    size_t output_count, size_t group_size, double *output);

/* Exact Q15 squared-L2 top-k.  Pattern rows and query coordinates are signed
 * int16.  `eligible` is optional; when present it contains only 0/1 bytes.
 * Pattern order is the deterministic tie-break (the Python contract stores
 * patterns in SHA-256 order). Matches must not overlap any input and are
 * returned sorted by distance/index. */
BCIR_AI_API int bcir_ai_q15_topk(
    const int16_t *query, const int16_t *patterns,
    const uint8_t *eligible, size_t pattern_count, size_t dimension,
    size_t top_k, bcir_ai_match *matches, size_t match_capacity,
    size_t *match_count);

#ifdef __cplusplus
}
#endif

#endif
