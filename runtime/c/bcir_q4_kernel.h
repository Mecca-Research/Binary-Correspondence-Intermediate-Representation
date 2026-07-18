#ifndef BCIR_Q4_KERNEL_H
#define BCIR_Q4_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_Q4_MAX_ELEMENTS UINT32_C(67108864)

/* BCIRQ4T v1: low nibble first, signed two's-complement, -8 forbidden.
 * `packed` is borrowed for the duration of the call; no ownership is transferred. */
int bcir_q4_codes_validate(const uint8_t *packed, size_t element_count);

/* Reference compute twin. Both inputs are borrowed and may not overlap writable state;
 * q4_exp + q8_exp is applied with ldexp after exact integer accumulation. */
double bcir_q4_q8_dot(const uint8_t *packed_q4, const int8_t *q8,
                      size_t element_count, int16_t q4_exp, int16_t q8_exp);

#ifdef __cplusplus
}
#endif

#endif
