#include "bcir_q4_kernel.h"

#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static int8_t bcir_q4_decode(uint8_t nibble) {
    nibble &= UINT8_C(0x0f);
    return (int8_t)((nibble & UINT8_C(0x08)) ? (int)nibble - 16 : (int)nibble);
}

int bcir_q4_codes_validate(const uint8_t *packed, size_t element_count) {
    size_t index;
    if (element_count > (size_t)BCIR_Q4_MAX_ELEMENTS) return 0;
    if (packed == NULL && element_count != 0U) return 0;
    if ((element_count & 1U) != 0U && (packed[element_count / 2U] & UINT8_C(0xf0)) != 0U)
        return 0;
    for (index = 0U; index < element_count; ++index) {
        uint8_t byte = packed[index / 2U];
        uint8_t nibble = (uint8_t)((index & 1U) != 0U ? byte >> 4 : byte & UINT8_C(0x0f));
        if (nibble == UINT8_C(0x08)) return 0;
    }
    return 1;
}

#if defined(__AVX2__)
static int64_t bcir_q4_q8_dot_avx2(const uint8_t *packed, const int8_t *q8,
                                   size_t count, size_t *consumed) {
    int64_t total = 0;
    size_t index = 0U;
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i sixteen = _mm_set1_epi8(0x10);
    while (index + 32U <= count) {
        __m128i bytes = _mm_loadu_si128((const __m128i *)(const void *)(packed + index / 2U));
        __m128i low = _mm_and_si128(bytes, mask);
        __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        low = _mm_sub_epi8(low, _mm_and_si128(_mm_cmpgt_epi8(low, _mm_set1_epi8(7)), sixteen));
        high = _mm_sub_epi8(high, _mm_and_si128(_mm_cmpgt_epi8(high, _mm_set1_epi8(7)), sixteen));
        {
            __m128i q4_lo = _mm_unpacklo_epi8(low, high);
            __m128i q4_hi = _mm_unpackhi_epi8(low, high);
            __m128i q8_lo = _mm_loadu_si128((const __m128i *)(const void *)(q8 + index));
            __m128i q8_hi = _mm_loadu_si128((const __m128i *)(const void *)(q8 + index + 16U));
            __m256i q4_16_lo = _mm256_cvtepi8_epi16(q4_lo);
            __m256i q4_16_hi = _mm256_cvtepi8_epi16(q4_hi);
            __m256i q8_16_lo = _mm256_cvtepi8_epi16(q8_lo);
            __m256i q8_16_hi = _mm256_cvtepi8_epi16(q8_hi);
            __m256i products = _mm256_add_epi32(
                _mm256_madd_epi16(q4_16_lo, q8_16_lo),
                _mm256_madd_epi16(q4_16_hi, q8_16_hi));
            int32_t lanes[8];
            size_t lane;
            _mm256_storeu_si256((__m256i *)(void *)lanes, products);
            for (lane = 0U; lane < 8U; ++lane) total += lanes[lane];
        }
        index += 32U;
    }
    *consumed = index;
    return total;
}
#endif

double bcir_q4_q8_dot(const uint8_t *packed_q4, const int8_t *q8,
                      size_t element_count, int16_t q4_exp, int16_t q8_exp) {
    size_t index = 0U;
    int64_t accumulator = 0;
    if ((packed_q4 == NULL || q8 == NULL) && element_count != 0U) return NAN;
    if (!bcir_q4_codes_validate(packed_q4, element_count)) return NAN;
#if defined(__AVX2__)
    accumulator = bcir_q4_q8_dot_avx2(packed_q4, q8, element_count, &index);
#endif
    for (; index < element_count; ++index) {
        uint8_t byte = packed_q4[index / 2U];
        uint8_t nibble = (uint8_t)((index & 1U) != 0U ? byte >> 4 : byte & UINT8_C(0x0f));
        accumulator += (int64_t)bcir_q4_decode(nibble) * (int64_t)q8[index];
    }
    return ldexp((double)accumulator, (int)q4_exp + (int)q8_exp);
}

static int bcir_q4_exponent_le(const uint8_t *bytes, size_t group) {
    const uint8_t *value = bytes + 2U * group;
    uint16_t raw = (uint16_t)value[0] | (uint16_t)((uint16_t)value[1] << 8);
    return raw >= UINT16_C(0x8000) ? (int)raw - 65536 : (int)raw;
}

int bcir_q4_q8_group32_dot(const uint8_t *packed_q4,
                           const uint8_t *q4_exponents_le,
                           const int8_t *q8,
                           const uint8_t *q8_exponents_le,
                           size_t element_count, size_t exponent_count,
                           double *result) {
    size_t group_count, group;
    double total = 0.0;
    if (result == NULL || element_count > (size_t)BCIR_Q4_MAX_ELEMENTS)
        return -1;
    group_count = element_count == 0U ? 0U : 1U + (element_count - 1U) / 32U;
    if (((packed_q4 == NULL || q8 == NULL) && element_count != 0U) ||
        ((q4_exponents_le == NULL || q8_exponents_le == NULL) && group_count != 0U) ||
        exponent_count < group_count || !bcir_q4_codes_validate(packed_q4, element_count))
        return -1;
    for (group = 0U; group < group_count; ++group) {
        int q4_exp = bcir_q4_exponent_le(q4_exponents_le, group);
        int q8_exp = bcir_q4_exponent_le(q8_exponents_le, group);
        size_t begin = group * 32U;
        size_t end = begin + 32U;
        size_t index;
        int64_t accumulator = 0;
        if (end > element_count) end = element_count;
        if (q4_exp < -300 || q4_exp > 300 || q8_exp < -300 || q8_exp > 300)
            return -1;
        for (index = begin; index < end; ++index) {
            uint8_t byte = packed_q4[index / 2U];
            uint8_t nibble = (uint8_t)((index & 1U) != 0U ? byte >> 4 : byte & UINT8_C(0x0f));
            if (q8[index] == INT8_MIN) return -1;
            accumulator += (int64_t)bcir_q4_decode(nibble) * (int64_t)q8[index];
        }
        total += ldexp((double)accumulator, q4_exp + q8_exp);
        if (!isfinite(total)) return -1;
    }
    *result = total;
    return 0;
}
