#include "bcir_ai_kernels.h"

#include <math.h>
#include <string.h>

#define BCIR_AI_EXP_MIN (-300)
#define BCIR_AI_EXP_MAX 300

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

int bcir_ai_q8_matvec_f64_prevalidated(
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

int bcir_ai_q8_rows_dot_f64_prevalidated(
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
