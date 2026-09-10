#include <stdint.h>

__attribute__((noinline)) static int local_bias(int value) {
  return value + 3;
}

__attribute__((noinline)) int matrix_transform(int value) {
  return local_bias(value) * 2 + 7;
}

uint32_t matrix_checksum(const uint8_t *bytes,
                                                    uint32_t count) {
  uint32_t hash = 17;
  for (uint32_t index = 0; index < count; ++index)
    hash = hash * 3 + bytes[index];
  return hash;
}
