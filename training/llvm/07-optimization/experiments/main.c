#include <stdint.h>

extern int matrix_transform(int value);
extern uint32_t matrix_checksum(const uint8_t *bytes, uint32_t count);

int main(int argc, char **argv) {
  static const uint8_t payload[] = {3, 1, 4, 1, 5, 9};
  int transformed = matrix_transform(argc + 6);
  uint32_t checksum = matrix_checksum(payload, (uint32_t)sizeof(payload));
  (void)argv;
  return (transformed == 27 && checksum == 13344u) ? 0 : 1;
}
