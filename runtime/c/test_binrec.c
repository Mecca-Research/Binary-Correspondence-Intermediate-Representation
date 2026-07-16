/*===- test_binrec.c - Python<->C parity harness for the binary-record decoder ===
 *
 * Reads a record file (argv[1]) and decodes a FIXED, documented field set with
 * bcir_binrec_field, printing one `<idx> <status> <value>` line per field. The Python
 * side (bcir/tests/test_c23_runtime.py) decodes the same bytes with etl.binary and
 * asserts the per-field (status, value) agree -- the ABI-parity gate for the decoder
 * (the analog of test_runtime.c for StreamPack). Uses libc (a host harness, not the
 * freestanding runtime). Build: clang -std=c23 test_binrec.c bcir_binrec.c -I .
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_binrec.h"

/* The field set the Python parity test mirrors exactly (offset_bits, width_bits,
 * is_signed, big_endian). Covers u8 / u16 / u32 LE, a signed 16-bit, a big-endian
 * 16-bit, and a full-width signed 64-bit -- endianness + sign-extension parity. */
static const struct { bcir_binfield f; int big_endian; } kFields[] = {
  {{0, 8, 0}, 0},    {{16, 16, 0}, 0},  {{32, 32, 0}, 0},
  {{16, 16, 1}, 0},  {{0, 16, 0}, 1},   {{0, 64, 1}, 0},
};

static int invalid_input_regressions(void) {
  const uint8_t byte = 0;
  const bcir_binfield field = {0, 8, 0};
  int64_t out = 17;
  if (bcir_binrec_field(NULL, 1, &field, 0, &out) != BCIR_BINREC_ERR_INVALID || out != 0)
    return 0;
  out = 17;
  if (bcir_binrec_field(&byte, 1, NULL, 0, &out) != BCIR_BINREC_ERR_INVALID || out != 0)
    return 0;
  if (bcir_binrec_field(&byte, 1, &field, 0, NULL) != BCIR_BINREC_ERR_INVALID)
    return 0;
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <record-file>\n", argv[0]); return 2; }
  if (!invalid_input_regressions()) {
    fprintf(stderr, "invalid-input regression failed\n");
    return 3;
  }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen"); return 2; }
  static uint8_t buf[4096];
  size_t len = fread(buf, 1, sizeof buf, fp);
  fclose(fp);

  for (unsigned i = 0; i < sizeof kFields / sizeof kFields[0]; ++i) {
    int64_t out = 0;
    bcir_binrec_status st = bcir_binrec_field(buf, len, &kFields[i].f, kFields[i].big_endian, &out);
    printf("%u %d %lld\n", i, (int)st, (long long)out);
  }
  return 0;
}
