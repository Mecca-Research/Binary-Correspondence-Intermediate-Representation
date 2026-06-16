/*===- fuzz_binrec.c - libFuzzer harness for the ETL binary-record decoder -===
 *
 * bcir_binrec_field decodes byte-aligned fields out of untrusted record bytes (the C
 * twin of bcir/etl/binary.py -- driver/device packet ingestion). This harness drives
 * it on arbitrary input under ASan/UBSan: any out-of-bounds read, overflow, or other
 * UB is caught. The decoder must NEVER read out of bounds -- a malformed descriptor or
 * short buffer must return a bcir_binrec_status, the only acceptable outcome.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       runtime/c/fuzz_binrec.c runtime/c/bcir_binrec.c -I runtime/c -o fuzz_binrec
 *   ./fuzz_binrec -runs=1000000           (see tools/c/fuzz_streampack.sh --binrec)
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_binrec.h"

/* The reference NVMe SQE header fields (etl.binary.NVME_SQE_HEADER). */
static const bcir_binfield kNvmeFields[] = {
  {0, 8, 0},    /* opcode       */
  {16, 16, 0},  /* command_id   */
  {32, 32, 0},  /* namespace_id */
};

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  /* (1) Decode the fixed NVMe SQE fields against arbitrary (often short) buffers. */
  for (unsigned i = 0; i < sizeof kNvmeFields / sizeof kNvmeFields[0]; ++i) {
    int64_t out = 0;
    (void)bcir_binrec_field(data, size, &kNvmeFields[i], 0, &out);
    (void)bcir_binrec_field(data, size, &kNvmeFields[i], 1, &out);
  }

  /* (2) Derive an attacker-controlled descriptor from the first 5 bytes and decode the
   *     remaining bytes -- explores hostile offset/width/sign/endianness combinations. */
  if (size >= 5) {
    bcir_binfield f;
    f.offset_bits = (uint16_t)(data[0] | (data[1] << 8));
    f.width_bits = (uint16_t)(data[2] | (data[3] << 8));
    f.is_signed = data[4];
    const uint8_t *buf = data + 5;
    size_t buflen = size - 5;
    int64_t out = 0;
    (void)bcir_binrec_field(buf, buflen, &f, 0, &out);
    (void)bcir_binrec_field(buf, buflen, &f, 1, &out);
    /* And the byte-aligned variant of the same width (offset/width forced legal). */
    f.offset_bits = (uint16_t)((f.offset_bits & 0x7u) ? (f.offset_bits & ~0x7u) : f.offset_bits);
    f.width_bits = (uint16_t)(((f.width_bits % 8u) ? (f.width_bits - f.width_bits % 8u) : f.width_bits));
    (void)bcir_binrec_field(buf, buflen, &f, 0, &out);
  }
  return 0;
}
