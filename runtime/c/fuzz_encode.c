/*===- fuzz_encode.c - libFuzzer harness for the StreamPack encoder -------===
 *
 * bcir_sp_reencode parses an untrusted StreamPack and re-serializes it. This harness
 * drives it on arbitrary bytes under ASan/UBSan: a malformed input must return a status
 * (never an out-of-bounds read), and the writer must never exceed the output buffer
 * (the BCIR_ERR_NOSPACE path is exercised with a deliberately small buffer too).
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       fuzz_encode.c bcir_encode.c bcir_runtime.c -I . -o fuzz_encode
 *   ./fuzz_encode -runs=1000000           (see tools/c/fuzz_streampack.sh)
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_encode.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t out[1 << 16];
  size_t out_len = 0;
  /* Full-size buffer: exercises the happy path + bounds. */
  (void)bcir_sp_reencode(data, size, out, sizeof out, &out_len);
  /* Tiny buffer: forces the writer's BCIR_ERR_NOSPACE checks across every record. */
  static uint8_t tiny[80];
  (void)bcir_sp_reencode(data, size, tiny, sizeof tiny, &out_len);
  return 0;
}
