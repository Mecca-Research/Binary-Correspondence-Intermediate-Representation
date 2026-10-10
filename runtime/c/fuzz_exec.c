/*===- fuzz_exec.c - libFuzzer harness for the StreamPack executor --------===
 *
 * bcir_sp_execute runs an untrusted StreamPack (the deployed plan) end to end. This
 * harness drives it on arbitrary bytes under ASan/UBSan with fixed caller buffers: a
 * malformed pack must return a bcir_status (never an out-of-bounds read), and the
 * executor must never write past the caller-provided scratch / phases buffers. The block
 * walk (bcir_sp_for_each_block, the decoder program interpreter's second walk) is driven
 * over the same bytes, every stride read in and one past its declared range.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       fuzz_exec.c bcir_exec.c bcir_runtime.c -I . -o fuzz_exec
 *   ./fuzz_exec -runs=1000000           (see tools/c/fuzz_streampack.sh)
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_exec.h"

static int touch(const bcir_exec_item *item, void *ctx) {
  uint64_t *acc = (uint64_t *)ctx;
  *acc += item->claim_id + item->phase_id + item->width + item->lane;
  return 0;
}

static int touch_block(const bcir_block_view *block, void *ctx) {
  uint64_t *acc = (uint64_t *)ctx;
  uint16_t i;
  *acc += block->base + block->count;
  for (i = 0; i <= block->n_strides; ++i) *acc += bcir_block_stride(block, i);
  return (*acc & 0xFFu) == 0x5Au; /* stop now and then: parsing must go on regardless */
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  /* Deliberately small buffers so the NOSPACE path is exercised on large packs too. */
  static bcir_exec_item scratch[64];
  static bcir_phase_stat phases[16];
  bcir_exec_result res;
  uint64_t acc = 0;
  (void)bcir_sp_execute(data, size, scratch, 64, phases, 16, touch, &acc, &res);
  (void)bcir_sp_for_each_block(data, size, touch_block, &acc);
  (void)bcir_sp_for_each_block(data, size, NULL, NULL);
  return 0;
}
