/*===- fuzz_streampack.c - libFuzzer harness for the StreamPack C decoder --===
 *
 * The StreamPack decoder (bcir_runtime.c) is a trust boundary: it parses an
 * untrusted binary artifact (the deployed plan). This harness drives it on
 * arbitrary bytes under ASan/UBSan so any out-of-bounds read, integer overflow, or
 * other UB is caught -- the C analog of bcir/kbcir/fuzz.py for the Python boundaries.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       runtime/c/fuzz_streampack.c runtime/c/bcir_runtime.c -I runtime/c -o fuzz
 *   ./fuzz -runs=1000000           (see tools/c/fuzz_streampack.sh)
 *
 * The decoder must NEVER read out of bounds: every malformed input returns a
 * bcir_status (BCIR_ERR_*), so the only acceptable outcome is a clean status.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_runtime.h"

/* Touch each segment view's bounds-derived pointers so ASan flags any bad range. */
static int visit(const bcir_segment_view *seg, void *ctx) {
  uint64_t *acc = (uint64_t *)ctx;
  *acc += seg->claim_id + seg->phase_id + seg->lane + seg->width + seg->stride_k;
  for (uint16_t i = 0; i < seg->n_reads; i++)
    *acc += bcir_seg_read_rid(seg, i);
  for (uint16_t i = 0; i < seg->n_writes; i++)
    *acc += bcir_seg_write_rid(seg, i);
  if (seg->name_len)
    *acc += (uint8_t)seg->name[0];           /* in-bounds by construction */
  if (seg->opcode_len)
    *acc += (uint8_t)seg->opcode[seg->opcode_len - 1];
  /* v3 + prefetch view fields: touch their bounds-derived ends so ASan flags a bad range. */
  *acc += seg->dispatch;
  if (seg->prefetch_len)
    *acc += (uint8_t)seg->prefetch[seg->prefetch_len - 1];
  if (seg->channel_len)
    *acc += (uint8_t)seg->channel[seg->channel_len - 1];
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  bcir_streampack_header hdr;
  (void)bcir_sp_validate(data, size, &hdr);   /* header + CRC validation path */
  uint64_t acc = 0;
  (void)bcir_sp_for_each_segment(data, size, visit, &acc);  /* the full body walk */
  /* The semantic verifier (R10/R11 + range gate) is a second trust-boundary pass: it makes
   * extra bounded walks over the same untrusted bytes, so it must also never over-read. */
  (void)bcir_sp_verify_semantic(data, size, 0xFFFFFFFFu, 0xFFFFFFFFu);
  (void)bcir_sp_check_generation(data, size, 1u, 0u);
  /* A standalone CRC over the buffer must also never over-read. */
  if (size)
    (void)bcir_crc32(data, size);
  return 0;
}
