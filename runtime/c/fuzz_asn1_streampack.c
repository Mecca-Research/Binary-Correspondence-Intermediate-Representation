/* libFuzzer harness for the DER -> native StreamPack fast path.
 *
 * This is a TRUST BOUNDARY WITH A TWIST: unlike the other decoders, it does not merely
 * read hostile octets, it WRITES an artifact derived from them. Two properties matter,
 * and the harness checks both:
 *
 *   1. No out-of-bounds read of the projection and no out-of-bounds write of the
 *      output, for any input at all. ASan/UBSan enforce this.
 *   2. A BLESSED OUTPUT IS REALLY A PACK. If the reconstruction returns BCIR_OK, the
 *      octets it produced are fed straight back into the native decoder and the
 *      semantic verifier. A fast path that emitted a subtly malformed artifact -- one
 *      the normal decoder would reject -- would be worse than one that simply crashed,
 *      because the corruption would surface far away from here.
 *
 * The output buffer is deliberately FIXED and small rather than sized from
 * bcir_asn1_streampack_bound, so the NOSPACE path is exercised on most inputs instead
 * of being unreachable.
 */
#include <stddef.h>
#include <stdint.h>

#include "bcir_asn1_streampack.h"

/* Big enough for the corpus seeds, small enough that oversized inputs hit NOSPACE. */
static uint8_t g_out[1 << 16];

static int count_segment(const bcir_segment_view *seg, void *ctx) {
  (void)seg;
  ++*(unsigned *)ctx;
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  size_t out_len = 0;
  bcir_status st = bcir_asn1_to_streampack(data, size, g_out, sizeof g_out, &out_len);

  if (st != BCIR_OK) {
    /* A failure must report nothing written -- a caller that trusted a stale out_len
     * after an error would read uninitialised octets. */
    if (out_len != 0)
      __builtin_trap();
    return 0;
  }

  if (out_len == 0 || out_len > sizeof g_out)
    __builtin_trap();

  /* The reconstruction claimed success, so the artifact must satisfy every gate the
   * native rail applies to a pack it did not produce. */
  bcir_streampack_header hdr;
  if (bcir_sp_validate(g_out, out_len, &hdr) != BCIR_OK)
    __builtin_trap();
  if (bcir_sp_verify_semantic(g_out, out_len, UINT32_MAX, UINT32_MAX) != BCIR_OK)
    __builtin_trap();

  unsigned seen = 0;
  if (bcir_sp_for_each_segment(g_out, out_len, count_segment, &seen) != BCIR_OK)
    __builtin_trap();
  if (seen != hdr.n_segments)
    __builtin_trap();

  return 0;
}
