/*===- fuzz_telemetry_frame.c - libFuzzer harness for the telemetry-frame decoder -===
 *
 * bcir_telemetry_frame.c is a trust boundary the other harnesses did not cover: the
 * frames it decodes arrive from a DEVICE over a byte transport (UART today, per
 * docs/kernel/TELEMETRY_FRAME_ABI.md), so every byte -- magic, version, flags, the
 * u16 record count, the CRC -- is attacker-controlled, and the resync path scans a
 * stream that may contain no valid frame at all.
 *
 * The contract under test: for ANY (buf, len), every entry point returns a status and
 * never reads out of bounds. In particular
 *   - a `n_records` field that claims more records than the buffer holds must be
 *     TRUNCATED, not a body over-read;
 *   - `bcir_tf_get_record` must re-validate (it is a standalone accessor) and must
 *     reject an out-of-range index rather than index past the frame;
 *   - `bcir_tf_find_magic` must terminate and stay in bounds for every `start`,
 *     including start >= len and a 0..3-byte tail with no room for the magic.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       runtime/c/fuzz_telemetry_frame.c runtime/c/bcir_telemetry_frame.c \
 *       runtime/c/bcir_runtime.c -I runtime/c -o fuzz_telemetry_frame
 *   ./fuzz_telemetry_frame -runs=1000000     (see tools/c/fuzz_streampack.sh)
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_telemetry_frame.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  bcir_tf_header hdr;
  size_t frame_len = 0;

  /* (1) The framed decode: header + declared record count + CRC over untrusted bytes. */
  bcir_tf_status st = bcir_tf_decode_frame(data, size, &hdr, &frame_len);

  /* (2) The exact-fit decode (rejects trailing bytes) on the same input. */
  (void)bcir_tf_decode_exact(data, size, &hdr);

  /* (3) Record access. Walk the declared count when the frame validated, and ALSO probe
   *     indices past it -- a standalone accessor must re-validate and range-check rather
   *     than trust a previously decoded header. */
  {
    bcir_tf_record rec;
    uint32_t declared = (st == BCIR_TF_OK) ? (uint32_t)hdr.n_records : 0u;
    for (uint32_t i = 0; i < declared; i++)
      (void)bcir_tf_get_record(data, size, (uint16_t)i, &rec);
    /* out-of-range / boundary indices, valid frame or not */
    (void)bcir_tf_get_record(data, size, 0u, &rec);
    (void)bcir_tf_get_record(data, size, (uint16_t)declared, &rec);
    (void)bcir_tf_get_record(data, size, UINT16_MAX, &rec);
    (void)bcir_tf_get_record(data, size, 0u, NULL);       /* NULL out is legal (validate-only) */
  }

  /* (4) The resync scan: every start offset must stay in bounds and terminate, and the
   *     stream walk (find magic -> decode -> skip) must make progress on garbage. */
  {
    size_t at = 0, guard = 0;
    while (at < size && guard++ < 4096u) {
      size_t found = bcir_tf_find_magic(data, size, at);
      if (found >= size) break;
      size_t used = 0;
      if (bcir_tf_decode_frame(data + found, size - found, &hdr, &used) == BCIR_TF_OK && used)
        at = found + used;
      else
        at = found + 1u;                                  /* resync past this magic */
    }
    /* degenerate starts */
    (void)bcir_tf_find_magic(data, size, size);
    (void)bcir_tf_find_magic(data, size, size ? size - 1u : 0u);
    (void)bcir_tf_find_magic(data, size, SIZE_MAX);
  }

  /* (5) Re-encode round-trip: feed the decoded records back through the writer with an
   *     exact-fit and a deliberately tiny buffer (the NOSPACE path). */
  if (st == BCIR_TF_OK && hdr.n_records && hdr.n_records <= 64u) {
    bcir_tf_record recs[64];
    static uint8_t out[BCIR_TELEMETRY_FRAME_HEADER_SIZE
                       + 64u * BCIR_TF_RECORD_SIZE + BCIR_TELEMETRY_FRAME_CRC_SIZE];
    static uint8_t tiny[8];
    uint16_t n = hdr.n_records, i;
    for (i = 0; i < n; i++)
      if (bcir_tf_get_record(data, size, i, &recs[i]) != BCIR_TF_OK) return 0;
    (void)bcir_tf_encode_frame(recs, n, hdr.seq, hdr.timestamp, out, sizeof out);
    (void)bcir_tf_encode_frame(recs, n, hdr.seq, hdr.timestamp, tiny, sizeof tiny);
  }
  return 0;
}
