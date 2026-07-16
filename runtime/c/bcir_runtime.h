/*===- bcir_runtime.h - freestanding BCIR StreamPack runtime --------------===
 *
 * A no-libc loader/executor for frozen StreamPack v1 and its append-only v2/v3 forms
 * (bcir_streampack.h / docs/kernel/BCIR_STREAMPACK_ABI.md). Depends only on
 * <stddef.h> + <stdint.h> (both freestanding-safe), so it links into kernels,
 * drivers, and embedded/WASM hosts with no runtime dependency.
 *
 * The Python reference is bcir/abi/streampack_abi.py; a cross-language parity
 * test (Python encodes, this C decodes) gates the ABI freeze.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_RUNTIME_H
#define BCIR_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_streampack.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_status {
  BCIR_OK = 0,
  BCIR_ERR_TRUNCATED = 1,
  BCIR_ERR_MAGIC = 2,
  BCIR_ERR_VERSION = 3,
  BCIR_ERR_CRC = 4,
  BCIR_ERR_NOSPACE = 5,  /* a caller-provided buffer was too small (executor) */
  /* --- semantic trust-boundary failures (post-CRC; the C twin of verify_pack R10/R11
   *     + the lane/width range checks). A CRC-valid but semantically-corrupt pack must
   *     fail HERE rather than execute silently. --- */
  BCIR_ERR_LANE = 6,        /* a segment lane u8 outside the 6 valid lanes (mirror Lane(bad)) */
  BCIR_ERR_WIDTH = 7,       /* a segment width that is zero or not a power of two */
  BCIR_ERR_DISPATCH = 8,    /* a v3 segment dispatch code outside the legal set */
  BCIR_ERR_PROVENANCE = 9,  /* R10: a segment claim_id has no trace note, or an unresolved prefetch */
  BCIR_ERR_STALE = 10,      /* R11: map_gen/data_gen do not match the caller's expected generation */
  BCIR_ERR_OVERFLOW = 11,   /* a derived size/cost cannot be represented by the public ABI */
  BCIR_ERR_TRAILING = 12,   /* CRC-valid bytes remain after all declared body records */
  BCIR_ERR_RESERVED = 13,   /* a reserved header/record field is nonzero */
  BCIR_ERR_UTF8 = 14        /* a length-prefixed wire string is not valid UTF-8 */
} bcir_status;

/* zlib-compatible CRC-32 (reflected, poly 0xEDB88320). NULL is valid only with len 0;
 * an invalid NULL/nonzero pair returns 0 instead of dereferencing it. */
BCIR_NODISCARD uint32_t bcir_crc32(const uint8_t *BCIR_RESTRICT data, size_t len);

/* Validate magic + version + CRC and copy the header out. A trust boundary: every
 * field is bounds-checked, so any malformed/hostile buffer returns an error status
 * (never reads out of bounds) -- exercised by runtime/c/fuzz_streampack.c. */
BCIR_NODISCARD bcir_status bcir_sp_validate(const uint8_t *BCIR_RESTRICT data,
                                            size_t len,
                                            bcir_streampack_header *BCIR_RESTRICT hdr);

/* A zero-copy view of one lane segment (pointers into the buffer). */
typedef struct bcir_segment_view {
  const char *name; uint16_t name_len;
  uint64_t claim_id;
  uint32_t phase_id;
  uint8_t  lane;        /* bcir_lane */
  uint32_t width;
  uint32_t stride_k;
  const char *opcode; uint16_t opcode_len;
  uint16_t n_reads;  const uint8_t *reads;   /* little-endian u32[n_reads]  */
  uint16_t n_writes; const uint8_t *writes;  /* little-endian u32[n_writes] */
  const char *prefetch; uint16_t prefetch_len;  /* "" (len 0) == no prefetch */
  uint8_t  dispatch;    /* v3: bcir_dispatch (BCIR_DISPATCH_CORE on a v1/v2 pack) */
  const char *channel; uint16_t channel_len;    /* v3: "host" on a v1/v2 pack */
} bcir_segment_view;

/* Decode reads[i] / writes[i] from the raw little-endian pointers. */
uint32_t bcir_seg_read_rid(const bcir_segment_view *seg, uint16_t i);
uint32_t bcir_seg_write_rid(const bcir_segment_view *seg, uint16_t i);

/* Per-segment callback; return nonzero to stop the walk early. */
typedef int (*bcir_seg_fn)(const bcir_segment_view *seg, void *ctx);

/* Validate, then walk every lane segment in order, invoking `fn`. Bounds-checked end
 * to end: a hostile/truncated buffer returns BCIR_ERR_TRUNCATED, never an OOB read.
 *
 * Range gate (the rail-symmetry fix): every segment's `lane` must be one of the 6 valid
 * lanes and its `width` a nonzero power of two -- mirroring the Python decoder's
 * `Lane(r.u8())` raise (BCIR_ERR_LANE / BCIR_ERR_WIDTH). A v3 segment's `dispatch` code
 * must be legal (BCIR_ERR_DISPATCH). So a CRC-valid pack with an out-of-range lane/width/
 * dispatch now fails on the C rail exactly as it does on the Python rail. */
BCIR_NODISCARD bcir_status bcir_sp_for_each_segment(const uint8_t *BCIR_RESTRICT data,
                                                    size_t len, bcir_seg_fn fn,
                                                    void *ctx);

/* R11 (generation / staleness): the pack's map_gen/data_gen must match the live registry
 * generation the caller supplies, else the pack is STALE and must be rehydrated, never
 * executed silently. Pass (uint32_t)-1 for a field to skip it (a caller that only knows
 * one generation). Validates magic/version/CRC first. Returns BCIR_ERR_STALE on a mismatch.
 * Mirrors bcir/verify::verify_pack's R11. */
BCIR_NODISCARD bcir_status bcir_sp_check_generation(const uint8_t *BCIR_RESTRICT data,
                                                    size_t len,
                                                    uint32_t expected_map_gen,
                                                    uint32_t expected_data_gen);

/* R10 (stream provenance) + the range gate, over the WHOLE decoded pack (the freestanding
 * twin of verify_pack's R10). After the CRC/bounds check it enforces, on the decoded body:
 *   - every segment's claim_id resolves to a decoded trace record (no dangling/redirected id);
 *   - segment claim ids, trace claim ids, and prefetch names are unique (no ambiguous/replayed binding);
 *   - every segment's non-empty `prefetch` name resolves to a declared prefetch record;
 *   - lane/width/dispatch are in range (as bcir_sp_for_each_segment, applied to every segment);
 *   - pipeline_depth >= 1 and every prefetch `buffers` is 1 or 2 (the v2 well-formedness);
 *   - every declared trace record is present and the final record ends exactly at the CRC trailer.
 * When `expected_map_gen`/`expected_data_gen` are not (uint32_t)-1, R11 (staleness) is also
 * enforced. Returns BCIR_ERR_PROVENANCE / BCIR_ERR_STALE / BCIR_ERR_LANE / BCIR_ERR_WIDTH /
 * BCIR_ERR_DISPATCH / BCIR_ERR_TRAILING on the first violation, BCIR_OK on a clean pack.
 * Bounds-checked end to end. */
BCIR_NODISCARD bcir_status bcir_sp_verify_semantic(const uint8_t *BCIR_RESTRICT data,
                                                   size_t len,
                                                   uint32_t expected_map_gen,
                                                   uint32_t expected_data_gen);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_RUNTIME_H */
