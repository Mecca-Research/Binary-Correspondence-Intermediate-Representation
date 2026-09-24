/*===- bcir_runtime.h - freestanding BCIR StreamPack runtime --------------===
 *
 * A no-libc loader/executor for frozen StreamPack v1 and its append-only v2/v3/v4 forms
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
  BCIR_ERR_UTF8 = 14,       /* a length-prefixed wire string is not valid UTF-8 */
  BCIR_ERR_GENERATION = 15, /* a v4 generation vector is malformed: RIDs not strictly ascending,
                             * or the header map_gen/data_gen are not the vector's maxima */
  BCIR_ERR_PLAN = 16,       /* an ExecutionPlanV1 record violates a plan law (bcir_execution_plan.h):
                             * mode, streams/knee, a step's stream/duration/makespan, a lifetime's
                             * or a movement edge's shape */
  BCIR_ERR_CONTROL = 17,    /* a ControlRecordV1 violates a wire law (bcir_control_plane.h): kind,
                             * body length, scope, reason, capability, sequence, lease, the
                             * generation/expect witness, or a body law */
  BCIR_ERR_MAC = 18,        /* a ControlRecordV1 MAC is all zero, or is not the one the key makes */
  BCIR_ERR_TELEMETRY = 19,  /* a TelemetryEnvelopeV0 violates a field law
                             * (bcir_telemetry_envelope.h), or names a REQUIRED signal the
                             * consumer's table does not define */
  BCIR_ERR_RING = 20,       /* a live ring's geometry or shared state violates a ring law
                             * (bcir_ring.h): a corrupt region, a protocol violation, misuse */
  BCIR_ERR_FULL = 21,       /* a backpressure ring has no free slot: the record is refused and
                             * counted, never written over an unconsumed one */
  BCIR_ERR_BUSY = 22,       /* a ring endpoint is held by a peer (attach without a takeover) */
  BCIR_ERR_LIFETIME = 23,   /* a pack-table handle or view names no live slot at its epoch: the
                             * owner released it (a view outlived its owner), the slot was
                             * reused, or the handle was never issued (bcir_handoff.h) */
  BCIR_ERR_SHARD = 24,      /* a shard manifest violates a manifest law, or a shard set does not
                             * reassemble to the whole it declares (bcir_shard_manifest.h) */
  BCIR_ERR_PLANNER = 25     /* a K_BCIR planner input or realization record violates a planner
                             * law (bcir_kplan.h): a count, the scope, a claim, the op table */
  /* Append-only: the codes are the rails' shared names for a refusal (the Python codecs'
   * `status`), so a code is never renumbered or reused -- a new status takes the next one. */
} bcir_status;

/* zlib-compatible CRC-32 (reflected, poly 0xEDB88320). NULL is valid only with len 0;
 * an invalid NULL/nonzero pair returns 0 instead of dereferencing it. */
BCIR_NODISCARD uint32_t bcir_crc32(const uint8_t *BCIR_RESTRICT data, size_t len);

/* 1 when `s[0..n)` is well-formed UTF-8 (RFC 3629: no overlong form, no surrogate, nothing
 * above U+10FFFF) -- the one validator every wire string in the runtime is held to. */
BCIR_NODISCARD int bcir_utf8_valid(const uint8_t *BCIR_RESTRICT s, size_t n);

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

/* Per-generation callback (v4); return nonzero to stop the walk early. */
typedef int (*bcir_gen_fn)(const bcir_generation_view *g, void *ctx);

/* Walk the v4 per-resource generation vector in RID order. Runs the full semantic
 * verification first (the vector is the body's tail, so exact consumption locates it);
 * a v1-v3 pack has no vector and the walk invokes nothing. */
BCIR_NODISCARD bcir_status bcir_sp_for_each_generation(const uint8_t *BCIR_RESTRICT data,
                                                       size_t len, bcir_gen_fn fn,
                                                       void *ctx);

/* R11 per resource (S0-2, StreamPack v4): `live[0..n_live)` is the caller's live registry
 * -- one (rid, map_gen, data_gen) per declared resource, any order. The pack is STALE
 * (BCIR_ERR_STALE) unless every vector entry names a live resource with exactly its
 * generation AND every live resource has an entry: a resource that moved while another
 * still held the maximum, or one declared after hydration, is stale even though the
 * header maxima agree. A pack with no vector is stale against any registry that
 * declares resources (a v1-v3 artifact must be rehydrated), and clean against none.
 * Validates magic/version/CRC and the semantic laws first (their statuses pass through).
 * Mirrors bcir/verify::verify_pack's per-resource R11 and the law rail's walk. */
BCIR_NODISCARD bcir_status bcir_sp_check_generation_vector(
    const uint8_t *BCIR_RESTRICT data, size_t len,
    const bcir_generation_view *BCIR_RESTRICT live, size_t n_live);

/* R10 (stream provenance) + the range gate, over the WHOLE decoded pack (the freestanding
 * twin of verify_pack's R10). After the CRC/bounds check it enforces, on the decoded body:
 *   - every segment's claim_id resolves to a decoded trace record (no dangling/redirected id);
 *   - segment claim ids, trace claim ids, and prefetch names are unique (no ambiguous/replayed binding);
 *   - every segment's non-empty `prefetch` name resolves to a declared prefetch record;
 *   - lane/width/dispatch are in range (as bcir_sp_for_each_segment, applied to every segment);
 *   - pipeline_depth >= 1 and every prefetch `buffers` is 1 or 2 (the v2 well-formedness);
 *   - a v4 generation vector has strictly ascending RIDs and the header map_gen/data_gen are
 *     its maxima (BCIR_ERR_GENERATION otherwise);
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
