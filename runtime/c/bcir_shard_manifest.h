/*===- bcir_shard_manifest.h - the manifest-of-shards (BSHM, version zero) ---===
 *
 * The C twin of bcir/abi/shard_manifest.py (GEM+ roadmap G16, staged plan S3-C); the prose spec
 * is docs/kernel/BCIR_SHARD_MANIFEST_ABI.md. A StreamPack too large to ship as one artifact
 * travels as SHARDS named by digest, bound by one small manifest:
 *
 *   shard r = the canonical v4 sub-pack of the whole over segments [begin, end): those segment
 *             records verbatim, the prefetch records they name and their trace records, no
 *             blocks, the whole generation vector, under the whole's tags -- a StreamPack a node
 *             admits and runs by itself;
 *   frame   = the whole with no segments: every prefetch, block and trace record and the vector.
 *
 * Manifest layout (little-endian; every field fixed-width):
 *   header(64)  magic "BSHM" | version u16 = 0 | flags u16 = 0 | n_shards u32 | n_segments u32 |
 *               whole_length u64 | frame_length u64 | map_gen u32 | data_gen u32 | topo_gen u32 |
 *               pack_version u16 = 4 | reserved u16 | n_gens u32 | reserved[12]
 *   digests(96) whole_sha256[32] | frame_sha256[32] | registry_digest[32]
 *   shards      n_shards x (seg_begin u32 | seg_end u32 | length u64 | sha256[32])   (48 each)
 *   trailer     crc32 of every preceding byte
 *
 * Only a v4 pack with a vector, in the HYDRATED LAYOUT, shards: one trace record per segment in
 * segment order (S1), and the prefetch records the segments name appear in segment order, each
 * named at most once (S2) -- the layout every BCIR hydrator emits, which lets a shard be cut in
 * one forward pass (no search, no scratch). Reassembly is one total predicate: every blob has
 * its declared length and digest and is a pack; the whole built from the frame and the shards'
 * segment records has the declared length and digest, verifies, is in the hydrated layout, and
 * carries the manifest's tags, vector length and registry digest; and the frame and every shard
 * are byte for byte the ones this unit cuts from it. Every reassembly refusal is BCIR_ERR_SHARD.
 *
 * Freestanding: <stddef.h> + <stdint.h> (through bcir_runtime.h), no heap, no libc. Version zero:
 * no compatibility promise until a cluster exercises it.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_SHARD_MANIFEST_H
#define BCIR_SHARD_MANIFEST_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_SHM_MAGIC         "BSHM"
#define BCIR_SHM_VERSION       0u
#define BCIR_SHM_HEADER_SIZE   64u
#define BCIR_SHM_DIGESTS_SIZE  96u
#define BCIR_SHM_ENTRY_SIZE    48u
#define BCIR_SHM_FIXED         164u  /* header + digests + trailer: a manifest of no shards */
#define BCIR_SHM_SHARDS_MAX    4096u /* the declared bound */
#define BCIR_SHM_MAX_BYTES     (BCIR_SHM_FIXED + BCIR_SHM_ENTRY_SIZE * BCIR_SHM_SHARDS_MAX)
#define BCIR_SHM_BLOB_MAX      0xFFFFFFFFu /* every length fits a 32-bit target's size_t */
#define BCIR_SHM_PACK_VERSION  4u

/* A validated manifest, read in place (the digests point into `data`). */
typedef struct bcir_shm_view {
  const uint8_t *data;
  size_t len;
  uint32_t n_shards;
  uint32_t n_segments;
  uint64_t whole_length;
  uint64_t frame_length;
  uint32_t map_gen, data_gen, topo_gen;
  uint32_t n_gens;
  const uint8_t *whole_sha256;
  const uint8_t *frame_sha256;
  const uint8_t *registry_digest;
} bcir_shm_view;

typedef struct bcir_shm_entry {
  uint32_t seg_begin;
  uint32_t seg_end;
  uint64_t length;
  uint8_t sha256[32];
} bcir_shm_entry;

/* Decode and validate a manifest -- the laws in the specification's order: TRUNCATED (under
 * the fixed part), MAGIC, VERSION, CRC, RESERVED (flags and the pads), SHARD (n_shards outside
 * 1..4096), TRUNCATED/TRAILING (the size is not 164 + 48 n), SHARD (pack_version != 4 or no
 * vector), SHARD (a length over 2^32-1 or under the smallest v4 pack; a whole shorter than its
 * frame; a pack with segments that is its own frame, or one without that is not), SHARD (the
 * ranges do not partition [0, n_segments) into contiguous non-empty shards -- an empty pack is
 * one empty shard). `out` is zeroed on failure. */
BCIR_NODISCARD bcir_status bcir_shm_decode(const uint8_t *BCIR_RESTRICT data, size_t len,
                                           bcir_shm_view *BCIR_RESTRICT out);

/* The shard entry `index` of a decoded manifest (BCIR_ERR_SHARD out of range). */
BCIR_NODISCARD bcir_status bcir_shm_entry_at(const bcir_shm_view *BCIR_RESTRICT m, uint32_t index,
                                             bcir_shm_entry *BCIR_RESTRICT out);

/* The partition the seam's DistributedOrchestrator::shard() uses: ceil(n / world) segments per
 * shard, contiguous from 0 (so there may be fewer shards than ranks); an empty pack is one
 * empty shard; world 0 counts as 1. Writes *count ranges into begins/ends
 * (BCIR_ERR_NOSPACE if `cap` is short; BCIR_ERR_SHARD past the 4096-shard bound). */
BCIR_NODISCARD bcir_status bcir_shm_partition(uint32_t n_segments, uint32_t world,
                                              uint32_t *BCIR_RESTRICT begins,
                                              uint32_t *BCIR_RESTRICT ends, uint32_t cap,
                                              uint32_t *BCIR_RESTRICT count);

/* Whether `whole` may shard: a v4 pack with a vector that bcir_sp_verify_semantic accepts, in
 * the hydrated layout. BCIR_OK or BCIR_ERR_SHARD. */
BCIR_NODISCARD bcir_status bcir_shm_check_whole(const uint8_t *BCIR_RESTRICT whole, size_t len);

/* The canonical sub-pack of `whole` over segments [begin, end), and the frame. With out == NULL
 * (cap 0) the size is written to *out_len and nothing else; otherwise the bytes, CRC and all
 * (BCIR_ERR_NOSPACE if `cap` is short, before any byte is written). BCIR_ERR_SHARD when the
 * whole may not shard or the range is outside it. *out_len is zeroed on failure. */
BCIR_NODISCARD bcir_status bcir_shm_sub_pack(const uint8_t *BCIR_RESTRICT whole, size_t len,
                                             uint32_t begin, uint32_t end,
                                             uint8_t *BCIR_RESTRICT out, size_t cap,
                                             size_t *BCIR_RESTRICT out_len);
BCIR_NODISCARD bcir_status bcir_shm_frame(const uint8_t *BCIR_RESTRICT whole, size_t len,
                                          uint8_t *BCIR_RESTRICT out, size_t cap,
                                          size_t *BCIR_RESTRICT out_len);

/* The manifest binding `whole`, its `frame` and `n_shards` shards (shards[i] of shard_lens[i]
 * bytes over [begins[i], ends[i])): lengths, SHA-256 digests and the whole vector's registry
 * digest, then decoded before it is returned -- the encoder refuses what the decoder refuses.
 * The frame and shards must be the ones this unit cuts from the whole (BCIR_ERR_SHARD
 * otherwise); with out == NULL the size (164 + 48 n) is written to *out_len. */
BCIR_NODISCARD bcir_status bcir_shm_encode(const uint8_t *BCIR_RESTRICT whole, size_t whole_len,
                                           const uint8_t *BCIR_RESTRICT frame, size_t frame_len,
                                           const uint32_t *begins, const uint32_t *ends,
                                           const uint8_t *const *shards, const size_t *shard_lens,
                                           uint32_t n_shards, uint8_t *BCIR_RESTRICT out,
                                           size_t cap, size_t *BCIR_RESTRICT out_len);

/* A content-addressed store: find the blob whose SHA-256 is `digest`; return 0 and point
 * `*data` and `*len` at it (borrowed for the duration of the reassembly), or nonzero when
 * absent. */
typedef int (*bcir_shm_fetch_fn)(const uint8_t digest[32], const uint8_t **data, size_t *len,
                                 void *ctx);

/* Reassemble the whole pack into out[0..cap) from a manifest and a store, or refuse: the
 * manifest's own status, then BCIR_ERR_SHARD for every law above, BCIR_ERR_NOSPACE only once
 * the blobs prove the declared length and `cap` is short of it. *out_len is zeroed on failure. */
BCIR_NODISCARD bcir_status bcir_shm_reassemble(const uint8_t *BCIR_RESTRICT manifest,
                                               size_t manifest_len, bcir_shm_fetch_fn fetch,
                                               void *ctx, uint8_t *BCIR_RESTRICT out, size_t cap,
                                               size_t *BCIR_RESTRICT out_len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_SHARD_MANIFEST_H */
