/*===- bcir_tokload.h - token shards and the batches over them: the native data loader -----------------===
 *
 * The C subset audit's native data loader (docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md), the twin of
 * bcir/frontends/models/token_shards.py, which states the format and the order; in short:
 *
 *   shard (BTKS, version 0), little-endian:
 *     header(64)  magic "BTKS" | version u16 = 0 | width u16 (2 or 4) | vocab u32 | reserved u32 = 0 |
 *                 count u64 | reserved[8] = 0 | sha256[32] of the payload (the shard's generation)
 *     payload     count token ids, `width` bytes each, every one below vocab
 *   windows       T + 1 tokens; shard s's window w starts at token w * T, w < (len(s) - 1) / T; numbered in
 *                 shard order, N of them
 *   sample p      window perm_e[p mod N] of epoch e = p / N, perm_e the RNG's Fisher-Yates permutation of
 *                 [0, N) under the loader's key, stream e, from element 0 (bcir_rng_permutation), or the
 *                 identity, unshuffled
 *   batch b       samples [b * B, (b + 1) * B): inputs the first T tokens of each, targets the last T, written
 *                 straight into the trainer's arrays (bcir_decoder_run's `tokens` and `targets`)
 *
 * bcir_tokshard_open checks a shard whole -- header, exact length, every token, the digest (on the hardware path
 * `digest_features` allows: bcir_digest_hw.h) -- before any token is read, and reads it in place: the caller maps
 * or reads the file and keeps the bytes alive. A batch is a pure function of its index, so a run resumes exactly.
 *
 * Freestanding: <stddef.h> and <stdint.h>, no heap, no libc. The loader's one workspace, the permutation, is
 * the caller's (N words when shuffled); the hosted bcir-tokload (bcir_tokload_tool.c) maps shards from files.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_TOKLOAD_H
#define BCIR_TOKLOAD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_TOKSHARD_HEADER 64u

enum {
  BCIR_TOKLOAD_OK = 0,
  BCIR_TOKLOAD_ERR_ARG = 1,    /* a NULL, a zero size, a context of 0 or 2^32 - 1, a batch past 2^64 samples */
  BCIR_TOKLOAD_ERR_FORMAT = 2, /* not a BTKS v0 shard: magic, version, width, reserved, vocab, length */
  BCIR_TOKLOAD_ERR_TOKEN = 3,  /* a token id at or past the vocabulary */
  BCIR_TOKLOAD_ERR_DIGEST = 4, /* a payload that is not its SHA-256 */
  BCIR_TOKLOAD_ERR_EMPTY = 5,  /* no shard holds a window */
  BCIR_TOKLOAD_ERR_SPACE = 6,  /* more windows than 2^32 - 1, or than the permutation's workspace */
  BCIR_TOKLOAD_ERR_VOCAB = 7   /* shards of different vocabularies */
};

typedef struct bcir_tokshard {
  const uint8_t *payload; /* count * width bytes, inside the caller's shard */
  uint64_t count;
  uint32_t vocab;
  uint32_t width;
  uint8_t digest[32];
} bcir_tokshard;

int bcir_tokshard_open(bcir_tokshard *shard, const uint8_t *data, size_t length, unsigned digest_features);
/* Token `i` (< count) of an opened shard. */
uint32_t bcir_tokshard_token(const bcir_tokshard *shard, uint64_t i);
/* The windows of T + 1 tokens a shard of `count` tokens has. */
uint64_t bcir_tokshard_windows(uint64_t count, uint32_t context);

typedef struct bcir_tokload {
  const bcir_tokshard *shards;
  size_t n_shards;
  uint32_t context;
  uint32_t key[2];
  int shuffle;
  uint32_t windows;
  uint32_t *perm;   /* the caller's workspace: `windows` words when shuffled */
  uint64_t epoch;   /* the epoch perm holds; UINT64_MAX before the first */
} bcir_tokload;

/* A loader over `n` opened shards. `perm`/`perm_capacity` (in words) are the shuffle's workspace, unused (and
 * may be NULL) unshuffled. */
int bcir_tokload_init(bcir_tokload *loader, const bcir_tokshard *shards, size_t n, uint32_t context,
                      const uint32_t key[2], int shuffle, uint32_t *perm, size_t perm_capacity);
/* Batch `index` of `size` samples: size * context tokens into each of `inputs` and `targets`. */
int bcir_tokload_batch(bcir_tokload *loader, uint64_t index, uint32_t size, uint32_t *inputs, uint32_t *targets);

#ifdef __cplusplus
}
#endif

#endif
