/*===- bcir_tokload.c - token shards and the batches over them -----------------------------------------===
 *
 * See bcir_tokload.h. The order of the checks is the oracle's, so a shard wrong in two ways is refused for the
 * same one on both rails.
 *===----------------------------------------------------------------------===*/
#include "bcir_tokload.h"

#include "bcir_digest_hw.h"
#include "bcir_rng.h"
#include "bcir_sha256.h"

static uint32_t rd16(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static uint32_t rd32(const uint8_t *p) { return rd16(p) | rd16(p + 2) << 16; }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }

uint32_t bcir_tokshard_token(const bcir_tokshard *shard, uint64_t i) {
  const uint8_t *p = shard->payload + (size_t)i * shard->width;
  return shard->width == 2u ? rd16(p) : rd32(p);
}

uint64_t bcir_tokshard_windows(uint64_t count, uint32_t context) {
  return context && count > context ? (count - 1u) / context : 0u;
}

int bcir_tokshard_open(bcir_tokshard *shard, const uint8_t *data, size_t length, unsigned digest_features) {
  if (!shard || (!data && length)) return BCIR_TOKLOAD_ERR_ARG;
  if (length < BCIR_TOKSHARD_HEADER || data[0] != 'B' || data[1] != 'T' || data[2] != 'K' || data[3] != 'S')
    return BCIR_TOKLOAD_ERR_FORMAT;
  uint32_t version = rd16(data + 4), width = rd16(data + 6), vocab = rd32(data + 8);
  uint64_t count = rd64(data + 16);
  int reserved = rd32(data + 12) != 0u;
  for (unsigned i = 24; i < 32; i++) reserved |= data[i] != 0u;
  if (version != 0u || (width != 2u && width != 4u) || reserved) return BCIR_TOKLOAD_ERR_FORMAT;
  if (vocab == 0u || (width == 2u && vocab > 65536u)) return BCIR_TOKLOAD_ERR_FORMAT;
  /* the count bounded by what the length holds before it is multiplied: no wrap on a 32-bit size_t */
  size_t body = length - BCIR_TOKSHARD_HEADER;
  if (count > body / width || (size_t)count * width != body) return BCIR_TOKLOAD_ERR_FORMAT;
  /* the shard is the caller's only once it is whole: a refusal leaves *shard as it was */
  bcir_tokshard s;
  s.payload = data + BCIR_TOKSHARD_HEADER;
  s.count = count;
  s.width = width;
  s.vocab = vocab;
  uint32_t worst = 0;
  for (uint64_t i = 0; i < count; i++) {
    uint32_t t = bcir_tokshard_token(&s, i);
    if (t > worst) worst = t;
  }
  if (count && worst >= vocab) return BCIR_TOKLOAD_ERR_TOKEN;
  bcir_sha256_digest_hw(s.payload, body, s.digest, digest_features);
  if (!bcir_sha256_equal(s.digest, data + 32, 32u)) return BCIR_TOKLOAD_ERR_DIGEST;
  *shard = s;
  return BCIR_TOKLOAD_OK;
}

int bcir_tokload_init(bcir_tokload *loader, const bcir_tokshard *shards, size_t n, uint32_t context,
                      const uint32_t key[2], int shuffle, uint32_t *perm, size_t perm_capacity) {
  if (!loader || !shards || !n || !key) return BCIR_TOKLOAD_ERR_ARG;
  if (context == 0u || context == UINT32_MAX) return BCIR_TOKLOAD_ERR_ARG;
  for (size_t s = 1; s < n; s++)
    if (shards[s].vocab != shards[0].vocab) return BCIR_TOKLOAD_ERR_VOCAB;
  uint64_t windows = 0;
  for (size_t s = 0; s < n; s++) {
    windows += bcir_tokshard_windows(shards[s].count, context);
    if (windows > UINT32_MAX) return BCIR_TOKLOAD_ERR_SPACE;
  }
  if (!windows) return BCIR_TOKLOAD_ERR_EMPTY;
  if (shuffle && (!perm || perm_capacity < windows)) return BCIR_TOKLOAD_ERR_SPACE;
  loader->shards = shards;
  loader->n_shards = n;
  loader->context = context;
  loader->key[0] = key[0];
  loader->key[1] = key[1];
  loader->shuffle = shuffle != 0;
  loader->windows = (uint32_t)windows;
  loader->perm = perm;
  loader->epoch = UINT64_MAX;
  return BCIR_TOKLOAD_OK;
}

/* window w (< windows): its shard and first token, in shard order */
static void window_start(const bcir_tokload *l, uint32_t w, const bcir_tokshard **shard, uint64_t *start) {
  for (size_t s = 0;; s++) {
    uint64_t here = bcir_tokshard_windows(l->shards[s].count, l->context);
    if (w < here) {
      *shard = &l->shards[s];
      *start = (uint64_t)w * l->context;
      return;
    }
    w -= (uint32_t)here;
  }
}

int bcir_tokload_batch(bcir_tokload *loader, uint64_t index, uint32_t size, uint32_t *inputs, uint32_t *targets) {
  if (!loader || !size || !inputs || !targets) return BCIR_TOKLOAD_ERR_ARG;
  /* samples index * size .. index * size + size - 1, the last no past 2^64 - 1 */
  if (index > (UINT64_MAX - (size - 1u)) / size) return BCIR_TOKLOAD_ERR_ARG;
  uint64_t first = index * size;
  size_t t = loader->context;
  for (uint32_t r = 0; r < size; r++) {
    uint64_t p = first + r, epoch = p / loader->windows;
    uint32_t j = (uint32_t)(p % loader->windows), w = j;
    if (loader->shuffle) {
      if (loader->epoch != epoch) {
        uint64_t at = 0;
        bcir_rng_permutation(loader->key, epoch, &at, loader->perm, loader->windows);
        loader->epoch = epoch;
      }
      w = loader->perm[j];
    }
    const bcir_tokshard *shard;
    uint64_t start;
    window_start(loader, w, &shard, &start);
    for (size_t k = 0; k < t; k++) {
      inputs[(size_t)r * t + k] = bcir_tokshard_token(shard, start + k);
      targets[(size_t)r * t + k] = bcir_tokshard_token(shard, start + k + 1u);
    }
  }
  return BCIR_TOKLOAD_OK;
}
