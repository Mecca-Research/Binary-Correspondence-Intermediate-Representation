/*===- fuzz_tokload.c - libFuzzer target: a hostile token shard, and the batches over what it opens ---------===
 *
 * The input's first octet chooses the loader (its low bits the context, then the batch size, shuffling and the
 * digest's path), the rest is the shard. A shard opens only whole -- so its digest must hold, which a fuzzer
 * rarely finds -- and the input is therefore also tried with its digest recomputed, which reaches the token checks
 * and the loader. Whatever opens must give batches whose every token is below the vocabulary; a refused shard
 * must leave the caller's untouched. ASan/UBSan catch any read past the shard.
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_digest_hw.h"
#include "bcir_sha256.h"
#include "bcir_tokload.h"

static void exercise(const uint8_t *data, size_t size, uint8_t control) {
  bcir_tokshard s, keep;
  memset(&keep, 0x5A, sizeof keep);
  s = keep;
  unsigned features = (control & 0x80u) ? bcir_digest_probe() : 0u;
  int rc = bcir_tokshard_open(&s, data, size, features);
  if (rc != BCIR_TOKLOAD_OK) {
    if (memcmp(&s, &keep, sizeof s) != 0) abort();
    return;
  }
  for (uint64_t i = 0; i < s.count; i++)
    if (bcir_tokshard_token(&s, i) >= s.vocab) abort();
  uint32_t context = 1u + (control & 7u), batch = 1u + ((control >> 3) & 3u);
  uint64_t windows = bcir_tokshard_windows(s.count, context);
  if (!windows || windows > 4096u) return;
  static uint32_t perm[4096], in[4 * 8], tg[4 * 8];
  const uint32_t key[2] = {control, (uint32_t)size};
  bcir_tokload l;
  if (bcir_tokload_init(&l, &s, 1, context, key, (control >> 5) & 1u, perm, 4096u) != BCIR_TOKLOAD_OK) abort();
  for (uint64_t b = 0; b < 3; b++) {
    if (bcir_tokload_batch(&l, b * 977u, batch, in, tg) != BCIR_TOKLOAD_OK) abort();
    for (uint32_t k = 0; k < batch * context; k++)
      if (in[k] >= s.vocab || tg[k] >= s.vocab) abort();
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 1) return 0;
  uint8_t control = data[0];
  exercise(data + 1, size - 1, control);
  if (size - 1 >= BCIR_TOKSHARD_HEADER && size - 1 <= 65536u) {
    static uint8_t fixed[65536];
    memcpy(fixed, data + 1, size - 1);
    bcir_sha256_digest(fixed + BCIR_TOKSHARD_HEADER, size - 1 - BCIR_TOKSHARD_HEADER, fixed + 32);
    exercise(fixed, size - 1, control);
  }
  return 0;
}
