/*===- test_tokload.c - the data loader's contracts: every refusal, and the batches' laws -------------------===
 *
 * The shard's every refusal, each one leaving the caller's shard as it was, on both widths; the loader's; and the
 * batches' laws: one epoch visits every window once, a batch is the same whether reached in order or straight
 * from its index (an exact resume), an unshuffled loader walks the windows in shard order, two keys order them
 * differently, a batch straddles two epochs, and the last sample's index is the last a batch may take. Prints
 * `test_tokload: OK`. The parity with the oracle is bcir/tests/test_c_tokload.py's, over bcir-tokload.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>

#include "bcir_digest_hw.h"
#include "bcir_sha256.h"
#include "bcir_tokload.h"

static int failures;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; }          \
  } while (0)

/* a shard of `n` tokens (t[i] = (i * 7 + seed) % vocab) at `width`, into `out`; its length */
static size_t make(uint8_t *out, uint32_t n, uint32_t vocab, uint32_t width, uint32_t seed) {
  memset(out, 0, 64);
  memcpy(out, "BTKS", 4);
  out[6] = (uint8_t)width;
  for (int b = 0; b < 4; b++) out[8 + b] = (uint8_t)(vocab >> (8 * b));
  for (int b = 0; b < 4; b++) out[16 + b] = (uint8_t)(n >> (8 * b));
  for (uint32_t i = 0; i < n; i++) {
    uint32_t t = (i * 7u + seed) % vocab;
    for (uint32_t b = 0; b < width; b++) out[64 + i * width + b] = (uint8_t)(t >> (8 * b));
  }
  bcir_sha256_digest(out + 64, (size_t)n * width, out + 32);
  return 64u + (size_t)n * width;
}

static void test_shard_refusals(void) {
  static uint8_t buf[64 + 4 * 400], bad[sizeof buf];
  for (uint32_t width = 2; width <= 4; width += 2) {
    size_t len = make(buf, 400, 1000, width, 3);
    bcir_tokshard s, keep;
    CHECK(bcir_tokshard_open(&s, buf, len, 0) == BCIR_TOKLOAD_OK && s.count == 400 && s.width == width);
    CHECK(bcir_tokshard_token(&s, 5) == 38u && s.vocab == 1000u);
    memset(&keep, 0xA5, sizeof keep);
    struct { size_t at; uint8_t value; int want; } cases[] = {
      {0, 'b', BCIR_TOKLOAD_ERR_FORMAT},  {4, 1, BCIR_TOKLOAD_ERR_FORMAT},  {6, 3, BCIR_TOKLOAD_ERR_FORMAT},
      {12, 1, BCIR_TOKLOAD_ERR_FORMAT},   {27, 1, BCIR_TOKLOAD_ERR_FORMAT}, {16, 0xFF, BCIR_TOKLOAD_ERR_FORMAT},
      {23, 1, BCIR_TOKLOAD_ERR_FORMAT},   {40, 0x5A, BCIR_TOKLOAD_ERR_DIGEST},
      {64 + 2 * width + 1, 0xFF, BCIR_TOKLOAD_ERR_TOKEN},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
      memcpy(bad, buf, len);
      bad[cases[c].at] ^= cases[c].value;
      s = keep;
      CHECK(bcir_tokshard_open(&s, bad, len, 0) == cases[c].want);
      CHECK(memcmp(&s, &keep, sizeof s) == 0); /* a refusal leaves the shard as it was */
    }
    /* a payload one byte short, the header alone, a vocab of 0, a width-2 vocab past 65,536 */
    CHECK(bcir_tokshard_open(&s, buf, len - 1, 0) == BCIR_TOKLOAD_ERR_FORMAT);
    CHECK(bcir_tokshard_open(&s, buf, 63, 0) == BCIR_TOKLOAD_ERR_FORMAT);
    memcpy(bad, buf, len);
    memset(bad + 8, 0, 4);
    CHECK(bcir_tokshard_open(&s, bad, len, 0) == BCIR_TOKLOAD_ERR_FORMAT);
    if (width == 2) {
      bad[8] = 1; bad[9] = 0; bad[10] = 1; /* 65,537 */
      CHECK(bcir_tokshard_open(&s, bad, len, 0) == BCIR_TOKLOAD_ERR_FORMAT);
    }
    CHECK(bcir_tokshard_open(NULL, buf, len, 0) == BCIR_TOKLOAD_ERR_ARG);
    CHECK(bcir_tokshard_open(&s, NULL, 5, 0) == BCIR_TOKLOAD_ERR_ARG);
    /* the hardware digest path opens what the portable one does */
    CHECK(bcir_tokshard_open(&s, buf, len, bcir_digest_probe()) == BCIR_TOKLOAD_OK);
  }
  /* an empty shard opens, and holds no window */
  size_t len = make(buf, 0, 9, 4, 0);
  bcir_tokshard s;
  CHECK(bcir_tokshard_open(&s, buf, len, 0) == BCIR_TOKLOAD_OK && s.count == 0);
  CHECK(bcir_tokshard_windows(0, 3) == 0 && bcir_tokshard_windows(3, 3) == 0 && bcir_tokshard_windows(4, 3) == 1);
  CHECK(bcir_tokshard_windows(7, 3) == 2 && bcir_tokshard_windows(5, 0) == 0);
}

static uint8_t store[3][64 + 4 * 200];
static bcir_tokshard shards[3];

static void open_three(uint32_t vocab_last) {
  static const uint32_t counts[3] = {101, 7, 200};
  for (int k = 0; k < 3; k++) {
    size_t len = make(store[k], counts[k], k == 2 ? vocab_last : 500, 4, (uint32_t)k * 11u);
    CHECK(bcir_tokshard_open(&shards[k], store[k], len, 0) == BCIR_TOKLOAD_OK);
  }
}

static void test_loader(void) {
  const uint32_t key[2] = {0x1234u, 0xabcdu}, other[2] = {0x1235u, 0xabcdu};
  uint32_t perm[256], perm2[256];
  bcir_tokload l, m;
  open_three(500);
  /* context 4: 25 + 1 + 49 windows */
  CHECK(bcir_tokload_init(&l, shards, 3, 4, key, 1, perm, 256) == BCIR_TOKLOAD_OK && l.windows == 75);
  CHECK(bcir_tokload_init(&l, shards, 3, 4, key, 1, perm, 74) == BCIR_TOKLOAD_ERR_SPACE);
  CHECK(bcir_tokload_init(&l, shards, 3, 4, key, 1, NULL, 256) == BCIR_TOKLOAD_ERR_SPACE);
  CHECK(bcir_tokload_init(&l, shards, 3, 0, key, 1, perm, 256) == BCIR_TOKLOAD_ERR_ARG);
  CHECK(bcir_tokload_init(&l, shards, 3, UINT32_MAX, key, 1, perm, 256) == BCIR_TOKLOAD_ERR_ARG);
  CHECK(bcir_tokload_init(&l, shards, 0, 4, key, 1, perm, 256) == BCIR_TOKLOAD_ERR_ARG);
  CHECK(bcir_tokload_init(&l, shards + 1, 1, 7, key, 0, NULL, 0) == BCIR_TOKLOAD_ERR_EMPTY);
  CHECK(bcir_tokload_init(&l, shards, 3, 4, key, 0, NULL, 0) == BCIR_TOKLOAD_OK); /* unshuffled: no workspace */
  /* unshuffled, the windows in shard order: window 26 is shard 2's first */
  uint32_t in[5 * 4], tg[5 * 4], in2[5 * 4], tg2[5 * 4];
  CHECK(bcir_tokload_batch(&l, 0, 1, in, tg) == BCIR_TOKLOAD_OK && in[0] == bcir_tokshard_token(&shards[0], 0));
  CHECK(bcir_tokload_batch(&l, 26, 1, in, tg) == BCIR_TOKLOAD_OK && in[0] == bcir_tokshard_token(&shards[2], 0));
  CHECK(tg[3] == bcir_tokshard_token(&shards[2], 4) && in[3] == tg[2]);
  /* shuffled: one epoch visits every window once, each row the window the permutation names */
  CHECK(bcir_tokload_init(&l, shards, 3, 4, key, 1, perm, 256) == BCIR_TOKLOAD_OK);
  unsigned char seen[75];
  memset(seen, 0, sizeof seen);
  for (uint64_t b = 0; b < 15; b++) {
    CHECK(bcir_tokload_batch(&l, b, 5, in, tg) == BCIR_TOKLOAD_OK);
    for (uint32_t r = 0; r < 5; r++) {
      uint32_t w = l.perm[b * 5 + r];
      /* windows 0-24 shard 0, 25 shard 1, 26-74 shard 2 */
      const bcir_tokshard *sh = w < 25 ? &shards[0] : w < 26 ? &shards[1] : &shards[2];
      uint64_t start = (uint64_t)(w < 25 ? w : w < 26 ? 0 : w - 26) * 4u;
      for (uint32_t k = 0; k < 4; k++) {
        CHECK(in[r * 4 + k] == bcir_tokshard_token(sh, start + k));
        CHECK(tg[r * 4 + k] == bcir_tokshard_token(sh, start + k + 1));
      }
      if (w < 75) seen[w]++;
    }
  }
  for (int w = 0; w < 75; w++) CHECK(seen[w] == 1);
  /* an exact resume: batch 18 of 4 (samples 72 .. 75, epochs 0 and 1) straight from its index on a fresh
   * loader is batch 18 reached in order */
  for (uint64_t b = 0; b <= 18; b++) CHECK(bcir_tokload_batch(&l, b, 4, in, tg) == BCIR_TOKLOAD_OK);
  CHECK(l.epoch == 1);
  CHECK(bcir_tokload_init(&m, shards, 3, 4, key, 1, perm2, 256) == BCIR_TOKLOAD_OK);
  CHECK(bcir_tokload_batch(&m, 18, 4, in2, tg2) == BCIR_TOKLOAD_OK);
  CHECK(memcmp(in, in2, 16 * sizeof in[0]) == 0 && memcmp(tg, tg2, 16 * sizeof tg[0]) == 0);
  /* and a later epoch reached straight, then an earlier one again */
  CHECK(bcir_tokload_batch(&m, 1000, 4, in2, tg2) == BCIR_TOKLOAD_OK && m.epoch == 53);
  CHECK(bcir_tokload_batch(&m, 18, 4, in2, tg2) == BCIR_TOKLOAD_OK);
  CHECK(memcmp(in, in2, 16 * sizeof in[0]) == 0 && memcmp(tg, tg2, 16 * sizeof tg[0]) == 0);
  /* another key, another order */
  CHECK(bcir_tokload_init(&m, shards, 3, 4, other, 1, perm2, 256) == BCIR_TOKLOAD_OK);
  CHECK(bcir_tokload_batch(&m, 0, 5, in2, tg2) == BCIR_TOKLOAD_OK);
  CHECK(bcir_tokload_batch(&l, 0, 5, in, tg) == BCIR_TOKLOAD_OK);
  CHECK(memcmp(in, in2, sizeof in) != 0);
  /* the last sample: index * size + size - 1 == 2^64 - 1 is the last batch; one past it is refused */
  CHECK(bcir_tokload_batch(&l, (UINT64_MAX - 4) / 5, 5, in, tg) == BCIR_TOKLOAD_OK);
  CHECK(bcir_tokload_batch(&l, (UINT64_MAX - 4) / 5 + 1, 5, in, tg) == BCIR_TOKLOAD_ERR_ARG);
  CHECK(bcir_tokload_batch(&l, UINT64_MAX, 1, in, tg) == BCIR_TOKLOAD_OK);
  CHECK(bcir_tokload_batch(&l, 0, 0, in, tg) == BCIR_TOKLOAD_ERR_ARG);
  /* one window, shuffled, its first batch the last sample's: epoch 2^64 - 1, which the loader had taken for its own
   * "no permutation yet" and so read the workspace unwritten -- here every word past any window */
  for (size_t k = 0; k < 256; k++) perm2[k] = UINT32_MAX;
  CHECK(bcir_tokload_init(&m, shards + 1, 1, 4, key, 1, perm2, 256) == BCIR_TOKLOAD_OK && m.windows == 1u);
  CHECK(bcir_tokload_batch(&m, UINT64_MAX, 1, in, tg) == BCIR_TOKLOAD_OK && perm2[0] == 0u);
  CHECK(in[0] == bcir_tokshard_token(&shards[1], 0) && tg[3] == bcir_tokshard_token(&shards[1], 4));
  CHECK(bcir_tokload_batch(&l, 0, 1, NULL, tg) == BCIR_TOKLOAD_ERR_ARG);
  /* shards of two vocabularies */
  open_three(501);
  CHECK(bcir_tokload_init(&l, shards, 3, 4, key, 1, perm, 256) == BCIR_TOKLOAD_ERR_VOCAB);
}

int main(void) {
  test_shard_refusals();
  test_loader();
  if (failures) { printf("test_tokload: %d FAILED\n", failures); return 1; }
  printf("test_tokload: OK\n");
  return 0;
}
