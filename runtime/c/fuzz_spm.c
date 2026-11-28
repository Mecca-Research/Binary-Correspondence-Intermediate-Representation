/*===- fuzz_spm.c - libFuzzer target for the C SentencePiece tokenizer (PORT-2b) -----------===
 *
 * A tokenizer.model is a file anyone hands a model runner. Every input is tried two ways:
 *
 *   1. as a tokenizer.model: a load either refuses with a status (keeping nothing) or yields a
 *      tokenizer, which encodes a few fixed texts and the input itself, with and without the dummy
 *      prefix, and decodes ids drawn from the input. The count-only encode must count what the full
 *      one writes, a text never yields more than 3 * len + 3 ids, a decode's measure is what it
 *      writes, and every decode is UTF-8 (the replacement decoder emits nothing else);
 *   2. as text and ids for a fixed model with a byte alphabet, under the same contracts.
 *
 * Seeds are the corpus bcir/tests/test_c_spm_tokenizer.py compares against the oracle.
 *===----------------------------------------------------------------------===*/
#include "bcir_jer.h"
#include "bcir_spm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP 2048u

static void die(const char *what) {
  fprintf(stderr, "fuzz_spm: %s\n", what);
  abort();
}

static void check_decode(const bcir_spm *t, const uint32_t *ids, size_t n) {
  static uint8_t out[65536];
  size_t measured = 0, w = 0;
  bcir_jer_diag jd;
  bcir_spm_status st = bcir_spm_decode(t, ids, n, NULL, 0, &measured);
  if (st == BCIR_SPM_ERR_UNKNOWN_ID) return;
  if (st != BCIR_SPM_OK && st != BCIR_SPM_ERR_NOSPACE) die("a decode measure failed");
  if (measured > sizeof out) return;
  if (bcir_spm_decode(t, ids, n, out, sizeof out, &w) != BCIR_SPM_OK || w != measured)
    die("the decode measure disagrees");
  if (bcir_jer_validate_utf8(out, w, &jd) != BCIR_JER_OK) die("a decode is not UTF-8");
}

static void round_trip(const bcir_spm *t, const uint8_t *text, size_t len, int prefix) {
  static uint32_t ids[3u * CAP + 3u];
  size_t count = 0, counted = 0;
  if (len > CAP) len = CAP;
  bcir_spm_status st = bcir_spm_encode(t, text, len, prefix, ids, 3u * CAP + 3u, &count);
  bcir_spm_status probe = bcir_spm_encode(t, text, len, prefix, NULL, 0, &counted);
  if (st == BCIR_SPM_ERR_NOT_UTF8) {
    if (probe != BCIR_SPM_ERR_NOT_UTF8 || count || counted) die("a refused text counted ids");
    return;
  }
  if (st != BCIR_SPM_OK) die("a UTF-8 text was not encoded");
  if (count > 3u * len + 3u) die("more ids than the bound");
  if (counted != count || (count ? probe != BCIR_SPM_ERR_NOSPACE : probe != BCIR_SPM_OK))
    die("the count-only encode disagrees");
  for (size_t i = 0; i < count; i++)
    if (ids[i] >= bcir_spm_piece_count(t)) die("an encoded id names no piece");
  check_decode(t, ids, count);
}

static void decode_ids(const bcir_spm *t, const uint8_t *data, size_t size) {
  static uint32_t ids[CAP];
  size_t n = 0, range = bcir_spm_piece_count(t) + 2u;
  for (size_t i = 0; i + 1u < size && n < CAP; i += 2u)
    ids[n++] = (uint32_t)(((size_t)data[i] | (size_t)data[i + 1u] << 8) % range);
  check_decode(t, ids, n);
}

/* A fixed model: <unk>, <s>, the byte alphabet, and a few scored pieces. */
static const bcir_spm *fixed(void) {
  static bcir_spm *t;
  static uint8_t model[16384];
  static const struct { const char *text; float score; unsigned type; } rows[] = {
      {"<unk>", 0.0f, 2}, {"<s>", 0.0f, 3}, {"\xE2\x96\x81", -1.0f, 1}, {"\xE2\x96\x81h", -2.0f, 1},
      {"\xE2\x96\x81he", -1.5f, 1}, {"ab", -1.0f, 1}, {"abc", -1.0f, 4}, {"bc", -0.5f, 5}};
  size_t n = 0;
  if (t) return t;
  for (unsigned r = 0; r < 256u + sizeof rows / sizeof rows[0]; r++) {
    char byte[8];
    const char *text = r < sizeof rows / sizeof rows[0] ? rows[r].text : byte;
    float score = r < sizeof rows / sizeof rows[0] ? rows[r].score : 0.0f;
    unsigned type = r < sizeof rows / sizeof rows[0] ? rows[r].type : 6u;
    uint32_t bits;
    if (r >= sizeof rows / sizeof rows[0]) snprintf(byte, sizeof byte, "<0x%02X>", r - (unsigned)(sizeof rows / sizeof rows[0]));
    size_t tl = strlen(text), msg = 2u + tl + 5u + 2u;
    if (n + msg + 2u > sizeof model || tl > 100u) die("the fixed model outgrew its buffer");
    memcpy(&bits, &score, 4);
    model[n++] = 0x0A;
    model[n++] = (uint8_t)msg;
    model[n++] = 0x0A;
    model[n++] = (uint8_t)tl;
    memcpy(model + n, text, tl);
    n += tl;
    model[n++] = 0x15;
    for (unsigned k = 0; k < 4u; k++) model[n++] = (uint8_t)(bits >> (8u * k));
    model[n++] = 0x18;
    model[n++] = (uint8_t)type;
  }
  if (bcir_spm_load(model, n, NULL, &t, NULL) != BCIR_SPM_OK) die("the fixed model");
  return t;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static const char *const texts[] = {"hello world", " he abc  bc", "", "a\tb\nc", "\xC3\xA9\xE6\x97\xA5\xF0\x9F\x98\x80"};
  bcir_spm *t = (bcir_spm *)&t;
  bcir_spm_diag d;
  bcir_spm_status st = bcir_spm_load(data, size, NULL, &t, &d);
  if (st == BCIR_SPM_OK) {
    for (size_t i = 0; i < sizeof texts / sizeof texts[0]; i++) {
      round_trip(t, (const uint8_t *)texts[i], strlen(texts[i]), 1);
      round_trip(t, (const uint8_t *)texts[i], strlen(texts[i]), 0);
    }
    round_trip(t, data, size, 1);
    decode_ids(t, data, size);
    bcir_spm_free(t);
  } else if (t != NULL || d.status != st) {
    die("a refused load kept a tokenizer or misreported");
  }
  const bcir_spm *f = fixed();
  round_trip(f, data, size, 1);
  round_trip(f, data, size, 0);
  decode_ids(f, data, size);
  return 0;
}
