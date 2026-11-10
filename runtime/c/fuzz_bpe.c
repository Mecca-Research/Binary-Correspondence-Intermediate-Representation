/*===- fuzz_bpe.c - libFuzzer target for the C BPE tokenizer (PORT-2) ---------------------===
 *
 * A tokenizer.json is a file anyone hands a model runner. Every input is tried two ways:
 *
 *   1. as a tokenizer.json: a load either refuses with a status or yields a tokenizer, which then
 *      encodes a few fixed texts and the input itself, decodes ids drawn from the input, and is
 *      held to its contracts -- the count-only encode counts what the full one writes, a text never
 *      yields more ids than octets, a decode's measure is what it writes, and with no specials
 *      (whose ids may name a vocab token's text) decode(encode(text)) == text;
 *   2. as text and ids for a fixed tokenizer (the mini one, whose special ids are fresh, so the
 *      round trip holds whatever the text): text that is not UTF-8 is refused, anything else
 *      round-trips.
 *
 * Seeds are the corpus bcir/tests/test_c_bpe_tokenizer.py compares against the oracle.
 *===----------------------------------------------------------------------===*/
#include "bcir_bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP 4096u

static void die(const char *what) {
  fprintf(stderr, "fuzz_bpe: %s\n", what);
  abort();
}

/* encode -> contracts -> decode; `lossless` asks for decode(encode(text)) == text. */
static void round_trip(const bcir_bpe *t, const uint8_t *text, size_t len, int lossless) {
  static uint32_t ids[CAP];
  static uint8_t out[4u * CAP + 16u];
  size_t count = 0, counted = 0, w = 0, measured = 0;
  if (len > CAP) len = CAP;
  bcir_bpe_status st = bcir_bpe_encode(t, text, len, ids, CAP, &count);
  bcir_bpe_status probe = bcir_bpe_encode(t, text, len, NULL, 0, &counted);
  if (st == BCIR_BPE_ERR_NOT_UTF8) {
    if (probe != BCIR_BPE_ERR_NOT_UTF8 || count || counted) die("a refused text counted ids");
    return;
  }
  if (st != BCIR_BPE_OK) die("a UTF-8 text was not encoded");
  if (count > len) die("more ids than octets");
  if (counted != count || (count ? probe != BCIR_BPE_ERR_NOSPACE : probe != BCIR_BPE_OK))
    die("the count-only encode disagrees");
  st = bcir_bpe_decode(t, ids, count, NULL, 0, &measured);
  if (st != BCIR_BPE_OK && st != BCIR_BPE_ERR_NOSPACE && st != BCIR_BPE_ERR_NOT_UTF8) die("an encoded id is unknown");
  if (st == BCIR_BPE_ERR_NOT_UTF8) {
    if (lossless) die("a lossless round trip decoded to something that is not UTF-8");
    return;
  }
  if (measured > sizeof out) return; /* specials render longer than they matched only by their text */
  if (bcir_bpe_decode(t, ids, count, out, sizeof out, &w) != BCIR_BPE_OK || w != measured)
    die("the decode measure disagrees");
  if (lossless && (w != len || memcmp(out, text, len) != 0)) die("decode(encode(text)) != text");
}

/* ids from input octets, two at a time, mostly inside the id range */
static void decode_ids(const bcir_bpe *t, const uint8_t *data, size_t size, uint32_t range) {
  static uint32_t ids[CAP / 2u];
  static uint8_t out[16384];
  size_t n = 0, w = 0, measured = 0;
  for (size_t i = 0; i + 1u < size && n < CAP / 2u; i += 2u) {
    uint32_t v = (uint32_t)data[i] | (uint32_t)data[i + 1u] << 8;
    ids[n++] = range ? v % (range + 4u) : v;
  }
  bcir_bpe_status st = bcir_bpe_decode(t, ids, n, NULL, 0, &measured);
  if (st != BCIR_BPE_OK && st != BCIR_BPE_ERR_NOSPACE) return; /* unknown id, or a run not UTF-8 */
  if (measured > sizeof out) return;
  if (bcir_bpe_decode(t, ids, n, out, sizeof out, &w) != BCIR_BPE_OK || w != measured)
    die("the decode measure disagrees");
}

static const char MINI_HEAD[] = "{\"model\":{\"type\":\"BPE\",\"vocab\":{";
static const char MINI_TAIL[] =
    ",\"he\":256,\"ll\":257,\"hell\":258,\"hello\":259},\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\"]},"
    "\"added_tokens\":[{\"content\":\"<bos>\",\"id\":260},{\"content\":\"<eos>\",\"id\":261},"
    "{\"content\":\"<|\",\"id\":262},{\"content\":\"\\u00e9\\u00e9\",\"id\":263}]}";

static const bcir_bpe *mini(void) {
  static bcir_bpe *t;
  static char json[8192];
  if (t) return t;
  size_t n = 0, shifted = 0;
  memcpy(json, MINI_HEAD, sizeof MINI_HEAD - 1u);
  n = sizeof MINI_HEAD - 1u;
  for (unsigned b = 0; b < 256u; b++) {
    int printable = (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || (b >= 0xAEu && b <= 0xFFu);
    unsigned cp = printable ? b : 256u + (unsigned)shifted++;
    n += (size_t)snprintf(json + n, sizeof json - n, "%s\"\\u%04X\":%u", b ? "," : "", cp, b);
  }
  memcpy(json + n, MINI_TAIL, sizeof MINI_TAIL - 1u);
  n += sizeof MINI_TAIL - 1u;
  if (bcir_bpe_load((const uint8_t *)json, n, NULL, &t, NULL) != BCIR_BPE_OK) die("the mini tokenizer");
  return t;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static const char *const texts[] = {"hello world", " <bos>hello  ", "a\tb\nc", "\xC3\xA9\xE6\x97\xA5\xF0\x9F\x98\x80"};
  bcir_bpe *t = (bcir_bpe *)&t;
  bcir_bpe_diag d;
  bcir_bpe_status st = bcir_bpe_load(data, size, NULL, &t, &d);
  if (st == BCIR_BPE_OK) {
    int lossless = bcir_bpe_special_count(t) == 0u;
    for (size_t i = 0; i < sizeof texts / sizeof texts[0]; i++)
      round_trip(t, (const uint8_t *)texts[i], strlen(texts[i]), lossless);
    round_trip(t, data, size, lossless);
    decode_ids(t, data, size, (uint32_t)(bcir_bpe_vocab_size(t) + bcir_bpe_special_count(t)));
    bcir_bpe_free(t);
  } else if (t != NULL || d.status != st) {
    die("a refused load kept a tokenizer or misreported");
  }
  const bcir_bpe *m = mini();
  round_trip(m, data, size, 1);
  decode_ids(m, data, size, 264u);
  return 0;
}
