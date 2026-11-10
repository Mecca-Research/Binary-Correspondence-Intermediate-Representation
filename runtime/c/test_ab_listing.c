/*===- test_ab_listing.c - the C BCAB listing's contracts ------------------------===
 *
 * The octets of `list`, `hexdump` and the payload views are held to the oracle by the Python corpus
 * (bcir/tests/test_c_bundle_listing.py, which tools/c/sections/bundle_listing.sh runs against the
 * built bcir-bundle). This harness holds what a corpus of valid bundles cannot show: a corrupted
 * bundle reaching no sink, a sink that refuses, the width bounds, the text views' line, edge and word
 * rules over Python's own whitespace and line breaks, and a pack's mnemonics. It takes a valid bundle
 * (written by the section) and prints OK, or names the failure.
 *
 *   usage: test_ab_listing BUNDLE
 *===----------------------------------------------------------------------===*/
#include "bcir_ab_listing.h"
#include "bcir_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, what)                                                       \
  do {                                                                          \
    if (!(cond)) { fprintf(stderr, "FAIL %s (line %d)\n", what, __LINE__); failures++; } \
  } while (0)

typedef struct { char text[1 << 22]; size_t n, calls, refuse_after; } sink_t;  /* 4 MiB: a width-1 dump */

static int collect(void *ctx, const char *text, size_t len) {
  sink_t *s = (sink_t *)ctx;
  s->calls++;
  if (s->refuse_after && s->calls > s->refuse_after) return 1;
  if (s->n + len >= sizeof s->text) return 1;
  memcpy(s->text + s->n, text, len);
  s->n += len;
  s->text[s->n] = '\0';
  return 0;
}

static sink_t *sink_new(size_t refuse_after) {
  sink_t *s = (sink_t *)calloc(1, sizeof *s);
  if (!s) { fprintf(stderr, "out of memory\n"); exit(2); }
  s->refuse_after = refuse_after;
  return s;
}

static void test_a_corrupt_bundle_prints_nothing(const uint8_t *good, size_t len) {
  uint8_t *bad = (uint8_t *)malloc(len);
  if (!bad) exit(2);
  for (size_t at = 0; at < len; at += len / 7u + 1u) {
    memcpy(bad, good, len);
    bad[at] ^= 0x20u;
    bcir_ab_view view;
    bcir_ab_status opened = bcir_ab_open(bad, len, &view);
    sink_t *s = sink_new(0);
    CHECK(opened != BCIR_AB_OK, "a flipped octet is refused (a CRC or the digest covers it)");
    CHECK(bcir_ab_format_listing(bad, len, collect, s) == opened && s->calls == 0u,
          "the listing returns the reader's refusal and prints nothing");
    CHECK(bcir_ab_format_hexdump(bad, len, 16, collect, s) == opened && s->calls == 0u,
          "the dump returns the reader's refusal and prints nothing");
    free(s);
  }
  free(bad);
}

static void test_the_sink_and_width_contracts(const uint8_t *good, size_t len) {
  sink_t *all = sink_new(0), *once = sink_new(1), *none = sink_new(0);
  CHECK(bcir_ab_format_hexdump(good, len, 1, collect, all) == BCIR_AB_OK, "a width-1 dump");
  CHECK(all->calls > 1u, "the dump arrives in more than one chunk");
  CHECK(strncmp(all->text, "# header 'header' offset=0x00000000 length=128\n00000000  42  |B|\n",
                64) == 0, "the dump's first rows");
  CHECK(bcir_ab_format_hexdump(good, len, 1, collect, once) == BCIR_AB_ERR_OUTPUT &&
            once->calls == 2u, "a refusing sink stops the dump");
  CHECK(bcir_ab_format_hexdump(good, len, 0, collect, none) == BCIR_AB_ERR_INVALID &&
            bcir_ab_format_hexdump(good, len, 65, collect, none) == BCIR_AB_ERR_INVALID &&
            none->calls == 0u, "the width is in [1, 64]");
  CHECK(bcir_ab_format_listing(good, len, NULL, NULL) == BCIR_AB_ERR_INVALID, "no sink");
  sink_t *list = sink_new(0);
  CHECK(bcir_ab_format_listing(good, len, collect, list) == BCIR_AB_OK, "the listing");
  CHECK(strncmp(list->text, "BCAB v1 bytes=", 14) == 0 && strstr(list->text, "\nvariants:\n"),
        "the listing's heading");
  free(all); free(once); free(none); free(list);
}

static void test_the_text_views(void) {
  /* A text with every rule in it: CRLF, NEL, LS, IS1/IS4, NBSP, ideographic space, FF, comments by
   * each prefix, a lone slash, CR CR, and an unterminated last line (Python's answer, worked out
   * with str.splitlines / strip / split on 3.11 and 3.15 alike). */
  static const char text[] =
      "  mov a, b\r\n# c\n\xC2\x85; x\nnop\xE2\x80\xA8.l q\xE3\x80\x80tail\x0c" "add\x1c  b\x1f"
      "c\n// z\n/x y\n\xC2\xA0\xE2\x80\x80sub\xC2\xA0r1\r\rlast";
  sink_t *s = sink_new(0);
  CHECK(bcir_ab_text_mnemonics((const uint8_t *)text, sizeof text - 1u, collect, s) == BCIR_AB_OK,
        "text mnemonics");
  CHECK(strcmp(s->text, "mov\nnop\nadd\nb\n/x\nsub\nlast\n") == 0, "the first word of each line");
  free(s);
  s = sink_new(0);
  CHECK(bcir_ab_text_mnemonics((const uint8_t *)"# only\n\n   \n", 12, collect, s) == BCIR_AB_OK &&
            s->n == 0u, "no word, no output");
  free(s);
  s = sink_new(0);
  CHECK(bcir_ab_text_listing((const uint8_t *)"a\nb", 3, collect, s) == BCIR_AB_OK &&
            strcmp(s->text, "a\nb\n") == 0, "a missing final newline is added");
  free(s);
  s = sink_new(0);
  CHECK(bcir_ab_text_listing((const uint8_t *)"a\n", 2, collect, s) == BCIR_AB_OK &&
            strcmp(s->text, "a\n") == 0, "a present one is not doubled");
  free(s);
}

/* A v1 pack with no records: header, an empty source plan, the CRC. */
static size_t empty_pack(uint8_t out[70]) {
  memset(out, 0, 70);
  memcpy(out, "BSPK", 4);
  out[4] = 1; /* version 1 */
  uint32_t crc = bcir_crc32(out, 66);
  for (int i = 0; i < 4; i++) out[66 + i] = (uint8_t)(crc >> (8 * i));
  return 70;
}

static void test_the_pack_mnemonics(void) {
  uint8_t pack[70];
  size_t n = empty_pack(pack);
  sink_t *s = sink_new(0);
  CHECK(bcir_ab_pack_mnemonics(pack, n, collect, s) == BCIR_AB_OK && strcmp(s->text, "\n") == 0,
        "a pack with no segment shows one empty line");
  free(s);
  pack[69] ^= 1u;
  s = sink_new(0);
  CHECK(bcir_ab_pack_mnemonics(pack, n, collect, s) == BCIR_AB_ERR_PAYLOAD && s->calls == 0u,
        "a refused pack shows nothing");
  free(s);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_ab_listing BUNDLE\n");
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) return 2;
  static uint8_t data[1 << 20];
  size_t len = fread(data, 1, sizeof data, f);
  fclose(f);
  bcir_ab_view view;
  if (bcir_ab_open(data, len, &view) != BCIR_AB_OK) {
    fprintf(stderr, "the fixture bundle does not open\n");
    return 2;
  }
  test_a_corrupt_bundle_prints_nothing(data, len);
  test_the_sink_and_width_contracts(data, len);
  test_the_text_views();
  test_the_pack_mnemonics();
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  puts("OK");
  return 0;
}
