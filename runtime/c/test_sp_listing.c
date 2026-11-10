/*===- test_sp_listing.c - the MC1 record walk and the C listing's contracts ---===
 *
 * The octets of the listing and the dump are held to the oracle by the Python corpus
 * (bcir/tests/test_c_streampack_listing.py, which tools/c/sections/sp_listing.sh runs against the
 * built bcir-pack). This harness holds what a corpus of valid packs cannot show: packs built octet by
 * octet that break one decoder law each, the walk refusing exactly those and not the R10 laws, a
 * refused pack never reaching the sink, a sink that refuses, short scratch, the width bounds, the
 * chunked output and the spellings of quoted strings. Prints OK and exits 0, or names the failure.
 *===----------------------------------------------------------------------===*/
#include "bcir_sp_listing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, what)                                                       \
  do {                                                                          \
    if (!(cond)) { fprintf(stderr, "FAIL %s (line %d)\n", what, __LINE__); failures++; } \
  } while (0)

/* --- a pack, octet by octet ------------------------------------------------------------- */

typedef struct { uint8_t b[16384]; size_t n; } buf_t;

static void w8(buf_t *w, uint32_t v) { w->b[w->n++] = (uint8_t)v; }
static void w16(buf_t *w, uint32_t v) { w8(w, v & 0xFFu); w8(w, (v >> 8) & 0xFFu); }
static void w32(buf_t *w, uint32_t v) { w16(w, v & 0xFFFFu); w16(w, v >> 16); }
static void w64(buf_t *w, uint64_t v) { w32(w, (uint32_t)v); w32(w, (uint32_t)(v >> 32)); }
static void wbytes(buf_t *w, const char *s, size_t n) { memcpy(w->b + w->n, s, n); w->n += n; }
static void wstr(buf_t *w, const char *s) { size_t n = strlen(s); w16(w, (uint32_t)n); wbytes(w, s, n); }
static void put32(buf_t *w, size_t at, uint32_t v) {
  for (int i = 0; i < 4; i++) w->b[at + (size_t)i] = (uint8_t)(v >> (8 * i));
}
static void seal(buf_t *w) { w32(w, bcir_crc32(w->b, w->n)); }

/* What a test may change about the base pack: one field per law. */
typedef struct {
  uint32_t depth, buffers, lane, width, stride_k, dispatch, rid1, gen_max;
  const char *name;
  int trailing;
} knobs_t;

static knobs_t base_knobs(void) {
  knobs_t k = {1u, 2u, 0u, 16u, 0u, 1u, 3u, 5u, "seg0", 0};
  return k;
}

/* A v4 pack: one segment (claim 7, fenced), an untraced one (claim 9),
 * one double-buffered prefetch, one block, three trace records (claim 7 twice: the second wins), two
 * generation records. */
static void build(buf_t *w, knobs_t k, const char *plan) {
  w->n = 0;
  wbytes(w, "BSPK", 4); w16(w, 4); w16(w, 0);
  w32(w, 1); w32(w, k.gen_max); w32(w, 2);     /* topo, map, data */
  w32(w, 2); w32(w, 1); w32(w, 1); w32(w, 3);  /* segments, prefetches, blocks, trace */
  w16(w, k.depth); w16(w, 0); w32(w, 2);       /* pipeline_depth, reserved, n_gens */
  while (w->n < 64) w8(w, 0);
  wstr(w, plan);
  /* segment 0 */
  wstr(w, k.name); w64(w, 7); w32(w, 0); w8(w, k.lane); w32(w, k.width); w32(w, k.stride_k);
  wstr(w, "f32.add"); w16(w, 2); w32(w, 10); w32(w, 11); w16(w, 1); w32(w, 12);
  wstr(w, "pf0"); w16(w, 2); wstr(w, "a'b"); wstr(w, "c"); w16(w, 0);
  w8(w, k.dispatch); wstr(w, "pim0");
  /* segment 1: untraced */
  wstr(w, "seg1"); w64(w, 9); w32(w, 1); w8(w, 2); w32(w, 4); w32(w, 0);
  wstr(w, "tile.mma"); w16(w, 0); w16(w, 0); wstr(w, ""); w16(w, 0); w16(w, 0);
  w8(w, 0); wstr(w, "host");
  /* prefetch 0 */
  wstr(w, "pf0"); w32(w, 4); w16(w, 1); w32(w, 10); wstr(w, "T0"); wstr(w, "linear"); w8(w, k.buffers);
  /* block 0 */
  w64(w, 0); w64(w, 32); w16(w, 2); w64(w, 1); w64(w, UINT64_MAX);
  /* trace: claim 7 twice (the last one is the note the listing shows), then claim 11 */
  w64(w, 7); w64(w, 1); w64(w, 2);
  w64(w, 7); w64(w, 3); w64(w, 4);
  w64(w, 11); w64(w, 5); w64(w, 6);
  /* generations: rid 1 (map 5, data 2), rid k.rid1 */
  w32(w, 1); w32(w, 5); w32(w, 2);
  w32(w, k.rid1); w32(w, 0); w32(w, 0);
  if (k.trailing) w8(w, 0);
  seal(w);
}

/* --- sinks ------------------------------------------------------------------------------ */

typedef struct { char *text; size_t n, cap; size_t calls, biggest; size_t refuse_after; } sink_t;

static int collect(void *ctx, const char *text, size_t len) {
  sink_t *s = (sink_t *)ctx;
  s->calls++;
  if (len > s->biggest) s->biggest = len;
  if (s->refuse_after && s->calls > s->refuse_after) return 1;
  if (s->n + len + 1 > s->cap) {
    size_t cap = (s->n + len + 1) * 2;
    char *next = (char *)realloc(s->text, cap);
    if (!next) return 1;
    s->text = next; s->cap = cap;
  }
  memcpy(s->text + s->n, text, len);
  s->n += len;
  s->text[s->n] = '\0';
  return 0;
}

static sink_t sink_new(size_t refuse_after) {
  sink_t s = {NULL, 0, 0, 0, 0, refuse_after};
  return s;
}

static bcir_status list_into(const buf_t *w, sink_t *s) {
  size_t need = 0;
  bcir_status st = bcir_sp_listing_scratch_bytes(w->b, w->n, &need);
  if (st != BCIR_OK) return st;
  uint8_t *scratch = (uint8_t *)malloc(need ? need : 1u);
  if (!scratch) return BCIR_ERR_NOSPACE;
  st = bcir_sp_format_listing(w->b, w->n, scratch, need, collect, s);
  free(scratch);
  return st;
}

/* --- the walk ----------------------------------------------------------------------------- */

typedef struct { size_t next, records, last_kind; int ordered; uint32_t crc; } spans_t;

static int spans(const bcir_sp_record *r, void *ctx) {
  spans_t *s = (spans_t *)ctx;
  if (r->offset != s->next || r->length == 0 || (size_t)r->kind < s->last_kind) s->ordered = 0;
  s->next = r->offset + r->length;
  s->last_kind = (size_t)r->kind;
  s->records++;
  if (r->kind == BCIR_SP_RECORD_CRC32) s->crc = r->crc32;
  return 0;
}

static int stop_at_first(const bcir_sp_record *r, void *ctx) {
  (void)r;
  ++*(size_t *)ctx;
  return 1;
}

static void test_spans_partition_the_pack(void) {
  buf_t w;
  build(&w, base_knobs(), "plan0");
  spans_t s = {0, 0, 0, 1, 0};
  CHECK(bcir_sp_for_each_record(w.b, w.n, spans, &s) == BCIR_OK, "walk of the base pack");
  CHECK(s.ordered && s.next == w.n, "spans partition the pack in wire order");
  /* header, plan, 2 segments, 1 prefetch, 1 block, 3 trace, 2 generations, crc */
  CHECK(s.records == 12u, "every record visited once");
  CHECK(s.crc == bcir_crc32(w.b, w.n - 4u), "the CRC record carries the stored value");
  size_t calls = 0;
  CHECK(bcir_sp_for_each_record(w.b, w.n, stop_at_first, &calls) == BCIR_OK && calls == 1u,
        "an early stop is BCIR_OK after one call");
  CHECK(bcir_sp_for_each_record(w.b, w.n, NULL, NULL) == BCIR_OK, "a NULL callback only checks");
  CHECK(bcir_sp_checked_for_each_record(NULL, spans, &s) == BCIR_ERR_TRUNCATED, "no token");
  bcir_sp_checked refused;
  w.b[w.n - 1] ^= 0x01u;
  CHECK(bcir_sp_check(w.b, w.n, &refused) == BCIR_ERR_CRC, "a corrupt CRC");
  CHECK(bcir_sp_checked_for_each_record(&refused, spans, &s) == BCIR_ERR_TRUNCATED,
        "a refused token walks nothing");
  bcir_sp_strings none = {0, NULL};
  const char *text = NULL;
  uint16_t len = 0;
  CHECK(!bcir_sp_strings_next(&none, &text, &len), "an empty string array yields nothing");
  bcir_prefetch_view pf;
  memset(&pf, 0, sizeof pf);
  CHECK(bcir_prefetch_target(&pf, 0) == 0u && bcir_prefetch_target(NULL, 0) == 0u,
        "a target past the array reads 0");
}

/* One decoder law broken per pack: the walk refuses it with the law's status, and neither the
 * listing nor the dump says a word. */
static void refused(knobs_t k, bcir_status want, const char *what) {
  buf_t w;
  build(&w, k, "plan0");
  CHECK(bcir_sp_for_each_record(w.b, w.n, NULL, NULL) == want, what);
  sink_t s = sink_new(0);
  CHECK(list_into(&w, &s) == want && s.calls == 0u, what);
  CHECK(bcir_sp_format_hexdump(w.b, w.n, 16, collect, &s) == want && s.calls == 0u, what);
  size_t need = 99;
  CHECK(bcir_sp_listing_scratch_bytes(w.b, w.n, &need) == want && need == 0u, what);
  free(s.text);
}

static void test_each_decoder_law_refuses_and_prints_nothing(void) {
  knobs_t k;
  k = base_knobs(); k.depth = 0;    refused(k, BCIR_ERR_PROVENANCE, "pipeline_depth 0");
  k = base_knobs(); k.buffers = 3;  refused(k, BCIR_ERR_PROVENANCE, "prefetch buffers 3");
  k = base_knobs(); k.lane = 6;     refused(k, BCIR_ERR_LANE, "lane 6");
  k = base_knobs(); k.width = 12;   refused(k, BCIR_ERR_WIDTH, "width 12");
  k = base_knobs(); k.stride_k = 1; refused(k, BCIR_ERR_RESERVED, "stride_k 1");
  k = base_knobs(); k.dispatch = 2; refused(k, BCIR_ERR_DISPATCH, "dispatch 2");
  k = base_knobs(); k.rid1 = 1;     refused(k, BCIR_ERR_GENERATION, "RIDs not ascending");
  k = base_knobs(); k.gen_max = 6;  refused(k, BCIR_ERR_GENERATION, "map_gen not the maximum");
  k = base_knobs(); k.trailing = 1; refused(k, BCIR_ERR_TRAILING, "a CRC-valid trailing octet");
  k = base_knobs(); k.name = "s\xC0\xAF"; refused(k, BCIR_ERR_UTF8, "an overlong UTF-8 name");
  {
    /* The cursor's first fault is its verdict on every walk: the reads after a string that failed
     * UTF-8 do not restate it as a truncation. */
    buf_t w;
    k = base_knobs(); k.name = "s\xC0\xAF";
    build(&w, k, "plan0");
    CHECK(bcir_sp_for_each_segment(w.b, w.n, NULL, NULL) == BCIR_ERR_UTF8, "segment walk: UTF-8");
    CHECK(bcir_sp_for_each_block(w.b, w.n, NULL, NULL) == BCIR_ERR_UTF8, "block walk: UTF-8");
    CHECK(bcir_sp_verify_semantic(w.b, w.n, UINT32_MAX, UINT32_MAX) == BCIR_ERR_UTF8,
          "semantic walk: UTF-8");
  }
  {
    buf_t w;
    build(&w, base_knobs(), "plan0");
    put32(&w, 32, 4);  /* a fourth trace record the body does not hold */
    w.n -= 4; seal(&w);
    CHECK(bcir_sp_for_each_record(w.b, w.n, NULL, NULL) == BCIR_ERR_TRUNCATED, "a missing record");
  }
}

static void test_r10_is_the_verifiers_not_the_walks(void) {
  /* claim 9 untraced, claim 7 traced twice: R10 refuses both, the decoder neither. */
  buf_t w;
  build(&w, base_knobs(), "plan0");
  CHECK(bcir_sp_verify_semantic(w.b, w.n, UINT32_MAX, UINT32_MAX) == BCIR_ERR_PROVENANCE,
        "the semantic verifier refuses the untraced claim");
  sink_t s = sink_new(0);
  CHECK(list_into(&w, &s) == BCIR_OK, "the listing lists it");
  CHECK(s.text && strstr(s.text, "seg1: tile.mma claim=9 phase=1 lane=t width=4 rd=[] wr=[] "
                                 "dispatch=core channel=host prefetch=- untraced\n") != NULL,
        "the untraced segment's line");
  CHECK(s.text && strstr(s.text, "src=0x0000000000000003 trace=0x0000000000000004\n") != NULL,
        "a claim traced twice shows its last note");
  CHECK(s.text && strstr(s.text, "    fences before=[\"a'b\", 'c'] after=[]\n") != NULL,
        "fences quoted as ascii() quotes them");
  CHECK(s.text && strstr(s.text, "strides=[1, 18446744073709551615]\n") != NULL, "u64 strides");
  CHECK(s.text && strstr(s.text, "buffers=2\nblocks:\n") != NULL, "section order");
  free(s.text);
}

static void test_quoted_strings_are_spelled_as_ascii_spells_them(void) {
  static const struct { const char *plan, *shown; } cases[] = {
      {"plan0", "source_plan 'plan0'\n"},
      {"it's", "source_plan \"it's\"\n"},
      {"say \"hi\"", "source_plan 'say \"hi\"'\n"},
      {"both ' and \"", "source_plan 'both \\' and \"'\n"},
      {"a\\b\tc\nd\re", "source_plan 'a\\\\b\\tc\\nd\\re'\n"},
      {"\x01\x1f\x7f", "source_plan '\\x01\\x1f\\x7f'\n"},
      {"\xC3\xA9\xC4\x80\xF0\x9F\xAB\xA8\xCD\xB8", "source_plan '\\xe9\\u0100\\U0001fae8\\u0378'\n"},
      {"", "source_plan ''\n"},
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    buf_t w;
    build(&w, base_knobs(), cases[i].plan);
    sink_t s = sink_new(0);
    CHECK(list_into(&w, &s) == BCIR_OK && s.text && strstr(s.text, cases[i].shown), cases[i].shown);
    free(s.text);
  }
}

static void test_the_sink_scratch_and_width_contracts(void) {
  buf_t w;
  knobs_t k = base_knobs();
  char longname[3001];
  memset(longname, 'n', 3000);
  longname[3000] = '\0';
  k.name = longname;
  build(&w, k, "plan0");
  sink_t s = sink_new(0);
  CHECK(list_into(&w, &s) == BCIR_OK, "a 3000-octet name lists");
  CHECK(s.biggest <= BCIR_LISTING_CHUNK && s.calls >= 3u, "output arrives in bounded chunks");
  CHECK(s.text && strstr(s.text, longname) != NULL, "the name crosses chunks intact");
  free(s.text);

  sink_t once = sink_new(1);
  CHECK(list_into(&w, &once) == BCIR_ERR_NOSPACE && once.calls == 2u,
        "a refusing sink stops the listing");
  free(once.text);
  sink_t dump_once = sink_new(1);
  CHECK(bcir_sp_format_hexdump(w.b, w.n, 16, collect, &dump_once) == BCIR_ERR_NOSPACE &&
            dump_once.calls == 2u, "a refusing sink stops the dump");
  free(dump_once.text);

  size_t need = 0;
  CHECK(bcir_sp_listing_scratch_bytes(w.b, w.n, &need) == BCIR_OK && need == 36u,
        "12 scratch octets a trace record");
  uint8_t scratch[36];
  sink_t none = sink_new(0);
  CHECK(bcir_sp_format_listing(w.b, w.n, scratch, 35, collect, &none) == BCIR_ERR_NOSPACE &&
            none.calls == 0u, "short scratch refuses before any output");
  CHECK(bcir_sp_format_listing(w.b, w.n, NULL, 36, collect, &none) == BCIR_ERR_NOSPACE &&
            none.calls == 0u, "no scratch refuses before any output");
  CHECK(bcir_sp_format_listing(w.b, w.n, scratch, 36, NULL, NULL) == BCIR_ERR_NOSPACE, "no sink");
  CHECK(bcir_sp_format_hexdump(w.b, w.n, 0, collect, &none) == BCIR_ERR_OVERFLOW &&
            bcir_sp_format_hexdump(w.b, w.n, 65, collect, &none) == BCIR_ERR_OVERFLOW &&
            none.calls == 0u, "the width is in [1, 64]");
  sink_t wide = sink_new(0);
  CHECK(bcir_sp_format_hexdump(w.b, w.n, 64, collect, &wide) == BCIR_OK, "width 64 dumps");
  CHECK(wide.text && strncmp(wide.text, "# header 'header' offset=0x00000000 length=64\n"
                                        "00000000  42 53 50 4b", 66) == 0, "the dump's first row");
  free(wide.text);
  free(none.text);
}

int main(void) {
  test_spans_partition_the_pack();
  test_each_decoder_law_refuses_and_prints_nothing();
  test_r10_is_the_verifiers_not_the_walks();
  test_quoted_strings_are_spelled_as_ascii_spells_them();
  test_the_sink_scratch_and_width_contracts();
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  puts("OK");
  return 0;
}
