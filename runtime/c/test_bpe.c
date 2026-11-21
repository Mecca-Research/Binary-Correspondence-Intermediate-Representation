/*===- test_bpe.c - the C BPE tokenizer's contracts (PORT-2) ---------------------------===
 *
 * What the parity test cannot see from a command line: the golden ids over an inline mini tokenizer,
 * the count-only and short-buffer paths of encode and decode, every argument refusal, the law a
 * refused file names, and the allocation discipline -- each allocation failed in turn during a
 * load and an encode must give BCIR_BPE_ERR_NOMEM with nothing kept and nothing leaked. With a
 * file argument (a tokenizer.json the oracle loads), the load sweep runs over that file too.
 *
 *   usage: test_bpe [TOKENIZER.json]      prints OK
 *===----------------------------------------------------------------------===*/
#include "bcir_bpe.h"
#include "bcir_host_alloc.h"
#include "bcir_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      failures++;                                                         \
    }                                                                     \
  } while (0)

/* --- a fault-injecting allocator: fails attempt `fail_at` (0: none), counts what is live ---- */

typedef struct { size_t attempt, fail_at, live; } fault_state;

static void *fault_allocate(void *context, size_t size) {
  fault_state *s = (fault_state *)context;
  s->attempt++;
  if (s->fail_at && s->attempt == s->fail_at) return NULL;
  void *p = malloc(size);
  if (p) s->live++;
  return p;
}
static void *fault_reallocate(void *context, void *allocation, size_t size) {
  fault_state *s = (fault_state *)context;
  s->attempt++;
  if (s->fail_at && s->attempt == s->fail_at) return NULL;
  void *p = realloc(allocation, size);
  if (p && !allocation) s->live++;
  return p;
}
static void fault_deallocate(void *context, void *allocation) {
  fault_state *s = (fault_state *)context;
  if (!allocation) return;
  if (!s->live) {
    fprintf(stderr, "allocator live-count underflow\n");
    abort();
  }
  s->live--;
  free(allocation);
}
static bcir_host_allocator fault_allocator(fault_state *s) {
  bcir_host_allocator a;
  a.context = s;
  a.allocate = fault_allocate;
  a.reallocate = fault_reallocate;
  a.deallocate = fault_deallocate;
  return a;
}

/* --- an inline mini tokenizer: the 256-byte alphabet (id == byte), four merges, four specials -- */

typedef struct { char *s; size_t n, cap; } buf_t;

static void put(buf_t *b, const char *text, size_t n) {
  if (b->n + n + 1u > b->cap) {
    b->cap = (b->n + n + 1u) * 2u;
    b->s = (char *)realloc(b->s, b->cap);
    if (!b->s) abort();
  }
  memcpy(b->s + b->n, text, n);
  b->n += n;
  b->s[b->n] = '\0';
}
static void puts_(buf_t *b, const char *text) { put(b, text, strlen(text)); }

static uint32_t alphabet_cp(unsigned byte) {
  unsigned shifted = 0;
  for (unsigned b = 0; b < 256u; b++) {
    int printable = (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || (b >= 0xAEu && b <= 0xFFu);
    if (b == byte) return printable ? b : 256u + shifted;
    if (!printable) shifted++;
  }
  return 0;
}

/* A JSON string holding the alphabet character of `byte`, escaped as \uXXXX. */
static void put_alphabet(buf_t *b, unsigned byte) {
  char esc[16];
  snprintf(esc, sizeof esc, "\"\\u%04X\"", (unsigned)alphabet_cp(byte));
  puts_(b, esc);
}

/* `extra` is spliced into the root object after the model, `vocab_extra` after the vocab's
 * entries, `merges` replaces the merge list's contents and `added` the added tokens' contents. */
static buf_t mini_json(const char *extra, const char *vocab_extra, const char *merges, const char *added,
                       int drop_byte) {
  buf_t b = {0, 0, 0};
  char num[32];
  puts_(&b, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
  int first = 1;
  for (unsigned byte = 0; byte < 256u; byte++) {
    if ((int)byte == drop_byte) continue;
    if (!first) puts_(&b, ",");
    first = 0;
    put_alphabet(&b, byte);
    snprintf(num, sizeof num, ":%u", byte);
    puts_(&b, num);
  }
  puts_(&b, ",\"he\":256,\"ll\":257,\"hell\":258,\"hello\":259");
  puts_(&b, vocab_extra);
  puts_(&b, "},\"merges\":[");
  puts_(&b, merges ? merges : "\"h e\",\"l l\",\"he ll\",\"hell o\"");
  puts_(&b, "]}");
  puts_(&b, extra);
  puts_(&b, ",\"added_tokens\":[");
  puts_(&b, added ? added
                  : "{\"content\":\"<bos>\",\"id\":260},{\"content\":\"<eos>\",\"id\":261},"
                    "{\"content\":\"<start_of_turn>\",\"id\":262},{\"content\":\"<end_of_turn>\",\"id\":263}");
  puts_(&b, "]}");
  return b;
}

static bcir_bpe *load_or_die(const buf_t *b) {
  bcir_bpe *t = NULL;
  bcir_bpe_diag d;
  if (bcir_bpe_load((const uint8_t *)b->s, b->n, NULL, &t, &d) != BCIR_BPE_OK) {
    fprintf(stderr, "the mini tokenizer did not load: status %d law %d at %zu\n", (int)d.status, (int)d.law,
            d.offset);
    exit(1);
  }
  return t;
}

static int encodes_to(const bcir_bpe *t, const char *text, const uint32_t *want, size_t n) {
  uint32_t ids[64];
  size_t count = 0;
  if (bcir_bpe_encode(t, (const uint8_t *)text, strlen(text), ids, 64, &count) != BCIR_BPE_OK || count != n)
    return 0;
  return n == 0u || memcmp(ids, want, n * sizeof(uint32_t)) == 0;
}

static void test_golden(const bcir_bpe *t, const buf_t *b) {
  static const uint32_t hello[] = {259}, space_hello[] = {32, 259}, helo[] = {256, 108, 111},
                        framed[] = {260, 259, 261}, spaces[] = {32, 32, 259, 32};
  CHECK(encodes_to(t, "hello", hello, 1));
  CHECK(encodes_to(t, " hello", space_hello, 2));
  CHECK(encodes_to(t, "helo", helo, 3));
  CHECK(encodes_to(t, "<bos>hello<eos>", framed, 3));
  CHECK(encodes_to(t, "  hello ", spaces, 4));
  CHECK(encodes_to(t, "", NULL, 0));
  CHECK(bcir_bpe_vocab_size(t) == 260u && bcir_bpe_merge_count(t) == 4u && bcir_bpe_special_count(t) == 4u);
  uint8_t digest[32], want[32];
  bcir_sha256 sha;
  bcir_sha256_init(&sha);
  bcir_sha256_update(&sha, (const uint8_t *)b->s, b->n);
  bcir_sha256_final(&sha, want);
  bcir_bpe_digest(t, digest);
  CHECK(memcmp(digest, want, 32) == 0);
}

static void test_buffers(const bcir_bpe *t) {
  const char *text = "<bos> hello helo";
  uint32_t ids[16], few[2];
  size_t count = 99, n = 0;
  CHECK(bcir_bpe_encode(t, (const uint8_t *)text, strlen(text), ids, 16, &n) == BCIR_BPE_OK && n == 7u);
  /* count-only, and a short buffer: NOSPACE with the count needed, the prefix written */
  CHECK(bcir_bpe_encode(t, (const uint8_t *)text, strlen(text), NULL, 0, &count) == BCIR_BPE_ERR_NOSPACE &&
        count == n);
  CHECK(bcir_bpe_encode(t, (const uint8_t *)text, strlen(text), few, 2, &count) == BCIR_BPE_ERR_NOSPACE &&
        count == n && few[0] == ids[0] && few[1] == ids[1]);
  /* decode: the measure, a short buffer, and the octets */
  uint8_t out[64], small[3];
  size_t w = 99;
  CHECK(bcir_bpe_decode(t, ids, n, NULL, 0, &w) == BCIR_BPE_ERR_NOSPACE && w == strlen(text));
  CHECK(bcir_bpe_decode(t, ids, n, small, 3, &w) == BCIR_BPE_ERR_NOSPACE && w == strlen(text));
  CHECK(bcir_bpe_decode(t, ids, n, out, sizeof out, &w) == BCIR_BPE_OK && w == strlen(text) &&
        memcmp(out, text, w) == 0);
  CHECK(bcir_bpe_decode(t, ids, 0, NULL, 0, &w) == BCIR_BPE_OK && w == 0u);
  /* UTF-8 is judged per run between specials, and at the end */
  static const uint32_t e_acute[] = {195, 169}, split[] = {195, 260, 169}, dangling[] = {195},
                        stray[] = {169}, overlong[] = {0xC0, 0xAF}, surrogate[] = {0xED, 0xA0, 0x80},
                        too_high[] = {0xF4, 0x90, 0x80, 0x80}, unknown[] = {32, 264};
  CHECK(bcir_bpe_decode(t, e_acute, 2, out, sizeof out, &w) == BCIR_BPE_OK && w == 2u);
  CHECK(bcir_bpe_decode(t, split, 3, out, sizeof out, &w) == BCIR_BPE_ERR_NOT_UTF8);
  CHECK(bcir_bpe_decode(t, dangling, 1, out, sizeof out, &w) == BCIR_BPE_ERR_NOT_UTF8);
  CHECK(bcir_bpe_decode(t, stray, 1, out, sizeof out, &w) == BCIR_BPE_ERR_NOT_UTF8);
  CHECK(bcir_bpe_decode(t, overlong, 2, out, sizeof out, &w) == BCIR_BPE_ERR_NOT_UTF8);
  CHECK(bcir_bpe_decode(t, surrogate, 3, out, sizeof out, &w) == BCIR_BPE_ERR_NOT_UTF8);
  CHECK(bcir_bpe_decode(t, too_high, 4, out, sizeof out, &w) == BCIR_BPE_ERR_NOT_UTF8);
  CHECK(bcir_bpe_decode(t, unknown, 2, out, sizeof out, &w) == BCIR_BPE_ERR_UNKNOWN_ID);
  /* text that is not UTF-8 is refused before any id is written */
  ids[0] = 12345u;
  CHECK(bcir_bpe_encode(t, (const uint8_t *)"a\xC3(", 3, ids, 16, &count) == BCIR_BPE_ERR_NOT_UTF8 && count == 0u &&
        ids[0] == 12345u);
  CHECK(bcir_bpe_encode(t, (const uint8_t *)"\xED\xA0\x80", 3, ids, 16, &count) == BCIR_BPE_ERR_NOT_UTF8);
}

static void test_arguments(const bcir_bpe *t, const buf_t *b) {
  bcir_bpe *out = (bcir_bpe *)&out;
  bcir_bpe_diag d;
  size_t count = 0, w = 0;
  uint32_t ids[4] = {0};
  uint8_t byte = 0;
  CHECK(bcir_bpe_load((const uint8_t *)b->s, b->n, NULL, NULL, &d) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_load(NULL, 5, NULL, &out, &d) == BCIR_BPE_ERR_INVALID && out == NULL);
  bcir_host_allocator broken = bcir_host_allocator_default();
  broken.reallocate = NULL;
  out = (bcir_bpe *)&out;
  CHECK(bcir_bpe_load((const uint8_t *)b->s, b->n, &broken, &out, &d) == BCIR_BPE_ERR_INVALID && out == NULL);
  CHECK(bcir_bpe_encode(NULL, &byte, 1, ids, 4, &count) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_encode(t, NULL, 1, ids, 4, &count) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_encode(t, &byte, 1, NULL, 4, &count) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_encode(t, &byte, 1, ids, 4, NULL) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_encode(t, &byte, (size_t)UINT32_MAX, ids, 4, &count) == BCIR_BPE_ERR_INVALID); /* never read */
  CHECK(bcir_bpe_decode(NULL, ids, 1, &byte, 1, &w) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_decode(t, NULL, 1, &byte, 1, &w) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_decode(t, ids, 1, NULL, 1, &w) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_decode(t, ids, 1, &byte, 1, NULL) == BCIR_BPE_ERR_INVALID);
  CHECK(bcir_bpe_vocab_size(NULL) == 0u && bcir_bpe_merge_count(NULL) == 0u && bcir_bpe_special_count(NULL) == 0u);
  bcir_bpe_free(NULL);
  bcir_bpe_digest(NULL, NULL);
  bcir_bpe_limits(NULL);
}

/* A file the laws refuse names the law, and nothing is kept. */
static void refused(buf_t b, bcir_bpe_status status, bcir_bpe_law law, int line) {
  bcir_bpe *t = (bcir_bpe *)&t;
  bcir_bpe_diag d;
  bcir_bpe_status st = bcir_bpe_load((const uint8_t *)b.s, b.n, NULL, &t, &d);
  if (st != status || d.status != status || d.law != law || t != NULL) {
    fprintf(stderr, "test_bpe.c:%d: want status %d law %d, got status %d law %d\n", line, (int)status, (int)law,
            (int)st, (int)d.law);
    failures++;
  }
  free(b.s);
}
#define REFUSED(doc, status, law) refused((doc), (status), (law), __LINE__)

static buf_t raw(const char *text) {
  buf_t b = {0, 0, 0};
  puts_(&b, text);
  return b;
}

static void test_laws(void) {
  REFUSED(raw("[]"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ROOT);
  REFUSED(raw("{}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MODEL);
  REFUSED(raw("{\"model\":{\"type\":\"WordPiece\"}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MODEL);
  REFUSED(raw("{\"model\":[]}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MODEL);
  REFUSED(raw("{\"model\":{\"type\":\"BPE\",\"vocab\":[]}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_VOCAB);
  REFUSED(raw("{\"model\":{\"type\":\"BPE\",\"vocab\":{\" \":1}}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_VOCAB);
  REFUSED(raw("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":1.0}}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ID);
  REFUSED(raw("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":2147483648}}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ID);
  REFUSED(raw("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":-1}}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ID);
  REFUSED(raw("{\"model\":{\"type\":\"BPE\",\"type\":\"BPE\"}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_DUPLICATE_KEY);
  REFUSED(raw("{\"x\":{\"k\":1,\"k\":1},\"model\":{}}"), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_DUPLICATE_KEY);
  REFUSED(raw("{\"x\":NaN}"), BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE);
  REFUSED(raw("{\"x\":\"\\ud800\"}"), BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE);
  REFUSED(raw("{\"x\":\"\xC3\x28\"}"), BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE);
  REFUSED(raw("{} x"), BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE);
  REFUSED(raw(""), BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE);
  {
    /* The scan runs before the sink: a file that breaks a law early and a reader limit late is
     * refused by the reader, before anything is built. */
    buf_t deep = raw("{\"model\":{\"type\":\"WordPiece\"},\"x\":");
    for (int i = 0; i < 64; i++) puts_(&deep, "[");
    for (int i = 0; i < 64; i++) puts_(&deep, "]");
    puts_(&deep, "}");
    REFUSED(deep, BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE);
  }
  REFUSED(mini_json("", "", NULL, NULL, 0x41), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_BASE_BYTES);
  REFUSED(mini_json("", ",\"a\":97", NULL, NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_DUPLICATE_KEY);
  REFUSED(mini_json("", ",\"x\\u0100\":97", NULL, NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ID);
  REFUSED(mini_json("", "", "\"h e\",\"h e\"", NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MERGES);
  REFUSED(mini_json("", "", "\"h e\",[\"h\",\"e\"]", NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MERGES);
  REFUSED(mini_json("", "", "\"h  e\"", NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MERGES);
  REFUSED(mini_json("", "", "[\"h\"]", NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MERGES);
  REFUSED(mini_json("", "", "\"o h\"", NULL, -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_MERGES);
  REFUSED(mini_json("", "", NULL, "{\"content\":\"\",\"id\":300}", -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ADDED);
  REFUSED(mini_json("", "", NULL, "{\"id\":300}", -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ADDED);
  REFUSED(mini_json("", "", NULL, "{\"content\":\"x\",\"id\":300},{\"content\":\"x\",\"id\":301}", -1),
          BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ADDED);
  REFUSED(mini_json("", "", NULL, "{\"content\":\"hello\",\"id\":300}", -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ADDED);
  REFUSED(mini_json("", "", NULL, "{\"content\":\"<x>\",\"id\":259}", -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ADDED);
  REFUSED(mini_json("", "", NULL, "{\"content\":\"<x>\",\"id\":1.5}", -1), BCIR_BPE_ERR_SHAPE, BCIR_BPE_LAW_ID);
  /* and what the laws keep: extra members, nested and ignored, a merge in the pair form, an
   * added token that is its vocab token */
  buf_t ok = mini_json(",\"normalizer\":{\"a\":[1,{\"b\":null}],\"c\":-0.5e3}", "", "[\"h\",\"e\"],\"l l\"",
                       "{\"content\":\"hello\",\"id\":259,\"special\":true},{\"content\":\"\\u00e9\",\"id\":233}", -1);
  bcir_bpe *t = NULL;
  CHECK(bcir_bpe_load((const uint8_t *)ok.s, ok.n, NULL, &t, NULL) == BCIR_BPE_OK && t != NULL);
  if (t) {
    static const uint32_t e_acute[] = {233};
    uint8_t out[8];
    size_t w = 0;
    CHECK(bcir_bpe_special_count(t) == 2u && bcir_bpe_merge_count(t) == 2u);
    CHECK(encodes_to(t, "\xC3\xA9", e_acute, 1)); /* the special's text, ahead of its bytes */
    CHECK(bcir_bpe_decode(t, e_acute, 1, out, sizeof out, &w) == BCIR_BPE_OK && w == 2u && out[0] == 0xC3);
  }
  bcir_bpe_free(t);
  free(ok.s);
}

/* Every allocation of a load, failed in turn: NOMEM, nothing kept, nothing leaked; then every
 * allocation of an encode over a long word. Returns the allocations a clean load makes. */
static size_t sweep(const uint8_t *json, size_t len, const char *text) {
  fault_state s = {0, 0, 0};
  bcir_host_allocator a = fault_allocator(&s);
  bcir_bpe *t = NULL;
  CHECK(bcir_bpe_load(json, len, &a, &t, NULL) == BCIR_BPE_OK);
  size_t attempts = s.attempt;
  bcir_bpe_free(t);
  CHECK(s.live == 0u);
  for (size_t k = 1; k <= attempts; k++) {
    s.attempt = 0;
    s.fail_at = k;
    t = (bcir_bpe *)&t;
    bcir_bpe_diag d;
    bcir_bpe_status st = bcir_bpe_load(json, len, &a, &t, &d);
    if (st != BCIR_BPE_ERR_NOMEM || d.status != BCIR_BPE_ERR_NOMEM || t != NULL || s.live != 0u) {
      fprintf(stderr, "allocation %zu of %zu failed: status %d, %zu live\n", k, attempts, (int)st, s.live);
      failures++;
      bcir_bpe_free(st == BCIR_BPE_OK ? t : NULL);
    }
  }
  s.fail_at = 0;
  CHECK(bcir_bpe_load(json, len, &a, &t, NULL) == BCIR_BPE_OK);
  uint32_t ids[512];
  size_t count = 0, base = s.attempt, n = strlen(text);
  CHECK(bcir_bpe_encode(t, (const uint8_t *)text, n, ids, 512, &count) == BCIR_BPE_OK);
  size_t encode_attempts = s.attempt - base;
  CHECK(encode_attempts > 0u);
  for (size_t k = 1; k <= encode_attempts; k++) {
    s.attempt = base;
    s.fail_at = base + k;
    size_t live = s.live;
    CHECK(bcir_bpe_encode(t, (const uint8_t *)text, n, ids, 512, &count) == BCIR_BPE_ERR_NOMEM && s.live == live);
  }
  s.fail_at = 0;
  bcir_bpe_free(t);
  CHECK(s.live == 0u);
  return attempts;
}

static uint8_t *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  size_t cap = 1u << 16, n = 0;
  uint8_t *p = (uint8_t *)malloc(cap);
  for (;;) {
    if (!p) break;
    n += fread(p + n, 1u, cap - n, f);
    if (n < cap) break;
    cap *= 2u;
    p = (uint8_t *)realloc(p, cap);
  }
  fclose(f);
  *len = n;
  return p;
}

int main(int argc, char **argv) {
  buf_t mini = mini_json("", "", NULL, NULL, -1);
  bcir_bpe *t = load_or_die(&mini);
  test_golden(t, &mini);
  test_buffers(t);
  test_arguments(t, &mini);
  bcir_bpe_free(t);
  test_laws();
  CHECK(sweep((const uint8_t *)mini.s, mini.n, "<bos>hello hellohellohello helo hhhheeeelllllloooo") > 4u);
  free(mini.s);
  if (argc > 1) {
    size_t len = 0;
    uint8_t *json = read_file(argv[1], &len);
    if (!json) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
    }
    CHECK(sweep(json, len, "abababababababababababababab aaaaaaaaaaaaaaaaaaaa the quick brown fox") > 4u);
    free(json);
  }
  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  puts("OK");
  return 0;
}
