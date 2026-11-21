/*===- test_spm.c - the C SentencePiece tokenizer's contracts (PORT-2b) -------------------===
 *
 * What the parity test cannot see from a command line: golden ids over an inline model, the
 * count-only and short-buffer paths, every argument refusal, the rule a refused model names, the
 * replacement decoder's subparts one by one, and the allocation discipline -- each allocation failed
 * in turn during a load and an encode must give BCIR_SPM_ERR_NOMEM with nothing kept and nothing
 * leaked. With a file argument (a tokenizer.model the oracle loads), the sweep runs over it too.
 *
 *   usage: test_spm [TOKENIZER.model]      prints OK
 *===----------------------------------------------------------------------===*/
#include "bcir_host_alloc.h"
#include "bcir_sha256.h"
#include "bcir_spm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      failures++;                                                              \
    }                                                                          \
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

/* --- a protobuf writer -------------------------------------------------------------------- */

typedef struct { uint8_t *s; size_t n, cap; } buf_t;

static void put(buf_t *b, const void *p, size_t n) {
  if (!n) return;
  if (b->n + n > b->cap) {
    b->cap = (b->n + n) * 2u + 64u;
    b->s = (uint8_t *)realloc(b->s, b->cap);
    if (!b->s) abort();
  }
  memcpy(b->s + b->n, p, n);
  b->n += n;
}
static void varint(buf_t *b, uint64_t v) {
  do {
    uint8_t c = (uint8_t)(v & 0x7Fu);
    v >>= 7;
    if (v) c |= 0x80u;
    put(b, &c, 1);
  } while (v);
}
static void ld(buf_t *b, unsigned field, const void *p, size_t n) {
  varint(b, (uint64_t)field << 3 | 2u);
  varint(b, n);
  put(b, p, n);
}
static void piece(buf_t *b, const char *text, float score, unsigned type) {
  buf_t m = {0, 0, 0};
  uint32_t bits;
  uint8_t le[4];
  memcpy(&bits, &score, 4);
  for (unsigned i = 0; i < 4u; i++) le[i] = (uint8_t)(bits >> (8u * i));
  ld(&m, 1, text, strlen(text));
  varint(&m, 2u << 3 | 5u);
  put(&m, le, 4);
  varint(&m, 3u << 3 | 0u);
  varint(&m, type);
  ld(b, 1, m.s, m.n);
  free(m.s);
}

#define SP "\xE2\x96\x81" /* U+2581 */

/* unk, <s>, </s>, the byte alphabet (ids 3..258), then the scored pieces of test_model_spm.py */
static buf_t mini(int bytes) {
  static const struct { const char *t; float s; } scored[] = {
      {SP, -1.0f},           {SP "h", -1.9f},  {SP "he", -1.8f}, {SP "hel", -1.7f}, {SP "hell", -1.6f},
      {SP "hello", -0.5f},   {SP "w", -1.95f}, {SP "wo", -1.85f}, {SP "wor", -1.75f}, {SP "worl", -1.65f},
      {SP "world", -0.6f},   {"he", -2.0f},    {"ll", -2.1f},     {"o", -3.0f},       {"l", -3.1f},
      {"h", -3.2f},          {"e", -3.3f},     {"w", -3.4f},      {"r", -3.5f},       {"d", -3.6f},
      {"el", -2.2f}};
  buf_t b = {0, 0, 0};
  char text[8];
  piece(&b, "<unk>", 0.0f, 2);
  piece(&b, "<s>", 0.0f, 3);
  piece(&b, "</s>", 0.0f, 3);
  for (unsigned v = 0; bytes && v < 256u; v++) {
    snprintf(text, sizeof text, "<0x%02X>", v);
    piece(&b, text, 0.0f, 6);
  }
  for (size_t i = 0; i < sizeof scored / sizeof scored[0]; i++) piece(&b, scored[i].t, scored[i].s, 1);
  ld(&b, 2, "abc", 3); /* a trainer_spec-shaped field the reader skips */
  return b;
}

static bcir_spm *load_or_die(const buf_t *b) {
  bcir_spm *t = NULL;
  bcir_spm_diag d;
  if (bcir_spm_load(b->s, b->n, NULL, &t, &d) != BCIR_SPM_OK) {
    fprintf(stderr, "the mini model did not load: status %d law %d at %zu\n", (int)d.status, (int)d.law, d.offset);
    exit(1);
  }
  return t;
}

static int encodes_to(const bcir_spm *t, const char *text, int prefix, const uint32_t *want, size_t n) {
  uint32_t ids[64];
  size_t count = 0;
  if (bcir_spm_encode(t, (const uint8_t *)text, strlen(text), prefix, ids, 64, &count) != BCIR_SPM_OK ||
      count != n)
    return 0;
  return n == 0u || memcmp(ids, want, n * sizeof(uint32_t)) == 0;
}

static int decodes_to(const bcir_spm *t, const uint32_t *ids, size_t n, const char *want) {
  uint8_t out[128];
  size_t w = 0;
  return bcir_spm_decode(t, ids, n, out, sizeof out, &w) == BCIR_SPM_OK && w == strlen(want) &&
         memcmp(out, want, w) == 0;
}

/* id 259 + k is the k-th scored piece: SP 259, SP h 260 ... SP hello 264, SP world 269, he 270 */
enum { ID_SP = 259, ID_SP_HELLO = 264, ID_SP_WORLD = 269, ID_HE = 270, ID_LL = 271, ID_O = 272 };

static void test_golden(const bcir_spm *t, const buf_t *b) {
  static const uint32_t hello_world[] = {ID_SP_HELLO, ID_SP_WORLD}, sp_only[] = {ID_SP},
                        no_prefix[] = {ID_HE, ID_LL, ID_O}, cjk[] = {ID_SP, 3u + 0xE6u, 3u + 0x97u, 3u + 0xA5u};
  CHECK(encodes_to(t, "hello world", 1, hello_world, 2));
  CHECK(encodes_to(t, " hello world", 1, hello_world, 2)); /* a leading space is the prefix */
  CHECK(encodes_to(t, "", 1, sp_only, 1));                  /* the dummy prefix alone */
  CHECK(encodes_to(t, "", 0, NULL, 0));
  CHECK(encodes_to(t, "hello", 0, no_prefix, 3));
  CHECK(encodes_to(t, "\xE6\x97\xA5", 1, cjk, 4)); /* byte fallback */
  CHECK(bcir_spm_piece_count(t) == 280u && bcir_spm_unk_id(t) == 0u && bcir_spm_has_byte_fallback(t) &&
        bcir_spm_special_count(t) == 3u);
  CHECK(decodes_to(t, hello_world, 2, "hello world")); /* the prefix space is dropped */
  {
    static const uint32_t controls[] = {1, ID_SP_HELLO, 2, 0};
    CHECK(decodes_to(t, controls, 4, "hello")); /* control and unknown pieces render as nothing */
  }
  uint8_t digest[32], want[32];
  bcir_sha256 sha;
  bcir_sha256_init(&sha);
  bcir_sha256_update(&sha, b->s, b->n);
  bcir_sha256_final(&sha, want);
  bcir_spm_digest(t, digest);
  CHECK(memcmp(digest, want, 32) == 0);
}

/* Python's errors='replace': each maximal ill-formed subpart is one U+FFFD. */
static void test_replacement(const bcir_spm *t) {
#define FFFD "\xEF\xBF\xBD"
  static const struct { uint8_t in[6]; size_t n; const char *want; } cases[] = {
      {{0xC3, 0xA9}, 2, "\xC3\xA9"},
      {{0xC3}, 1, FFFD},
      {{0xA9}, 1, FFFD},
      {{0xC0, 0xAF}, 2, FFFD FFFD},                 /* never a lead */
      {{0xE0, 0x80, 0x80}, 3, FFFD FFFD FFFD},      /* overlong: E0 needs A0..BF */
      {{0xE0, 0xA0}, 2, FFFD},                      /* one subpart, cut short */
      {{0xED, 0xA0, 0x80}, 3, FFFD FFFD FFFD},      /* a surrogate */
      {{0xF0, 0x90, 0x80, 0x41}, 4, FFFD "A"},      /* a subpart, then a fresh start */
      {{0xF4, 0x90, 0x80, 0x80}, 4, FFFD FFFD FFFD FFFD}, /* past U+10FFFF */
      {{0xF5, 0x41}, 2, FFFD "A"},
      {{0x20, 0x41}, 2, "A"},                        /* the leading space is dropped */
      {{0x20, 0x20}, 2, " "},
      {{0xC3, 0x20}, 2, FFFD " "},                   /* only a leading space */
      {{0xE2, 0x96, 0x81}, 3, SP},                   /* a byte piece is never the space */
  };
#undef FFFD
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    uint32_t ids[6];
    for (size_t k = 0; k < cases[i].n; k++) ids[k] = 3u + cases[i].in[k];
    if (!decodes_to(t, ids, cases[i].n, cases[i].want)) {
      fprintf(stderr, "test_spm.c: replacement case %zu decodes wrong\n", i);
      failures++;
    }
  }
}

static void test_buffers(const bcir_spm *t) {
  const char *text = "hello hell\xE6\x97\xA5";
  uint32_t ids[64], few[2];
  size_t n = 0, count = 99, w = 99;
  uint8_t out[64], small[2];
  CHECK(bcir_spm_encode(t, (const uint8_t *)text, strlen(text), 1, ids, 64, &n) == BCIR_SPM_OK && n > 2u);
  CHECK(bcir_spm_encode(t, (const uint8_t *)text, strlen(text), 1, NULL, 0, &count) == BCIR_SPM_ERR_NOSPACE &&
        count == n);
  CHECK(bcir_spm_encode(t, (const uint8_t *)text, strlen(text), 1, few, 2, &count) == BCIR_SPM_ERR_NOSPACE &&
        count == n && few[0] == ids[0] && few[1] == ids[1]);
  CHECK(bcir_spm_decode(t, ids, n, NULL, 0, &w) == BCIR_SPM_ERR_NOSPACE && w == strlen(text));
  CHECK(bcir_spm_decode(t, ids, n, small, 2, &w) == BCIR_SPM_ERR_NOSPACE && w == strlen(text));
  CHECK(bcir_spm_decode(t, ids, n, out, sizeof out, &w) == BCIR_SPM_OK && w == strlen(text) &&
        memcmp(out, text, w) == 0);
  CHECK(bcir_spm_decode(t, ids, 0, NULL, 0, &w) == BCIR_SPM_OK && w == 0u);
  {
    static const uint32_t unknown[] = {ID_SP, 280};
    CHECK(bcir_spm_decode(t, unknown, 2, out, sizeof out, &w) == BCIR_SPM_ERR_UNKNOWN_ID);
  }
  ids[0] = 12345u;
  CHECK(bcir_spm_encode(t, (const uint8_t *)"a\xC3(", 3, 1, ids, 64, &count) == BCIR_SPM_ERR_NOT_UTF8 &&
        count == 0u && ids[0] == 12345u);
  /* the bound: 3 * len + 3 ids, reached by spaces when U+2581 is no piece and bytes carry it */
}

static void test_arguments(const bcir_spm *t, const buf_t *b) {
  bcir_spm *out = (bcir_spm *)&out;
  bcir_spm_diag d;
  size_t count = 0, w = 0;
  uint32_t ids[4] = {0};
  uint8_t byte = 0;
  CHECK(bcir_spm_load(b->s, b->n, NULL, NULL, &d) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_load(NULL, 5, NULL, &out, &d) == BCIR_SPM_ERR_INVALID && out == NULL);
  bcir_host_allocator broken = bcir_host_allocator_default();
  broken.deallocate = NULL;
  out = (bcir_spm *)&out;
  CHECK(bcir_spm_load(b->s, b->n, &broken, &out, &d) == BCIR_SPM_ERR_INVALID && out == NULL);
  CHECK(bcir_spm_encode(NULL, &byte, 1, 1, ids, 4, &count) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_encode(t, NULL, 1, 1, ids, 4, &count) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_encode(t, &byte, 1, 1, NULL, 4, &count) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_encode(t, &byte, 1, 1, ids, 4, NULL) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_encode(t, &byte, (size_t)UINT32_MAX / 3u, 1, ids, 4, &count) == BCIR_SPM_ERR_INVALID); /* never read */
  CHECK(bcir_spm_decode(NULL, ids, 1, &byte, 1, &w) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_decode(t, NULL, 1, &byte, 1, &w) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_decode(t, ids, 1, NULL, 1, &w) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_decode(t, ids, 1, &byte, 1, NULL) == BCIR_SPM_ERR_INVALID);
  CHECK(bcir_spm_piece_count(NULL) == 0u && bcir_spm_special_count(NULL) == 0u && !bcir_spm_has_byte_fallback(NULL));
  bcir_spm_free(NULL);
  bcir_spm_digest(NULL, NULL);
}

static void refused(buf_t b, bcir_spm_status status, bcir_spm_law law, int line) {
  bcir_spm *t = (bcir_spm *)&t;
  bcir_spm_diag d;
  bcir_spm_status st = bcir_spm_load(b.s, b.n, NULL, &t, &d);
  if (st != status || d.status != status || d.law != law || t != NULL) {
    fprintf(stderr, "test_spm.c:%d: want status %d law %d, got status %d law %d\n", line, (int)status, (int)law,
            (int)st, (int)d.law);
    failures++;
  }
  free(b.s);
}
#define REFUSED(doc, status, law) refused((doc), (status), (law), __LINE__)

static buf_t raw(const void *p, size_t n) {
  buf_t b = {0, 0, 0};
  put(&b, p, n ? n : 0u);
  return b;
}

static buf_t unk_then(const void *p, size_t n) {
  buf_t b = {0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  put(&b, p, n);
  return b;
}

static void test_laws(void) {
  buf_t b;
  REFUSED(raw("", 0), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_NO_PIECES);
  REFUSED(raw("\x12\x03" "abc", 5), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_NO_PIECES);
  REFUSED(raw("\x80", 1), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_VARINT);
  REFUSED(raw("\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x02", 10), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_VARINT);
  REFUSED(unk_then("\x02\x00", 2), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_FIELD);     /* field zero */
  REFUSED(unk_then("\x2B", 1), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_FIELD);         /* wire 3 */
  REFUSED(unk_then("\x0A\x64x", 3), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_FIELD);    /* a length past the end */
  REFUSED(unk_then("\x49\x00", 2), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_FIELD);     /* a torn 64-bit field */
  REFUSED(unk_then("\x0A\x02\x15\x00", 4), BCIR_SPM_ERR_WIRE, BCIR_SPM_LAW_FIELD); /* a torn score */
  REFUSED(unk_then("\x0A\x05\x15\x00\x00\x00\x00", 7), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_PIECE_FIELDS);
  REFUSED(unk_then("\x0A\x06\x0A\x01" "a\x0A\x01" "b", 8), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_PIECE_FIELDS);
  REFUSED(unk_then("\x0A\x04\x0A\x02\xC0\xAF", 6), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_PIECE_TEXT);
  REFUSED(unk_then("\x0A\x08\x0A\x01" "a\x15\x00\x00\xC0\x7F", 10), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_SCORE);
  REFUSED(unk_then("\x0A\x05\x0A\x01" "a\x18\x07", 7), BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_TYPE);
  b = (buf_t){0, 0, 0};
  piece(&b, "a", 0.0f, 1);
  REFUSED(b, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_UNKNOWN_PIECE);
  b = (buf_t){0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  piece(&b, "<unk2>", 0.0f, 2);
  REFUSED(b, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_UNKNOWN_PIECE);
  b = (buf_t){0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  piece(&b, "<unk>", 0.0f, 3);
  REFUSED(b, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_DUPLICATE_TEXT);
  b = (buf_t){0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  piece(&b, "<0x1f>", 0.0f, 6);
  REFUSED(b, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_BYTE_PIECE);
  b = (buf_t){0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  piece(&b, "<0x1F>", 0.0f, 6);
  REFUSED(b, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_BYTE_ALPHABET);
  /* what the rules keep: an empty piece, NUL, a padded varint, a field 1 of another wire type */
  b = (buf_t){0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  piece(&b, "", -1.0f, 1);
  put(&b, "\x0A\x03\x0A\x01\x00", 5);              /* a piece "\0" with no score and no type */
  put(&b, "\x08\x05", 2);                          /* field 1, varint wire: skipped */
  put(&b, "\x10\x80\x80\x00", 4);                  /* a padded varint */
  bcir_spm *t = NULL;
  CHECK(bcir_spm_load(b.s, b.n, NULL, &t, NULL) == BCIR_SPM_OK && bcir_spm_piece_count(t) == 3u &&
        !bcir_spm_has_byte_fallback(t));
  if (t) {
    static const uint32_t unk[] = {0, 0};
    CHECK(encodes_to(t, "zz", 0, unk, 2)); /* no byte alphabet: each unknown symbol is <unk> */
  }
  bcir_spm_free(t);
  free(b.s);
  /* the bound of 3 * len + 3 ids: every space a byte triple, and the prefix one too */
  b = (buf_t){0, 0, 0};
  piece(&b, "<unk>", 0.0f, 2);
  for (unsigned v = 0; v < 256u; v++) {
    char text[8];
    snprintf(text, sizeof text, "<0x%02X>", v);
    piece(&b, text, 0.0f, 6);
  }
  t = NULL;
  CHECK(bcir_spm_load(b.s, b.n, NULL, &t, NULL) == BCIR_SPM_OK);
  if (t) {
    size_t count = 0;
    CHECK(bcir_spm_encode(t, (const uint8_t *)"  ", 2, 0, NULL, 0, &count) == BCIR_SPM_ERR_NOSPACE && count == 6u);
    CHECK(bcir_spm_encode(t, (const uint8_t *)"a ", 2, 1, NULL, 0, &count) == BCIR_SPM_ERR_NOSPACE && count == 7u);
  }
  bcir_spm_free(t);
  free(b.s);
}

/* Every allocation of a load, failed in turn: NOMEM, nothing kept, nothing leaked; then every
 * allocation of an encode. Returns the allocations a clean load makes. */
static size_t sweep(const uint8_t *model, size_t len, const char *text) {
  fault_state s = {0, 0, 0};
  bcir_host_allocator a = fault_allocator(&s);
  bcir_spm *t = NULL;
  CHECK(bcir_spm_load(model, len, &a, &t, NULL) == BCIR_SPM_OK);
  size_t attempts = s.attempt;
  bcir_spm_free(t);
  CHECK(s.live == 0u);
  for (size_t k = 1; k <= attempts; k++) {
    s.attempt = 0;
    s.fail_at = k;
    t = (bcir_spm *)&t;
    bcir_spm_diag d;
    bcir_spm_status st = bcir_spm_load(model, len, &a, &t, &d);
    if (st != BCIR_SPM_ERR_NOMEM || d.status != BCIR_SPM_ERR_NOMEM || t != NULL || s.live != 0u) {
      fprintf(stderr, "allocation %zu of %zu failed: status %d, %zu live\n", k, attempts, (int)st, s.live);
      failures++;
      bcir_spm_free(st == BCIR_SPM_OK ? t : NULL);
    }
  }
  s.fail_at = 0;
  CHECK(bcir_spm_load(model, len, &a, &t, NULL) == BCIR_SPM_OK);
  uint32_t ids[1024];
  size_t count = 0, base = s.attempt, n = strlen(text);
  CHECK(bcir_spm_encode(t, (const uint8_t *)text, n, 1, ids, 1024, &count) == BCIR_SPM_OK);
  size_t encode_attempts = s.attempt - base;
  CHECK(encode_attempts > 0u);
  for (size_t k = 1; k <= encode_attempts; k++) {
    s.attempt = base;
    s.fail_at = base + k;
    size_t live = s.live;
    CHECK(bcir_spm_encode(t, (const uint8_t *)text, n, 1, ids, 1024, &count) == BCIR_SPM_ERR_NOMEM && s.live == live);
  }
  s.fail_at = 0;
  bcir_spm_free(t);
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
  buf_t m = mini(1);
  bcir_spm *t = load_or_die(&m);
  test_golden(t, &m);
  test_replacement(t);
  test_buffers(t);
  test_arguments(t, &m);
  bcir_spm_free(t);
  test_laws();
  CHECK(sweep(m.s, m.n, "hello world hell \xE6\x97\xA5 the quick") > 2u);
  free(m.s);
  if (argc > 1) {
    size_t len = 0;
    uint8_t *model = read_file(argv[1], &len);
    if (!model) {
      fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
    }
    CHECK(sweep(model, len, "the quick brown fox jumps over the lazy dog abababab aaaaaaaa") > 2u);
    free(model);
  }
  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  puts("OK");
  return 0;
}
