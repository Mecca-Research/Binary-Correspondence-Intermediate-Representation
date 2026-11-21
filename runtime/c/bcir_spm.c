/*===- bcir_spm.c - SentencePiece-BPE tokenizer (the Llama family's tokenizer.model) -===
 *
 * See bcir_spm.h. The wire reader walks the whole file, once to judge it and count the pieces and
 * once to collect them into one allocation; the laws then run over them in the oracle's order. Encode merges through a heap of candidates keyed by (score
 * descending, position ascending), the choice the oracle's rescan makes at every step; decode
 * streams the pieces' octets through Python's errors='replace' UTF-8 decoder.
 *
 * Held id for id and octet for octet to bcir/frontends/models/spm.py by
 * bcir/tests/test_c_spm_tokenizer.py, and fuzzed under libFuzzer + ASan/UBSan by fuzz_spm.c.
 *===----------------------------------------------------------------------===*/
#include "bcir_spm.h"

#include "bcir_jer.h"
#include "bcir_sha256.h"
#include "bcir_tok_hash.h"

enum { TYPE_NORMAL = 1, TYPE_UNKNOWN = 2, TYPE_CONTROL = 3, TYPE_USER_DEFINED = 4, TYPE_UNUSED = 5, TYPE_BYTE = 6 };
enum { TAG_TEXT = 16 };

#define NONE UINT32_MAX

/* A piece: its text, what it renders as (the text with U+2581 as a space), its score and type. */
typedef struct {
  const uint8_t *text, *render;
  uint32_t len, render_len;
  float score;
  uint8_t type, byte;
} piece_t;

typedef struct { uint64_t hash; uint32_t pid, used; } text_slot;

struct bcir_spm {
  bcir_host_allocator allocator;
  bcir_host_arena arena; /* the pieces, their texts and the table */
  uint8_t digest[32];
  uint64_t k0, k1;
  piece_t *pieces;
  uint32_t n;
  text_slot *table;
  size_t table_cap;
  uint32_t byte_id[256];
  int has_bytes;
  uint32_t unk;
  size_t specials;
};

static int mergeable(uint8_t type) {
  return type == TYPE_NORMAL || type == TYPE_USER_DEFINED || type == TYPE_UNUSED;
}

/* The piece whose text is s[0..n), or NONE. */
static uint32_t find_text(const bcir_spm *t, const uint8_t *s, size_t n) {
  uint64_t h = bcir_tok_siphash(t->k0, t->k1, TAG_TEXT, s, n);
  for (size_t j = (size_t)h & (t->table_cap - 1u); t->table[j].used; j = (j + 1u) & (t->table_cap - 1u)) {
    const piece_t *p = &t->pieces[t->table[j].pid];
    if (t->table[j].hash == h && p->len == n && (n == 0u || memcmp(p->text, s, n) == 0)) return t->table[j].pid;
  }
  return NONE;
}

/* --- the wire reader ------------------------------------------------------------------------- */

typedef struct { bcir_spm_law law; size_t at; } fail_t;

static int fail(fail_t *f, bcir_spm_law law, size_t at) {
  f->law = law;
  f->at = at;
  return 0;
}

/* A varint at b[*i], at most ten octets and at most 2^64 - 1; a non-minimal spelling is read. */
static int read_varint(const uint8_t *b, size_t *i, size_t end, uint64_t *out, fail_t *f) {
  size_t start = *i;
  uint64_t v = 0;
  for (unsigned k = 0; k < 10u; k++) {
    if (*i >= end) return fail(f, BCIR_SPM_LAW_VARINT, start);
    uint8_t c = b[(*i)++];
    if (k == 9u && c > 1u) return fail(f, BCIR_SPM_LAW_VARINT, start);
    v |= (uint64_t)(c & 0x7Fu) << (7u * k);
    if (!(c & 0x80u)) {
      *out = v;
      return 1;
    }
  }
  return fail(f, BCIR_SPM_LAW_VARINT, start); /* unreachable: a tenth octet with 0x80 exceeds 1 */
}

/* `n` octets from *i, inside `end`. */
static int take(size_t *i, uint64_t n, size_t end, size_t at, fail_t *f) {
  if (n > (uint64_t)(end - *i)) return fail(f, BCIR_SPM_LAW_FIELD, at);
  *i += (size_t)n;
  return 1;
}

/* Skip one field's payload by its wire type: varint, 64-bit, length-delimited or 32-bit. */
static int skip(const uint8_t *b, size_t *i, unsigned wire, size_t end, size_t at, fail_t *f) {
  uint64_t v;
  switch (wire) {
  case 0: return read_varint(b, i, end, &v, f);
  case 1: return take(i, 8u, end, at, f);
  case 2: return read_varint(b, i, end, &v, f) && take(i, v, end, at, f);
  case 5: return take(i, 4u, end, at, f);
  default: return fail(f, BCIR_SPM_LAW_FIELD, at); /* the group wire types, and 6 and 7 */
  }
}

typedef struct { size_t text, len, at; float score; uint64_t type; } entry_t;

/* One SentencePiece message, b[i..end). */
static int read_piece(const uint8_t *b, size_t i, size_t end, entry_t *e, fail_t *f) {
  unsigned seen = 0;
  e->text = e->len = 0;
  e->score = 0.0f;
  e->type = TYPE_NORMAL;
  while (i < end) {
    size_t at = i;
    uint64_t tag, v;
    if (!read_varint(b, &i, end, &tag, f)) return 0;
    uint64_t field = tag >> 3;
    unsigned wire = (unsigned)(tag & 7u);
    if (field == 0u) return fail(f, BCIR_SPM_LAW_FIELD, at);
    if (field == 1u && wire == 2u) {
      bcir_jer_diag jd;
      if (seen & 1u) return fail(f, BCIR_SPM_LAW_PIECE_FIELDS, at);
      if (!read_varint(b, &i, end, &v, f)) return 0;
      size_t text = i;
      if (!take(&i, v, end, at, f)) return 0;
      if (bcir_jer_validate_utf8(b + text, i - text, &jd) != BCIR_JER_OK) return fail(f, BCIR_SPM_LAW_PIECE_TEXT, at);
      e->text = text;
      e->len = i - text;
      seen |= 1u;
    } else if (field == 2u && wire == 5u) {
      if (seen & 2u) return fail(f, BCIR_SPM_LAW_PIECE_FIELDS, at);
      size_t s = i;
      if (!take(&i, 4u, end, at, f)) return 0;
      uint32_t bits = (uint32_t)b[s] | (uint32_t)b[s + 1u] << 8 | (uint32_t)b[s + 2u] << 16 | (uint32_t)b[s + 3u] << 24;
      memcpy(&e->score, &bits, sizeof bits);
      seen |= 2u;
    } else if (field == 3u && wire == 0u) {
      if (seen & 4u) return fail(f, BCIR_SPM_LAW_PIECE_FIELDS, at);
      if (!read_varint(b, &i, end, &e->type, f)) return 0;
      seen |= 4u;
    } else if (!skip(b, &i, wire, end, at, f)) {
      return 0;
    }
  }
  if (!(seen & 1u)) return fail(f, BCIR_SPM_LAW_PIECE_FIELDS, e->at);
  {
    uint32_t bits;
    memcpy(&bits, &e->score, sizeof bits);
    if ((bits & 0x7F800000u) == 0x7F800000u) return fail(f, BCIR_SPM_LAW_SCORE, e->at); /* NaN or infinity */
  }
  if (e->type < 1u || e->type > 6u) return fail(f, BCIR_SPM_LAW_TYPE, e->at);
  return 1;
}

/* Every piece of a ModelProto: counted into *n, and written to items[0..*n) when `items` is set (a
 * first pass counts, so the second writes into one allocation). 0 on a refusal, which f names. */
static int read_model(const uint8_t *b, size_t len, entry_t *items, size_t *n, fail_t *f) {
  size_t i = 0;
  *n = 0;
  while (i < len) {
    size_t at = i;
    uint64_t tag, size;
    entry_t scratch;
    if (!read_varint(b, &i, len, &tag, f)) return 0;
    uint64_t field = tag >> 3;
    unsigned wire = (unsigned)(tag & 7u);
    if (field == 0u) return fail(f, BCIR_SPM_LAW_FIELD, at);
    if (field != 1u || wire != 2u) { /* not a piece: skipped */
      if (!skip(b, &i, wire, len, at, f)) return 0;
      continue;
    }
    if (!read_varint(b, &i, len, &size, f)) return 0;
    size_t body = i;
    if (!take(&i, size, len, at, f)) return 0;
    entry_t *e = items ? &items[*n] : &scratch;
    e->at = at;
    if (!read_piece(b, body, i, e, f)) return 0;
    (*n)++;
  }
  return 1;
}

/* --- load ------------------------------------------------------------------------------------ */

static bcir_spm_status finish(bcir_spm_diag *diag, bcir_spm_status st, bcir_spm_law law, size_t at) {
  if (diag) {
    diag->status = st;
    diag->law = law;
    diag->offset = at;
  }
  return st;
}

#define NOMEM(diag) finish((diag), BCIR_SPM_ERR_NOMEM, BCIR_SPM_LAW_NONE, BCIR_SPM_NO_OFFSET)

/* The byte `<0xHH>` names (two uppercase ASCII hex digits), or -1. */
static int byte_of(const uint8_t *s, size_t n) {
  static const char hex[] = "0123456789ABCDEF";
  int hi = -1, lo = -1;
  if (n != 6u || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
  for (int k = 0; k < 16; k++) {
    if ((uint8_t)hex[k] == s[3]) hi = k;
    if ((uint8_t)hex[k] == s[4]) lo = k;
  }
  return hi >= 0 && lo >= 0 ? hi * 16 + lo : -1;
}

/* A copy of `text` in the arena, U+2581 (E2 96 81) written as a space when `render` is set. */
static const uint8_t *keep(bcir_spm *t, const uint8_t *text, size_t n, int render, uint32_t *out_n) {
  uint8_t *p = (uint8_t *)bcir_host_arena_allocate(&t->arena, n ? n : 1u, 1u);
  size_t k = 0;
  if (!p) return 0;
  for (size_t i = 0; i < n; i++) {
    if (render && i + 2u < n && text[i] == 0xE2u && text[i + 1u] == 0x96u && text[i + 2u] == 0x81u) {
      p[k++] = ' ';
      i += 2u;
    } else {
      p[k++] = text[i];
    }
  }
  *out_n = (uint32_t)k;
  return p;
}

static bcir_spm_status apply_laws(bcir_spm *t, const uint8_t *model, const entry_t *e, size_t n, bcir_spm_diag *diag) {
  size_t bytes = 0;
  int have_unk = 0;
  if (!n) return finish(diag, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_NO_PIECES, BCIR_SPM_NO_OFFSET);
  t->table_cap = bcir_tok_pow2(2u * n + 2u);
  if (!bcir_size_mul(n, sizeof(piece_t), &bytes) ||
      !(t->pieces = (piece_t *)bcir_host_arena_allocate(&t->arena, bytes, 0)) ||
      !bcir_size_mul(t->table_cap, sizeof(text_slot), &bytes) ||
      !(t->table = (text_slot *)bcir_host_arena_allocate(&t->arena, bytes, 0)))
    return NOMEM(diag);
  memset(t->table, 0, bytes);
  memset(t->pieces, 0, n * sizeof(piece_t));
  t->n = (uint32_t)n;
  for (size_t pid = 0; pid < n; pid++) {
    const uint8_t *text = model + e[pid].text;
    piece_t *p = &t->pieces[pid];
    uint8_t type = (uint8_t)e[pid].type;
    uint64_t h = bcir_tok_siphash(t->k0, t->k1, TAG_TEXT, text, e[pid].len);
    size_t j = (size_t)h & (t->table_cap - 1u);
    for (; t->table[j].used; j = (j + 1u) & (t->table_cap - 1u)) {
      const piece_t *o = &t->pieces[t->table[j].pid];
      if (t->table[j].hash == h && o->len == e[pid].len && (o->len == 0u || memcmp(o->text, text, o->len) == 0))
        return finish(diag, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_DUPLICATE_TEXT, e[pid].at);
    }
    if (!(p->text = keep(t, text, e[pid].len, 0, &p->len))) return NOMEM(diag);
    p->render = p->text;
    p->render_len = p->len;
    if (mergeable(type) && !(p->render = keep(t, text, e[pid].len, 1, &p->render_len))) return NOMEM(diag);
    p->score = e[pid].score;
    p->type = type;
    p->byte = 0;
    t->table[j].hash = h;
    t->table[j].pid = (uint32_t)pid;
    t->table[j].used = 1u;
    if (type == TYPE_BYTE) {
      int b = byte_of(text, e[pid].len);
      if (b < 0 || t->byte_id[b] != NONE) return finish(diag, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_BYTE_PIECE, e[pid].at);
      t->byte_id[b] = (uint32_t)pid;
      p->byte = (uint8_t)b;
      t->has_bytes++;
    } else if (type == TYPE_UNKNOWN) {
      if (have_unk) return finish(diag, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_UNKNOWN_PIECE, e[pid].at);
      have_unk = 1;
      t->unk = (uint32_t)pid;
      t->specials++;
    } else if (type == TYPE_CONTROL) {
      t->specials++;
    }
  }
  if (!have_unk) return finish(diag, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_UNKNOWN_PIECE, BCIR_SPM_NO_OFFSET);
  if (t->has_bytes && t->has_bytes != 256)
    return finish(diag, BCIR_SPM_ERR_SHAPE, BCIR_SPM_LAW_BYTE_ALPHABET, BCIR_SPM_NO_OFFSET);
  return finish(diag, BCIR_SPM_OK, BCIR_SPM_LAW_NONE, BCIR_SPM_NO_OFFSET);
}

bcir_spm_status bcir_spm_load(const uint8_t *model, size_t len, const bcir_host_allocator *allocator, bcir_spm **out,
                              bcir_spm_diag *diag) {
  bcir_host_allocator a = bcir_host_allocator_or_default(allocator);
  entry_t *items = 0;
  size_t n = 0, bytes = 0;
  fail_t f = {BCIR_SPM_LAW_NONE, BCIR_SPM_NO_OFFSET};
  bcir_spm_status st;
  if (diag) {
    diag->status = BCIR_SPM_OK;
    diag->law = BCIR_SPM_LAW_NONE;
    diag->offset = BCIR_SPM_NO_OFFSET;
  }
  if (!out) return finish(diag, BCIR_SPM_ERR_INVALID, BCIR_SPM_LAW_NONE, BCIR_SPM_NO_OFFSET);
  *out = 0;
  if ((!model && len) || (uint64_t)len >= (uint64_t)UINT32_MAX || !bcir_host_allocator_valid(&a))
    return finish(diag, BCIR_SPM_ERR_INVALID, BCIR_SPM_LAW_NONE, BCIR_SPM_NO_OFFSET);
  if (!read_model(model, len, NULL, &n, &f)) /* the wire format proper is WIRE; a piece rule SHAPE */
    return finish(diag, f.law == BCIR_SPM_LAW_VARINT || f.law == BCIR_SPM_LAW_FIELD ? BCIR_SPM_ERR_WIRE : BCIR_SPM_ERR_SHAPE,
                  f.law, f.at);
  /* the pieces, in a scratch arena released once the laws have run */
  bcir_host_arena scratch;
  if (!bcir_host_arena_init(&scratch, &a, (size_t)1u << 16)) return NOMEM(diag);
  if (n && (!bcir_size_mul(n, sizeof(entry_t), &bytes) ||
            !(items = (entry_t *)bcir_host_arena_allocate(&scratch, bytes, 0)))) {
    bcir_host_arena_destroy(&scratch);
    return NOMEM(diag);
  }
  if (n && !read_model(model, len, items, &n, &f)) { /* the first pass read these octets whole */
    bcir_host_arena_destroy(&scratch);
    return finish(diag, BCIR_SPM_ERR_INVALID, BCIR_SPM_LAW_NONE, BCIR_SPM_NO_OFFSET);
  }
  bcir_spm *t = (bcir_spm *)bcir_host_allocate_zeroed(&a, sizeof *t);
  if (!t || !bcir_host_arena_init(&t->arena, &a, (size_t)1u << 16)) {
    bcir_host_deallocate(&a, t);
    bcir_host_arena_destroy(&scratch);
    return NOMEM(diag);
  }
  t->allocator = a;
  {
    bcir_sha256 sha;
    bcir_sha256_init(&sha);
    bcir_sha256_update(&sha, model, len);
    bcir_sha256_final(&sha, t->digest);
    t->k0 = bcir_tok_le64(t->digest);
    t->k1 = bcir_tok_le64(t->digest + 8);
  }
  for (unsigned b = 0; b < 256u; b++) t->byte_id[b] = NONE;
  t->unk = NONE;
  st = apply_laws(t, model, items, n, diag);
  bcir_host_arena_destroy(&scratch);
  if (st != BCIR_SPM_OK) {
    bcir_spm_free(t);
    return st;
  }
  *out = t;
  return BCIR_SPM_OK;
}

void bcir_spm_free(bcir_spm *t) {
  if (!t) return;
  bcir_host_allocator a = t->allocator;
  bcir_host_arena_destroy(&t->arena);
  bcir_host_deallocate(&a, t);
}

void bcir_spm_digest(const bcir_spm *t, uint8_t sha256[32]) {
  if (t && sha256) memcpy(sha256, t->digest, 32);
}
size_t bcir_spm_piece_count(const bcir_spm *t) { return t ? t->n : 0u; }
uint32_t bcir_spm_unk_id(const bcir_spm *t) { return t ? t->unk : NONE; }
int bcir_spm_has_byte_fallback(const bcir_spm *t) { return t ? t->has_bytes == 256 : 0; }
size_t bcir_spm_special_count(const bcir_spm *t) { return t ? t->specials : 0u; }

/* --- encode ---------------------------------------------------------------------------------- */

/* A candidate merge of symbols `left` and `right`, valid while neither has changed since it was
 * pushed (`vl`, `vr` are their versions then). */
typedef struct { float score; uint32_t left, right, vl, vr; } cand_t;

typedef struct {
  uint8_t *text;                      /* the normalized text */
  uint32_t len;
  uint32_t *start, *next, *prev, *ver; /* per symbol, in text order */
  cand_t *heap;
  size_t heap_n;
} scratch_t;

/* `m` 32-bit words through the allocator, or NULL. */
static uint32_t *alloc_u32(const bcir_host_allocator *a, size_t m) {
  size_t bytes;
  return bcir_size_mul(m, sizeof(uint32_t), &bytes) ? (uint32_t *)bcir_host_allocate(a, bytes) : 0;
}

static int cand_first(const cand_t *x, const cand_t *y) { /* the highest score, then the leftmost */
  if (x->score > y->score) return 1;
  if (x->score < y->score) return 0;
  return x->left < y->left; /* a tie: no score is NaN, the loader refused those */
}

static void heap_push(scratch_t *s, cand_t e) {
  size_t i = s->heap_n++;
  while (i > 0u) {
    size_t p = (i - 1u) / 2u;
    if (!cand_first(&e, &s->heap[p])) break;
    s->heap[i] = s->heap[p];
    i = p;
  }
  s->heap[i] = e;
}

static cand_t heap_pop(scratch_t *s) {
  cand_t top = s->heap[0], last = s->heap[--s->heap_n];
  size_t i = 0, n = s->heap_n;
  if (!n) return top;
  for (;;) {
    size_t c = 2u * i + 1u;
    if (c >= n) break;
    if (c + 1u < n && cand_first(&s->heap[c + 1u], &s->heap[c])) c++;
    if (!cand_first(&s->heap[c], &last)) break;
    s->heap[i] = s->heap[c];
    i = c;
  }
  s->heap[i] = last;
  return top;
}

static uint32_t sym_end(const scratch_t *s, uint32_t k) { return s->next[k] == NONE ? s->len : s->start[s->next[k]]; }

static void push_candidate(const bcir_spm *t, scratch_t *s, uint32_t left) {
  uint32_t right = s->next[left];
  if (right == NONE) return;
  uint32_t a = s->start[left], b = sym_end(s, right);
  uint32_t pid = find_text(t, s->text + a, b - a);
  if (pid == NONE || !mergeable(t->pieces[pid].type)) return;
  cand_t e;
  e.score = t->pieces[pid].score;
  e.left = left;
  e.right = right;
  e.vl = s->ver[left];
  e.vr = s->ver[right];
  heap_push(s, e);
}

typedef struct { uint32_t *ids; size_t cap, n; } id_out;

static void emit(id_out *o, uint32_t id) {
  if (o->n < o->cap) o->ids[o->n] = id;
  o->n++;
}

bcir_spm_status bcir_spm_encode(const bcir_spm *t, const uint8_t *text, size_t len, int add_dummy_prefix,
                                uint32_t *ids, size_t cap, size_t *count) {
  bcir_jer_diag jd;
  scratch_t s;
  id_out o;
  size_t spaces = 0, m = 0, bytes;
  if (count) *count = 0;
  if (!t || !count || (!text && len) || (!ids && cap) || (uint64_t)len >= ((uint64_t)UINT32_MAX - 3u) / 3u)
    return BCIR_SPM_ERR_INVALID;
  if (bcir_jer_validate_utf8(text, len, &jd) != BCIR_JER_OK) return BCIR_SPM_ERR_NOT_UTF8;
  int starts = len && (text[0] == ' ' || (len >= 3u && text[0] == 0xE2u && text[1] == 0x96u && text[2] == 0x81u));
  int prefix = add_dummy_prefix && !starts;
  for (size_t i = 0; i < len; i++) spaces += text[i] == ' ';
  memset(&s, 0, sizeof s);
  s.len = (uint32_t)(len + 2u * spaces + (prefix ? 3u : 0u));
  o.ids = ids;
  o.cap = cap;
  o.n = 0;
  if (s.len == 0u) return BCIR_SPM_OK;

  /* the normalized text: spaces as U+2581, the dummy prefix */
  const bcir_host_allocator *a = &t->allocator;
  if (!(s.text = (uint8_t *)bcir_host_allocate(a, s.len))) return BCIR_SPM_ERR_NOMEM;
  {
    size_t k = 0;
    if (prefix) {
      s.text[k++] = 0xE2u;
      s.text[k++] = 0x96u;
      s.text[k++] = 0x81u;
    }
    for (size_t i = 0; i < len; i++) {
      if (text[i] == ' ') {
        s.text[k++] = 0xE2u;
        s.text[k++] = 0x96u;
        s.text[k++] = 0x81u;
      } else {
        s.text[k++] = text[i];
      }
    }
  }
  for (uint32_t i = 0; i < s.len; i++) m += (s.text[i] & 0xC0u) != 0x80u; /* one symbol a character */
  if (!bcir_size_mul(m, 3u * sizeof(cand_t), &bytes) || !(s.heap = (cand_t *)bcir_host_allocate(a, bytes)) ||
      !(s.start = alloc_u32(a, m)) || !(s.next = alloc_u32(a, m)) || !(s.prev = alloc_u32(a, m)) ||
      !(s.ver = alloc_u32(a, m))) {
    bcir_host_deallocate(a, s.text);
    bcir_host_deallocate(a, s.heap);
    bcir_host_deallocate(a, s.start);
    bcir_host_deallocate(a, s.next);
    bcir_host_deallocate(a, s.prev);
    return BCIR_SPM_ERR_NOMEM;
  }
  {
    uint32_t k = 0;
    for (uint32_t i = 0; i < s.len; i++) {
      if ((s.text[i] & 0xC0u) == 0x80u) continue;
      s.start[k] = i;
      s.prev[k] = k ? k - 1u : NONE;
      s.next[k] = k + 1u < m ? k + 1u : NONE;
      s.ver[k] = 0;
      k++;
    }
  }
  for (uint32_t k = 0; k + 1u < m; k++) push_candidate(t, &s, k);
  while (s.heap_n) {
    cand_t e = heap_pop(&s);
    if (s.ver[e.left] != e.vl || s.ver[e.right] != e.vr) continue; /* a symbol changed: stale */
    uint32_t after = s.next[e.right];
    s.next[e.left] = after;
    if (after != NONE) s.prev[after] = e.left;
    s.ver[e.left]++;
    s.ver[e.right]++; /* the right symbol is gone */
    if (s.prev[e.left] != NONE) push_candidate(t, &s, s.prev[e.left]);
    push_candidate(t, &s, e.left);
  }
  for (uint32_t k = 0; k != NONE; k = s.next[k]) { /* symbol 0 is never absorbed */
    uint32_t a0 = s.start[k], b0 = sym_end(&s, k);
    uint32_t pid = find_text(t, s.text + a0, b0 - a0);
    if (pid != NONE && mergeable(t->pieces[pid].type)) {
      emit(&o, pid);
    } else if (t->has_bytes) { /* byte fallback: the symbol's UTF-8 octets */
      for (uint32_t i = a0; i < b0; i++) emit(&o, t->byte_id[s.text[i]]);
    } else {
      emit(&o, t->unk);
    }
  }
  bcir_host_deallocate(a, s.text);
  bcir_host_deallocate(a, s.heap);
  bcir_host_deallocate(a, s.start);
  bcir_host_deallocate(a, s.next);
  bcir_host_deallocate(a, s.prev);
  bcir_host_deallocate(a, s.ver);
  *count = o.n;
  return o.n > cap ? BCIR_SPM_ERR_NOSPACE : BCIR_SPM_OK;
}

/* --- decode ---------------------------------------------------------------------------------- */

/* Python's UTF-8 decoder with errors='replace', fed octet by octet: each maximal ill-formed
 * subpart becomes U+FFFD; the first character, when it is a space, is dropped. */
typedef struct {
  uint8_t *out;
  size_t cap, at;
  int first;
  uint8_t pend[4];
  unsigned npend, need;
  uint8_t lo, hi;
} replace_t;

static void put(replace_t *r, uint8_t b) {
  if (r->first) {
    r->first = 0;
    if (b == ' ') return;
  }
  if (r->at < r->cap) r->out[r->at] = b;
  r->at++;
}

static void put_fffd(replace_t *r) {
  put(r, 0xEFu);
  put(r, 0xBFu);
  put(r, 0xBDu);
}

static void feed(replace_t *r, uint8_t b) {
  for (;;) {
    if (!r->npend) {
      if (b < 0x80u) {
        put(r, b);
        return;
      }
      r->lo = 0x80u;
      r->hi = 0xBFu;
      if (b >= 0xC2u && b <= 0xDFu) {
        r->need = 1;
      } else if (b >= 0xE0u && b <= 0xEFu) {
        r->need = 2;
        if (b == 0xE0u) r->lo = 0xA0u; /* no overlong three-octet form */
        if (b == 0xEDu) r->hi = 0x9Fu; /* no surrogate */
      } else if (b >= 0xF0u && b <= 0xF4u) {
        r->need = 3;
        if (b == 0xF0u) r->lo = 0x90u; /* no overlong four-octet form */
        if (b == 0xF4u) r->hi = 0x8Fu; /* nothing past U+10FFFF */
      } else {
        put_fffd(r); /* a byte no character starts with */
        return;
      }
      r->pend[r->npend++] = b;
      return;
    }
    if (b < r->lo || b > r->hi) { /* the subpart so far is ill-formed; b starts afresh */
      r->npend = 0;
      put_fffd(r);
      continue;
    }
    r->pend[r->npend++] = b;
    r->lo = 0x80u;
    r->hi = 0xBFu;
    if (r->npend == r->need + 1u) {
      for (unsigned k = 0; k < r->npend; k++) put(r, r->pend[k]);
      r->npend = 0;
    }
    return;
  }
}

bcir_spm_status bcir_spm_decode(const bcir_spm *t, const uint32_t *ids, size_t n, uint8_t *out, size_t cap,
                                size_t *written) {
  replace_t r;
  if (written) *written = 0;
  if (!t || !written || (!ids && n) || (!out && cap)) return BCIR_SPM_ERR_INVALID;
  for (size_t i = 0; i < n; i++)
    if (ids[i] >= t->n) return BCIR_SPM_ERR_UNKNOWN_ID;
  memset(&r, 0, sizeof r);
  r.out = out;
  r.cap = cap;
  r.first = 1;
  for (size_t i = 0; i < n; i++) {
    const piece_t *p = &t->pieces[ids[i]];
    if (p->type == TYPE_CONTROL || p->type == TYPE_UNKNOWN) continue; /* renders as nothing */
    if (p->type == TYPE_BYTE) {
      feed(&r, p->byte);
    } else {
      for (uint32_t k = 0; k < p->render_len; k++) feed(&r, p->render[k]);
    }
  }
  if (r.npend) put_fffd(&r); /* a sequence the octets end inside */
  *written = r.at;
  return r.at > cap ? BCIR_SPM_ERR_NOSPACE : BCIR_SPM_OK;
}
