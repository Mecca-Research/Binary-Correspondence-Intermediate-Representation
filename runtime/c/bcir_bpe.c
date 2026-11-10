/*===- bcir_bpe.c - byte-level BPE tokenizer (the GPT-2 / Qwen tokenizer.json family) ===
 *
 * See bcir_bpe.h. The bounded JER reader reads the file once; this unit's sink turns its events
 * into raw records (vocab entries, merges, added tokens), and the laws then run over the records
 * in the oracle's order, so the order of members in the file never changes a verdict.
 *
 * Every table is an open-addressed hash keyed by SipHash-2-4 under a key taken from the file's own
 * SHA-256. The tables stay deterministic, and a file cannot choose its collisions without first
 * choosing its own digest. Encode and decode only look keys up; they never insert, so text cannot
 * grow a table either.
 *
 * Held id for id to bcir/frontends/models/tokenizer.py by bcir/tests/test_c_bpe_tokenizer.py, and
 * fuzzed under libFuzzer + ASan/UBSan by fuzz_bpe.c.
 *===----------------------------------------------------------------------===*/
#include "bcir_bpe.h"

#include "bcir_sha256.h"

/* --- the byte alphabet: bytes_to_unicode ----------------------------------------------------
 * Every byte maps to one code point: the printable ranges to themselves, the other 68 to
 * U+0100.. in byte order. `u2b` inverts it over U+0000..U+0143, -1 outside the alphabet. */
#define ALPHABET_TOP 0x144u

typedef struct { int16_t u2b[ALPHABET_TOP]; } alphabet_t;

static int printable_byte(unsigned b) {
  return (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || (b >= 0xAEu && b <= 0xFFu);
}

static void build_alphabet(alphabet_t *al) {
  unsigned shifted = 0;
  for (unsigned c = 0; c < ALPHABET_TOP; c++) al->u2b[c] = -1;
  for (unsigned b = 0; b < 256u; b++) al->u2b[printable_byte(b) ? b : 256u + shifted++] = (int16_t)b;
}

/* Alphabet text (UTF-8) to the bytes it stands for, one byte per code point; 0 when a code point
 * lies outside the alphabet. `out` needs room for `n` octets. */
static int unalphabet(const alphabet_t *al, const uint8_t *text, size_t n, uint8_t *out, size_t *out_n) {
  size_t i = 0, k = 0;
  while (i < n) {
    uint32_t cp;
    size_t w;
    if (bcir_jer_utf8_next(text, n, i, &cp, &w) != BCIR_JER_OK) return 0;
    if (cp >= ALPHABET_TOP || al->u2b[cp] < 0) return 0;
    out[k++] = (uint8_t)al->u2b[cp];
    i += w;
  }
  *out_n = k;
  return 1;
}

/* --- SipHash-2-4 -------------------------------------------------------------------------- */

static uint64_t rotl64(uint64_t x, unsigned b) { return (x << b) | (x >> (64u - b)); }
static uint64_t le64(const uint8_t *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8u; i++) v |= (uint64_t)p[i] << (8u * i);
  return v;
}
#define SIPROUND                                                                \
  do {                                                                          \
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);               \
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;                                    \
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;                                    \
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);               \
  } while (0)

/* SipHash-2-4 under (k0, k1) of the message `tag` (8 octets, little-endian) || data[0..len). */
static uint64_t siphash(uint64_t k0, uint64_t k1, uint64_t tag, const uint8_t *data, size_t len) {
  uint64_t v0 = k0 ^ UINT64_C(0x736f6d6570736575), v1 = k1 ^ UINT64_C(0x646f72616e646f6d);
  uint64_t v2 = k0 ^ UINT64_C(0x6c7967656e657261), v3 = k1 ^ UINT64_C(0x7465646279746573);
  uint64_t m;
  size_t i = 0;
  v3 ^= tag; SIPROUND; SIPROUND; v0 ^= tag;
  for (; i + 8u <= len; i += 8u) {
    m = le64(data + i);
    v3 ^= m; SIPROUND; SIPROUND; v0 ^= m;
  }
  m = ((uint64_t)len + 8u) << 56;
  for (size_t k = 0; i + k < len; k++) m |= (uint64_t)data[i + k] << (8u * k);
  v3 ^= m; SIPROUND; SIPROUND; v0 ^= m;
  v2 ^= 0xFFu;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return v0 ^ v1 ^ v2 ^ v3;
}

enum { TAG_NAME = 0, TAG_VOCAB = 1, TAG_PAIR = 2, TAG_ID = 3, TAG_TRIE = 4 };

static uint64_t hash_u32x2(uint64_t k0, uint64_t k1, uint64_t tag, uint32_t a, uint32_t b) {
  uint8_t k[8];
  for (unsigned i = 0; i < 4u; i++) {
    k[i] = (uint8_t)(a >> (8u * i));
    k[4u + i] = (uint8_t)(b >> (8u * i));
  }
  return siphash(k0, k1, tag, k, 8u);
}

static size_t pow2_at_least(size_t n) {
  size_t c = 16u;
  while (c < n) c *= 2u;
  return c;
}

/* --- records the sink collects --------------------------------------------------------------- */

typedef struct { const uint8_t *s; uint32_t n; } span_t;            /* octets in the arena */
typedef struct { span_t token; uint32_t id; size_t at; } vocab_rec;   /* the bytes a token is */
typedef struct { span_t left, right; size_t at; } merge_rec;          /* alphabet text (UTF-8) */
typedef struct { span_t content; uint32_t id; size_t at; } added_rec; /* UTF-8 text */

/* A growable array, through the lent allocator. */
typedef struct { void *items; size_t n, cap, size; } grow_t;

static int grow_push(const bcir_host_allocator *a, grow_t *g, const void *item) {
  if (g->n == g->cap) {
    size_t cap = 0;
    if (!bcir_host_grow_capacity(g->cap, g->n + 1u, g->size, &cap)) return 0;
    if (!bcir_host_realloc_array(a, &g->items, g->cap, cap, g->size, 0)) return 0;
    g->cap = cap;
  }
  memcpy((uint8_t *)g->items + g->n * g->size, item, g->size);
  g->n++;
  return 1;
}

/* --- the member names of one document: an open-addressed set of (object, name) --------------- */

typedef struct { uint64_t hash, object; span_t name; } name_slot;
typedef struct { name_slot *slots; size_t cap, n; } name_set;

/* *fresh = 0 when `object` already has a member `name`. 0 only when the allocator refuses. */
static int name_set_insert(const bcir_host_allocator *a, name_set *set, uint64_t k0, uint64_t k1,
                           uint64_t object, span_t name, int *fresh) {
  if ((set->n + 1u) * 2u > set->cap) { /* the load stays at most one half */
    size_t cap = set->cap ? set->cap * 2u : 64u, bytes;
    if (!bcir_size_mul(cap, sizeof(name_slot), &bytes)) return 0;
    name_slot *slots = (name_slot *)bcir_host_allocate_zeroed(a, bytes);
    if (!slots) return 0;
    for (size_t i = 0; i < set->cap; i++) {
      if (!set->slots[i].name.s) continue;
      size_t j = (size_t)set->slots[i].hash & (cap - 1u);
      while (slots[j].name.s) j = (j + 1u) & (cap - 1u);
      slots[j] = set->slots[i];
    }
    bcir_host_deallocate(a, set->slots);
    set->slots = slots;
    set->cap = cap;
  }
  uint64_t h = siphash(k0, k1, TAG_NAME ^ (object << 8), name.s, name.n);
  size_t j = (size_t)h & (set->cap - 1u);
  for (; set->slots[j].name.s; j = (j + 1u) & (set->cap - 1u)) {
    const name_slot *e = &set->slots[j];
    if (e->hash == h && e->object == object && e->name.n == name.n && memcmp(e->name.s, name.s, name.n) == 0) {
      *fresh = 0;
      return 1;
    }
  }
  set->slots[j].hash = h;
  set->slots[j].object = object;
  set->slots[j].name = name;
  set->n++;
  *fresh = 1;
  return 1;
}

/* --- the tokenizer --------------------------------------------------------------------------- */

typedef struct { uint64_t hash; uint32_t index; uint32_t used; } vocab_slot;  /* bytes -> token */
typedef struct { uint32_t id, entry, used; } id_slot;                         /* id -> entry */
typedef struct { uint32_t left, right, rank, merged, used; } merge_slot;      /* id pair -> merge */
typedef struct { uint32_t node, child, byte, used; } trie_slot;               /* (node, octet) -> child */

/* What an id renders as: a special as its UTF-8 text (flushing the byte run before it), every
 * other token as its bytes. */
typedef struct { span_t bytes; uint32_t special; } id_entry;

struct bcir_bpe {
  bcir_host_allocator allocator;
  bcir_host_arena arena; /* every table and every kept octet */
  uint8_t digest[32];
  uint64_t k0, k1; /* the SipHash key: the digest's first 16 octets */
  uint32_t byte_id[256]; /* each byte's base token */
  vocab_rec *tokens; vocab_slot *vocab; size_t vocab_cap, vocab_n;
  id_entry *entries; size_t entries_n; id_slot *ids; size_t ids_cap;
  merge_slot *merges; size_t merges_cap, merges_n;
  added_rec *specials; size_t specials_n; /* in the file's order */
  /* The specials as a trie over their octets: node 0 is the root, whose children are direct;
   * `term[node]` is the special that ends there, or -1. */
  uint32_t root_child[256];
  trie_slot *trie; size_t trie_cap; int32_t *term; uint32_t nodes;
};

static int vocab_find(const bcir_bpe *t, const uint8_t *s, size_t n, uint32_t *id) {
  uint64_t h = siphash(t->k0, t->k1, TAG_VOCAB, s, n);
  for (size_t j = (size_t)h & (t->vocab_cap - 1u); t->vocab[j].used; j = (j + 1u) & (t->vocab_cap - 1u)) {
    const vocab_rec *r = &t->tokens[t->vocab[j].index];
    if (t->vocab[j].hash == h && r->token.n == n && memcmp(r->token.s, s, n) == 0) {
      *id = r->id;
      return 1;
    }
  }
  return 0;
}

static const id_slot *id_find(const bcir_bpe *t, uint32_t id) {
  uint64_t h = hash_u32x2(t->k0, t->k1, TAG_ID, id, 0u);
  for (size_t j = (size_t)h & (t->ids_cap - 1u); t->ids[j].used; j = (j + 1u) & (t->ids_cap - 1u))
    if (t->ids[j].id == id) return &t->ids[j];
  return 0;
}

/* 0 when `id` already names an entry. */
static int id_insert(bcir_bpe *t, uint32_t id, uint32_t entry) {
  uint64_t h = hash_u32x2(t->k0, t->k1, TAG_ID, id, 0u);
  size_t j = (size_t)h & (t->ids_cap - 1u);
  for (; t->ids[j].used; j = (j + 1u) & (t->ids_cap - 1u))
    if (t->ids[j].id == id) return 0;
  t->ids[j].id = id;
  t->ids[j].entry = entry;
  t->ids[j].used = 1u;
  return 1;
}

static const merge_slot *merge_find(const bcir_bpe *t, uint32_t a, uint32_t b) {
  if (!t->merges_cap) return 0;
  uint64_t h = hash_u32x2(t->k0, t->k1, TAG_PAIR, a, b);
  for (size_t j = (size_t)h & (t->merges_cap - 1u); t->merges[j].used; j = (j + 1u) & (t->merges_cap - 1u))
    if (t->merges[j].left == a && t->merges[j].right == b) return &t->merges[j];
  return 0;
}

/* The child of `node` along octet `b`, or 0 (the root is nobody's child). */
static uint32_t trie_child(const bcir_bpe *t, uint32_t node, uint8_t b) {
  if (node == 0u) return t->root_child[b];
  uint64_t h = hash_u32x2(t->k0, t->k1, TAG_TRIE, node, b);
  for (size_t j = (size_t)h & (t->trie_cap - 1u); t->trie[j].used; j = (j + 1u) & (t->trie_cap - 1u))
    if (t->trie[j].node == node && t->trie[j].byte == b) return t->trie[j].child;
  return 0u;
}

/* The node `content` ends at, created as needed. The trie was sized for every special's octets. */
static uint32_t trie_insert(bcir_bpe *t, const span_t *content) {
  uint32_t node = 0u;
  for (uint32_t k = 0; k < content->n; k++) {
    uint8_t b = content->s[k];
    uint32_t child = trie_child(t, node, b);
    if (!child) {
      child = t->nodes++;
      t->term[child] = -1;
      if (node == 0u) {
        t->root_child[b] = child;
      } else {
        uint64_t h = hash_u32x2(t->k0, t->k1, TAG_TRIE, node, b);
        size_t j = (size_t)h & (t->trie_cap - 1u);
        while (t->trie[j].used) j = (j + 1u) & (t->trie_cap - 1u);
        t->trie[j].node = node;
        t->trie[j].byte = b;
        t->trie[j].child = child;
        t->trie[j].used = 1u;
      }
    }
    node = child;
  }
  return node;
}

/* The longest special at text[i..len), or -1. */
static int32_t special_at(const bcir_bpe *t, const uint8_t *text, size_t len, size_t i) {
  int32_t best = -1;
  uint32_t node = 0u;
  for (size_t k = i; k < len; k++) {
    node = trie_child(t, node, text[k]);
    if (!node) break;
    if (t->term[node] >= 0) best = t->term[node];
  }
  return best;
}

/* --- the sink -------------------------------------------------------------------------------- */

typedef enum { CTX_ROOT, CTX_MODEL, CTX_VOCAB, CTX_MERGES, CTX_PAIR, CTX_ADDED, CTX_ITEM, CTX_IGNORED } ctx_role;

typedef enum { /* what the value after a member name means */
  V_IGNORE, V_MODEL, V_ADDED, V_TYPE, V_VOCAB, V_MERGES, V_VOCAB_ID, V_CONTENT, V_ITEM_ID
} value_role;

typedef struct {
  ctx_role role;
  int is_object;
  uint64_t serial;    /* the object's key in the name set */
  value_role pending; /* objects: the meaning of the value after the last member name */
  uint32_t count;     /* CTX_PAIR: the parts seen */
  span_t pair[2];
  int has_content, has_id; /* CTX_ITEM */
  span_t content;
  uint32_t id;
  size_t at;
} ctx_t;

/* The reader's depth ceiling is the stack's: one entry per open container. */
#define MAX_DEPTH 64u

typedef struct {
  bcir_bpe *t;
  const bcir_host_allocator *a;
  alphabet_t alphabet;
  ctx_t stack[MAX_DEPTH];
  size_t depth;
  uint64_t serials;
  name_set names;
  grow_t vocab, merges, added;
  int seen_root, seen_model, type_is_bpe;
  span_t pending_token; /* the vocab member name awaiting its id, as bytes */
  bcir_bpe_law law;
  size_t law_at;
  int nomem;
} sink_t;

static int refuse(sink_t *s, bcir_bpe_law law, size_t at) {
  if (s->law == BCIR_BPE_LAW_NONE) {
    s->law = law;
    s->law_at = at;
  }
  return 1;
}

static int out_of_memory(sink_t *s) {
  s->nomem = 1;
  return 2;
}

/* Keep `n` octets in the arena. */
static int keep(sink_t *s, const uint8_t *text, size_t n, span_t *out) {
  uint8_t *p = (uint8_t *)bcir_host_arena_allocate(&s->t->arena, n ? n : 1u, 1u);
  if (!p) return 0;
  if (n) memcpy(p, text, n);
  out->s = p;
  out->n = (uint32_t)n;
  return 1;
}

/* A token id from a raw number token: `-?digits` only (a fraction or an exponent makes the
 * oracle's json a float, which is no id), in [0, 2^31-1]. "-0" is the integer 0. */
static int token_id(const uint8_t *raw, size_t n, uint32_t *out) {
  size_t i = 0;
  int negative = 0;
  uint64_t v = 0;
  if (n && raw[0] == '-') {
    negative = 1;
    i = 1;
  }
  if (i >= n) return 0;
  for (; i < n; i++) {
    if (raw[i] < '0' || raw[i] > '9') return 0;
    v = v * 10u + (uint64_t)(raw[i] - '0');
    if (v > 0x7FFFFFFFu) return 0;
  }
  if (negative && v != 0u) return 0;
  *out = (uint32_t)v;
  return 1;
}

/* 0 when the stack is full, which the reader's depth ceiling rules out. */
static int push_ctx(sink_t *s, ctx_role role, int is_object, size_t at) {
  if (s->depth >= MAX_DEPTH) return 0;
  ctx_t *c = &s->stack[s->depth++];
  memset(c, 0, sizeof *c);
  c->role = role;
  c->is_object = is_object;
  c->at = at;
  c->serial = ++s->serials;
  c->pending = V_IGNORE;
  return 1;
}

static int is_name(const uint8_t *text, size_t len, const char *name) {
  size_t n = strlen(name);
  return len == n && memcmp(text, name, n) == 0;
}

static int on_member_name(sink_t *s, ctx_t *c, size_t at, const uint8_t *text, size_t len) {
  if (c->role == CTX_VOCAB) {
    /* A nonempty token inside the byte alphabet, kept as the bytes it stands for. The alphabet
     * is a bijection, so two names repeat exactly when their bytes do, and the law pass finds
     * that through the vocab table without a second copy of every name. */
    uint8_t *bytes;
    size_t n = 0;
    if (!len) return refuse(s, BCIR_BPE_LAW_VOCAB, at);
    if (!(bytes = (uint8_t *)bcir_host_arena_allocate(&s->t->arena, len, 1u))) return out_of_memory(s);
    if (!unalphabet(&s->alphabet, text, len, bytes, &n)) return refuse(s, BCIR_BPE_LAW_VOCAB, at);
    s->pending_token.s = bytes;
    s->pending_token.n = (uint32_t)n;
    c->pending = V_VOCAB_ID;
    return 0;
  }
  {
    span_t name;
    int fresh = 0;
    if (!keep(s, text, len, &name)) return out_of_memory(s);
    if (!name_set_insert(s->a, &s->names, s->t->k0, s->t->k1, c->serial, name, &fresh)) return out_of_memory(s);
    if (!fresh) return refuse(s, BCIR_BPE_LAW_DUPLICATE_KEY, at);
  }
  c->pending = V_IGNORE;
  if (c->role == CTX_ROOT) {
    if (is_name(text, len, "model")) c->pending = V_MODEL;
    else if (is_name(text, len, "added_tokens")) c->pending = V_ADDED;
  } else if (c->role == CTX_MODEL) {
    if (is_name(text, len, "type")) c->pending = V_TYPE;
    else if (is_name(text, len, "vocab")) c->pending = V_VOCAB;
    else if (is_name(text, len, "merges")) c->pending = V_MERGES;
  } else if (c->role == CTX_ITEM) {
    if (is_name(text, len, "content")) c->pending = V_CONTENT;
    else if (is_name(text, len, "id")) c->pending = V_ITEM_ID;
  }
  return 0;
}

static int on_end(sink_t *s, ctx_t *c) {
  if (c->role == CTX_PAIR) {
    merge_rec m;
    if (c->count != 2u) return refuse(s, BCIR_BPE_LAW_MERGES, c->at);
    m.left = c->pair[0];
    m.right = c->pair[1];
    m.at = c->at;
    if (!grow_push(s->a, &s->merges, &m)) return out_of_memory(s);
  } else if (c->role == CTX_ITEM) {
    added_rec r;
    if (!c->has_content || !c->has_id) return refuse(s, BCIR_BPE_LAW_ADDED, c->at);
    r.content = c->content;
    r.id = c->id;
    r.at = c->at;
    if (!grow_push(s->a, &s->added, &r)) return out_of_memory(s);
  }
  s->depth--;
  return 0;
}

/* A value inside an array: its meaning comes from the array. */
static int on_element(sink_t *s, ctx_t *c, bcir_jer_event ev, size_t at, const uint8_t *text, size_t len) {
  if (c->role == CTX_MERGES) {
    if (ev == BCIR_JER_EV_STRING) { /* "a b": two nonempty parts around exactly one space */
      size_t sp = len, spaces = 0;
      merge_rec m;
      for (size_t i = 0; i < len; i++)
        if (text[i] == ' ') {
          spaces++;
          sp = i;
        }
      if (spaces != 1u || sp == 0u || sp + 1u == len) return refuse(s, BCIR_BPE_LAW_MERGES, at);
      if (!keep(s, text, sp, &m.left) || !keep(s, text + sp + 1u, len - sp - 1u, &m.right)) return out_of_memory(s);
      m.at = at;
      return grow_push(s->a, &s->merges, &m) ? 0 : out_of_memory(s);
    }
    if (ev == BCIR_JER_EV_ARRAY_BEGIN) return push_ctx(s, CTX_PAIR, 0, at) ? 0 : refuse(s, BCIR_BPE_LAW_MERGES, at);
    return refuse(s, BCIR_BPE_LAW_MERGES, at);
  }
  if (c->role == CTX_PAIR) { /* [a, b]: two nonempty strings */
    if (ev != BCIR_JER_EV_STRING || !len || c->count >= 2u) return refuse(s, BCIR_BPE_LAW_MERGES, at);
    if (!keep(s, text, len, &c->pair[c->count])) return out_of_memory(s);
    c->count++;
    return 0;
  }
  if (c->role == CTX_ADDED) {
    if (ev != BCIR_JER_EV_OBJECT_BEGIN) return refuse(s, BCIR_BPE_LAW_ADDED, at);
    return push_ctx(s, CTX_ITEM, 1, at) ? 0 : refuse(s, BCIR_BPE_LAW_ADDED, at);
  }
  if (ev == BCIR_JER_EV_OBJECT_BEGIN || ev == BCIR_JER_EV_ARRAY_BEGIN) /* inside an ignored array */
    return push_ctx(s, CTX_IGNORED, ev == BCIR_JER_EV_OBJECT_BEGIN, at) ? 0 : refuse(s, BCIR_BPE_LAW_ROOT, at);
  return 0;
}

/* A member's value: its meaning comes from the member name before it. */
static int on_member_value(sink_t *s, value_role role, bcir_jer_event ev, size_t at, const uint8_t *text, size_t len) {
  ctx_t *c = &s->stack[s->depth - 1u];
  switch (role) {
  case V_MODEL:
    if (ev != BCIR_JER_EV_OBJECT_BEGIN) return refuse(s, BCIR_BPE_LAW_MODEL, at);
    s->seen_model = 1;
    return push_ctx(s, CTX_MODEL, 1, at) ? 0 : refuse(s, BCIR_BPE_LAW_MODEL, at);
  case V_ADDED:
    if (ev != BCIR_JER_EV_ARRAY_BEGIN) return refuse(s, BCIR_BPE_LAW_ADDED, at);
    return push_ctx(s, CTX_ADDED, 0, at) ? 0 : refuse(s, BCIR_BPE_LAW_ADDED, at);
  case V_TYPE:
    if (ev != BCIR_JER_EV_STRING || !is_name(text, len, "BPE")) return refuse(s, BCIR_BPE_LAW_MODEL, at);
    s->type_is_bpe = 1;
    return 0;
  case V_VOCAB:
    if (ev != BCIR_JER_EV_OBJECT_BEGIN) return refuse(s, BCIR_BPE_LAW_VOCAB, at);
    return push_ctx(s, CTX_VOCAB, 1, at) ? 0 : refuse(s, BCIR_BPE_LAW_VOCAB, at);
  case V_MERGES:
    if (ev != BCIR_JER_EV_ARRAY_BEGIN) return refuse(s, BCIR_BPE_LAW_MERGES, at);
    return push_ctx(s, CTX_MERGES, 0, at) ? 0 : refuse(s, BCIR_BPE_LAW_MERGES, at);
  case V_VOCAB_ID: {
    vocab_rec r;
    if (ev != BCIR_JER_EV_NUMBER || !token_id(text, len, &r.id)) return refuse(s, BCIR_BPE_LAW_ID, at);
    r.token = s->pending_token;
    r.at = at;
    return grow_push(s->a, &s->vocab, &r) ? 0 : out_of_memory(s);
  }
  case V_CONTENT:
    if (ev != BCIR_JER_EV_STRING || !len) return refuse(s, BCIR_BPE_LAW_ADDED, at);
    if (!keep(s, text, len, &c->content)) return out_of_memory(s);
    c->has_content = 1;
    return 0;
  case V_ITEM_ID:
    if (ev != BCIR_JER_EV_NUMBER || !token_id(text, len, &c->id)) return refuse(s, BCIR_BPE_LAW_ID, at);
    c->has_id = 1;
    return 0;
  case V_IGNORE:
  default:
    if (ev == BCIR_JER_EV_OBJECT_BEGIN || ev == BCIR_JER_EV_ARRAY_BEGIN)
      return push_ctx(s, CTX_IGNORED, ev == BCIR_JER_EV_OBJECT_BEGIN, at) ? 0 : refuse(s, BCIR_BPE_LAW_ROOT, at);
    return 0;
  }
}

static int bpe_sink(void *ctx, bcir_jer_event ev, size_t at, const uint8_t *text, size_t len) {
  sink_t *s = (sink_t *)ctx;
  if (s->nomem) return 2;
  if (!s->seen_root) {
    s->seen_root = 1;
    if (ev != BCIR_JER_EV_OBJECT_BEGIN) return refuse(s, BCIR_BPE_LAW_ROOT, at);
    return push_ctx(s, CTX_ROOT, 1, at) ? 0 : refuse(s, BCIR_BPE_LAW_ROOT, at);
  }
  if (!s->depth) return refuse(s, BCIR_BPE_LAW_ROOT, at); /* past the root: the reader refuses too */
  ctx_t *c = &s->stack[s->depth - 1u];
  if (ev == BCIR_JER_EV_MEMBER_NAME) return on_member_name(s, c, at, text, len);
  if (ev == BCIR_JER_EV_OBJECT_END || ev == BCIR_JER_EV_ARRAY_END) return on_end(s, c);
  if (!c->is_object) return on_element(s, c, ev, at, text, len);
  value_role role = c->pending;
  c->pending = V_IGNORE; /* one value per name */
  return on_member_value(s, role, ev, at, text, len);
}

/* --- load ------------------------------------------------------------------------------------ */

void bcir_bpe_limits(bcir_jer_limits *out) {
  if (!out) return;
  out->input_bytes = (uint64_t)256u << 20;
  out->depth = MAX_DEPTH;
  out->nodes = (uint64_t)8u << 20;
  out->members = (uint64_t)1u << 20;
  out->elements = (uint64_t)2u << 20;
  out->string_bytes = (uint64_t)1u << 20;
  out->number_bytes = 128u;
  out->integer_digits = 64u;
  out->exponent_magnitude = 4096u;
  out->work = (uint64_t)1u << 30;
}

static bcir_bpe_status finish(bcir_bpe_diag *diag, bcir_bpe_status st, bcir_bpe_law law, size_t at) {
  if (diag) {
    diag->status = st;
    diag->law = law;
    diag->offset = at;
  }
  return st;
}

#define NOMEM(diag) finish((diag), BCIR_BPE_ERR_NOMEM, BCIR_BPE_LAW_NONE, BCIR_JER_NO_OFFSET)
#define SHAPE(diag, law, at) finish((diag), BCIR_BPE_ERR_SHAPE, (law), (at))

/* `count` zeroed elements of `size` octets from the arena, or NULL. */
static void *arena_table(bcir_bpe *t, size_t count, size_t size) {
  size_t bytes;
  void *p;
  if (!bcir_size_mul(count ? count : 1u, size, &bytes)) return 0;
  if (!(p = bcir_host_arena_allocate(&t->arena, bytes, 0))) return 0;
  memset(p, 0, bytes);
  return p;
}

/* The vocab: tokens to distinct ids, and every base byte present. */
static bcir_bpe_status vocab_laws(bcir_bpe *t, const grow_t *records, bcir_bpe_diag *diag, size_t extra_ids) {
  size_t nv = records->n;
  t->vocab_cap = pow2_at_least(2u * nv + 2u);
  t->ids_cap = pow2_at_least(2u * (nv + extra_ids) + 2u);
  t->tokens = (vocab_rec *)arena_table(t, nv, sizeof(vocab_rec));
  t->vocab = (vocab_slot *)arena_table(t, t->vocab_cap, sizeof(vocab_slot));
  t->ids = (id_slot *)arena_table(t, t->ids_cap, sizeof(id_slot));
  t->entries = (id_entry *)arena_table(t, nv + extra_ids, sizeof(id_entry));
  if (!t->tokens || !t->vocab || !t->ids || !t->entries) return NOMEM(diag);
  if (nv) memcpy(t->tokens, records->items, nv * sizeof(vocab_rec));
  for (size_t i = 0; i < nv; i++) {
    const vocab_rec *r = &t->tokens[i];
    uint64_t h = siphash(t->k0, t->k1, TAG_VOCAB, r->token.s, r->token.n);
    size_t j = (size_t)h & (t->vocab_cap - 1u);
    for (; t->vocab[j].used; j = (j + 1u) & (t->vocab_cap - 1u)) {
      const vocab_rec *o = &t->tokens[t->vocab[j].index];
      if (t->vocab[j].hash == h && o->token.n == r->token.n && memcmp(o->token.s, r->token.s, r->token.n) == 0)
        return SHAPE(diag, BCIR_BPE_LAW_DUPLICATE_KEY, r->at);
    }
    if (!id_insert(t, r->id, (uint32_t)t->entries_n)) return SHAPE(diag, BCIR_BPE_LAW_ID, r->at);
    t->vocab[j].hash = h;
    t->vocab[j].index = (uint32_t)i;
    t->vocab[j].used = 1u;
    t->vocab_n++;
    t->entries[t->entries_n].bytes = r->token;
    t->entries[t->entries_n].special = 0u;
    t->entries_n++;
  }
  for (unsigned b = 0; b < 256u; b++) {
    uint8_t one = (uint8_t)b;
    if (!vocab_find(t, &one, 1u, &t->byte_id[b])) return SHAPE(diag, BCIR_BPE_LAW_BASE_BYTES, BCIR_JER_NO_OFFSET);
  }
  return BCIR_BPE_OK;
}

/* The merges: rank = index; each pair once; both parts and their concatenation vocab tokens. */
static bcir_bpe_status merge_laws(bcir_bpe *t, const alphabet_t *al, const grow_t *records, bcir_bpe_diag *diag) {
  const merge_rec *merges = (const merge_rec *)records->items;
  size_t nm = records->n;
  if (!nm) return BCIR_BPE_OK;
  t->merges_cap = pow2_at_least(2u * nm + 2u);
  if (!(t->merges = (merge_slot *)arena_table(t, t->merges_cap, sizeof(merge_slot)))) return NOMEM(diag);
  for (size_t i = 0; i < nm; i++) {
    const merge_rec *m = &merges[i];
    size_t ln = 0, rn = 0;
    uint32_t left, right, merged;
    uint8_t *cat = (uint8_t *)bcir_host_arena_allocate(&t->arena, (size_t)m->left.n + m->right.n, 1u);
    if (!cat) return NOMEM(diag);
    if (!unalphabet(al, m->left.s, m->left.n, cat, &ln) || !unalphabet(al, m->right.s, m->right.n, cat + ln, &rn) ||
        !vocab_find(t, cat, ln, &left) || !vocab_find(t, cat + ln, rn, &right) || !vocab_find(t, cat, ln + rn, &merged))
      return SHAPE(diag, BCIR_BPE_LAW_MERGES, m->at);
    uint64_t h = hash_u32x2(t->k0, t->k1, TAG_PAIR, left, right);
    size_t j = (size_t)h & (t->merges_cap - 1u);
    for (; t->merges[j].used; j = (j + 1u) & (t->merges_cap - 1u))
      if (t->merges[j].left == left && t->merges[j].right == right) return SHAPE(diag, BCIR_BPE_LAW_MERGES, m->at);
    merge_slot *e = &t->merges[j];
    e->left = left;
    e->right = right;
    e->rank = (uint32_t)i;
    e->merged = merged;
    e->used = 1u;
    t->merges_n++;
  }
  return BCIR_BPE_OK;
}

/* The added tokens: contents distinct; a content in the vocab only under its vocab id; an id never
 * naming two texts. Each becomes a special, entered in the encoder's trie. */
static bcir_bpe_status added_laws(bcir_bpe *t, const alphabet_t *al, const grow_t *records, bcir_bpe_diag *diag) {
  const added_rec *added = (const added_rec *)records->items;
  size_t na = records->n, octets = 0;
  if (!na) return BCIR_BPE_OK;
  for (size_t i = 0; i < na; i++) octets += added[i].content.n; /* each kept from one string of the file */
  t->trie_cap = pow2_at_least(2u * octets + 2u);
  t->specials = (added_rec *)arena_table(t, na, sizeof(added_rec));
  t->trie = (trie_slot *)arena_table(t, t->trie_cap, sizeof(trie_slot));
  t->term = (int32_t *)arena_table(t, octets + 1u, sizeof(int32_t));
  if (!t->specials || !t->trie || !t->term) return NOMEM(diag);
  t->term[0] = -1;
  t->nodes = 1u;
  for (size_t i = 0; i < na; i++) {
    const added_rec *r = &added[i];
    uint32_t end = trie_insert(t, &r->content), vid;
    if (t->term[end] >= 0) return SHAPE(diag, BCIR_BPE_LAW_ADDED, r->at); /* a repeated content */
    /* the content as vocab bytes, when it is alphabet text */
    uint8_t *as_bytes = (uint8_t *)bcir_host_arena_allocate(&t->arena, r->content.n, 1u);
    size_t bn = 0;
    if (!as_bytes) return NOMEM(diag);
    int alphabet = unalphabet(al, r->content.s, r->content.n, as_bytes, &bn);
    if (alphabet && vocab_find(t, as_bytes, bn, &vid) && vid != r->id) return SHAPE(diag, BCIR_BPE_LAW_ADDED, r->at);
    const id_slot *slot = id_find(t, r->id);
    if (slot) {
      id_entry *e = &t->entries[slot->entry];
      int same = e->special ? (e->bytes.n == r->content.n && memcmp(e->bytes.s, r->content.s, r->content.n) == 0)
                            : (alphabet && e->bytes.n == bn && memcmp(e->bytes.s, as_bytes, bn) == 0);
      if (!same) return SHAPE(diag, BCIR_BPE_LAW_ADDED, r->at);
      e->bytes = r->content; /* the id renders as the special from now on */
      e->special = 1u;
    } else {
      if (!id_insert(t, r->id, (uint32_t)t->entries_n)) return SHAPE(diag, BCIR_BPE_LAW_ID, r->at);
      t->entries[t->entries_n].bytes = r->content;
      t->entries[t->entries_n].special = 1u;
      t->entries_n++;
    }
    t->term[end] = (int32_t)t->specials_n;
    t->specials[t->specials_n++] = *r;
  }
  return BCIR_BPE_OK;
}

static bcir_bpe_status apply_laws(bcir_bpe *t, const sink_t *s, bcir_bpe_diag *diag) {
  bcir_bpe_status st;
  if (!s->seen_model || !s->type_is_bpe) return SHAPE(diag, BCIR_BPE_LAW_MODEL, BCIR_JER_NO_OFFSET);
  if ((st = vocab_laws(t, &s->vocab, diag, s->added.n)) != BCIR_BPE_OK) return st;
  if ((st = merge_laws(t, &s->alphabet, &s->merges, diag)) != BCIR_BPE_OK) return st;
  if ((st = added_laws(t, &s->alphabet, &s->added, diag)) != BCIR_BPE_OK) return st;
  return finish(diag, BCIR_BPE_OK, BCIR_BPE_LAW_NONE, BCIR_JER_NO_OFFSET);
}

static void sink_release(sink_t *s) {
  bcir_host_deallocate(s->a, s->names.slots);
  bcir_host_deallocate(s->a, s->vocab.items);
  bcir_host_deallocate(s->a, s->merges.items);
  bcir_host_deallocate(s->a, s->added.items);
}

bcir_bpe_status bcir_bpe_load(const uint8_t *json, size_t len, const bcir_host_allocator *allocator, bcir_bpe **out,
                              bcir_bpe_diag *diag) {
  bcir_host_allocator a = bcir_host_allocator_or_default(allocator);
  bcir_jer_limits limits;
  bcir_jer_diag jd;
  bcir_jer_status js;
  bcir_bpe_status st;
  uint64_t nodes = 0;
  if (diag) {
    memset(diag, 0, sizeof *diag);
    diag->offset = BCIR_JER_NO_OFFSET;
    diag->jer.offset = BCIR_JER_NO_OFFSET;
  }
  if (!out) return finish(diag, BCIR_BPE_ERR_INVALID, BCIR_BPE_LAW_NONE, BCIR_JER_NO_OFFSET);
  *out = 0;
  if ((!json && len) || !bcir_host_allocator_valid(&a))
    return finish(diag, BCIR_BPE_ERR_INVALID, BCIR_BPE_LAW_NONE, BCIR_JER_NO_OFFSET);
  bcir_bpe_limits(&limits);
  memset(&jd, 0, sizeof jd);

  /* the bounded reader's scan, then UTF-8: nothing is built until both pass */
  bcir_jer_level *stack = (bcir_jer_level *)bcir_host_allocate(&a, bcir_jer_stack_bytes(&limits));
  if (!stack) return NOMEM(diag);
  js = bcir_jer_scan(json, len, &limits, stack, limits.depth, &nodes, &jd);
  if (js == BCIR_JER_OK) js = bcir_jer_validate_utf8(json, len, &jd);
  if (js != BCIR_JER_OK) {
    bcir_host_deallocate(&a, stack);
    if (diag) diag->jer = jd;
    return finish(diag, BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE, jd.offset);
  }

  bcir_bpe *t = (bcir_bpe *)bcir_host_allocate_zeroed(&a, sizeof *t);
  uint8_t *scratch = (uint8_t *)bcir_host_allocate(&a, (size_t)limits.string_bytes);
  sink_t *s = (sink_t *)bcir_host_allocate_zeroed(&a, sizeof *s);
  if (!t || !scratch || !s || !bcir_host_arena_init(&t->arena, &a, (size_t)1u << 16)) {
    bcir_host_deallocate(&a, t);
    bcir_host_deallocate(&a, scratch);
    bcir_host_deallocate(&a, s);
    bcir_host_deallocate(&a, stack);
    return NOMEM(diag);
  }
  t->allocator = a;
  {
    bcir_sha256 sha;
    bcir_sha256_init(&sha);
    bcir_sha256_update(&sha, json, len);
    bcir_sha256_final(&sha, t->digest);
    t->k0 = le64(t->digest);
    t->k1 = le64(t->digest + 8);
  }
  s->t = t;
  s->a = &t->allocator;
  build_alphabet(&s->alphabet);
  s->vocab.size = sizeof(vocab_rec);
  s->merges.size = sizeof(merge_rec);
  s->added.size = sizeof(added_rec);

  js = bcir_jer_parse(json, len, &limits, stack, limits.depth, scratch, (size_t)limits.string_bytes, bpe_sink, s, &jd);
  bcir_host_deallocate(&a, scratch);
  bcir_host_deallocate(&a, stack);
  if (s->nomem) {
    st = NOMEM(diag);
  } else if (s->law != BCIR_BPE_LAW_NONE) {
    st = SHAPE(diag, s->law, s->law_at);
  } else if (js != BCIR_JER_OK) {
    if (diag) diag->jer = jd;
    st = finish(diag, BCIR_BPE_ERR_JSON, BCIR_BPE_LAW_NONE, jd.offset);
  } else {
    st = apply_laws(t, s, diag);
  }
  sink_release(s);
  bcir_host_deallocate(&a, s);
  if (st != BCIR_BPE_OK) {
    bcir_bpe_free(t);
    return st;
  }
  *out = t;
  return BCIR_BPE_OK;
}

void bcir_bpe_free(bcir_bpe *t) {
  if (!t) return;
  bcir_host_allocator a = t->allocator;
  bcir_host_arena_destroy(&t->arena);
  bcir_host_deallocate(&a, t);
}

void bcir_bpe_digest(const bcir_bpe *t, uint8_t sha256[32]) {
  if (t && sha256) memcpy(sha256, t->digest, 32);
}
size_t bcir_bpe_vocab_size(const bcir_bpe *t) { return t ? t->vocab_n : 0u; }
size_t bcir_bpe_merge_count(const bcir_bpe *t) { return t ? t->merges_n : 0u; }
size_t bcir_bpe_special_count(const bcir_bpe *t) { return t ? t->specials_n : 0u; }

/* --- encode ---------------------------------------------------------------------------------- */

#define NONE UINT32_MAX

/* A candidate merge of the symbols starting at octets `left` and `right`, valid while neither
 * symbol has changed since it was pushed (`vl`, `vr` are their versions then). */
typedef struct { uint32_t rank, left, right, vl, vr; } cand_t;

typedef struct {
  uint32_t *id, *next, *prev, *ver; /* per symbol, indexed by the octet it starts at */
  cand_t *heap;
  size_t cap, heap_cap, heap_n;
} word_scratch;

static int cand_less(const cand_t *x, const cand_t *y) { /* the lowest rank, then the leftmost */
  return x->rank != y->rank ? x->rank < y->rank : x->left < y->left;
}

static void heap_push(word_scratch *w, cand_t e) {
  size_t i = w->heap_n++;
  while (i > 0u) {
    size_t p = (i - 1u) / 2u;
    if (!cand_less(&e, &w->heap[p])) break;
    w->heap[i] = w->heap[p];
    i = p;
  }
  w->heap[i] = e;
}

static cand_t heap_pop(word_scratch *w) {
  cand_t top = w->heap[0], last = w->heap[--w->heap_n];
  size_t i = 0, n = w->heap_n;
  if (!n) return top;
  for (;;) {
    size_t c = 2u * i + 1u;
    if (c >= n) break;
    if (c + 1u < n && cand_less(&w->heap[c + 1u], &w->heap[c])) c++;
    if (!cand_less(&w->heap[c], &last)) break;
    w->heap[i] = w->heap[c];
    i = c;
  }
  w->heap[i] = last;
  return top;
}

static int grow_u32(const bcir_host_allocator *a, uint32_t **p, size_t old, size_t n) {
  void *v = *p;
  if (!bcir_host_realloc_array(a, &v, old, n, sizeof(uint32_t), 0)) return 0;
  *p = (uint32_t *)v;
  return 1;
}

/* Room for a word of `n` octets: n symbols and the 3n - 3 candidates its merges can push. A
 * failure leaves every array at least as large as the recorded capacity. */
static int scratch_reserve(const bcir_host_allocator *a, word_scratch *w, size_t n) {
  size_t heap;
  void *v;
  if (n <= w->cap) return 1;
  if (!bcir_size_mul(n, 3u, &heap)) return 0;
  if (!grow_u32(a, &w->id, w->cap, n) || !grow_u32(a, &w->next, w->cap, n) || !grow_u32(a, &w->prev, w->cap, n) ||
      !grow_u32(a, &w->ver, w->cap, n))
    return 0;
  v = w->heap;
  if (!bcir_host_realloc_array(a, &v, w->heap_cap, heap, sizeof(cand_t), 0)) return 0;
  w->heap = (cand_t *)v;
  w->cap = n;
  w->heap_cap = heap;
  return 1;
}

static void scratch_release(const bcir_host_allocator *a, word_scratch *w) {
  bcir_host_deallocate(a, w->id);
  bcir_host_deallocate(a, w->next);
  bcir_host_deallocate(a, w->prev);
  bcir_host_deallocate(a, w->ver);
  bcir_host_deallocate(a, w->heap);
}

typedef struct { uint32_t *ids; size_t cap, n; } id_out;

static void emit(id_out *o, uint32_t id) {
  if (o->n < o->cap) o->ids[o->n] = id;
  o->n++;
}

static void push_candidate(const bcir_bpe *t, word_scratch *w, uint32_t left) {
  uint32_t right = w->next[left];
  const merge_slot *m;
  cand_t e;
  if (right == NONE || !(m = merge_find(t, w->id[left], w->id[right]))) return;
  e.rank = m->rank;
  e.left = left;
  e.right = right;
  e.vl = w->ver[left];
  e.vr = w->ver[right];
  heap_push(w, e);
}

/* One word: its bytes merged by the lowest rank, leftmost first -- the oracle's choice at every
 * step, made through a heap of candidates instead of a rescan of the word. */
static int encode_word(const bcir_bpe *t, word_scratch *w, const uint8_t *word, size_t n, id_out *o) {
  if (n == 1u) {
    emit(o, t->byte_id[word[0]]);
    return 1;
  }
  if (!scratch_reserve(&t->allocator, w, n)) return 0;
  for (size_t k = 0; k < n; k++) {
    w->id[k] = t->byte_id[word[k]];
    w->prev[k] = k ? (uint32_t)(k - 1u) : NONE;
    w->next[k] = k + 1u < n ? (uint32_t)(k + 1u) : NONE;
    w->ver[k] = 0;
  }
  w->heap_n = 0;
  for (size_t k = 0; k + 1u < n; k++) push_candidate(t, w, (uint32_t)k);
  while (w->heap_n) {
    cand_t e = heap_pop(w);
    if (w->ver[e.left] != e.vl || w->ver[e.right] != e.vr) continue; /* a symbol changed: stale */
    const merge_slot *m = merge_find(t, w->id[e.left], w->id[e.right]);
    uint32_t after = w->next[e.right];
    w->id[e.left] = m->merged;
    w->next[e.left] = after;
    if (after != NONE) w->prev[after] = e.left;
    w->ver[e.left]++;
    w->ver[e.right]++; /* the right symbol is gone: every candidate naming it is stale */
    if (w->prev[e.left] != NONE) push_candidate(t, w, w->prev[e.left]);
    push_candidate(t, w, e.left);
  }
  for (uint32_t k = 0; k != NONE; k = w->next[k]) emit(o, w->id[k]); /* symbol 0 is never absorbed */
  return 1;
}

/* The reference pre-tokenizer over octets: words keep their leading space, and a run of spaces
 * before a word gives all but its last space a word of their own. A multi-octet character never
 * contains 0x20, so splitting octets is splitting characters. */
static int encode_span(const bcir_bpe *t, word_scratch *w, const uint8_t *s, size_t n, id_out *o) {
  size_t start = 0;
  for (size_t i = 0; i < n; i++) {
    if (s[i] == ' ') {
      if (i > start && s[i - 1u] != ' ') { /* the word before this space ends */
        if (!encode_word(t, w, s + start, i - start, o)) return 0;
        start = i;
      }
    } else if (i - start > 1u && s[i - 1u] == ' ') { /* spaces before a word: all but the last */
      if (!encode_word(t, w, s + start, i - 1u - start, o)) return 0;
      start = i - 1u;
    }
  }
  return n > start ? encode_word(t, w, s + start, n - start, o) : 1;
}

bcir_bpe_status bcir_bpe_encode(const bcir_bpe *t, const uint8_t *text, size_t len, uint32_t *ids, size_t cap,
                                size_t *count) {
  bcir_jer_diag jd;
  word_scratch w;
  id_out o;
  bcir_bpe_status st = BCIR_BPE_OK;
  size_t i = 0;
  if (count) *count = 0;
  if (!t || !count || (!text && len) || (!ids && cap) || (uint64_t)len >= (uint64_t)NONE) return BCIR_BPE_ERR_INVALID;
  if (bcir_jer_validate_utf8(text, len, &jd) != BCIR_JER_OK) return BCIR_BPE_ERR_NOT_UTF8;
  memset(&w, 0, sizeof w);
  o.ids = ids;
  o.cap = cap;
  o.n = 0;
  while (i < len && st == BCIR_BPE_OK) {
    int32_t sp = special_at(t, text, len, i);
    if (sp >= 0) { /* a special is never split */
      emit(&o, t->specials[sp].id);
      i += t->specials[sp].content.n;
      continue;
    }
    size_t j = i + 1u; /* the run up to the next special; none starts inside a character */
    while (j < len && special_at(t, text, len, j) < 0) j++;
    if (!encode_span(t, &w, text + i, j - i, &o)) st = BCIR_BPE_ERR_NOMEM;
    i = j;
  }
  scratch_release(&t->allocator, &w);
  if (st != BCIR_BPE_OK) return st;
  *count = o.n;
  return o.n > cap ? BCIR_BPE_ERR_NOSPACE : BCIR_BPE_OK;
}

/* --- decode ---------------------------------------------------------------------------------- */

/* The oracle's strict UTF-8, fed octet by octet so a run is judged as its tokens arrive: no
 * overlong form, no surrogate, nothing past U+10FFFF. */
typedef struct { unsigned need; uint8_t lo, hi; int bad; } utf8_state;

static void utf8_feed(utf8_state *u, uint8_t b) {
  if (u->bad) return;
  if (!u->need) {
    if (b < 0x80u) return;
    u->lo = 0x80u;
    u->hi = 0xBFu;
    if (b >= 0xC2u && b <= 0xDFu) {
      u->need = 1;
    } else if (b >= 0xE0u && b <= 0xEFu) {
      u->need = 2;
      if (b == 0xE0u) u->lo = 0xA0u; /* no overlong three-octet form */
      if (b == 0xEDu) u->hi = 0x9Fu; /* no surrogate */
    } else if (b >= 0xF0u && b <= 0xF4u) {
      u->need = 3;
      if (b == 0xF0u) u->lo = 0x90u; /* no overlong four-octet form */
      if (b == 0xF4u) u->hi = 0x8Fu; /* nothing past U+10FFFF */
    } else {
      u->bad = 1;
    }
    return;
  }
  if (b < u->lo || b > u->hi) {
    u->bad = 1;
    return;
  }
  u->lo = 0x80u;
  u->hi = 0xBFu;
  u->need--;
}

/* 1 when the run so far is complete UTF-8; the state is reset for the next run. */
static int utf8_flush(utf8_state *u) {
  int ok = !u->bad && !u->need;
  memset(u, 0, sizeof *u);
  return ok;
}

bcir_bpe_status bcir_bpe_decode(const bcir_bpe *t, const uint32_t *ids, size_t n, uint8_t *out, size_t cap,
                                size_t *written) {
  utf8_state u;
  size_t at = 0;
  if (written) *written = 0;
  if (!t || !written || (!ids && n) || (!out && cap)) return BCIR_BPE_ERR_INVALID;
  memset(&u, 0, sizeof u);
  for (size_t i = 0; i < n; i++) {
    const id_slot *slot = id_find(t, ids[i]);
    if (!slot) return BCIR_BPE_ERR_UNKNOWN_ID;
    const id_entry *e = &t->entries[slot->entry];
    if (e->special) {
      if (!utf8_flush(&u)) return BCIR_BPE_ERR_NOT_UTF8;
    } else {
      for (uint32_t k = 0; k < e->bytes.n; k++) utf8_feed(&u, e->bytes.s[k]);
    }
    for (uint32_t k = 0; k < e->bytes.n; k++, at++)
      if (at < cap) out[at] = e->bytes.s[k];
  }
  if (!utf8_flush(&u)) return BCIR_BPE_ERR_NOT_UTF8;
  *written = at;
  return at > cap ? BCIR_BPE_ERR_NOSPACE : BCIR_BPE_OK;
}
