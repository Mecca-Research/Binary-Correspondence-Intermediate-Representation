/*===- bcir_emit.c - freestanding plan-driven ASN.1 encoder (E2) -----------===
 *
 * See bcir_emit.h for the contract. The clause governing each spelling is named at the site:
 * a plan-driven encoder is only worth having if a reader can check it against the standard
 * without holding the Python oracle in their head at the same time.
 *
 * THE SHAPE THAT MAKES IMPLICIT TAGS WORK. Every X.690 emitter here is split in two:
 * `*_content` writes a node's CONTENT octets and `*_full` writes identifier + length +
 * content. X.690 8.14.4 says an implicit tag REPLACES the base tag, so a member with one
 * writes its own header and then the inner node's *content* -- there is no inner header to
 * strip and no bytes to shift. 8.14.3's explicit tag wraps instead, and calls `*_full`.
 * Getting this wrong is the single easiest way to produce a decoder-plausible document of
 * the wrong value.
 *===----------------------------------------------------------------------===*/
#include "bcir_emit.h"

/* --- small helpers, no libc -------------------------------------------------------------- */

static int emit_streq(const char *a, const char *b, size_t len) {
  size_t i;
  for (i = 0; i < len; i++)
    if (a[i] != b[i]) return 0;
  return 1;
}

static size_t emit_cstr_len(const char *s) {
  size_t n = 0;
  while (s[n] != '\0') n++;
  return n;
}

/* --- the descriptor reader ---------------------------------------------------------------- */

typedef struct plan_reader {
  const char *text;
  size_t len;
  size_t at;
} plan_reader;

static void plan_skip_spaces(plan_reader *r) {
  while (r->at < r->len && r->text[r->at] == ' ') r->at++;
}

static void plan_next_line(plan_reader *r) {
  while (r->at < r->len && r->text[r->at] != '\n') r->at++;
  if (r->at < r->len) r->at++;
}

static size_t plan_token(plan_reader *r, size_t *start) {
  plan_skip_spaces(r);
  *start = r->at;
  while (r->at < r->len && r->text[r->at] != ' ' && r->text[r->at] != '\n') r->at++;
  return r->at - *start;
}

/* `key=value`: checks the key and returns the value's span. */
static int plan_field(plan_reader *r, const char *key, size_t *start, size_t *len) {
  size_t tok_start, tok_len, key_len = emit_cstr_len(key);
  tok_len = plan_token(r, &tok_start);
  if (tok_len < key_len + 1) return 0;
  if (!emit_streq(r->text + tok_start, key, key_len)) return 0;
  if (r->text[tok_start + key_len] != '=') return 0;
  *start = tok_start + key_len + 1;
  *len = tok_len - key_len - 1;
  return 1;
}

static int plan_uint(const char *text, size_t len, uint32_t *out) {
  uint32_t value = 0;
  size_t i;
  if (len == 0) return 0;
  for (i = 0; i < len; i++) {
    if (text[i] < '0' || text[i] > '9') return 0;
    if (value > (0xFFFFFFFFu - (uint32_t)(text[i] - '0')) / 10u) return 0;
    value = value * 10u + (uint32_t)(text[i] - '0');
  }
  *out = value;
  return 1;
}

static int plan_kind(const char *text, size_t len, uint8_t *out) {
  static const struct {
    const char *name;
    uint8_t kind;
  } table[] = {{"boolean", BCIR_EMIT_BOOLEAN},         {"integer", BCIR_EMIT_INTEGER},
               {"enumerated", BCIR_EMIT_ENUMERATED},   {"null", BCIR_EMIT_NULL},
               {"octetstring", BCIR_EMIT_OCTETSTRING}, {"string", BCIR_EMIT_STRING},
               {"oid", BCIR_EMIT_OID},                 {"sequence", BCIR_EMIT_SEQUENCE},
               {"sequence-of", BCIR_EMIT_SEQUENCE_OF}, {"choice", BCIR_EMIT_CHOICE}};
  size_t i;
  for (i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
    size_t n = emit_cstr_len(table[i].name);
    if (n == len && emit_streq(text, table[i].name, n)) {
      *out = table[i].kind;
      return 1;
    }
  }
  return 0;
}

static int plan_tag_class(const char *text, size_t len, uint8_t *out) {
  if (len == 9 && emit_streq(text, "universal", 9)) { *out = 0x00; return 1; }
  if (len == 11 && emit_streq(text, "application", 11)) { *out = 0x40; return 1; }
  if (len == 7 && emit_streq(text, "context", 7)) { *out = 0x80; return 1; }
  if (len == 7 && emit_streq(text, "private", 7)) { *out = 0xC0; return 1; }
  return 0;
}

/* One bound: `-` for unbounded, else optional `-` sign and decimal magnitude. The magnitude
 * is unsigned 64-bit because X.696 10.3 d)'s widest fixed word is eight octets UNSIGNED --
 * see bcir_emit_bound. Overflow is a REFUSAL, never a wrap: a truncated bound names a
 * different type, and would pick a different OER field width. */
static int plan_bound(const char *text, size_t len, bcir_emit_bound *out) {
  uint64_t value = 0;
  size_t i = 0;
  out->present = 0;
  out->negative = 0;
  out->magnitude = 0;
  if (len == 0) return 0;
  if (len == 1 && text[0] == '-') return 1;         /* unbounded */
  if (text[0] == '-') {
    out->negative = 1;
    i = 1;
    if (len == 1) return 0;
  }
  for (; i < len; i++) {
    if (text[i] < '0' || text[i] > '9') return 0;
    if (value > (0xFFFFFFFFFFFFFFFFull - (uint64_t)(text[i] - '0')) / 10ull) return 0;
    value = value * 10ull + (uint64_t)(text[i] - '0');
  }
  out->present = 1;
  out->magnitude = value;
  if (value == 0) out->negative = 0;                /* one spelling for zero */
  return 1;
}

static int plan_hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;                                        /* the writer emits lower case only */
}

typedef struct plan_build {
  plan_reader r;
  bcir_emit_tables t;
  uint32_t node_count;
  uint32_t member_count;
  uint32_t constraint_count;
  uint32_t enum_count;
  bcir_emit_diag *diag;
} plan_build;

static bcir_emit_status plan_fail(plan_build *b, bcir_emit_status status) {
  if (b->diag) {
    b->diag->status = status;
    b->diag->offset = b->r.at;
  }
  return status;
}

/* The optional `constraint` line, which follows its node's line when there is one. Peeked
 * rather than counted: a node line's `members`/`element` counts exist so the reader needs no
 * path arithmetic, and a count of "constraints on this node" could only ever be 0 or 1. */
static bcir_emit_status plan_read_constraint(plan_build *b, uint32_t self) {
  static const char *const bound_keys[8] = {"vlo", "vhi", "slo", "shi",
                                            "rvlo", "rvhi", "rslo", "rshi"};
  size_t save = b->r.at, start = 0, len = 0, i;
  bcir_emit_constraint *k;
  bcir_emit_bound *slots[8];

  len = plan_token(&b->r, &start);
  if (len != 10 || !emit_streq(b->r.text + start, "constraint", 10)) {
    b->r.at = save;                                 /* not ours; leave the line alone */
    return BCIR_EMIT_OK;
  }
  if (b->constraint_count >= b->t.constraint_cap) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
  k = &b->t.constraints[b->constraint_count];
  slots[0] = &k->value_low;      slots[1] = &k->value_high;
  slots[2] = &k->size_low;       slots[3] = &k->size_high;
  slots[4] = &k->root_value_low; slots[5] = &k->root_value_high;
  slots[6] = &k->root_size_low;  slots[7] = &k->root_size_high;

  (void)plan_token(&b->r, &start);                  /* the path, as on a node line */
  for (i = 0; i < 8; i++) {
    if (!plan_field(&b->r, bound_keys[i], &start, &len) ||
        !plan_bound(b->r.text + start, len, slots[i]))
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  }
  if (!plan_field(&b->r, "vext", &start, &len) || len != 1)
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  k->value_extensible = (uint8_t)(b->r.text[start] == '1');
  if (!plan_field(&b->r, "sext", &start, &len) || len != 1)
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  k->size_extensible = (uint8_t)(b->r.text[start] == '1');
  if (!plan_field(&b->r, "alpha", &start, &len)) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  k->alphabet_len = 0;
  if (!(len == 1 && b->r.text[start] == '-')) {
    if ((len & 1u) != 0) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (len / 2 > BCIR_EMIT_ALPHABET_MAX) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
    for (i = 0; i < len; i += 2) {
      int hi = plan_hex_nibble(b->r.text[start + i]);
      int lo = plan_hex_nibble(b->r.text[start + i + 1]);
      if (hi < 0 || lo < 0) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
      k->alphabet[i / 2] = (char)((hi << 4) | lo);
    }
    k->alphabet_len = (uint8_t)(len / 2);
  }
  plan_next_line(&b->r);
  b->t.nodes[self].constraint = (int32_t)b->constraint_count++;
  return BCIR_EMIT_OK;
}

/* The node line's `enum=` field: `-`, or `name:number|name:number|...`. Parsed into the
 * caller's enum table, whose slots are contiguous per node so a node needs only a first
 * index and a count. */
static bcir_emit_status plan_read_enum(plan_build *b, uint32_t self, const char *text,
                                       size_t len) {
  size_t at = 0;

  b->t.nodes[self].first_enum = b->enum_count;
  b->t.nodes[self].enum_count = 0;
  if (len == 1 && text[0] == '-') return BCIR_EMIT_OK;
  while (at < len) {
    size_t name_start = at, name_len, number_len, i;
    bcir_emit_enum_item *item;
    bcir_emit_bound number;

    while (at < len && text[at] != ':') at++;
    if (at == len) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    name_len = at - name_start;
    if (name_len == 0) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (name_len > BCIR_EMIT_NAME_MAX) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
    at++;                                             /* the ':' */
    number_len = at;
    while (at < len && text[at] != '|') at++;
    number_len = at - number_len;
    /* A bound rather than a plain integer: an enumeration number is signed, and X.680
     * 20.1 puts no ceiling on it. `plan_bound` refuses overflow instead of wrapping. */
    if (!plan_bound(text + at - number_len, number_len, &number) || !number.present)
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (number.magnitude > 0x7FFFFFFFFFFFFFFFull) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
    if (b->enum_count >= b->t.enum_cap) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
    item = &b->t.enums[b->enum_count++];
    for (i = 0; i < name_len; i++) item->name[i] = text[name_start + i];
    item->name_len = (uint8_t)name_len;
    item->number = number.negative ? -(int64_t)number.magnitude : (int64_t)number.magnitude;
    b->t.nodes[self].enum_count++;
    if (at < len) at++;                               /* the '|' */
  }
  return BCIR_EMIT_OK;
}

static bcir_emit_status plan_read_node(plan_build *b, uint32_t *out_index, uint32_t depth) {
  size_t start = 0, len = 0, enum_start = 0, enum_len = 0;
  uint32_t self, members = 0, element = 0, universal = 0, i;
  uint8_t kind = 0, extensible = 0;

  if (depth > BCIR_EMIT_MAX_PLAN_DEPTH) return plan_fail(b, BCIR_EMIT_TOO_DEEP);
  len = plan_token(&b->r, &start);
  if (len != 4 || !emit_streq(b->r.text + start, "node", 4))
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  (void)plan_token(&b->r, &start); /* the path, which the counts make unnecessary */
  if (!plan_field(&b->r, "kind", &start, &len) || !plan_kind(b->r.text + start, len, &kind))
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  if (!plan_field(&b->r, "universal", &start, &len) ||
      !plan_uint(b->r.text + start, len, &universal))
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  if (!plan_field(&b->r, "members", &start, &len) ||
      !plan_uint(b->r.text + start, len, &members))
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  if (!plan_field(&b->r, "element", &start, &len) ||
      !plan_uint(b->r.text + start, len, &element))
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  (void)plan_field(&b->r, "type", &start, &len);  /* identity, not structure */
  if (!plan_field(&b->r, "enum", &enum_start, &enum_len))
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  if (!plan_field(&b->r, "ext", &start, &len) || len != 1)
    return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
  extensible = (uint8_t)(b->r.text[start] == '1');
  plan_next_line(&b->r);

  if (b->node_count >= b->t.node_cap) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
  self = b->node_count++;
  b->t.nodes[self].kind = kind;
  b->t.nodes[self].extensible = extensible;
  b->t.nodes[self].universal = universal;
  b->t.nodes[self].member_count = members;
  b->t.nodes[self].first_member = b->member_count;
  b->t.nodes[self].element = -1;
  b->t.nodes[self].constraint = -1;

  {
    /* `enum_start` points into the descriptor text, which outlives the parse. */
    bcir_emit_status status = plan_read_enum(b, self, b->r.text + enum_start, enum_len);
    if (status != BCIR_EMIT_OK) return status;
    status = plan_read_constraint(b, self);
    if (status != BCIR_EMIT_OK) return status;
  }

  if (members > 0) {
    if (members > b->t.member_cap - b->member_count) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
    b->member_count += members;
  }

  for (i = 0; i < members; i++) {
    uint32_t slot = b->t.nodes[self].first_member + i, value = 0, child = 0;
    bcir_emit_member *m = &b->t.members[slot];
    bcir_emit_status status;
    size_t n;

    len = plan_token(&b->r, &start);
    if (len != 6 || !emit_streq(b->r.text + start, "member", 6))
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    (void)plan_token(&b->r, &start); /* path */
    (void)plan_token(&b->r, &start); /* index, positional here */
    if (!plan_field(&b->r, "name", &start, &len)) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (len > BCIR_EMIT_NAME_MAX) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
    for (n = 0; n < len; n++) m->name[n] = b->r.text[start + n];
    m->name_len = (uint8_t)len;
    if (!plan_field(&b->r, "id", &start, &len)) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (!plan_field(&b->r, "opt", &start, &len) || len != 1)
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    m->optional = (uint8_t)(b->r.text[start] == '1');
    if (!plan_field(&b->r, "def", &start, &len) || len != 1)
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    m->has_default = (uint8_t)(b->r.text[start] == '1');
    if (!plan_field(&b->r, "tag", &start, &len)) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (len == 1 && b->r.text[start] == '-') {
      m->tag = -1;
    } else if (plan_uint(b->r.text + start, len, &value) && value <= 0x7FFFFFFFu) {
      m->tag = (int32_t)value;
    } else {
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    }
    if (!plan_field(&b->r, "class", &start, &len) ||
        !plan_tag_class(b->r.text + start, len, &m->tag_class))
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    if (!plan_field(&b->r, "exp", &start, &len) || len != 1)
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    m->explicit_tag = (uint8_t)(b->r.text[start] == '1');
    if (!plan_field(&b->r, "dval", &start, &len))
      return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
    m->default_len = 0;
    if (!(len == 1 && b->r.text[start] == '-')) {
      if ((len & 1u) != 0) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
      if (len / 2 > BCIR_EMIT_DEFAULT_MAX) return plan_fail(b, BCIR_EMIT_PLAN_TOO_BIG);
      for (n = 0; n < len; n += 2) {
        int hi = plan_hex_nibble(b->r.text[start + n]);
        int lo = plan_hex_nibble(b->r.text[start + n + 1]);
        if (hi < 0 || lo < 0) return plan_fail(b, BCIR_EMIT_PLAN_MALFORMED);
        m->default_stream[n / 2] = (uint8_t)((hi << 4) | lo);
      }
      m->default_len = (uint8_t)(len / 2);
    }
    plan_next_line(&b->r);

    status = plan_read_node(b, &child, depth + 1);
    if (status != BCIR_EMIT_OK) return status;
    b->t.members[slot].node = child;
  }

  if (element) {
    uint32_t child = 0;
    bcir_emit_status status = plan_read_node(b, &child, depth + 1);
    if (status != BCIR_EMIT_OK) return status;
    b->t.nodes[self].element = (int32_t)child;
  }

  *out_index = self;
  return BCIR_EMIT_OK;
}

bcir_emit_status bcir_emit_parse_plan(const char *text, size_t len,
                                      const bcir_emit_tables *tables, bcir_emit_plan *out,
                                      bcir_emit_diag *diag) {
  plan_build b;
  size_t start = 0, tok;
  uint32_t version = 0, root = 0;
  bcir_emit_status status;
  int i;

  if (diag) { diag->status = BCIR_EMIT_OK; diag->offset = 0; diag->needed = 0; }
  if (!text || !tables || !out) return BCIR_EMIT_PLAN_MALFORMED;
  if (!tables->nodes || !tables->members) return BCIR_EMIT_PLAN_MALFORMED;
  /* A null table with a nonzero capacity is a caller error that would otherwise surface
   * only on the first schema needing it, so it is refused up front rather than trusted. */
  if (!tables->constraints && tables->constraint_cap != 0) return BCIR_EMIT_PLAN_MALFORMED;
  if (!tables->enums && tables->enum_cap != 0) return BCIR_EMIT_PLAN_MALFORMED;
  b.r.text = text; b.r.len = len; b.r.at = 0;
  b.t = *tables;
  b.node_count = 0;
  b.member_count = 0;
  b.constraint_count = 0;
  b.enum_count = 0;
  b.diag = diag;

  tok = plan_token(&b.r, &start);
  if (tok != 12 || !emit_streq(text + start, "plan-version", 12))
    return plan_fail(&b, BCIR_EMIT_PLAN_MALFORMED);
  tok = plan_token(&b.r, &start);
  if (!plan_uint(text + start, tok, &version)) return plan_fail(&b, BCIR_EMIT_PLAN_MALFORMED);
  if (version != BCIR_EMIT_PLAN_VERSION) return plan_fail(&b, BCIR_EMIT_PLAN_VERSION_BAD);
  plan_next_line(&b.r);
  /* compiler, module, type, source-sha256: identity, not structure. */
  for (i = 0; i < 4; i++) plan_next_line(&b.r);

  status = plan_read_node(&b, &root, 0);
  if (status != BCIR_EMIT_OK) return status;
  out->nodes = b.t.nodes;
  out->node_count = b.node_count;
  out->members = b.t.members;
  out->member_count = b.member_count;
  out->constraints = b.t.constraints;
  out->constraint_count = b.constraint_count;
  out->enums = b.t.enums;
  out->enum_count = b.enum_count;
  out->root = root;
  return BCIR_EMIT_OK;
}

/* --- the emitter context ------------------------------------------------------------------ */

typedef struct emit_ctx {
  const bcir_emit_plan *plan;
  const uint8_t *stream;
  size_t stream_len;
  size_t at;
  uint8_t *out;
  size_t out_cap;
  size_t written;
  uint32_t *scratch;
  size_t scratch_cap;
  size_t visits;
  uint32_t max_depth;
  bcir_emit_status status;
  size_t fail_offset;
} emit_ctx;

static int ctx_fail(emit_ctx *c, bcir_emit_status status) {
  if (c->status == BCIR_EMIT_OK) {
    c->status = status;
    c->fail_offset = c->at;
  }
  return 0;
}

static int rd_u8(emit_ctx *c, uint8_t *out) {
  if (c->stream_len - c->at < 1) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
  *out = c->stream[c->at++];
  return 1;
}

static int rd_u32(emit_ctx *c, uint32_t *out) {
  if (c->stream_len - c->at < 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
  *out = ((uint32_t)c->stream[c->at] << 24) | ((uint32_t)c->stream[c->at + 1] << 16) |
         ((uint32_t)c->stream[c->at + 2] << 8) | (uint32_t)c->stream[c->at + 3];
  c->at += 4;
  return 1;
}

static int rd_take(emit_ctx *c, size_t count, const uint8_t **out) {
  if (count > c->stream_len - c->at) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
  *out = c->stream + c->at;
  c->at += count;
  return 1;
}

/* Always counts; writes only where there is room. A short buffer therefore yields a
 * complete, accurate `written` rather than a partial write the caller has to unpick. */
static void put(emit_ctx *c, uint8_t byte) {
  if (c->out && c->written < c->out_cap) c->out[c->written] = byte;
  c->written++;
}

static void put_bytes(emit_ctx *c, const uint8_t *data, size_t len) {
  size_t i;
  for (i = 0; i < len; i++) put(c, data[i]);
}


/* --- the DEFAULT omission rule, which is per-candidate ------------------------------------- */

static int emit_memeq(const uint8_t *a, const uint8_t *b, size_t len) {
  size_t i;
  for (i = 0; i < len; i++)
    if (a[i] != b[i]) return 0;
  return 1;
}

/* Advance past exactly one node's value, emitting nothing -- the mirror of the flattener.
 *
 * `limit` is an absolute stream offset past which it stops early. This is only ever called
 * to compare against a DEFAULT of at most BCIR_EMIT_DEFAULT_MAX octets, so a value that has
 * already outrun the default cannot match it and there is nothing to gain by walking the
 * rest. That also bounds a SEQUENCE OF whose elements consume NO stream octets -- the same
 * unbounded-count shape the fuzzer found once already. */
static int skip_node(emit_ctx *c, uint32_t node_index, uint32_t depth, size_t limit) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  uint8_t byte = 0;
  uint32_t count = 0, i;
  const uint8_t *data = NULL;

  if (depth > c->max_depth) return ctx_fail(c, BCIR_EMIT_TOO_DEEP);
  if (c->at >= limit) return 1;
  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN:
      return rd_u8(c, &byte);
    case BCIR_EMIT_INTEGER:
    case BCIR_EMIT_ENUMERATED:
      return rd_u8(c, &byte) && rd_take(c, byte, &data);
    case BCIR_EMIT_NULL:
      return 1;
    case BCIR_EMIT_OCTETSTRING:
    case BCIR_EMIT_STRING:
    case BCIR_EMIT_OID:
      return rd_u32(c, &count) && rd_take(c, count, &data);
    case BCIR_EMIT_SEQUENCE:
      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        if (m->optional || m->has_default) {
          if (!rd_u8(c, &byte)) return 0;
          if (!byte) continue;
        }
        if (!skip_node(c, m->node, depth + 1, limit)) return 0;
        if (c->at >= limit) return 1;
      }
      return 1;
    case BCIR_EMIT_SEQUENCE_OF:
      if (!rd_u32(c, &count)) return 0;
      if (node->element < 0) return ctx_fail(c, BCIR_EMIT_PLAN_MALFORMED);
      for (i = 0; i < count; i++) {
        if (!skip_node(c, (uint32_t)node->element, depth + 1, limit)) return 0;
        if (c->at >= limit) return 1;
      }
      return 1;
    case BCIR_EMIT_CHOICE:
      if (!rd_u32(c, &count)) return 0;
      if (count >= node->member_count) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      return skip_node(c, c->plan->members[node->first_member + count].node, depth + 1,
                       limit);
    default:
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  }
}

/* Read a component's presence octet and apply the candidate's DEFAULT rule.
 *
 * X.690 11.5 forbids DER an encoding for a component whose value equals its default; X.696
 * 31.9 and X.697's CJER say the same. PLAIN BER DOES NOT -- 11 is titled "Restrictions on
 * BER employed by both CER and DER", so the freedom is BER's, which is why `omit_default` is
 * a parameter rather than a constant. The rule is therefore per-CANDIDATE and cannot live in
 * the format-neutral value stream, which carries only a presence flag.
 *
 * When the rule applies and the value matches, the component's octets are CONSUMED and
 * discarded: the stream still describes it, and leaving it unread would leave a suffix
 * `bcir_emit` correctly refuses as STREAM_LONG. */
static int member_present(emit_ctx *c, const bcir_emit_member *m, int omit_default,
                          uint32_t depth, int *present) {
  size_t start;
  uint8_t byte = 0;

  *present = 1;
  if (!(m->optional || m->has_default)) return 1;
  if (!rd_u8(c, &byte)) return 0;
  if (!byte) { *present = 0; return 1; }
  if (!omit_default || !m->has_default || m->default_len == 0) return 1;
  start = c->at;
  if (!skip_node(c, m->node, depth, start + (size_t)m->default_len + 1)) return 0;
  if (c->at - start == (size_t)m->default_len &&
      emit_memeq(c->stream + start, m->default_stream, m->default_len)) {
    *present = 0;
    return 1;
  }
  c->at = start;
  return 1;
}

/* --- X.690 primitives ---------------------------------------------------------------------- */

static size_t identifier_size(uint32_t number) {
  size_t n = 1;
  uint32_t value = number;
  if (number < 31) return 1;
  do {
    n++;
    value >>= 7;
  } while (value != 0);
  return n;
}

/* 8.1.2, with 8.1.2.4's high-tag-number form above 30. */
static void put_identifier(emit_ctx *c, uint8_t tag_class, int constructed, uint32_t number) {
  uint8_t group[5];
  int n = 0, i;
  uint32_t value = number;
  if (number < 31) {
    put(c, (uint8_t)(tag_class | (constructed ? 0x20 : 0) | number));
    return;
  }
  do {
    group[n++] = (uint8_t)(value & 0x7F);
    value >>= 7;
  } while (value != 0);
  put(c, (uint8_t)(tag_class | (constructed ? 0x20 : 0) | 0x1F));
  for (i = n - 1; i >= 0; i--) put(c, (uint8_t)(group[i] | (i > 0 ? 0x80 : 0)));
}

static size_t definite_length_size(size_t count) {
  size_t n = 1, value = count;
  if (count < 0x80) return 1;
  while (value != 0) {
    n++;
    value >>= 8;
  }
  return n;
}

/* 8.1.3.3-8.1.3.5, restricted by 10.1 to the minimal form. */
static void put_definite_length(emit_ctx *c, size_t count) {
  uint8_t body[8];
  int n = 0, i;
  size_t value = count;
  if (count < 0x80) {
    put(c, (uint8_t)count);
    return;
  }
  while (value != 0) {
    body[n++] = (uint8_t)(value & 0xFF);
    value >>= 8;
  }
  put(c, (uint8_t)(0x80 | (unsigned)n));
  for (i = n - 1; i >= 0; i--) put(c, body[i]);
}

static int node_is_constructed(const bcir_emit_node *node) {
  return node->kind == BCIR_EMIT_SEQUENCE || node->kind == BCIR_EMIT_SEQUENCE_OF;
}

/* 8.19.4: the first two arcs share a subidentifier; the rest are base 128. `measure_only`
 * counts without writing, which is what the DER measure pass needs. */
static int oid_octets(emit_ctx *c, const uint8_t *text, size_t len, int measure_only,
                      size_t *size) {
  size_t i = 0, produced = 0;
  uint32_t arc = 0, first = 0;
  int arcs_seen = 0, have_digit = 0;

  for (;;) {
    if (i < len && text[i] >= '0' && text[i] <= '9') {
      if (arc > (0xFFFFFFFFu - (uint32_t)(text[i] - '0')) / 10u)
        return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      arc = arc * 10u + (uint32_t)(text[i] - '0');
      have_digit = 1;
      i++;
      continue;
    }
    if (!have_digit) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    arcs_seen++;
    if (arcs_seen == 1) {
      first = arc;
    } else {
      uint32_t value = (arcs_seen == 2) ? (40u * first + arc) : arc;
      uint32_t rest = value;
      uint8_t group[5];
      int n = 0, k;
      do {
        group[n++] = (uint8_t)(rest & 0x7F);
        rest >>= 7;
      } while (rest != 0);
      produced += (size_t)n;
      if (!measure_only)
        for (k = n - 1; k >= 0; k--) put(c, (uint8_t)(group[k] | (k > 0 ? 0x80 : 0)));
    }
    arc = 0;
    have_digit = 0;
    if (i >= len) break;
    if (text[i] != '.') return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    i++;
  }
  if (arcs_seen < 2) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED); /* 8.19.3 */
  if (size) *size = produced;
  return 1;
}


/* --- bounding a SEQUENCE OF's element count ------------------------------------------------ */
/*
 * A fuzzer found this: the element count is a 32-bit field in the value stream, and a
 * SEQUENCE OF whose element consumes NO stream octets -- a NULL, or a SEQUENCE of only
 * NULLs -- turns four attacker-chosen bytes into four billion iterations that produce
 * output and read nothing. A few octets in, gigabytes out.
 *
 * The bound comes from the PLAN, which is trusted, rather than from the stream, which is
 * not. `min_stream_octets` is the fewest octets any value of a node can consume; when it is
 * positive the count is checked against what the remaining stream could possibly supply,
 * and when it is zero the count is checked against an explicit ceiling. An element that
 * costs nothing to declare must not cost unboundedly to emit.
 */
#define BCIR_EMIT_MAX_ZERO_COST_ELEMENTS 65536u

static size_t min_stream_octets(const bcir_emit_plan *plan, uint32_t node_index,
                                uint32_t depth) {
  const bcir_emit_node *node = &plan->nodes[node_index];
  size_t total = 0;
  uint32_t i;
  if (depth > BCIR_EMIT_MAX_PLAN_DEPTH) return 1;
  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN: return 1;
    case BCIR_EMIT_INTEGER:
    case BCIR_EMIT_ENUMERATED: return 1;  /* the length octet, at least */
    case BCIR_EMIT_NULL: return 0;        /* nothing at all -- the whole problem */
    case BCIR_EMIT_OCTETSTRING:
    case BCIR_EMIT_STRING:
    case BCIR_EMIT_OID:
    case BCIR_EMIT_SEQUENCE_OF:
    case BCIR_EMIT_CHOICE: return 4;
    case BCIR_EMIT_SEQUENCE:
      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &plan->members[node->first_member + i];
        if (m->optional || m->has_default) total += 1; /* the presence octet */
        else total += min_stream_octets(plan, m->node, depth + 1);
      }
      return total;
    default: return 1;
  }
}

static int bound_elements(emit_ctx *c, uint32_t element_node, uint32_t count) {
  size_t minimum = min_stream_octets(c->plan, element_node, 0);
  if (minimum == 0)
    return count > BCIR_EMIT_MAX_ZERO_COST_ELEMENTS
               ? ctx_fail(c, BCIR_EMIT_UNSUPPORTED)
               : 1;
  return (size_t)count > (c->stream_len - c->at) / minimum
             ? ctx_fail(c, BCIR_EMIT_STREAM_SHORT)
             : 1;
}

/* --- DER: measure, then write ------------------------------------------------------------- */

static size_t der_measure_content(emit_ctx *c, uint32_t node_index, uint32_t depth);
static size_t der_measure_member(emit_ctx *c, const bcir_emit_member *m, uint32_t depth);
static int der_write_content(emit_ctx *c, uint32_t node_index, uint32_t depth);
static int der_write_member(emit_ctx *c, const bcir_emit_member *m, uint32_t depth);

static size_t der_measure_full(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  size_t content;
  /* A CHOICE has no tag of its own (X.680 29.1), so its "full" size IS its content size --
   * there is no header to add. */
  if (node->kind == BCIR_EMIT_CHOICE) return der_measure_content(c, node_index, depth);
  content = der_measure_content(c, node_index, depth);
  if (c->status != BCIR_EMIT_OK) return 0;
  return identifier_size(node->universal) + definite_length_size(content) + content;
}

static size_t der_measure_content(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  size_t slot, content = 0;
  uint8_t byte = 0;
  uint32_t count = 0, i;
  const uint8_t *data = NULL;

  if (depth > c->max_depth) { ctx_fail(c, BCIR_EMIT_TOO_DEEP); return 0; }
  /* The slot is claimed even when the scratch is short, so `needed` ends up being the total
   * number of visits rather than the point where counting stopped. */
  slot = c->visits++;
  if (slot >= c->scratch_cap) ctx_fail(c, BCIR_EMIT_SCRATCH_SHORT);

  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN:
      if (!rd_u8(c, &byte)) return 0;
      content = 1;
      break;
    case BCIR_EMIT_INTEGER:
    case BCIR_EMIT_ENUMERATED:
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      content = byte;
      break;
    case BCIR_EMIT_NULL:
      content = 0;
      break;
    case BCIR_EMIT_OCTETSTRING:
    case BCIR_EMIT_STRING:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      content = count;
      break;
    case BCIR_EMIT_OID: {
      size_t size = 0;
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      if (!oid_octets(c, data, count, 1, &size)) return 0;
      content = size;
      break;
    }
    case BCIR_EMIT_SEQUENCE:
      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        int present = 1;
        if (!member_present(c, m, 1, depth + 1, &present)) return 0;
        if (!present) continue;
        content += der_measure_member(c, m, depth + 1);
        if (c->status != BCIR_EMIT_OK && c->status != BCIR_EMIT_SCRATCH_SHORT) return 0;
      }
      break;
    case BCIR_EMIT_SEQUENCE_OF:
      if (!rd_u32(c, &count)) return 0;
      if (node->element < 0) { ctx_fail(c, BCIR_EMIT_PLAN_MALFORMED); return 0; }
      if (!bound_elements(c, (uint32_t)node->element, count)) return 0;
      for (i = 0; i < count; i++) {
        content += der_measure_full(c, (uint32_t)node->element, depth + 1);
        if (c->status != BCIR_EMIT_OK && c->status != BCIR_EMIT_SCRATCH_SHORT) return 0;
      }
      break;
    case BCIR_EMIT_CHOICE: {
      const bcir_emit_member *m;
      if (!rd_u32(c, &count)) return 0;
      if (count >= node->member_count) { ctx_fail(c, BCIR_EMIT_UNSUPPORTED); return 0; }
      m = &c->plan->members[node->first_member + count];
      content = der_measure_member(c, m, depth + 1);
      break;
    }
    default:
      ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      return 0;
  }
  if (slot < c->scratch_cap) c->scratch[slot] = (uint32_t)content;
  return content;
}

static size_t der_measure_member(emit_ctx *c, const bcir_emit_member *m, uint32_t depth) {
  const bcir_emit_node *inner = &c->plan->nodes[m->node];
  size_t inner_content;
  if (m->tag < 0) return der_measure_full(c, m->node, depth);
  /* Both tagged forms need the inner CONTENT length, and the inner node's own slot is the
   * one `der_measure_content` is about to claim -- so no slot is consumed here, and the
   * write pass peeks the same slot before descending. */
  inner_content = der_measure_content(c, m->node, depth);
  if (c->status != BCIR_EMIT_OK && c->status != BCIR_EMIT_SCRATCH_SHORT) return 0;
  if (m->explicit_tag) {
    /* 8.14.3: the tag WRAPS the base encoding, so the inner header is inside. */
    size_t inner_total = (inner->kind == BCIR_EMIT_CHOICE)
                             ? inner_content
                             : identifier_size(inner->universal) +
                                   definite_length_size(inner_content) + inner_content;
    return identifier_size((uint32_t)m->tag) + definite_length_size(inner_total) + inner_total;
  }
  /* 8.14.4: the tag REPLACES the base tag, so the content is carried unchanged. */
  return identifier_size((uint32_t)m->tag) + definite_length_size(inner_content) +
         inner_content;
}

static int der_write_full(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  size_t content;
  if (node->kind == BCIR_EMIT_CHOICE) return der_write_content(c, node_index, depth);
  if (c->visits >= c->scratch_cap) return ctx_fail(c, BCIR_EMIT_SCRATCH_SHORT);
  content = c->scratch[c->visits]; /* peek the slot der_write_content is about to consume */
  put_identifier(c, 0, node_is_constructed(node), node->universal);
  put_definite_length(c, content);
  return der_write_content(c, node_index, depth);
}

static int der_write_content(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  uint8_t byte = 0;
  uint32_t count = 0, i;
  const uint8_t *data = NULL;

  if (depth > c->max_depth) return ctx_fail(c, BCIR_EMIT_TOO_DEEP);
  if (c->visits >= c->scratch_cap) return ctx_fail(c, BCIR_EMIT_SCRATCH_SHORT);
  c->visits++;

  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN:
      if (!rd_u8(c, &byte)) return 0;
      /* 11.1: DER's TRUE is 0xFF exactly, not "any non-zero octet". */
      put(c, byte ? 0xFF : 0x00);
      return 1;
    case BCIR_EMIT_INTEGER:
    case BCIR_EMIT_ENUMERATED:
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      put_bytes(c, data, byte);
      return 1;
    case BCIR_EMIT_NULL:
      return 1;
    case BCIR_EMIT_OCTETSTRING:
    case BCIR_EMIT_STRING:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      put_bytes(c, data, count);
      return 1;
    case BCIR_EMIT_OID:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      return oid_octets(c, data, count, 0, NULL);
    case BCIR_EMIT_SEQUENCE:
      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        int present = 1;
        if (!member_present(c, m, 1, depth + 1, &present)) return 0;
        if (!present) continue;
        if (!der_write_member(c, m, depth + 1)) return 0;
      }
      return 1;
    case BCIR_EMIT_SEQUENCE_OF:
      if (!rd_u32(c, &count)) return 0;
      if (node->element < 0) return ctx_fail(c, BCIR_EMIT_PLAN_MALFORMED);
      for (i = 0; i < count; i++)
        if (!der_write_full(c, (uint32_t)node->element, depth + 1)) return 0;
      return 1;
    case BCIR_EMIT_CHOICE: {
      const bcir_emit_member *m;
      if (!rd_u32(c, &count)) return 0;
      if (count >= node->member_count) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      m = &c->plan->members[node->first_member + count];
      return der_write_member(c, m, depth + 1);
    }
    default:
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  }
}

static int der_write_member(emit_ctx *c, const bcir_emit_member *m, uint32_t depth) {
  const bcir_emit_node *inner = &c->plan->nodes[m->node];
  size_t inner_content;
  if (m->tag < 0) return der_write_full(c, m->node, depth);
  if (c->visits >= c->scratch_cap) return ctx_fail(c, BCIR_EMIT_SCRATCH_SHORT);
  inner_content = c->scratch[c->visits]; /* the inner node's own slot, not yet consumed */
  if (m->explicit_tag) {
    size_t inner_total = (inner->kind == BCIR_EMIT_CHOICE)
                             ? inner_content
                             : identifier_size(inner->universal) +
                                   definite_length_size(inner_content) + inner_content;
    put_identifier(c, m->tag_class, 1, (uint32_t)m->tag);
    put_definite_length(c, inner_total);
    return der_write_full(c, m->node, depth);
  }
  put_identifier(c, m->tag_class, node_is_constructed(inner), (uint32_t)m->tag);
  put_definite_length(c, inner_content);
  return der_write_content(c, m->node, depth);
}

/* --- BER: one pass, indefinite lengths ---------------------------------------------------- */
/*
 * 8.1.3.6 lets a CONSTRUCTED encoding leave its length open and close with an end-of-contents
 * pair. That is exactly why BER needs no measure pass and no scratch, and it is the whole
 * difference between this candidate and DER -- so the cost gap between the two rows is a
 * property of the encodings rather than of this file.
 */

static int ber_content(emit_ctx *c, uint32_t node_index, uint32_t depth);
static int ber_member(emit_ctx *c, const bcir_emit_member *m, uint32_t depth);

static int ber_full(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  if (node->kind == BCIR_EMIT_CHOICE) return ber_content(c, node_index, depth);
  if (node_is_constructed(node)) {
    put_identifier(c, 0, 1, node->universal);
    put(c, 0x80);
    if (!ber_content(c, node_index, depth)) return 0;
    put(c, 0x00);
    put(c, 0x00);
    return 1;
  }
  /* A primitive has no nested encoding to leave open, so its length is known immediately
   * and written definite -- which is what the Python reference does and what 8.1.3.2
   * requires (the indefinite form is available to constructed encodings only). */
  {
    size_t before = c->written, content;
    uint8_t scratch_head[1];
    (void)scratch_head;
    (void)before;
    /* Primitive content sizes are all readable from the stream without descending. */
    switch ((bcir_emit_kind)node->kind) {
      case BCIR_EMIT_BOOLEAN: content = 1; break;
      case BCIR_EMIT_INTEGER:
      case BCIR_EMIT_ENUMERATED:
        if (c->stream_len - c->at < 1) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        content = c->stream[c->at];
        break;
      case BCIR_EMIT_NULL: content = 0; break;
      case BCIR_EMIT_OCTETSTRING:
      case BCIR_EMIT_STRING: {
        uint32_t n;
        if (c->stream_len - c->at < 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        n = ((uint32_t)c->stream[c->at] << 24) | ((uint32_t)c->stream[c->at + 1] << 16) |
            ((uint32_t)c->stream[c->at + 2] << 8) | (uint32_t)c->stream[c->at + 3];
        content = n;
        break;
      }
      case BCIR_EMIT_OID: {
        uint32_t n;
        size_t size = 0;
        const uint8_t *text;
        if (c->stream_len - c->at < 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        n = ((uint32_t)c->stream[c->at] << 24) | ((uint32_t)c->stream[c->at + 1] << 16) |
            ((uint32_t)c->stream[c->at + 2] << 8) | (uint32_t)c->stream[c->at + 3];
        if ((size_t)n > c->stream_len - c->at - 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        text = c->stream + c->at + 4;
        if (!oid_octets(c, text, n, 1, &size)) return 0;
        content = size;
        break;
      }
      default: return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    }
    put_identifier(c, 0, 0, node->universal);
    put_definite_length(c, content);
    return ber_content(c, node_index, depth);
  }
}

static int ber_content(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  uint8_t byte = 0;
  uint32_t count = 0, i;
  const uint8_t *data = NULL;

  if (depth > c->max_depth) return ctx_fail(c, BCIR_EMIT_TOO_DEEP);
  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN:
      if (!rd_u8(c, &byte)) return 0;
      put(c, byte ? 0xFF : 0x00);
      return 1;
    case BCIR_EMIT_INTEGER:
    case BCIR_EMIT_ENUMERATED:
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      put_bytes(c, data, byte);
      return 1;
    case BCIR_EMIT_NULL:
      return 1;
    case BCIR_EMIT_OCTETSTRING:
    case BCIR_EMIT_STRING:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      put_bytes(c, data, count);
      return 1;
    case BCIR_EMIT_OID:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      return oid_octets(c, data, count, 0, NULL);
    case BCIR_EMIT_SEQUENCE:
      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        int present = 1;
        if (!member_present(c, m, 0, depth + 1, &present)) return 0;
        if (!present) continue;
        if (!ber_member(c, m, depth + 1)) return 0;
      }
      return 1;
    case BCIR_EMIT_SEQUENCE_OF:
      if (!rd_u32(c, &count)) return 0;
      if (node->element < 0) return ctx_fail(c, BCIR_EMIT_PLAN_MALFORMED);
      if (!bound_elements(c, (uint32_t)node->element, count)) return 0;
      for (i = 0; i < count; i++)
        if (!ber_full(c, (uint32_t)node->element, depth + 1)) return 0;
      return 1;
    case BCIR_EMIT_CHOICE: {
      const bcir_emit_member *m;
      if (!rd_u32(c, &count)) return 0;
      if (count >= node->member_count) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      m = &c->plan->members[node->first_member + count];
      return ber_member(c, m, depth + 1);
    }
    default:
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  }
}

static int ber_member(emit_ctx *c, const bcir_emit_member *m, uint32_t depth) {
  const bcir_emit_node *inner = &c->plan->nodes[m->node];
  if (m->tag < 0) return ber_full(c, m->node, depth);
  if (m->explicit_tag) {
    put_identifier(c, m->tag_class, 1, (uint32_t)m->tag);
    put(c, 0x80);
    if (!ber_full(c, m->node, depth)) return 0;
    put(c, 0x00);
    put(c, 0x00);
    return 1;
  }
  if (node_is_constructed(inner)) {
    put_identifier(c, m->tag_class, 1, (uint32_t)m->tag);
    put(c, 0x80);
    if (!ber_content(c, m->node, depth)) return 0;
    put(c, 0x00);
    put(c, 0x00);
    return 1;
  }
  /* An implicitly tagged PRIMITIVE: definite length, and the size is readable from the
   * stream without descending -- so ber_full's primitive path already does the right thing
   * once the identifier is replaced. It is spelled out here rather than shared because the
   * tag CLASS differs and 8.14.4 is the reason. */
  {
    size_t content;
    switch ((bcir_emit_kind)inner->kind) {
      case BCIR_EMIT_BOOLEAN: content = 1; break;
      case BCIR_EMIT_INTEGER:
      case BCIR_EMIT_ENUMERATED:
        if (c->stream_len - c->at < 1) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        content = c->stream[c->at];
        break;
      case BCIR_EMIT_NULL: content = 0; break;
      case BCIR_EMIT_OCTETSTRING:
      case BCIR_EMIT_STRING: {
        uint32_t n;
        if (c->stream_len - c->at < 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        n = ((uint32_t)c->stream[c->at] << 24) | ((uint32_t)c->stream[c->at + 1] << 16) |
            ((uint32_t)c->stream[c->at + 2] << 8) | (uint32_t)c->stream[c->at + 3];
        content = n;
        break;
      }
      case BCIR_EMIT_OID: {
        uint32_t n;
        size_t size = 0;
        if (c->stream_len - c->at < 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        n = ((uint32_t)c->stream[c->at] << 24) | ((uint32_t)c->stream[c->at + 1] << 16) |
            ((uint32_t)c->stream[c->at + 2] << 8) | (uint32_t)c->stream[c->at + 3];
        if ((size_t)n > c->stream_len - c->at - 4) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
        if (!oid_octets(c, c->stream + c->at + 4, n, 1, &size)) return 0;
        content = size;
        break;
      }
      default: return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    }
    put_identifier(c, m->tag_class, 0, (uint32_t)m->tag);
    put_definite_length(c, content);
    return ber_content(c, m->node, depth);
  }
}

/* --- X.697 JER: one pass of text ----------------------------------------------------------- */

/* The abstract integer arrives as minimal two's complement, of ANY width; JER needs it in
 * decimal. The first version accumulated into a uint64_t and silently truncated above 64
 * bits -- 2^64+7 came out as 7, which is a perfectly well-formed JER document of a
 * different value. That is the failure mode this whole harness exists to avoid, so the
 * conversion is long division on the magnitude octets and a width past the buffer is a
 * refusal rather than a wrap. */
static int put_int_decimal(emit_ctx *c, const uint8_t *octets, size_t len) {
  uint8_t mag[64];
  char digits[160];
  size_t i, start = 0;
  int d = 0;

  if (len > sizeof(mag)) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  if (len > 0 && (octets[0] & 0x80) != 0) {
    unsigned carry = 1;
    for (i = len; i-- > 0;) {
      unsigned value = (unsigned)((uint8_t)(~octets[i] & 0xFFu)) + carry;
      mag[i] = (uint8_t)(value & 0xFFu);
      carry = value >> 8;
    }
    put(c, '-');
  } else {
    for (i = 0; i < len; i++) mag[i] = octets[i];
  }

  while (start < len && mag[start] == 0) start++;
  if (start == len) {
    put(c, '0');
    return 1;
  }
  while (start < len) {
    unsigned remainder = 0;
    for (i = start; i < len; i++) {
      unsigned current = (remainder << 8) | mag[i];
      mag[i] = (uint8_t)(current / 10u);
      remainder = current % 10u;
    }
    if (d >= (int)sizeof(digits)) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    digits[d++] = (char)('0' + (int)remainder);
    while (start < len && mag[start] == 0) start++;
  }
  for (i = (size_t)d; i-- > 0;) put(c, (uint8_t)digits[i]);
  return 1;
}

static void put_json_string(emit_ctx *c, const uint8_t *data, size_t len) {
  static const char hex[] = "0123456789abcdef";
  size_t i;
  put(c, '"');
  for (i = 0; i < len; i++) {
    uint8_t ch = data[i];
    if (ch == '"' || ch == '\\') { put(c, '\\'); put(c, ch); }
    else if (ch == '\n') { put(c, '\\'); put(c, 'n'); }
    else if (ch == '\r') { put(c, '\\'); put(c, 'r'); }
    else if (ch == '\t') { put(c, '\\'); put(c, 't'); }
    else if (ch == '\b') { put(c, '\\'); put(c, 'b'); }
    else if (ch == '\f') { put(c, '\\'); put(c, 'f'); }
    else if (ch < 0x20) {
      put(c, '\\'); put(c, 'u'); put(c, '0'); put(c, '0');
      put(c, (uint8_t)hex[ch >> 4]);
      put(c, (uint8_t)hex[ch & 0x0F]);
    } else {
      put(c, ch);
    }
  }
  put(c, '"');
}

static int jer_node(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  static const char upper[] = "0123456789ABCDEF";
  uint8_t byte = 0;
  uint32_t count = 0, i;
  const uint8_t *data = NULL;
  int first = 1;

  if (depth > c->max_depth) return ctx_fail(c, BCIR_EMIT_TOO_DEEP);
  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN:
      if (!rd_u8(c, &byte)) return 0;
      if (byte) { put(c, 't'); put(c, 'r'); put(c, 'u'); put(c, 'e'); }
      else { put(c, 'f'); put(c, 'a'); put(c, 'l'); put(c, 's'); put(c, 'e'); }
      return 1;
    case BCIR_EMIT_INTEGER:
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      return put_int_decimal(c, data, byte);
    case BCIR_EMIT_ENUMERATED: {
      /* 22.2: the JSON string denotes "the identifier of the chosen enumeration item".
       * NOT the number -- X.690 8.4 encodes the number and X.697 deliberately does not,
       * which is why the plan carries the identifiers at all. Sharing this branch with
       * INTEGER emitted `4` where the standard requires `"red"`. */
      int64_t value = 0;
      uint32_t k;
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      if (byte == 0 || byte > 8) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      value = (data[0] & 0x80u) ? -1 : 0;              /* sign-extend, then shift in */
      for (i = 0; i < byte; i++) value = (int64_t)(((uint64_t)value << 8) | data[i]);
      for (k = 0; k < node->enum_count; k++) {
        const bcir_emit_enum_item *item = &c->plan->enums[node->first_enum + k];
        if (item->number != value) continue;
        put(c, '"');
        put_bytes(c, (const uint8_t *)item->name, item->name_len);
        put(c, '"');
        return 1;
      }
      /* 22.1 gives an enumerated value no numeric spelling, so there is nothing to fall
       * back to: a number outside the enumeration has no JER document at all. */
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    }
    case BCIR_EMIT_NULL:
      put(c, 'n'); put(c, 'u'); put(c, 'l'); put(c, 'l');
      return 1;
    case BCIR_EMIT_OCTETSTRING:
      /* 21: an OCTET STRING is an upper-case hexadecimal string. */
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      put(c, '"');
      for (i = 0; i < count; i++) {
        put(c, (uint8_t)upper[data[i] >> 4]);
        put(c, (uint8_t)upper[data[i] & 0x0F]);
      }
      put(c, '"');
      return 1;
    case BCIR_EMIT_STRING:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      put_json_string(c, data, count);
      return 1;
    case BCIR_EMIT_OID:
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      put_json_string(c, data, count);
      return 1;
    case BCIR_EMIT_SEQUENCE:
      put(c, '{');
      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        int present = 1;
        if (!member_present(c, m, 1, depth + 1, &present)) return 0;
        if (!present) continue;
        if (!first) put(c, ',');
        first = 0;
        /* 22.2: the member's IDENTIFIER is what a JER document carries -- the whole reason
         * this emitter cannot be schema-free. */
        put_json_string(c, (const uint8_t *)m->name, m->name_len);
        put(c, ':');
        if (!jer_node(c, m->node, depth + 1)) return 0;
      }
      put(c, '}');
      return 1;
    case BCIR_EMIT_SEQUENCE_OF:
      if (!rd_u32(c, &count)) return 0;
      if (node->element < 0) return ctx_fail(c, BCIR_EMIT_PLAN_MALFORMED);
      if (!bound_elements(c, (uint32_t)node->element, count)) return 0;
      put(c, '[');
      for (i = 0; i < count; i++) {
        if (i) put(c, ',');
        if (!jer_node(c, (uint32_t)node->element, depth + 1)) return 0;
      }
      put(c, ']');
      return 1;
    case BCIR_EMIT_CHOICE: {
      const bcir_emit_member *m;
      if (!rd_u32(c, &count)) return 0;
      if (count >= node->member_count) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      m = &c->plan->members[node->first_member + count];
      put(c, '{');
      put_json_string(c, (const uint8_t *)m->name, m->name_len);
      put(c, ':');
      if (!jer_node(c, m->node, depth + 1)) return 0;
      put(c, '}');
      return 1;
    }
    default:
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  }
}

/* --- X.696 CANONICAL-OER: one pass --------------------------------------------------------- */

static void put_oer_length(emit_ctx *c, size_t count) { put_definite_length(c, count); }

/* 10.3 / 10.4: the fixed word width the constraint selects, and its sign. Width 0 is
 * 10.3 e) / 10.4 e), the length-prefixed variable-size form.
 *
 * The split between the two clauses is whether a lower bound EXISTS and is non-negative --
 * not whether the bounds happen to be small -- so a type with no lower bound is signed
 * however tight its upper bound is. This mirrors the Python `_oer_integer_form` clause for
 * clause; both read the plan, and the oracle reads the live constraint. */
static void oer_integer_form(const emit_ctx *c, const bcir_emit_node *node, unsigned *width,
                             int *is_signed) {
  static const uint64_t unsigned_limit[4] = {0xFFull, 0xFFFFull, 0xFFFFFFFFull,
                                             0xFFFFFFFFFFFFFFFFull};
  static const unsigned widths[4] = {1, 2, 4, 8};
  const bcir_emit_constraint *k;
  int i;

  *width = 0;
  *is_signed = 1;
  if (node->constraint < 0) return;
  k = &c->plan->constraints[node->constraint];
  if (k->value_low.present && !k->value_low.negative) {   /* 10.2 a) -> 10.3, unsigned */
    *is_signed = 0;
    if (!k->value_high.present || k->value_high.negative) return;      /* 10.3 e) */
    for (i = 0; i < 4; i++)
      if (k->value_high.magnitude <= unsigned_limit[i]) { *width = widths[i]; return; }
    return;                                                           /* 10.3 e) */
  }
  if (!k->value_low.present || !k->value_high.present) return;         /* 10.4 e) */
  for (i = 0; i < 4; i++) {
    /* `top` is 2^(bits-1): the magnitude of the most negative value the word holds, and one
     * past the largest positive one. Written this way because 8 octets makes 2^63 the
     * boundary and computing `1 << 64` to get the limit would be undefined. */
    uint64_t top = 1ull << (widths[i] * 8u - 1u);
    int low_ok = k->value_low.negative ? (k->value_low.magnitude <= top)
                                       : (k->value_low.magnitude < top);
    int high_ok = k->value_high.negative || k->value_high.magnitude < top;
    if (low_ok && high_ok) { *width = widths[i]; return; }
  }
}

/* Write a value the stream carries as minimal two's complement into the fixed `width`-octet
 * word the constraint selected. REFUSES rather than truncates: a value outside its own
 * constraint is a well-formed document of a different value, which is the failure this
 * whole constraint-carrying plan version exists to stop. */
static int oer_fixed_word(emit_ctx *c, const uint8_t *data, size_t n, unsigned width,
                          int is_signed) {
  uint8_t fill;
  size_t i;

  if (n == 0) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
  /* A non-negative value's minimal two's-complement form always has its top bit clear, so
   * a set one here means the value is negative -- and an unsigned field cannot hold it. */
  fill = (uint8_t)((data[0] & 0x80u) ? 0xFFu : 0x00u);
  if (!is_signed && fill == 0xFFu) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  if (n > width) {
    for (i = 0; i < n - width; i++)
      if (data[i] != fill) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    if (is_signed && (((data[n - width] ^ fill) & 0x80u) != 0))
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
    put_bytes(c, data + (n - width), width);
    return 1;
  }
  for (i = n; i < width; i++) put(c, fill);
  put_bytes(c, data, n);
  return 1;
}

/* 14.1 / 27.2: the single length a SIZE constraint fixes. A RANGE is not enough -- only an
 * exact length lets a decoder find the end of the field without a determinant. */
static int oer_fixed_size(const emit_ctx *c, const bcir_emit_node *node, uint64_t *out) {
  const bcir_emit_constraint *k;
  if (node->constraint < 0) return 0;
  k = &c->plan->constraints[node->constraint];
  if (!k->size_low.present || !k->size_high.present) return 0;
  if (k->size_low.negative || k->size_high.negative) return 0;
  if (k->size_low.magnitude != k->size_high.magnitude) return 0;
  *out = k->size_low.magnitude;
  return 1;
}

/* 27.2's known-multiplier character types, restricted to the ones this plan compiles as
 * `string`. UTF8String is deliberately absent: 27.1 keeps it out because a character costs
 * 1..4 octets there, so its length is never implied by a character count. */
static int oer_known_multiplier(uint32_t universal) {
  return universal == 18u || universal == 19u || universal == 22u || universal == 26u;
}

static int oer_node(emit_ctx *c, uint32_t node_index, uint32_t depth) {
  const bcir_emit_node *node = &c->plan->nodes[node_index];
  uint8_t byte = 0;
  uint32_t count = 0, i;
  const uint8_t *data = NULL;

  if (depth > c->max_depth) return ctx_fail(c, BCIR_EMIT_TOO_DEEP);
  switch ((bcir_emit_kind)node->kind) {
    case BCIR_EMIT_BOOLEAN:
      if (!rd_u8(c, &byte)) return 0;
      /* 11: any non-zero octet is TRUE, and CANONICAL-OER fixes it at 0xFF. */
      put(c, byte ? 0xFF : 0x00);
      return 1;
    case BCIR_EMIT_INTEGER: {
      /* 10.3 / 10.4. A constrained integer is a FIXED-WIDTH word with no length
       * determinant; only the unbounded cases 10.3 e) / 10.4 e) are length-prefixed. */
      unsigned width = 0;
      int is_signed = 1;
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      oer_integer_form(c, node, &width, &is_signed);
      if (width != 0) return oer_fixed_word(c, data, byte, width, is_signed);
      if (!is_signed) {
        /* 10.3 e) is UNSIGNED, and the stream carries the minimal SIGNED form -- which
         * gives a value at or above 0x80 a leading zero octet the unsigned form drops. */
        while (byte > 1 && data[0] == 0x00) { data++; byte--; }
        if ((data[0] & 0x80u) != 0) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      }
      put_oer_length(c, byte);
      put_bytes(c, data, byte);
      return 1;
    }
    case BCIR_EMIT_ENUMERATED:
      /* 11 gives ENUMERATED its own form and reads NO constraint: the short form below 128,
       * else a long form whose count octet has its top bit set and whose body is SIGNED.
       * It is not 10's integer, and sharing a branch with one emitted `01 05` where the
       * standard says `05` -- a well-formed document of a different value for every
       * enumerated ever encoded through this plan. */
      if (!rd_u8(c, &byte) || !rd_take(c, byte, &data)) return 0;
      if (byte == 0) return ctx_fail(c, BCIR_EMIT_STREAM_SHORT);
      if (byte == 1 && data[0] < 0x80u) { put(c, data[0]); return 1; }   /* 11.3 */
      if (byte > 0x7Fu) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);       /* 11.4 */
      put(c, (uint8_t)(0x80u | byte));
      put_bytes(c, data, byte);
      return 1;
    case BCIR_EMIT_NULL:
      return 1; /* 12: no octets at all */
    case BCIR_EMIT_OCTETSTRING: /* 14 */
    case BCIR_EMIT_STRING: {    /* 27 */
      uint64_t fixed = 0;
      int has_fixed = oer_fixed_size(c, node, &fixed);
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      /* 14.1 and 27.2 drop the length determinant when the size is pinned to one value.
       * Every repertoire `oer_known_multiplier` admits is single-octet, so the stream's
       * octet count IS the character count 27.2 is about. */
      if (has_fixed && (node->kind == BCIR_EMIT_OCTETSTRING ||
                        oer_known_multiplier(node->universal))) {
        if ((uint64_t)count != fixed) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
        put_bytes(c, data, count);
        return 1;
      }
      put_oer_length(c, count);
      put_bytes(c, data, count);
      return 1;
    }
    case BCIR_EMIT_OID: {       /* 14 */
      size_t size = 0;
      if (!rd_u32(c, &count) || !rd_take(c, count, &data)) return 0;
      if (!oid_octets(c, data, count, 1, &size)) return 0;
      put_oer_length(c, size);
      return oid_octets(c, data, count, 0, NULL);
    }
    case BCIR_EMIT_SEQUENCE: {
      /* 16.2: a preamble of one bit per OPTIONAL/DEFAULT component, MSB first, zero-padded
       * to an octet boundary -- and 16.2.2 makes that padding zero in CANONICAL-OER.
       *
       * The preamble precedes the components on the wire, but the presence octets are
       * INTERLEAVED with values in the stream, so it cannot simply be written first. The
       * first version walked the members twice, once to collect the bits and once to emit;
       * a fuzzer found the hang that implies, because every nested SEQUENCE then re-walks
       * its whole subtree and the cost is exponential in depth.
       *
       * The preamble's SIZE, though, comes from the PLAN -- it is the number of optional
       * components, which no value can change. So the space is reserved, the body is
       * emitted in one pass, and the bits are patched into the reserved octets as each
       * presence flag is read. */
      size_t optional_count = 0, preamble_octets, preamble_at, seen = 0;

      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        if (m->optional || m->has_default) optional_count++;
      }
      preamble_octets = (optional_count + 7u) / 8u;
      preamble_at = c->written;
      for (i = 0; i < preamble_octets; i++) put(c, 0x00);

      for (i = 0; i < node->member_count; i++) {
        const bcir_emit_member *m = &c->plan->members[node->first_member + i];
        int present = 1;
        /* 31.9: a DEFAULT component equal to its default is ABSENT, and the PREAMBLE bit
         * must say so too -- writing a 1 and then no field would misalign every component
         * after it. */
        if (!member_present(c, m, 1, depth + 1, &present)) return 0;
        if (m->optional || m->has_default) {
          byte = (uint8_t)present;
          if (byte && c->out) {
            /* Guarded exactly as `put` is: count always, write only where there is room.
             * A short buffer drops these bits along with everything else past the cap, and
             * `written` still reports the true total so the caller can retry. */
            size_t at = preamble_at + (seen >> 3);
            if (at < c->out_cap) c->out[at] |= (uint8_t)(0x80u >> (seen & 7u));
          }
          seen++;
          if (!byte) continue;
        }
        if (!oer_node(c, m->node, depth + 1)) return 0;
      }
      return 1;
    }
    case BCIR_EMIT_SEQUENCE_OF: {
      /* 19.1: the quantity is itself a length-prefixed unsigned integer, NOT a fixed word.
       * An empty SEQUENCE OF is `01 00`; four zero octets instead produce a document a
       * conforming decoder reads as a three-element sequence. */
      uint8_t quantity[4];
      int n = 0, k;
      uint32_t value;
      if (!rd_u32(c, &count)) return 0;
      if (node->element < 0) return ctx_fail(c, BCIR_EMIT_PLAN_MALFORMED);
      if (!bound_elements(c, (uint32_t)node->element, count)) return 0;
      value = count;
      do {
        quantity[n++] = (uint8_t)(value & 0xFFu);
        value >>= 8;
      } while (value != 0);
      put_oer_length(c, (size_t)n);
      for (k = n - 1; k >= 0; k--) put(c, quantity[k]);
      for (i = 0; i < count; i++)
        if (!oer_node(c, (uint32_t)node->element, depth + 1)) return 0;
      return 1;
    }
    case BCIR_EMIT_CHOICE: {
      /* 20.1: the alternative's TAG identifies it -- a CHOICE is the one place OER puts a
       * tag on the wire (8.7.1). */
      const bcir_emit_member *m;
      uint32_t tag;
      if (!rd_u32(c, &count)) return 0;
      if (count >= node->member_count) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      m = &c->plan->members[node->first_member + count];
      tag = (m->tag >= 0) ? (uint32_t)m->tag : count;
      if (tag >= 0x40u) return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
      put(c, (uint8_t)(m->tag_class | tag));
      return oer_node(c, m->node, depth + 1);
    }
    default:
      return ctx_fail(c, BCIR_EMIT_UNSUPPORTED);
  }
}

/* --- the entry point ------------------------------------------------------------------------ */

bcir_emit_status bcir_emit(const bcir_emit_plan *plan, bcir_emit_rules rules,
                           const uint8_t *stream, size_t stream_len, uint8_t *out,
                           size_t out_cap, size_t *written, uint32_t *scratch,
                           size_t scratch_cap, uint32_t max_depth, bcir_emit_diag *diag) {
  emit_ctx c;
  int ok;

  if (diag) { diag->status = BCIR_EMIT_OK; diag->offset = 0; diag->needed = 0; }
  if (!plan || !stream || !written) return BCIR_EMIT_PLAN_MALFORMED;
  if (max_depth > BCIR_EMIT_MAX_PLAN_DEPTH) max_depth = BCIR_EMIT_MAX_PLAN_DEPTH;
  *written = 0;

  c.plan = plan;
  c.stream = stream;
  c.stream_len = stream_len;
  c.at = 0;
  c.out = out;
  c.out_cap = out_cap;
  c.written = 0;
  c.scratch = scratch;
  c.scratch_cap = scratch ? scratch_cap : 0;
  c.visits = 0;
  c.max_depth = max_depth;
  c.status = BCIR_EMIT_OK;
  c.fail_offset = 0;

  if (rules == BCIR_EMIT_DER) {
    /* Pass one: content lengths into the scratch, output suppressed. */
    emit_ctx measure = c;
    measure.out = NULL;
    measure.out_cap = 0;
    (void)der_measure_full(&measure, plan->root, 0);
    if (measure.status == BCIR_EMIT_SCRATCH_SHORT) {
      if (diag) {
        diag->status = BCIR_EMIT_SCRATCH_SHORT;
        diag->needed = measure.visits;
      }
      return BCIR_EMIT_SCRATCH_SHORT;
    }
    if (measure.status != BCIR_EMIT_OK) {
      if (diag) { diag->status = measure.status; diag->offset = measure.fail_offset; }
      return measure.status;
    }
    if (measure.at != stream_len) {
      if (diag) { diag->status = BCIR_EMIT_STREAM_LONG; diag->offset = measure.at; }
      return BCIR_EMIT_STREAM_LONG;
    }
    c.visits = 0;
    ok = der_write_full(&c, plan->root, 0);
  } else if (rules == BCIR_EMIT_BER) {
    ok = ber_full(&c, plan->root, 0);
  } else if (rules == BCIR_EMIT_JER) {
    ok = jer_node(&c, plan->root, 0);
  } else if (rules == BCIR_EMIT_COER) {
    ok = oer_node(&c, plan->root, 0);
  } else {
    return BCIR_EMIT_UNSUPPORTED;
  }

  if (!ok || c.status != BCIR_EMIT_OK) {
    if (diag) { diag->status = c.status; diag->offset = c.fail_offset; }
    return c.status == BCIR_EMIT_OK ? BCIR_EMIT_UNSUPPORTED : c.status;
  }
  /* A leftover suffix means the plan and the stream describe different values, and the
   * encoding built from a prefix would be a VALID document of the wrong value. */
  if (c.at != stream_len) {
    if (diag) { diag->status = BCIR_EMIT_STREAM_LONG; diag->offset = c.at; }
    return BCIR_EMIT_STREAM_LONG;
  }
  *written = c.written;
  if (c.written > out_cap) {
    if (diag) { diag->status = BCIR_EMIT_OUT_SHORT; diag->needed = c.written; }
    return BCIR_EMIT_OUT_SHORT;
  }
  return BCIR_EMIT_OK;
}
