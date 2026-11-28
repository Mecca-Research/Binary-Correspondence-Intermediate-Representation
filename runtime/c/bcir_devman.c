/*===- bcir_devman.c - device manifests bound at instantiation -------------------------------------------===
 *
 * See bcir_devman.h. The reader's events build the document's nodes in the caller's workspace; the oracle's laws
 * then run over the nodes in its order, so a document wrong in several ways is refused for the same first reason
 * on both rails.
 *===----------------------------------------------------------------------===*/
#include "bcir_devman.h"

#include "bcir_sha256.h"

#define MAX_DEPTH 8u
#define MAX_ELEMENTS 256u
#define MAX_INT_DIGITS 18u

void bcir_devman_limits(bcir_jer_limits *out) {
  if (!out) return;
  out->input_bytes = BCIR_DEVMAN_MAX_BYTES;
  out->depth = MAX_DEPTH;
  out->nodes = (uint64_t)1u << 18;
  out->members = 64u;
  out->elements = MAX_ELEMENTS;
  out->string_bytes = 4096u;
  out->number_bytes = 64u;
  out->integer_digits = MAX_INT_DIGITS;
  out->exponent_magnitude = 4096u;
  out->work = (uint64_t)1u << 25;
}

/* ---- the document's nodes ---------------------------------------------------------------------------- */

enum { N_OBJ, N_ARR, N_MEM, N_STR, N_INT, N_FLOAT, N_TRUE, N_FALSE, N_NULL };
#define NONE UINT32_MAX

typedef struct node {
  uint8_t kind;
  uint32_t first, last, next, count; /* children (a member's one child is its value) */
  uint32_t str_off, str_len;         /* a member's name, a string's text: in the string arena */
  int64_t ival;
} node;

typedef struct builder {
  node *nodes;
  uint32_t n, cap;
  uint8_t *strings;
  size_t used, strings_cap;
  uint32_t stack[MAX_DEPTH + 1u];
  uint32_t depth;
  uint32_t root;
  int space; /* the workspace ran short */
} builder;

#define SINK_DUPLICATE 1
#define SINK_SPACE 2
#define SINK_NUMBER 3

static int bytes_eq(const uint8_t *a, size_t na, const uint8_t *b, size_t nb) {
  if (na != nb) return 0;
  for (size_t k = 0; k < na; k++)
    if (a[k] != b[k]) return 0;
  return 1;
}

static uint32_t new_node(builder *b, uint8_t kind) {
  if (b->n == b->cap) {
    b->space = 1;
    return NONE;
  }
  node *x = &b->nodes[b->n];
  x->kind = kind;
  x->first = x->last = x->next = NONE;
  x->count = 0;
  x->str_off = x->str_len = 0;
  x->ival = 0;
  return b->n++;
}

static void add_child(builder *b, uint32_t parent, uint32_t child) {
  node *p = &b->nodes[parent];
  if (p->last == NONE) p->first = child;
  else b->nodes[p->last].next = child;
  p->last = child;
  p->count++;
}

/* a value: the root, an array's element, or the value of the member its object opened last */
static void place(builder *b, uint32_t v) {
  if (b->depth == 0) {
    b->root = v;
    return;
  }
  uint32_t parent = b->stack[b->depth - 1u];
  if (b->nodes[parent].kind == N_OBJ) parent = b->nodes[parent].last; /* the pending member */
  add_child(b, parent, v);
}

static int copy_text(builder *b, uint32_t at, const uint8_t *text, size_t len) {
  if (len > b->strings_cap - b->used) return 0;
  for (size_t k = 0; k < len; k++) b->strings[b->used + k] = text[k];
  b->nodes[at].str_off = (uint32_t)b->used;
  b->nodes[at].str_len = (uint32_t)len;
  b->used += len;
  return 1;
}

/* JSON's integer: -?(0|[1-9][0-9]*), no fraction and no exponent (the oracle's int, not its float) */
static int number(const uint8_t *t, size_t len, uint8_t *kind, int64_t *value) {
  size_t k = 0;
  int neg = len && t[0] == '-';
  for (size_t j = 0; j < len; j++)
    if (t[j] == '.' || t[j] == 'e' || t[j] == 'E') {
      *kind = N_FLOAT;
      return 1;
    }
  k = (size_t)neg;
  if (len - k > MAX_INT_DIGITS || k == len) return 0; /* the scan's digit bound, held here too */
  int64_t v = 0;
  for (; k < len; k++) v = v * 10 + (int64_t)(t[k] - '0');
  *kind = N_INT;
  *value = neg ? -v : v;
  return 1;
}

static int sink(void *ctx, bcir_jer_event ev, size_t offset, const uint8_t *text, size_t len) {
  builder *b = (builder *)ctx;
  uint32_t x;
  (void)offset;
  switch (ev) {
  case BCIR_JER_EV_OBJECT_BEGIN:
  case BCIR_JER_EV_ARRAY_BEGIN:
    x = new_node(b, ev == BCIR_JER_EV_OBJECT_BEGIN ? N_OBJ : N_ARR);
    if (x == NONE || b->depth > MAX_DEPTH) return SINK_SPACE;
    place(b, x);
    b->stack[b->depth++] = x;
    return 0;
  case BCIR_JER_EV_OBJECT_END: {
    const node *o = &b->nodes[b->stack[b->depth - 1u]];
    /* no member name twice in one object (the oracle's object_pairs_hook) */
    for (uint32_t m = o->first; m != NONE; m = b->nodes[m].next)
      for (uint32_t k = b->nodes[m].next; k != NONE; k = b->nodes[k].next)
        if (bytes_eq(b->strings + b->nodes[m].str_off, b->nodes[m].str_len, b->strings + b->nodes[k].str_off,
                     b->nodes[k].str_len))
          return SINK_DUPLICATE;
    b->depth--;
    return 0;
  }
  case BCIR_JER_EV_ARRAY_END:
    b->depth--;
    return 0;
  case BCIR_JER_EV_MEMBER_NAME:
    x = new_node(b, N_MEM);
    if (x == NONE || !copy_text(b, x, text, len)) return SINK_SPACE;
    add_child(b, b->stack[b->depth - 1u], x);
    return 0;
  case BCIR_JER_EV_STRING:
    x = new_node(b, N_STR);
    if (x == NONE || !copy_text(b, x, text, len)) return SINK_SPACE;
    place(b, x);
    return 0;
  case BCIR_JER_EV_NUMBER: {
    uint8_t kind = N_FLOAT;
    int64_t v = 0;
    if (!number(text, len, &kind, &v)) return SINK_NUMBER;
    x = new_node(b, kind);
    if (x == NONE) return SINK_SPACE;
    b->nodes[x].ival = v;
    place(b, x);
    return 0;
  }
  case BCIR_JER_EV_TRUE:
  case BCIR_JER_EV_FALSE:
  case BCIR_JER_EV_NULL:
    x = new_node(b, ev == BCIR_JER_EV_TRUE ? N_TRUE : ev == BCIR_JER_EV_FALSE ? N_FALSE : N_NULL);
    if (x == NONE) return SINK_SPACE;
    place(b, x);
    return 0;
  }
  return SINK_SPACE;
}

/* ---- the workspace ----------------------------------------------------------------------------------- */

typedef struct layout {
  size_t stack, scratch, nodes, strings, banks, rows, values, total;
  uint32_t node_cap;
  size_t value_cap;
} layout;

static size_t up8(size_t n) { return (n + 7u) & ~(size_t)7u; }

/* The workspace for a document of `len` octets and `values` values (the scan's count, or its bound): a node per
 * value and per member name -- a member has one value, so at most two a value -- an arena for the decoded strings,
 * which are no longer than the document, and the loaded manifest's arrays. */
static int plan(size_t len, uint64_t values, layout *L) {
  bcir_jer_limits lim;
  bcir_devman_limits(&lim);
  if (len > BCIR_DEVMAN_MAX_BYTES || values > lim.nodes) return 0;
  L->node_cap = (uint32_t)(2u * values + 2u);
  L->value_cap = values < (uint64_t)MAX_ELEMENTS * MAX_ELEMENTS ? (size_t)values + 1u
                                                               : (size_t)MAX_ELEMENTS * MAX_ELEMENTS;
  size_t at = 0;
  L->stack = at;
  at += up8(bcir_jer_stack_bytes(&lim));
  L->scratch = at;
  at += up8((size_t)lim.string_bytes);
  L->nodes = at;
  at += up8((size_t)L->node_cap * sizeof(node));
  L->strings = at;
  at += up8(len + 1u);
  L->banks = at;
  at += up8((size_t)MAX_ELEMENTS * sizeof(bcir_devman_bank));
  L->rows = at;
  at += up8((size_t)(MAX_ELEMENTS + 1u) * sizeof(uint32_t));
  L->values = at;
  at += up8(L->value_cap * sizeof(int64_t));
  L->total = at + 8u; /* the workspace's own alignment */
  return 1;
}

size_t bcir_devman_workspace_size(size_t len) {
  bcir_jer_limits lim;
  layout L;
  bcir_devman_limits(&lim);
  /* the most values an envelope of `len` octets can hold within the scan's bound: one an octet */
  uint64_t values = (uint64_t)len < lim.nodes ? (uint64_t)len : lim.nodes;
  return plan(len, values, &L) ? L.total : 0;
}

/* ---- the canonical JSON the oracle's digest takes ------------------------------------------------------ */

static void put(bcir_sha256 *h, const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  bcir_sha256_update(h, (const uint8_t *)s, n);
}

static void put_int(bcir_sha256 *h, int64_t v) {
  char buf[24];
  size_t n = sizeof buf;
  uint64_t u = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
  do {
    buf[--n] = (char)('0' + u % 10u);
    u /= 10u;
  } while (u);
  if (v < 0) buf[--n] = '-';
  bcir_sha256_update(h, (const uint8_t *)buf + n, sizeof buf - n);
}

static void put_u16(bcir_sha256 *h, uint32_t u) {
  static const char hex[] = "0123456789abcdef";
  uint8_t e[6] = {'\\', 'u', (uint8_t)hex[(u >> 12) & 15u], (uint8_t)hex[(u >> 8) & 15u], (uint8_t)hex[(u >> 4) & 15u],
                  (uint8_t)hex[u & 15u]};
  bcir_sha256_update(h, e, 6);
}

/* json.dumps's string, ASCII-escaped as Python escapes it: the two quotes and the five short escapes, every other
 * code point outside ' '..'~' as \uXXXX in lowercase hex, an astral one as its surrogate pair */
static void put_str(bcir_sha256 *h, const uint8_t *s, size_t n) {
  bcir_sha256_update(h, (const uint8_t *)"\"", 1);
  size_t k = 0;
  while (k < n) {
    uint32_t cp;
    size_t w = 1;
    if (bcir_jer_utf8_next(s, n, k, &cp, &w) != BCIR_JER_OK) { /* the reader admitted UTF-8 only */
      cp = s[k];
      w = 1;
    }
    k += w;
    const char *esc = cp == '"' ? "\\\"" : cp == '\\' ? "\\\\" : cp == '\n' ? "\\n" : cp == '\r' ? "\\r"
                      : cp == '\t' ? "\\t" : cp == '\b' ? "\\b" : cp == '\f' ? "\\f" : 0;
    if (esc) {
      put(h, esc);
    } else if (cp >= 0x20u && cp <= 0x7Eu) {
      uint8_t c = (uint8_t)cp;
      bcir_sha256_update(h, &c, 1);
    } else if (cp > 0xFFFFu) {
      put_u16(h, 0xD800u + ((cp - 0x10000u) >> 10));
      put_u16(h, 0xDC00u + ((cp - 0x10000u) & 0x3FFu));
    } else {
      put_u16(h, cp);
    }
  }
  bcir_sha256_update(h, (const uint8_t *)"\"", 1);
}

/* sha256(json.dumps({"device", "target", "cal_gen", "banks", "distance"}, sort_keys=True)) */
static void digest(const bcir_devman *m, uint8_t out[32]) {
  bcir_sha256 h;
  bcir_sha256_init(&h);
  put(&h, "{\"banks\": [");
  for (uint32_t i = 0; i < m->n_banks; i++) {
    const bcir_devman_bank *b = &m->banks[i];
    put(&h, i ? ", [" : "[");
    put_str(&h, b->name, b->name_len);
    put(&h, ", ");
    put_int(&h, (int64_t)b->domain);
    put(&h, ", ");
    put_int(&h, b->capacity);
    put(&h, ", ");
    put_int(&h, b->native_tile);
    put(&h, ", ");
    put_int(&h, b->align);
    put(&h, "]");
  }
  put(&h, "], \"cal_gen\": ");
  put_int(&h, m->cal_gen);
  put(&h, ", \"device\": ");
  put_str(&h, m->device, m->device_len);
  put(&h, ", \"distance\": [");
  for (uint32_t r = 0; r < m->n_rows; r++) {
    put(&h, r ? ", [" : "[");
    for (uint32_t k = m->row_off[r]; k < m->row_off[r + 1u]; k++) {
      if (k > m->row_off[r]) put(&h, ", ");
      put_int(&h, m->distance[k]);
    }
    put(&h, "]");
  }
  put(&h, "], \"target\": ");
  put_str(&h, m->target, m->target_len);
  put(&h, "}");
  bcir_sha256_final(&h, out);
}

/* ---- check_device_manifest --------------------------------------------------------------------------- */

static unsigned consistency(const bcir_devman *m) {
  unsigned bits = 0;
  uint32_t n = m->n_banks;
  if (m->device_len == 0) bits |= BCIR_DEVMAN_BAD_DEVICE;
  if (n == 0) return bits | BCIR_DEVMAN_BAD_NO_BANKS;
  int empty = 0;
  for (uint32_t i = 0; i < n; i++)
    if (m->banks[i].name_len == 0) empty = 1;
  if (empty) {
    bits |= BCIR_DEVMAN_BAD_NAME;
  } else {
    for (uint32_t i = 0; i < n; i++)
      for (uint32_t j = i + 1u; j < n; j++)
        if (bytes_eq(m->banks[i].name, m->banks[i].name_len, m->banks[j].name, m->banks[j].name_len))
          bits |= BCIR_DEVMAN_BAD_DUPLICATE;
  }
  for (uint32_t i = 0; i < n; i++) {
    const bcir_devman_bank *b = &m->banks[i];
    if (b->capacity < 1) bits |= BCIR_DEVMAN_BAD_CAPACITY;
    if (b->native_tile < 1) bits |= BCIR_DEVMAN_BAD_TILE;
    if (b->align < 1) bits |= BCIR_DEVMAN_BAD_ALIGN;
    else if (b->align & (b->align - 1)) bits |= BCIR_DEVMAN_BAD_ALIGN_POW2;
  }
  int square = m->n_rows == n;
  for (uint32_t r = 0; square && r < n; r++)
    if (m->row_off[r + 1u] - m->row_off[r] != n) square = 0;
  if (!square) return bits | BCIR_DEVMAN_BAD_SHAPE;
  const int64_t *d = m->distance;
  for (uint32_t i = 0; i < n; i++) {
    if (d[(size_t)i * n + i] != 0) bits |= BCIR_DEVMAN_BAD_DIAGONAL;
    for (uint32_t j = 0; j < n; j++) {
      if (i != j && d[(size_t)i * n + j] < 1) bits |= BCIR_DEVMAN_BAD_DISTANCE;
      if (d[(size_t)i * n + j] != d[(size_t)j * n + i]) bits |= BCIR_DEVMAN_BAD_ASYMMETRIC;
    }
  }
  return bits;
}

/* ---- load ------------------------------------------------------------------------------------------- */

static const node *member(const builder *b, const node *obj, const char *name) {
  size_t n = 0;
  while (name[n]) n++;
  for (uint32_t m = obj->first; m != NONE; m = b->nodes[m].next)
    if (bytes_eq(b->strings + b->nodes[m].str_off, b->nodes[m].str_len, (const uint8_t *)name, n))
      return b->nodes[m].first == NONE ? 0 : &b->nodes[b->nodes[m].first];
  return 0;
}

static int is_str(const builder *b, const node *x, const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  return x && x->kind == N_STR && bytes_eq(b->strings + x->str_off, x->str_len, (const uint8_t *)s, n);
}

static const node *child(const builder *b, const node *x, uint32_t k) {
  uint32_t c = x->first;
  while (k-- && c != NONE) c = b->nodes[c].next;
  return c == NONE ? 0 : &b->nodes[c];
}

static const char *const fields[8] = {"kind", "schema", "device", "target", "cal_gen", "banks", "distance", "digest"};

int bcir_devman_load(const uint8_t *data, size_t len, const uint8_t *expect_target, size_t expect_target_len,
                     int has_expect_cal_gen, uint64_t expect_cal_gen, void *work, size_t work_len, bcir_devman *out,
                     uint32_t *where, unsigned *bits, bcir_jer_diag *diag) {
  if (where) *where = 0;
  if (bits) *bits = 0;
  if (diag) {
    diag->status = BCIR_JER_OK;
    diag->offset = BCIR_JER_NO_OFFSET;
    diag->needed = 0;
    diag->sink_code = 0;
  }
  if ((!data && len) || !out || !work || (!expect_target && expect_target_len)) return BCIR_DEVMAN_ERR_ARG;
  if (len > BCIR_DEVMAN_MAX_BYTES) return BCIR_DEVMAN_ERR_SIZE;
  layout L;
  if (!plan(len, 0, &L) || work_len < L.scratch + 8u) return BCIR_DEVMAN_ERR_SPACE; /* the reader's stack */
  uint8_t *w = (uint8_t *)work;
  w += (8u - ((uintptr_t)w & 7u)) & 7u;

  /* the bounded reader: the scan, UTF-8, then -- in a workspace sized by what the scan counted -- the grammar
   * building the nodes */
  bcir_jer_limits lim;
  bcir_jer_diag jd;
  uint64_t scanned = 0;
  bcir_devman_limits(&lim);
  bcir_jer_level *stack = (bcir_jer_level *)(void *)(w + L.stack);
  bcir_jer_status js = bcir_jer_scan(data, len, &lim, stack, lim.depth, &scanned, &jd);
  if (js == BCIR_JER_OK) js = bcir_jer_validate_utf8(data, len, &jd);
  if (js != BCIR_JER_OK) {
    if (diag) *diag = jd;
    return BCIR_DEVMAN_ERR_READ;
  }
  if (!plan(len, scanned, &L) || work_len < L.total) return BCIR_DEVMAN_ERR_SPACE;
  builder b;
  b.nodes = (node *)(void *)(w + L.nodes);
  b.n = 0;
  b.cap = L.node_cap;
  b.strings = w + L.strings;
  b.used = 0;
  b.strings_cap = len + 1u;
  b.depth = 0;
  b.root = NONE;
  b.space = 0;
  js = bcir_jer_parse(data, len, &lim, stack, lim.depth, w + L.scratch, (size_t)lim.string_bytes, sink, &b, &jd);
  if (js != BCIR_JER_OK) {
    if (diag) *diag = jd;
    return js == BCIR_JER_SINK_REFUSED && jd.sink_code == SINK_SPACE ? BCIR_DEVMAN_ERR_SPACE : BCIR_DEVMAN_ERR_READ;
  }
  const node *root = b.root == NONE ? 0 : &b.nodes[b.root];
  if (!root || root->kind != N_OBJ) return BCIR_DEVMAN_ERR_ROOT;
  if (!is_str(&b, member(&b, root, "kind"), "bcir.device_manifest")) return BCIR_DEVMAN_ERR_KIND;
  const node *schema = member(&b, root, "schema");
  if (!schema || schema->kind != N_INT) return BCIR_DEVMAN_ERR_SCHEMA_TYPE;
  if (schema->ival > (int64_t)BCIR_DEVMAN_SCHEMA) return BCIR_DEVMAN_ERR_SCHEMA_NEWER;
  if (schema->ival != (int64_t)BCIR_DEVMAN_SCHEMA) return BCIR_DEVMAN_ERR_SCHEMA;
  int all = root->count == 8u;
  for (unsigned f = 0; all && f < 8u; f++)
    if (!member(&b, root, fields[f])) all = 0;
  if (!all) return BCIR_DEVMAN_ERR_FIELDS;
  const node *device = member(&b, root, "device"), *target = member(&b, root, "target");
  const node *cal = member(&b, root, "cal_gen"), *banks = member(&b, root, "banks");
  const node *dist = member(&b, root, "distance"), *dg = member(&b, root, "digest");
  if (expect_target_len &&
      !(target->kind == N_STR && bytes_eq(b.strings + target->str_off, target->str_len, expect_target,
                                          expect_target_len)))
    return BCIR_DEVMAN_ERR_TARGET;
  if (has_expect_cal_gen && cal->kind == N_INT && (cal->ival < 0 || (uint64_t)cal->ival != expect_cal_gen))
    return BCIR_DEVMAN_ERR_STALE;
  if (device->kind != N_STR || target->kind != N_STR || cal->kind != N_INT || cal->ival < 0 ||
      banks->kind != N_ARR || dist->kind != N_ARR)
    return BCIR_DEVMAN_ERR_TYPES;

  bcir_devman m;
  bcir_devman_bank *bk = (bcir_devman_bank *)(void *)(w + L.banks);
  uint32_t *row_off = (uint32_t *)(void *)(w + L.rows);
  int64_t *values = (int64_t *)(void *)(w + L.values);
  m.device = b.strings + device->str_off;
  m.device_len = device->str_len;
  m.target = b.strings + target->str_off;
  m.target_len = target->str_len;
  m.cal_gen = cal->ival;
  m.n_banks = 0;
  for (uint32_t i = 0; i < banks->count; i++) {
    const node *x = child(&b, banks, i);
    if (where) *where = i;
    if (!x || x->kind != N_ARR || x->count != 5u) return BCIR_DEVMAN_ERR_BANK_SHAPE;
    const node *nm = child(&b, x, 0);
    for (uint32_t k = 1; k < 5u; k++)
      if (child(&b, x, k)->kind != N_INT) return BCIR_DEVMAN_ERR_BANK_TYPES;
    if (nm->kind != N_STR) return BCIR_DEVMAN_ERR_BANK_TYPES;
    int64_t domain = child(&b, x, 1)->ival;
    if (domain < 0 || domain >= (int64_t)BCIR_DEVMAN_DOMAINS) return BCIR_DEVMAN_ERR_DOMAIN;
    if (i >= MAX_ELEMENTS) return BCIR_DEVMAN_ERR_SPACE; /* the reader's element bound keeps it below */
    bk[i].name = b.strings + nm->str_off;
    bk[i].name_len = nm->str_len;
    bk[i].domain = (uint32_t)domain;
    bk[i].capacity = child(&b, x, 2)->ival;
    bk[i].native_tile = child(&b, x, 3)->ival;
    bk[i].align = child(&b, x, 4)->ival;
    m.n_banks = i + 1u;
  }
  size_t nv = 0;
  row_off[0] = 0;
  for (uint32_t r = 0; r < dist->count; r++) {
    const node *row = child(&b, dist, r);
    if (where) *where = r;
    if (!row || row->kind != N_ARR) return BCIR_DEVMAN_ERR_DISTANCE;
    for (uint32_t c = row->first; c != NONE; c = b.nodes[c].next) {
      if (b.nodes[c].kind != N_INT) return BCIR_DEVMAN_ERR_DISTANCE;
      if (nv == L.value_cap || r >= MAX_ELEMENTS) return BCIR_DEVMAN_ERR_SPACE;
      values[nv++] = b.nodes[c].ival;
    }
    row_off[r + 1u] = (uint32_t)nv;
  }
  if (where) *where = 0;
  m.n_rows = dist->count;
  m.banks = bk;
  m.row_off = row_off;
  m.distance = values;
  if (dg->kind != N_STR || dg->str_len != 64u) return BCIR_DEVMAN_ERR_DIGEST_FORM;
  const uint8_t *hex = b.strings + dg->str_off;
  for (unsigned k = 0; k < 64u; k++)
    if (!((hex[k] >= '0' && hex[k] <= '9') || (hex[k] >= 'a' && hex[k] <= 'f'))) return BCIR_DEVMAN_ERR_DIGEST_FORM;
  digest(&m, m.digest);
  static const char nib[] = "0123456789abcdef";
  for (unsigned k = 0; k < 32u; k++)
    if (hex[2u * k] != (uint8_t)nib[m.digest[k] >> 4] || hex[2u * k + 1u] != (uint8_t)nib[m.digest[k] & 15u]) {
      for (unsigned j = 0; j < 32u; j++) out->digest[j] = m.digest[j]; /* what the contents re-derive */
      return BCIR_DEVMAN_ERR_TAMPERED;
    }
  unsigned bad = consistency(&m);
  if (bad) {
    if (bits) *bits = bad;
    return BCIR_DEVMAN_ERR_INVALID;
  }
  *out = m;
  return BCIR_DEVMAN_OK;
}

long bcir_devman_bank_index(const bcir_devman *m, const uint8_t *name, size_t name_len) {
  if (!m || (!name && name_len)) return -1;
  for (uint32_t i = 0; i < m->n_banks; i++)
    if (bytes_eq(m->banks[i].name, m->banks[i].name_len, name, name_len)) return (long)i;
  return -1;
}

int bcir_devman_dma_banks(const bcir_devman *m, bcir_dma_bank *out, size_t cap) {
  if (!m || (!out && cap)) return BCIR_DEVMAN_ERR_ARG;
  if (cap < m->n_banks) return BCIR_DEVMAN_ERR_SPACE;
  for (uint32_t i = 0; i < m->n_banks; i++)
    if ((uint64_t)m->banks[i].native_tile > UINT32_MAX || (uint64_t)m->banks[i].align > UINT32_MAX)
      return BCIR_DEVMAN_ERR_WIDTH;
  for (uint32_t i = 0; i < m->n_banks; i++) { /* a loaded manifest's fields are >= 1 */
    out[i].capacity = (uint64_t)m->banks[i].capacity;
    out[i].native_tile = (uint32_t)m->banks[i].native_tile;
    out[i].align = (uint32_t)m->banks[i].align;
  }
  return BCIR_DEVMAN_OK;
}

unsigned bcir_devman_probe_agree(const bcir_devman *m, const bcir_devman_probe *obs, size_t n, size_t *messages) {
  unsigned bits = 0;
  size_t count = 0;
  for (size_t k = 0; m && obs && k < n; k++) {
    long i = bcir_devman_bank_index(m, obs[k].name, obs[k].name_len);
    if (i < 0) { /* discovery may veto, never steer */
      bits |= BCIR_DEVMAN_PROBE_UNKNOWN;
      count++;
      continue;
    }
    if (obs[k].has_capacity && obs[k].capacity != (uint64_t)m->banks[i].capacity) {
      bits |= BCIR_DEVMAN_PROBE_CAPACITY;
      count++;
    }
    if (obs[k].has_tile && obs[k].native_tile != (uint64_t)m->banks[i].native_tile) {
      bits |= BCIR_DEVMAN_PROBE_TILE;
      count++;
    }
  }
  if (messages) *messages = count;
  return bits;
}
