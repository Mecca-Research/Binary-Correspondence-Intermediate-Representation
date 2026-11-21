/* CSUB-2c: a pointer member of an array element -- the record table's first need -- behaviour-checked on both rails.
 * `order[i].name` (`bcir_ab_listing.c`), `table[i].name` (`bcir_emit.c`), `d->notes[i].message` (`bcir_diag.c`),
 * `arts[i].name` (`bcir_provenance.c`), `set->slots[i].name.s` (`bcir_bpe.c`) and `pre[0].a = panel`
 * (`bcir_decoder_train.c`): a member of type pointer, at any depth of the element's nested members, read as a value
 * of its pointer type -- compared, passed, subscripted, dereferenced, followed with `->` -- and stored, or stored
 * through. Each rail loads it as an element member is loaded, strided by the element, and walks on through the
 * pointer it holds as a plain struct's pointer member walks on. The pointed-to buffers are block statics, set at each
 * call: the harness runs the original and then the emit, and the escape corpus holds every local array private.
 * Stepping through such a member (`t[i].next->v++`) is lowered; stepping the member itself (`t[i].p++`) is refused
 * alike, as a plain struct's pointer member's step is; the address of the member is a `T **`, of what it reaches a
 * `T *`. */
#include <stdint.h>
#include <stddef.h>

static uint32_t mix(const char *s) {
  uint32_t h = 2166136261u;
  for (uint32_t i = 0; s[i]; i++) h = (h ^ (uint32_t)(uint8_t)s[i]) * 16777619u;
  return h;
}

/* the runtime's shape: a static table of codes and names, scanned for the ones a mask holds */
uint32_t csub2_flag_names(uint32_t f) {
  static const struct {
    uint16_t bit;
    const char *name;
  } order[] = {{1u, "exec"}, {2u, "r12"}, {4u, "portable"}, {8u, "debug"}};
  uint32_t h = 0;
  for (unsigned i = 0; i < sizeof order / sizeof order[0]; i++) {
    if (!(f & order[i].bit)) continue;
    h = h * 33u + mix(order[i].name) + (uint32_t)(uint8_t)order[i].name[1];
  }
  return h;
}

/* a member array of records reached through a pointer, its pointer member null-checked and subscripted */
struct note {
  uint32_t code;
  const char *message;
};
struct diag {
  struct note notes[4];
  uint32_t n;
};
static uint32_t diag_sum(const struct diag *d) {
  uint32_t s = 0;
  for (uint32_t i = 0; i < d->n && i < 4u; i++) {
    if (!d->notes[i].message) return s + 1000u;
    s += d->notes[i].code * (uint32_t)(uint8_t)d->notes[i].message[0] + mix(d->notes[i].message);
  }
  return s;
}
uint32_t csub2_notes(uint32_t k) {
  static struct diag d;
  d.n = k % 5u;
  d.notes[0].code = 1u;
  d.notes[0].message = "first";
  d.notes[1].code = 2u;
  d.notes[1].message = "second";
  d.notes[2].code = 3u;
  d.notes[2].message = (k & 1u) ? "third" : 0;
  d.notes[3].code = 4u;
  d.notes[3].message = "fourth";
  return diag_sum(&d);
}

/* a pointer member of a nested member, `slots[i].name.s`; a pointer passed on; one stored, then stored through */
struct str {
  const char *s;
  size_t n;
};
struct slot {
  struct str name;
  uint32_t id;
  uint32_t *cell;
};
static uint32_t span(const char *s, size_t n) {
  uint32_t h = 7u;
  for (size_t i = 0; i < n; i++) h = h * 31u + (uint8_t)s[i];
  return h;
}
uint32_t csub2_slots(uint32_t k) {
  static struct slot slots[3];
  static uint32_t cells[6];
  for (uint32_t i = 0; i < 6u; i++) cells[i] = k + i;
  slots[0].name.s = "alpha";
  slots[0].name.n = 5u;
  slots[1].name.s = 0;
  slots[1].name.n = 0u;
  slots[2].name.s = "gamma";
  slots[2].name.n = 3u;
  uint32_t h = 0;
  for (uint32_t i = 0; i < 3u; i++) {
    slots[i].id = i * 7u;
    slots[i].cell = &cells[2u * i];
    slots[i].cell[1] = slots[i].id + k;   /* a store through a subscript of the member */
    *slots[i].cell += 3u;                 /* ... through `*` of it */
    ++slots[i].cell[0];                   /* ... a step through it */
    if (!slots[i].name.s) continue;
    h += (uint32_t)(uint8_t)slots[i].name.s[0] + span(slots[i].name.s, slots[i].name.n);
  }
  uint32_t **pc = &slots[1].cell;        /* the member's own address: a `T **` */
  uint32_t *last = &slots[2].cell[1];     /* ... and that of an object it reaches */
  **pc += 2u;
  *last ^= 1u;
  static struct str one;
  one.s = "omega";
  one.n = 5u;
  const struct str *nm = &one;
  const char *const *ps = &nm->s;         /* a plain struct's `const char *` member's address (a `const char **`) */
  return h + cells[0] + cells[1] + cells[3] + cells[5] + *slots[2].cell + (uint32_t)(uint8_t)(*ps)[2];
}

/* records that link to records: `t[i].next->v` read and stored through */
struct link {
  struct link *next;
  uint32_t v;
};
uint32_t csub2_links(uint32_t k) {
  static struct link t[4];
  for (uint32_t i = 0; i < 4u; i++) {
    t[i].v = k * (i + 1u);
    t[i].next = i < 3u ? &t[i + 1u] : 0;
  }
  t[0].next->v += 5u;
  t[1].next->v++;
  uint32_t *pv = &t[2].next->v;
  *pv *= 3u;
  uint32_t s = 0;
  for (uint32_t i = 0; i < 4u; i++) s = s * 3u + (t[i].next ? t[i].next->v : t[i].v);
  return s;
}

/* a panel of pointer pairs, stored as `pre[0].a = panel` writes them, and read back through */
struct pnl {
  const uint32_t *a;
  const uint32_t *b;
};
uint32_t csub2_panels(uint32_t k) {
  static uint32_t panel[4];
  static struct pnl pre[2];
  for (uint32_t i = 0; i < 4u; i++) panel[i] = k ^ (i * 2654435761u);
  pre[0].a = panel;
  pre[0].b = panel + 1;
  pre[1].a = pre[0].b;
  pre[1].b = 0;
  return pre[0].a[0] + pre[0].b[2] + pre[1].a[1] + (pre[1].b == 0);
}

/* a member array of records holding pointers, `s.b[i].cell`: stored, stored and stepped through, and their addresses
 * taken -- and a `T **` member's, a `T ***` */
struct bucket {
  uint32_t *cell;
  struct link *head;
  const char *name;
};
struct table {
  struct bucket b[2];
  uint32_t n;
};
struct deep {
  uint32_t **pp;
};
uint32_t csub2_buckets(uint32_t k) {
  static uint32_t cells[4];
  static struct link nodes[2];
  static struct table tb;
  static uint32_t *rows[2];
  static struct deep dp;
  for (uint32_t i = 0; i < 4u; i++) cells[i] = k + i;
  nodes[0].v = k;
  nodes[1].v = k * 2u;
  nodes[0].next = &nodes[1];
  nodes[1].next = 0;
  tb.b[0].cell = &cells[0];
  tb.b[1].cell = &cells[2];
  tb.b[0].head = &nodes[0];
  tb.b[1].head = &nodes[1];
  tb.b[0].name = "north";
  tb.b[1].name = "south";
  tb.b[k & 1u].cell[1] = k * 3u;   /* a store through a subscript of the member */
  tb.b[0].head->v += 7u;           /* ... a compound store through `->` */
  tb.b[k & 1u].cell[0]++;          /* ... a step through it */
  --tb.b[1].head->v;
  uint32_t *q = &tb.b[1].cell[1];  /* the address of an object it reaches */
  uint32_t **pc = &tb.b[0].cell;   /* the member's own address, a `T **` */
  const char **pn = &tb.b[k & 1u].name;
  *q += 1u;
  rows[0] = &cells[3];
  rows[1] = &cells[1];
  dp.pp = rows;
  uint32_t ***ppp = &dp.pp;        /* a `T **` member's address, a `T ***` */
  const char *nm = *pn;
  return cells[0] + cells[1] + cells[2] + cells[3] + **pc + (uint32_t)(uint8_t)nm[0] + tb.b[0].head->next->v +
         nodes[0].v + ***ppp + *(*ppp)[1];
}

/* The entry the harness drives: every function above, folded. */
uint32_t csub2_records(uint32_t x, uint32_t y) {
  return csub2_flag_names(x) * 3u + csub2_notes(y) * 5u + csub2_slots(x ^ y) * 7u + csub2_links(y) * 11u +
         csub2_panels(x) * 13u + csub2_buckets(x + y) * 17u;
}
