/* CSUB-2c: a struct or union member of an array element -- the record table's struct members -- behaviour-checked on
 * both rails. `set->slots[j].name = name` (`bcir_bpe.c`) and `d->notes[i].span` passed by value (`bcir_diag.c`): the
 * member read as the aggregate value, stored, copied element to element, passed and returned by value, and its address
 * taken, on a direct array, a member array and through a loaded pointer. The storage is block statics, set at each
 * call: the harness runs the original and then the emit, and the escape corpus holds every local array private. */
#include <stdint.h>
#include <stddef.h>

struct str {
  const char *s;
  size_t n;
};
struct span {
  uint32_t lo, hi;
};
union word {
  struct span sp;
  uint64_t w;
};
struct slot {
  uint32_t id;
  struct str name;
  struct span sp;
};
struct set {
  struct slot *slots;
  size_t n;
};
struct note {
  uint32_t code;
  struct span span;
};
struct diag {
  struct note notes[3];
  uint32_t n;
};
struct cell {
  union word u;
  struct {
    struct span inner;
  } in;
};

static uint32_t width(struct span s) { return s.hi - s.lo; }
static uint32_t hash(struct str s) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < s.n; i++) h = (h ^ (uint32_t)(uint8_t)s.s[i]) * 16777619u;
  return h;
}

/* bcir_bpe.c's put: a struct member of an element stored through a loaded pointer, and read back by value */
static uint32_t slot_put(struct set *set, struct str name, uint32_t id) {
  size_t j = set->n++;
  set->slots[j].id = id;
  set->slots[j].name = name;
  struct str back = set->slots[j].name;
  return hash(back) + (uint32_t)set->slots[j].name.n;
}

/* bcir_diag.c's render: a struct member of a member array's element passed by value */
static uint32_t notes_width(const struct diag *d) {
  uint32_t w = 0;
  for (uint32_t i = 0; i < d->n && i < 3u; i++) w = w * 7u + width(d->notes[i].span) + d->notes[i].code;
  return w;
}

/* a struct member returned by value, and its address */
static struct span span_of(const struct slot *t, size_t i) { return t[i].sp; }
static uint32_t bump(struct slot *t, size_t i) {
  struct span *p = &t[i].sp;
  p->lo += 2u;
  p->hi += 5u;
  return width(t[i].sp);
}

/* copied element to element, on a member array and a direct one; a union member and a nested struct's */
static uint32_t copies(struct diag *d, struct cell *c, uint32_t k) {
  d->notes[1].span = d->notes[0].span;
  d->notes[2].span.hi += k;
  c[1].u = c[0].u;
  c[0].in.inner = c[1].u.sp;
  union word x = c[1].u;
  struct span *q = &c[0].in.inner;
  return width(d->notes[1].span) + width(d->notes[2].span) + x.sp.lo + q->hi;
}

uint32_t csub2_structmembers(uint32_t x, uint32_t y) {
  static struct slot slots[4];
  static struct set set;
  static struct diag d;
  static struct cell cells[2];
  set.slots = slots;
  set.n = 0;
  struct str a = {"merge", 5u};
  struct str b = {"rank", 4u};
  uint32_t r = slot_put(&set, a, x) * 3u + slot_put(&set, b, y) * 5u;
  for (uint32_t i = 0; i < 2u; i++) {
    slots[i].sp.lo = x + i;
    slots[i].sp.hi = x + i * 9u + 20u;
  }
  d.n = 3u;
  for (uint32_t i = 0; i < 3u; i++) {
    d.notes[i].code = i + 1u;
    d.notes[i].span.lo = y + i;
    d.notes[i].span.hi = y + 3u * i + 4u;
  }
  cells[0].u.sp.lo = x;
  cells[0].u.sp.hi = y;
  struct span s1 = span_of(slots, 1u);
  r += notes_width(&d) * 7u + bump(slots, y & 1u) * 11u + width(s1) * 13u + copies(&d, cells, x) * 17u;
  return r + slots[0].sp.lo + slots[1].sp.hi;
}
