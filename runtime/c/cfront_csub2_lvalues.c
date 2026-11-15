/* CSUB-2e: an element of a pointer to structs reached through a loaded pointer -- the lvalue `p->a[i]` and its
 * members -- behaviour-checked on both rails. `&set->slots[j]` (`bcir_bpe.c`), `&d->fixits[i]` (`bcir_diag.c`),
 * `&b->t.constraints[n]` (`bcir_emit.c`), `&c->claims[v->claim_id]` (`bcir_decoder_pack.c`) and `&t->slots[i]`
 * (`bcir_handoff.c`): the element's address; then a member of it stepped, assigned as a value, compound-assigned,
 * addressed, and stored or stepped through a pointer member it holds. The storage is block statics, set at each call:
 * the harness runs the original and then the emit, and the escape corpus holds every local array private. */
#include <stdint.h>
#include <stddef.h>

struct slot {
  uint32_t id;
  int16_t h;
  const char *name;
  struct slot *next;
  uint8_t *buf;
};
union word {
  uint32_t w;
  uint8_t b[4];
};
struct set {
  struct slot *slots;
  size_t n;
  struct {
    struct slot *rows;
    uint32_t count;
  } t;
  union word *words;
};

static uint32_t mix(const char *s) {
  uint32_t h = 2166136261u;
  for (uint32_t i = 0; s[i]; i++) h = (h ^ (uint32_t)(uint8_t)s[i]) * 16777619u;
  return h;
}

/* the element's address, through a member pointer and a nested member's, compared and differenced */
static uint32_t find(struct set *s, uint32_t want) {
  for (size_t j = 0; j < s->n; j++) {
    const struct slot *e = &s->slots[j];
    if (e->id == want) return (uint32_t)(e - s->slots) + 1u;
  }
  struct slot *last = &s->t.rows[s->t.count - 1u];
  return last == &s->t.rows[0] ? 0u : last->id;
}

/* a member of an element stepped, assigned as a value and compound-assigned as one */
static uint32_t bump(struct set *s, size_t j, uint32_t v) {
  s->slots[j].id++;
  uint32_t a = ++s->slots[j].id;
  uint32_t b = s->slots[j].id--;
  uint32_t c = (s->slots[j].id = a + v);
  uint32_t d = (s->slots[j].id += v);
  s->slots[j].h -= 7;
  int16_t e = --s->slots[j].h;
  return a + b * 3u + c * 5u + d * 7u + (uint32_t)(int32_t)e;
}

/* a member's address, and a store or a step through a pointer member of an element */
static uint32_t through(struct set *s, size_t j, uint8_t v) {
  uint32_t *p = &s->slots[j].id;
  *p += 11u;
  s->slots[j].buf[1] = v;
  s->slots[j].buf[0] += v;
  s->slots[j].next->id = *p + 1u;
  s->slots[j].next->next->id += 3u;
  s->slots[j].next->h++;
  uint8_t *q = &s->slots[j].buf[2];
  *q ^= 0x5au;
  struct slot *n = &s->slots[j].next[0];
  return *p + s->slots[j].buf[0] + s->slots[j].buf[1] + *q + n->id + (uint32_t)(int32_t)s->slots[j].next->h +
         mix(s->slots[j].name);
}

/* a union element's address and a member's */
static uint32_t words(struct set *s, size_t j) {
  union word *u = &s->words[j];
  uint32_t *w = &s->words[j].w;
  *w = 0x01020304u + (uint32_t)j;
  return u->b[0] + u->b[3] + ++s->words[j].w;
}

uint32_t csub2_lvalues(uint32_t x, uint32_t y) {
  static struct slot slots[3];
  static struct slot rows[2];
  static union word wd[2];
  static uint8_t b0[4], b1[4], b2[4];
  static struct set s;
  for (uint32_t k = 0; k < 4u; k++) {
    b0[k] = (uint8_t)(x + k);
    b1[k] = (uint8_t)(y + k);
    b2[k] = (uint8_t)(x ^ k);
  }
  slots[0].buf = b0;
  slots[1].buf = b1;
  slots[2].buf = b2;
  for (uint32_t i = 0; i < 3u; i++) {
    slots[i].id = x + i * 5u;
    slots[i].h = (int16_t)(y & 0x7fu);
    slots[i].next = &slots[(i + 1u) % 3u];
  }
  slots[0].name = "alpha";
  slots[1].name = "beta";
  slots[2].name = "gamma";
  rows[0].id = y;
  rows[1].id = y ^ 3u;
  s.slots = slots;
  s.n = 3u;
  s.t.rows = rows;
  s.t.count = 2u;
  s.words = wd;
  uint32_t r = find(&s, x + 5u) * 3u + find(&s, y + 1000u) * 5u;
  r += bump(&s, y % 3u, x) * 7u;
  r += through(&s, x % 3u, (uint8_t)y) * 11u;
  r += words(&s, y & 1u) * 13u;
  return r + slots[0].id + slots[1].id + slots[2].id;
}
