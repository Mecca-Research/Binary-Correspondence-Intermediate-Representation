/* Anonymous structs named by a typedef (CF-ANON): `typedef struct { ... } T;` as a local, a parameter, a returned
 * value, a member of another struct, an array element, a pointer's target and an operand of `sizeof`; nested
 * anonymous members (`span`, `pair[2]`), an anonymous union, one only a pointer typedef names (`an_ref`), and
 * several anonymous structs in one unit. Each rail's emit names the type as C does -- the typedef's name,
 * `__typeof__(*(an_ref)0)`, or `__typeof__` of the member whose type it is -- where both spelled a tag no compiler
 * knows (the oracle `struct ` with no tag at all, the twin `struct $anon0`); the oracle also lost the layout of
 * every anonymous struct past the first. */
#include <stdint.h>

typedef struct { uint16_t x, y; } an_pt;
typedef struct { uint32_t id; uint8_t tag; } an_rec;
typedef union { uint32_t u; uint8_t b[4]; } an_word;
typedef struct { an_pt at; an_rec r[2]; struct { uint16_t lo, hi; } span; struct { uint8_t a, b; } pair[2]; uint8_t n; } an_box;
typedef struct { uint32_t k; uint16_t w; } *an_ref;   /* named only through a pointer */

static an_box an_g;

static an_pt an_make(uint32_t s) {
  an_pt p = {(uint16_t)s, (uint16_t)(s >> 16)};
  return p;
}
static uint32_t an_len(an_pt p) { return (uint32_t)p.x * 3u + p.y; }
static uint32_t an_bump(an_pt *p, uint32_t s) {
  p->x = (uint16_t)(p->x + s);
  return p->x ^ p->y;
}

uint32_t an_local(uint32_t s) {
  an_pt a = an_make(s), b;
  b = a;
  b.y = (uint16_t)(b.y + 7u);
  return an_len(a) + an_bump(&b, s) + (uint32_t)sizeof(an_pt) + (uint32_t)sizeof b;
}
uint32_t an_array(uint32_t s) {
  an_rec rs[3] = {{s, 1}, {s ^ 7u, 2}, {3u, 3}};
  an_rec *q = &rs[1];
  q->tag = (uint8_t)(q->tag + s);
  return rs[0].id + q->id + rs[1].tag + rs[2].tag + (uint32_t)sizeof rs;
}
uint32_t an_nested(uint32_t s) {
  an_box b = {{1, 2}, {{s, 4}, {5u, 6}}, {7, 8}, {{1, 2}, {3, (uint8_t)s}}, 9};
  an_box c;
  c = b;
  c.span = b.span;
  c.span.hi = (uint16_t)s;
  c.pair[0] = b.pair[1];
  c.at = an_make(s + 1u);
  return c.at.x + c.r[1].id + c.span.lo + c.span.hi + c.pair[0].b * 5u + c.pair[1].a + c.n + (uint32_t)sizeof c.span
         + (uint32_t)sizeof c.pair;
}
uint32_t an_union(uint32_t s) {
  an_word w;
  w.u = s;
  an_word v = w;
  return v.b[0] + v.b[3] * 7u + (uint32_t)sizeof(an_word);
}
uint32_t an_deref(an_ref r, uint32_t s) {
  r->k += s;
  return r->k ^ r->w;
}
uint32_t an_global(uint32_t s) {
  an_g.at.x = (uint16_t)s;
  an_g.span.lo = (uint16_t)(s >> 4);
  return an_g.at.x + an_g.span.lo + an_g.n;
}
uint32_t an_entry(uint32_t s) { return an_local(s) + an_array(s) + an_nested(s) + an_union(s); }
