/* CSUB-2b: an untagged struct or union a block declaration defines, behaviour-checked on both rails.
 * `static const struct { uint16_t bit; const char *name; } order[] = {...}` is how the runtime's units write their
 * small tables (`bcir_ab_listing.c`'s flag names, `bcir_emit.c`'s plan kinds, `bcir_channel.c`'s channel bits): the
 * declaration defines the type and declares its objects, and nothing else can name it (C11 6.7.2.3p6). Each rail's
 * emit spells it once, as a typedef on the function's first line, `typedef struct { ... } __bcir_anonN;` -- an
 * identifier reserved to the implementation -- and every object, temp and copy of the type by that name. */
#include <stdint.h>
#include <stddef.h>

/* the runtime's shape: a static table of codes, scanned for the ones a mask holds */
uint32_t csub2_flags(uint32_t f) {
  static const struct {
    uint16_t bit;
    uint16_t weight;
  } order[] = {{1u, 3u}, {2u, 5u}, {4u, 7u}, {8u, 11u}};
  uint32_t h = 0;
  for (unsigned i = 0; i < sizeof order / sizeof order[0]; i++) {
    if (!(f & order[i].bit)) continue;
    h = h * 31u + order[i].weight;
  }
  return h;
}

/* an automatic table, initialized at each call, its records read by index */
uint32_t csub2_auto_table(uint32_t k) {
  struct {
    uint32_t code;
    uint8_t len;
  } t[] = {{0x10u, 1u}, {0x20u, 2u}, {0x30u, 3u}};
  uint32_t s = 0;
  for (unsigned i = 0; i < 3u; i++) s += t[i].code * t[i].len;
  return s + k;
}

/* a union, a nested untagged member, several declarators off one definition, a pointer to one */
uint32_t csub2_shapes(uint32_t k) {
  union {
    uint32_t w;
    uint8_t b[4];
  } u;
  u.w = k * 2654435761u;
  struct {
    struct {
      uint32_t a;
    } in;
    uint32_t b;
  } n = {{k}, k + 1u};
  struct {
    uint32_t a;
  } x, *px = &x, arr[2];
  x.a = k;
  arr[0].a = k + 1u;
  arr[1].a = 3u;
  return u.b[0] + u.b[3] + n.in.a * 2u + n.b + px->a + arr[0].a + arr[1].a;
}

/* copies of the type -- by `typeof`, by assignment -- bit-fields, an array member, a string member */
uint32_t csub2_copies(uint32_t k) {
  struct {
    uint32_t a, b;
  } x = {k, 2u};
  __typeof__(x) y = x;
  y.a += 1u;
  struct {
    uint32_t lo : 4, hi : 12;
    uint8_t t[3];
    const char *s;
  } z = {k & 15u, (k >> 4) & 4095u, {1, 2, 3}, "zz"}, w;
  w = z;
  w.t[2] ^= 5u;
  return y.a + x.a + y.b + w.lo + w.hi + w.t[2] + z.t[1] + (uint32_t)(uint8_t)w.s[0];
}

/* two functions' tables of their own, a static of strings, a member of the unit's typedef */
typedef struct {
  uint32_t a;
} pt_t;
uint32_t csub2_two_tables(uint32_t k) {
  static const struct {
    pt_t p;
    uint32_t q;
  } t[] = {{{1u}, 2u}, {{3u}, 4u}};
  static const struct {
    const char *ok, *bad;
  } words = {"ok", "bad"};
  for (struct {
         uint32_t i, n;
       } s = {0u, k & 7u};
       s.i < s.n; s.i++)
    k += s.i;
  return t[k & 1u].q + t[k & 1u].p.a + k + (uint32_t)(uint8_t)words.ok[1] + (uint32_t)(uint8_t)words.bad[2];
}

/* The entry the harness drives: every function above, folded. */
uint32_t csub2_untagged(uint32_t x, uint32_t y) {
  return csub2_flags(x) * 3u + csub2_auto_table(y) * 5u + csub2_shapes(x ^ y) * 7u + csub2_copies(y) * 11u +
         csub2_two_tables(x) * 13u;
}
