/* Brace elision in aggregate initializers (C11 6.7.9p17-21, CF-BRACE): a list that opens no brace for a
 * sub-aggregate fills that sub-aggregate's members in order from the enclosing list; a designator sends the
 * walk back to the list's own object and the entries after it continue past the subobject it names; an
 * anonymous struct or union member is one subobject, entered like any other; a union takes one value, for
 * its first member; a string literal initializes a character array it meets; a struct-typed value
 * initializes its struct subobject whole; a braced scalar `{e}` is `e` and `{}` is zero. Both rails walk
 * the current object the same way, and each emit returns what the original does. */
#include <stdint.h>

struct pt { uint16_t x, y; };
struct seg { struct pt a, b; uint8_t tag; };
struct rec { uint8_t k; uint32_t v[3]; struct pt p; };
struct named { char name[6]; uint32_t id; };
struct anon { uint8_t lo; struct { uint16_t m, n; }; union { uint32_t u; uint8_t b[4]; }; };
union word { uint32_t w; uint8_t b[4]; };
struct holder { union word u; uint16_t t; };
struct grid { uint8_t cell[2][3]; uint8_t tail; };

uint32_t be_flat_2d(uint32_t s) {                   /* `T m[2][3] = {...}`: the rows fill in order */
  uint32_t m[2][3] = {s, s + 1u, s + 2u, s + 3u, s + 4u};
  return m[0][0] + m[0][2] * 3u + m[1][0] * 5u + m[1][1] * 7u + m[1][2] * 11u;
}
uint32_t be_struct_array(uint32_t s) {              /* an array of structs, their members elided */
  struct pt ps[3] = {(uint16_t)s, 2, 3, (uint16_t)(s >> 3), 5};
  return ps[0].x + ps[0].y * 3u + ps[1].x * 5u + ps[1].y * 7u + ps[2].x * 11u + ps[2].y * 13u;
}
uint32_t be_nested(uint32_t s) {                    /* two levels of struct elided */
  struct seg g = {1, (uint16_t)s, 3, 4, 9};
  return g.a.x + g.a.y * 3u + g.b.x * 5u + g.b.y * 7u + g.tag * 11u;
}
uint32_t be_mixed(uint32_t s) {                     /* braces on one member, elision into the next */
  struct rec r = {7, {s, s * 2u}, 5, 6};
  return r.k + r.v[0] * 3u + r.v[1] * 5u + r.v[2] * 7u + r.p.x * 11u + r.p.y * 13u;
}
uint32_t be_designated(uint32_t s) {                /* a designator, then the walk continues after it */
  struct rec r = {.v[1] = s, 9, .p.y = 3, .k = 4};
  return r.k + r.v[0] * 3u + r.v[1] * 5u + r.v[2] * 7u + r.p.x * 11u + r.p.y * 13u;
}
uint32_t be_array_designators(uint32_t s) {         /* `[i] =` into an array of structs, then elision */
  struct pt ps[4] = {[2] = {(uint16_t)s, 1}, 7, 8, [0].y = 5};
  return ps[0].x + ps[0].y * 3u + ps[1].x * 5u + ps[2].x * 7u + ps[2].y * 11u + ps[3].x * 13u + ps[3].y * 17u;
}
uint32_t be_member_rows(uint32_t s) {               /* a 2-D member array elided inside its struct */
  struct grid g[2] = {1, 2, 3, 4, 5, 6, 7, (uint8_t)s};
  return g[0].cell[0][2] + g[0].cell[1][2] * 3u + g[0].tail * 5u + g[1].cell[0][0] * 7u + g[1].tail * 11u;
}
uint32_t be_inferred(uint32_t s) {                  /* `T a[] = {...}` sized by its elided entries: 5 -> 3 */
  struct pt ps[] = {1, 2, (uint16_t)s, 4, 5};
  return (uint32_t)sizeof ps + ps[2].x + ps[1].y * 3u + ps[1].x * 5u;
}
uint32_t be_inferred_rows(uint32_t s) {             /* `T m[][2] = {...}`: 5 entries -> 3 rows */
  uint8_t m[][2] = {1, 2, (uint8_t)s, 4, 5};
  return (uint32_t)sizeof m + m[1][0] + m[2][0] * 3u + m[2][1] * 5u;
}
uint32_t be_inferred_designated(uint32_t s) {       /* a designator past the positional entries sizes it */
  uint16_t a[] = {1, [4] = (uint16_t)s, 6};
  return (uint32_t)sizeof a + a[0] + a[3] * 3u + a[4] * 5u + a[5] * 7u;
}
uint32_t be_string_member(uint32_t s) {             /* a string literal initializes a char array member */
  struct named n[2] = {"abc", 7, "xyzw", s};
  return n[0].name[0] + n[0].name[2] * 3u + (uint32_t)n[0].name[3] * 5u + n[0].id * 7u + n[1].name[3] * 11u
         + (uint32_t)n[1].name[5] * 13u + n[1].id * 17u;
}
uint32_t be_string_exact(uint32_t s) {              /* a literal exactly as long as its array drops the NUL */
  struct named n = {"abcdef", s};
  return n.name[5] + n.id * 3u;
}
uint32_t be_anon(uint32_t s) {                      /* an anonymous struct and union: one subobject each */
  struct anon a = {1, 2, 3, s};
  return a.lo + a.m * 3u + a.n * 5u + a.u * 7u;
}
uint32_t be_anon_designated(uint32_t s) {           /* a designator into an anonymous member continues inside */
  struct anon a = {.n = (uint16_t)s, .b[1] = 9, 4};
  return a.lo + a.m * 3u + a.n * 5u + a.b[0] * 7u + a.b[1] * 11u + a.b[2] * 13u + a.b[3] * 17u;
}
uint32_t be_union(uint32_t s) {                     /* a union takes one value: its first member */
  struct holder h = {s, 5};                         /* `s` elides into h.u, for its first member */
  struct holder hs[2] = {s, 5, {{7}, 9}};           /* and inside an array of holders */
  union word w[3] = {s, {6}, 8};
  return h.u.w + h.t * 3u + hs[0].t * 5u + hs[1].t * 7u + w[0].w * 11u + w[1].w * 13u + w[2].w * 17u;
}
uint32_t be_union_designated(uint32_t s) {          /* `.b = {...}` takes the other member */
  struct holder h = {.u.b = {1, 2, (uint8_t)s, 4}, 6};
  return h.u.b[0] + h.u.b[1] * 3u + h.u.b[2] * 5u + h.u.b[3] * 7u + h.t * 11u;
}
uint32_t be_struct_value(uint32_t s) {              /* a struct value initializes its subobject whole */
  struct pt q = {(uint16_t)s, 4};
  struct seg g = {q, 5, 6, 7};
  return g.a.x + g.a.y * 3u + g.b.x * 5u + g.b.y * 7u + g.tag * 11u;
}
uint32_t be_braced_scalar(uint32_t s) {             /* `T x = {e}` is `T x = e`; `{}` is zero */
  uint32_t a = {s};
  uint32_t b = {};
  struct pt z = {};
  uint8_t c[3] = {};
  return a + b * 3u + z.x * 5u + z.y * 7u + c[2] * 11u;
}
uint32_t be_compound_literal(uint32_t s) {          /* a compound literal walks the same way */
  struct seg g = (struct seg){1, 2, (uint16_t)s, 4, 5};
  uint32_t t = (struct pt[2]){(uint16_t)s, 7, 8}[1].x;
  return g.a.x + g.a.y * 3u + g.b.x * 5u + g.b.y * 7u + g.tag * 11u + t * 13u;
}
uint32_t be_entry(uint32_t s) {
  return be_flat_2d(s) + be_struct_array(s) * 3u + be_nested(s) * 5u + be_mixed(s) * 7u
         + be_designated(s) * 11u + be_array_designators(s) * 13u + be_member_rows(s) * 17u
         + be_inferred(s) * 19u + be_inferred_rows(s) * 23u + be_inferred_designated(s) * 29u
         + be_string_member(s) * 31u + be_string_exact(s) * 37u + be_anon(s) * 41u
         + be_anon_designated(s) * 43u + be_union(s) * 47u + be_union_designated(s) * 53u
         + be_struct_value(s) * 59u + be_braced_scalar(s) * 61u + be_compound_literal(s) * 67u;
}
