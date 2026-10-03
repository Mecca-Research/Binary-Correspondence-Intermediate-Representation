/* A postfix operator after a parenthesized operand (CF-PAREN) applies to what the parentheses hold: `(a)[i]`,
 * `(s).x`, `(p)->x`, `(x)++`, `((a))[i]`, `((uint32_t[3]){s, 7})[1]`, `((struct pt[2]){...})[1].x`,
 * `("abc")[i]` -- as macro-heavy driver code spells them (`#define REGS (base)` then `REGS[3]`) -- read,
 * stored, compound-assigned, stepped and addressed; `(*p).x` is `p->x` and `(*p)[i]` is `p[0][i]`, the row
 * a `T (*p)[N]` parameter points at. A parenthesis that opens a call's arguments, a condition or a cast
 * stays one (`if (s) ++y;`, `(T)++w`, `at(s)[1]`). The twin had refused every form (a postfix operator after
 * `)`); the oracle had loaded the whole struct for `(*p).x` -- a 4-byte temp that dropped every member past
 * it -- and emitted an uncompilable scalar subscript for `(*p)[i]`. The global is reset first. */
#include <stdint.h>

#define PP_REGS (pp_base)
#define PP_R(d) ((d)->regs)
typedef uint32_t pp_T;
struct pp_pt { uint16_t x, y; };
struct pp_big { uint32_t a, b, c; struct pp_pt in; uint32_t arr[3]; };
struct pp_dev { uint32_t regs[4]; };

uint32_t pp_g[3];
uint32_t pp_buf[4] = {1u, 2u, 3u, 4u};

static void pp_reset(void) { pp_g[0] = 1u; pp_g[1] = 2u; pp_g[2] = 3u; }
uint32_t *pp_at(uint32_t k) { return &pp_buf[k & 1u]; }
uint32_t pp_index(uint32_t s) {                        /* a parenthesized name, subscripted */
  uint32_t a[3] = {s, 2u, 3u};
  return (a)[1] + (a)[0] * 3u + ((a))[2] * 5u;
}
uint32_t pp_store(uint32_t s) {                        /* =, OP= and an assignment's value */
  uint32_t a[3] = {s, 2u, 3u};
  (a)[1] = s + 4u;
  (a)[2] += 3u;
  uint32_t v = ((a)[0] = 9u);
  return a[1] + a[2] * 3u + v * 5u + a[0];
}
uint32_t pp_steps(uint32_t s) {                        /* ++ / -- after the parentheses: statements and values */
  uint32_t a[3] = {s, 2u, 3u};
  uint32_t x = s;
  (a)[1]++;
  (a)[2]--;
  (x)++;
  uint32_t o = (x)--;
  uint32_t p = (a)[0]++;
  uint32_t q = ((x))++;
  return a[1] * 3u + a[2] + x * 5u + o * 7u + p * 11u + q + a[0];
}
uint32_t pp_literal(uint32_t s) {                      /* a parenthesized compound literal, subscripted */
  return ((uint32_t[3]){s, 7u})[1] + ((uint32_t[3]){s, 7u})[0] * 3u
         + ((uint32_t[2][2]){{s, 2u}, {3u, 4u}})[1][0] * 5u;
}
uint32_t pp_literal_struct(uint32_t s) {               /* ... and a member of one of its elements */
  return ((struct pp_pt[2]){(uint16_t)s, 7u, 8u})[1].x + ((struct pp_pt){(uint16_t)s, 3u}).y * 3u
         + ((struct pp_pt[2]){(uint16_t)s, 7u, 8u})[0].x;
}
uint32_t pp_members(uint32_t s) {                      /* (s).x and (p)->x */
  struct pp_pt q = {(uint16_t)s, 3u};
  struct pp_pt *p = &q;
  (q).x += 2u;
  (p)->y = (uint16_t)(s >> 2);
  uint16_t *r = &(p)->x;
  return (q).x + (p)->y * 3u + *r * 5u;
}
uint32_t pp_deref(uint32_t s) {                        /* (*p).m is p->m, past the first word too */
  struct pp_big b = {1u, 2u, s, {4u, 5u}, {6u, 7u, 8u}};
  struct pp_big *p = &b;
  (*p).c += 1u;
  (*p).in.y = (uint16_t)s;
  (*p).arr[1] = s + 1u;
  uint32_t v = ((*p).a = 3u);
  return (*p).c + (*p).in.y * 3u + (*p).arr[1] * 5u + v * 7u + (*p).b;
}
uint32_t pp_rows(uint32_t (*p)[3], uint32_t i) {       /* (*p)[i] is p[0][i]: the row p points at */
  (*p)[i % 3u] = 9u;
  (*p)[0] += 1u;
  uint32_t *q = &(*p)[1];
  *q += 2u;
  return (*p)[i % 3u] + (*p)[0] * 3u + (*p)[1] * 5u;
}
uint32_t pp_macro(uint32_t s) {                        /* the macro spellings */
  uint32_t pp_base[4] = {1u, 2u, 3u, 4u};
  struct pp_dev dv = {{5u, 6u, 7u, 8u}};
  struct pp_dev *dp = &dv;
  PP_REGS[3] = s;
  PP_R(dp)[2] += 1u;
  PP_R(dp)[1]++;
  return PP_REGS[3] + PP_R(dp)[2] * 3u + PP_R(dp)[1] * 5u;
}
uint32_t pp_global(uint32_t s) {                       /* a file-scope array, a cast and a string */
  (pp_g)[1] = s;
  (pp_g)[2] = (uint32_t)(uint16_t)(pp_g)[1];
  return (uint32_t)(uint8_t)(pp_g)[2] + (pp_g)[1] * 3u + (uint32_t)("abc")[s % 3u] * 5u;
}
uint32_t pp_control(uint32_t s) {                      /* in a for, a do-while, an if-else and a switch */
  uint32_t a[4] = {s, 1u, 2u, 3u};
  uint32_t t = 0u;
  for (uint32_t i = (a)[1]; i < (a)[3]; (i)++) t += (a)[i];
  do (a)[1]++; while ((a)[1] < 5u);
  if ((a)[0] & 1u) (a)[2] = 7u; else (a)[2] = 8u;
  switch ((a)[2]) { case 7u: (a)[3] = 1u; break; default: (a)[3] = 2u; }
  return t + a[1] * 3u + a[2] * 5u + a[3] * 7u;
}
uint32_t pp_forms(uint32_t s) {                        /* a statement expression, _Generic, typeof, sizeof */
  uint16_t a[2] = {(uint16_t)s, 2u};
  uint32_t v = ({ (a)[1] += 1u; (a)[0] + (a)[1]; });
  typeof((a)[0]) z = (a)[1];
  return v + (uint32_t)_Generic((a)[0], uint16_t: 10u, default: 20u) + z * 3u + (uint32_t)sizeof((a)[1]) * 5u;
}
uint32_t pp_not_operands(uint32_t s) {                 /* parentheses that stay: a condition, a cast, a call */
  uint32_t y = 1u, w = s;
  if (s) ++y;
  while (w > 3u) --w;
  pp_T z = (pp_T)++w;
  return y + w * 3u + z * 5u + pp_at(s)[1] * 7u;
}
uint32_t pp_entry(uint32_t s) {
  uint32_t a[4] = {s, 1u, 2u, 3u};
  struct pp_pt q = {(uint16_t)s, 5u};
  struct pp_pt *p = &q;
  (a)[s % 4u] += (p)->y;
  (*p).x++;
  return (a)[s % 4u] + (q).x * 3u + ((uint32_t[2]){s, 9u})[s % 2u] * 5u;
}
