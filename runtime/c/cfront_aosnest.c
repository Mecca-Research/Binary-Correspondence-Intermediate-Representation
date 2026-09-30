/* A member of a member of an array element (CF-NESTMEM): `a[i].m.k` through any depth of nested struct and
 * union members -- and a member array at the end of the chain, `a[i].m.arr[j]` -- is one access at the
 * element's stride plus the member's flattened offset in the element: read, stored, compound-assigned,
 * stepped as a statement and as a value, assigned as a value, and addressed, on a local array, a file-scope
 * array, an array reached through a pointer parameter, a member array of structs (`b.s[i].m.k`), and the
 * elements a pointer member points at (`h.next[i].m.k`). A volatile member keeps its accesses volatile. Both
 * rails had refused every nested form (the oracle's "unsupported base expression Index", the twin's
 * "array-of-structs non-scalar element field"), the twin a flat `a[i].f++;` statement, and the twin
 * `h.next[i].f`. Every global the functions touch is reset first, so a run does not depend on the last. */
#include <stdint.h>

union an_word { uint32_t w; uint8_t b[4]; };
struct an_holder { union an_word u; uint16_t t; };
struct an_pt { uint16_t x, y; };
struct an_seg { struct an_pt a, b; };
struct an_in { uint8_t arr[4]; uint16_t grid[2][3]; uint32_t t; };
struct an_outer { uint32_t k; struct an_in m; };
struct an_c3 { uint8_t lo; uint32_t v; };
struct an_c2 { uint16_t k; struct an_c3 in; };
struct an_c1 { uint32_t h; struct an_c2 mid; };
struct an_box { struct an_seg s[2]; uint32_t n; };
struct an_node { uint32_t v; struct an_seg sg; struct an_node *next; };
struct an_vseg { struct an_pt a; volatile struct an_pt b; };
struct an_apt { uint16_t x; _Atomic uint32_t c; };
struct an_aseg { struct an_apt a, b; };

struct an_seg an_g[3];
struct an_outer an_go[2];

static void an_reset(void) {                           /* the members the functions below write */
  for (uint32_t i = 0u; i < 3u; i++) {
    an_g[i].a.x = (uint16_t)i; an_g[i].b.x = (uint16_t)(i + 20u); an_g[i].b.y = (uint16_t)(i + 30u);
  }
  for (uint32_t i = 0u; i < 2u; i++) {
    an_go[i].m.t = i + 5u;
    for (uint32_t j = 0u; j < 4u; j++) an_go[i].m.arr[j] = (uint8_t)(i * 4u + j);
  }
}
uint32_t an_union(uint32_t s) {                        /* a union member, and a member array of the union */
  struct an_holder h[3] = {0};
  h[0].t = 1u;
  h[2].u.w = s;
  h[1].u.b[2] = (uint8_t)(s >> 5);
  h[1].t = (uint16_t)(s >> 3);
  return h[2].u.w * 3u + h[1].u.b[2] + h[1].t + h[0].t;
}
uint32_t an_compound(uint32_t s) {                     /* OP= on a nested member of an element */
  struct an_seg g[2] = {0};
  g[1].b.y = (uint16_t)s;
  g[1].b.y += 4u;
  g[1].b.y <<= 2u;
  g[1].b.y |= 1u;
  g[0].a.x = 9u;
  g[0].a.x *= (uint16_t)(s | 1u);
  return g[1].b.y + g[0].a.x * 7u;
}
uint32_t an_steps(uint32_t s) {                        /* ++ / -- as statements and as values */
  struct an_seg g[2] = {0};
  g[1].b.y = (uint16_t)s;
  g[0].a.x = (uint16_t)(s >> 16);
  g[1].b.y++;
  g[1].b.y--;
  g[0].a.x++;
  uint32_t o = g[1].b.y++;
  uint32_t n = ++g[1].b.y;
  uint32_t m = g[1].b.y--;
  uint32_t q = --g[0].a.x;
  return o * 3u + n * 5u + m * 7u + q * 11u + g[1].b.y + g[0].a.x;
}
uint32_t an_values(uint32_t s) {                       /* an assignment to a nested member as a value */
  struct an_seg g[2] = {0};
  uint32_t v = (g[0].a.x = (uint16_t)(s + 70000u));
  uint32_t w = (g[0].a.y = 3u);
  uint32_t z = (g[0].a.y += 65535u);
  return v + w * 7u + z * 13u + g[0].a.y;
}
uint32_t an_address(uint32_t s) {                      /* &a[i].m.k, &a[i].m.arr[j] */
  struct an_seg g[3] = {0};
  struct an_outer o[2] = {0};
  uint16_t *q = &g[2].b.x;
  *q = (uint16_t)s;
  uint16_t *r = &g[1].a.y;
  *r = 9u;
  uint8_t *u = &o[1].m.arr[3];
  *u = (uint8_t)(s >> 8);
  return g[2].b.x + g[1].a.y * 3u + o[1].m.arr[3] * 5u;
}
uint32_t an_deep(uint32_t s) {                         /* three levels of nested members */
  struct an_c1 a[2] = {0};
  a[1].mid.in.v = s;
  a[1].mid.in.lo = 3u;
  a[0].mid.k = 4u;
  a[1].mid.in.v += a[0].mid.k;
  return a[1].mid.in.v + a[1].mid.in.lo + a[0].mid.k * 5u;
}
uint32_t an_member_array(uint32_t i) {                 /* a member array at the end of the chain, 1-D and 2-D */
  struct an_outer o[2] = {0};
  o[1].m.arr[2] = 9u;
  o[1].m.arr[i % 4u] += 3u;
  o[0].m.arr[1]++;
  uint32_t v = (o[1].m.arr[0] = 300u);
  uint32_t w = ++o[1].m.arr[0];
  o[1].m.grid[1][2] = 11u;
  o[1].m.grid[1][i % 3u] *= 2u;
  return o[1].m.arr[2] + o[0].m.arr[1] * 3u + v * 5u + w * 7u + o[1].m.grid[1][2] * 11u;
}
uint32_t an_global(uint32_t i) {                       /* a file-scope array of structs */
  an_g[1].b.y = 5u;
  an_g[2].a.x = (uint16_t)i;
  an_g[2].a.x += 2u;
  an_g[1].b.y++;
  an_go[1].m.arr[i % 4u] += 3u;
  an_go[0].m.t = 5u;
  return an_g[i % 3u].b.y + an_g[2].a.x * 3u + an_g[0].b.x * 5u + an_go[1].m.arr[2] * 7u + an_go[0].m.t;
}
uint32_t an_param(struct an_seg *p, uint32_t i) {      /* the elements a pointer parameter points at */
  p[i].b.y = 5u;
  p[i].a.x += 3u;
  p[i].b.y++;
  uint32_t v = (p[i].b.x = 2u);
  uint16_t *q = &p[i].a.y;
  *q = 8u;
  return p[i].b.y + p[i].a.x * 3u + v * 5u + p[i].a.y * 7u;
}
uint32_t an_param_call(uint32_t i) {
  struct an_seg sg[2] = {0};
  sg[0].a.x = 1u;
  sg[1].a.x = (uint16_t)i;
  uint32_t r = an_param(sg, i % 2u);
  return r + sg[0].a.x;
}
uint32_t an_member_elements(uint32_t i) {              /* a member array of structs: b.s[i].m.k */
  struct an_box bx = {0};
  bx.s[1].b.y = 5u;
  bx.s[1].b.y += 2u;
  bx.s[0].a.x = (uint16_t)i;
  bx.s[0].a.x++;
  uint16_t *q = &bx.s[1].a.y;
  *q = 4u;
  struct an_box *p = &bx;
  p->s[i % 2u].b.x = 3u;
  return bx.s[i % 2u].b.y + bx.s[0].a.x * 3u + bx.s[1].a.y * 5u + p->s[1].b.y * 7u + bx.s[i % 2u].b.x;
}
uint32_t an_pointer_member(uint32_t s) {               /* the elements a pointer member points at: h.next[i] */
  struct an_node arr[3] = {0};
  struct an_node h = {0};
  arr[1].v = s;
  h.next = &arr[0];
  struct an_node *hp = &h;
  h.next[1].v += 2u;
  h.next[2].sg.b.y = (uint16_t)(s >> 4);
  hp->next[2].v = 9u;
  return h.next[1].v + hp->next[2].v * 3u + hp->next[2].sg.b.y * 5u;
}
uint32_t an_volatile(uint32_t s) {                     /* a volatile member's members stay volatile accesses */
  struct an_vseg v[2] = {0};
  v[1].b.y = (uint16_t)s;
  v[1].a.x = 1u;
  v[1].b.x = (uint16_t)(v[1].b.y + 2u);
  return v[1].b.y + v[1].a.x * 3u + v[1].b.x * 5u;
}
uint32_t an_atomic(uint32_t s) {                       /* an `_Atomic` member: one atomic operation each */
  struct an_aseg g[2] = {0};
  g[1].b.c = s;
  g[1].b.c += 2u;
  g[1].b.c++;
  uint32_t o = g[1].b.c++;
  return g[1].b.c + o * 3u;
}
uint32_t an_entry(uint32_t i) {
  struct an_seg g[2] = {0};
  struct an_outer o[2] = {0};
  g[i % 2u].b.y = (uint16_t)i;
  g[1].a.x = 7u;
  o[i % 2u].m.arr[i % 4u] = (uint8_t)i;
  return g[i % 2u].b.y + g[1].a.x * 3u + o[i % 2u].m.arr[i % 4u] * 5u;
}
