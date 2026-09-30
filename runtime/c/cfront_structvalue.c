/* Struct and union VALUES (CF-STRUCTVAL): an element of an array of structs -- local, static, file-scope, through a
 * pointer parameter, an array parameter or a pointer member -- a struct or union member, a member array's element,
 * `*p`, a select of two structs and a struct returned through a function pointer are each a value of the struct
 * itself: copied whole into a declaration or an assignment, returned, passed by value, stored into an element or a
 * member, placed whole by a brace list, and typed as the struct by `typeof` and `_Generic`. Both rails had read
 * such a value into an integer temp (`uint32_t t = ps[i];`, which does not compile), the twin had kept an array
 * parameter of structs a struct by value, and its brace lists and `_Generic` had not seen the struct. A struct is
 * initialized and assigned from a value of its own type (CF-STRUCTINIT). Both rails lower it the same way, and
 * each emit returns what the original does. (`va_arg` of a struct needs <stdarg.h>, which the corpus harness does
 * not include: its unit runs beside this fixture's test.) */
#include <stdint.h>

struct sv_pt { uint16_t x, y; };
struct sv_p8 { uint32_t a, b; };
union sv_u { uint32_t w; uint16_t h[2]; };
struct sv_nn { struct sv_pt p; uint32_t k; };
struct sv_w { uint8_t b[3]; uint32_t k; };
struct sv_gg { uint32_t k; struct sv_pt b; struct sv_pt v[3]; };
struct sv_holder { uint32_t n; struct sv_pt *ps; };
struct sv_ops { struct sv_pt (*mk)(uint32_t); uint32_t k; };
typedef struct sv_pt sv_pt_t;

struct sv_pt sv_gps[3];
struct sv_pt sv_gp;

static uint32_t sv_sum(struct sv_pt a) { return a.x + a.y * 3u; }   /* a struct passed by value */
static struct sv_pt sv_make(uint32_t i) {
  struct sv_pt q = {(uint16_t)i, 2};
  return q;
}

uint32_t sv_local(uint32_t s) {                     /* a declaration and an assignment from a local element */
  struct sv_pt ps[2];
  ps[0].x = 7;
  ps[0].y = (uint16_t)(s >> 3);
  ps[1].x = (uint16_t)s;
  ps[1].y = 4;
  struct sv_pt a = ps[s % 2u];
  struct sv_pt b = {0, 0};
  b = ps[(s + 1u) % 2u];
  return a.x + a.y * 3u + b.x * 5u + sv_sum(ps[s % 2u]) * 7u;
}
struct sv_pt sv_pick(uint32_t i) {                  /* an element returned by value */
  struct sv_pt ps[3] = {{1, 2}, {3, 4}, {(uint16_t)i, 6}};
  return ps[i % 3u];
}
uint32_t sv_storage(uint32_t i) {                   /* file-scope arrays and structs, a static array */
  static struct sv_pt st[2];
  for (uint32_t k = 0u; k < 3u; k++) {
    sv_gps[k].x = (uint16_t)(i + k);
    sv_gps[k].y = (uint16_t)(k * 7u);
  }
  sv_gp = sv_gps[i % 3u];
  struct sv_pt a = sv_gp;
  st[i % 2u].x = (uint16_t)(st[i % 2u].x + i);
  struct sv_pt b = st[i % 2u];
  return a.x + a.y * 3u + b.x * 5u;
}
static uint32_t sv_param_forms(const struct sv_p8 *pp, struct sv_pt m[][2], struct sv_pt v[2], uint32_t i) {
  struct sv_p8 a = pp[i];                           /* through a pointer parameter; an array parameter is a */
  struct sv_p8 b = *pp;                             /* pointer to its first row or element */
  struct sv_pt c = m[i][1];
  struct sv_pt d = v[i];
  return a.a + a.b * 3u + b.b * 5u + c.x * 7u + c.y * 11u + d.y * 13u;
}
uint32_t sv_params(uint32_t i) {
  struct sv_p8 arr[2] = {{1u, i}, {i * 3u, 4u}};
  struct sv_pt m[2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {(uint16_t)i, 8}}};
  struct sv_pt v[2] = {{9, 10}, {11, (uint16_t)i}};
  return sv_param_forms(arr, m, v, i % 2u);
}
uint32_t sv_members(uint32_t i) {                   /* a member, a member array's element, a pointer member's */
  struct sv_gg g = {7u, {(uint16_t)i, 9}, {{1, 2}, {3, 4}, {5, 6}}};
  struct sv_gg *pg = &g;
  struct sv_pt q = g.b;
  struct sv_pt r = g.v[i % 3u];
  struct sv_pt s = pg->v[(i + 1u) % 3u];
  struct sv_holder h = {2u, &g.v[0]};
  struct sv_holder *ph = &h;
  struct sv_pt t = h.ps[i % 2u];
  struct sv_pt u = ph->ps[(i + 1u) % 2u];
  struct sv_pt w = *h.ps;
  return q.x + q.y * 3u + r.x * 5u + s.y * 7u + t.x * 11u + u.y * 13u + w.x * 17u + pg->k;
}
uint32_t sv_shapes(uint32_t i) {                    /* a union, a struct holding a struct, one holding an array */
  union sv_u us[2];
  us[0].w = i;
  us[1].w = i * 3u + 1u;
  union sv_u a = us[i % 2u];
  struct sv_nn ns[2] = {{{1, 2}, 3u}, {{(uint16_t)i, 5}, 6u}};
  struct sv_w ws[2] = {{{1, 2, 3}, 4u}, {{5, (uint8_t)i, 7}, 8u}};
  struct sv_nn b = ns[i % 2u];
  struct sv_w c = ws[(i + 1u) % 2u];
  return a.w + b.p.x * 3u + b.p.y * 5u + b.k * 7u + c.b[1] * 11u + c.k * 13u;
}
uint32_t sv_select(uint32_t i) {                    /* a select of two structs; `typeof` and `_Generic` */
  struct sv_pt a = {1, (uint16_t)i}, b = {3, 4};
  struct sv_pt c = (i & 1u) ? a : b;
  c = (i & 2u) ? b : c;
  struct sv_pt ps[2] = {a, b};
  __typeof__(ps[0]) d = ps[i % 2u];
  return c.x + c.y * 3u + d.x * 5u + _Generic(ps[i % 2u], struct sv_pt: 7u, default: 9u);
}
uint32_t sv_brace(uint32_t i) {                     /* struct values placed whole by a brace list */
  struct sv_pt ps[2] = {{1, 2}, {(uint16_t)i, 4}};
  struct sv_nn g = {{5, 6}, 1u};
  const struct sv_pt *p = &ps[1];
  sv_gp.x = (uint16_t)(i >> 1);
  sv_gp.y = 3;
  struct sv_nn o[5] = {{ps[i % 2u], 5u}, {g.p, 6u}, {*p, 7u}, {sv_gp, 8u}, {(i & 1u) ? ps[0] : g.p, 9u}};
  uint32_t s = 0u;
  for (uint32_t k = 0u; k < 5u; k++) {
    struct sv_nn e = o[k];
    s = s * 7u + e.p.x + e.p.y * 3u + e.k;
  }
  return s;
}
uint32_t sv_volatile(uint32_t i) {                  /* a volatile array of structs, a volatile struct's member */
  volatile struct sv_pt vps[2];
  vps[0].x = 1;
  vps[0].y = 2;
  vps[1].x = (uint16_t)i;
  vps[1].y = 4;
  struct sv_pt a = vps[i % 2u];
  volatile struct sv_nn vn;
  vn.k = 1u;
  vn.p.x = 5;
  vn.p.y = (uint16_t)i;
  struct sv_pt b = vn.p;
  return a.x + a.y * 3u + b.y * 5u;
}
static uint32_t sv_call(struct sv_pt (*mk)(uint32_t), uint32_t i) {
  struct sv_pt q = mk(i);
  return q.x + q.y * 3u;
}
uint32_t sv_calls(uint32_t i) {                     /* a struct returned through a function pointer */
  struct sv_pt (*mk)(uint32_t) = sv_make;
  struct sv_ops ops = {sv_make, 1u};
  struct sv_ops *po = &ops;
  struct sv_pt a = mk(i);
  struct sv_pt b = ops.mk(i + 1u);
  struct sv_pt c = po->mk(i + 2u);
  return a.x + b.x * 3u + c.x * 5u + sv_call(sv_make, i) * 7u;
}
uint32_t sv_stores(uint32_t i) {                    /* struct values stored into elements and members */
  struct sv_pt a = {(uint16_t)i, 2};
  struct sv_gg g = {0u, {0, 0}, {{0, 0}, {0, 0}, {0, 0}}};
  struct sv_gg *pg = &g;
  struct sv_pt ps[2] = {{0, 0}, {0, 0}};
  struct sv_pt *p = &ps[0];
  g.b = a;
  pg->v[1] = a;
  g.v[0] = g.b;
  ps[1] = g.v[1];
  *p = ps[1];
  p[1] = pg->b;
  return ps[0].x + ps[1].y * 3u + g.v[0].x * 5u + pg->v[1].y * 7u;
}
uint32_t sv_copies(uint32_t i) {                    /* initialized and assigned from a value of its own type */
  sv_pt_t a = {(uint16_t)i, 2};
  const struct sv_pt c = a;
  struct sv_pt d = {0, 0};
  d = c;
  union sv_u u1, u2;
  u1.w = i;
  u2 = u1;
  struct sv_pt e = sv_make(i + 3u);
  struct sv_pt f = (struct sv_pt){7, (uint16_t)(i >> 2)};
  e = (i & 1u) ? f : e;
  return c.y + d.x * 3u + u2.w * 5u + e.x * 7u + f.y * 11u;
}
uint32_t sv_entry(uint32_t i) {
  struct sv_pt p = sv_pick(i);
  return sv_local(i) + sv_sum(p) * 3u + sv_storage(i) * 5u + sv_params(i) * 7u + sv_members(i) * 11u
         + sv_shapes(i) * 13u + sv_select(i) * 17u + sv_brace(i) * 19u + sv_volatile(i) * 23u + sv_calls(i) * 29u
         + sv_stores(i) * 31u + sv_copies(i) * 37u;
}
