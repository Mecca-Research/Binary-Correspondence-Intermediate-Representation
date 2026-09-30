/* An array stored into a pointer slot decays to its address (C11 6.3.2.1p3, CF-MEMDECAY): a pointer member
 * set from a file-scope, string-initialized or variable-length array -- by assignment, through `->` or a
 * dereference, in a brace list, a designated list or a compound literal, or into a member array of pointers --
 * holds the array's address, never the array's first bytes. Both rails lower it the same way, and each emit
 * returns what the original does. The arrays are at file scope: a local array stored into a pointer slot
 * escapes, and the G10 escape row counts every local array of this corpus it cannot prove private, so the
 * same forms on local arrays run in `test_arrays_stored_into_pointer_slots_decay_on_both_rails`. */
#include <stdint.h>

struct holder { uint32_t tag; uint32_t *p; };
struct msg { uint8_t n; const char *txt; };
struct pt { uint16_t x, y; };
struct ptrs { uint8_t k; uint32_t *slot[3]; };
struct segs { struct pt *pts; uint32_t n; };
uint32_t md_garr[4] = {4u, 5u, 6u, 7u};

uint32_t md_a3[3];
uint32_t md_b2[2];
uint32_t md_c2[2];
char md_buf[5] = {'b', 'c', 'i', 'r', 0};

uint32_t md_member(uint32_t s) {                    /* h.p = arr */
  md_a3[0] = s; md_a3[1] = s + 1u; md_a3[2] = s + 2u;
  struct holder h;
  h.tag = 7u;
  h.p = md_a3;
  return h.p[2] + h.tag;
}
uint32_t md_braced(uint32_t s) {                    /* {7u, arr} and {.p = arr, ...} */
  md_a3[0] = s; md_a3[1] = s * 3u; md_a3[2] = s ^ 5u;
  struct holder a = {7u, md_a3};
  struct holder b = {.p = md_a3, .tag = s};
  return a.p[1] + b.p[2] * 3u + b.tag;
}
uint32_t md_arrow(uint32_t s) {                     /* hp->p = arr */
  md_b2[0] = s; md_b2[1] = s + 9u;
  struct holder h;
  struct holder *hp = &h;
  hp->tag = 1u;
  hp->p = md_b2;
  return hp->p[1] + h.p[0] + h.tag;
}
uint32_t md_deref(uint32_t s) {                     /* *pp = arr */
  md_b2[0] = s + 4u; md_b2[1] = s;
  uint32_t other = 3u;
  uint32_t *q = &other;
  uint32_t **pp = &q;
  *pp = md_b2;
  return q[0] + q[1];
}
uint32_t md_global(uint32_t s) {                    /* an initialized file-scope array into a member */
  struct holder h = {s, md_garr};
  struct holder k;
  k.tag = 2u;
  k.p = md_garr;
  return h.p[3] + k.p[1] + h.tag + k.tag;
}
uint32_t md_string(uint32_t s) {                    /* a character array into a `const char *` */
  struct msg m;
  m.n = (uint8_t)s;
  m.txt = md_buf;
  return (uint32_t)m.txt[2] + m.n;
}
uint32_t md_compound(uint32_t s) {                  /* a compound literal's member */
  md_a3[0] = s; md_a3[1] = 2u; md_a3[2] = 3u;
  struct holder h = (struct holder){5u, md_a3};
  return h.p[0] * h.tag + h.p[2];
}
uint32_t md_ptr_array(uint32_t s) {                 /* a member array of pointers, element by element */
  md_b2[0] = s; md_b2[1] = 1u;
  md_c2[0] = 2u; md_c2[1] = s + 3u;
  struct ptrs r;
  r.k = 1u;
  r.slot[0] = md_b2;
  r.slot[1] = md_c2;
  r.slot[2] = md_b2;
  uint32_t *w = r.slot[0];
  uint32_t *x = r.slot[1];
  uint32_t *y = r.slot[2];
  return w[0] + x[1] * 5u + y[1] + r.k;
}
uint32_t md_structs(uint32_t s) {                   /* an array of structs into a pointer-to-struct member */
  struct pt pts[2];
  pts[0].x = (uint16_t)s;
  pts[0].y = 1u;
  pts[1].x = 2u;
  pts[1].y = (uint16_t)(s + 3u);
  struct segs g;
  g.n = 2u;
  g.pts = pts;
  struct pt *q = g.pts;
  return q[1].y + q[0].x + g.n;
}
uint32_t md_vla(uint32_t s) {                       /* a variable-length array */
  uint32_t n = s % 4u + 1u;
  uint32_t v[n];
  for (uint32_t i = 0u; i < n; i++) v[i] = s + i;
  struct holder h;
  h.tag = n;
  h.p = v;
  return h.p[n - 1u] + h.tag;
}
uint32_t md_entry(uint32_t s) {
  return md_member(s) + md_braced(s) * 3u + md_arrow(s) * 5u + md_deref(s) * 7u + md_global(s) * 11u +
         md_string(s) * 13u + md_compound(s) * 17u + md_ptr_array(s) * 19u + md_structs(s) * 23u + md_vla(s) * 29u;
}
