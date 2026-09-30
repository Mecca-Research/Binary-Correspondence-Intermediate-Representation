/* The null pointer constants CF-NULLPTR and CF-NULLARG left an `int` (CF-NULLCALL): the constant 0 compared with a
 * pointer by `==` or `!=` (C11 6.5.9p5) -- on either side, beside a parameter, a local, a global, a loaded member
 * or a member chain, a function pointer -- passed through a function pointer to a pointer parameter (6.5.2.2p7) --
 * a declarator, a typedef'd local, a parameter, a member by `.` and `->` -- given to `free` and as `realloc`'s
 * pointer, and the arm of `?:` beside a pointer (6.5.15p6). Both rails lowered each to the same claim graph, and
 * each emit declared the constant an `int` temp: compared with a pointer (a constraint violation Clang and GCC only
 * warn about) or passed where the callee takes a pointer (which they reject). Each emit now declares it as the
 * pointer it converts to; only the temps' C types move. `nc_step` holds a split the struct-arithmetic refusals
 * found: the twin read `p -= 1` of a pointer to a struct as `p->`, and refused it (`unknown field`). Every
 * indirect call here resolves to the one function it reaches (the G10 escape rows, tools/perf/check_escape.py). */
#include <stdint.h>
#include <stdlib.h>

struct nc_node { uint32_t v; struct nc_node *next; };
typedef uint32_t (*nc_fn)(uint32_t *);
struct nc_reader { nc_fn read; uint32_t k; };
struct nc_walker { uint32_t (*walk)(struct nc_node *, uint32_t); };

uint32_t *nc_g;                                      /* file scope: a global pointer, and what it points at */
uint32_t nc_tab[4];
struct nc_node nc_nodes[4];
static uint32_t nc_read(uint32_t *p) { return p ? *p * 3u + 1u : 7u; }
static uint32_t nc_walk(struct nc_node *n, uint32_t s) {
  uint32_t k = s;
  while (n != 0) {
    k += n->v;
    n = n->next;
  }
  return k;
}

uint32_t nc_compare(uint32_t *p, uint32_t s) {       /* a parameter, a local and a global, 0 on either side */
  uint32_t v = s, k = 0u;
  uint32_t *q = s & 1u ? &v : 0;                      /* 0 beside a pointer in `?:` */
  nc_tab[0] = s;
  nc_g = s & 2u ? nc_tab : 0;
  if (p == 0) k += 1u;
  if (p != 0) k += *p;
  if (0 == q) k += 4u;
  if (0 != q) k += *q * 8u;
  if (nc_g == 0) k += 16u;
  if (nc_g != 0) k += *nc_g * 32u;
  return k + (p == 0) * 64u + (0 != q) * 128u;
}
uint32_t nc_member(uint32_t s) {                     /* a loaded member, by `.` and `->`, and a chain */
  struct nc_node c = {5u, 0}, b = {s, &c}, a = {1u, &b};
  struct nc_node *pa = &a;
  uint32_t k = 0u;
  if (a.next == 0) k += 1u;
  if (a.next->next != 0) k += 2u;
  if (0 == pa->next->next->next) k += 4u;
  if (c.next != 0) k += 8u;
  return k + nc_walk(pa, s);
}
uint32_t nc_fnptr(uint32_t s) {                      /* a function pointer compared with 0 */
  nc_fn f = s & 1u ? nc_read : 0;
  uint32_t (*g)(uint32_t *) = nc_read;
  uint32_t k = 0u;
  if (f == 0) k += 1u;
  if (f != 0) k += 2u;
  if (0 != g) k += 4u;
  return k;
}
static uint32_t nc_through(nc_fn f, uint32_t s) {   /* 0 passed through a parameter, typedef'd and declared */
  return f(0) + f(nc_tab) * 3u + s;
}
static uint32_t nc_through_decl(uint32_t (*f)(uint32_t *), uint32_t s) { return f(0) * 5u + s; }
uint32_t nc_calls(uint32_t s) {                      /* ... through a declarator, a typedef'd local, a member */
  uint32_t (*fn)(uint32_t *) = nc_read;
  nc_fn f2 = nc_read;
  struct nc_reader r = {nc_read, 2u};
  struct nc_reader *rp = &r;
  struct nc_walker w = {nc_walk};
  nc_tab[1] = s;
  uint32_t k = fn(0) + f2(0) * 3u + r.read(0) * 5u + rp->read(0) * 7u + w.walk(0, s) * 9u;
  return k + fn(nc_tab + 1) * 11u + nc_through(nc_read, s) * 13u + nc_through_decl(nc_read, s) * 15u + r.k;
}
uint32_t nc_libc(uint32_t s) {                       /* `free(0)`, `realloc(0, n)` */
  free(0);
  uint32_t *q = realloc(0, 4u * sizeof(uint32_t));
  if (q == 0) return 1u;
  q[0] = s;
  q[3] = s + 3u;
  uint32_t r = q[0] + q[3];
  free(q);
  return r;
}
uint32_t nc_step(uint32_t s) {                       /* a pointer to a struct stepped back and forth */
  struct nc_node *p = &nc_nodes[2];
  p->v = s;
  p -= 1;
  p->v = s + 1u;
  p += 2;
  p->v = s * 3u;
  return nc_nodes[1].v + nc_nodes[2].v * 5u + nc_nodes[3].v * 7u;
}
uint32_t nc_entry(uint32_t s) {
  uint32_t v = s;
  return nc_compare(&v, s) + nc_compare(0, s) * 3u + nc_member(s) * 5u + nc_fnptr(s) * 7u + nc_calls(s) * 9u
         + nc_libc(s) * 11u + nc_step(s) * 13u;
}
