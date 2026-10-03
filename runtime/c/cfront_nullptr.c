/* A null pointer constant (C11 6.3.2.3p3, CF-NULLPTR): the integer constant 0 taken by a pointer -- a
 * declaration's initializer (braced, empty or not, any integer literal), an assignment to a local, a
 * parameter or a file-scope pointer, a `return` from a function returning a pointer, an element of an array
 * of pointers or of a `T **`, stored or listed -- is a null pointer of that pointer's type. Both rails lowered
 * it to the same claim graph, but each emit assigned an `int` temp to the pointer, which C forbids (Clang and
 * GCC reject it); each emit now declares the constant as the pointer and returns what the original does. */
#include <stdint.h>

struct node { uint32_t v; struct node *next; };
typedef uint32_t (*op_t)(uint32_t);
uint32_t *np_g;

uint32_t np_decl(uint32_t s) {                      /* T *p = 0 / {0} / {} / 0L / 0u */
  uint32_t *p = 0;
  uint32_t *q = {0};
  uint32_t *r = {};
  const char *c = 0L;
  void *w = 0u;
  struct node *n = 0;
  op_t fn = 0;
  uint32_t k = s;
  if (p) k += 1u;
  if (q) k += 2u;
  if (r) k += 4u;
  if (c) k += 8u;
  if (w) k += 16u;
  if (n) k += 32u;
  if (fn) k += 64u;
  return k;
}
uint32_t np_assign(uint32_t *p, uint32_t s) {       /* p = 0 for a local, a parameter and a global */
  uint32_t v = s;
  uint32_t *q = &v;
  uint32_t n = *q;
  q = 0;
  p = 0;
  np_g = &v;
  n += *np_g;
  np_g = 0;
  if (q) n += 1u;
  if (p) n += 2u;
  if (np_g) n += 4u;
  return n;
}
static uint32_t *np_find(uint32_t *a, uint32_t n, uint32_t key) {   /* `return 0;` from a pointer function */
  for (uint32_t i = 0u; i < n; i++) {
    if (a[i] == key) return &a[i];
  }
  return 0;
}
static struct node *np_first(struct node *n, uint32_t s) {
  if (s & 1u) return 0;
  return n;
}
uint32_t np_tab[4];                                /* file scope: a local array lent to a call escapes (G10) */
uint32_t np_return(uint32_t s) {
  np_tab[0] = s; np_tab[1] = s + 1u; np_tab[2] = s + 2u; np_tab[3] = s + 3u;
  struct node b = {s * 3u, 0};
  uint32_t *hit = np_find(np_tab, 4u, s + 2u);
  uint32_t *miss = np_find(np_tab, 4u, s + 9u);
  struct node *first = np_first(&b, s);
  uint32_t r = 0u;
  if (hit) r += *hit;
  if (miss) r += 100u;
  if (first) r += first->v;
  return r;
}
uint32_t np_elements(uint32_t s) {                  /* an array of pointers and a `T **`, stored and listed */
  uint32_t v = s, w = s + 5u;
  uint32_t *ps[3] = {&v, 0, &w};
  uint32_t *qs[2];
  qs[0] = 0;
  qs[1] = &v;
  uint32_t *rs[2];
  rs[0] = &w;
  rs[1] = &w;
  uint32_t **pp = rs;
  pp[1] = 0;
  uint32_t k = *ps[0] + *ps[2];
  if (ps[1]) k += 1u;
  if (qs[0]) k += 2u;
  k += *qs[1] * 4u;                                  /* the pointer stored beside the null one, read through */
  if (rs[1]) k += 8u;
  return k + *rs[0];
}
uint32_t np_entry(uint32_t s) {
  uint32_t g = s;
  return np_decl(s) + np_assign(&g, s) * 3u + np_return(s) * 5u + np_elements(s) * 7u;
}
