/* An array-of-pointers struct member (CF-PTRARR): `T *arr[N]`. The C twin laid each element out by its
 * POINTEE's size -- `uint32_t *arr[2]` took 4-byte elements -- so a later member sat at the wrong offset,
 * `sizeof` was wrong, a store truncated the pointer to 4 bytes (`uint32_t _v = p`), and a load read the
 * element into an integer, which Clang rejects. Each element now takes the ABI's pointer size and is loaded
 * and stored whole, typed `T *`, digest for digest with the oracle. `*t->arr[i]` dereferences the element:
 * a postfix operator binds tighter than `*`, and the twin had dereferenced `t` and then failed on the `->`
 * (so `*s->p` through a plain pointer member failed too). `t->arr[i][j]` and `t->arr[i]++` stay refused on
 * both rails.
 *
 * The harness fills every struct with random bytes, so each function stores a pointer into an element before
 * it dereferences one: no garbage pointer is ever followed. */
#include <stdint.h>

struct T { uint8_t c; uint32_t *arr[2]; uint8_t d; };
struct S { uint32_t v; uint16_t w; };
struct U { uint8_t c; struct S *sp[2]; double *dp[3]; uint32_t n; };
struct P { uint8_t c; uint32_t *p; };

uint32_t pm_roundtrip(struct T *t, uint32_t *p) {
  t->d = 7u;                       /* the scalar AFTER the array: a 4-byte element model put it at 12, not 24 */
  t->arr[1] = p;                   /* an 8-byte pointer store at 8 + 1*8 */
  t->arr[0] = p + 3;
  uint32_t *q = t->arr[1];         /* loaded back whole, typed `uint32_t *` */
  return *q + *t->arr[0] + t->d + (uint32_t)sizeof(struct T);
}

uint32_t pm_deref_store(struct T *t, uint32_t *p, uint32_t v) {
  t->arr[0] = p + 1;
  *t->arr[0] = v;                  /* a store THROUGH the loaded element */
  *t->arr[0] += 5u;                /* and a compound assignment through it */
  return p[1];
}

uint32_t pm_step(struct T *t, uint32_t *p) {
  t->arr[1] = p;
  t->arr[1] += 2;                  /* pointer arithmetic on the element: it steps by its pointee */
  t->arr[1] -= 1;
  return *t->arr[1] + (uint32_t)(t->arr[1] - p);
}

uint32_t pm_index(struct T *t, uint32_t *p, uint32_t i) {
  t->arr[i & 1u] = p + (i & 7u);   /* a variable element index */
  uint32_t *q = t->arr[i & 1u];
  return *q + (uint32_t)(t->arr[i & 1u] == p + (i & 7u));
}

uint32_t pm_value(uint32_t *p) {
  struct T s = { 1, { p, p + 2 }, 9 };   /* an initializer list fills the pointer elements */
  struct T u = { .arr = { p + 4, p } };  /* and a designated one */
  return *s.arr[1] + *u.arr[0] + s.d + u.c;
}

uint32_t pm_aggregate(struct U *u, struct S *s, double *x) {
  u->n = 3u;
  s->v = 11u; s->w = 5u; *x = 2.5;  /* known values: the harness fills a buffer through `uint32_t` stores, so
                                     * reading a `double` or `uint16_t` it never wrote lets GCC and Clang
                                     * differ; read back through the elements, they prove the addresses */
  u->sp[1] = s;                    /* a pointer-to-struct element */
  u->dp[2] = x;                    /* a pointer-to-double element */
  struct S *e = u->sp[1];
  double *y = u->dp[2];
  return e->v + e->w + (uint32_t)(*y > 2.0) + u->n + (uint32_t)sizeof(struct U);
}

uint32_t pm_member_deref(struct T *t, struct P *s, uint32_t *p) {
  t->arr[0] = p;
  s->p = p + 1;
  return *t->arr[0] + *(t->arr[0]) + *s->p;   /* `*t->arr[0]`, its parenthesized spelling, and `*s->p` */
}

/* The harness runs the unit's LAST function against the original: this one runs every form above. The calls
 * share `t` and `p`, so each is its own statement -- the operands of one `+` are unsequenced, and GCC and Clang
 * would call them in different orders. */
uint32_t pm_all(struct T *t, struct U *u, struct S *s, struct P *q, uint32_t *p, double *x, uint32_t v,
                uint32_t i) {
  uint32_t r = pm_roundtrip(t, p);
  r += pm_deref_store(t, p, v);
  r += pm_step(t, p);
  r += pm_index(t, p, i);
  r += pm_value(p);
  r += pm_aggregate(u, s, x);
  r += pm_member_deref(t, q, p);
  return r;
}
