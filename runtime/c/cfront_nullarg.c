/* A null pointer constant passed as a call argument (C11 6.5.2.2p7, CF-NULLARG): an argument converts to its
 * parameter's type as if by assignment, so the integer constant 0 given to a pointer parameter -- a pointer to a
 * scalar, to const, to void, to a pointer or to a struct, a function pointer (a typedef'd one or a declarator),
 * an array parameter (`T a[]`, `T a[N]`, `T m[][N]`, a VLA), of a void callee, of one returning a pointer or a
 * struct, the named parameter of a variadic callee (whose extra arguments stay integers), however the constant
 * is spelled -- is a null pointer of that parameter's type. Both rails lowered each call to the same claim
 * graph, but each emit passed an `int` temp where the callee takes a pointer, which C forbids (Clang and GCC 14
 * reject it); each emit now declares the constant as the parameter's pointer and returns what the original
 * does. The callees defined after their callers, and those only another unit defines, are in
 * cfront_nullarg_link.c. The callees that call through a function pointer are `static`, so every pointer they
 * are passed is in this unit, and the array lent to calls is at file scope: the G10 escape rows count every
 * indirect call no analysis can narrow and every local array lent to a call (tools/perf/check_escape.py). */
#include <stdint.h>

struct na_node { uint32_t v; struct na_node *next; };
struct na_pair { uint32_t a, b; };
typedef uint32_t (*na_op)(uint32_t);

uint32_t na_deref(const uint32_t *p, uint32_t s) {          /* a pointer (to const) */
  if (p) return *p + s;
  return s ^ 0x5Au;
}
uint32_t na_raw(void *p, char **v, uint32_t s) {            /* a pointer to void, to a pointer */
  uint32_t k = s;
  if (p) k += 1u;
  if (v) k += 2u;
  return k;
}
uint32_t na_twice(uint32_t x) { return x * 2u + 1u; }
static uint32_t na_apply(na_op fn, uint32_t s) {            /* a typedef'd function pointer */
  if (fn) return fn(s);
  return s + 3u;
}
static uint32_t na_apply_decl(uint32_t (*fn)(uint32_t), uint32_t s) {   /* a function-pointer declarator */
  if (fn) return fn(s);
  return s + 5u;
}
uint32_t na_walk(struct na_node *n, uint32_t s) {           /* a pointer to a struct */
  uint32_t k = s;
  if (n) {
    k += n->v;
    if (n->next) k += n->next->v * 3u;
  }
  return k;
}
uint32_t na_sum(uint32_t a[], uint32_t n) {                 /* array parameters -- each one a pointer */
  uint32_t k = 7u;
  if (!a) return k;
  for (uint32_t i = 0u; i < n; i++) k += a[i];
  return k;
}
uint32_t na_last(uint32_t a[4], uint32_t s) {
  if (a) return a[3] + s;
  return s + 9u;
}
uint32_t na_cell(uint32_t m[][2], uint32_t s) {
  if (m) return m[1][0] + s;
  return s + 11u;
}
uint32_t na_vla(uint32_t n, uint32_t v[n]) {
  if (v) return v[n - 1u];
  return n + 13u;
}
void na_store(uint32_t *p, uint32_t s) {                    /* a void callee */
  if (p) *p = s;
}
uint32_t *na_pick(uint32_t *p, uint32_t *q) {               /* a callee returning a pointer */
  if (p) return p;
  return q;
}
struct na_pair na_make(uint32_t *p, uint32_t s) {           /* a callee returning a struct */
  struct na_pair r = {s, s + 1u};
  if (p) r.a = *p;
  return r;
}
uint32_t na_count(uint32_t *p, uint32_t n, ...) {           /* variadic: its extra arguments stay integers */
  if (p) return *p + n;                                      /*   (cfront_nullarg_link.c reads them) */
  return n * 3u;
}

uint32_t na_pointers(uint32_t s) {                          /* the constant however it is spelled */
  uint32_t v = s + 1u;
  uint32_t k = na_deref(0, s) + na_deref(&v, s) * 3u;
  k += na_deref(0u, s) * 5u + na_deref(0L, s) * 7u + na_deref((0), s) * 9u;
  return k + na_raw(0, 0, s) * 11u + na_raw(&v, 0, s) * 13u;
}
uint32_t na_functions(uint32_t s) {
  return na_apply(0, s) + na_apply(na_twice, s) * 3u + na_apply_decl(0, s) * 5u
         + na_apply_decl(na_twice, s) * 7u;
}
uint32_t na_structs(uint32_t s) {
  struct na_node b = {s, 0};
  struct na_node a = {3u, &b};
  return na_walk(0, s) + na_walk(&a, s) * 3u + na_walk(&b, s) * 5u;
}
uint32_t na_a4[4];
uint32_t na_arrays(uint32_t s) {
  na_a4[0] = s; na_a4[1] = s + 1u; na_a4[2] = s + 2u; na_a4[3] = s + 3u;
  uint32_t m[2][2] = {{s, 1u}, {2u, s}};
  uint32_t k = na_sum(0, 4u) + na_sum(na_a4, 4u) * 3u + na_last(0, s) * 5u + na_last(na_a4, s) * 7u;
  return k + na_cell(0, s) * 9u + na_cell(m, s) * 11u + na_vla(3u, 0) * 13u + na_vla(4u, na_a4) * 15u;
}
uint32_t na_calls(uint32_t s) {
  uint32_t v = s, w = s + 2u;
  na_store(0, s);
  na_store(&v, s + 7u);
  uint32_t *p = na_pick(0, &w);
  struct na_pair r = na_make(0, s);
  struct na_pair q = na_make(&w, s);
  uint32_t k = v + *p * 3u + r.a * 5u + r.b * 7u + q.a * 9u;
  k += na_count(0, 2u, 0, 5) * 11u + na_count(&v, 1u, 0) * 13u;
  return k + na_deref(na_pick(0, 0), s) * 15u + na_deref(na_pick(0, &v), s) * 17u;
}
uint32_t na_entry(uint32_t s) {
  return na_pointers(s) + na_functions(s) * 3u + na_structs(s) * 5u + na_arrays(s) * 7u
         + na_calls(s) * 9u;
}
