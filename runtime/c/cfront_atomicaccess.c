/* CF-ATOMIC: an access to an `_Atomic` object is ONE atomic operation, wherever the object is --
 * through a pointer, a member, a member array, an array of structs, a subscripted pointer, or a named
 * global. A load or a store keeps its claim and gains the atomic order (lane A, hazard atomic); a compound
 * assignment or an increment is one `c.c11atom.rmw:<op>` read-modify-write, never a load, an operation and
 * a store (a concurrent write between them was lost). Both emits spell every one of them through an
 * `_Atomic` lvalue -- never a byte copy, which is no atomic access at all. Each function writes the object
 * before it modifies it, so a run leaves the same state however often it repeats. The generics type their
 * value by the pointee: a 64-bit `atomic_load` no longer truncates to 32 bits. Integer read-modify-writes
 * only -- a floating one needs libatomic under GCC, in the original source too. */
#include <stdint.h>
#include <stdatomic.h>

struct acc_s {
  uint8_t tag;
  _Atomic uint32_t n;
  _Atomic uint64_t w;
  _Atomic uint16_t h;
  _Atomic _Bool b;
  _Atomic float f;
  _Atomic double d;
};
struct acc_a {
  uint8_t c;
  _Atomic uint32_t arr[4];
  uint16_t t;
};
struct acc_e {
  uint16_t k;
  _Atomic uint32_t v;
};
struct acc_p {
  uint8_t c;
  _Atomic uint32_t *ap;
};

_Atomic uint32_t acc_g;

/* through a pointer: a store, read-modify-writes, the value of a compound assignment, a load */
uint32_t acc_deref(_Atomic uint32_t *p, uint32_t v) {
  *p = v;
  *p += 3u;
  *p ^= v;
  *p *= 3u;
  uint32_t r = (*p -= 1u);
  return r + *p;
}

/* members of every width, a _Bool member that normalizes, the postfix value of a member increment */
uint64_t acc_member(struct acc_s *s, uint32_t v) {
  s->n = v;
  s->n += 5u;
  s->w = v;
  s->w <<= 33u;
  s->h = (uint16_t)v;
  s->h |= 0x100u;
  s->b = v;
  uint32_t a = s->n++;
  return a + s->n + s->w + s->h + s->b;
}

/* floating members: an atomic store and load (no floating read-modify-write) */
double acc_float(struct acc_s *s, float x) {
  s->f = x;
  s->d = x;
  return s->f + s->d;
}

/* a member array's element and an array of structs' field */
uint32_t acc_elems(struct acc_a *a, struct acc_e *e, uint32_t i, uint32_t v) {
  a->arr[i & 3u] = v;
  a->arr[i & 3u] += 7u;
  a->arr[(i + 1u) & 3u] = 2u;
  e[i & 1u].v = v;
  e[i & 1u].v -= 1u;
  return a->arr[i & 3u] + a->arr[(i + 1u) & 3u] + e[i & 1u].v;
}

/* a subscripted pointer to `_Atomic` elements */
uint32_t acc_index(_Atomic uint32_t *p, uint32_t i, uint32_t v) {
  p[i & 3u] = v;
  p[i & 3u] += 9u;
  return p[i & 3u];
}

/* a named global: atomic by declaration, its read-modify-writes single operations */
uint32_t acc_global(uint32_t v) {
  acc_g = v;
  acc_g += 2u;
  acc_g++;
  uint32_t a = acc_g++;
  return a * 3u + acc_g;
}

/* the generics on a 64-bit object: the value is the pointee's type, never a truncating uint32 */
uint64_t acc_load64(_Atomic uint64_t *p, uint64_t v) {
  atomic_store(p, v);
  uint64_t a = atomic_load(p);
  uint64_t b = atomic_fetch_add(p, v);
  return a + b + atomic_exchange(p, v);
}

/* ... and on a float: the value read is a float, never a uint32 conversion of it */
float acc_loadf(_Atomic float *p, float x) {
  atomic_store(p, x);
  float a = atomic_load(p);
  return a + atomic_exchange(p, x * 0.5f);
}

/* through a pointer member: the pointer is a plain member, its pointee `_Atomic` */
uint32_t acc_ptrmember(struct acc_p *s, uint32_t v) {
  *s->ap = v;
  *s->ap += 4u;
  s->ap[1] = v ^ 1u;
  return *s->ap + s->ap[1];
}

/* an `_Atomic _Bool` normalizes every store (C23 6.3.1.2), through a pointer and an element alike */
uint32_t acc_bool(_Atomic _Bool *p, uint32_t i, uint32_t v) {
  *p = v;
  p[(i & 1u) + 1u] = v & 2u;
  return *p + 2u * p[(i & 1u) + 1u];
}

/* one step of three shared counters: each update is one read-modify-write, so none is lost to a
 * concurrent one (the threaded witness runs this from several threads at once) */
void acc_bump(_Atomic uint32_t *p, struct acc_s *s, struct acc_a *a) {
  *p += 1u;
  s->n += 1u;
  a->arr[1] += 1u;
}
