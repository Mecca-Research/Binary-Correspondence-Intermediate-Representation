/* Function pointers that calls return, and what a call through a function pointer returns (CF-FPRET). A call to a
 * function returning a function pointer yields that pointer -- held, compared, selected and returned -- where both
 * rails typed it `uint32_t`: an emit no compiler takes, whose comparisons read a pointer cut to 32 bits. A call through
 * a function pointer returns what the function's type says: a pointer (`T *(*pf)(T *)`, whose typedef the oracle did
 * not parse and whose result the twin typed as an integer), a struct whose member is read (`m(s).a`, which the twin
 * refused), a function pointer. A pointer to a variadic function, `uint32_t (*g)(uint32_t, ...)`, which both rails
 * refused, is declared, selected and called, and a variadic function named in a `?:` arm is that pointer, not an
 * integer. The G10 escape rows count every indirect call no analysis resolves to one function
 * (tools/perf/check_escape.py), so each call made here goes through a pointer that holds one: a returned pointer is
 * returned to the caller, which calls it (the test's driver), and calling through one in place is the test's own unit
 * (`_FPRET_CALLS`). */
#include <stdint.h>
#include <stdarg.h>

typedef uint32_t (*fr_op)(uint32_t);
typedef fr_op (*fr_mk)(uint32_t);
typedef uint32_t *(*fr_pf)(uint32_t *);
typedef uint32_t (*fr_vop)(uint32_t, ...);
struct fr_pair { uint32_t a, b; };
fr_op fr_late(uint32_t s);                            /* declared by a prototype, defined after its callers */

static uint32_t fr_inc(uint32_t x) { return x + 1u; }
static uint32_t fr_dbl(uint32_t x) { return x * 2u + 3u; }
static uint32_t fr_neg(uint32_t x) { return 0u - x; }
static uint32_t *fr_id(uint32_t *p) { return p; }
static struct fr_pair fr_mkpair(uint32_t s) { struct fr_pair p = {s ^ 0x5Au, s * 7u}; return p; }
static uint32_t fr_va(uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  uint32_t r = n + va_arg(ap, uint32_t);
  va_end(ap);
  return r;
}
static uint32_t fr_vb(uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  uint32_t r = n * va_arg(ap, uint32_t) + 1u;
  va_end(ap);
  return r;
}

fr_op fr_pick(uint32_t s) { return s & 1u ? fr_inc : s & 2u ? fr_dbl : fr_neg; }   /* a function-pointer return */
fr_op fr_pick_held(uint32_t s) {                      /* a call's function pointer held, and one through a pointer to */
  fr_op g = fr_pick(s >> 1);                          /* a function returning one */
  fr_mk m = fr_pick;
  fr_op h = m(s + 1u);
  return s & 4u ? g : h;
}
fr_vop fr_pick_va(uint32_t s) { return s & 1u ? fr_va : fr_vb; }   /* variadic functions as the arms of `?:` */
fr_op fr_pick_late(uint32_t s) { fr_op g = fr_late(s + 2u); return g; }   /* through the prototype */
uint32_t fr_compare(uint32_t s) {                     /* the returned pointers compared, where both rails cut them */
  fr_mk m = fr_pick;
  uint32_t k = 0u;
  if (fr_pick(s) == fr_inc) k += 1u;
  if (fr_pick(s >> 1) != fr_neg) k += 2u;
  if (!fr_pick(s)) k += 4u;
  if (fr_pick_va(s) == fr_va) k += 8u;
  if (m(s + 1u) == fr_dbl) k += 16u;
  if ((s & 8u ? fr_pick(s) : fr_dbl) == fr_pick(s >> 2)) k += 32u;
  return k;
}
uint32_t fr_ptr_ret(uint32_t s) {                     /* a pointer returned through a function pointer */
  fr_pf h = fr_id;
  uint32_t *(*k)(uint32_t *) = fr_id;
  uint32_t v = s, w = s ^ 3u;
  return *h(&v) + *k(&w) * 3u + (*(*k)(&v) == v) * 5u;
}
uint32_t fr_struct_ret(uint32_t s) {                  /* a struct returned through a function pointer, its member read */
  struct fr_pair (*m)(uint32_t) = fr_mkpair;
  return m(s).a + (*m)(s).b * 3u + m(s + 1u).b * 5u;
}
uint32_t fr_variadic(uint32_t s) {                    /* calls through pointers to variadic functions */
  fr_vop g = fr_va;
  uint32_t (*h)(uint32_t, ...) = fr_vb;
  return g(s, 3u) + h(s, 5u) * 7u + (*g)(s + 1u, 9u) * 11u;
}
fr_op fr_late(uint32_t s) { return s & 4u ? fr_neg : fr_inc; }
uint32_t fr_entry(uint32_t s) { return fr_compare(s) + fr_ptr_ret(s) * 3u + fr_struct_ret(s) * 5u + fr_variadic(s) * 7u; }
