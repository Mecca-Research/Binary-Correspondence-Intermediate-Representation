/* Designators of functions another unit defines (CF-EXTDESIG). A function the unit only prototypes is used as a value
 * -- passed, held, selected, stored in a member and a table, compared, its address taken (`&f`) -- as C allows (C11
 * 6.3.2.1p4, 6.5.3.2p3), and each emit declares it `extern` as it declares one the unit calls; a variadic prototype
 * keeps its `...` in that declaration, which the oracle dropped, so a call passing more arguments did not compile. Both
 * rails refused every such designator (`use of undeclared identifier`), and `&f` of any function for reasons of their
 * own. The G10 escape rows count every indirect call whose pointer may hold a function no analysis sees -- one another
 * unit defines among them -- so the pointers made here are returned to the test's driver, which calls them; a call
 * through one in place is the test's own unit (`_EXTDESIG_CALLS`). The driver defines the external functions. */
#include <stdint.h>

typedef uint32_t (*ed_op)(uint32_t);
typedef uint32_t (*ed_vop)(uint32_t, ...);
struct ed_ops { ed_op fn; ed_op alt; };

uint32_t ed_ext(uint32_t v);                  /* defined by the test's driver */
uint32_t ed_ext2(uint32_t v);
uint32_t ed_vext(uint32_t n, ...);            /* a variadic one */
uint32_t ed_apply(ed_op f, uint32_t s);       /* one that takes a function pointer */

static uint32_t ed_inc(uint32_t v) { return v + 1u; }

ed_op ed_pick(uint32_t s) { return s & 1u ? ed_ext : s & 2u ? ed_ext2 : ed_inc; }   /* the arms of `?:` */
ed_op ed_held(uint32_t s) {                   /* held by an initializer and an assignment, `&f` */
  ed_op g = ed_ext;
  ed_op h;
  h = &ed_ext2;
  return s & 4u ? g : h;
}
ed_op ed_member(uint32_t s) { struct ed_ops o = {ed_ext, &ed_inc}; return s & 1u ? o.fn : o.alt; }
ed_op ed_table(uint32_t s) { ed_op t[3] = {ed_ext, &ed_ext2, ed_inc}; return t[s % 3u]; }
ed_vop ed_pick_va(uint32_t s) { return s & 8u ? 0 : ed_vext; }                      /* a variadic one, or none */
uint32_t ed_compare(uint32_t s) {             /* compared with a pointer, with another designator and with 0 */
  ed_op g = ed_pick(s);
  uint32_t k = 0u;
  if (g == ed_ext) k += 1u;
  if (g != &ed_ext2) k += 2u;
  if (ed_ext != 0) k += 4u;
  if (ed_held(s >> 1) == ed_ext) k += 8u;
  if (ed_pick_va(s) == ed_vext) k += 16u;
  return k;
}
uint32_t ed_pass(uint32_t s) { return ed_apply(ed_ext, s) + ed_apply(&ed_inc, s + 1u) * 3u; }   /* to another unit */
uint32_t ed_calls(uint32_t s) {               /* direct calls, through `*f` and `(&f)`, and a variadic one */
  return ed_ext(s) + (*ed_ext)(s) * 3u + (&ed_ext2)(s) * 5u + ed_vext(s, 2u, 3u) * 7u;
}
uint32_t ed_sizes(uint32_t s) { return (uint32_t)sizeof(&ed_ext) + (uint32_t)sizeof ed_ext(s) + s; }
uint32_t ed_entry(uint32_t s) { return ed_compare(s) + ed_pass(s) * 3u + ed_calls(s) * 5u + ed_sizes(s) * 7u; }
