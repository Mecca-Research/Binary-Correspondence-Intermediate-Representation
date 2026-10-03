/* Casts to function-pointer types and to pointers to `_Atomic` objects, and the forms the emit writes with them
 * (CF-RTFP). A cast names a function-pointer type -- `(R (*)(P))f`, `(op_t)f`, a pointer to one -- or a pointer to an
 * `_Atomic` object; the emit stores a function pointer at a byte offset through a pointer to its own type,
 * `*(R (**)(P))((char *)p + K) = f;` (through a generic `void (**)(void)` slot until GCC at -O2 was seen dropping such
 * a store, a form both rails still read), and reaches an `_Atomic` member at one, `(*(_Atomic T *)((char *)p + K))`:
 * both rails had refused each, so no emit of a unit with a function-pointer or `_Atomic` member re-read. Beside them: a
 * typedef of a table of function pointers, which the twin refused; a compound literal of one and `sizeof` of an array
 * compound literal, which the twin gave the size of a pointer (a silent miscompile); `( E ) = v;` and a braced
 * function-pointer initializer, which the twin refused. The objects pointed at live at file scope -- the G10 escape
 * rows count a local whose address is stored -- and every indirect call holds one function of this unit: calls
 * through a table or a select of two, or through a pointer read from memory, are the test's own unit
 * (`_RTFP_CALLS`), which the G10 rows do not count. */
#include <stdint.h>

typedef uint32_t (*rf_op)(uint32_t);
typedef uint32_t (*rf_tab[2])(uint32_t);            /* a table of function pointers, typedef'd whole */
typedef uint32_t rf_row[3];
struct rf_slot { rf_op fn; uint32_t tag; };         /* the function pointer at byte 0 on every target */
struct rf_am { uint32_t k; _Atomic uint32_t v; };   /* `v` at byte 4 on every target */

static uint32_t rf_inc(uint32_t v) { return v + 1u; }
static uint32_t rf_dbl(uint32_t v) { return v * 2u; }
static struct rf_slot rf_gs = {rf_inc, 3u};
static struct rf_am rf_ga = {1u, 2u};
static uint32_t rf_gw[3] = {4u, 5u, 6u};

uint32_t rf_casts(uint32_t s) {          /* casts to a function-pointer type, a typedef'd one, in a select, of 0 */
  rf_op g = (rf_op)rf_dbl;
  uint32_t (*h)(uint32_t) = (uint32_t (*)(uint32_t))rf_inc;
  rf_op z = (rf_op)0;
  rf_op k = (s & 1u) ? (rf_op)rf_inc : (uint32_t (*)(uint32_t))rf_inc;
  return g(s) + h(s) * 3u + (z == 0) * 5u + k(s) * 7u;
}
uint32_t rf_slots(uint32_t s) {          /* the emit's member accesses at a byte offset: a function pointer, `_Atomic` */
  struct rf_slot *p = &rf_gs;
  struct rf_am *a = &rf_ga;
  *(void (**)(void))((char *)p + 0) = (void (*)(void))rf_dbl;
  uint32_t r = (p->fn == rf_dbl) * 3u;
  *(_Atomic uint32_t *)((char *)a + 4) = s;
  *(_Atomic uint32_t *)((char *)a + 4) += 3u;
  (*(_Atomic uint32_t *)((char *)a + 4))++;
  (*(_Atomic uint32_t *)((char *)a + 4)) -= 1u;
  r += (*(_Atomic uint32_t *)((char *)a + 4)) * 5u;
  *(void (**)(void))((char *)p + 0) = (void (*)(void))rf_inc;
  return r + a->v * 7u + (p->fn == rf_inc) * 11u;
}
uint32_t rf_tables(uint32_t s) {         /* a typedef'd table, its size, a compound literal of one, `typeof` an element */
  rf_tab t = {rf_dbl, rf_dbl};
  __typeof__(t[0]) g = t[1];
  uint32_t r = t[s & 1u](s) + g(s) * 3u + (uint32_t)sizeof(rf_tab) / (uint32_t)sizeof(rf_op) * 5u;
  r += (rf_op[2]){rf_inc, rf_inc}[s & 1u](s) * 7u;
  r += (uint32_t (*[2])(uint32_t)){rf_dbl, rf_dbl}[1](s) * 11u;
  rf_op *q = (rf_op[2]){rf_inc, rf_inc};
  return r + q[1](s) * 13u;
}
uint32_t rf_sizes(uint32_t s) {          /* `sizeof` of an array compound literal is the array's (C11 6.5.2.5p4) */
  uint32_t r = (uint32_t)sizeof((uint32_t[3]){1u, 2u, 3u});
  r += (uint32_t)sizeof((uint8_t[2][5]){{1u}, {2u}}) * 3u;
  r += (uint32_t)sizeof (rf_row){7u, 8u, 9u} * 5u;
  r += (uint32_t)sizeof((uint32_t[]){1u, 2u, 3u, 4u}[1]) * 7u;
  r += (uint32_t)(sizeof(uint32_t (*[3])(uint32_t)) / sizeof(rf_op)) * 11u;
  r += (uint32_t)sizeof((struct rf_am){1u, 2u}) * 13u;
  return r + s;
}
uint32_t rf_braced(uint32_t s) {         /* a braced function-pointer initializer; a null one */
  uint32_t (*f)(uint32_t) = {rf_inc};
  rf_op g = {rf_dbl,};
  uint32_t (*z)(uint32_t) = 0;
  return f(s) + g(s) * 3u + (z == 0) * 5u + (z != f) * 7u;
}
uint32_t rf_parens(uint32_t s) {         /* `( E ) = v;` and `( E ) OP= v;`: the parentheses are redundant */
  uint32_t *p = &rf_gw[0];
  struct rf_slot *o = &rf_gs;
  rf_gw[1] = 2u;
  o->tag = s & 7u;
  (*p) = s;
  (rf_gw[1]) += s;
  (o->tag) <<= 1u;
  (*p++) = s + 1u;
  ((*p)) *= 3u;
  return rf_gw[0] + rf_gw[1] * 3u + rf_gw[2] * 5u + o->tag * 7u;
}
uint32_t rf_entry(uint32_t s) {
  return rf_casts(s) + rf_slots(s) * 3u + rf_tables(s) * 5u + rf_sizes(s) * 7u + rf_braced(s) * 11u
         + rf_parens(s) * 13u;
}
