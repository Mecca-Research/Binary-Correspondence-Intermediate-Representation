/* Function-pointer tables, and calls through a dereference or an element (CF-FPTAB). A call takes any postfix
 * expression whose value is a function pointer (C11 6.5.2.2p1): `(*fp)(x)`, `(**fp)(x)` and `(*f)(x)` of a function
 * name the function again, which converts back to the pointer (6.5.3.2p4, 6.3.2.1p4) -- as `*fp` does wherever it is
 * a value (`ft_pick_deref`); `t[i](x)` and `(*t[i])(x)` call through an element. The tables are arrays of function
 * pointers, typedef'd and spelled inline (`uint32_t (*t[N])(uint32_t)`), local, static and file-scope, of one and
 * two dimensions, struct members, pointers to them (`op_t *`, `uint32_t (**p)(uint32_t)`) and parameters of them.
 * Both rails refused `(*fp)(x)` and a call through an element, the twin every inline declarator, and both typed a
 * member read as an integer, which no `?:` compared. The G10 escape rows count every indirect call no analysis
 * resolves to one function (tools/perf/check_escape.py), so each call made here goes through a pointer that holds
 * one: an element is picked and returned to the caller, which calls it (the test's driver), and calling through a
 * table in place is the test's own unit (`_FPTAB_CALLS`). */
#include <stdint.h>

typedef uint32_t (*ft_op)(uint32_t);

static uint32_t ft_inc(uint32_t x) { return x + 1u; }
static uint32_t ft_dbl(uint32_t x) { return x * 2u + 3u; }
static uint32_t ft_neg(uint32_t x) { return 0u - x; }
static uint32_t ft_mix(uint32_t x) { return (x ^ 0x5Au) * 7u; }

static ft_op ft_ops[4] = {ft_inc, ft_dbl, ft_neg, ft_mix};   /* file-scope tables: typedef'd, inline, 2-D */
static uint32_t (*ft_raw[2])(uint32_t) = {ft_mix, ft_inc};
static ft_op ft_grid[2][2] = {{ft_neg, ft_inc}, {ft_mix, ft_dbl}};
struct ft_dev { uint32_t id; ft_op fn[2]; uint32_t (*raw[2])(uint32_t); ft_op one; };   /* member tables */
static struct ft_dev ft_d = {7u, {ft_dbl, ft_neg}, {ft_mix, ft_inc}, ft_dbl};

ft_op ft_pick_global(uint32_t s) { return ft_ops[s & 3u]; }   /* an element as a value: returned, then called */
ft_op ft_pick_raw(uint32_t s) { return *(ft_raw + (s & 1u)); }
ft_op ft_pick_local(uint32_t s) {                     /* a local table, typedef'd and inline */
  ft_op t[3] = {ft_neg, ft_mix, ft_inc};
  uint32_t (*u[2])(uint32_t) = {ft_dbl, ft_mix};
  return s & 8u ? t[s % 3u] : u[(s >> 1) & 1u];
}
ft_op ft_pick_grid(uint32_t s) {                      /* two dimensions */
  ft_op g[2][2] = {{ft_inc, ft_dbl}, {ft_neg, ft_mix}};
  uint32_t (*h[2][2])(uint32_t) = {{ft_mix, ft_neg}, {ft_dbl, ft_inc}};
  return s & 4u ? g[s & 1u][(s >> 1) & 1u] : h[(s >> 1) & 1u][s & 1u];
}
ft_op ft_pick_member(uint32_t s) {                    /* member tables, read as the pointers they hold */
  struct ft_dev *d = &ft_d;
  return s & 1u ? d->fn[(s >> 1) & 1u] : s & 2u ? ft_d.raw[(s >> 2) & 1u] : d->one;
}
ft_op ft_pick_ptr(uint32_t s) {                       /* through pointers to a table: `p[i]`, `*(q + 1)`, `*p` */
  ft_op *p = ft_ops;
  uint32_t (**q)(uint32_t) = ft_raw;
  return s & 1u ? p[(s >> 1) & 3u] : s & 2u ? *(q + 1) : *p;
}
ft_op ft_pick_deref(uint32_t s) {                     /* `*fp` as a value: the function, converted back to its */
  ft_op fp = ft_mix;                                  /* pointer (6.3.2.1p4) -- held, selected and returned */
  ft_op g = *fp;
  ft_op t[2] = {ft_inc, ft_neg};
  if (s & 1u) g = **ft_dbl;
  if (s & 2u) g = *t[(s >> 2) & 1u];
  if (s & 8u) g = s & 16u ? *ft_d.one : *ft_ops[s & 3u];
  return g;
}
static uint32_t ft_is(ft_op *t, uint32_t s, ft_op f) { return t[s & 3u] == f; }   /* table parameters, 2-D inline */
static uint32_t ft_is_grid(uint32_t (*t[2][2])(uint32_t), uint32_t s, ft_op f) { return t[(s >> 1) & 1u][s & 1u] == f; }
static uint32_t ft_is_ptrs(uint32_t (**t)(uint32_t), uint32_t s, ft_op f) { return *(t + (s & 1u)) == f; }
uint32_t ft_params(uint32_t s) {                      /* each element compared where it lands */
  return ft_is(ft_ops, s, ft_neg) + ft_is_grid(ft_grid, s, ft_mix) * 2u + ft_is_ptrs(ft_raw, s, ft_inc) * 4u;
}
uint32_t ft_call_deref(uint32_t s) {                  /* `(*fp)(x)`, `(**fp)(x)`, `(*f)(x)`: each one function */
  ft_op fp = ft_dbl;
  uint32_t (*gp)(uint32_t) = ft_mix;
  return (*fp)(s) + (**fp)(s + 1u) * 3u + (*gp)(s) * 5u + (*ft_inc)(s) * 7u + (***ft_neg)(s) * 11u;
}
uint32_t ft_call_one(uint32_t s) {                    /* an element of a one-entry table, called in place */
  ft_op t[1] = {ft_neg};
  uint32_t (*u[1])(uint32_t) = {ft_inc};
  return t[0](s) + (*t[0])(s + 1u) * 3u + u[0](s) * 5u + (**t)(s + 2u) * 7u;
}
uint32_t ft_compare(uint32_t s) {                     /* elements and member reads compared, a `?:` of them typed */
  ft_op t[2] = {ft_inc, ft_dbl};
  uint32_t k = 0u;
  if (ft_ops[s & 3u] == ft_inc) k += 1u;
  if (t[s & 1u] != ft_dbl) k += 2u;
  if (ft_d.fn[s & 1u] == ft_neg) k += 4u;
  if ((s & 2u ? ft_d.fn[0] : ft_d.one) == ft_dbl) k += 8u;
  if ((s & 4u ? ft_d.raw[1] : ft_neg) == ft_inc) k += 16u;
  if (*ft_raw[s & 1u] == ft_inc) k += 32u;
  return k;
}
uint32_t ft_entry(uint32_t s) { return ft_call_deref(s) + ft_call_one(s) * 13u + ft_compare(s) * 17u + ft_params(s); }
