/* Qualifiers below a type's top level, kept (CF-QUALS). What a pointer points to and each `*` under the outermost --
 * `const char *const *v` -- and the qualifiers of a function pointer's own parameters and return are part of a function
 * type (C11 6.7.6.1p2, 6.7.6.3p15): an `extern` declaration that drops one declares another type, which conflicts
 * with the prototype it repeats, and a function pointer that drops one is not the type of the function assigned to it.
 * Both rails had dropped every qualifier past the first level, in their `extern` declarations and in the function
 * pointer types they spell, so no emit of a unit naming one compiled together with its prototypes. The emit spells its
 * own objects without qualifiers, so an argument passed where C converts none -- `char **` to `const char *const *`
 * (6.5.16.1p1) -- and a qualified result are cast at the call. The driver defines the external functions. The objects
 * passed live at file scope and every indirect call holds one function of this unit: a local lent to a call, and a
 * call through a member, are the test's own unit (`_QUALS_CALLS`), which the G10 escape rows do not count. */
#include <stdint.h>

typedef uint32_t (*ql_cop)(const uint32_t *);           /* a function type whose parameter points to `const` */
typedef uint32_t (*ql_cnt)(const char *const *, uint32_t);
typedef char *ql_str;
typedef uint32_t (*ql_op)(uint32_t);

uint32_t ql_count(const char *const *v, uint32_t n);     /* defined by the test's driver */
uint32_t ql_first(char *const *v);
uint32_t ql_cpp(const char **v);
uint32_t ql_pp(const uint32_t *const *const pp, uint32_t n);
uint32_t ql_set(char *restrict *pp, uint32_t n);
uint32_t ql_apply(uint32_t (*fn)(const uint32_t *), const uint32_t *p);
const uint32_t *ql_tab(uint32_t i);
const char *const *ql_names(uint32_t i);
uint32_t ql_strs(const ql_str *v);                       /* `char *const *`: a `const` pointer typedef */
uint32_t ql_ops(ql_op const *pp, uint32_t n);            /* `uint32_t (*const *)(uint32_t)` */

static const char *ql_gnames[2] = {"ab", "c"};
static char ql_gchars[2] = {'x', 0};
static char *ql_gv[1] = {ql_gchars};
static const char *ql_gw[1] = {"r"};
static const uint32_t ql_gb[2] = {4u, 1u};
static const uint32_t *ql_gp[2] = {&ql_gb[0], &ql_gb[1]};
static char ql_gc = 'q';
static char *ql_gpc = &ql_gc;

static uint32_t ql_inc(uint32_t v) { return v + 1u; }
static ql_op ql_gops[2] = {ql_inc, ql_inc};
static uint32_t ql_sum(const uint32_t *p) { return p[0] + p[1]; }
static uint32_t ql_cnt_impl(const char *const *v, uint32_t n) {
  uint32_t k = 0u;
  for (uint32_t i = 0u; i < n; i++) k += (uint32_t)v[i][0];
  return k;
}

uint32_t ql_externs(uint32_t s) {          /* arguments to qualified parameters of functions another unit defines */
  return ql_count(ql_gnames, 2u) + ql_first(ql_gv) * 3u + ql_cpp(ql_gw) * 5u + ql_pp(ql_gp, 2u) * 7u
         + ql_set(ql_gv, 1u) * 11u + s;
}
uint32_t ql_results(uint32_t s) {          /* ... and their qualified results */
  const uint32_t *t = ql_tab(s);
  const char *const *n = ql_names(s);
  return t[0] + (uint32_t)n[0][0] + s;
}
uint32_t ql_fnptrs(uint32_t s) {           /* function pointers whose parameters keep their qualifiers */
  ql_cop g = ql_sum;
  uint32_t (*h)(const uint32_t *) = ql_sum;
  ql_cnt c = ql_cnt_impl;
  uint32_t (*d)(const char *const *, uint32_t) = ql_cnt_impl;
  ql_cnt tab[2] = {ql_cnt_impl, ql_cnt_impl};
  uint32_t (*const k)(const uint32_t *) = ql_sum;        /* a `const` function pointer */
  return g(ql_gb) + h(ql_gb) * 3u + c(ql_gnames, 2u) * 5u + d(ql_gnames, 1u) * 7u + tab[s & 1u](ql_gnames, 2u) * 11u
         + ql_apply(ql_sum, ql_gb) * 13u + k(ql_gb) * 17u + s;
}
uint32_t ql_typedefs(uint32_t s) {         /* a qualifier on a pointer typedef qualifies the pointer */
  const ql_str *q = &ql_gpc;               /* `char *const *` */
  char *const *r = &ql_gpc;
  ql_str const t = ql_gpc;
  return (uint32_t)**q + (uint32_t)**r * 3u + (uint32_t)*t * 5u + ql_strs(ql_gv) * 7u + ql_ops(ql_gops, 2u) * 11u
         + s;
}
uint32_t ql_entry(uint32_t s) {
  return ql_externs(s) + ql_results(s) * 3u + ql_fnptrs(s) * 5u + ql_typedefs(s) * 11u;
}
