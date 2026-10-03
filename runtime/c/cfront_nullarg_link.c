/* A null pointer constant passed to a callee the call cannot see the definition of yet (CF-NULLARG): one
 * defined after its caller -- declared first by a prototype, static or not, variadic or not -- and one only
 * prototyped here, which another unit defines (the test's driver links it in). The C twin parses in one pass,
 * so it types these arguments once the unit is parsed; the oracle reads every definition and prototype before
 * it lowers. Both rails lower each call to the same claim graph, and each emit declares the constant as the
 * parameter's pointer -- a pointer, a pointer to a struct, a typedef'd function pointer, an array parameter --
 * while a variadic callee's extra `0`, which it reads as an `int`, stays one. The emit does not declare a
 * function before its definition, so the test supplies the declarations of the callees defined late
 * (bcir/tests/test_c_cfront.py); cfront_nullarg.c holds the callees defined first. `nl_later_node`, which
 * another unit may call, tests its function pointer rather than calling through it, and the arrays lent to
 * calls are at file scope: the G10 escape rows count every indirect call no analysis can narrow and every
 * local array lent to a call (tools/perf/check_escape.py). */
#include <stdint.h>
#include <stdarg.h>

struct nl_node { uint32_t v; struct nl_node *next; };
typedef uint32_t (*nl_op)(uint32_t);

uint32_t nl_ext(uint32_t *p, uint32_t s);                   /* another unit's -- the driver defines them */
uint32_t nl_ext_node(struct nl_node *n, nl_op fn, uint32_t s);
uint32_t nl_ext_sum(uint32_t a[], uint32_t n);
static uint32_t nl_later(uint32_t *p, uint32_t s);          /* defined after its caller */
uint32_t nl_later_node(struct nl_node *n, nl_op fn, uint32_t a[], uint32_t s);
static uint32_t nl_sum(uint32_t *p, uint32_t n, ...);

uint32_t nl_twice(uint32_t x) { return x * 2u + 1u; }

uint32_t nl_fa[2];
uint32_t nl_ca[2];
uint32_t nl_forward(uint32_t s) {
  uint32_t v = s + 4u;
  struct nl_node b = {s, 0};
  nl_fa[0] = s; nl_fa[1] = 3u;
  uint32_t k = nl_later(0, s) + nl_later(&v, s) * 3u;
  k += nl_later_node(0, 0, 0, s) * 5u + nl_later_node(&b, nl_twice, nl_fa, s) * 7u;
  return k + nl_sum(0, 2u, 0, 5) * 9u + nl_sum(&v, 1u, 0) * 11u;
}
uint32_t nl_cross(uint32_t s) {
  uint32_t v = s + 6u;
  struct nl_node b = {s, 0};
  nl_ca[0] = s; nl_ca[1] = 9u;
  uint32_t k = nl_ext(0, s) + nl_ext(&v, s) * 3u + nl_ext_node(0, 0, s) * 5u;
  return k + nl_ext_node(&b, nl_twice, s) * 7u + nl_ext_sum(0, 2u) * 9u + nl_ext_sum(nl_ca, 2u) * 11u;
}
static uint32_t nl_later(uint32_t *p, uint32_t s) {
  if (p) return *p + s;
  return s ^ 0x3Cu;
}
uint32_t nl_later_node(struct nl_node *n, nl_op fn, uint32_t a[], uint32_t s) {
  uint32_t k = s;
  if (n) k += n->v;
  if (fn) k += 17u;
  if (a) k += a[1];
  return k;
}
static uint32_t nl_sum(uint32_t *p, uint32_t n, ...) {
  va_list ap;
  va_start(ap, n);
  uint32_t k = n;
  for (uint32_t i = 0u; i < n; i++) k += (uint32_t)va_arg(ap, int) * (i + 2u);
  va_end(ap);
  if (p) k += *p;
  return k;
}
uint32_t nl_entry(uint32_t s) { return nl_forward(s) + nl_cross(s) * 3u; }
