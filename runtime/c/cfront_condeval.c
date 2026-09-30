/* The operands C may leave unevaluated (C11 6.5.15p4, 6.5.13p4, 6.5.14p4; CF-TERNARY): `c ? a : b` evaluates one
 * arm, and `a && b` / `a || b` evaluate `b` only when `a` does not decide. Both rails had computed every operand
 * first and chosen after -- `d ? n / d : 0u` divided by zero, `p ? *p : s` and `p && *p` read through NULL, a
 * call in the arm C skips ran anyway. An operand that can neither trap nor change state is still computed eagerly
 * (a select); any other lowers as a branch. Arms of type void -- `c ? f() : g();`, an `assert`'s `c ? (void)0 :
 * fail()` -- have no value: the branch runs one for its effects. Each function is run with inputs that make every
 * guard false, and
 * resets the counter it reads first, so a run does not depend on the last; `ce_device` runs only with its guard
 * false (the register is at an address nothing maps). Both rails lower each the same way. */
#include <stdint.h>

static uint32_t ce_calls;
static uint32_t ce_bump(uint32_t x) { ce_calls++; return x + 1u; }
static uint32_t *ce_pick(uint32_t *p) { ce_calls++; return p; }
struct ce_pt { uint32_t x, y; };
static struct ce_pt ce_mk(uint32_t a) { struct ce_pt r = {a, a + 1u}; ce_calls++; return r; }
static double ce_half(uint32_t a) { ce_calls++; return a / 2.0; }

uint32_t ce_div(uint32_t n, uint32_t d) {            /* a guarded division and remainder */
  return (d ? n / d : 0u) + (d ? n % d : 7u) * 3u;
}
int32_t ce_sdiv(int32_t a, int32_t b) {              /* signed: b == -1 with the least a is guarded too */
  return (b != 0 && !(a == -2147483647 - 1 && b == -1)) ? a / b : 0;
}
uint32_t ce_deref(uint32_t *p, uint32_t s) {         /* a NULL-guarded read */
  return p ? *p : s;
}
uint32_t ce_call_arm(uint32_t s) {                   /* a call only in the arm C evaluates */
  ce_calls = 0u;
  uint32_t r = s > 3u ? ce_bump(s) : 7u;
  return r + ce_calls * 100u;
}
uint32_t ce_both_calls(uint32_t s) {                 /* a call in each arm: exactly one runs */
  ce_calls = 0u;
  uint32_t r = (s & 1u) ? ce_bump(s) : ce_bump(s + 5u);
  return r + ce_calls * 1000u;
}
uint32_t ce_and_guard(uint32_t *p, uint32_t s) {     /* `p && *p`: the idiom a NULL p must not reach */
  if (p && *p == 3u) return 1u;
  uint32_t v = p && *p > 1u;
  return v * 10u + s;
}
uint32_t ce_or_call(uint32_t s) {                    /* `||` decides without its right operand */
  ce_calls = 0u;
  uint32_t r = (s < 3u) || ce_bump(s) > 9u;
  return r * 10u + ce_calls;
}
uint32_t ce_and_or(uint32_t s) {                     /* `&&` inside `||`, each short-circuiting */
  ce_calls = 0u;
  uint32_t r = (s > 3u && ce_bump(s) > 4u) || ce_bump(1u) > 1u;
  return r * 10u + ce_calls;
}
uint32_t ce_bounds(uint32_t n) {                     /* a bounds guard in each loop form: never reads a[4] */
  uint32_t a[4] = {3u, 2u, 1u, 9u}, i = 0u, j = 0u, k = 0u;
  while (i < 4u && a[i] != (n & 3u)) i++;
  for (j = 0u; j < 4u && a[j] > (n & 1u); j++) {}
  do { k++; } while (k < 4u && a[k] != 0u);
  return i * 100u + j * 10u + k;
}
uint32_t ce_chain(uint32_t n, uint32_t d) {          /* a chain of guarded arms */
  return d == 0u ? 11u : n > 100u ? n / d : d > 50u ? n % d : n + d;
}
uint32_t ce_null_arm(uint32_t s) {                   /* a pointer result whose other arm is a null pointer */
  uint32_t x = s;
  ce_calls = 0u;
  uint32_t *q = s > 3u ? ce_pick(&x) : 0;
  return (q ? *q : 99u) + ce_calls * 1000u;
}
uint32_t ce_struct_arm(uint32_t s) {                 /* a struct result from a call in one arm */
  struct ce_pt z = {4u, 5u};
  ce_calls = 0u;
  struct ce_pt w = s > 3u ? ce_mk(s) : z;
  return w.x + w.y * 3u + ce_calls * 1000u;
}
uint32_t ce_float_arm(uint32_t s) {                  /* a float result: the int arm converts */
  ce_calls = 0u;
  double v = s > 3u ? ce_half(s & 0xFFFFu) : 1;
  return (uint32_t)(v * 10.0) + ce_calls * 1000u;
}
uint32_t ce_writes(uint32_t s) {                     /* an assignment or an increment in the arm C skips */
  uint32_t x = 0u, y = 0u, i = 5u, j = 9u;
  (s & 1u) ? (x = 1u) : (y = 2u);
  uint32_t v = (s & 2u) ? i++ : j++;
  return x * 10000u + y * 1000u + v * 100u + i * 10u + j;
}
uint32_t ce_comma(uint32_t s) {                      /* a comma operand with a call, in the arm C skips */
  ce_calls = 0u;
  uint32_t v = s > 3u ? (ce_bump(1u), 5u) : 6u;
  return v + ce_calls * 10u;
}
uint32_t ce_nested(uint32_t *p, uint32_t s) {        /* a guard inside a guarded arm, and `!` of one */
  uint32_t r = p ? (*p > 2u ? *p / (s | 1u) : 1u) : 2u;
  return r * 10u + !(p && *p);
}
uint32_t ce_mixed(uint32_t n, uint32_t d) {         /* pure and guarded operands in one function, in both orders */
  uint32_t a = d > 3u ? n : d, b = d ? n / d : 1u, c = n > d ? a : b;
  return a + b * 3u + c * 5u + (d && n % d) + ((n & 1u) || (d & 1u));
}
static void ce_inc(void) { ce_calls++; }
static void ce_dec(void) { ce_calls += 10u; }
static void ce_fail(uint32_t c) { ce_calls += c * 100u; }
static void ce_twice(void) { ce_inc(); ce_inc(); }  /* a void function whose only effect is the calls it makes */
uint32_t ce_void(uint32_t s) {                       /* void arms: the one C evaluates runs, for its effects alone */
  ce_calls = 0u;
  s > 3u ? ce_inc() : ce_dec();
  (s & 1u) ? (void)0 : ce_fail(s & 7u);              /* the shape of an `assert` */
  s > 5u ? (ce_inc(), ce_inc()) : (s ? ce_dec() : (void)0);
  (s & 2u) ? ce_twice() : (void)s;
  s > 9u ? (void)0 : (void)1;                        /* two pure void arms: nothing to evaluate */
  (void)ce_inc();
  return ce_calls;
}
volatile uint32_t ce_vg = 5u;
#define CE_REG (*(volatile uint32_t *)0x40000000u)
uint32_t ce_volatile(uint32_t s) {                   /* a volatile read is an access, made only where C makes it */
  volatile uint32_t x = s;
  return (s > 3u ? x : 0u) + ((s & 1u) ? ce_vg : 2u) + (s > 3u && x);
}
uint32_t ce_device(uint32_t s) {                     /* a device register C does not read is never touched */
  return s <= 3u ? 7u : CE_REG;
}
uint32_t ce_pure(uint32_t s) {                       /* pure operands stay selects: no branch */
  uint32_t a = s + 1u, b = s * 3u;
  return (s > 5u ? a : b) + ((s & 1u) && (s & 2u)) + ((s & 4u) || (a > b));
}
uint32_t ce_entry(uint32_t s) {
  uint32_t v = s & 7u;
  return ce_div(s, s & 3u) + ce_deref(&v, s) + ce_call_arm(s) + ce_and_guard(&v, s) + ce_bounds(s) + ce_pure(s);
}
