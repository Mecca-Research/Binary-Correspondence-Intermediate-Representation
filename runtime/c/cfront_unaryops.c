/* Unary operators, controlling expressions and void values as C types them (CF-UNARY, CF-STRUCTCOND, CF-VOIDVAL).
 * Both parsers had dropped a unary `+`, so `sizeof(+c)`, `_Generic(+h, ...)` and `typeof(+h)` read the operand's own
 * type where C reads the promoted one (C11 6.5.3.3p2); the twin gave `-z` of a complex a real temp and promoted a
 * `_BitInt` under `-` and `~`, and typed `__real__ c` of an integer a `double`; the oracle typed `__real__ z` the
 * complex under `sizeof` and `_Generic`, a bit-field operand its declared `unsigned` and `&g` as `g` itself -- each a
 * silent miscompile. Beside them: an array compared with 0 was an `int` temp in both emits, `void *vp = malloc(4u)`
 * carried a bounds extent on the twin alone, `++(x)` was refused by the twin, and the oracle's emit spelled a
 * file-scope object read only by a condition, a `switch` or an early `return` as an undeclared temp. Each form here
 * lowers to one claim graph on the four targets and runs as the original under Clang and GCC. */
#include <stdint.h>
#include <stdlib.h>
#include <complex.h>

struct uo_bf { unsigned a : 3; signed b : 4; unsigned w : 32; };
static uint32_t uo_g = 5u;
static uint32_t uo_arr[4] = {1u, 2u, 3u, 4u};
static uint32_t uo_m2[2][2] = {{1u}, {2u}};
static uint32_t uo_n;
static uint32_t uo_c;
static void uo_add(uint32_t x) { uo_n += x; }
static void uo_setc(uint32_t x) { uo_c = x & 3u; }

uint32_t uo_plus(uint32_t s) {           /* `+a` promotes: `sizeof`, `_Generic` and `typeof` read the promoted type */
  uint8_t c = (uint8_t)s;
  uint16_t h = (uint16_t)s;
  _Bool b = s & 1u;
  char ch = (char)s;
  __typeof__(+h) k = -1;
  uint32_t r = (uint32_t)sizeof(+c) + (uint32_t)sizeof(+b) * 3u + (uint32_t)sizeof(+ch) * 5u;
  r += _Generic(+h, int: 7u, unsigned: 11u, default: 13u);
  r += (uint32_t)(k < 0) * 17u + (uint32_t)(+c) + (uint32_t)(+(+h)) * 19u;
  return r + (uint32_t)sizeof(+ +c) * 23u + (uint32_t)(+c - 300) * 29u;
}
uint32_t uo_parts(uint32_t s) {          /* `-` and `~` of a complex are complex; `__real__` and `__imag__` a part */
  uint32_t t = s & 255u;                 /* (a byte keeps each part converted below in `uint32_t`'s range: converting
                                          * a negative one is undefined, C11 6.3.1.4p1 -- AArch64 saturates it, x86-64
                                          * wraps it) */
  double _Complex z = t + (2.0 * t) * I;
  float _Complex fz = t + (3.0f * t) * I;
  uint8_t c = (uint8_t)s;
  z = -z;
  fz = ~fz;
  uint32_t r = (uint32_t)(__imag__ z + 1000.0) + (uint32_t)(__real__ z + 1000.0) * 3u;
  r += (uint32_t)(__imag__ fz + 1000.0f) * 5u;
  r += (uint32_t)sizeof(__real__ z) * 7u + (uint32_t)sizeof(__imag__ fz) * 11u + (uint32_t)sizeof(__real__ c) * 13u;
  r += _Generic(__real__ z, double: 17u, default: 19u) + _Generic(__real__ c, int: 23u, uint8_t: 29u, default: 31u);
  return r + (uint32_t)(__real__ c) + (uint32_t)(__imag__ c) * 37u;
}
uint32_t uo_bitfields(uint32_t s) {      /* a bit-field operand has the type of its value: `int` when narrower */
  struct uo_bf x = {5u, -3, 7u};
  x.a = s;
  x.w = s;
  uint32_t r = _Generic(x.a - 1, int: 1u, unsigned: 2u, default: 3u);
  r += _Generic(-x.a, int: 5u, unsigned: 7u, default: 11u) * 3u;
  r += _Generic(x.a << 1, int: 13u, unsigned: 17u, default: 19u) * 5u;
  r += _Generic(x.w - 1, int: 23u, unsigned: 29u, default: 31u) * 7u;
  return r + (uint32_t)sizeof(+x.a) + (uint32_t)(-x.a < 0) * 37u + (uint32_t)(+x.b) * 41u;
}
uint32_t uo_addr(uint32_t s) {           /* `&x` is a pointer to x's type */
  uint32_t v = s;
  struct uo_bf x = {1u, 1, 2u};
  __typeof__(&uo_g) p = &uo_g;
  uint32_t r = _Generic(&uo_g, uint32_t *: 1u, default: 2u) + _Generic(&uo_arr[1], uint32_t *: 3u, default: 5u) * 3u;
  r += _Generic(&x, struct uo_bf *: 7u, default: 11u) * 5u + _Generic(&v, uint32_t *: 13u, default: 17u) * 7u;
  return r + *p + s;
}
uint32_t uo_null(uint32_t s) {           /* an array compared with 0 is the pointer it decays to; `void *` of malloc */
  void *vp = malloc(4u);                 /* (the arrays at file scope: the G10 rows count a local whose address is
                                          * taken; the test's own units compare local arrays and a VLA) */
  uint32_t r = (uo_arr == 0) + (uo_arr != 0) * 2u + (0 == uo_arr) * 4u + (uo_m2 != 0) * 8u + (0 != uo_m2) * 16u;
  r += (vp != 0) * 32u;
  free(vp);
  return r + s;
}
uint32_t uo_conds(uint32_t s) {          /* scalar controlling expressions; a `switch` of an integer */
  uint32_t *p = &uo_g;
  double d = s;
  _Bool b = s & 1u;
  char c = (char)(s & 7u);
  uint32_t r = 0u;
  if (p)
    r += 1u;
  if (d)
    r += 2u;
  while (d < 0.0)
    d += 1.0;
  switch (b) { case 1: r += 16u; break; default: r += 32u; }
  switch (c) { case 3: r += 64u; break; default: r += 128u; }
  return r + (d ? 256u : 512u);
}
uint32_t uo_gconds(uint32_t s) {         /* a file-scope object read only as a condition, a discriminant, a return */
  uint32_t r = 0u;
  uo_setc(s);
  if (uo_c)
    r += 1u;
  switch (uo_c) { case 1: r += 2u; break; case 2: r += 4u; break; default: r += 8u; }
  for (; uo_c;) {
    r += 16u;
    break;
  }
  while (uo_c) {
    r += 32u;
    break;
  }
  do {
    r += 64u;
    if (r > 128u)
      break;
  } while (uo_c);
  if (uo_arr)
    r += 256u;
  if (s & 4u)
    return uo_c;
  return r;
}
uint32_t uo_steps(uint32_t s) {          /* `++(x)` and `(x)--`: the parentheses are redundant */
  uint32_t x = s;
  uint32_t *p = &uo_g;
  uint32_t *q = &uo_arr[0];
  uint32_t a[2] = {1u, 2u};
  ++(x);
  (x)--;
  ++((x));
  uint32_t y = ++(x) * 2u;
  ++(a[1]);
  ++(q[1]);
  uint32_t k = q[1];
  --(q[1]);
  --(*p);
  ++(*p);
  return x + y + a[1] + k + *p;
}
uint32_t uo_voids(uint32_t s) {          /* a void expression where C reads no value */
  uint32_t i = 0u;
  uo_n = 0u;
  uo_add(s);
  (void)uo_add(1u);
  uint32_t k = (uo_add(2u), s + 1u);
  s ? uo_add(3u) : uo_add(4u);
  s ? uo_add(5u) : (void)0;
  for (uo_add(6u); i < 2u; i++) {
    uo_add(i);
  }
  ({ uo_add(7u); });
  return k + uo_n;
}
uint32_t uo_entry(uint32_t s) {
  return uo_plus(s) + uo_parts(s) * 3u + uo_bitfields(s) * 5u + uo_addr(s) * 7u + uo_null(s) * 11u
         + uo_conds(s) * 13u + uo_steps(s) * 17u + uo_voids(s) * 19u + uo_gconds(s) * 23u;
}
