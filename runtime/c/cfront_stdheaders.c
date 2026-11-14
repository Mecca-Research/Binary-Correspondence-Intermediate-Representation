/* CSUB-1: the C subset's conformance rung, behaviour-checked. The standard headers both rails supply (their limits,
 * `INTn_C`, `offsetof`, `max_align_t`, `ptrdiff_t`, `bool`, the `<iso646.h>` spellings and `<math.h>`'s constants and classification macros),
 * static assertions at file, member and block scope, and function-like macro invocations whose arguments -- and
 * whose `(` -- run across lines (C11 6.10.3p10). Each value read through a header must equal what Clang computes
 * through its own: the behaviour-equivalence harness compiles this unit, preprocessed, beside each rail's emit and
 * the real headers. */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <iso646.h>

#define PICK(c, a, b) ((c) ? (a) : (b))
#define MIX(a, b, c) (((a) * 31u) ^ ((b) + (c)))

struct rec {
  uint8_t tag;
  uint16_t len;
  uint32_t vals[3];
  _Static_assert(sizeof(uint16_t) == 2, "a member list takes one");
  uint64_t stamp;
  max_align_t pad;
};
_Static_assert(offsetof(struct rec, vals) == 4, "vals follows len");
_Static_assert(sizeof(int_least16_t) == 2 && sizeof(intmax_t) == 8, "least and max widths");
static_assert(CHAR_BIT * sizeof(uint32_t) == 32);
static_assert(_Alignof(struct rec) == _Alignof(max_align_t),
              "a max_align_t member aligns the record");

uint32_t csub1_limits(uint32_t x) {
  static_assert(INT32_MAX == 2147483647, "a block takes one too");
  uint64_t wide = UINT64_C(1) << 40;
  int64_t neg = INT64_MIN + (int64_t)x;
  uint32_t r = PICK(x > UINT16_MAX,
                    (uint32_t)(wide >> 20),
                    (uint32_t)(neg & 0xff));
  return r + (uint32_t)(SIZE_MAX > UINT32_MAX) + (uint32_t)(PTRDIFF_MAX == INT64_MAX) + (uint32_t)SCHAR_MAX +
         (uint32_t)(INT_MIN < 0) + (uint32_t)(UINT8_C(200) + INT16_C(-3));
}

uint32_t csub1_layout(uint32_t i) {
  size_t off = offsetof(struct rec, vals[2]) + offsetof(struct rec, stamp) + offsetof(struct rec, pad);
  ptrdiff_t d = (ptrdiff_t)i - (ptrdiff_t)off;
  return (uint32_t)off + (uint32_t)sizeof(max_align_t) + (uint32_t)_Alignof(max_align_t) +
         (uint32_t)sizeof(struct rec) + (uint32_t)(d < 0);
}

uint32_t csub1_bool(uint32_t x) {
  bool even = (x & 1u) == 0u;
  bool big = x > 1000u;
  return (uint32_t)(even and not big) + (uint32_t)(true + true) + (uint32_t)sizeof(true) +
         (uint32_t)(big or false) + (uint32_t)(x bitand 7u) + (uint32_t)__bool_true_false_are_defined;
}

uint32_t csub1_float(uint32_t x) {
  float f = (float)x;
  float inf = INFINITY;
  float nan = NAN;
  return (uint32_t)(f < FLT_MAX) + (uint32_t)(inf > FLT_MAX) + (uint32_t)(nan != nan) +
         (uint32_t)(HUGE_VAL == (double)inf) + (uint32_t)(HUGE_VALF == inf) + (uint32_t)FLT_MANT_DIG +
         (uint32_t)DBL_DIG + (uint32_t)(FLT_EPSILON < 1.0f) + (uint32_t)FLT_RADIX;
}

uint32_t csub1_classify(uint32_t x) {
  float f = (float)x - 100.0f;
  double big = (double)x * 1e308;
  float inf = INFINITY, nan = NAN;
  return (uint32_t)(isfinite(f) != 0) + 2u * (uint32_t)(isnan(nan) != 0) + 4u * (uint32_t)(isinf(-inf) < 0) +
         8u * (uint32_t)(isinf(big) != 0) + 16u * (uint32_t)(signbit(f) != 0) + 32u * (uint32_t)(isnormal(f) != 0) +
         64u * (uint32_t)(isfinite(big) != 0) + 128u * (uint32_t)(isnan(f) != 0) +
         /* an integer libm result is signed: the twin had read a negative one as unsigned */
         256u * (uint32_t)(ilogb((double)f * 0.001) < 0) + 512u * (uint32_t)(lround((double)f) < 0);
}

uint32_t csub1(uint32_t x, uint32_t y) {
  return PICK(x > y,
              csub1_limits(x),
              csub1_layout(y))
         + PICK
           (x & 1u, csub1_bool(x), csub1_float(y))
         + csub1_classify(x ^ y)
         + MIX(
               x,
               y >> 3,
               (uint32_t)LONG_MAX
           );
}
