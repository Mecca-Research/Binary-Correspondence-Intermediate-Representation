/* Variadic functions (#variadic): `RET f(T last, ...)` with <stdarg.h> -- a `va_list` cursor walked by
 * va_start / va_arg / va_end, va_copy (two independent passes), a `va_list` PARAMETER (the vprintf-style
 * helper a variadic wrapper forwards its cursor to), and a same-unit variadic CALL passing arguments past
 * the declared fixed params (the default argument promotions ride on the real emitted call, so Clang
 * applies them). va_start/va_arg/va_end/va_copy lower as opaque builtins (the libm-call mold) emitted
 * verbatim against <stdarg.h>; the result of `va_arg(ap, T)` carries type T. Both rails agree structurally
 * and the emit is Clang-behaviour-equivalent. Float/wide-int vararg += accumulation keeps the operand type
 * (the compound-assignment result-typing); external variadic calls (snprintf-family) live in
 * cfront_extvariadic.c. */

/* the canonical integer accumulator (a printf-style consumer) */
int isum(int n, ...) {
  va_list ap; va_start(ap, n);
  int s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, int);
  va_end(ap);
  return s;
}
/* va_copy: two independent walks over the same argument list */
int twice(int n, ...) {
  va_list ap, aq; va_start(ap, n); va_copy(aq, ap);
  int s = 0; for (int i = 0; i < n; i++) s += va_arg(ap, int);
  int t = 0; for (int i = 0; i < n; i++) t += va_arg(aq, int);
  va_end(aq); va_end(ap);
  return s * 3 + t;
}
/* a `va_list` PARAMETER (a vprintf-style helper) -- the wrapper forwards its own cursor by value */
int vsumv(int n, va_list ap) {
  int s = 0; for (int i = 0; i < n; i++) s += va_arg(ap, int);
  return s;
}
int forward(int n, ...) {
  va_list ap; va_start(ap, n);
  int r = vsumv(n, ap);
  va_end(ap);
  return r;
}
/* a float vararg selected by index (no accumulation) -- va_arg(ap, double) + the default promotion */
double nth(int k, ...) {
  va_list ap; va_start(ap, k);
  double r = 0;
  for (int i = 0; i <= k; i++) r = va_arg(ap, double);
  va_end(ap);
  return r;
}
/* a float vararg ACCUMULATION (+=): the compound desugars to `s = s + va_arg(ap, double)`, and the binary
 * result keeps `double` (the compound-assignment result-typing absorbs the float operand) */
double dsum(int n, ...) {
  va_list ap; va_start(ap, n);
  double s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, double);
  va_end(ap);
  return s;
}
/* a WIDE-int (long) vararg accumulation: the compound keeps the 8-byte width */
long lsum(int n, ...) {
  va_list ap; va_start(ap, n);
  long s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, long);
  va_end(ap);
  return s;
}
/* a same-unit variadic CALL with extra arguments (the call applies the default argument promotions) */
int caller(int a, int b, int c) {
  return isum(3, a, b, c) + twice(2, a, b) + forward(2, b, c)
       + (int)(nth(2, (double)a, (double)b, (double)c) * 2);
}
