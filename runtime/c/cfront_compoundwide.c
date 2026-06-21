/* Wide / floating compound assignment (#compoundwide): an `OP=` (or ++/--) on a `long`/`double` lvalue
 * keeps the operand's width and float-ness in its result, instead of truncating to a 4-byte uint32. The
 * regular binary-op path already typed `a + b` correctly; the ~9 compound-assignment / ++/-- sites shared
 * none of that and hard-coded a 4-byte integer temp, so `long s; s += x` truncated to 32 bits and
 * `double s; s += x` truncated to int (UB for negatives/fractions). All sites now route through the same
 * usual-arithmetic-conversion typing. This also unblocks float / wide-int variadic accumulation
 * (`s += va_arg(ap, double)` / `va_arg(ap, long)`), the natural #variadic follow-on. Twin-only fix (the
 * oracle was already correct); both rails agree and the emit is Clang-behaviour-equivalent. */

struct S { long m; double d; };

long   l_local(long x, long y)    { long s = x; s += y; s *= 3; s -= 7; return s; }
double d_local(double x, double y){ double s = x; s += y; s *= 2.0; s /= 4.0; return s; }
long   l_inc(long x)              { long s = x; s++; ++s; s--; return s; }
long   l_member(long x, long y)   { struct S s; s.m = x; s.m += y; s.m *= 2; return s.m; }
long   l_array(long x, long y)    { long a[2]; a[0] = x; a[1] = 0; a[0] += y; a[1] = a[0] << 1; return a[0] + a[1]; }
long   l_ptr(long x, long y)      { long v = x; long *p = &v; *p += y; *p <<= 1; return v; }

/* float vararg accumulation via `+=` -- needs the wide/float compound-assign result type */
double d_vararg(int n, ...) {
  va_list ap; va_start(ap, n);
  double s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, double);
  va_end(ap);
  return s;
}
/* wide-int vararg accumulation via `+=` */
long l_vararg(int n, ...) {
  va_list ap; va_start(ap, n);
  long s = 0;
  for (int i = 0; i < n; i++) s += va_arg(ap, long);
  va_end(ap);
  return s;
}

long driver(long x, long y) {
  return l_local(x, y) + l_inc(x) + l_member(x, y) + l_array(x, y) + l_ptr(x, y)
       + (long)(d_local((double)x, (double)y) * 8.0)
       + (long)(d_vararg(3, (double)x, (double)y, 1.5) * 4.0)
       + l_vararg(2, x, y);
}
