/* long double (#longdouble): the extended floating type (80-bit on x86-64, ABI-sized: 16/16, 12/4 on
 * ILP32, or 8/8 where it aliases double). The C twin's value model has no native 80/128-bit float, so
 * -- exactly like float/double -- it emits real `long double` C and lets the backend / Clang do the
 * arithmetic; this closes a twin parity gap (the twin previously could not even parse `long double`,
 * while the oracle already supported it end to end). Covers arithmetic, an `L`-suffixed constant, the
 * usual conversions to/from double and int, a `long double *`, the `+l` libm variants (sqrtl/fabsl), and
 * `+=` accumulation (the wide compound-assign result type). Both rails agree and the emit == Clang. */

long double ld_arith(long double a, long double b)   { long double s = a + b; s *= 2.0L; s -= a / 3.0L; return s; }
long double ld_promote(double a, int b)              { return (long double)a * b + 0.5L; }
int         ld_to_int(long double a, long double b)  { long double t = a * b; return (int)t; }
double      ld_narrow(long double a)                 { return (double)(a * 1.5L); }
long double ld_libm(long double a)                   { return sqrtl(a * a + 1.0L) + fabsl(a); }
long double ld_ptr(long double a)                    { long double v = a * 2.0L; long double *p = &v; *p += 1.0L; return *p; }
long double ld_acc(int n, long double a)             { long double s = 0; for (int i = 0; i < n; i++) s += a; return s; }
