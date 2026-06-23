/* C99 `_Complex` numbers (#complex): a complex value is modeled as a float *scalar* carrying an
 * is_complex flag -- it rides every existing float path (binop result typing, no truncation, value
 * delegated to the backend) and the emitter prints the `<elem> _Complex` spelling with NATIVE
 * operators, so Clang lowers `+`/`-`/`*` (and `__mul`) identically in the original AND the re-emitted
 * bcir_*, making them behaviour-equivalent by construction. Exercised here: `double _Complex` params,
 * `+ - *`, mixed complex*real promotion, the GNU `__real__`/`__imag__` part operators, the <complex.h>
 * creal/cimag (real result) and conj (complex result) calls, a complex-RETURNING in-unit call (the
 * call-result is typed complex), and a complex RETURN. (Division, the `I` literal, long-double complex,
 * the complex transcendentals, and complex struct members are follow-ons -- both rails defer them.) */
#include <complex.h>

double _Complex cadd(double _Complex a, double _Complex b)
{
    return a + b;                              /* native complex addition */
}

double _Complex cscale(double _Complex z, double s)
{
    return z * s;                              /* mixed complex * real -> complex (the real promotes) */
}

double _Complex complex_demo(double _Complex a, double _Complex b, double s)
{
    double _Complex sum  = cadd(a, b);         /* a complex-returning in-unit call (result typed complex) */
    double _Complex diff = a - b;
    double _Complex prod = a * b;
    double _Complex scaled = cscale(b, s);     /* mixed-promote inside the callee */
    double _Complex cc = conj(a);              /* <complex.h> conj -> complex */
    double re = creal(prod) + __real__ scaled + creal(sum);   /* creal + GNU __real__ -> real doubles */
    double im = cimag(cc) - __imag__ diff + cimag(b);         /* cimag + GNU __imag__ -> real doubles */
    return sum + prod + cc + (re + im * 2.0);  /* complex + real -> complex (the entry returns complex) */
}
