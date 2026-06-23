/* Complex transcendentals (#complextrans) -- the C99 <complex.h> elementary functions. Each is lowered
 * like a libm call (c.call.libm, an opaque external edge, emitted VERBATIM against <complex.h>) and typed
 * by name: cexp/clog/csqrt/cpow and the circular/hyperbolic families (+ their inverses) all return the
 * *complex* type of the argument's width, while cabs/carg (covered by #complex) return the real element.
 * Because the call is emitted byte-for-byte in both the original and the bcir_* twin, the two link the
 * SAME libm symbol and are equivalent by construction; the harness compares the complex result bit-exactly
 * (memcmp, nan/inf-safe -- clog(0), csqrt of a negative, etc. land on identical inf/nan bits on both rails).
 * The `f`-suffixed forms exercise the `float _Complex` typing path (suffix -> element width). */
#include <complex.h>

double _Complex ctrans(double _Complex z, double _Complex w)
{
    double _Complex e = cexp(z);                              /* exp */
    double _Complex l = clog(z);                              /* log */
    double _Complex r = csqrt(z);                             /* sqrt */
    double _Complex p = cpow(z, w);                           /* power (two complex args) */
    double _Complex c = csin(z) + ccos(z) + ctan(z);          /* circular */
    double _Complex h = csinh(z) + ccosh(z) + ctanh(z);       /* hyperbolic */
    double _Complex a = casin(z) + cacos(z) + catan(z);       /* inverse circular */
    double _Complex b = casinh(z) + cacosh(z) + catanh(z);    /* inverse hyperbolic */
    return e + l + r + p + c + h + a + b;
}

float _Complex ctransf(float _Complex z)                      /* f-suffix -> float _Complex result */
{
    return cexpf(z) + clogf(z) + csqrtf(z) + csinf(z) + ctanhf(z);
}
