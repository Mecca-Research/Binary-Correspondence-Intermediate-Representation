/* The <complex.h> imaginary unit `I` (#imagunit) -- the macro `_Complex_I`, a `const float _Complex` of
 * value i (0+1i). The frontend does not preprocess <complex.h>, so `I` reaches the lowerer as a bare
 * identifier; it (and the canonical `_Complex_I` spelling) is recognized as a complex constant and emitted
 * VERBATIM via a `c.cconst`. The re-emitted bcir_* compiles against <complex.h> exactly as the original
 * does, so the two resolve to the identical imaginary unit and are equivalent by construction (the bit-exact
 * complex compare confirms it). `float _Complex * double` promotes to `double _Complex` per the usual
 * arithmetic conversions. (Recognition is only honored when `I` is NOT a declared name -- but note any TU
 * that includes <complex.h> already makes `I` a macro, so a variable named `I` is not expressible there;
 * the guard is defensive. Clang doesn't implement `_Imaginary`, so `_Imaginary_I` is not recognized.) */
#include <complex.h>

double _Complex cpure(double im)
{
    return im * _Complex_I;                      /* the canonical `_Complex_I` spelling -> a pure imaginary */
}

double _Complex crotate(double _Complex z)
{
    return z * I;                                /* multiply by i: a 90-degree rotation in the plane */
}

double _Complex cbuild(double re, double im)     /* the entry: the textbook complex constructor re + im*I */
{
    return re + im * I;
}
