/* Long-double complex (#complexlong) -- `long double _Complex`, the widest complex type (32 bytes: a pair
 * of the target's long double, 80-bit x87 on x86-64 / 128-bit quad on aarch64). Arithmetic `+ - * /` rides
 * the same native operators as double complex (Clang lowers `/` to __divxc3 etc.), the `l`-suffixed
 * <complex.h> calls type correctly (cabsl real-valued; cexpl/conjl complex), and the imaginary unit
 * composes (a long-double real + `im*I` promotes to `long double _Complex`). Both rails emit identical
 * `long double _Complex` source, so they are equivalent by construction; the bit-exact compare confirms it
 * (identical codegen sets the x87 padding bytes identically on both rails). */
#include <complex.h>

long double _Complex lmul(long double _Complex a, long double _Complex b)
{
    return a * b;                                /* native long-double complex multiply */
}

long double labs2(long double _Complex z)
{
    return cabsl(z);                             /* an `l`-suffixed real-valued <complex.h> call */
}

long double _Complex lentry(long double _Complex a, long double _Complex b, long double s)
{
    long double _Complex p = a * b;              /* native multiply */
    long double _Complex q = a / b;              /* native divide (__divxc3) */
    long double _Complex e = cexpl(a);           /* `l`-suffixed transcendental -> long double complex */
    long double _Complex c = conjl(b);           /* `l`-suffixed algebraic */
    long double _Complex u = s + a * I;          /* long-double real + imaginary unit -> ld complex */
    return p + q + e + c + u;
}
