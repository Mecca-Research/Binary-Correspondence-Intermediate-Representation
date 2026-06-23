/* Complex division `/` (#complexdiv) -- the one complex arithmetic operator held back from #complex.
 * It is NOT a feature gap: `/` rides the same `c.bin.div` claim as the others and emits the native
 * `a / b`, which Clang lowers to __divdc3 (Annex-G inf/nan handling) identically in the original AND the
 * re-emitted bcir_*, so they are equivalent by construction. The only reason it waited is the generic
 * equivalence harness compared a complex result with `!=`, and a near-zero divisor can yield nan/inf
 * where `nan != nan` is spuriously true -- so the harness now memcmp's a complex result (BIT-exact: both
 * rails run the identical operation, so inf/nan/signed-zero compare equal). Exercised: complex / complex,
 * complex / real, and real / complex (element promotion), plus a complex-returning in-unit call. */
#include <complex.h>

double _Complex cdiv(double _Complex a, double _Complex b)
{
    return a / b;                              /* complex / complex */
}

double _Complex complexdiv_demo(double _Complex a, double _Complex b, double s)
{
    double _Complex q1 = cdiv(a, b);           /* a complex-returning in-unit call (result typed complex) */
    double _Complex q2 = a / s;                /* complex / real -> complex */
    double _Complex q3 = s / b;                /* real / complex -> complex (element promotion) */
    return q1 + q2 + q3;
}
