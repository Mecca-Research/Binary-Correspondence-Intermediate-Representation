/* Typed user-call returns: a void callee lowers to a c.call.void claim emitted as a bare statement
 * (touch(x);), not an invalid `uint32_t t = bcir_touch(...)`; a double-returning callee gives a
 * double result temp (dbl(x) -> `double t = bcir_dbl(x)`). The callees are defined before the entry,
 * so the entry is typed by them. Parity + Clang behaviour-equivalence. */
void touch(double a)
{
    double b = a + 1.0;        /* a void function (no observable effect) */
}

double dbl(double a)
{
    return a * 2.0;
}

double entry(double x)
{
    touch(x);                  /* void call -> a bare statement */
    return dbl(x) + 1.0;       /* double user return -> a double result temp */
}
