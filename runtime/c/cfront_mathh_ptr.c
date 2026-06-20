/* <math.h> pointer out-param: modf(value, double *iptr) writes the integral part through the pointer
 * and returns the fractional part. `&ip` lowers to a c.addrof claim emitted as `&ip` (not a dropped
 * `&`); since ip is a double local, &ip is a clean `double *`. frexp/remquo ride the same path.
 * Parity + Clang behaviour-equivalence. */
double mathptr(double x)
{
    double ip;
    double frac = modf(x, &ip);
    return frac + ip;
}
