/* <math.h> mixed-arg + integer-return: ldexp/scalbn (double, int exponent) and scalbln (double,
 * long) keep a floating result fixed by the suffix; ilogb returns int -- exactly the 4-byte value
 * model (a lossless round-trip), declared uint32_t in the emit. The arg is non-negative, so
 * x + 1.0 >= 1 and ilogb is >= 0 (no FP_ILOGB0). Parity + Clang behaviour-equivalence. */
double mix(double x)
{
    double s = ldexp(x, 3) + scalbn(x, 2) + scalbln(x, 4L);
    return s + (double)ilogb(x + 1.0);
}
