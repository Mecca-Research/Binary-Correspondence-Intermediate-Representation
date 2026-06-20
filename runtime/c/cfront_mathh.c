/* <math.h> core library calls (sqrt/fabs/hypot/floor/fmin/fmax): an opaque external library edge
 * that lowers to c.call.libm and emits the *real* libm function (the harness links -lm, not a
 * bcir_ twin). The result is float for an f-suffixed variant (sqrtf), double for the base (hypot);
 * such calls are not call-graph edges (R18 sees no callee). Parity + Clang behaviour-equivalence.
 * The entry args are non-negative, so no domain NaN / overflow. */
float mf(float a)
{
    return sqrtf(a) + fabsf(a - 3.0f);
}

double md(double x, double y)
{
    double h = hypot(x, y);
    return floor(h) + fmin(x, y) - fmax(x, y) + sqrt(x);
}
