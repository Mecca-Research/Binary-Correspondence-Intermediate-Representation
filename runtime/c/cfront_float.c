/* Floating-point (float and double): literals, the four arithmetic operators, comparisons, and
 * float params, locals, and returns. The actual IEEE-754 math is delegated to the resident backend:
 * BCIR lowers the dataflow plus an integer cost and emits real floating C, so the result is
 * bit-identical to Clang and the deterministic integer/Q-fixed plan and executor core stay untouched
 * (a floating value never enters a plan decision). */
float favg(float a, float b)
{
    float s = a + b;
    float h = s * 0.5f;
    return h;
}

double dscale(double x, double y)
{
    double w = x * 2.0 + y;
    return w - 1.5e1;
}
