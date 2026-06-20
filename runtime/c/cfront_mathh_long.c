/* <math.h> 8-byte integer returns: lround/lrint return long (llround/llrint return long long). The
 * result temp is 8 bytes, declared uint64_t in the emit -- a lossless round-trip to the function's
 * long return, so the 8-byte value is not truncated to the 4-byte model. The arg is non-negative, so
 * the rounded longs are small positives. Parity + Clang behaviour-equivalence. */
long mathlong(double x)
{
    return lround(x) + lrint(x * 2.0);
}
