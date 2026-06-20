/* The `restrict` pointer qualifier (#restrict): an aliasing hint on pointer parameters (the
 * performance-sensitive driver / kernel / DSP idiom -- non-overlapping buffers). The frontend
 * recognizes restrict / __restrict / __restrict__ as a pointer qualifier and consumes it: BCIR's
 * value model carries no aliasing facts, so the emit is behaviour-equivalent (the hint is advisory).
 * Both rails agree structurally and the emitted C is Clang-behaviour-equivalent. */
unsigned sum2(const unsigned *restrict a, const unsigned *restrict b) { return a[0] + b[0]; }
unsigned dot3(const unsigned *__restrict x, const unsigned *__restrict__ y) { return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]; }
