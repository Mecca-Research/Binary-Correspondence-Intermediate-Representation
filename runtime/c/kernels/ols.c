/* The driver tools/build/emit_kernel.py appends to the oracle's emit_lapack_ols_c kernel (the manifest's
 * kernel_ols): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
#include <math.h>
int main(void) {
  /* fit y = 2x + 1 over 8 points exactly on the line: design rows [1, x_i], b_i = 2*x_i + 1. */
  float A[16], b[8], x[2] = {0.0f, 0.0f};
  for (int i = 0; i < 8; ++i) { A[i*2+0] = 1.0f; A[i*2+1] = (float)i; b[i] = 2.0f*(float)i + 1.0f; }
  ols(A, b, x);                                   /* x = [c0, c1] */
  printf("c0=%.6f c1=%.6f\n", x[0], x[1]);
  /* recovered coefficients must be [1, 2] to float round-off (consistent, well-conditioned). */
  return (fabsf(x[0] - 1.0f) < 1e-3f && fabsf(x[1] - 2.0f) < 1e-3f) ? 0 : 1;
}
