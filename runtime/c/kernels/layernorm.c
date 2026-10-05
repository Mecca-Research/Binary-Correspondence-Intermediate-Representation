/* The driver tools/build/emit_kernel.py appends to the oracle's emit_layernorm_c kernel (the manifest's
 * kernel_layernorm): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
#include <math.h>
int main(void) {
  /* two rows over a dim=4 feature axis; gamma=1/beta=0 -> a pure normalization (mean 0, var 1 per row). */
  float X[8] = {1.0f, 2.0f, 3.0f, 4.0f,  -2.0f, 0.0f, 2.0f, 4.0f};
  float G[4] = {1.0f, 1.0f, 1.0f, 1.0f}, B[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float O[8] = {0};
  ln(X, G, B, 1e-5f, O);                         /* out[r*4+c] = (X[r,c]-mean)/sqrtf(var+eps) */
  int ok = 1;
  for (int r = 0; r < 2 && ok; ++r) {
    float mean = 0.0f; for (int c = 0; c < 4; ++c) mean += O[r*4+c]; mean /= 4.0f;
    float var = 0.0f;  for (int c = 0; c < 4; ++c) { float d = O[r*4+c] - mean; var += d*d; } var /= 4.0f;
    printf("r%d mean=%.6f var=%.6f\n", r, mean, var);
    if (fabsf(mean) >= 1e-3f || fabsf(var - 1.0f) >= 1e-2f) ok = 0;   /* normalized: mean~0, var~1 */
  }
  return ok ? 0 : 1;
}
