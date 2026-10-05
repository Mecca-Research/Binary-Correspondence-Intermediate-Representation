/* The driver tools/build/emit_kernel.py appends to the oracle's emit_svm_rbf_predict_c kernel (the manifest's
 * kernel_svm): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
#include <math.h>
int main(void) {
  /* one SV at the origin, alpha_y=2, b=0.5, gamma=1: f(x)=2*expf(-||x||^2)+0.5. At x=(1,0): 2*exp(-1)+0.5. */
  float X[2] = {1.0f, 0.0f};
  float SV[2] = {0.0f, 0.0f};
  float AY[1] = {2.0f};
  float f = svm_rbf(X, SV, AY, 0.5f, 1.0f);
  float exp_f = 2.0f * expf(-1.0f) + 0.5f;
  printf("f=%.6f (exp=%.6f)\n", f, exp_f);
  return (fabsf(f - exp_f) < 1e-4f) ? 0 : 1;
}
