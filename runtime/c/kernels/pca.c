/* The driver tools/build/emit_kernel.py appends to the oracle's emit_lapack_eigh_c kernel (the manifest's
 * kernel_pca): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
#include <math.h>
int main(void) {
  /* a hand-built symmetric matrix diag(5,3,1): eigenvalues [5,3,1] DESCENDING, eigenvectors = standard basis. */
  float C[9] = {5.0f, 0.0f, 0.0f,  0.0f, 3.0f, 0.0f,  0.0f, 0.0f, 1.0f};
  float vals[3] = {0}, vecs[9] = {0};
  eigh(C, vals, vecs);                            /* vals descending, vecs[t*3+j] = component t coord j */
  printf("l0=%.6f l1=%.6f l2=%.6f\n", vals[0], vals[1], vals[2]);
  int ok = fabsf(vals[0] - 5.0f) < 1e-4f && fabsf(vals[1] - 3.0f) < 1e-4f && fabsf(vals[2] - 1.0f) < 1e-4f;
  for (int t = 0; t < 3 && ok; ++t)               /* eigenvectors are the standard basis, sign convention */
    for (int j = 0; j < 3; ++j) {
      float want = (j == t) ? 1.0f : 0.0f;
      if (fabsf(fabsf(vecs[t*3+j]) - want) >= 1e-4f) ok = 0;
    }
  ok = ok && vecs[0] > 0.0f && vecs[4] > 0.0f && vecs[8] > 0.0f;   /* largest-magnitude entry positive */
  return ok ? 0 : 1;
}
