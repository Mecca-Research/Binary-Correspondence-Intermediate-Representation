/* The driver tools/build/emit_kernel.py appends to the oracle's emit_kmeans_assign_c kernel (the manifest's
 * kernel_kmeans): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
int main(void) {
  /* three baked centroids: (0,0), (10,10), (-5,5). Points off any exact equidistant tie. */
  float C[6] = {0.0f, 0.0f,  10.0f, 10.0f,  -5.0f, 5.0f};
  float X0[2] = {0.3f, 0.1f};    /* -> centroid 0 (near origin) */
  float X1[2] = {9.7f, 10.2f};   /* -> centroid 1 (near (10,10)) */
  float X2[2] = {-4.8f, 5.1f};   /* -> centroid 2 (near (-5,5)) */
  int a = km_assign(X0, C);
  int b = km_assign(X1, C);
  int c = km_assign(X2, C);
  printf("a=%d b=%d c=%d\n", a, b, c);
  /* the assign kernel is EXACT (integer cluster id) -- the argmin matches kmeans_assign exactly. */
  return (a == 0 && b == 1 && c == 2) ? 0 : 1;
}
