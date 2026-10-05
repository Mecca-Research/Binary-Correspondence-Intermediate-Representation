/* The driver tools/build/emit_kernel.py appends to the oracle's emit_tree_predict_c kernel (the manifest's
 * kernel_tree): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
#include <math.h>
int main(void) {
  /* the toy 5-node tree (test_classical._toy_tree): node0 split x[0]<=0.5 -> node1 else leaf2(30);
     node1 split x[1]<=0.5 -> leaf3(10) else leaf4(20). */
  int   FE[5] = {0, 1, -1, -1, -1};
  float TH[5] = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f};
  int   LE[5] = {1, 3, 0, 0, 0};
  int   RI[5] = {2, 4, 0, 0, 0};
  float LV[5] = {0.0f, 0.0f, 30.0f, 10.0f, 20.0f};
  float X0[2] = {0.2f, 0.2f};   /* -> leaf3 = 10 */
  float X1[2] = {0.2f, 0.9f};   /* -> leaf4 = 20 */
  float X2[2] = {0.9f, 0.0f};   /* -> leaf2 = 30 */
  float a = tree_p(X0, FE, TH, LE, RI, LV);
  float b = tree_p(X1, FE, TH, LE, RI, LV);
  float c = tree_p(X2, FE, TH, LE, RI, LV);
  printf("a=%.1f b=%.1f c=%.1f\n", a, b, c);
  /* the tree is EXACT (no transcendental) -- the leaf values match bit-for-bit. */
  return (a == 10.0f && b == 20.0f && c == 30.0f) ? 0 : 1;
}
