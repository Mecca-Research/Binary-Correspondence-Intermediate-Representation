/* The driver tools/build/emit_kernel.py appends to the oracle's emit_lstm_cell_c kernel (the manifest's
 * kernel_lstm): no unit of its own -- it calls the function the kernel defines above it. */
#include <stdio.h>
#include <math.h>
int main(void) {
  /* a 1x1 LSTM with W_*=1, U_*=0, b_*=0, x=0.5, h_prev=0, c_prev=0: every gate pre-activation is 0.5. */
  float X[1] = {0.5f}, HP[1] = {0.0f}, CP[1] = {0.0f};
  float Wf[1]={1.0f}, Uf[1]={0.0f}, bf[1]={0.0f};
  float Wi[1]={1.0f}, Ui[1]={0.0f}, bi[1]={0.0f};
  float Wo[1]={1.0f}, Uo[1]={0.0f}, bo[1]={0.0f};
  float Wg[1]={1.0f}, Ug[1]={0.0f}, bg[1]={0.0f};
  float H[1], C[1];
  lstm_cell(X, HP, CP, Wf,Uf,bf, Wi,Ui,bi, Wo,Uo,bo, Wg,Ug,bg, H, C);
  /* hand-computed reference: f=i=o=sigmoid(0.5), g=tanhf(0.5); c=i*g; h=o*tanhf(c). */
  float s = 1.0f/(1.0f+expf(-0.5f));
  float g = tanhf(0.5f);
  float c_exp = s*g;
  float h_exp = s*tanhf(c_exp);
  printf("h=%.6f c=%.6f (exp h=%.6f c=%.6f)\n", H[0], C[0], h_exp, c_exp);
  int ok = (fabsf(H[0]-h_exp) < 1e-4f) && (fabsf(C[0]-c_exp) < 1e-4f);
  return ok ? 0 : 1;
}
