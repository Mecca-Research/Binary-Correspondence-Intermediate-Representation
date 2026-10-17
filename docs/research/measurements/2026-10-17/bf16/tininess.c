/* BF16-0 (NDT-BF16): the lane model at the underflow boundary, against AMX and the AVX-512 FMA under
 * MXCSR FTZ|DAZ -- the tininess rule bcir_tensor.c's bf16_fma states (README.json).
 *
 *   gcc -O2 -ffp-contract=off -mamx-tile -mamx-bf16 -mavx512f tininess.c -lm && ./a.out
 *
 * Every lane runs three ways: the model, the AMX tile product, and one vfmadd per lane step under
 * MXCSR FTZ|DAZ. lane step: d = e + a*b in double; tiny iff |d| < 2^-126 - 2^-151 (IEEE tininess after
 * rounding, unbounded exponent) -> the zero of d's sign; else (float)d. Old model: flush (float)fmaf. */
#define _GNU_SOURCE 1
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
typedef struct { uint8_t palette, start; uint8_t rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } tilecfg;
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static float bfv(uint16_t h) { return f_of((uint32_t)h << 16); }
static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float lane_new(float e, float a, float b) {
  double d = (double)e + (double)a * (double)b; float r = (float)d;
  return fabs(d) < 0x1.ffffffp-127 ? copysignf(0.0f, (float)d) : r;
}
static float lane_old(float e, float a, float b) {
  float r = fmaf(a, b, e); return fabsf(r) < 0x1p-126f ? copysignf(0.0f, r) : r;
}
static float add_ftz(float x, float y) { float r = x + y; return fabsf(r) < 0x1p-126f ? copysignf(0.0f, r) : r; }
static float daz(float x) { return fabsf(x) < 0x1p-126f ? copysignf(0.0f, x) : x; }
__attribute__((noinline, target("avx512f"))) static void vfma(const float *a, const float *b, const float *e, float *r) {
  _mm512_storeu_ps(r, _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), _mm512_loadu_ps(e)));
}
static uint16_t gen(int elo, int ehi) {
  int e = elo + (int)(rnd() % (uint64_t)(ehi - elo + 1));
  return (uint16_t)(((rnd() & 1) << 15) | ((uint32_t)(127 + e) << 7) | (rnd() & 0x7F));
}
int main(void) {
  if (syscall(SYS_arch_prctl, 0x1023, 18)) { perror("arch_prctl"); return 2; }
  tilecfg cfg; memset(&cfg, 0, sizeof cfg); cfg.palette = 1;
  for (int i = 0; i < 3; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
  _tile_loadconfig(&cfg);
  long n = 0, okn = 0, oko = 0, okv = 0, diffold = 0;
  static float C[16][16], R[16][16]; static uint16_t A[16][32], B[16][32];
  for (int trial = 0; trial < 4000; trial++) {
    if (trial % 2 == 0) {
      /* each output: lane terms at the underflow boundary -- products 2^-130..2^-120 then a tiny one */
      for (int i = 0; i < 16; i++) for (int t = 0; t < 32; t++) {
        int k = t / 2;  /* pair index */
        if (k % 4 == 3) { A[i][t] = gen(-80, -70); } else { A[i][t] = gen(-66, -60); }
      }
      for (int k = 0; k < 16; k++) for (int j = 0; j < 32; j++) B[k][j] = (k % 4 == 3) ? gen(-80, -70) : gen(-66, -60);
    } else {
      /* the window: a lane at +-2^-126 (1 x 2^-126), then a product of magnitude in (2^-151, 2^-150]
       * of the other sign: the exact lane value in [2^-126 - 2^-150, 2^-126 - 2^-151) -- tiny after
       * rounding with an unbounded exponent, 2^-126 on the subnormal grid. Rows alternate signs. */
      memset(A, 0, sizeof A); memset(B, 0, sizeof B);
      for (int i = 0; i < 16; i++) {
        uint16_t s = (uint16_t)((i & 1) << 15);
        A[i][0] = (uint16_t)(0x3F80 | s); A[i][1] = (uint16_t)(0x3F80 | s);
        for (int t = 2; t < 4; t++) {
          unsigned fa, fb;
          do { fa = (unsigned)(rnd() & 0x7F); fb = (unsigned)(rnd() & 0x7F); } while ((128 + fa) * (128 + fb) > 2 * 128 * 128);
          A[i][t] = (uint16_t)((s ^ 0x8000) | ((127 - 75) << 7) | fa);
          B[1][2 * (t & 1) + 0] = 0; (void)fb;
          A[i][t + 2] = (uint16_t)fb;  /* stash fb for the B fill below (unused lanes are zero) */
        }
      }
      for (int j = 0; j < 16; j++) {
        B[0][2 * j] = 0x0080; B[0][2 * j + 1] = 0x0080;               /* 2^-126 in both lanes */
        unsigned fb0 = (unsigned)(rnd() & 0x7F), fb1 = (unsigned)(rnd() & 0x7F);
        B[1][2 * j] = (uint16_t)(((127 - 76) << 7) | (fb0 & 0x0F));    /* small fractions keep */
        B[1][2 * j + 1] = (uint16_t)(((127 - 76) << 7) | (fb1 & 0x0F)); /* the product <= 2^-150 */
      }
      for (int i = 0; i < 16; i++) { A[i][4] = 0; A[i][5] = 0;
        A[i][2] = (uint16_t)((A[i][2] & 0xFF80) | (A[i][2] & 0x3F)); A[i][3] = (uint16_t)((A[i][3] & 0xFF80) | (A[i][3] & 0x3F)); }
    }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) C[i][j] = (trial % 2 || (rnd() & 3)) ? 0.0f : f_of((uint32_t)(rnd() & 0x80FFFFFFu));
    memcpy(R, C, sizeof C);
    _tile_loadd(0, R, 64); _tile_loadd(1, A, 64); _tile_loadd(2, B, 64); _tile_dpbf16ps(0, 1, 2); _tile_stored(0, R, 64);
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
      float en = 0, on = 0, eo = 0, oo = 0;
      float ev[16], ov[16];  /* for the vector replay: each lane step as one vfmadd under FTZ|DAZ */
      float ve = 0, vo = 0;
      for (int k = 0; k < 16; k++) {
        float a0 = bfv(A[i][2 * k]), a1 = bfv(A[i][2 * k + 1]), b0 = bfv(B[k][2 * j]), b1 = bfv(B[k][2 * j + 1]);
        en = lane_new(en, a0, b0); on = lane_new(on, a1, b1);
        eo = lane_old(eo, a0, b0); oo = lane_old(oo, a1, b1);
        float xa[16] = {a0, a1}, xb[16] = {b0, b1}, xe[16] = {ve, vo}, xr[16];
        unsigned int old = _mm_getcsr(); _mm_setcsr(old | 0x8040u); vfma(xa, xb, xe, xr); _mm_setcsr(old);
        ve = xr[0]; vo = xr[1]; (void)ev; (void)ov;
      }
      float c = daz(C[i][j]);
      float rn = add_ftz(c, add_ftz(en, on)), ro = add_ftz(c, add_ftz(eo, oo)), rv = add_ftz(c, add_ftz(ve, vo));
      n++; okn += u_of(rn) == u_of(R[i][j]); oko += u_of(ro) == u_of(R[i][j]); okv += u_of(rv) == u_of(R[i][j]);
      diffold += u_of(rn) != u_of(ro);
    }
  }
  _tile_release();
  printf("underflow-boundary lanes: %ld outputs; new model %ld, old model %ld, vfmadd FTZ|DAZ replay %ld match AMX; new!=old on %ld\n",
         n, okn, oko, okv, diffold);
  return 0;
}
