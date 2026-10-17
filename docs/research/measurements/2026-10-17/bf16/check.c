/* BF16-1 (NDT-BF16): every bf16 instance, entry, chunking and group against an independent
 * reference of the contract that reads the binary32 operands directly (no panels), rounds lanes in
 * __float128 and decides tininess by an explicit 24-bit rounding with the exponent unbounded
 * (README.json; check.txt is five builds of it).
 *
 *   CC -std=gnu11 -O3 -ffp-contract=off -fno-math-errno [-DBCIR_TENSOR_NO_DISPATCH |
 *      -DBCIR_TENSOR_MAX_VARIANT=4] -I runtime/c check.c runtime/c/bcir_tensor.c -lm -pthread
 *   ./a.out [rounds] */
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef __float128 Q;
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float fbits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* bf16 by classification: NaN quiet, zero/subnormal its zero, else nearest-even on 16 bits */
static float ref_bf(float x) {
  uint32_t u = bits(x), keep, rest;
  if (isnan(x)) return fbits((u | 0x00400000u) & 0xFFFF0000u);
  if (fpclassify(x) == FP_SUBNORMAL || x == 0.0f) return fbits(u & 0x80000000u);
  keep = u & 0xFFFF0000u; rest = u & 0xFFFFu;
  if (rest > 0x8000u || (rest == 0x8000u && (keep & 0x10000u))) keep += 0x10000u;
  return fbits(keep);
}
static float sz(float like) { return signbit(like) ? -0.0f : 0.0f; }
static float ref_daz(float x) { return fpclassify(x) == FP_SUBNORMAL ? sz(x) : x; }
/* e + a*b in quad (rounded to 113 bits, which rounding to 24 keeps exact); tiny when its 24-bit
 * rounding with the exponent unbounded is below 2^-126: scaled by 2^150 (exact) that rounding is to
 * an integer near 2^24, so tiny exactly when |q| * 2^150 < 2^24 - 1/2 (the tie 2^24 - 1/2 rounds to
 * the even 2^24, not tiny) */
static float ref_lane(float e, float a, float b) {
  Q q = (Q)e + (Q)a * (Q)b, mag = q < 0 ? -q : q;
  if (q == q && mag * (Q)0x1p150 < (Q)16777216 - (Q)0.5) return signbit((double)q) || (q == 0 && signbit(e) && signbit(a * b)) ? -0.0f : 0.0f;
  return (float)q;
}
static float ref_ftz(float x) { return fpclassify(x) == FP_SUBNORMAL ? sz(x) : x; }

static void ref_mm(int ta, int tb, size_t m, size_t n, size_t k, float alpha, const float *a,
                   const float *b, float beta, float *c) {
  for (size_t i = 0; i < m; i++) for (size_t j = 0; j < n; j++) {
    float acc = beta == 0.0f ? 0.0f : beta * c[i * n + j];
    for (size_t l0 = 0; l0 < k; l0 += 32) {
      float e = 0.0f, o = 0.0f;
      for (size_t p = 0; p < 16; p++) {
        for (int h = 0; h < 2; h++) {
          size_t l = l0 + 2 * p + (size_t)h;
          float av = l < k ? ref_bf(alpha * a[ta ? l * m + i : i * k + l]) : 0.0f;
          float bv = l < k ? ref_bf(b[tb ? j * k + l : l * n + j]) : 0.0f;
          if (h == 0) e = ref_lane(e, av, bv); else o = ref_lane(o, av, bv);
        }
      }
      acc = ref_ftz(ref_daz(acc) + ref_ftz(e + o));
    }
    c[i * n + j] = acc;
  }
}

static uint64_t rs = 0x123456789abcdefull;
static uint64_t rnd(void) { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 0x2545F4914F6CDD1Dull; }
/* operands: kind 0 mixed magnitudes; 1 trainer-like; 2 edge-heavy (subnormals, zeros, tiny, inf) */
static float gen(int kind) {
  uint64_t r = rnd();
  if (kind == 1) { double u = ((r >> 11) + 0.5) * 0x1p-53; return (float)((u - 0.5) * 0.08); }
  if (kind == 2) {
    switch (r % 16) {
    case 0: return 0.0f; case 1: return -0.0f;
    case 2: return fbits((uint32_t)(r >> 8) & 0x807FFFFFu);           /* subnormal (or zero) */
    case 3: return fbits(0x00800000u | ((uint32_t)(r >> 8) & 0x80000000u)); /* +-2^-126 */
    case 4: return ldexpf(1.0f + (float)((r >> 20) & 0x7F) / 128.0f, -75 - (int)((r >> 30) & 3)) * ((r >> 40) & 1 ? -1.0f : 1.0f);
    case 5: return (r >> 9) % 64 == 0 ? INFINITY : 1.5f;
    default: break;
    }
  }
  {
    int e = (int)((r >> 20) % 33) - 16;
    float f = ldexpf(1.0f + (float)((r >> 32) & 0xFFFFFF) / 16777216.0f, e);
    return (r >> 60) & 1 ? -f : f;
  }
}

static long cmp(const float *x, const float *y, size_t n) {
  long bad = 0;
  for (size_t i = 0; i < n; i++) {
    if (isnan(x[i]) && isnan(y[i])) continue;
    bad += bits(x[i]) != bits(y[i]);
  }
  return bad;
}

int main(int argc, char **argv) {
  int rounds = argc > 1 ? atoi(argv[1]) : 40;
  int mask = bcir_tensor_mm_bf16_variants();
  static const size_t dims[] = {1, 2, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 130};
  static const float alphas[] = {1.0f, 0.75f, -1.25f, 1.1f}, betas[] = {0.0f, 1.0f, 0.5f, -2.0f};
  long cases = 0, bad = 0, refused = 0;
  size_t poolbytes = bcir_tensor_pool_size(4);
  void *pm = aligned_alloc(64, (poolbytes + 63) / 64 * 64);
  bcir_tensor_pool *pool = bcir_tensor_pool_init(pm, poolbytes, 4);
  printf("variants mask %d\n", mask);
  for (int round = 0; round < rounds; round++) {
    size_t m = dims[rnd() % 14], n = dims[rnd() % 14], k = dims[rnd() % 14] * (1 + rnd() % 3);
    int ta = (int)(rnd() & 1), tb = (int)(rnd() & 1), kind = (int)(rnd() % 3);
    float alpha = alphas[rnd() % 4], beta = betas[rnd() % 4];
    float *a = malloc(m * k * 4), *b = malloc(k * n * 4), *c0 = malloc(m * n * 4), *want = malloc(m * n * 4), *got = malloc(m * n * 4);
    for (size_t i = 0; i < m * k; i++) a[i] = gen(kind);
    for (size_t i = 0; i < k * n; i++) b[i] = gen(kind);
    for (size_t i = 0; i < m * n; i++) c0[i] = gen(kind);
    memcpy(want, c0, m * n * 4);
    ref_mm(ta, tb, m, n, k, alpha, a, b, beta, want);
    size_t full = bcir_tensor_mm_bf16_workspace(m, n, k);
    size_t least = 2 * 64 * ((k + 31) / 32 * 32) + 64;
    size_t sizes[4] = {full, least, least + 2 * 32 * ((k + 31) / 32 * 32), (full + least) / 2};
    unsigned char *ws = malloc(full + least + 128);
    for (int v = 1; v <= 8; v *= 2) {
      if (!(mask & v)) continue;
      for (int s = 0; s < 4; s++) {
        size_t bytes = sizes[s] < least ? least : sizes[s];
        memcpy(got, c0, m * n * 4);
        int ran = bcir_tensor_mm_bf16_variant(v, ta, tb, m, n, k, alpha, a, b, beta, got, ws + (s & 1) * 8, bytes);
        cases++;
        if (ran != v) { refused++; continue; }
        long d = cmp(got, want, m * n);
        if (d) { bad++; if (bad < 10) printf("MISMATCH v%d ws%d ta%d tb%d m%zu n%zu k%zu alpha %g beta %g kind %d: %ld of %zu\n", v, s, ta, tb, m, n, k, alpha, beta, kind, d, m * n); }
      }
    }
    /* the pooled entry at four threads, whole and chunked */
    for (int s = 0; s < 2; s++) {
      memcpy(got, c0, m * n * 4);
      int ok = bcir_tensor_mm_bf16_pool(pool, ta, tb, m, n, k, alpha, a, b, beta, got, ws, s ? least : full);
      cases++;
      if (!ok && (full || least)) { refused++; continue; }
      long d = cmp(got, want, m * n);
      if (d) { bad++; if (bad < 10) printf("MISMATCH pool ws%d ta%d tb%d m%zu n%zu k%zu: %ld\n", s, ta, tb, m, n, k, d); }
    }
    /* below the least: refused, nothing written */
    if (k) {
      memcpy(got, c0, m * n * 4);
      int ok = bcir_tensor_mm_bf16_pool(pool, ta, tb, m, n, k, alpha, a, b, beta, got, ws, least - 1);
      cases++;
      if (ok || memcmp(got, c0, m * n * 4)) { bad++; printf("NOT REFUSED below the least (m%zu n%zu k%zu)\n", m, n, k); }
    }
    free(a); free(b); free(c0); free(want); free(got); free(ws);
  }
  /* groups: q/k/v forward (shared A), and a backward chain into one dx with shared B for the dW */
  for (int round = 0; round < rounds / 4 + 1; round++) {
    size_t r = 64 + rnd() % 70, d = 32 + rnd() % 40, kv = 16 + rnd() % 30;
    size_t outs[3] = {d, kv, kv};
    float *x = malloc(r * d * 4), *w[3], *y[3], *yref[3], *dy[3], *dw[3], *dwref[3], *dx = malloc(r * d * 4), *dxref = malloc(r * d * 4);
    for (size_t i = 0; i < r * d; i++) x[i] = gen(1);
    for (int q = 0; q < 3; q++) {
      w[q] = malloc(outs[q] * d * 4); y[q] = malloc(r * outs[q] * 4); yref[q] = malloc(r * outs[q] * 4);
      dy[q] = malloc(r * outs[q] * 4); dw[q] = malloc(outs[q] * d * 4); dwref[q] = malloc(outs[q] * d * 4);
      for (size_t i = 0; i < outs[q] * d; i++) { w[q][i] = gen(1); dw[q][i] = gen(1); }
      for (size_t i = 0; i < r * outs[q]; i++) dy[q][i] = gen(1);
      memcpy(dwref[q], dw[q], outs[q] * d * 4);
    }
    for (size_t i = 0; i < r * d; i++) dx[i] = dxref[i] = gen(1);
    bcir_tensor_mm_call cf[3], cx[3], cw[3];
    for (int q = 0; q < 3; q++) {
      cf[q] = (bcir_tensor_mm_call){0, 1, r, outs[q], d, 1.0f, x, w[q], 0.0f, y[q]};
      cx[q] = (bcir_tensor_mm_call){0, 0, r, d, outs[q], 1.0f, dy[q], w[q], q ? 1.0f : 0.5f, dx};
      cw[q] = (bcir_tensor_mm_call){1, 0, outs[q], d, r, 1.0f, dy[q], x, 1.0f, dw[q]};
      ref_mm(0, 1, r, outs[q], d, 1.0f, x, w[q], 0.0f, yref[q]);
    }
    for (int q = 0; q < 3; q++) ref_mm(0, 0, r, d, outs[q], 1.0f, dy[q], w[q], q ? 1.0f : 0.5f, dxref);
    for (int q = 0; q < 3; q++) ref_mm(1, 0, outs[q], d, r, 1.0f, dy[q], x, 1.0f, dwref[q]);
    size_t gf = bcir_tensor_mm_bf16_group_workspace(NULL, 0, cf, 3), gb = bcir_tensor_mm_bf16_group_workspace(cx, 3, cw, 3);
    size_t unshared = 0;
    for (int q = 0; q < 3; q++) unshared += bcir_tensor_mm_bf16_workspace(r, outs[q], d) - 64;
    unsigned char *ws = malloc(gb + gf + 256);
    for (int s = 0; s < 2; s++) {  /* the group whole, then with too little for the group (one call at a time) */
      int ok = bcir_tensor_mm_bf16_group_pool(pool, NULL, 0, cf, 3, ws, s ? gf - 1 : gf);
      long dd = 0;
      for (int q = 0; q < 3; q++) dd += cmp(y[q], yref[q], r * outs[q]);
      cases++; if (!ok || dd) { bad++; printf("GROUP fwd s%d ok%d: %ld differ (gf %zu, unshared %zu)\n", s, ok, dd, gf, unshared + 64); }
    }
    {
      int ok = bcir_tensor_mm_bf16_group_pool(pool, cx, 3, cw, 3, ws, gb);
      long dd = cmp(dx, dxref, r * d);
      for (int q = 0; q < 3; q++) dd += cmp(dw[q], dwref[q], outs[q] * d);
      cases++; if (!ok || dd) { bad++; printf("GROUP bwd ok%d: %ld differ\n", ok, dd); }
    }
    if (round == 0) printf("group workspace: forward %zu (unshared %zu), backward %zu\n", gf, unshared + 64, gb);
    free(ws); free(x); free(dx); free(dxref);
    for (int q = 0; q < 3; q++) { free(w[q]); free(y[q]); free(yref[q]); free(dy[q]); free(dw[q]); free(dwref[q]); }
  }
  /* the crafted edges through every instance: the tininess window, -0 products under c = -0, a
   * subnormal c, products below the normal range; each output's expected bits stated */
  {
    enum { M = 4, N = 4, K = 40 };
    float a[M * K], b[K * N], c[M * N], got[M * N];
    static const uint32_t want[M * N] = {
      0x00000000u, 0, 0, 0,   /* (0,0): 2^-126 then -1.328*2^-151 in the even lane: tiny -> +0 */
      0, 0x00000000u, 0, 0,   /* (1,1): c = -0, every product -0: lanes from +0 -> +0 */
      0, 0, 0x00800000u, 0,   /* (2,2): c = 2^-149 (subnormal, DAZ) + 2^-126 -> 2^-126 */
      0, 0, 0, 0x3f800000u};  /* (3,3): products 2^-140 each flushed from +0 -> c = 1 */
    memset(a, 0, sizeof a); memset(b, 0, sizeof b);
    for (int l = 0; l < K; l++) { a[1 * K + l] = 1.0f; b[l * N + 1] = -0.0f; a[3 * K + l] = 0x1p-70f; b[l * N + 3] = 0x1p-70f; }
    a[0 * K + 0] = 1.0f; b[0 * N + 0] = 0x1p-126f; a[0 * K + 2] = -0x1.4p-75f; b[2 * N + 0] = 0x1.1p-76f;
    a[2 * K + 0] = 1.0f; b[0 * N + 2] = 0x1p-126f;
    static unsigned char ws[16384];
    for (int v = 1; v <= 8; v *= 2) {
      if (!(mask & v)) continue;
      for (int i = 0; i < M * N; i++) c[i] = 0.0f;
      c[1 * N + 1] = -0.0f; c[2 * N + 2] = 0x1p-149f; c[3 * N + 3] = 1.0f;
      memcpy(got, c, sizeof c);
      if (bcir_tensor_mm_bf16_variant(v, 0, 0, M, N, K, 1.0f, a, b, 1.0f, got, ws, sizeof ws) != v) { bad++; printf("edge v%d refused\n", v); continue; }
      for (int i = 0; i < M; i++) {
        uint32_t g = bits(got[i * N + i]);
        cases++;
        if (g != want[i * N + i]) { bad++; printf("EDGE v%d (%d,%d): got %08x want %08x\n", v, i, i, g, want[i * N + i]); }
      }
      float ref[M * N]; memcpy(ref, c, sizeof c); ref_mm(0, 0, M, N, K, 1.0f, a, b, 1.0f, ref);
      cases++; if (cmp(got, ref, M * N)) { bad++; printf("EDGE v%d differs from the reference\n", v); }
    }
  }
  bcir_tensor_pool_stop(pool); free(pm);
  printf("bf16 check: %ld cases, %ld mismatched, %ld refused\n", cases, bad, refused);
  return bad != 0;
}
