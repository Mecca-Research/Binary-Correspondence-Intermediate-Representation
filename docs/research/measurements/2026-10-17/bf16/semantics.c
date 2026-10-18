/* BF16-0 (NDT-BF16): the bit semantics of this host's TDPBF16PS (AMX), VDPBF16PS (AVX512-BF16)
 * and VCVTNEPS2BF16, each against candidate exact models -- the record behind bcir_tensor.h's bf16
 * contract (docs/research/measurements/2026-10-17/bf16/README.json).
 *
 * Products of two bf16 values are exact in fp32 when in range, so a "+= a*b" with RNE is
 * RN(acc + a*b) whether fused or not; what distinguishes the models is the summation order,
 * where the roundings fall, the zero sign of an untouched accumulator, and DAZ/FTZ.
 *
 *   gcc -O2 -ffp-contract=off -mamx-tile -mamx-bf16 -mavx512f -mavx512bf16 semantics.c -lm
 *   ./a.out [trials] [cvt]      (cvt=1 runs the exhaustive 2^32 conversion check)            */
#define _GNU_SOURCE 1
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

typedef __float128 Q;
typedef struct { uint8_t palette, start; uint8_t rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } tilecfg;

static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 0x2545F4914F6CDD1Dull; }
static int uni(int lo, int hi) { return lo + (int)(rnd() % (uint64_t)(hi - lo + 1)); }

static int sub32(uint32_t u) { return (u & 0x7F800000u) == 0 && (u & 0x007FFFFFu) != 0; }
/* flush a subnormal to the zero of its sign */
static float fl(float x, int on) { uint32_t u = u_of(x); return on && sub32(u) ? f_of(u & 0x80000000u) : x; }
static float bfv(uint16_t h, int daz) { return fl(f_of((uint32_t)h << 16), daz); }
/* a fused step with FTZ as the tininess probe (tininess.c) established it: tiny after rounding with
 * the exponent unbounded (|exact| < 2^-126 - 2^-151) flushes to the zero of its sign */
static float fma_fl(float a, float b, float e, int on) {
  double d = (double)e + (double)a * (double)b;
  if (on && fabs(d) < 0x1.ffffffp-127) return (float)copysign(0.0, d);
  return (float)d;
}

/* ---------------- AMX ---------------- */
static void amx_config(int krows) {
  tilecfg c; memset(&c, 0, sizeof c); c.palette = 1;
  c.rows[0] = 16; c.colsb[0] = 64;              /* C: 16 x 16 fp32 */
  c.rows[1] = 16; c.colsb[1] = (uint16_t)(krows * 4); /* A: 16 x 2*krows bf16 */
  c.rows[2] = (uint8_t)krows; c.colsb[2] = 64;  /* B: krows pairs x 16 columns */
  _tile_loadconfig(&c);
}
static void amx_dp(float C[16][16], const uint16_t A[16][32], const uint16_t B[16][32]) {
  _tile_loadd(0, C, 64); _tile_loadd(1, A, 64); _tile_loadd(2, B, 64);
  _tile_dpbf16ps(0, 1, 2);
  _tile_stored(0, C, 64);
}

/* the 32 terms of output (m, n): t = 2k + j pairs A[m][2k+j] with B[k][2n+j] */
#define NT 32
typedef struct { float c; float a[NT], b[NT]; int nt; } cell;

/* ---------------- models: one output from c and nt terms ---------------- */
typedef float (*modelfn)(const cell *x, int ftz);
static float M_chain(const cell *x, int ftz) { /* acc = c; each term in order */
  float acc = x->c;
  for (int t = 0; t < x->nt; t++) acc = fma_fl(x->a[t], x->b[t], acc, ftz);
  return acc;
}
static float M_chain_mul(const cell *x, int ftz) { /* product rounded and flushed first */
  float acc = x->c;
  for (int t = 0; t < x->nt; t++) acc = fl(acc + fl(x->a[t] * x->b[t], ftz), ftz);
  return acc;
}
static float M_oddfirst(const cell *x, int ftz) {
  float acc = x->c;
  for (int t = 0; t < x->nt; t += 2) {
    acc = fma_fl(x->a[t + 1], x->b[t + 1], acc, ftz);
    acc = fma_fl(x->a[t], x->b[t], acc, ftz);
  }
  return acc;
}
static float M_pairexact(const cell *x, int ftz) { /* acc = RN(acc + p0 + p1), one rounding a pair */
  float acc = x->c;
  for (int t = 0; t < x->nt; t += 2)
    acc = fl((float)((Q)acc + (Q)x->a[t] * x->b[t] + (Q)x->a[t + 1] * x->b[t + 1]), ftz);
  return acc;
}
static float M_pairrn(const cell *x, int ftz) { /* acc = RN(acc + RN(p0 + p1)) */
  float acc = x->c;
  for (int t = 0; t < x->nt; t += 2) {
    float s = fl((float)((Q)x->a[t] * x->b[t] + (Q)x->a[t + 1] * x->b[t + 1]), ftz);
    acc = fl(acc + s, ftz);
  }
  return acc;
}
static float M_lanes(const cell *x, int ftz) { /* even/odd lanes from +0, then c + (E + O) */
  float e = 0.0f, o = 0.0f;
  for (int t = 0; t < x->nt; t += 2) {
    e = fma_fl(x->a[t], x->b[t], e, ftz);
    o = fma_fl(x->a[t + 1], x->b[t + 1], o, ftz);
  }
  return fl(x->c + fl(e + o, ftz), ftz);
}
static float M_lanesc(const cell *x, int ftz) { /* even lane from c, odd from +0, then E + O */
  float e = x->c, o = 0.0f;
  for (int t = 0; t < x->nt; t += 2) {
    e = fma_fl(x->a[t], x->b[t], e, ftz);
    o = fma_fl(x->a[t + 1], x->b[t + 1], o, ftz);
  }
  return fl(e + o, ftz);
}
static float M_exact(const cell *x, int ftz) { /* one rounding of the exact total */
  Q s = x->c;
  for (int t = 0; t < x->nt; t++) s += (Q)x->a[t] * x->b[t];
  return fl((float)s, ftz);
}
static float M_exactblock(const cell *x, int ftz) { /* c + RN(exact block sum) */
  Q s = 0;
  for (int t = 0; t < x->nt; t++) s += (Q)x->a[t] * x->b[t];
  return fl(x->c + fl((float)s, ftz), ftz);
}
static float M_tree(const cell *x, int ftz) { /* pairwise tree, RN at each node, then c + tree */
  float v[NT]; int n = x->nt;
  for (int t = 0; t < n; t++) v[t] = fl(x->a[t] * x->b[t], ftz);
  while (n > 1) { for (int i = 0; i < n / 2; i++) v[i] = fl(v[2 * i] + v[2 * i + 1], ftz); n /= 2; }
  return fl(x->c + v[0], ftz);
}
static float M_pairchain0(const cell *x, int ftz) { /* S from +0 over RN pair sums, then c + S */
  float s = 0.0f;
  for (int t = 0; t < x->nt; t += 2)
    s = fl(s + fl((float)((Q)x->a[t] * x->b[t] + (Q)x->a[t + 1] * x->b[t + 1]), ftz), ftz);
  return fl(x->c + s, ftz);
}
static float M_chain0(const cell *x, int ftz) { /* S from +0 over every term, then c + S */
  float s = 0.0f;
  for (int t = 0; t < x->nt; t++) s = fma_fl(x->a[t], x->b[t], s, ftz);
  return fl(x->c + s, ftz);
}

static const struct { const char *name; modelfn fn; } MODELS[] = {
  {"chain (c, t=0..31, fused)", M_chain},       {"chain, product rounded", M_chain_mul},
  {"chain, odd term first", M_oddfirst},        {"chain of exact pair sums", M_pairexact},
  {"chain of RN pair sums", M_pairrn},          {"even/odd lanes from +0, c+(E+O)", M_lanes},
  {"even lane from c, odd +0, E+O", M_lanesc},  {"exact total, one rounding", M_exact},
  {"c + RN(exact block)", M_exactblock},        {"pairwise tree, c + tree", M_tree},
  {"pair sums chained from +0, c+S", M_pairchain0}, {"terms chained from +0, c+S", M_chain0},
};
#define NM (int)(sizeof MODELS / sizeof MODELS[0])

static int same(float x, float y) {
  uint32_t a = u_of(x), b = u_of(y);
  if (isnan(x) && isnan(y)) return 2; /* both NaN: payload compared separately */
  return a == b;
}

/* ---------------- generators ---------------- */
static uint16_t gen_bf(int elo, int ehi) {
  return (uint16_t)(((rnd() & 1) << 15) | ((uint32_t)(127 + uni(elo, ehi)) << 7) | (rnd() & 0x7F));
}
static float gen_f(int elo, int ehi) {
  return f_of((uint32_t)(((rnd() & 1) << 31) | ((uint32_t)(127 + uni(elo, ehi)) << 23) | (rnd() & 0x7FFFFF)));
}
static uint16_t bf_rne(float f) { uint32_t u = u_of(f); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }
static double gauss(void) { double u1 = ((rnd() >> 11) + 1) * 0x1p-53, u2 = (rnd() >> 11) * 0x1p-53; return sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2); }

static void fill(int dist, float C[16][16], uint16_t A[16][32], uint16_t B[16][32]) {
  for (int i = 0; i < 16; i++) for (int j = 0; j < 32; j++) {
    switch (dist) {
    case 0: A[i][j] = gen_bf(-8, 8); B[i][j] = gen_bf(-8, 8); break;   /* wide */
    case 1: A[i][j] = gen_bf(0, 0); B[i][j] = gen_bf(0, 0); break;     /* narrow */
    case 2: A[i][j] = gen_bf(-4, 4); B[i][j] = gen_bf(-4, 4); break;   /* big accumulator */
    default: A[i][j] = bf_rne((float)gauss()); B[i][j] = bf_rne((float)(0.02 * gauss())); break; /* trainer-like */
    }
  }
  for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
    switch (dist) {
    case 0: C[i][j] = gen_f(-6, 18); break;
    case 1: C[i][j] = gen_f(0, 4); break;
    case 2: C[i][j] = gen_f(12, 22); break;
    default: C[i][j] = (float)(0.1 * gauss()); break;
    }
  }
}

static void cell_of(const float C[16][16], const uint16_t A[16][32], const uint16_t B[16][32], int m, int n,
                    int krows, int daz, cell *x) {
  x->c = fl(C[m][n], daz); x->nt = 2 * krows;
  for (int k = 0; k < krows; k++) for (int j = 0; j < 2; j++) {
    x->a[2 * k + j] = bfv(A[m][2 * k + j], daz); x->b[2 * k + j] = bfv(B[k][2 * n + j], daz);
  }
}

static void random_amx(int trials) {
  static const char *dn[] = {"wide", "narrow", "big-accumulator", "trainer-like"};
  for (int dist = 0; dist < 4; dist++) {
    long hit[NM] = {0}, total = 0;
    for (int tr = 0; tr < trials; tr++) {
      static float C[16][16], R[16][16]; static uint16_t A[16][32], B[16][32];
      fill(dist, C, A, B); memcpy(R, C, sizeof C);
      amx_dp(R, A, B);
      for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++) {
        cell x; cell_of(C, A, B, m, n, 16, 1, &x); total++;
        for (int k = 0; k < NM; k++) hit[k] += same(MODELS[k].fn(&x, 1), R[m][n]) != 0;
      }
    }
    printf("TDPBF16PS random %-16s %ld outputs:", dn[dist], total);
    for (int k = 0; k < NM; k++) printf(" [%d]%ld", k, hit[k]);
    printf("\n");
  }
  for (int k = 0; k < NM; k++) printf("  model [%d] %s\n", k, MODELS[k].name);
}

/* one hand-built tile: C[0][0] and terms of output (0,0), every other element zero; returns R[0][0] */
static float amx_one(float c, const uint16_t *ta, const uint16_t *tb, int nt) {
  static float C[16][16]; static uint16_t A[16][32], B[16][32];
  memset(C, 0, sizeof C); memset(A, 0, sizeof A); memset(B, 0, sizeof B);
  C[0][0] = c;
  for (int t = 0; t < nt; t++) { A[0][t] = ta[t]; B[t / 2][t % 2] = tb[t]; }
  amx_dp(C, A, B);
  return C[0][0];
}
static void edge(const char *what, float c, const uint16_t *ta, const uint16_t *tb, int nt) {
  float r = amx_one(c, ta, tb, nt);
  printf("  %-44s AMX %08x |", what, u_of(r));
  for (int daz = 0; daz < 2; daz++) for (int ftz = 0; ftz < 2; ftz++) {
    cell x; x.c = fl(c, daz); x.nt = 32;
    for (int t = 0; t < 32; t++) { x.a[t] = bfv(t < nt ? ta[t] : 0, daz); x.b[t] = bfv(t < nt ? tb[t] : 0, daz); }
    printf(" d%df%d chain %08x lanes %08x mul %08x;", daz, ftz, u_of(M_chain(&x, ftz)), u_of(M_lanes(&x, ftz)),
           u_of(M_chain_mul(&x, ftz)));
  }
  printf("\n");
}

static void edges(void) {
  printf("TDPBF16PS edges (a model column is d<DAZ>f<FTZ>):\n");
  uint16_t a[32], b[32];
  for (int t = 0; t < 32; t++) { a[t] = 0x3F80; b[t] = 0x8000; }  /* +1 x -0 = -0 */
  edge("c=-0, every product -0", f_of(0x80000000u), a, b, 32);
  for (int t = 0; t < 32; t++) { a[t] = 0; b[t] = 0; }
  edge("c=-0, every product +0", f_of(0x80000000u), a, b, 32);
  a[0] = 0x0001; b[0] = 0x7000;   /* bf16 subnormal 2^-133 x 2^97 = 2^-36 */
  edge("c=0, a subnormal x 2^97", 0.0f, a, b, 1);
  a[0] = 0x7000; b[0] = 0x0040;   /* 2^97 x 2^-127 (bf16 subnormal) = 2^-30 */
  edge("c=0, 2^97 x b subnormal", 0.0f, a, b, 1);
  a[0] = 0x1C80; b[0] = 0x1C80;   /* 2^-70 x 2^-70 = 2^-140 (fp32 subnormal) */
  edge("c=0, product 2^-140", 0.0f, a, b, 1);
  edge("c=2^-126, product 2^-140", f_of(0x00800000u), a, b, 1);
  a[1] = 0x3F80; b[1] = 0x0080;   /* second term 1 x 2^-126 */
  edge("c=0, 2^-140 then 2^-126", 0.0f, a, b, 2);
  a[0] = 0x3F80; b[0] = 0x0080; a[1] = 0x1C80; b[1] = 0x1C80;
  edge("c=0, 2^-126 then 2^-140", 0.0f, a, b, 2);
  memset(a, 0, sizeof a); memset(b, 0, sizeof b);
  edge("c=fp32 subnormal 1, no terms", f_of(1u), a, b, 0);
  a[0] = 0x3F80; b[0] = 0x0080;
  edge("c=fp32 subnormal 1, + 2^-126", f_of(1u), a, b, 1);
  a[0] = 0x3F80; b[0] = 0x8080;   /* 1 x -2^-126 */
  edge("c=2^-126+2^-149, - 2^-126", f_of(0x00800001u), a, b, 1);
  a[0] = 0x7F00; b[0] = 0x4000;   /* 2^127 x 2 = overflow in the product */
  edge("c=0, product 2^128 (overflow)", 0.0f, a, b, 1);
  a[0] = 0x7F00; b[0] = 0x3F80; a[1] = 0x7F00; b[1] = 0x3F80;
  edge("c=0, 2^127 + 2^127", 0.0f, a, b, 2);
  a[0] = 0x7F80; b[0] = 0x3F80; a[1] = 0xFF80; b[1] = 0x3F80;
  edge("c=0, +inf + -inf", 0.0f, a, b, 2);
  a[0] = 0x7FC1; b[0] = 0x3F80; a[1] = 0; b[1] = 0;
  edge("c=0, qNaN(payload 1) x 1", 0.0f, a, b, 1);
  a[0] = 0x7F81; b[0] = 0x3F80;
  edge("c=0, sNaN(payload 1) x 1", 0.0f, a, b, 1);
  a[0] = 0x3F80; b[0] = 0x3F80;
  edge("c=qNaN 0x7fc00123, + 1", f_of(0x7FC00123u), a, b, 1);
  a[0] = 0x7FC1; b[0] = 0x3F80;
  edge("c=qNaN 0x7fc00123, + qNaN(1)", f_of(0x7FC00123u), a, b, 1);
  a[0] = 0x7FC1; b[0] = 0x3F80; a[1] = 0x7FC2; b[1] = 0x3F80;
  edge("c=0, qNaN(1) then qNaN(2)", 0.0f, a, b, 2);
  a[0] = 0x7F80; b[0] = 0x0000;
  edge("c=0, inf x 0", 0.0f, a, b, 1);
  memset(a, 0, sizeof a); memset(b, 0, sizeof b);
  a[0] = 0x3F80; b[0] = 0x0080; a[2] = 0x1C80; b[2] = 0x1C80; /* even lane: 2^-126 then + 2^-140 fused */
  edge("c=0, even lane 2^-126 then 2^-140", 0.0f, a, b, 3);
  memset(a, 0, sizeof a); memset(b, 0, sizeof b);
  a[0] = 0x3F80; b[0] = 0x0080; a[1] = 0x1C80; b[1] = 0x1C80; /* E=2^-126, O=2^-140 (flushed?) */
  edge("c=0, E=2^-126, O=2^-140", 0.0f, a, b, 2);
  memset(a, 0, sizeof a); memset(b, 0, sizeof b);
  a[0] = 0x3F80; b[0] = 0x8000;  /* E = +0 + (1 x -0) */
  edge("c=-0, one -0 product, rest none", f_of(0x80000000u), a, b, 1);
}

/* the instruction with B's rows reduced (A colsb = 4*krows): the same models over fewer terms? */
static void ragged(int trials) {
  for (int kr = 1; kr <= 16; kr *= 2) {
    amx_config(kr);
    long hit[NM] = {0}, total = 0;
    for (int tr = 0; tr < trials; tr++) {
      static float C[16][16], R[16][16]; static uint16_t A[16][32], B[16][32];
      fill(0, C, A, B); memcpy(R, C, sizeof C);
      _tile_loadd(0, R, 64); _tile_loadd(1, A, 64); _tile_loadd(2, B, 64);
      _tile_dpbf16ps(0, 1, 2); _tile_stored(0, R, 64);
      for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++) {
        cell x; cell_of(C, A, B, m, n, kr, 1, &x); total++;
        for (int k = 0; k < NM; k++) hit[k] += same(MODELS[k].fn(&x, 1), R[m][n]) != 0;
      }
    }
    printf("TDPBF16PS K rows %2d (%2d terms) %ld outputs:", kr, 2 * kr, total);
    for (int k = 0; k < NM; k++) printf(" [%d]%ld", k, hit[k]);
    printf("\n");
  }
  amx_config(16);
}

/* MXCSR independence: the same tiles under RZ+DAZ+FTZ must give the same bits */
static void mxcsr_check(int trials) {
  long diff = 0, total = 0;
  for (int tr = 0; tr < trials; tr++) {
    static float C[16][16], R0[16][16], R1[16][16]; static uint16_t A[16][32], B[16][32];
    fill(tr % 4, C, A, B);
    memcpy(R0, C, sizeof C); amx_dp(R0, A, B);
    unsigned int old = _mm_getcsr();
    _mm_setcsr((old & ~0x6000u) | 0x6000u | 0x8040u); /* RZ, FTZ, DAZ */
    memcpy(R1, C, sizeof C); amx_dp(R1, A, B);
    _mm_setcsr(old);
    for (int i = 0; i < 256; i++) { total++; diff += u_of(((float *)R0)[i]) != u_of(((float *)R1)[i]); }
  }
  printf("TDPBF16PS under MXCSR RZ+FTZ+DAZ: %ld of %ld outputs differ\n", diff, total);
}

/* ---------------- VDPBF16PS ---------------- */
static float V_oddfirst(float acc, float a0, float b0, float a1, float b1, int ftz) {
  acc = fma_fl(a1, b1, acc, ftz); return fma_fl(a0, b0, acc, ftz);
}
static float V_evenfirst(float acc, float a0, float b0, float a1, float b1, int ftz) {
  acc = fma_fl(a0, b0, acc, ftz); return fma_fl(a1, b1, acc, ftz);
}
static float V_exact(float acc, float a0, float b0, float a1, float b1, int ftz) {
  return fl((float)((Q)acc + (Q)a0 * b0 + (Q)a1 * b1), ftz);
}
static float V_pairrn(float acc, float a0, float b0, float a1, float b1, int ftz) {
  return fl(acc + fl((float)((Q)a0 * b0 + (Q)a1 * b1), ftz), ftz);
}
static void vdp(int trials) {
  typedef float (*vf)(float, float, float, float, float, int);
  static const struct { const char *name; vf fn; } V[] = {
    {"odd first", V_oddfirst}, {"even first", V_evenfirst}, {"exact pair + acc", V_exact}, {"acc + RN(pair)", V_pairrn}};
  for (int dist = 0; dist < 4; dist++) {
    long hit[4] = {0}, total = 0;
    for (int tr = 0; tr < trials * 16; tr++) {
      static float C[16][16]; static uint16_t A[16][32], B[16][32];
      fill(dist, C, A, B);
      __m512 acc = _mm512_loadu_ps(C[0]);
      __m512bh va, vb; memcpy(&va, A[0], 64); memcpy(&vb, B[0], 64);
      float r[16]; _mm512_storeu_ps(r, _mm512_dpbf16_ps(acc, va, vb));
      for (int i = 0; i < 16; i++) {
        total++;
        float a0 = bfv(A[0][2 * i], 1), a1 = bfv(A[0][2 * i + 1], 1), b0 = bfv(B[0][2 * i], 1), b1 = bfv(B[0][2 * i + 1], 1);
        for (int k = 0; k < 4; k++) hit[k] += same(V[k].fn(fl(C[0][i], 1), a0, b0, a1, b1, 1), r[i]) != 0;
      }
    }
    printf("VDPBF16PS random dist %d %ld lanes:", dist, total);
    for (int k = 0; k < 4; k++) printf(" %s %ld;", V[k].name, hit[k]);
    printf("\n");
  }
  /* edges, lane 0 */
  struct { const char *what; uint32_t c; uint16_t a0, b0, a1, b1; } E[] = {
    {"c=-0, -0 products", 0x80000000u, 0x3F80, 0x8000, 0x3F80, 0x8000},
    {"c=0, a subnormal x 2^97", 0, 0x0001, 0x7000, 0, 0},
    {"c=0, product 2^-140", 0, 0x1C80, 0x1C80, 0, 0},
    {"c=2^-126, product 2^-140", 0x00800000u, 0x1C80, 0x1C80, 0, 0},
    {"c=subnormal 1, + 2^-126", 1u, 0x3F80, 0x0080, 0, 0},
    {"c=0, qNaN(1) x 1", 0, 0x7FC1, 0x3F80, 0, 0},
    {"c=0, +inf + -inf", 0, 0x7F80, 0x3F80, 0xFF80, 0x3F80},
    {"c=0, qNaN(1) even, qNaN(2) odd", 0, 0x7FC1, 0x3F80, 0x7FC2, 0x3F80},
  };
  for (size_t e = 0; e < sizeof E / sizeof E[0]; e++) {
    float c16[16] = {0}; uint16_t a[32] = {0}, b[32] = {0};
    c16[0] = f_of(E[e].c); a[0] = E[e].a0; b[0] = E[e].b0; a[1] = E[e].a1; b[1] = E[e].b1;
    __m512bh va, vb; memcpy(&va, a, 64); memcpy(&vb, b, 64);
    float r[16]; _mm512_storeu_ps(r, _mm512_dpbf16_ps(_mm512_loadu_ps(c16), va, vb));
    printf("  VDPBF16PS %-32s %08x | d0f0 odd %08x even %08x | d1f1 odd %08x even %08x\n", E[e].what, u_of(r[0]),
           u_of(V_oddfirst(f_of(E[e].c), bfv(E[e].a0, 0), bfv(E[e].b0, 0), bfv(E[e].a1, 0), bfv(E[e].b1, 0), 0)),
           u_of(V_evenfirst(f_of(E[e].c), bfv(E[e].a0, 0), bfv(E[e].b0, 0), bfv(E[e].a1, 0), bfv(E[e].b1, 0), 0)),
           u_of(V_oddfirst(fl(f_of(E[e].c), 1), bfv(E[e].a0, 1), bfv(E[e].b0, 1), bfv(E[e].a1, 1), bfv(E[e].b1, 1), 1)),
           u_of(V_evenfirst(fl(f_of(E[e].c), 1), bfv(E[e].a0, 1), bfv(E[e].b0, 1), bfv(E[e].a1, 1), bfv(E[e].b1, 1), 1)));
  }
}

/* ---------------- VCVTNEPS2BF16, every fp32 input ---------------- */
static uint16_t cvt_sw(uint32_t u, int daz, int ftz) {
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) return (uint16_t)((u >> 16) | 0x0040u);
  if (daz && (u & 0x7F800000u) == 0) return (uint16_t)((u >> 16) & 0x8000u);
  uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
  if (ftz && (r & 0x7F80u) == 0) r &= 0x8000u;
  return (uint16_t)r;
}
static void cvt_all(void) {
  static const int VAR[3][2] = {{0, 0}, {1, 0}, {0, 1}};
  long bad[3][3] = {{0}}; /* variant x class (nan, subnormal, other) */
  uint32_t first[3] = {0}; int have[3] = {0};
  const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  for (uint64_t base = 0; base < (1ull << 32); base += 16) {
    __m512i v = _mm512_add_epi32(_mm512_set1_epi32((int)(uint32_t)base), iota);
    __m256bh h = _mm512_cvtneps_pbh(_mm512_castsi512_ps(v));
    uint16_t out[16]; memcpy(out, &h, 32);
    for (int i = 0; i < 16; i++) {
      uint32_t u = (uint32_t)base + (uint32_t)i;
      int cls = ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) ? 0 : sub32(u) ? 1 : 2;
      for (int k = 0; k < 3; k++)
        if (cvt_sw(u, VAR[k][0], VAR[k][1]) != out[i]) { bad[k][cls]++; if (!have[k]) { have[k] = 1; first[k] = u; } }
    }
  }
  for (int k = 0; k < 3; k++)
    printf("VCVTNEPS2BF16 vs RNE daz=%d ftz=%d: mismatches nan %ld subnormal %ld other %ld (first %08x)\n",
           VAR[k][0], VAR[k][1], bad[k][0], bad[k][1], bad[k][2], first[k]);
  /* MXCSR independence on a sample */
  long d = 0; unsigned int old = _mm_getcsr();
  for (int s = 0; s < 1 << 20; s++) {
    __m512i v = _mm512_set1_epi32((int)(uint32_t)rnd());
    __m256bh h0 = _mm512_cvtneps_pbh(_mm512_castsi512_ps(v));
    _mm_setcsr((old & ~0x6000u) | 0x6000u | 0x8040u);
    __m256bh h1 = _mm512_cvtneps_pbh(_mm512_castsi512_ps(v));
    _mm_setcsr(old);
    d += memcmp(&h0, &h1, 32) != 0;
  }
  printf("VCVTNEPS2BF16 under MXCSR RZ+FTZ+DAZ: %ld of %d samples differ\n", d, 1 << 20);
}

int main(int argc, char **argv) {
  int trials = argc > 1 ? atoi(argv[1]) : 200, cvt = argc > 2 ? atoi(argv[2]) : 0;
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) { perror("arch_prctl"); return 2; }
  amx_config(16);
  random_amx(trials);
  ragged(trials / 4 + 1);
  mxcsr_check(trials / 4 + 1);
  edges();
  _tile_release();
  vdp(trials);
  if (cvt) cvt_all();
  return 0;
}
