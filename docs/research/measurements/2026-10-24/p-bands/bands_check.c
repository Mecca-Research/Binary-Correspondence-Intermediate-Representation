/* P-BANDS bit check: a unit without BANDS (the tree before, or the tree after with the bit off) and a unit under it
 * (the tree after: contract | BANDS) in one process, each case the forward then the backward from the same inputs:
 * y, the verdict, every probability j <= i (the banded unit's through the band addressing where the bit takes effect:
 * a context a multiple of 16, a head at most 256 wide, on this AVX-512 host), every score adjoint j <= i after the
 * backward, dq, dk and dv, byte for byte -- 16 shapes x 8 contracts x 1 and 4 threads x with and without the
 * workspace, an infinite key and a NaN query under four contracts, and (HEAVY) nine cases at T2048. The first unit
 * runs the widest instance; the second the instances VF (forward) and VB (backward). SMALL keeps the shapes with
 * T x D <= 16384. usage: bands_check LIB_ROWS LIB_BANDS [HEAVY 0|1] [VF VB] [SMALL 0|1] */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct bcir_tensor_pool bcir_tensor_pool;
typedef int (*fwd_fn)(bcir_tensor_pool *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
                      const float *, const float *, float *, float *, int *, void *, size_t);
typedef int (*bwd_fn)(bcir_tensor_pool *, int, unsigned, size_t, size_t, size_t, size_t, size_t, const float *,
                      const float *, const float *, float *, const float *, float *, float *, float *, void *, size_t);
typedef size_t (*psize_fn)(unsigned);
typedef bcir_tensor_pool *(*pinit_fn)(void *, size_t, unsigned);
typedef void (*pstop_fn)(bcir_tensor_pool *);
typedef size_t (*ws_fn)(size_t, size_t, size_t, size_t, size_t, unsigned);
typedef struct { fwd_fn fwd; bwd_fn bwd; psize_fn psize; pinit_fn pinit; pstop_fn pstop; ws_fn ws; } lib;
static void *sym(void *h, const char *n) { void *s = dlsym(h, n); if (!s) { fprintf(stderr, "no %s\n", n); exit(2); } return s; }
static lib open_lib(const char *path) {
  void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL); lib l;
  if (!h) { fprintf(stderr, "%s\n", dlerror()); exit(2); }
  *(void **)&l.fwd = sym(h, "bcir_tensor_attention_contract_variant_scan_ws_pool");
  *(void **)&l.bwd = sym(h, "bcir_tensor_attention_backward_contract_variant_ws_pool");
  *(void **)&l.psize = sym(h, "bcir_tensor_pool_size"); *(void **)&l.pinit = sym(h, "bcir_tensor_pool_init");
  *(void **)&l.pstop = sym(h, "bcir_tensor_pool_stop"); *(void **)&l.ws = sym(h, "bcir_tensor_attention_pool_workspace");
  return l;
}
static uint64_t rng = 88172645463325252ull;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (float)((rng >> 40) & 0xffffff) / 16777216.0f * 2.0f - 1.0f; }
static int same(const float *a, const float *b) { return memcmp(a, b, 4) == 0; }
static size_t PB(size_t t, size_t i, size_t j) { return (i >> 4) * 16 * t + j * 16 + (i & 15); }
static long cases, fails;
static int VF, VB;  /* the new unit's forward and backward instances */
static void run(lib *T, lib *P, size_t b, size_t t, size_t nh, size_t nk, size_t d, unsigned ct, unsigned threads, int use_ws, int special) {
  size_t nq = b * t * nh * d, nkv = b * t * nk * d, np = b * nh * t * t, i, j, x, r, h; int bad1 = 0, bad2 = 0;
  float *q = malloc(nq * 4), *k = malloc(nkv * 4), *v = malloc(nkv * 4), *dy = malloc(nq * 4);
  float *p1 = malloc(np * 4), *p2 = malloc(np * 4), *y1 = malloc(nq * 4), *y2 = malloc(nq * 4);
  float *dq1 = malloc(nq * 4), *dq2 = malloc(nq * 4), *dk1 = malloc(nkv * 4), *dk2 = malloc(nkv * 4), *dv1 = malloc(nkv * 4), *dv2 = malloc(nkv * 4);
  for (i = 0; i < nq; i++) { q[i] = frand() * 1.5f; dy[i] = frand(); }
  for (i = 0; i < nkv; i++) { k[i] = frand() * 1.2f; v[i] = frand(); }
  if (special == 1) k[(t / 3) * nk * d] = INFINITY;  /* an infinite key */
  if (special == 2) q[(t / 2) * nh * d + 1] = NAN;
  for (i = 0; i < np; i++) { p1[i] = NAN; p2[i] = NAN; }  /* canaries */
  memset(dq1, 0xff, nq * 4); memset(dq2, 0xff, nq * 4); memset(dk1, 0xff, nkv * 4); memset(dk2, 0xff, nkv * 4); memset(dv1, 0xff, nkv * 4); memset(dv2, 0xff, nkv * 4);
  void *m1 = NULL, *m2 = NULL; bcir_tensor_pool *pl1 = NULL, *pl2 = NULL;
  if (threads > 1) {
    size_t s1 = T->psize(threads), s2 = P->psize(threads);
    m1 = aligned_alloc(64, (s1 + 63) / 64 * 64); m2 = aligned_alloc(64, (s2 + 63) / 64 * 64);
    pl1 = T->pinit(m1, s1, threads); pl2 = P->pinit(m2, s2, threads);
  }
  size_t w1 = use_ws ? T->ws(b, t, nh, nk, d, threads) : 0, w2 = use_ws ? P->ws(b, t, nh, nk, d, threads) : 0;
  void *ws1 = w1 ? aligned_alloc(64, (w1 + 63) / 64 * 64) : NULL, *ws2 = w2 ? aligned_alloc(64, (w2 + 63) / 64 * 64) : NULL;
  int banded = t % 16 == 0 && d <= 256;
  T->fwd(pl1, 0, ct, b, t, nh, nk, d, q, k, v, p1, y1, &bad1, ws1, w1);
  if (!P->fwd(pl2, VF, ct | 32u, b, t, nh, nk, d, q, k, v, p2, y2, &bad2, ws2, w2)) { printf("fwd refused\n"); exit(1); }
  long bad = 0, pbad = 0, sbad = 0;
  for (i = 0; i < nq; i++) bad += !same(&y1[i], &y2[i]);
  bad += bad1 != bad2;
  for (r = 0; r < b; r++) for (h = 0; h < nh; h++) for (i = 0; i < t; i++) for (j = 0; j <= i; j++) {
    size_t o = (r * nh + h) * t * t; pbad += !same(&p1[o + i * t + j], &p2[o + (banded ? PB(t, i, j) : i * t + j)]);
  }
  T->bwd(pl1, 0, ct, b, t, nh, nk, d, q, k, v, p1, dy, dq1, dk1, dv1, ws1, w1);
  if (!P->bwd(pl2, VB, ct | 32u, b, t, nh, nk, d, q, k, v, p2, dy, dq2, dk2, dv2, ws2, w2)) { printf("bwd refused\n"); exit(1); }
  for (r = 0; r < b; r++) for (h = 0; h < nh; h++) for (i = 0; i < t; i++) for (j = 0; j <= i; j++) {
    size_t o = (r * nh + h) * t * t; sbad += !same(&p1[o + i * t + j], &p2[o + (banded ? PB(t, i, j) : i * t + j)]);
  }
  for (x = 0; x < nq; x++) bad += !same(&dq1[x], &dq2[x]);
  for (x = 0; x < nkv; x++) bad += !same(&dk1[x], &dk2[x]) + !same(&dv1[x], &dv2[x]);
  cases++;
  if (bad || pbad || sbad) { fails++; printf("FAIL vf %d vb %d B%zu T%zu H%zu KH%zu D%zu contract %u threads %u ws %d special %d: y/d* %ld, p %ld, ds %ld\n", VF, VB, b, t, nh, nk, d, ct, threads, use_ws, special, bad, pbad, sbad); }
  if (pl1) { T->pstop(pl1); P->pstop(pl2); }
  free(m1); free(m2); free(ws1); free(ws2);
  free(q); free(k); free(v); free(dy); free(p1); free(p2); free(y1); free(y2); free(dq1); free(dq2); free(dk1); free(dk2); free(dv1); free(dv2);
}
int main(int argc, char **argv) {
  lib T = open_lib(argv[1]), P = open_lib(argv[2]); int heavy = argc > 3 && atoi(argv[3]);
  VF = argc > 4 ? atoi(argv[4]) : 0; VB = argc > 5 ? atoi(argv[5]) : 0; int small = argc > 6 && atoi(argv[6]);
  static const size_t shapes[][5] = {{1,16,1,1,32},{2,32,2,1,32},{1,48,2,2,16},{2,64,4,2,48},{1,128,2,2,64},{1,256,4,4,32},
      {2,512,2,1,64},{1,80,2,2,24},{1,128,1,1,8},{1,64,2,2,256},{1,40,2,2,32},{1,100,2,1,32},{1,96,4,4,40},{1,1024,4,4,32},
      {1,32,2,2,300},{3,144,6,3,20}};
  static const unsigned contracts[] = {17, 16, 1, 0, 18, 19, 3, 2};
  size_t s, c; unsigned th; int ws, sp;
  for (s = 0; s < sizeof shapes / sizeof *shapes; s++) if (!small || shapes[s][1] * shapes[s][4] <= 256 * 64) for (c = 0; c < sizeof contracts / sizeof *contracts; c++)
    for (th = 1; th <= 4; th += 3) for (ws = 0; ws < 2; ws++)
      run(&T, &P, shapes[s][0], shapes[s][1], shapes[s][2], shapes[s][3], shapes[s][4], contracts[c], th, ws, 0);
  for (sp = 1; sp <= 2; sp++) for (c = 0; c < 4; c++) run(&T, &P, 1, 256, 2, 2, 32, contracts[c], 4, 1, sp);
  if (heavy) for (c = 0; c < 3; c++) { run(&T, &P, 2, 2048, 4, 4, 32, contracts[c], 4, 1, 0); run(&T, &P, 1, 2048, 2, 2, 64, contracts[c], 4, 0, 0); run(&T, &P, 1, 2048, 1, 1, 32, contracts[c], 1, 0, 0); }
  printf("%ld cases, %ld failed\n", cases, fails);
  return fails != 0;
}
