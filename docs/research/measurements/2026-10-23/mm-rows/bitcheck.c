/* MM-ROWS: the k-blocked GEMM's every entry over a fixed case list, one FNV-1a hash of C a run -- the serial
 * instances (AVX-512, AVX2, the plain C, where the host has them; the plain C on small cases), the pool at four and
 * two threads, the pool through each instance (PX-3b's entry), pair jobs, and group jobs whose chains include a call
 * of depth zero (one tiling for every chained call) -- the GEO step's twelve shapes and 700 drawn from row counts
 * 1..4500, column counts 1..260 and depths 0..1000 at five (alpha, beta) pairs. Two builds print the same lines
 * exactly when every C they produce has the same bytes. usage: bitcheck [cases]  (700: 712 cases, 6,081 lines) */
#include "bcir_tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float rf(void) { /* values across exponents, both signs, some exact zeros */
  uint64_t r = rnd(); if (r % 97 == 0) return (r & 64) ? -0.0f : 0.0f;
  float f = (float)((int64_t)(r >> 11) % 2000001 - 1000000) / 1000000.0f;
  int e = (int)((r >> 3) % 9) - 4; while (e > 0) { f *= 4.0f; e--; } while (e < 0) { f *= 0.25f; e++; } return f;
}
static float *mk(size_t n) { float *p = malloc((n ? n : 1) * sizeof(float)); size_t i; for (i = 0; i < n; i++) p[i] = rf(); return p; }
static uint64_t fnv(const float *p, size_t n) { const unsigned char *b = (const unsigned char *)p; uint64_t h = 1469598103934665603ull; size_t i; for (i = 0; i < n * 4; i++) { h ^= b[i]; h *= 1099511628211ull; } return h; }
static const size_t MS[] = {1, 2, 5, 8, 13, 31, 32, 33, 127, 128, 129, 200, 257, 1000, 1023, 1024, 1025, 1500, 4095, 4096, 4097, 4500};
static const size_t NS[] = {1, 7, 16, 17, 32, 33, 47, 48, 49, 64, 95, 96, 97, 128, 130, 144, 160, 260};
static const size_t KS[] = {0, 1, 5, 64, 128, 447, 448, 449, 1000};
static const float AB[][2] = {{1.0f, 0.0f}, {1.0f, 1.0f}, {0.5f, -0.75f}, {-2.0f, 0.0f}, {1.25f, 3.0f}};
#define NE(a) (sizeof(a) / sizeof((a)[0]))
int main(int argc, char **argv) {
  int cases = argc > 1 ? atoi(argv[1]) : 700, cs, v, t;
  size_t ps4 = bcir_tensor_pool_size(4), ps2 = bcir_tensor_pool_size(2);
  void *m4 = aligned_alloc(64, (ps4 + 63) / 64 * 64), *m2 = aligned_alloc(64, (ps2 + 63) / 64 * 64);
  bcir_tensor_pool *p4 = bcir_tensor_pool_init(m4, ps4, 4), *p2 = bcir_tensor_pool_init(m2, ps2, 2);
  int have = bcir_tensor_mm_kblock_variants();
  static const struct { int ta, tb; size_t m, n, k; } fixed[] = {
    {0,0,8192,128,128}, {0,0,8192,256,128}, {0,0,8192,128,256}, {0,0,8192,260,128}, {0,1,8192,128,128}, {0,1,8192,128,256},
    {0,1,8192,256,128}, {0,1,8192,260,128}, {1,0,128,128,8192}, {1,0,256,128,8192}, {1,0,128,256,8192}, {1,0,260,128,8192} };
  if (!p4 || !p2) { fprintf(stderr, "pool refused\n"); return 2; }
  printf("variants %d\n", have);
  for (cs = 0; cs < cases + (int)NE(fixed); cs++) {
    int ta, tb, ab; size_t m, n, k;
    if (cs < (int)NE(fixed)) { ta = fixed[cs].ta; tb = fixed[cs].tb; m = fixed[cs].m; n = fixed[cs].n; k = fixed[cs].k; ab = cs % (int)NE(AB); }
    else {
      ta = (int)(rnd() & 1); tb = (int)(rnd() & 1); m = MS[rnd() % NE(MS)]; n = NS[rnd() % NE(NS)]; k = KS[rnd() % NE(KS)]; ab = (int)(rnd() % NE(AB));
      if (m * n * (k ? k : 1) > 400000000ull) { m = MS[rnd() % 14]; }
    }
    { float *a = mk(m * k), *b = mk(k * n), *c0 = mk(m * n), *c = malloc(m * n * sizeof(float));
      float al = AB[ab][0], be = AB[ab][1];
      for (v = 4; v >= 1; v /= 2) {
        if (!(have & v)) continue;
        if (v == 1 && m * n * (k ? k : 1) > 30000000ull) continue;  /* the plain-C instance: small cases only */
        memcpy(c, c0, m * n * sizeof(float)); bcir_tensor_mm_kblock_variant(v, ta, tb, m, n, k, al, a, b, be, c);
        printf("%d serial v%d %d%d %zux%zux%zu ab%d %016llx\n", cs, v, ta, tb, m, n, k, ab, (unsigned long long)fnv(c, m * n));
      }
      for (v = 4; v >= 1; v /= 2) { /* the pool through each instance (PX-3b's per-variant entry) */
        if (!(have & v)) continue;
        if (v == 1 && m * n * (k ? k : 1) > 30000000ull) continue;
        memcpy(c, c0, m * n * sizeof(float));
        if (bcir_tensor_mm_contract_variant_pool(p4, 2, v, ta, tb, m, n, k, al, a, b, be, c, NULL, 0) != v) { printf("%d refused v%d\n", cs, v); continue; }
        printf("%d vpool4 v%d %016llx\n", cs, v, (unsigned long long)fnv(c, m * n));
      }
      for (t = 0; t < 2; t++) {
        memcpy(c, c0, m * n * sizeof(float)); bcir_tensor_mm_kblock_pool(t ? p2 : p4, ta, tb, m, n, k, al, a, b, be, c);
        printf("%d pool%d %d%d %zux%zux%zu ab%d %016llx\n", cs, t ? 2 : 4, ta, tb, m, n, k, ab, (unsigned long long)fnv(c, m * n));
      }
      if (cs % 3 == 0) { /* a pair job: this call and a second of another shape */
        size_t m2 = MS[rnd() % 18], n2 = NS[rnd() % NE(NS)], k2 = KS[rnd() % NE(KS)];
        float *a2 = mk(m2 * k2), *b2 = mk(k2 * n2), *d = mk(m2 * n2);
        bcir_tensor_mm_call x = {ta, tb, m, n, k, al, a, b, be, c}, y = {tb, ta, m2, n2, k2, 0.75f, a2, b2, 1.0f, d};
        memcpy(c, c0, m * n * sizeof(float)); bcir_tensor_mm_pair_pool(p4, 2, &x, &y);
        printf("%d pair %016llx %016llx\n", cs, (unsigned long long)fnv(c, m * n), (unsigned long long)fnv(d, m2 * n2));
        free(a2); free(b2); free(d);
      }
      if (cs % 4 == 1) { /* a group job: a chain of three into c (k, 0 and another k) plus two independent calls */
        size_t kk = KS[rnd() % NE(KS)], m2 = MS[rnd() % 18], n2 = NS[rnd() % NE(NS)], k2 = KS[rnd() % NE(KS)];
        float *a1 = mk(m * kk), *b1 = mk(kk * n), *a2 = mk(m2 * k2), *b2 = mk(k2 * n2), *d = mk(m2 * n2), *e = mk(m2 * n2);
        bcir_tensor_mm_call ch[3] = {{ta, tb, m, n, k, al, a, b, be, c}, {0, 0, m, n, 0, 1.0f, a, b, 0.5f, c}, {tb, ta, m, n, kk, -1.5f, a1, b1, 1.0f, c}};
        bcir_tensor_mm_call ys[2] = {{0, 1, m2, n2, k2, 1.0f, a2, b2, 0.0f, d}, {1, 0, m2, n2, k2, 2.0f, a2, b2, 1.0f, e}};
        if (ys[1].ta) { ys[1].m = m2; ys[1].k = k2; }  /* a2 is m2 x k2 either way; with ta it is read as k2 x m2 */
        memcpy(c, c0, m * n * sizeof(float)); bcir_tensor_mm_group_pool(p4, 2, ch, 3, ys, 2);
        printf("%d group %016llx %016llx %016llx\n", cs, (unsigned long long)fnv(c, m * n), (unsigned long long)fnv(d, m2 * n2), (unsigned long long)fnv(e, m2 * n2));
        if (have & 2) { float *d0 = mk(m2 * n2), *e0 = mk(m2 * n2); ys[0].c = d0; ys[1].c = e0;
          memcpy(c, c0, m * n * sizeof(float));
          if (bcir_tensor_mm_group_variant_pool(p4, 2, 2, ch, 3, ys, 2, NULL, NULL, 0) == 2)
            printf("%d group-v2 %016llx %016llx %016llx\n", cs, (unsigned long long)fnv(c, m * n), (unsigned long long)fnv(d0, m2 * n2), (unsigned long long)fnv(e0, m2 * n2));
          else printf("%d group-v2 refused\n", cs);
          free(d0); free(e0); }
        free(a1); free(b1); free(a2); free(b2); free(d); free(e);
      }
      free(a); free(b); free(c0); free(c); }
  }
  bcir_tensor_pool_stop(p4); bcir_tensor_pool_stop(p2);
  return 0;
}
