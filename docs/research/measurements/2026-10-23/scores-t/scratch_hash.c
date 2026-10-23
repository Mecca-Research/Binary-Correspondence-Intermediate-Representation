/* SCORES-T's cross-tree check for the scratch forward (INFER-ATTN) and the stored forward: per case one line of
 * FNV-1a digests -- the scratch forward's context y and verdict; the stored forward's p (every float, future columns
 * included, after a NaN fill -- under LOWER too where STRICT_P is 1, the default; with STRICT_P 0 under LOWER only the
 * probabilities j <= i), y and verdict -- over contexts short and long (multiples of 16 and not), head widths 4 to 256,
 * GQA groups, the four contracts with and without LOWER, serially and on a pool of four (HASH_MAXTH 1: serially
 * alone), without a workspace and lent the pool's need, finite and with an infinite key. Built against one tree's
 * bcir_tensor.c; two builds are one arithmetic when every line matches. usage: scratch_hash (no arguments) */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#ifndef STRICT_P
#define STRICT_P 1
#endif
#ifndef HASH_MAXTH
#define HASH_MAXTH 4
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fnv(uint64_t h, const void *x, size_t n) {
  const unsigned char *b = x; size_t i;
  for (i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
static void *alloc64(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 64)) return NULL; return p; }
int main(void) {
  static const size_t sh[][5] = {{1, 16, 1, 1, 4}, {1, 31, 2, 1, 16}, {1, 32, 2, 2, 32}, {2, 48, 4, 2, 17},
      {1, 100, 2, 1, 33}, {1, 128, 2, 1, 64}, {1, 512, 4, 4, 32}, {1, 1000, 2, 1, 32}, {1, 1024, 2, 2, 130},
      {2, 1040, 4, 2, 32}, {1, 1056, 2, 1, 256}, {1, 2048, 2, 2, 32}, {1, 2050, 4, 2, 16}};
  static const unsigned cts[] = {0, BCIR_TENSOR_ATTENTION_EXP, BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP | BCIR_TENSOR_ATTENTION_FUSED};
  size_t s, c, i, th, ws, inf, lw, mode;
  for (s = 0; s < sizeof sh / sizeof sh[0]; s++) {
    size_t b = sh[s][0], t = sh[s][1], nh = sh[s][2], nk = sh[s][3], d = sh[s][4];
    size_t nq = b * t * nh * d, nkv = b * t * nk * d, np = b * nh * t * t;
    size_t sf = bcir_tensor_attention_scratch_floats(b, t, nh);
    float *q = alloc64(nq * 4), *k = alloc64(nkv * 4), *v = alloc64(nkv * 4), *y = alloc64(nq * 4);
    float *p = alloc64(np * 4), *sp = alloc64(sf * 4);
    if (!q || !k || !v || !y || !p || !sp) return 1;
    for (i = 0; i < nq; i++) q[i] = sinf((float)i * 0.37f + (float)s) * 1.9f;
    for (i = 0; i < nkv; i++) { k[i] = cosf((float)i * 0.53f - (float)s) * 1.4f; v[i] = sinf((float)i * 0.29f) - 0.1f; }
    for (inf = 0; inf < 2; inf++) {
      if (inf) k[(nkv / 2) | 1] = INFINITY;
      for (c = 0; c < 4; c++) for (lw = 0; lw < 2; lw++) for (mode = 0; mode < 2; mode++) for (th = 1; th <= HASH_MAXTH; th += 3) {
        size_t psize = bcir_tensor_pool_size((unsigned)th); void *pm = NULL; bcir_tensor_pool *pool = NULL;
        unsigned ct = cts[c] | (lw ? BCIR_TENSOR_ATTENTION_LOWER : 0u) | (mode ? 0u : BCIR_TENSOR_ATTENTION_SCRATCH);
        if (th > 1) {
          if (!psize) continue;
          pm = alloc64((psize + 63) / 64 * 64);
          pool = bcir_tensor_pool_init(pm, psize, (unsigned)th);
          if (!pool) return 2;
        }
        if (!mode && lw) { if (pool) bcir_tensor_pool_stop(pool); free(pm); continue; }  /* LOWER: a stored forward's word */
        for (ws = 0; ws < 2; ws++) {
          size_t need = ws ? bcir_tensor_attention_pool_workspace(b, t, nh, nk, d, (unsigned)th) : 0;
          void *w = need ? alloc64(need) : NULL; int bad = -1; uint64_t h = 1469598103934665603ull;
          float *pp = mode ? p : sp; size_t pn = mode ? np : sf;
          memset(pp, 0xff, pn * 4);  /* NaN bytes */
          memset(y, 0xff, nq * 4);
          if (w) memset(w, 0xff, need);
          bcir_tensor_attention_contract_variant_scan_ws_pool(pool, 0, ct, b, t, nh, nk, d, q, k, v, pp, y, &bad, w, need);
          h = fnv(h, y, nq * 4); h = fnv(h, &bad, sizeof bad);
          if (mode) {
            if (!lw || STRICT_P) h = fnv(h, p, np * 4);
            else {
              size_t r, row;
              for (r = 0; r < b * nh; r++) for (row = 0; row < t; row++) h = fnv(h, p + (r * t + row) * t, (row + 1) * 4);
            }
          }
          printf("%zu %zu %zu %zu %zu %s ct%u th%zu ws%zu inf%zu %016llx\n", b, t, nh, nk, d, mode ? "stored" : "scratch",
                 ct, th, ws, inf, (unsigned long long)h);
          free(w);
        }
        if (pool) bcir_tensor_pool_stop(pool);
        free(pm);
      }
    }
    free(q); free(k); free(v); free(y); free(p); free(sp);
  }
  return 0;
}
