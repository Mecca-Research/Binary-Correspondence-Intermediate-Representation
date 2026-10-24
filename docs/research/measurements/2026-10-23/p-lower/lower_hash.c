/* P-LOWER's hash witness: the stored forward under a contract with BCIR_TENSOR_ATTENTION_LOWER (16) -- which the
 * tree before P-LOWER ignores, filling every future column, and the tree after it honors from BCIR_ATT_STRIP_FROM,
 * skipping the strips' fill -- p filled with NaN bytes first, then the in-place backward from that p under the same
 * contract: per case one line of FNV-1a digests of y, the verdict, p's probabilities j <= i, dq, dk and dv. The two
 * trees' lines are compared whole; p's future columns are not hashed (P-LOWER leaves them as p held them).
 * Built against one tree's bcir_tensor.c. usage: lower_hash (no arguments) */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
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
  static const size_t sh[][5] = {{1, 1, 1, 1, 2}, {1, 5, 2, 1, 3}, {2, 33, 4, 2, 17}, {1, 64, 2, 2, 8}, {1, 70, 2, 1, 32},
      {1, 129, 2, 1, 47}, {1, 300, 2, 1, 64}, {1, 1023, 2, 1, 4}, {1, 1024, 2, 1, 4}, {1, 1031, 3, 1, 16},
      {2, 1056, 4, 2, 32}, {1, 2048, 2, 2, 32}, {4, 2048, 4, 4, 32}};
  size_t s, c, i, th, ws, r, row;
  for (s = 0; s < sizeof sh / sizeof sh[0]; s++) {
    size_t b = sh[s][0], t = sh[s][1], nh = sh[s][2], nk = sh[s][3], d = sh[s][4];
    size_t nq = b * t * nh * d, nkv = b * t * nk * d, np = b * nh * t * t;
    float *q = alloc64(nq * 4), *k = alloc64(nkv * 4), *v = alloc64(nkv * 4), *p = alloc64(np * 4), *y = alloc64(nq * 4);
    float *dy = alloc64(nq * 4), *dq = alloc64(nq * 4), *dk = alloc64(nkv * 4), *dv = alloc64(nkv * 4);
    if (!q || !k || !v || !p || !y || !dy || !dq || !dk || !dv) return 1;
    for (th = 1; th <= 4; th += 3) {
      size_t psz = bcir_tensor_pool_size((unsigned)th); void *pm = alloc64(psz);
      bcir_tensor_pool *pool = th > 1 ? bcir_tensor_pool_init(pm, psz, (unsigned)th) : NULL;
      size_t need = bcir_tensor_attention_pool_workspace(b, t, nh, nk, d, (unsigned)th); void *w = alloc64(need);
      for (c = 0; c < 4; c++) for (ws = 0; ws < 2; ws++) {
        unsigned ct = (unsigned)c | 16u;  /* EXP 1, FUSED 2; LOWER 16 */
        int bad = 0; uint64_t hp = 1469598103934665603ull;
        for (i = 0; i < nq; i++) { q[i] = sinf((float)i * 0.37f) * 1.7f; dy[i] = cosf((float)i * 0.11f) * 0.5f; }
        for (i = 0; i < nkv; i++) { k[i] = cosf((float)i * 0.53f) * 1.3f; v[i] = sinf((float)i * 0.29f) - 0.1f; }
        memset(p, 0xff, np * 4); memset(y, 0xff, nq * 4); memset(dq, 0xff, nq * 4); memset(dk, 0xff, nkv * 4);
        memset(dv, 0xff, nkv * 4);
        if (!bcir_tensor_attention_contract_variant_scan_ws_pool(pool, 0, ct, b, t, nh, nk, d, q, k, v, p, y, &bad,
                ws ? w : NULL, ws ? need : 0)) return 1;
        for (r = 0; r < b * nh; r++) for (row = 0; row < t; row++) hp = fnv(hp, p + (r * t + row) * t, (row + 1) * 4);
        if (!bcir_tensor_attention_backward_contract_variant_ws_pool(pool, 0, ct, b, t, nh, nk, d, q, k, v, p, dy, dq, dk,
                dv, ws ? w : NULL, ws ? need : 0)) return 1;
        printf("%zux%zux%zux%zux%zu th %zu ct %u ws %zu: bad %d y=%016llx p=%016llx dq=%016llx dk=%016llx dv=%016llx\n",
            b, t, nh, nk, d, th, ct, ws, bad, (unsigned long long)fnv(1469598103934665603ull, y, nq * 4),
            (unsigned long long)hp, (unsigned long long)fnv(1469598103934665603ull, dq, nq * 4),
            (unsigned long long)fnv(1469598103934665603ull, dk, nkv * 4),
            (unsigned long long)fnv(1469598103934665603ull, dv, nkv * 4));
      }
      if (pool) bcir_tensor_pool_stop(pool);
      free(pm); free(w);
    }
    free(q); free(k); free(v); free(p); free(y); free(dy); free(dq); free(dk); free(dv);
  }
  return 0;
}
