/* FWD-STRIP's cross-tree check: the stored binary32 forward at long context -- its probabilities p (every
 * float of every head, future columns included, after a NaN fill), its context y and its verdict -- by
 * FNV-1a, one line a case, so the lines of a build of the tree before and of the tree after compare
 * byte for byte. Cases: contexts 1016 to 2050 (below the threshold, at it, past it, not a multiple of 8
 * or 16), GQA groups, head widths 4 to 64, contracts v1, EXP, FUSED and EXP+FUSED, serially and on a
 * pool of four, through the entries without a workspace and through the workspace entries lent the
 * pool's need, finite and with an infinite key. usage: fwd_hash (built with one tree's bcir_tensor.c) */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fnv(const void *x, size_t n) {
  const unsigned char *b = (const unsigned char *)x; uint64_t h = 1469598103934665603ull; size_t i;
  for (i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
static void *alloc64(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 64)) return NULL; return p; }
int main(void) {
  static const size_t sh[][5] = {{1, 1016, 2, 1, 4}, {1, 1023, 2, 2, 8}, {1, 1024, 2, 1, 4}, {1, 1025, 3, 1, 17},
      {2, 1030, 4, 2, 32}, {1, 1031, 1, 1, 64}, {1, 1056, 2, 1, 33}, {1, 2048, 2, 2, 32}, {1, 2050, 4, 2, 16}};
  static const unsigned cts[] = {0, BCIR_TENSOR_ATTENTION_EXP, BCIR_TENSOR_ATTENTION_FUSED,
      BCIR_TENSOR_ATTENTION_EXP | BCIR_TENSOR_ATTENTION_FUSED};
  size_t s, c, i, th, ws, inf;
  for (s = 0; s < sizeof sh / sizeof sh[0]; s++) {
    size_t b = sh[s][0], t = sh[s][1], nh = sh[s][2], nk = sh[s][3], d = sh[s][4];
    size_t nq = b * t * nh * d, nkv = b * t * nk * d, np = b * nh * t * t;
    float *q = alloc64(nq * 4), *k = alloc64(nkv * 4), *v = alloc64(nkv * 4), *p = alloc64(np * 4), *y = alloc64(nq * 4);
    if (!q || !k || !v || !p || !y) return 1;
    for (th = 1; th <= 4; th += 3) {
      size_t psz = bcir_tensor_pool_size((unsigned)th); void *pm = alloc64(psz);
      bcir_tensor_pool *pool = th > 1 ? bcir_tensor_pool_init(pm, psz, (unsigned)th) : NULL;
      size_t need = bcir_tensor_attention_pool_workspace(b, t, nh, nk, d, (unsigned)th); void *w = alloc64(need);
      for (inf = 0; inf < 2; inf++) for (c = 0; c < 4; c++) for (ws = 0; ws < 2; ws++) {
        int bad = 0, ok;
        for (i = 0; i < nq; i++) q[i] = sinf((float)i * 0.37f) * 1.7f;
        for (i = 0; i < nkv; i++) { k[i] = cosf((float)i * 0.53f) * 1.3f; v[i] = sinf((float)i * 0.29f) - 0.1f; }
        if (inf) k[(t / 2) * nk * d] = INFINITY;
        memset(p, 0xff, np * 4); memset(y, 0xff, nq * 4);
        ok = bcir_tensor_attention_contract_variant_scan_ws_pool(pool, 0, cts[c], b, t, nh, nk, d, q, k, v, p, y, &bad,
            ws ? w : NULL, ws ? need : 0);
        printf("%zux%zux%zux%zux%zu threads %zu ct %u ws %zu inf %zu: ok %d bad %d p=%016llx y=%016llx\n", b, t, nh, nk, d,
            th, cts[c], ws, inf, ok != 0, bad, (unsigned long long)fnv(p, np * 4), (unsigned long long)fnv(y, nq * 4));
      }
      if (pool) bcir_tensor_pool_stop(pool);
      free(pm); free(w);
    }
    free(q); free(k); free(v); free(p); free(y);
  }
  return 0;
}
