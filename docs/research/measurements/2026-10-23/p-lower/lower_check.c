/* LOWER prototype check: the stored forward (strips from STRIP_FROM) without the future-column fill, p NaN-filled
 * first, then the backward: y, dq, dk, dv and the verdict by FNV-1a (p not hashed), one line a case -- compared
 * with the same lines from the tree as committed (the fill in place). */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t fnv(const void *x, size_t n) { const unsigned char *b = x; uint64_t h = 1469598103934665603ull; size_t i; for (i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; } return h; }
static void *alloc64(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 64)) return NULL; return p; }
int main(void) {
  static const size_t sh[][5] = {{1, 1, 1, 1, 2}, {1, 5, 2, 1, 3}, {2, 33, 4, 2, 17}, {1, 64, 2, 2, 8}, {1, 70, 2, 1, 32},
      {1, 129, 2, 1, 47}, {1, 300, 2, 1, 64}, {1, 1024, 2, 1, 4}, {1, 1031, 3, 1, 16}, {2, 1056, 4, 2, 32}, {1, 2048, 2, 2, 32}};
  static const unsigned cts[] = {0, 1, 2, 3};
  size_t s, c, i, th, ws;
  for (s = 0; s < sizeof sh / sizeof sh[0]; s++) {
    size_t b = sh[s][0], t = sh[s][1], nh = sh[s][2], nk = sh[s][3], d = sh[s][4];
    size_t nq = b * t * nh * d, nkv = b * t * nk * d, np = b * nh * t * t;
    float *q = alloc64(nq * 4), *k = alloc64(nkv * 4), *v = alloc64(nkv * 4), *p = alloc64(np * 4), *y = alloc64(nq * 4);
    float *dy = alloc64(nq * 4), *dq = alloc64(nq * 4), *dk = alloc64(nkv * 4), *dv = alloc64(nkv * 4);
    for (th = 1; th <= 4; th += 3) {
      size_t psz = bcir_tensor_pool_size((unsigned)th); void *pm = alloc64(psz);
      bcir_tensor_pool *pool = th > 1 ? bcir_tensor_pool_init(pm, psz, (unsigned)th) : NULL;
      size_t need = bcir_tensor_attention_pool_workspace(b, t, nh, nk, d, (unsigned)th); void *w = alloc64(need);
      for (c = 0; c < 4; c++) for (ws = 0; ws < 2; ws++) {
        int bad = 0;
        for (i = 0; i < nq; i++) { q[i] = sinf((float)i * 0.37f) * 1.7f; dy[i] = cosf((float)i * 0.11f) * 0.5f; }
        for (i = 0; i < nkv; i++) { k[i] = cosf((float)i * 0.53f) * 1.3f; v[i] = sinf((float)i * 0.29f) - 0.1f; }
        memset(p, 0xff, np * 4); memset(y, 0xff, nq * 4); memset(dq, 0xff, nq * 4); memset(dk, 0xff, nkv * 4); memset(dv, 0xff, nkv * 4);
        bcir_tensor_attention_contract_variant_scan_ws_pool(pool, 0, cts[c], b, t, nh, nk, d, q, k, v, p, y, &bad, ws ? w : NULL, ws ? need : 0);
        bcir_tensor_attention_backward_contract_variant_ws_pool(pool, 0, cts[c], b, t, nh, nk, d, q, k, v, p, dy, dq, dk, dv, ws ? w : NULL, ws ? need : 0);
        printf("%zux%zux%zux%zux%zu th %zu ct %u ws %zu: bad %d y=%016llx dq=%016llx dk=%016llx dv=%016llx\n", b, t, nh, nk, d, th, cts[c], ws, bad,
            (unsigned long long)fnv(y, nq * 4), (unsigned long long)fnv(dq, nq * 4), (unsigned long long)fnv(dk, nkv * 4), (unsigned long long)fnv(dv, nkv * 4));
      }
      if (pool) bcir_tensor_pool_stop(pool);
      free(pm); free(w);
    }
    free(q); free(k); free(v); free(p); free(y); free(dy); free(dq); free(dk); free(dv);
  }
  return 0;
}
