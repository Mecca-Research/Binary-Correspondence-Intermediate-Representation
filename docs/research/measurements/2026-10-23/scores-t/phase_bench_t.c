/* The forward's phases at one shape over an instrumented unit (ATT-BOUND's instrument_phases.py: [0] scores [1] softmax
 * [2] context [3] max [4] exp+sum [5] division; SCORES-T's instrument_t.py: [0] scores [1] softmax [2] context [3] out
 * [4] max [5] exp+sum [6] division), lent the pool's workspace as the trainer lends it; TSC cycles a call over ROUNDS
 * after one, summed over the threads, every counter printed raw. usage: phase_bench_t B T H KH D THREADS ROUNDS CONTRACT */
#define _POSIX_C_SOURCE 200809L
#include "bcir_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
extern unsigned long long bcir_ar_cycles[8];
static void *alloc64(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n ? n : 64)) exit(3); memset(p, 0, n ? n : 64); return p; }
int main(int argc, char **argv) {
  size_t b, t, nh, nk, d, nq, nkv, np, i, wsb; unsigned threads, ct; int rounds, r, bad, k;
  bcir_tensor_pool *pool = NULL; void *ws; float *q, *kk, *v, *y, *p; unsigned long long t0, all;
  if (argc != 9) return 2;
  b = strtoul(argv[1], 0, 10); t = strtoul(argv[2], 0, 10); nh = strtoul(argv[3], 0, 10); nk = strtoul(argv[4], 0, 10);
  d = strtoul(argv[5], 0, 10); threads = (unsigned)atoi(argv[6]); rounds = atoi(argv[7]); ct = (unsigned)atoi(argv[8]);
  nq = b * t * nh * d; nkv = b * t * nk * d;
  np = (ct & BCIR_TENSOR_ATTENTION_SCRATCH) ? bcir_tensor_attention_scratch_floats(b, t, nh) : b * nh * t * t;
  q = alloc64(nq * 4); kk = alloc64(nkv * 4); v = alloc64(nkv * 4); y = alloc64(nq * 4); p = alloc64(np * 4);
  if (threads > 1) { size_t ps = bcir_tensor_pool_size(threads); pool = bcir_tensor_pool_init(alloc64(ps), ps, threads); }
  wsb = bcir_tensor_attention_pool_workspace(b, t, nh, nk, d, threads); ws = wsb ? alloc64(wsb) : NULL;
  for (i = 0; i < nq; i++) q[i] = sinf((float)i * 0.37f) * 1.7f;
  for (i = 0; i < nkv; i++) { kk[i] = cosf((float)i * 0.53f) * 1.3f; v[i] = sinf((float)i * 0.29f) - 0.1f; }
  bcir_tensor_attention_contract_variant_scan_ws_pool(pool, 0, ct, b, t, nh, nk, d, q, kk, v, p, y, &bad, ws, wsb);
  memset(bcir_ar_cycles, 0, sizeof bcir_ar_cycles);
  t0 = __rdtsc();
  for (r = 0; r < rounds; r++) bcir_tensor_attention_contract_variant_scan_ws_pool(pool, 0, ct, b, t, nh, nk, d, q, kk, v, p, y, &bad, ws, wsb);
  all = __rdtsc() - t0;
  printf("{\"shape\":[%zu,%zu,%zu,%zu,%zu],\"threads\":%u,\"contract\":%u,\"workspace\":%zu,\"rounds\":%d,\"wall_cycles_per_call\":%.0f,"
         "\"probabilities\":%zu,\"counters\":[", b, t, nh, nk, d, threads, ct, wsb, rounds, (double)all / rounds, b * nh * t * (t + 1) / 2);
  for (k = 0; k < 8; k++) printf("%s%.0f", k ? "," : "", (double)bcir_ar_cycles[k] / rounds);
  puts("]}");
  if (pool) bcir_tensor_pool_stop(pool);
  return 0;
}
