/*===- bcir_rng.c - Philox4x32-10 and the streams over it (twin of bcir/frontends/models/rng.py) ---===
 *
 * See bcir_rng.h. Freestanding: the arithmetic is 32- and 64-bit unsigned, every product formed in 64 bits, every
 * wrap the generator's own (Philox's key schedule adds modulo 2^32 by definition).
 *===----------------------------------------------------------------------===*/
#include "bcir_rng.h"

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

void bcir_philox4x32_10(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]) {
  uint32_t c0 = counter[0], c1 = counter[1], c2 = counter[2], c3 = counter[3];
  uint32_t k0 = key[0], k1 = key[1];
  for (int round = 0; round < 10; round++) {
    if (round) {
      k0 += PHILOX_W0;
      k1 += PHILOX_W1;
    }
    uint64_t p0 = (uint64_t)PHILOX_M0 * c0, p1 = (uint64_t)PHILOX_M1 * c2;
    uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0, n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
    c0 = n0;
    c1 = (uint32_t)p1;
    c2 = n2;
    c3 = (uint32_t)p0;
  }
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

static void block(const uint32_t key[2], uint64_t stream, uint64_t number, uint32_t out[4]) {
  uint32_t counter[4];
  counter[0] = (uint32_t)number;
  counter[1] = (uint32_t)(number >> 32);
  counter[2] = (uint32_t)stream;
  counter[3] = (uint32_t)(stream >> 32);
  bcir_philox4x32_10(counter, key, out);
}

uint32_t bcir_rng_u32(const uint32_t key[2], uint64_t stream, uint64_t index) {
  uint32_t out[4];
  block(key, stream, index >> 2, out);
  return out[index & 3u];
}

uint64_t bcir_rng_fill_u32(const uint32_t key[2], uint64_t stream, uint64_t first, uint32_t *out, size_t n) {
  uint32_t lanes[4];
  uint64_t have = UINT64_MAX;   /* the block number `lanes` holds, none yet */
  for (size_t k = 0; k < n; k++) {
    uint64_t index = first + k;
    if (index >> 2 != have) {
      have = index >> 2;
      block(key, stream, have, lanes);
    }
    out[k] = lanes[index & 3u];
  }
  return first + n;
}

float bcir_rng_unit_f32(uint32_t bits) { return (float)(bits >> 8) * 0x1p-24f; }

double bcir_rng_unit_f64(uint32_t hi, uint32_t lo) {
  uint64_t bits = ((uint64_t)hi << 32) | lo;
  return (double)(bits >> 11) * 0x1p-53;
}

uint32_t bcir_rng_below(const uint32_t key[2], uint64_t stream, uint64_t *index, uint32_t bound) {
  if (bound == 0) return 0;
  uint64_t m = (uint64_t)bcir_rng_u32(key, stream, (*index)++) * bound;
  uint32_t low = (uint32_t)m;
  if (low < bound) {
    uint32_t threshold = (uint32_t)(0u - bound) % bound;   /* 2^32 mod bound: the rejected band */
    while (low < threshold) {
      m = (uint64_t)bcir_rng_u32(key, stream, (*index)++) * bound;
      low = (uint32_t)m;
    }
  }
  return (uint32_t)(m >> 32);
}

void bcir_rng_permutation(const uint32_t key[2], uint64_t stream, uint64_t *index, uint32_t *out, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) out[i] = i;
  for (uint32_t i = n; i-- > 1;) {
    uint32_t j = bcir_rng_below(key, stream, index, i + 1);
    uint32_t t = out[i];
    out[i] = out[j];
    out[j] = t;
  }
}
