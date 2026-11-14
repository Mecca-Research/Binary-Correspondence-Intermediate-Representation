/*===- bcir_tok_hash.h - the tokenizers' keyed table hash (PORT-2) -------------------===
 *
 * One copy of what bcir_bpe.c and bcir_spm.c key their open-addressed tables with: SipHash-2-4
 * under a key taken from the loaded file's own SHA-256. The tables stay deterministic, and a file
 * cannot choose its collisions without first choosing its own digest; text handed to encode or
 * decode only looks keys up, so it can grow no table. Header-only and freestanding (no libc).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_TOK_HASH_H
#define BCIR_TOK_HASH_H

#include <stddef.h>
#include <stdint.h>

static inline uint64_t bcir_tok_rotl(uint64_t x, unsigned b) { return (x << b) | (x >> (64u - b)); }

/* Eight octets, little-endian. */
static inline uint64_t bcir_tok_le64(const uint8_t *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8u; i++) v |= (uint64_t)p[i] << (8u * i);
  return v;
}

#define BCIR_TOK_SIPROUND(v0, v1, v2, v3)                                        \
  do {                                                                           \
    v0 += v1; v1 = bcir_tok_rotl(v1, 13); v1 ^= v0; v0 = bcir_tok_rotl(v0, 32);  \
    v2 += v3; v3 = bcir_tok_rotl(v3, 16); v3 ^= v2;                              \
    v0 += v3; v3 = bcir_tok_rotl(v3, 21); v3 ^= v0;                              \
    v2 += v1; v1 = bcir_tok_rotl(v1, 17); v1 ^= v2; v2 = bcir_tok_rotl(v2, 32);  \
  } while (0)

/* SipHash-2-4 under (k0, k1) of the message `tag` (8 octets, little-endian) || data[0..len). */
static inline uint64_t bcir_tok_siphash(uint64_t k0, uint64_t k1, uint64_t tag, const uint8_t *data, size_t len) {
  uint64_t v0 = k0 ^ UINT64_C(0x736f6d6570736575), v1 = k1 ^ UINT64_C(0x646f72616e646f6d);
  uint64_t v2 = k0 ^ UINT64_C(0x6c7967656e657261), v3 = k1 ^ UINT64_C(0x7465646279746573);
  uint64_t m;
  size_t i = 0;
  v3 ^= tag;
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  v0 ^= tag;
  for (; i + 8u <= len; i += 8u) {
    m = bcir_tok_le64(data + i);
    v3 ^= m;
    BCIR_TOK_SIPROUND(v0, v1, v2, v3);
    BCIR_TOK_SIPROUND(v0, v1, v2, v3);
    v0 ^= m;
  }
  m = ((uint64_t)len + 8u) << 56;
  for (size_t k = 0; i + k < len; k++) m |= (uint64_t)data[i + k] << (8u * k);
  v3 ^= m;
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  v0 ^= m;
  v2 ^= 0xFFu;
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  BCIR_TOK_SIPROUND(v0, v1, v2, v3);
  return v0 ^ v1 ^ v2 ^ v3;
}

/* The hash of two 32-bit keys under `tag`. */
static inline uint64_t bcir_tok_hash2(uint64_t k0, uint64_t k1, uint64_t tag, uint32_t a, uint32_t b) {
  uint8_t k[8];
  for (unsigned i = 0; i < 8u; i++) k[i] = (uint8_t)((i < 4u ? a >> (8u * i) : b >> (8u * (i - 4u))) & 0xFFu);
  return bcir_tok_siphash(k0, k1, tag, k, 8u);
}

/* The table size for `n` keys: a power of two at least 16 and at least n. */
static inline size_t bcir_tok_pow2(size_t n) {
  size_t c = 16u;
  while (c < n) c *= 2u;
  return c;
}

#endif /* BCIR_TOK_HASH_H */
