/*===- bcir_sha256.c - freestanding SHA-256 and HMAC-SHA256 ---------------===
 *
 * The transform, init, update and final are the BCAB reader's (bcir_artifact_bundle.c),
 * moved verbatim; HMAC is RFC 2104 over them. No libc, no heap.
 *===----------------------------------------------------------------------===*/
#include "bcir_sha256.h"

static uint32_t rd32be(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
static void zero_bytes(void *pointer, size_t length) {
  volatile uint8_t *p = (volatile uint8_t *)pointer;
  for (size_t i = 0; i < length; ++i) p[i] = 0;
}
static void copy_bytes(void *destination, const void *source, size_t length) {
  uint8_t *d = (uint8_t *)destination;
  const uint8_t *s = (const uint8_t *)source;
  for (size_t i = 0; i < length; ++i) d[i] = s[i];
}

static const uint32_t sha_k[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
static uint32_t rotr(uint32_t value, unsigned amount) {
  return (value >> amount) | (value << (32u - amount));
}
static void sha_transform(bcir_sha256 *state, const uint8_t *block) {
  uint32_t w[64];
  for (unsigned i = 0; i < 16; ++i) w[i] = rd32be(block + i * 4u);
  for (unsigned i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
    uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }
  uint32_t a=state->h[0], b=state->h[1], c=state->h[2], d=state->h[3];
  uint32_t e=state->h[4], f=state->h[5], g=state->h[6], h=state->h[7];
  for (unsigned i = 0; i < 64; ++i) {
    uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g);
    uint32_t t1=h+s1+ch+sha_k[i]+w[i];
    uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&b)^(a&c)^(b&c);
    uint32_t t2=s0+maj;
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  state->h[0]+=a; state->h[1]+=b; state->h[2]+=c; state->h[3]+=d;
  state->h[4]+=e; state->h[5]+=f; state->h[6]+=g; state->h[7]+=h;
}

void bcir_sha256_init(bcir_sha256 *state) {
  static const uint32_t initial[8] = {
    0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
  };
  zero_bytes(state, sizeof(*state));
  for (unsigned i = 0; i < 8; ++i) state->h[i] = initial[i];
}

void bcir_sha256_update(bcir_sha256 *state, const uint8_t *data, size_t length) {
  if (!data) return;  /* NULL is valid only with length 0 */
  state->bytes += (uint64_t)length;
  while (length) {
    size_t available = 64u - state->used;
    size_t take = length < available ? length : available;
    copy_bytes(state->block + state->used, data, take);
    state->used += take; data += take; length -= take;
    if (state->used == 64u) {
      sha_transform(state, state->block);
      state->used = 0;
    }
  }
}

void bcir_sha256_final(bcir_sha256 *state, uint8_t output[32]) {
  uint64_t bits = state->bytes * UINT64_C(8);
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    while (state->used < 64u) state->block[state->used++] = 0;
    sha_transform(state, state->block); state->used = 0;
  }
  while (state->used < 56u) state->block[state->used++] = 0;
  for (unsigned i = 0; i < 8; ++i)
    state->block[63u-i] = (uint8_t)(bits >> (i*8u));
  sha_transform(state, state->block);
  for (unsigned i = 0; i < 8; ++i) {
    output[i*4u]=(uint8_t)(state->h[i]>>24); output[i*4u+1]=(uint8_t)(state->h[i]>>16);
    output[i*4u+2]=(uint8_t)(state->h[i]>>8); output[i*4u+3]=(uint8_t)state->h[i];
  }
  zero_bytes(state, sizeof(*state));
}

void bcir_sha256_digest(const uint8_t *data, size_t length, uint8_t output[32]) {
  bcir_sha256 state;
  bcir_sha256_init(&state); bcir_sha256_update(&state, data, length);
  bcir_sha256_final(&state, output);
}

/* --- HMAC-SHA256 (RFC 2104): H((K' ^ opad) || H((K' ^ ipad) || m)) --- */

void bcir_hmac_sha256_init(bcir_hmac_sha256_ctx *ctx, const uint8_t *key, size_t key_length) {
  uint8_t block[64], pad[64];
  zero_bytes(block, sizeof(block));
  if (key && key_length > 64u) bcir_sha256_digest(key, key_length, block);
  else if (key) copy_bytes(block, key, key_length);
  for (unsigned i = 0; i < 64u; ++i) pad[i] = (uint8_t)(block[i] ^ 0x36u);
  bcir_sha256_init(&ctx->inner); bcir_sha256_update(&ctx->inner, pad, 64u);
  for (unsigned i = 0; i < 64u; ++i) pad[i] = (uint8_t)(block[i] ^ 0x5cu);
  bcir_sha256_init(&ctx->outer); bcir_sha256_update(&ctx->outer, pad, 64u);
  zero_bytes(block, sizeof(block)); zero_bytes(pad, sizeof(pad));
}

void bcir_hmac_sha256_update(bcir_hmac_sha256_ctx *ctx, const uint8_t *data, size_t length) {
  bcir_sha256_update(&ctx->inner, data, length);
}

void bcir_hmac_sha256_final(bcir_hmac_sha256_ctx *ctx, uint8_t output[32]) {
  uint8_t inner[32];
  bcir_sha256_final(&ctx->inner, inner);
  bcir_sha256_update(&ctx->outer, inner, sizeof(inner));
  bcir_sha256_final(&ctx->outer, output);
  zero_bytes(inner, sizeof(inner));
}

void bcir_hmac_sha256(const uint8_t *key, size_t key_length, const uint8_t *message,
                      size_t message_length, uint8_t output[32]) {
  bcir_hmac_sha256_ctx ctx;
  bcir_hmac_sha256_init(&ctx, key, key_length);
  bcir_hmac_sha256_update(&ctx, message, message_length);
  bcir_hmac_sha256_final(&ctx, output);
}

int bcir_sha256_equal(const uint8_t *a, const uint8_t *b, size_t n) {
  uint8_t difference = 0;
  for (size_t i = 0; i < n; ++i) difference |= (uint8_t)(a[i] ^ b[i]);
  return difference == 0;
}
