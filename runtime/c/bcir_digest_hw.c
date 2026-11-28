/*===- bcir_digest_hw.c - SHA-256 and CRC-32 on the CPU's own instructions ---------------------------===
 *
 * See bcir_digest_hw.h. The portable code is bcir_sha256.c's and bcir_crc32.h's: what no path here takes (a
 * partial block, a CRC tail, a mask of 0, a build without the paths) goes to it.
 *===----------------------------------------------------------------------===*/
#include "bcir_digest_hw.h"

#include "bcir_crc32.h"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define BCIR_DIGEST_X86 1
#include <cpuid.h>
#include <immintrin.h>
#define BCIR_TARGET_SHANI __attribute__((target("sha,ssse3,sse4.1")))
#define BCIR_TARGET_CLMUL __attribute__((target("pclmul,sse2")))
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define BCIR_DIGEST_ARM 1
#include <arm_acle.h>
#include <arm_neon.h>
#if defined(__linux__) && __STDC_HOSTED__
#include <sys/auxv.h>
#endif
#define BCIR_TARGET_ARM_SHA2 __attribute__((target("+sha2")))
#define BCIR_TARGET_ARM_CRC __attribute__((target("+crc")))
#endif

#ifdef BCIR_DIGEST_HW_WITNESS
unsigned long bcir_digest_hw_witness[4];
#define BCIR_WITNESS(bit) (bcir_digest_hw_witness[(bit) == 1u ? 0 : (bit) == 2u ? 1 : (bit) == 4u ? 2 : 3]++)
#else
#define BCIR_WITNESS(bit) ((void)0)
#endif

#if defined(BCIR_DIGEST_X86) || defined(BCIR_DIGEST_ARM)
/* FIPS 180-4 section 4.2.2, as bcir_sha256.c spells them */
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
#endif

#ifdef BCIR_DIGEST_X86
/* `n` 64-byte blocks into h, SHA-NI. The state travels as ABEF and CDGH; quad k of the schedule is
 * msg2(msg1(W[k-4], W[k-3]) + (W[k-2] .. W[k-1] shifted one word), W[k-1]), a window of four. */
BCIR_TARGET_SHANI static void sha256_blocks_shani(uint32_t h[8], const uint8_t *p, size_t n) {
  const __m128i swap = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL); /* big-endian words */
  __m128i t = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)(const void *)&h[0]), 0xB1);
  __m128i s1 = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)(const void *)&h[4]), 0x1B);
  __m128i s0 = _mm_alignr_epi8(t, s1, 8); /* ABEF */
  s1 = _mm_blend_epi16(s1, t, 0xF0);      /* CDGH */
  BCIR_WITNESS(BCIR_DIGEST_SHA256_SHANI);
  for (; n; n--, p += 64) {
    __m128i abef = s0, cdgh = s1, q[4];
    for (int k = 0; k < 4; k++)
      q[k] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(const void *)(p + 16 * k)), swap);
    for (int k = 0; k < 16; k++) {
      if (k >= 4) {
        __m128i w = _mm_add_epi32(_mm_sha256msg1_epu32(q[k & 3], q[(k + 1) & 3]),
                                  _mm_alignr_epi8(q[(k + 3) & 3], q[(k + 2) & 3], 4));
        q[k & 3] = _mm_sha256msg2_epu32(w, q[(k + 3) & 3]);
      }
      __m128i m = _mm_add_epi32(q[k & 3], _mm_loadu_si128((const __m128i *)(const void *)&sha_k[4 * k]));
      s1 = _mm_sha256rnds2_epu32(s1, s0, m);
      s0 = _mm_sha256rnds2_epu32(s0, s1, _mm_shuffle_epi32(m, 0x0E));
    }
    s0 = _mm_add_epi32(s0, abef);
    s1 = _mm_add_epi32(s1, cdgh);
  }
  t = _mm_shuffle_epi32(s0, 0x1B);
  s1 = _mm_shuffle_epi32(s1, 0xB1);
  _mm_storeu_si128((__m128i *)(void *)&h[0], _mm_blend_epi16(t, s1, 0xF0));
  _mm_storeu_si128((__m128i *)(void *)&h[4], _mm_alignr_epi8(s1, t, 8));
}

/* x folded sixteen bytes on: its low half times k's low, its high half times k's high, carry-less */
BCIR_TARGET_CLMUL static __m128i crc_fold(__m128i x, __m128i k) {
  return _mm_xor_si128(_mm_clmulepi64_si128(x, k, 0x00), _mm_clmulepi64_si128(x, k, 0x11));
}

BCIR_TARGET_CLMUL static __m128i crc_load(const uint8_t *p) {
  return _mm_loadu_si128((const __m128i *)(const void *)p);
}

/* `len` >= 64 bytes on from `state`. Four lanes fold sixty-four bytes a step (x^544, x^480); they fold into one
 * (x^160, x^96), which takes any whole sixteen bytes left; the sixteen-byte remainder is the CRC of what it
 * replaced, from state 0, so the table reduces it and then takes the tail. */
BCIR_TARGET_CLMUL static uint32_t crc32_clmul(uint32_t state, const uint8_t *p, size_t len) {
  const __m128i k512 = _mm_set_epi64x(0x1c6e41596LL, 0x154442bd4LL);
  const __m128i k128 = _mm_set_epi64x(0x0ccaa009eLL, 0x1751997d0LL);
  __m128i x0 = _mm_xor_si128(crc_load(p), _mm_cvtsi32_si128((int)state));
  __m128i x1 = crc_load(p + 16), x2 = crc_load(p + 32), x3 = crc_load(p + 48);
  BCIR_WITNESS(BCIR_DIGEST_CRC32_CLMUL);
  p += 64;
  len -= 64;
  for (; len >= 64; p += 64, len -= 64) {
    x0 = _mm_xor_si128(crc_fold(x0, k512), crc_load(p));
    x1 = _mm_xor_si128(crc_fold(x1, k512), crc_load(p + 16));
    x2 = _mm_xor_si128(crc_fold(x2, k512), crc_load(p + 32));
    x3 = _mm_xor_si128(crc_fold(x3, k512), crc_load(p + 48));
  }
  __m128i x = _mm_xor_si128(crc_fold(x0, k128), x1);
  x = _mm_xor_si128(crc_fold(x, k128), x2);
  x = _mm_xor_si128(crc_fold(x, k128), x3);
  for (; len >= 16; p += 16, len -= 16) x = _mm_xor_si128(crc_fold(x, k128), crc_load(p));
  uint8_t rest[16];
  _mm_storeu_si128((__m128i *)(void *)rest, x);
  return bcir_crc32_step(bcir_crc32_step(0u, rest, 16u), p, len);
}
#endif

#ifdef BCIR_DIGEST_ARM
/* `n` 64-byte blocks into h, the ARMv8 SHA-2 instructions: quad k + 4 of the schedule is
 * su1(su0(W[k], W[k+1]), W[k+2], W[k+3]). */
BCIR_TARGET_ARM_SHA2 static void sha256_blocks_armv8(uint32_t h[8], const uint8_t *p, size_t n) {
  uint32x4_t s0 = vld1q_u32(&h[0]), s1 = vld1q_u32(&h[4]);
  BCIR_WITNESS(BCIR_DIGEST_SHA256_ARMV8);
  for (; n; n--, p += 64) {
    uint32x4_t abcd = s0, efgh = s1, q[4];
    for (int k = 0; k < 4; k++) q[k] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 16 * k)));
    for (int k = 0; k < 16; k++) {
      uint32x4_t m = vaddq_u32(q[k & 3], vld1q_u32(&sha_k[4 * k]));
      if (k < 12)
        q[k & 3] = vsha256su1q_u32(vsha256su0q_u32(q[k & 3], q[(k + 1) & 3]), q[(k + 2) & 3], q[(k + 3) & 3]);
      uint32x4_t was = s0;
      s0 = vsha256hq_u32(s0, s1, m);
      s1 = vsha256h2q_u32(s1, was, m);
    }
    s0 = vaddq_u32(s0, abcd);
    s1 = vaddq_u32(s1, efgh);
  }
  vst1q_u32(&h[0], s0);
  vst1q_u32(&h[4], s1);
}

/* the CRC32 instructions take the reflected update bcir_crc32_step makes, eight bytes (least first) a step */
BCIR_TARGET_ARM_CRC static uint32_t crc32_armv8(uint32_t state, const uint8_t *p, size_t len) {
  BCIR_WITNESS(BCIR_DIGEST_CRC32_ARMV8);
  for (; len >= 8; p += 8, len -= 8) {
    uint64_t v = 0;
    for (int b = 7; b >= 0; b--) v = v << 8 | p[b];
    state = __crc32d(state, v);
  }
  for (; len; p++, len--) state = __crc32b(state, *p);
  return state;
}
#endif

unsigned bcir_digest_compiled(void) {
#if defined(BCIR_DIGEST_X86)
  return BCIR_DIGEST_SHA256_SHANI | BCIR_DIGEST_CRC32_CLMUL;
#elif defined(BCIR_DIGEST_ARM)
  return BCIR_DIGEST_SHA256_ARMV8 | BCIR_DIGEST_CRC32_ARMV8;
#else
  return 0u;
#endif
}

unsigned bcir_digest_probe(void) {
  unsigned found = 0u;
#if defined(BCIR_DIGEST_X86)
  unsigned a, b, c, d;
  if (__get_cpuid(1u, &a, &b, &c, &d)) {
    unsigned ssse3 = c >> 9 & 1u, sse41 = c >> 19 & 1u, pclmul = c >> 1 & 1u;
    if (pclmul) found |= BCIR_DIGEST_CRC32_CLMUL;
    if (ssse3 && sse41 && __get_cpuid_max(0u, 0) >= 7u) {
      __cpuid_count(7u, 0u, a, b, c, d);
      if (b >> 29 & 1u) found |= BCIR_DIGEST_SHA256_SHANI;
    }
  }
#elif defined(BCIR_DIGEST_ARM)
#if defined(__ARM_FEATURE_SHA2)
  found |= BCIR_DIGEST_SHA256_ARMV8;
#endif
#if defined(__ARM_FEATURE_CRC32)
  found |= BCIR_DIGEST_CRC32_ARMV8;
#endif
#if defined(__linux__) && __STDC_HOSTED__
  /* <asm/hwcap.h>: HWCAP_SHA2 is bit 6, HWCAP_CRC32 bit 7 */
  unsigned long hwcap = getauxval(AT_HWCAP);
  if (hwcap >> 6 & 1ul) found |= BCIR_DIGEST_SHA256_ARMV8;
  if (hwcap >> 7 & 1ul) found |= BCIR_DIGEST_CRC32_ARMV8;
#endif
#endif
  return found;
}

void bcir_sha256_update_hw(bcir_sha256 *state, const uint8_t *data, size_t length, unsigned features) {
  features &= bcir_digest_compiled();
  if (!(features & (BCIR_DIGEST_SHA256_SHANI | BCIR_DIGEST_SHA256_ARMV8)) || !data || length < 64u) {
    bcir_sha256_update(state, data, length);
    return;
  }
  if (state->used) { /* finish the buffered block the portable way */
    size_t take = 64u - state->used;
    bcir_sha256_update(state, data, take);
    data += take;
    length -= take;
  }
  size_t blocks = length / 64u;
#if defined(BCIR_DIGEST_X86)
  if (blocks) sha256_blocks_shani(state->h, data, blocks);
#elif defined(BCIR_DIGEST_ARM)
  if (blocks) sha256_blocks_armv8(state->h, data, blocks);
#endif
  state->bytes += (uint64_t)blocks * 64u;
  bcir_sha256_update(state, data + blocks * 64u, length - blocks * 64u);
}

void bcir_sha256_digest_hw(const uint8_t *data, size_t length, uint8_t output[32], unsigned features) {
  bcir_sha256 state;
  bcir_sha256_init(&state);
  bcir_sha256_update_hw(&state, data, length, features);
  bcir_sha256_final(&state, output);
}

uint32_t bcir_crc32_step_hw(uint32_t state, const uint8_t *data, size_t length, unsigned features) {
  features &= bcir_digest_compiled();
#if defined(BCIR_DIGEST_X86)
  if ((features & BCIR_DIGEST_CRC32_CLMUL) && length >= 64u) return crc32_clmul(state, data, length);
#elif defined(BCIR_DIGEST_ARM)
  if ((features & BCIR_DIGEST_CRC32_ARMV8) && length) return crc32_armv8(state, data, length);
#endif
  return bcir_crc32_step(state, data, length);
}

uint32_t bcir_crc32_hw(const uint8_t *data, size_t length, unsigned features) {
  return bcir_crc32_step_hw(0xFFFFFFFFu, data, length, features) ^ 0xFFFFFFFFu;
}
