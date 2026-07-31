/*===- bcir_jer_simd.cpp - J5: the hosted SIMD rail ------------------------===
 *
 * See bcir_jer_simd.h for the contract. The load-bearing property, restated because it is
 * the one every line here is arranged around:
 *
 *   THE ONLY CODE THAT DECIDES WHETHER A DOCUMENT IS VALID UTF-8 IS THE SCALAR RAIL.
 *
 * A vector pass answers one narrower question -- "is this block entirely ASCII?" -- which is
 * decidable by a single comparison and for which the answer "yes" implies "valid UTF-8"
 * with no further reasoning. Every other case is handed to `bcir_jer_validate_utf8`. There
 * is therefore no second UTF-8 implementation to keep in step, and the "same trace" clause
 * of J5's gate holds by construction rather than by testing (it is tested anyway).
 *
 * WHY THE TAIL IS RE-VALIDATED FROM A BLOCK BOUNDARY. When a block is not all-ASCII the
 * scalar validator is called on the REST OF THE DOCUMENT, not on that block alone. A
 * multi-byte sequence can straddle a block boundary, so validating block-by-block would
 * split a legal sequence and reject it -- and would report an offset inside a valid
 * character. Resuming at the boundary and running to the end costs the tail once and cannot
 * be wrong.
 *===----------------------------------------------------------------------===*/
#include "bcir_jer_simd.h"

#if defined(__x86_64__) || defined(_M_X64)
#define BCIR_SIMD_X86 1
#include <immintrin.h>
#elif defined(__aarch64__)
#define BCIR_SIMD_ARM 1
#include <arm_neon.h>
#endif

namespace {

/* An all-ASCII block is valid UTF-8 by inspection: every octet below 0x80 is a complete
 * one-octet sequence (X.697 7.6.2 defers to ISO/IEC 10646, and RFC 3629 2 says the same).
 * That is the entire semantic claim this file makes on its own. */

#if defined(BCIR_SIMD_X86)
/* 16 octets at a time. SSE2 is baseline on x86-64, so this path needs no feature check --
 * but it is still gated by the tier so a caller can pin scalar. */
size_t ascii_run_sse2(const uint8_t *data, size_t len) {
  size_t at = 0;
  for (; at + 16 <= len; at += 16) {
    __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + at));
    /* The sign bit of each lane IS the "octet >= 0x80" test, so one movemask settles all
     * sixteen without a compare against a constant. */
    if (_mm_movemask_epi8(block) != 0) return at;
  }
  return at;
}

__attribute__((target("avx2"))) size_t ascii_run_avx2(const uint8_t *data, size_t len) {
  size_t at = 0;
  for (; at + 32 <= len; at += 32) {
    __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + at));
    if (_mm256_movemask_epi8(block) != 0) return at;
  }
  /* Finish the remainder with the narrower width rather than dropping to scalar: a 31-octet
   * tail is the common case at the end of a document. */
  return at + ascii_run_sse2(data + at, len - at);
}
#endif

#if defined(BCIR_SIMD_ARM)
size_t ascii_run_neon(const uint8_t *data, size_t len) {
  size_t at = 0;
  for (; at + 16 <= len; at += 16) {
    uint8x16_t block = vld1q_u8(data + at);
    /* No movemask on NEON: reduce to a single maximum and compare once. An all-ASCII block
     * has a maximum below 0x80. */
    if (vmaxvq_u8(block) >= 0x80) return at;
  }
  return at;
}
#endif

size_t ascii_run_scalar(const uint8_t *data, size_t len) {
  size_t at = 0;
  while (at < len && data[at] < 0x80) at++;
  return at;
}

size_t ascii_run(bcir_jer_simd_tier tier, const uint8_t *data, size_t len) {
  switch (tier) {
#if defined(BCIR_SIMD_X86)
    case BCIR_JER_SIMD_AVX2: return ascii_run_avx2(data, len);
    case BCIR_JER_SIMD_SSE2: return ascii_run_sse2(data, len);
#endif
#if defined(BCIR_SIMD_ARM)
    case BCIR_JER_SIMD_NEON: return ascii_run_neon(data, len);
#endif
    default: return ascii_run_scalar(data, len);
  }
}

bcir_jer_simd_tier detect() {
#if defined(BCIR_SIMD_X86)
  /* __builtin_cpu_init is required before __builtin_cpu_supports on GCC; both compilers
   * accept the call. A tier the CPU does not advertise is never returned, which is the
   * whole of the "no unsupported-CPU fault" clause. */
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) return BCIR_JER_SIMD_AVX2;
  return BCIR_JER_SIMD_SSE2;
#elif defined(BCIR_SIMD_ARM)
  return BCIR_JER_SIMD_NEON;   /* Advanced SIMD is mandatory in AArch64 */
#else
  return BCIR_JER_SIMD_SCALAR;
#endif
}

bcir_jer_status validate(bcir_jer_simd_tier tier, const uint8_t *data, size_t len,
                         bcir_jer_diag *diag) {
  size_t skipped;
  if (data == nullptr || len == 0) return bcir_jer_validate_utf8(data, len, diag);

  skipped = ascii_run(tier, data, len);
  if (skipped == len) {
    /* Every octet was ASCII. Still call the scalar rail on the empty remainder so the
     * `diag` this returns is written by the same code that writes it on every other path
     * -- a success whose diagnostic was filled in by a different function is exactly the
     * drift this file exists to avoid. */
    return bcir_jer_validate_utf8(data + len, 0, diag);
  }

  /* From the first non-ASCII octet to the end, scalar. The offset the scalar rail reports
   * is relative to what it was handed, so it is rebased onto the whole document. */
  {
    bcir_jer_status status = bcir_jer_validate_utf8(data + skipped, len - skipped, diag);
    if (status != BCIR_JER_OK && diag != nullptr) diag->offset += skipped;
    return status;
  }
}

}  // namespace

extern "C" {

bcir_jer_simd_tier bcir_jer_simd_tier_available(void) {
  /* Resolved once. C++11 guarantees the initializer runs exactly once even under
   * concurrent first calls, so no explicit lock is needed and none is taken. */
  static const bcir_jer_simd_tier tier = detect();
  return tier;
}

const char *bcir_jer_simd_tier_name(bcir_jer_simd_tier tier) {
  switch (tier) {
    case BCIR_JER_SIMD_SCALAR: return "scalar";
    case BCIR_JER_SIMD_SSE2: return "sse2";
    case BCIR_JER_SIMD_AVX2: return "avx2";
    case BCIR_JER_SIMD_NEON: return "neon";
  }
  return "scalar";
}

int bcir_jer_simd_tier_compiled(bcir_jer_simd_tier tier) {
  switch (tier) {
    case BCIR_JER_SIMD_SCALAR: return 1;
#if defined(BCIR_SIMD_X86)
    case BCIR_JER_SIMD_SSE2: return 1;
    case BCIR_JER_SIMD_AVX2: return 1;
#else
    case BCIR_JER_SIMD_SSE2: return 0;
    case BCIR_JER_SIMD_AVX2: return 0;
#endif
#if defined(BCIR_SIMD_ARM)
    case BCIR_JER_SIMD_NEON: return 1;
#else
    case BCIR_JER_SIMD_NEON: return 0;
#endif
  }
  return 0;
}

bcir_jer_status bcir_jer_validate_utf8_simd(const uint8_t *data, size_t len,
                                            bcir_jer_diag *diag) {
  return validate(bcir_jer_simd_tier_available(), data, len, diag);
}

bcir_jer_status bcir_jer_validate_utf8_at(bcir_jer_simd_tier tier, const uint8_t *data,
                                          size_t len, bcir_jer_diag *diag) {
  /* A tier this build did not compile, or this CPU does not advertise, degrades to scalar.
   * Refusing would make a caller handle a machine question it did not ask. */
  if (!bcir_jer_simd_tier_compiled(tier) || tier > bcir_jer_simd_tier_available())
    tier = BCIR_JER_SIMD_SCALAR;
  return validate(tier, data, len, diag);
}

}  // extern "C"
