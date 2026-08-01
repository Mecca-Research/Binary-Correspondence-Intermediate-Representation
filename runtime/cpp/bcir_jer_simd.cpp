/*===- bcir_jer_simd.cpp - J5: the hosted SIMD rail ------------------------===
 *
 * See bcir_jer_simd.h for the contract. The load-bearing property, restated because it is
 * the one every line here is arranged around:
 *
 *   THE ONLY CODE THAT DECIDES WHETHER A DOCUMENT IS VALID UTF-8 IS THE SCALAR RAIL.
 *
 * The vector passes answer only WHERE the ASCII and non-ASCII runs are. They never answer
 * what is valid: every verdict comes from `bcir_jer_validate_utf8` itself, so there is no
 * second UTF-8 implementation to keep in step and the "same trace" clause of J5's gate holds
 * by construction rather than by testing (it is tested anyway).
 *
 * WHY ALTERNATING RUNS ARE SOUND, AND WHY NO SECOND VALIDATOR IS NEEDED. The obvious way to
 * accelerate multi-byte text is to vectorize UTF-8 validation itself. That would be a second
 * definition of "valid UTF-8", which is the risk 8's table names -- and a bug in the fast one
 * produces a WRONG ACCEPT, the silent failure.
 *
 * It is also unnecessary. An ASCII octet can never be a CONTINUATION octet, because
 * continuations are 0x80-0xBF. So the next ASCII octet is always a sequence boundary, and no
 * legal multi-byte sequence can span it. That means [first non-ASCII, next ASCII) can be
 * handed to the scalar rail IN ISOLATION and yields exactly the answer validating it in
 * context would -- including for a truncated sequence, which is invalid either way. So the
 * two runs alternate, each vectorized, and the scalar rail sees only the short multi-byte
 * stretches.
 *
 * The earlier version handed everything from the first non-ASCII octet to the END of the
 * document to scalar, so one "cafe" near the front cost a 29 KB document its entire
 * acceleration.
 *
 * WHY THIS SHAPE DOES NOT CARRY OVER TO THE STRUCTURAL SCAN. `bcir_jer_validate_utf8` has no
 * cost budget, so skipping an ASCII run is semantically free. `bcir_jer_scan` charges ONE
 * WORK UNIT PER OCTET against 4.3's `work` ceiling, and BCIR_JER_WORK_EXCEEDED carries the
 * exact octet at which the budget ran out. A vector pass that skipped a run without charging
 * for it would accept documents the scalar rail rejects. A bulk charge IS exact, though:
 * the charge is one unit per octet at that octet's own position, so a crossing run's failure
 * point is arithmetic rather than a re-walk.
 *
 * The remaining difference is layering, and it is now solved rather than open: `bcir_jer_scan`
 * holds its state internally, so it cannot be WRAPPED the way this file wraps the validator.
 * `bcir_jer_scan_cursor` exports that state instead, and `bcir_jer_index.cpp` rebuilds only
 * the DISPATCH on top of it -- a second dispatch loop, not the stage-2 second scanner this
 * comment used to predict. It reuses this file's tier detection rather than repeating it.
 * See roadmap 7.4.1.
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

/* The vector helpers stop at a BLOCK boundary, which is a lower bound rather than the exact
 * position. Both runs are finished by a scalar loop of at most one block, because the
 * alternation below needs each run to be EXACT: a run that stopped early while still on its
 * own class would make no progress and spin. */
size_t ascii_run(bcir_jer_simd_tier tier, const uint8_t *data, size_t len) {
  size_t at;
  switch (tier) {
#if defined(BCIR_SIMD_X86)
    case BCIR_JER_SIMD_AVX2: at = ascii_run_avx2(data, len); break;
    case BCIR_JER_SIMD_SSE2: at = ascii_run_sse2(data, len); break;
#endif
#if defined(BCIR_SIMD_ARM)
    case BCIR_JER_SIMD_NEON: at = ascii_run_neon(data, len); break;
#endif
    default: at = 0; break;
  }
  while (at < len && data[at] < 0x80) at++;
  return at;
}

/* The mirror image: how far the NON-ASCII run extends. Vectorized for the same reason the
 * ASCII run is -- a long CJK or emoji stretch is exactly the case the old code sent to
 * scalar one octet at a time. */
#if defined(BCIR_SIMD_X86)
size_t nonascii_run_sse2(const uint8_t *data, size_t len) {
  size_t at = 0;
  for (; at + 16 <= len; at += 16) {
    __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + at));
    /* Every lane's sign bit set means every octet is >= 0x80. */
    if (_mm_movemask_epi8(block) != 0xFFFF) return at;
  }
  return at;
}

__attribute__((target("avx2"))) size_t nonascii_run_avx2(const uint8_t *data, size_t len) {
  size_t at = 0;
  for (; at + 32 <= len; at += 32) {
    __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + at));
    if (_mm256_movemask_epi8(block) != -1) return at;
  }
  return at + nonascii_run_sse2(data + at, len - at);
}
#endif

#if defined(BCIR_SIMD_ARM)
size_t nonascii_run_neon(const uint8_t *data, size_t len) {
  size_t at = 0;
  for (; at + 16 <= len; at += 16) {
    uint8x16_t block = vld1q_u8(data + at);
    /* An all-non-ASCII block has a MINIMUM at or above 0x80. */
    if (vminvq_u8(block) < 0x80) return at;
  }
  return at;
}
#endif

size_t nonascii_run(bcir_jer_simd_tier tier, const uint8_t *data, size_t len) {
  size_t at;
  switch (tier) {
#if defined(BCIR_SIMD_X86)
    case BCIR_JER_SIMD_AVX2: at = nonascii_run_avx2(data, len); break;
    case BCIR_JER_SIMD_SSE2: at = nonascii_run_sse2(data, len); break;
#endif
#if defined(BCIR_SIMD_ARM)
    case BCIR_JER_SIMD_NEON: at = nonascii_run_neon(data, len); break;
#endif
    default: at = 0; break;
  }
  while (at < len && data[at] >= 0x80) at++;
  return at;
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
  size_t at = 0;
  if (data == nullptr || len == 0) return bcir_jer_validate_utf8(data, len, diag);

  /* ALTERNATE rather than hand off once. The first version sent everything from the first
   * non-ASCII octet to the END of the document to scalar, so one "café" near the front cost
   * a 29 KB document its entire acceleration. Alternating restores it.
   *
   * WHY SPLITTING HERE IS SOUND, and why it needs no second UTF-8 implementation: an ASCII
   * octet can never be a CONTINUATION octet, because continuations are 0x80-0xBF. So the
   * next ASCII octet is always a sequence boundary, and no legal multi-byte sequence can
   * span it. Validating [first non-ASCII, next ASCII) in isolation therefore yields exactly
   * the answer validating it in context would -- including for a truncated sequence, which
   * is invalid either way.
   *
   * Every verdict still comes from `bcir_jer_validate_utf8` itself. This function decides
   * only WHERE to look, never WHAT is valid. */
  while (at < len) {
    size_t run_end;
    bcir_jer_status status;

    at += ascii_run(tier, data + at, len - at);
    if (at == len) break;

    run_end = at + nonascii_run(tier, data + at, len - at);
    status = bcir_jer_validate_utf8(data + at, run_end - at, diag);
    if (status != BCIR_JER_OK) {
      /* The scalar rail's offset is relative to the run it was handed. */
      if (diag != nullptr) diag->offset += at;
      return status;
    }
    at = run_end;
  }
  /* The success `diag` is written by the scalar rail too, so a caller cannot tell which
   * path produced it -- a success whose diagnostic came from different code on different
   * inputs is exactly the drift this file exists to avoid. */
  return bcir_jer_validate_utf8(data + len, 0, diag);
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
