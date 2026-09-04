/*===- bcir_jer_simd.h - J5: the hosted SIMD rail behind the scalar C ABI --===
 *
 * 4.1: "The scalar rail is authoritative for native parser correctness. SIMD is an
 * optimization candidate, not a separate semantic implementation." Everything below is
 * built to make that sentence structurally true rather than merely intended.
 *
 * WHAT IS ACCELERATED, AND WHAT IS NOT. X.697 7.6.2's UTF-8 validation over the whole
 * document is a data-parallel accept-check, and it is the stage this rail speeds up. The
 * ACCEPT path is vectorized; the REJECT path is not. When a wide block contains anything a
 * vector cannot settle, the block is handed to `bcir_jer_validate_utf8` -- the scalar
 * function itself, not a reimplementation of it -- which produces the status and the byte
 * offset. So a diagnostic can only ever come from the authoritative rail, and the two
 * cannot drift because there is only one of them.
 *
 * That is a deliberate limitation and not a hidden one: this rail accelerates
 * ASCII-dominant documents, which JER text overwhelmingly is, and defers every multi-byte
 * sequence to scalar. A fully vectorized UTF-8 DFA would accelerate the rest too, and would
 * be a second definition of "valid UTF-8" -- exactly the second semantics rail 8's risk
 * table names.
 *
 * NO UNSUPPORTED-CPU FAULT. The tier is resolved once, from a runtime feature check, and a
 * tier the CPU does not advertise is never entered. `bcir_jer_validate_utf8_at` lets a
 * caller pin a tier -- used by the differential tests to run every tier the build has on one
 * machine, and available to a caller who must not execute a wide path at all.
 *
 * Hosted, C++17, and behind `extern "C"`: it is an adapter, so it must be callable from the
 * same places the scalar rail is. It links nothing from the freestanding core except the
 * scalar validator it defers to.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_JER_SIMD_H
#define BCIR_JER_SIMD_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_jer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The widths this build can dispatch to, in increasing order. The order is load-bearing:
 * `bcir_jer_simd_tier` returns the highest tier this CPU supports, and a test walks every
 * tier at or below it. */
typedef enum bcir_jer_simd_tier {
  BCIR_JER_SIMD_SCALAR = 0,
  BCIR_JER_SIMD_SSE2 = 1, /* x86-64 baseline, 16 octets */
  BCIR_JER_SIMD_AVX2 = 2, /* runtime-detected, 32 octets */
  BCIR_JER_SIMD_NEON = 3  /* aarch64 baseline, 16 octets */
} bcir_jer_simd_tier;

/* The highest tier this CPU supports AND this build compiled. Resolved once and cached;
 * the answer cannot change under a running process. */
bcir_jer_simd_tier bcir_jer_simd_tier_available(void);

/* A stable name, for a report that has to say which tier produced a number. */
const char *bcir_jer_simd_tier_name(bcir_jer_simd_tier tier);

/* Whether this build contains code for `tier` at all, independent of the CPU. Lets a test
 * tell "this machine cannot run AVX2" from "this build has no AVX2 path". */
int bcir_jer_simd_tier_compiled(bcir_jer_simd_tier tier);

/* X.697 7.6.2 over the whole document, dispatched to the best available tier.
 *
 * Contract: IDENTICAL status and IDENTICAL `diag` to `bcir_jer_validate_utf8` for every
 * input. Not "equivalent" -- identical, because on anything a vector cannot settle this
 * calls that function. */
bcir_jer_status bcir_jer_validate_utf8_simd(const uint8_t *data, size_t len, bcir_jer_diag *diag);

/* The same, pinned to a tier. A tier this build did not compile falls back to scalar rather
 * than failing: a caller asking for a width that is not there wants the answer, not an
 * error about the machine. */
bcir_jer_status bcir_jer_validate_utf8_at(bcir_jer_simd_tier tier, const uint8_t *data, size_t len,
                                          bcir_jer_diag *diag);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_JER_SIMD_H */
