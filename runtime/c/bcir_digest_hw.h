/*===- bcir_digest_hw.h - SHA-256 and CRC-32 on the CPU's own instructions, beside the portable code ---===
 *
 * The C subset audit's hardware digests (docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md). The same SHA-256
 * (FIPS 180-4) as bcir_sha256.c and the same CRC-32 (zlib's) as bcir_crc32.h -- which stay the reference, and
 * which every existing call site keeps -- computed on
 *
 *   SHA-256  the x86 SHA extensions (SHA-NI, with SSSE3 and SSE4.1), or the ARMv8 SHA-2 instructions;
 *   CRC-32   carry-less-multiply folding (PCLMULQDQ: V. Gopal, E. Ozturk, J. Guilford et al., "Fast CRC
 *            Computation for Generic Polynomials Using PCLMULQDQ Instruction", Intel, 2009), sixty-four bytes a
 *            step down to sixteen, which the slicing-by-8 table then reduces; or the ARMv8 CRC32 instructions.
 *
 * Selection is the caller's. bcir_digest_probe() says which paths this build carries and this CPU runs, and is
 * asked once; every other call takes that mask, or any subset of it (0 is the portable code). So no call probes,
 * nothing here is global, and a test forces each path in turn. A mask bit this build does not carry is ignored. A
 * bit the CPU lacks is the caller's error -- an illegal instruction -- which is why the mask comes from the probe.
 *
 * The fold's four constants are x^544, x^480, x^160 and x^96 modulo the CRC's polynomial, bit-reflected:
 * bcir/tests/test_c_digest_hw.py derives them from that definition and reads them out of the source.
 *
 * Freestanding: <stddef.h>, <stdint.h> and the compiler's own intrinsic and CPUID headers; no heap, no libc. The
 * one exception is the AArch64 probe, which asks Linux's auxiliary vector (getauxval) in a hosted build, takes the
 * compiler's baseline (__ARM_FEATURE_SHA2, __ARM_FEATURE_CRC32) otherwise, and reports nothing beyond it in a
 * freestanding build. The paths are built by GCC or Clang on x86-64 or AArch64; anywhere else the probe is 0 and
 * every mask is the portable code. The vector code is outside the verified C subset (the C subset audit's
 * fallback boundary), which is why it lives in this unit and not in bcir_sha256.c.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DIGEST_HW_H
#define BCIR_DIGEST_HW_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_DIGEST_SHA256_SHANI 0x1u /* x86-64: SHA-NI, SSSE3, SSE4.1 */
#define BCIR_DIGEST_SHA256_ARMV8 0x2u /* AArch64: the SHA-2 instructions */
#define BCIR_DIGEST_CRC32_CLMUL 0x4u  /* x86-64: PCLMULQDQ */
#define BCIR_DIGEST_CRC32_ARMV8 0x8u  /* AArch64: the CRC32 instructions */

/* The paths this build carries code for. */
unsigned bcir_digest_compiled(void);
/* The paths this build carries and this CPU runs: the mask every call below may take. Asks the CPU (CPUID) or
 * the OS (getauxval) each time; ask once. */
unsigned bcir_digest_probe(void);

/* bcir_sha256_update, its full blocks compressed on the SHA path `features` allows (the portable transform
 * otherwise); the state is bcir_sha256's, so bcir_sha256_final ends it. */
void bcir_sha256_update_hw(bcir_sha256 *state, const uint8_t *data, size_t length, unsigned features);
void bcir_sha256_digest_hw(const uint8_t *data, size_t length, uint8_t output[32], unsigned features);

/* bcir_crc32_step (a running state, no conditioning) on the CRC path `features` allows; and zlib's CRC-32. */
uint32_t bcir_crc32_step_hw(uint32_t state, const uint8_t *data, size_t length, unsigned features);
uint32_t bcir_crc32_hw(const uint8_t *data, size_t length, unsigned features);

#ifdef BCIR_DIGEST_HW_WITNESS
/* Test builds only: how many times each path's kernel ran, indexed by the bit's position. The parity test reads
 * it to know a forced path did the work, since its output cannot say. */
extern unsigned long bcir_digest_hw_witness[4];
#endif

#ifdef __cplusplus
}
#endif

#endif
