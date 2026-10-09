/*===- bcir_plan_sign.h - PlanStatementV1 + TrustStoreV1 verification (G21) ===
 *
 * The C twin of bcir/abi/plan_sign_abi.py: a detached statement that binds an ExecutionPlan's
 * bytes by SHA-256 -- with the scope, module, pack and certificate it was planned under -- and
 * an Ed25519 signature over it (bcir_ed25519.h), checked against a trust store that says which
 * keys may sign, when, and which are revoked. docs/kernel/BCIR_PLAN_SIGNATURE_ABI.md is the
 * prose spec. Freestanding: no heap, no libc.
 *
 *   PlanStatementV1 (272 bytes, little-endian)
 *     magic "BPSG" | version:u16=1 | flags:u16=0 | signed_at:u64
 *     key_id[32] plan[32] scope[32] module[32] pack[32] certificate[32]   (SHA-256 each)
 *     signature[64]                                       (Ed25519 over bytes 0..207)
 *   TrustStoreV1 (8 + 56n bytes)
 *     magic "BTRS" | version:u16=1 | count:u16
 *     count x { public_key[32] not_before:u64 not_after:u64 flags:u32 (bit 0 revoked) reserved:u32 }
 *     public keys strictly ascending
 *
 * The verdict names the first law broken, in the oracle's order: the store, the statement,
 * the plan digest, the pack digest (only when the statement binds one and a pack is given),
 * an unknown key, a revoked key, a date after `now`, a date outside the key's window, the
 * signature.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_PLAN_SIGN_H
#define BCIR_PLAN_SIGN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_PS_STATEMENT_SIZE 272u
#define BCIR_PS_SIGNED_SIZE 208u
#define BCIR_PS_STORE_ENTRY_SIZE 56u
#define BCIR_PS_KEY_REVOKED 1u

typedef enum bcir_ps_verdict {
  BCIR_PS_OK = 0,
  BCIR_PS_STORE = 1,       /* the trust store breaks a wire law */
  BCIR_PS_STATEMENT = 2,   /* the statement breaks a wire law */
  BCIR_PS_PLAN = 3,        /* the statement does not bind these plan bytes */
  BCIR_PS_PACK = 4,        /* the statement binds another pack */
  BCIR_PS_UNKNOWN_KEY = 5, /* the store does not hold the signing key */
  BCIR_PS_REVOKED = 6,     /* the signing key is revoked */
  BCIR_PS_FUTURE = 7,      /* the statement is dated after `now` */
  BCIR_PS_WINDOW = 8,      /* the statement is dated outside the key's window */
  BCIR_PS_SIGNATURE = 9    /* the Ed25519 signature does not verify */
} bcir_ps_verdict;

/* The oracle's `PlanSignError.code` for a verdict ("ok" for BCIR_PS_OK). */
const char *bcir_ps_verdict_name(bcir_ps_verdict verdict);

/* The store's key count, or BCIR_PS_STORE when it breaks a wire law. */
bcir_ps_verdict bcir_ps_store_validate(const uint8_t *store, size_t length, uint16_t *count);

/* Verify `statement` against `store` over exactly `plan` (and `pack`, which may be NULL with
 * length 0 when the caller holds none), at time `now` (seconds since the epoch). */
bcir_ps_verdict bcir_ps_verify(const uint8_t *statement, size_t statement_length,
                               const uint8_t *store, size_t store_length, const uint8_t *plan,
                               size_t plan_length, const uint8_t *pack, size_t pack_length,
                               uint64_t now);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_PLAN_SIGN_H */
