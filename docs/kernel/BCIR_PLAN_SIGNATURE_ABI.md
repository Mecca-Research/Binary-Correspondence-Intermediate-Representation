# BCIR plan signatures — PlanStatementV1 + TrustStoreV1 (normative, G21)

An [ExecutionPlan](BCIR_EXECUTION_PLAN_ABI.md) carries a CRC-32 and names its module and target
by 63-bit FNV-1a R13 hashes; the control plane's authority is a shared-key HMAC, which
[`BCIR_CONTROL_PLANE_ABI.md`](BCIR_CONTROL_PLANE_ABI.md) itself calls "not a signature scheme".
This is the signature scheme (GEM+ G21; the 2026-10-06 audit, item 6): a **detached statement**
that binds a plan's exact bytes by SHA-256 — together with the scope, the module, the pack and
the certificate it was planned under — and an **Ed25519** signature over it (RFC 8032 section
5.1), checked against a **trust store** that says which keys may sign, when, and which are
revoked.

The reference rail is [`bcir/abi/plan_sign_abi.py`](../../bcir/abi/plan_sign_abi.py) with
[`bcir/abi/ed25519.py`](../../bcir/abi/ed25519.py); the C view is
[`runtime/c/bcir_plan_sign.h`](../../runtime/c/bcir_plan_sign.h) with the freestanding
[`bcir_ed25519.h`](../../runtime/c/bcir_ed25519.h) (SHA-512 and Ed25519, no heap, no libc).

## Why detached, and not an ExecutionPlan v4 record

The plan's versions are cumulative: a v4 plan would be a v3 plan, and v3's laws require a
nonzero source/spec binding that a plan which moves nothing cannot carry. A statement that binds
the plan's bytes signs every plan version as it is, re-signs nothing when a plan is re-encoded
identically, and leaves the frozen plan ABI where it is.

## PlanStatementV1 (272 bytes, little-endian)

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | `u8[4]` | `"BPSG"` |
| 4 | `version` | `u16` | 1 |
| 6 | `flags` | `u16` | reserved (0) |
| 8 | `signed_at` | `u64` | seconds since the epoch, as the signer states it |
| 16 | `key_id` | `u8[32]` | SHA-256 of the signer's Ed25519 public key |
| 48 | `plan` | `u8[32]` | SHA-256 of the ExecutionPlan bytes (never zero) |
| 80 | `scope` | `u8[32]` | SHA-256 of the `ExecutionScopeV1` (`ExecutionScope.digest`), or zero: not bound |
| 112 | `module` | `u8[32]` | SHA-256 of the scope's program component `P`, or zero |
| 144 | `pack` | `u8[32]` | SHA-256 of the StreamPack bytes, or zero |
| 176 | `certificate` | `u8[32]` | SHA-256 of the certificate (its `digest`, or its canonical `to_dict()` JSON), or zero |
| 208 | `signature` | `u8[64]` | Ed25519 over bytes 0..207 |

## TrustStoreV1 (8 + 56n bytes, little-endian)

`magic "BTRS" | version:u16 = 1 | count:u16`, then `count` entries of
`public_key[32] | not_before:u64 | not_after:u64 | flags:u32 (bit 0: revoked) | reserved:u32 = 0`.
The public keys are **strictly ascending** (a store has one spelling and holds a key once);
every key's window `[not_before, not_after)` is non-empty; and every key is the canonical
encoding of a curve point **not of small order**.

The last law is the one RFC 8032 does not state. With the identity as the public key, the
group equation `[S]B = R + [k]A` loses its `A` term, and `R = [S]B` "verifies" every message
for every `S`: anyone can sign. Every other small-order key does the same after a cofactor's
worth of doubling. So the store refuses them (`ed25519.usable_key`,
`bcir_ed25519_usable_key`), and `test_plan_sign.py` keeps the forgery as its witness.

## The verdict

Both rails return the first law a statement breaks, in this order, with one name for each:

| Verdict | When |
|---|---|
| `store` | the trust store breaks a law above. It is checked first, on both rails, whoever built the store: the reference rail holds an in-memory `TrustStore` to the laws `decode_store` holds bytes to, since a store that trusts a small-order key accepts a forgery of every statement that names it |
| `statement` | the statement breaks a wire law: its length, magic, version or flags, or a zero plan digest |
| `plan` | the statement does not bind these plan bytes |
| `pack` | the statement binds a pack, the verifier holds one, and it is another |
| `unknown-key` | the store does not hold the key the statement names |
| `revoked` | the signing key is revoked, whatever date the statement claims (a compromised key can write any date) |
| `future` | the statement is dated after the time it is verified at |
| `window` | the statement is dated outside the key's window |
| `signature` | the Ed25519 signature does not verify |
| `ok` | none of the above |

**Rotation** is two keys whose windows meet. The store accepts each one inside its own window.
**Expiry** is the window's end. **Revocation** is the flag.

Verification is RFC 8032 section 5.1.7, made strict and total over untrusted bytes. An `S` that
is not below the group order is refused rather than reduced, which closes the malleable second
spelling of a valid signature. A public key or `R` that is not the canonical encoding of a curve
point is refused. The group equation is checked cofactorless, the same way on both rails.

## Evidence

- Both rails reproduce RFC 8032 section 7.1's vectors (TEST 1, 2, 3 and SHA(abc)).
- They sign byte-identically over seeded keys and messages.
- They return the same verdict on every case of
  [`bcir/tests/plan_sign_fixtures.py`](../../bcir/tests/plan_sign_fixtures.py): the two accepted
  statements, and every forgery of trust, of binding, of the signed fields, and of the
  statement's and the store's wire.
- These are checked by `test_plan_sign.py`, by `test_c_plan_sign.py`, and by the `plan_sign`
  gate section, which BCIR Make and CMake both run.
- The decoder campaign (`tools/security/run_decoder_campaign.py`) mutates a genuine statement
  and store on the reference rail (the `statement` and `store` surfaces): each mutation is
  decoded or refused with `PlanSignError`, never another exception.
- `fuzz_plan_sign` fuzzes the C twin under ASan and UBSan, over raw bytes. It also asserts
  unforgeability: a genuine statement, mutated by the input, verifies only as the bytes that
  were signed.

The Python rail is a reference, not a hardened signer: Python integers are not constant-time. A
key whose timing must not leak is held by the C twin (a constant-time ladder) or by hardware.
