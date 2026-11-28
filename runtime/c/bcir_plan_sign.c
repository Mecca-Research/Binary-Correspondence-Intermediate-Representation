/*===- bcir_plan_sign.c - PlanStatementV1 + TrustStoreV1 verification (G21) ===
 *
 * See bcir_plan_sign.h. Every read is bounded by the lengths the caller passes; nothing is
 * allocated; every path out is a verdict.
 *===----------------------------------------------------------------------===*/
#include "bcir_plan_sign.h"

#include "bcir_ed25519.h"
#include "bcir_sha256.h"
#include "bcir_bytes.h"

static int same32(const uint8_t *a, const uint8_t *b) {
  unsigned i, d = 0;
  for (i = 0; i < 32u; i++) d |= (unsigned)(a[i] ^ b[i]);
  return d == 0;
}

static int zero32(const uint8_t *a) {
  unsigned i, d = 0;
  for (i = 0; i < 32u; i++) d |= a[i];
  return d == 0;
}

/* -1, 0 or 1 as the 32-byte strings compare (bytewise, the order the oracle's bytes use). */
static int compare32(const uint8_t *a, const uint8_t *b) {
  unsigned i;
  for (i = 0; i < 32u; i++)
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

const char *bcir_ps_verdict_name(bcir_ps_verdict verdict) {
  switch (verdict) {
    case BCIR_PS_OK: return "ok";
    case BCIR_PS_STORE: return "store";
    case BCIR_PS_STATEMENT: return "statement";
    case BCIR_PS_PLAN: return "plan";
    case BCIR_PS_PACK: return "pack";
    case BCIR_PS_UNKNOWN_KEY: return "unknown-key";
    case BCIR_PS_REVOKED: return "revoked";
    case BCIR_PS_FUTURE: return "future";
    case BCIR_PS_WINDOW: return "window";
    case BCIR_PS_SIGNATURE: return "signature";
  }
  return "statement";
}

bcir_ps_verdict bcir_ps_store_validate(const uint8_t *store, size_t length, uint16_t *count) {
  uint16_t n;
  size_t i;
  if (count) *count = 0;
  if (!store || length < 8u) return BCIR_PS_STORE;
  if (store[0] != 'B' || store[1] != 'T' || store[2] != 'R' || store[3] != 'S') return BCIR_PS_STORE;
  if (bcir_ld16le(store + 4) != 1u) return BCIR_PS_STORE;
  n = bcir_ld16le(store + 6);
  if (length != 8u + (size_t)BCIR_PS_STORE_ENTRY_SIZE * n) return BCIR_PS_STORE;
  for (i = 0; i < n; i++) {
    const uint8_t *e = store + 8u + BCIR_PS_STORE_ENTRY_SIZE * i;
    if (!bcir_ed25519_usable_key(e)) return BCIR_PS_STORE;
    if (i > 0 && compare32(e - BCIR_PS_STORE_ENTRY_SIZE, e) >= 0) return BCIR_PS_STORE;
    if (bcir_ld64le(e + 40) <= bcir_ld64le(e + 32)) return BCIR_PS_STORE; /* an empty window */
    if ((bcir_ld32le(e + 48) & ~(uint32_t)BCIR_PS_KEY_REVOKED) != 0u || bcir_ld32le(e + 52) != 0u)
      return BCIR_PS_STORE;
  }
  if (count) *count = n;
  return BCIR_PS_OK;
}

bcir_ps_verdict bcir_ps_verify(const uint8_t *statement, size_t statement_length,
                               const uint8_t *store, size_t store_length, const uint8_t *plan,
                               size_t plan_length, const uint8_t *pack, size_t pack_length,
                               uint64_t now) {
  uint8_t digest[32], key_id[32];
  const uint8_t *key = 0;
  uint16_t n = 0;
  uint64_t signed_at;
  size_t i;
  if (bcir_ps_store_validate(store, store_length, &n) != BCIR_PS_OK) return BCIR_PS_STORE;
  if (!statement || statement_length != BCIR_PS_STATEMENT_SIZE) return BCIR_PS_STATEMENT;
  if (statement[0] != 'B' || statement[1] != 'P' || statement[2] != 'S' || statement[3] != 'G')
    return BCIR_PS_STATEMENT;
  if (bcir_ld16le(statement + 4) != 1u || bcir_ld16le(statement + 6) != 0u) return BCIR_PS_STATEMENT;
  if (zero32(statement + 48)) return BCIR_PS_STATEMENT; /* a statement binds a plan */
  if (!plan && plan_length) return BCIR_PS_PLAN;
  bcir_sha256_digest(plan, plan_length, digest);
  if (!same32(digest, statement + 48)) return BCIR_PS_PLAN;
  if ((pack || pack_length) && !zero32(statement + 144)) {
    if (!pack) return BCIR_PS_PACK;
    bcir_sha256_digest(pack, pack_length, digest);
    if (!same32(digest, statement + 144)) return BCIR_PS_PACK;
  }
  for (i = 0; i < n; i++) {
    const uint8_t *e = store + 8u + BCIR_PS_STORE_ENTRY_SIZE * i;
    bcir_sha256_digest(e, 32, key_id);
    if (same32(key_id, statement + 16)) {
      key = e;
      break;
    }
  }
  if (!key) return BCIR_PS_UNKNOWN_KEY;
  if (bcir_ld32le(key + 48) & BCIR_PS_KEY_REVOKED) return BCIR_PS_REVOKED;
  signed_at = bcir_ld64le(statement + 8);
  if (signed_at > now) return BCIR_PS_FUTURE;
  if (signed_at < bcir_ld64le(key + 32) || signed_at >= bcir_ld64le(key + 40)) return BCIR_PS_WINDOW;
  if (!bcir_ed25519_verify(key, statement, BCIR_PS_SIGNED_SIZE, statement + BCIR_PS_SIGNED_SIZE))
    return BCIR_PS_SIGNATURE;
  return BCIR_PS_OK;
}
