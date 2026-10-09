/*===- fuzz_plan_sign.c - libFuzzer: Ed25519 + PlanStatementV1 / TrustStoreV1 (G21) ===
 *
 * Two trust boundaries in one harness:
 *   1. the raw decoders over arbitrary bytes -- bcir_ed25519_verify, bcir_ed25519_usable_key
 *      and bcir_ps_verify with the input split into a statement, a store and a plan -- total,
 *      never an out-of-bounds read, every verdict one the enum names;
 *   2. unforgeability: a genuine statement, signed at start-up by a fixed key the store trusts,
 *      is mutated by the input (XOR at input-chosen offsets, signature region included). The
 *      verifier may accept exactly the bytes that were signed: if a mutated statement verifies,
 *      it must be byte-identical to the genuine one, or the harness aborts (a finding).
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_ed25519.h"
#include "bcir_plan_sign.h"
#include "bcir_sha256.h"

static const uint8_t SECRET[32] = {0x42, 0x43, 0x49, 0x52, 0x20, 0x70, 0x6c, 0x61, 0x6e, 0x20, 0x73,
                                   0x69, 0x67, 0x6e, 0x20, 0x66, 0x75, 0x7a, 0x7a, 0x20, 0x6b, 0x65,
                                   0x79, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
static const uint8_t PLAN[] = "BPLN: the plan bytes the genuine statement binds";
static uint8_t genuine[BCIR_PS_STATEMENT_SIZE], store[8 + BCIR_PS_STORE_ENTRY_SIZE];
static int ready;

static void put64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

static void setup(void) {
  uint8_t pk[32];
  bcir_ed25519_public_key(SECRET, pk);
  memset(store, 0, sizeof store);
  memcpy(store, "BTRS", 4);
  store[4] = 1;
  store[6] = 1;
  memcpy(store + 8, pk, 32);
  put64(store + 8 + 32, 1000);
  put64(store + 8 + 40, 3000);
  memset(genuine, 0, sizeof genuine);
  memcpy(genuine, "BPSG", 4);
  genuine[4] = 1;
  put64(genuine + 8, 1500);
  bcir_sha256_digest(pk, 32, genuine + 16);
  bcir_sha256_digest(PLAN, sizeof PLAN, genuine + 48);
  memset(genuine + 80, 0x5c, 32); /* a scope digest */
  bcir_ed25519_sign(SECRET, genuine, BCIR_PS_SIGNED_SIZE, genuine + BCIR_PS_SIGNED_SIZE);
  if (bcir_ps_verify(genuine, sizeof genuine, store, sizeof store, PLAN, sizeof PLAN, NULL, 0,
                     2000) != BCIR_PS_OK)
    abort(); /* the harness's own statement must verify */
  ready = 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t mutated[BCIR_PS_STATEMENT_SIZE];
  bcir_ps_verdict v;
  size_t i;
  if (!ready) setup();
  /* 1. raw bytes */
  if (size >= 96) (void)bcir_ed25519_verify(data, data + 96, size - 96, data + 32);
  if (size >= 32) (void)bcir_ed25519_usable_key(data);
  {
    size_t st = size < BCIR_PS_STATEMENT_SIZE ? size : BCIR_PS_STATEMENT_SIZE;
    size_t rest = size - st, st_len = rest / 2;
    v = bcir_ps_verify(data, st, data + st, st_len, data + st + st_len, rest - st_len, NULL, 0,
                       size);
    if ((unsigned)v > (unsigned)BCIR_PS_SIGNATURE) abort();
  }
  /* 2. unforgeability: (offset, mask) pairs from the input */
  memcpy(mutated, genuine, sizeof mutated);
  for (i = 0; i + 1 < size; i += 2) mutated[data[i] % BCIR_PS_STATEMENT_SIZE] ^= data[i + 1];
  if (size >= 2 && (data[1] & 0x80u)) mutated[BCIR_PS_SIGNED_SIZE + (data[0] % 64u)] ^= 0x01u;
  v = bcir_ps_verify(mutated, sizeof mutated, store, sizeof store, PLAN, sizeof PLAN, NULL, 0,
                     2000);
  if (v == BCIR_PS_OK && memcmp(mutated, genuine, sizeof mutated) != 0) abort(); /* a forgery */
  return 0;
}
