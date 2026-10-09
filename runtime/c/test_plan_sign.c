/*===- test_plan_sign.c - Ed25519 + PlanStatementV1 harness (G21) ---------------===
 *
 * Drives the freestanding Ed25519 (bcir_ed25519.c) and plan-statement verifier
 * (bcir_plan_sign.c) for tools/c/sections/plan_sign.sh and bcir/tests/test_c_plan_sign.py:
 *
 *   test_plan_sign vectors                       RFC 8032 section 7.1 + FIPS 180-4 "abc"
 *   test_plan_sign sign <secret-hex> <file>      prints the public key and the signature, hex
 *   test_plan_sign verify <statement> <store> <plan> <pack|-> <now>
 *                                                prints verdict=<name>; exit 0 exactly on "ok"
 *
 * The harness uses libc (it is a test); the units under test stay freestanding.
 *===----------------------------------------------------------------------===*/
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_ed25519.h"
#include "bcir_plan_sign.h"

static size_t from_hex(const char *h, uint8_t *out, size_t cap) {
  size_t n = strlen(h) / 2, i;
  if (n > cap) return (size_t)-1;
  for (i = 0; i < n; i++) {
    unsigned v;
    if (sscanf(h + 2 * i, "%2x", &v) != 1) return (size_t)-1;
    out[i] = (uint8_t)v;
  }
  return n;
}

static void put_hex(const uint8_t *p, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) printf("%02x", p[i]);
}

static uint8_t *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  uint8_t *buf;
  long size;
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }
  buf = (uint8_t *)malloc((size_t)size + 1u);
  if (buf && fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf);
    buf = NULL;
  }
  fclose(f);
  *n = (size_t)size;
  return buf;
}

/* RFC 8032 section 7.1: TEST 1, TEST 2, TEST 3 and TEST SHA(abc). */
static const char *const vectors[4][4] = {
    {"9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"},
    {"4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"},
    {"c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"},
    {"833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42",
     "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
     "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
     "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704"}};

static int run_vectors(void) {
  uint8_t sk[32], pk[32], msg[64], sig[64], out_pk[32], out_sig[64], digest[64];
  size_t n, i;
  int failed = 0;
  bcir_sha512_digest((const uint8_t *)"abc", 3, digest);
  from_hex(vectors[3][2], msg, sizeof msg);
  if (memcmp(digest, msg, 64) != 0) {
    printf("FAIL sha512(abc)\n");
    failed = 1;
  }
  for (i = 0; i < 4; i++) {
    from_hex(vectors[i][0], sk, 32);
    from_hex(vectors[i][1], pk, 32);
    n = from_hex(vectors[i][2], msg, sizeof msg);
    from_hex(vectors[i][3], sig, 64);
    bcir_ed25519_public_key(sk, out_pk);
    bcir_ed25519_sign(sk, msg, n, out_sig);
    if (memcmp(out_pk, pk, 32) != 0 || memcmp(out_sig, sig, 64) != 0 ||
        !bcir_ed25519_verify(pk, msg, n, sig)) {
      printf("FAIL vector %zu\n", i + 1);
      failed = 1;
    }
    sig[63] ^= 0x10; /* S + 2^252 * 16: no longer below L, or another scalar */
    if (bcir_ed25519_verify(pk, msg, n, sig)) {
      printf("FAIL vector %zu accepted a forged S\n", i + 1);
      failed = 1;
    }
  }
  if (!failed) printf("vectors=ok\n");
  return failed;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "vectors") == 0) return run_vectors();
  if (argc == 4 && strcmp(argv[1], "sign") == 0) {
    uint8_t sk[32], pk[32], sig[64];
    size_t n = 0;
    uint8_t *msg;
    if (from_hex(argv[2], sk, 32) != 32) return 2;
    msg = slurp(argv[3], &n);
    if (!msg) return 2;
    bcir_ed25519_public_key(sk, pk);
    bcir_ed25519_sign(sk, msg, n, sig);
    put_hex(pk, 32);
    printf(" ");
    put_hex(sig, 64);
    printf("\n");
    free(msg);
    return 0;
  }
  if (argc == 7 && strcmp(argv[1], "verify") == 0) {
    size_t ns = 0, nt = 0, np = 0, nk = 0;
    uint8_t *st = slurp(argv[2], &ns), *store = slurp(argv[3], &nt), *plan = slurp(argv[4], &np);
    uint8_t *pack = strcmp(argv[5], "-") == 0 ? NULL : slurp(argv[5], &nk);
    uint64_t now = strtoull(argv[6], NULL, 10);
    bcir_ps_verdict v;
    if (!st || !store || !plan || (strcmp(argv[5], "-") != 0 && !pack)) return 2;
    v = bcir_ps_verify(st, ns, store, nt, plan, np, pack, nk, now);
    printf("verdict=%s\n", bcir_ps_verdict_name(v));
    free(st);
    free(store);
    free(plan);
    free(pack);
    return v == BCIR_PS_OK ? 0 : 1;
  }
  fprintf(stderr, "usage: test_plan_sign vectors | sign <hex> <file> | verify <st> <store> <plan> <pack|-> <now>\n");
  return 2;
}
