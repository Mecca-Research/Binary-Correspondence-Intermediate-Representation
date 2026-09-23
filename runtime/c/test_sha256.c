/*===- test_sha256.c - the shared SHA-256 / HMAC-SHA256 against the standards ---===
 *
 * The vectors are the standards' own worked examples -- FIPS 180-4 (SHA-256, including the
 * one-million-'a' message that crosses many blocks) and RFC 4231 test cases 1-7 (HMAC-SHA256,
 * including case 5's truncated tag and cases 6-7's key longer than the block, which takes the
 * hash-the-key-first path) -- not a round trip through our own code. A round trip passes when
 * the encoder and the decoder share one misreading; a published vector does not.
 * Prints "OK" and exits 0 when every vector holds.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>

#include "bcir_sha256.h"

static int from_hex(const char *hex, uint8_t *out, size_t cap) {
  size_t n = strlen(hex) / 2u;
  if (n > cap) return -1;
  for (size_t i = 0; i < n; ++i) {
    unsigned v;
    if (sscanf(hex + 2u * i, "%2x", &v) != 1) return -1;
    out[i] = (uint8_t)v;
  }
  return (int)n;
}

static int failures = 0;

static void expect(const char *name, const uint8_t *got, const char *want_hex) {
  uint8_t want[32];
  int n = from_hex(want_hex, want, sizeof(want));
  if (n <= 0 || memcmp(got, want, (size_t)n) != 0) {
    printf("FAIL %s\n", name);
    ++failures;
  }
}

static void sha_vector(const char *message, const char *want) {
  uint8_t out[32];
  bcir_sha256_digest((const uint8_t *)message, strlen(message), out);
  expect(message[0] ? message : "(empty)", out, want);
}

static void hmac_vector(const char *name, const uint8_t *key, size_t key_len,
                        const uint8_t *msg, size_t msg_len, const char *want) {
  uint8_t out[32];
  bcir_hmac_sha256(key, key_len, msg, msg_len, out);
  expect(name, out, want);
  /* the incremental interface, one byte at a time, must agree with the one-shot */
  {
    bcir_hmac_sha256_ctx ctx;
    uint8_t again[32];
    bcir_hmac_sha256_init(&ctx, key, key_len);
    for (size_t i = 0; i < msg_len; ++i) bcir_hmac_sha256_update(&ctx, msg + i, 1u);
    bcir_hmac_sha256_final(&ctx, again);
    if (!bcir_sha256_equal(out, again, 32u)) { printf("FAIL %s incremental\n", name); ++failures; }
  }
}

int main(void) {
  /* FIPS 180-4 */
  sha_vector("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  sha_vector("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  sha_vector("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
             "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  sha_vector("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopq"
             "klmnopqrlmnopqrsmnopqrstnopqrstu",
             "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
  {
    static uint8_t chunk[1000];
    bcir_sha256 s;
    uint8_t out[32];
    memset(chunk, 'a', sizeof(chunk));
    bcir_sha256_init(&s);
    for (int i = 0; i < 1000; ++i) bcir_sha256_update(&s, chunk, sizeof(chunk));
    bcir_sha256_final(&s, out);
    expect("million-a", out, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  }
  /* RFC 4231 */
  {
    uint8_t k[131], d[64];
    memset(k, 0x0b, 20);
    hmac_vector("rfc4231-1", k, 20, (const uint8_t *)"Hi There", 8,
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    hmac_vector("rfc4231-2", (const uint8_t *)"Jefe", 4,
                (const uint8_t *)"what do ya want for nothing?", 28,
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    memset(k, 0xaa, 20); memset(d, 0xdd, 50);
    hmac_vector("rfc4231-3", k, 20, d, 50,
                "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
    for (int i = 0; i < 25; ++i) k[i] = (uint8_t)(i + 1);
    memset(d, 0xcd, 50);
    hmac_vector("rfc4231-4", k, 25, d, 50,
                "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
    memset(k, 0x0c, 20);
    hmac_vector("rfc4231-5", k, 20, (const uint8_t *)"Test With Truncation", 20,
                "a3b6167473100ee06e0c796c2955552b");
    memset(k, 0xaa, 131);
    {
      const char *m6 = "Test Using Larger Than Block-Size Key - Hash Key First";
      const char *m7 = "This is a test using a larger than block-size key and a larger than "
                       "block-size data. The key needs to be hashed before being used by the "
                       "HMAC algorithm.";
      hmac_vector("rfc4231-6", k, 131, (const uint8_t *)m6, strlen(m6),
                  "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
      hmac_vector("rfc4231-7", k, 131, (const uint8_t *)m7, strlen(m7),
                  "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");
    }
  }
  /* the constant-time comparison is a comparison */
  {
    const uint8_t a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 3, 5};
    if (!bcir_sha256_equal(a, a, 4u) || bcir_sha256_equal(a, b, 4u)) {
      printf("FAIL equal\n"); ++failures;
    }
  }
  if (failures) return 1;
  printf("OK\n");
  return 0;
}
