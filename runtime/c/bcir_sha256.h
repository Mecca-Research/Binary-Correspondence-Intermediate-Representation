/*===- bcir_sha256.h - freestanding SHA-256 and HMAC-SHA256 ---------------===
 *
 * The one SHA-256 in runtime/c. It began file-local to the BCAB reader
 * (bcir_artifact_bundle.c), the first artifact boundary that needed more than a CRC; the
 * control plane (bcir_control_plane.c, GEM+ G14) needs the same digest and a keyed MAC over
 * it, so the transform moved here verbatim rather than being spelled twice.
 *
 * Freestanding: <stddef.h> + <stdint.h> only, no heap, no libc. SHA-256 is FIPS 180-4;
 * HMAC-SHA256 is RFC 2104 / FIPS 198-1 (a key longer than the 64-byte block is hashed first).
 * The Python rails use hashlib.sha256 / hmac.new(..., hashlib.sha256); the RFC 4231 vectors
 * pin the two against each other (tools/c/check_runtime.sh).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_SHA256_H
#define BCIR_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_SHA256_DIGEST_SIZE 32u
#define BCIR_SHA256_BLOCK_SIZE 64u

/* Incremental state; zeroed by bcir_sha256_final so no message bytes linger. */
typedef struct bcir_sha256 {
  uint32_t h[8];
  uint64_t bytes;
  uint8_t block[64];
  size_t used;
} bcir_sha256;

void bcir_sha256_init(bcir_sha256 *state);
/* NULL is valid only with length 0. */
void bcir_sha256_update(bcir_sha256 *state, const uint8_t *data, size_t length);
void bcir_sha256_final(bcir_sha256 *state, uint8_t output[32]);
void bcir_sha256_digest(const uint8_t *data, size_t length, uint8_t output[32]);

/* HMAC-SHA256, incremental: init with the key, update with the message parts, final. */
typedef struct bcir_hmac_sha256_ctx {
  bcir_sha256 inner;
  bcir_sha256 outer;
} bcir_hmac_sha256_ctx;

void bcir_hmac_sha256_init(bcir_hmac_sha256_ctx *ctx, const uint8_t *key, size_t key_length);
void bcir_hmac_sha256_update(bcir_hmac_sha256_ctx *ctx, const uint8_t *data, size_t length);
void bcir_hmac_sha256_final(bcir_hmac_sha256_ctx *ctx, uint8_t output[32]);
void bcir_hmac_sha256(const uint8_t *key, size_t key_length, const uint8_t *message,
                      size_t message_length, uint8_t output[32]);

/* Constant-time equality of two n-byte strings: 1 equal, 0 not. A MAC comparison that leaks
 * its answer through timing is a comparison an attacker completes one octet at a time. */
int bcir_sha256_equal(const uint8_t *a, const uint8_t *b, size_t n);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_SHA256_H */
