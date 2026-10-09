/*===- bcir_ed25519.h - freestanding SHA-512 and Ed25519 (RFC 8032) --------===
 *
 * The C twin of bcir/abi/ed25519.py (GEM+ G21, plan signatures). Freestanding: <stddef.h> +
 * <stdint.h> only, no heap, no libc. SHA-512 is FIPS 180-4; Ed25519 is RFC 8032 section 5.1,
 * the field and group arithmetic in TweetNaCl's sixteen 16-bit limbs (public domain), with
 * every shift of a possibly negative limb written as a multiplication so the code is defined
 * C, and with the scalar multiplication a constant-time ladder.
 *
 * Verification is the strict RFC 8032 section 5.1.7 check the oracle applies, total over
 * untrusted bytes: a public key that is not a canonical encoding of a curve point (y >= p, or
 * x = 0 spelled with the sign bit), an S that is not below the group order L, or an R that is
 * not the canonical encoding of [S]B - [k]A is refused; the group equation is cofactorless.
 * Both rails are held to RFC 8032 section 7.1's vectors and to each other.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_ED25519_H
#define BCIR_ED25519_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_SHA512_DIGEST_SIZE 64u
#define BCIR_ED25519_PUBLIC_KEY_SIZE 32u
#define BCIR_ED25519_SECRET_KEY_SIZE 32u
#define BCIR_ED25519_SIGNATURE_SIZE 64u

/* Incremental SHA-512 state; zeroed by bcir_sha512_final so no message bytes linger. */
typedef struct bcir_sha512 {
  uint64_t h[8];
  uint64_t bytes;
  uint8_t block[128];
  size_t used;
} bcir_sha512;

void bcir_sha512_init(bcir_sha512 *state);
/* NULL is valid only with length 0. */
void bcir_sha512_update(bcir_sha512 *state, const uint8_t *data, size_t length);
void bcir_sha512_final(bcir_sha512 *state, uint8_t output[64]);
void bcir_sha512_digest(const uint8_t *data, size_t length, uint8_t output[64]);

/* The public key of a 32-byte secret key (the RFC 8032 seed). */
void bcir_ed25519_public_key(const uint8_t secret[32], uint8_t public_key[32]);
/* The deterministic signature of `message` (NULL only with length 0). */
void bcir_ed25519_sign(const uint8_t secret[32], const uint8_t *message, size_t length,
                       uint8_t signature[64]);
/* 1 when `signature` is `public_key`'s signature of `message`, 0 otherwise (every malformed
 * input included -- never a crash, never a partial verdict). */
int bcir_ed25519_verify(const uint8_t public_key[32], const uint8_t *message, size_t length,
                        const uint8_t signature[64]);

/* 1 when `public_key` may be trusted to sign: the canonical encoding of a curve point not of
 * small order (for a small-order key, the identity among them, anyone can forge). */
int bcir_ed25519_usable_key(const uint8_t public_key[32]);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_ED25519_H */
