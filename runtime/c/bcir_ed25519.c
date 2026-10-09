/*===- bcir_ed25519.c - freestanding SHA-512 and Ed25519 (RFC 8032) --------===
 *
 * See bcir_ed25519.h. The constants (SHA-512's round constants and initial value; the curve's
 * d, 2d, base point and sqrt(-1); the group order L) are derived, not typed: from the first
 * eighty primes and from p = 2^255 - 19, and the RFC 8032 vectors and the oracle's own
 * signatures pin the result (tools/c/sections/plan_sign.sh, bcir/tests/test_plan_sign.py).
 *===----------------------------------------------------------------------===*/
#include "bcir_ed25519.h"

/* --- SHA-512 (FIPS 180-4) ----------------------------------------------------------------- */

static const uint64_t sha512_k[80] = {UINT64_C(0x428a2f98d728ae22), UINT64_C(0x7137449123ef65cd), UINT64_C(0xb5c0fbcfec4d3b2f), UINT64_C(0xe9b5dba58189dbbc), UINT64_C(0x3956c25bf348b538), UINT64_C(0x59f111f1b605d019), UINT64_C(0x923f82a4af194f9b), UINT64_C(0xab1c5ed5da6d8118), UINT64_C(0xd807aa98a3030242), UINT64_C(0x12835b0145706fbe), UINT64_C(0x243185be4ee4b28c), UINT64_C(0x550c7dc3d5ffb4e2), UINT64_C(0x72be5d74f27b896f), UINT64_C(0x80deb1fe3b1696b1), UINT64_C(0x9bdc06a725c71235), UINT64_C(0xc19bf174cf692694), UINT64_C(0xe49b69c19ef14ad2), UINT64_C(0xefbe4786384f25e3), UINT64_C(0x0fc19dc68b8cd5b5), UINT64_C(0x240ca1cc77ac9c65), UINT64_C(0x2de92c6f592b0275), UINT64_C(0x4a7484aa6ea6e483), UINT64_C(0x5cb0a9dcbd41fbd4), UINT64_C(0x76f988da831153b5), UINT64_C(0x983e5152ee66dfab), UINT64_C(0xa831c66d2db43210), UINT64_C(0xb00327c898fb213f), UINT64_C(0xbf597fc7beef0ee4), UINT64_C(0xc6e00bf33da88fc2), UINT64_C(0xd5a79147930aa725), UINT64_C(0x06ca6351e003826f), UINT64_C(0x142929670a0e6e70), UINT64_C(0x27b70a8546d22ffc), UINT64_C(0x2e1b21385c26c926), UINT64_C(0x4d2c6dfc5ac42aed), UINT64_C(0x53380d139d95b3df), UINT64_C(0x650a73548baf63de), UINT64_C(0x766a0abb3c77b2a8), UINT64_C(0x81c2c92e47edaee6), UINT64_C(0x92722c851482353b), UINT64_C(0xa2bfe8a14cf10364), UINT64_C(0xa81a664bbc423001), UINT64_C(0xc24b8b70d0f89791), UINT64_C(0xc76c51a30654be30), UINT64_C(0xd192e819d6ef5218), UINT64_C(0xd69906245565a910), UINT64_C(0xf40e35855771202a), UINT64_C(0x106aa07032bbd1b8), UINT64_C(0x19a4c116b8d2d0c8), UINT64_C(0x1e376c085141ab53), UINT64_C(0x2748774cdf8eeb99), UINT64_C(0x34b0bcb5e19b48a8), UINT64_C(0x391c0cb3c5c95a63), UINT64_C(0x4ed8aa4ae3418acb), UINT64_C(0x5b9cca4f7763e373), UINT64_C(0x682e6ff3d6b2b8a3), UINT64_C(0x748f82ee5defb2fc), UINT64_C(0x78a5636f43172f60), UINT64_C(0x84c87814a1f0ab72), UINT64_C(0x8cc702081a6439ec), UINT64_C(0x90befffa23631e28), UINT64_C(0xa4506cebde82bde9), UINT64_C(0xbef9a3f7b2c67915), UINT64_C(0xc67178f2e372532b), UINT64_C(0xca273eceea26619c), UINT64_C(0xd186b8c721c0c207), UINT64_C(0xeada7dd6cde0eb1e), UINT64_C(0xf57d4f7fee6ed178), UINT64_C(0x06f067aa72176fba), UINT64_C(0x0a637dc5a2c898a6), UINT64_C(0x113f9804bef90dae), UINT64_C(0x1b710b35131c471b), UINT64_C(0x28db77f523047d84), UINT64_C(0x32caab7b40c72493), UINT64_C(0x3c9ebe0a15c9bebc), UINT64_C(0x431d67c49c100d4c), UINT64_C(0x4cc5d4becb3e42b6), UINT64_C(0x597f299cfc657e2a), UINT64_C(0x5fcb6fab3ad6faec), UINT64_C(0x6c44198c4a475817)};
static const uint64_t sha512_iv[8] = {UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b), UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1), UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f), UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)};

static uint64_t ror64(uint64_t x, unsigned n) { return (x >> n) | (x << (64u - n)); }

static uint64_t load64_be(const uint8_t *p) {
  uint64_t v = 0;
  unsigned i;
  for (i = 0; i < 8u; i++) v = (v << 8) | p[i];
  return v;
}

static void sha512_block(uint64_t h[8], const uint8_t block[128]) {
  uint64_t w[80], a, b, c, d, e, f, g, k, t1, t2;
  unsigned i;
  for (i = 0; i < 16u; i++) w[i] = load64_be(block + 8u * i);
  for (i = 16; i < 80u; i++) {
    uint64_t s0 = ror64(w[i - 15], 1) ^ ror64(w[i - 15], 8) ^ (w[i - 15] >> 7);
    uint64_t s1 = ror64(w[i - 2], 19) ^ ror64(w[i - 2], 61) ^ (w[i - 2] >> 6);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1; /* mod 2^64 by definition (FIPS 180-4) */
  }
  a = h[0]; b = h[1]; c = h[2]; d = h[3]; e = h[4]; f = h[5]; g = h[6]; k = h[7];
  for (i = 0; i < 80u; i++) {
    t1 = k + (ror64(e, 14) ^ ror64(e, 18) ^ ror64(e, 41)) + ((e & f) ^ (~e & g)) + sha512_k[i] +
         w[i];
    t2 = (ror64(a, 28) ^ ror64(a, 34) ^ ror64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

void bcir_sha512_init(bcir_sha512 *state) {
  unsigned i;
  for (i = 0; i < 8u; i++) state->h[i] = sha512_iv[i];
  for (i = 0; i < 128u; i++) state->block[i] = 0;
  state->bytes = 0;
  state->used = 0;
}

void bcir_sha512_update(bcir_sha512 *state, const uint8_t *data, size_t length) {
  size_t i;
  for (i = 0; i < length; i++) {
    state->block[state->used++] = data[i];
    if (state->used == 128u) {
      sha512_block(state->h, state->block);
      state->used = 0;
    }
  }
  state->bytes += (uint64_t)length;
}

void bcir_sha512_final(bcir_sha512 *state, uint8_t output[64]) {
  uint64_t bits_hi = state->bytes >> 61, bits_lo = state->bytes << 3;
  unsigned i;
  state->block[state->used++] = 0x80u;
  if (state->used > 112u) {
    while (state->used < 128u) state->block[state->used++] = 0;
    sha512_block(state->h, state->block);
    state->used = 0;
  }
  while (state->used < 112u) state->block[state->used++] = 0;
  for (i = 0; i < 8u; i++) {
    state->block[112u + i] = (uint8_t)(bits_hi >> (56u - 8u * i));
    state->block[120u + i] = (uint8_t)(bits_lo >> (56u - 8u * i));
  }
  sha512_block(state->h, state->block);
  for (i = 0; i < 64u; i++) output[i] = (uint8_t)(state->h[i / 8u] >> (56u - 8u * (i % 8u)));
  for (i = 0; i < 8u; i++) state->h[i] = 0;
  for (i = 0; i < 128u; i++) state->block[i] = 0;
  state->bytes = 0;
  state->used = 0;
}

void bcir_sha512_digest(const uint8_t *data, size_t length, uint8_t output[64]) {
  bcir_sha512 state;
  bcir_sha512_init(&state);
  bcir_sha512_update(&state, data, length);
  bcir_sha512_final(&state, output);
}

/* --- GF(2^255 - 19) in sixteen 16-bit limbs ----------------------------------------------- */

typedef int64_t gf[16];

static const gf gf0 = {0};
static const gf gf1 = {1};
static const gf ed_d = {0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070, 0xe898, 0x7779, 0x4079, 0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203};
static const gf ed_d2 = {0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0, 0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406};
static const gf ed_x = {0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c, 0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169};
static const gf ed_y = {0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666};
static const gf ed_i = {0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43, 0xd7a7, 0x3dfb, 0x0099, 0x2b4d, 0xdf0b, 0x4fc1, 0x2480, 0x2b83};

static void set25519(gf r, const gf a) {
  int i;
  for (i = 0; i < 16; i++) r[i] = a[i];
}

/* Carry each limb into the next; the top limb wraps into limb 0 times 38 (2^256 = 38 mod p).
 * `>>` of a negative limb is an arithmetic shift on every target this runtime builds for; the
 * carry is put back with a multiplication, which is defined where `c << 16` would not be. */
static void car25519(gf o) {
  int i;
  int64_t c;
  for (i = 0; i < 16; i++) {
    o[i] += 65536;
    c = o[i] >> 16;
    if (i < 15) o[i + 1] += c - 1;
    else o[0] += 38 * (c - 1);
    o[i] -= c * 65536;
  }
}

/* Constant-time conditional swap of p and q when b is 1. */
static void sel25519(gf p, gf q, int b) {
  int64_t t, c = -(int64_t)b;
  int i;
  for (i = 0; i < 16; i++) {
    t = c & (p[i] ^ q[i]);
    p[i] ^= t;
    q[i] ^= t;
  }
}

static void pack25519(uint8_t o[32], const gf n) {
  int i, j, b;
  gf m, t;
  set25519(t, n);
  car25519(t);
  car25519(t);
  car25519(t);
  for (j = 0; j < 2; j++) {
    m[0] = t[0] - 0xffed;
    for (i = 1; i < 15; i++) {
      m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
      m[i - 1] &= 0xffff;
    }
    m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
    b = (int)((m[15] >> 16) & 1);
    m[14] &= 0xffff;
    sel25519(t, m, 1 - b);
  }
  for (i = 0; i < 16; i++) {
    o[2 * i] = (uint8_t)(t[i] & 0xff);
    o[2 * i + 1] = (uint8_t)((t[i] >> 8) & 0xff);
  }
}

static int bytes_differ(const uint8_t *x, const uint8_t *y, unsigned n) {
  unsigned i, d = 0;
  for (i = 0; i < n; i++) d |= (unsigned)(x[i] ^ y[i]);
  return d != 0;
}

static int neq25519(const gf a, const gf b) {
  uint8_t c[32], d[32];
  pack25519(c, a);
  pack25519(d, b);
  return bytes_differ(c, d, 32);
}

static int par25519(const gf a) {
  uint8_t d[32];
  pack25519(d, a);
  return d[0] & 1;
}

static void unpack25519(gf o, const uint8_t n[32]) {
  int i;
  for (i = 0; i < 16; i++) o[i] = (int64_t)n[2 * i] + (int64_t)n[2 * i + 1] * 256;
  o[15] &= 0x7fff;
}

static void fadd(gf o, const gf a, const gf b) {
  int i;
  for (i = 0; i < 16; i++) o[i] = a[i] + b[i];
}

static void fsub(gf o, const gf a, const gf b) {
  int i;
  for (i = 0; i < 16; i++) o[i] = a[i] - b[i];
}

static void fmul(gf o, const gf a, const gf b) {
  int64_t t[31];
  int i, j;
  for (i = 0; i < 31; i++) t[i] = 0;
  for (i = 0; i < 16; i++)
    for (j = 0; j < 16; j++) t[i + j] += a[i] * b[j];
  for (i = 0; i < 15; i++) t[i] += 38 * t[i + 16];
  for (i = 0; i < 16; i++) o[i] = t[i];
  car25519(o);
  car25519(o);
}

static void fsq(gf o, const gf a) { fmul(o, a, a); }

static void inv25519(gf o, const gf i) {
  gf c;
  int a;
  set25519(c, i);
  for (a = 253; a >= 0; a--) {
    fsq(c, c);
    if (a != 2 && a != 4) fmul(c, c, i);
  }
  set25519(o, c);
}

static void pow2523(gf o, const gf i) {
  gf c;
  int a;
  set25519(c, i);
  for (a = 250; a >= 0; a--) {
    fsq(c, c);
    if (a != 1) fmul(c, c, i);
  }
  set25519(o, c);
}

/* --- the group, in extended coordinates (X, Y, Z, T) --------------------------------------- */

static void point_add(gf p[4], gf q[4]) {
  gf a, b, c, d, t, e, f, g, h;
  fsub(a, p[1], p[0]);
  fsub(t, q[1], q[0]);
  fmul(a, a, t);
  fadd(b, p[0], p[1]);
  fadd(t, q[0], q[1]);
  fmul(b, b, t);
  fmul(c, p[3], q[3]);
  fmul(c, c, ed_d2);
  fmul(d, p[2], q[2]);
  fadd(d, d, d);
  fsub(e, b, a);
  fsub(f, d, c);
  fadd(g, d, c);
  fadd(h, b, a);
  fmul(p[0], e, f);
  fmul(p[1], h, g);
  fmul(p[2], g, f);
  fmul(p[3], e, h);
}

static void point_cswap(gf p[4], gf q[4], int b) {
  int i;
  for (i = 0; i < 4; i++) sel25519(p[i], q[i], b);
}

static void point_pack(uint8_t r[32], gf p[4]) {
  gf tx, ty, zi;
  inv25519(zi, p[2]);
  fmul(tx, p[0], zi);
  fmul(ty, p[1], zi);
  pack25519(r, ty);
  r[31] = (uint8_t)(r[31] ^ (par25519(tx) << 7));
}

/* p = [s]q, a constant-time ladder over all 256 bits of s. */
static void scalarmult(gf p[4], gf q[4], const uint8_t s[32]) {
  int i;
  set25519(p[0], gf0);
  set25519(p[1], gf1);
  set25519(p[2], gf1);
  set25519(p[3], gf0);
  for (i = 255; i >= 0; i--) {
    int b = (s[i / 8] >> (i & 7)) & 1;
    point_cswap(p, q, b);
    point_add(q, p);
    point_add(p, p);
    point_cswap(p, q, b);
  }
}

static void scalarbase(gf p[4], const uint8_t s[32]) {
  gf q[4];
  set25519(q[0], ed_x);
  set25519(q[1], ed_y);
  set25519(q[2], gf1);
  fmul(q[3], ed_x, ed_y);
  scalarmult(p, q, s);
}

/* --- scalars mod L ------------------------------------------------------------------------- */

static const int64_t ed_l[32] = {0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10};

/* r = x mod L for a 64-limb byte-radix x (limbs may be wider than a byte on entry). */
static void mod_l(uint8_t r[32], int64_t x[64]) {
  int64_t carry;
  int i, j;
  for (i = 63; i >= 32; i--) {
    carry = 0;
    for (j = i - 32; j < i - 12; j++) {
      x[j] += carry - 16 * x[i] * ed_l[j - (i - 32)];
      carry = (x[j] + 128) >> 8;
      x[j] -= carry * 256;
    }
    x[j] += carry;
    x[i] = 0;
  }
  carry = 0;
  for (j = 0; j < 32; j++) {
    x[j] += carry - (x[31] >> 4) * ed_l[j];
    carry = x[j] >> 8;
    x[j] &= 255;
  }
  for (j = 0; j < 32; j++) x[j] -= carry * ed_l[j];
  for (i = 0; i < 32; i++) {
    x[i + 1] += x[i] >> 8;
    r[i] = (uint8_t)(x[i] & 255);
  }
}

/* A 64-byte little-endian digest reduced mod L into its first 32 bytes. */
static void reduce64(uint8_t r[64]) {
  int64_t x[64];
  int i;
  for (i = 0; i < 64; i++) x[i] = (int64_t)r[i];
  for (i = 0; i < 64; i++) r[i] = 0;
  mod_l(r, x);
}

/* Whether a 32-byte little-endian scalar is below L (the RFC 8032 section 5.1.7 bound on S). */
static int below_l(const uint8_t s[32]) {
  int i;
  for (i = 31; i >= 0; i--) {
    if ((int64_t)s[i] < ed_l[i]) return 1;
    if ((int64_t)s[i] > ed_l[i]) return 0;
  }
  return 0; /* S == L */
}

/* Whether the y of an encoded point is below p (the canonical encoding; the sign bit aside). */
static int canonical_y(const uint8_t s[32]) {
  int i;
  if ((s[31] & 0x7f) != 0x7f) return 1;
  for (i = 30; i >= 1; i--)
    if (s[i] != 0xff) return 1;
  return s[0] < 0xed;
}

/* r = -A for the encoded point A, or -1 when A is not a canonical encoding of a curve point. */
static int unpack_negated(gf r[4], const uint8_t p[32]) {
  gf t, chk, num, den, den2, den4, den6;
  uint8_t x_bytes[32];
  unsigned i, x_zero = 0;
  if (!canonical_y(p)) return -1;
  set25519(r[2], gf1);
  unpack25519(r[1], p);
  fsq(num, r[1]);
  fmul(den, num, ed_d);
  fsub(num, num, r[2]);
  fadd(den, r[2], den);
  fsq(den2, den);
  fsq(den4, den2);
  fmul(den6, den4, den2);
  fmul(t, den6, num);
  fmul(t, t, den);
  pow2523(t, t);
  fmul(t, t, num);
  fmul(t, t, den);
  fmul(t, t, den);
  fmul(r[0], t, den);
  fsq(chk, r[0]);
  fmul(chk, chk, den);
  if (neq25519(chk, num)) fmul(r[0], r[0], ed_i);
  fsq(chk, r[0]);
  fmul(chk, chk, den);
  if (neq25519(chk, num)) return -1;
  pack25519(x_bytes, r[0]);
  for (i = 0; i < 32u; i++) x_zero |= x_bytes[i];
  if (x_zero == 0 && (p[31] >> 7) != 0) return -1; /* x = 0 has one spelling: sign 0 */
  if (par25519(r[0]) == (p[31] >> 7)) fsub(r[0], gf0, r[0]);
  fmul(r[3], r[0], r[1]);
  return 0;
}

static void expand_secret(const uint8_t secret[32], uint8_t d[64]) {
  bcir_sha512_digest(secret, 32, d);
  d[0] &= 248;
  d[31] &= 127;
  d[31] |= 64;
}

void bcir_ed25519_public_key(const uint8_t secret[32], uint8_t public_key[32]) {
  uint8_t d[64];
  gf p[4];
  unsigned i;
  expand_secret(secret, d);
  scalarbase(p, d);
  point_pack(public_key, p);
  for (i = 0; i < 64u; i++) d[i] = 0;
}

void bcir_ed25519_sign(const uint8_t secret[32], const uint8_t *message, size_t length,
                       uint8_t signature[64]) {
  uint8_t d[64], pk[32], r[64], k[64];
  int64_t x[64];
  gf p[4];
  bcir_sha512 state;
  int i, j;
  expand_secret(secret, d);
  scalarbase(p, d);
  point_pack(pk, p);
  bcir_sha512_init(&state);
  bcir_sha512_update(&state, d + 32, 32);
  bcir_sha512_update(&state, message, length);
  bcir_sha512_final(&state, r);
  reduce64(r);
  scalarbase(p, r);
  point_pack(signature, p);
  bcir_sha512_init(&state);
  bcir_sha512_update(&state, signature, 32);
  bcir_sha512_update(&state, pk, 32);
  bcir_sha512_update(&state, message, length);
  bcir_sha512_final(&state, k);
  reduce64(k);
  for (i = 0; i < 64; i++) x[i] = 0;
  for (i = 0; i < 32; i++) x[i] = (int64_t)r[i];
  for (i = 0; i < 32; i++)
    for (j = 0; j < 32; j++) x[i + j] += (int64_t)k[i] * (int64_t)d[j];
  mod_l(signature + 32, x);
  for (i = 0; i < 64; i++) d[i] = r[i] = 0;
}

int bcir_ed25519_verify(const uint8_t public_key[32], const uint8_t *message, size_t length,
                        const uint8_t signature[64]) {
  uint8_t k[64], t[32];
  gf p[4], q[4];
  bcir_sha512 state;
  if (!public_key || !signature || (!message && length)) return 0;
  if (!below_l(signature + 32)) return 0;
  if (unpack_negated(q, public_key) != 0) return 0;
  bcir_sha512_init(&state);
  bcir_sha512_update(&state, signature, 32);
  bcir_sha512_update(&state, public_key, 32);
  bcir_sha512_update(&state, message, length);
  bcir_sha512_final(&state, k);
  reduce64(k);
  scalarmult(p, q, k);       /* [k](-A) */
  scalarbase(q, signature + 32); /* [S]B */
  point_add(p, q);
  point_pack(t, p);
  return !bytes_differ(signature, t, 32);
}

int bcir_ed25519_usable_key(const uint8_t public_key[32]) {
  static const uint8_t identity[32] = {1};
  uint8_t t[32];
  gf q[4];
  int i;
  if (!public_key || unpack_negated(q, public_key) != 0) return 0;
  for (i = 0; i < 3; i++) point_add(q, q); /* [8](-A): the identity exactly when A is small */
  point_pack(t, q);
  return bytes_differ(t, identity, 32);
}
