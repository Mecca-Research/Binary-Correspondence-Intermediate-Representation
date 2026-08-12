/*===- bcir_per.c - freestanding X.691 PER decoding primitives --------------===
 * See bcir_per.h for the contract. Every clause reference is to Rec. ITU-T X.691 (02/2021).
 *===----------------------------------------------------------------------===*/
#include "bcir_per.h"

/* The largest byte count any single whole-number field can occupy here. 11.5.7.4 and 11.8
 * both encode into "the minimum number of octets", and this twin is 64-bit, so anything
 * past eight octets is a value it could not represent. Refusing is the point: a decoder
 * that wrapped would accept an encoding no conforming encoder produced. */
#define BCIR_PER_MAX_INT_OCTETS 8u

bcir_per_status bcir_per_reader_init(bcir_per_reader *r, const uint8_t *data, size_t len,
                                     bcir_per_variant variant) {
  if (r == NULL) return BCIR_PER_INVALID;
  if (data == NULL && len != 0u) return BCIR_PER_INVALID;
  if (variant != BCIR_PER_ALIGNED && variant != BCIR_PER_UNALIGNED)
    return BCIR_PER_INVALID;
  /* bit_len is precomputed, so it must not overflow size_t on a 32-bit host. */
  if (len > (size_t)-1 / 8u) return BCIR_PER_RANGE;
  r->data = data;
  r->bit_len = len * 8u;
  r->pos = 0u;
  r->variant = variant;
  return BCIR_PER_OK;
}

size_t bcir_per_bits_left(const bcir_per_reader *r) {
  if (r == NULL || r->pos >= r->bit_len) return 0u;
  return r->bit_len - r->pos;
}

bcir_per_status bcir_per_get_bit(bcir_per_reader *r, unsigned *out) {
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  if (r->pos >= r->bit_len) return BCIR_PER_TRUNCATED;
  {
    size_t index = r->pos >> 3;
    unsigned shift = 7u - (unsigned)(r->pos & 7u);
    *out = (unsigned)((r->data[index] >> shift) & 1u);
    r->pos += 1u;
  }
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_get_bits(bcir_per_reader *r, unsigned width, uint64_t *out) {
  uint64_t value = 0u;
  unsigned i;
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  if (width > 64u) return BCIR_PER_INVALID;
  if (width == 0u) { *out = 0u; return BCIR_PER_OK; }
  /* One bound check for the whole field: the loop below cannot then run past the end. */
  if (width > r->bit_len - r->pos || r->pos > r->bit_len) return BCIR_PER_TRUNCATED;
  for (i = 0u; i < width; i++) {
    size_t index = (r->pos + i) >> 3;
    unsigned shift = 7u - (unsigned)((r->pos + i) & 7u);
    value = (value << 1) | (uint64_t)((r->data[index] >> shift) & 1u);
  }
  r->pos += width;
  *out = value;
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_align(bcir_per_reader *r) {
  if (r == NULL) return BCIR_PER_INVALID;
  if (r->variant != BCIR_PER_ALIGNED) return BCIR_PER_OK;   /* 11.1.3: never pads */
  while ((r->pos & 7u) != 0u) {
    unsigned bit = 0u;
    bcir_per_status st = bcir_per_get_bit(r, &bit);
    if (st != BCIR_PER_OK) return st;
    if (bit != 0u) return BCIR_PER_MALFORMED;               /* 11.1.4 pads with ZERO bits */
  }
  return BCIR_PER_OK;
}

unsigned bcir_per_bits_for_range(uint64_t range) {
  unsigned bits = 0u;
  uint64_t span;
  if (range <= 1u) return 0u;                               /* 11.5.4: empty bit-field */
  span = range - 1u;
  while (span != 0u) { bits++; span >>= 1; }
  return bits;
}

/* 11.3.6: a minimum-octet non-negative-binary-integer over `octets` octets. */
static bcir_per_status read_unsigned_octets(bcir_per_reader *r, unsigned octets,
                                            uint64_t *out) {
  uint64_t value = 0u;
  unsigned i;
  if (octets > BCIR_PER_MAX_INT_OCTETS) return BCIR_PER_RANGE;
  for (i = 0u; i < octets; i++) {
    uint64_t byte = 0u;
    bcir_per_status st = bcir_per_get_bits(r, 8u, &byte);
    if (st != BCIR_PER_OK) return st;
    value = (value << 8) | byte;
  }
  *out = value;
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_constrained(bcir_per_reader *r, int64_t lb, int64_t ub,
                                     int64_t *out) {
  uint64_t span, offset = 0u;
  bcir_per_status st;
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  if (lb > ub) return BCIR_PER_RANGE;
  /* Computed in uint64 so that lb = INT64_MIN and ub = INT64_MAX does not overflow the
   * subtraction -- signed overflow there would be undefined behaviour, not a big number. */
  span = (uint64_t)ub - (uint64_t)lb;                       /* == range - 1 (11.5.3) */
  if (span == 0u) { *out = lb; return BCIR_PER_OK; }        /* 11.5.4 */

  if (r->variant == BCIR_PER_UNALIGNED) {                   /* 11.5.6 */
    unsigned bits = bcir_per_bits_for_range(span + 1u);
    if (span == (uint64_t)-1) return BCIR_PER_RANGE;        /* a full 2^64 range */
    st = bcir_per_get_bits(r, bits, &offset);
    if (st != BCIR_PER_OK) return st;
  } else if (span <= 254u) {                                /* 11.5.7.1 range <= 255 */
    st = bcir_per_get_bits(r, bcir_per_bits_for_range(span + 1u), &offset);
    if (st != BCIR_PER_OK) return st;
  } else if (span == 255u) {                                /* 11.5.7.2 the one-octet case */
    st = bcir_per_align(r);
    if (st != BCIR_PER_OK) return st;
    st = bcir_per_get_bits(r, 8u, &offset);
    if (st != BCIR_PER_OK) return st;
  } else if (span <= 65535u) {                              /* 11.5.7.3 the two-octet case */
    st = bcir_per_align(r);
    if (st != BCIR_PER_OK) return st;
    st = bcir_per_get_bits(r, 16u, &offset);
    if (st != BCIR_PER_OK) return st;
  } else {                                                  /* 11.5.7.4 indefinite length */
    uint64_t octets = 0u;
    unsigned width = 0u;
    uint64_t probe = span;
    while (probe != 0u) { width++; probe >>= 1; }
    {
      unsigned max_octets = (unsigned)((width + 7u) / 8u);
      if (max_octets == 0u) max_octets = 1u;
      if (max_octets > BCIR_PER_MAX_INT_OCTETS) return BCIR_PER_RANGE;
      /* 13.2.6 a): the length is itself constrained, lb = 1, ub = the octet count that
       * holds the range. */
      {
        int64_t len_value = 0;
        st = bcir_per_constrained(r, 1, (int64_t)max_octets, &len_value);
        if (st != BCIR_PER_OK) return st;
        octets = (uint64_t)len_value;
      }
    }
    st = bcir_per_align(r);
    if (st != BCIR_PER_OK) return st;
    st = read_unsigned_octets(r, (unsigned)octets, &offset);
    if (st != BCIR_PER_OK) return st;
  }

  if (offset > span) return BCIR_PER_RANGE;
  *out = (int64_t)((uint64_t)lb + offset);
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_semi_constrained(bcir_per_reader *r, int64_t lb, int64_t *out) {
  uint64_t count = 0u, offset = 0u;
  bcir_per_status st;
  int more = 0;
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  st = bcir_per_length(r, 0, 0, 0, &count, &more);
  if (st != BCIR_PER_OK) return st;
  if (more) return BCIR_PER_MALFORMED;      /* a whole number is never fragmented */
  if (count == 0u || count > BCIR_PER_MAX_INT_OCTETS) return BCIR_PER_RANGE;
  st = bcir_per_align(r);
  if (st != BCIR_PER_OK) return st;
  st = read_unsigned_octets(r, (unsigned)count, &offset);
  if (st != BCIR_PER_OK) return st;
  /* 11.7.4 adds the offset to lb; refuse a sum this twin cannot represent.
   * Clamping a negative lb to 0 threw away headroom the type actually has: with
   * lb = -1 the representable offsets run to INT64_MAX + 1, so the largest value of
   * INTEGER (-1..MAX) -- offset 2^63, whose sum is exactly INT64_MAX -- was reported
   * BCIR_PER_RANGE. A conforming peer's encoding of a value inside the declared type
   * must not be refused; that is a false rejection, not conservatism. */
  {
    uint64_t headroom = (uint64_t)INT64_MAX;
    if (lb < 0) headroom += (uint64_t)(-(lb + 1)) + 1u;   /* no overflow at INT64_MIN */
    else        headroom -= (uint64_t)lb;
    if (offset > headroom) return BCIR_PER_RANGE;
  }
  *out = (int64_t)((uint64_t)lb + offset);
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_unconstrained(bcir_per_reader *r, int64_t *out) {
  uint64_t count = 0u, raw = 0u;
  bcir_per_status st;
  int more = 0;
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  st = bcir_per_length(r, 0, 0, 0, &count, &more);
  if (st != BCIR_PER_OK) return st;
  if (more) return BCIR_PER_MALFORMED;
  if (count == 0u || count > BCIR_PER_MAX_INT_OCTETS) return BCIR_PER_RANGE;
  st = bcir_per_align(r);
  if (st != BCIR_PER_OK) return st;
  st = read_unsigned_octets(r, (unsigned)count, &raw);
  if (st != BCIR_PER_OK) return st;
  /* 11.4: 2's complement. Sign-extend from the top bit of the FIRST octet read. A shift
   * by 64 would be undefined, which is why the count == 8 case falls through unchanged. */
  if (count < BCIR_PER_MAX_INT_OCTETS) {
    uint64_t sign_bit = (uint64_t)1 << (count * 8u - 1u);
    if ((raw & sign_bit) != 0u) raw |= ~((sign_bit << 1) - 1u);
  }
  *out = (int64_t)raw;
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_normally_small(bcir_per_reader *r, uint64_t *out) {
  unsigned flag = 0u;
  bcir_per_status st;
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  st = bcir_per_get_bit(r, &flag);
  if (st != BCIR_PER_OK) return st;
  if (flag == 0u) return bcir_per_get_bits(r, 6u, out);      /* 11.6.1 */
  {                                                          /* 11.6.2 */
    int64_t value = 0;
    st = bcir_per_semi_constrained(r, 0, &value);
    if (st != BCIR_PER_OK) return st;
    if (value < 0) return BCIR_PER_RANGE;
    *out = (uint64_t)value;
  }
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_normally_small_length(bcir_per_reader *r, uint64_t *out) {
  unsigned flag = 0u;
  bcir_per_status st;
  if (r == NULL || out == NULL) return BCIR_PER_INVALID;
  st = bcir_per_get_bit(r, &flag);
  if (st != BCIR_PER_OK) return st;
  if (flag == 0u) {                                          /* 11.9.3.4: the n-1 bias */
    st = bcir_per_get_bits(r, 6u, out);
    if (st != BCIR_PER_OK) return st;
    *out += 1u;
    return BCIR_PER_OK;
  }
  {
    int more = 0;
    return bcir_per_length(r, 0, 0, 0, out, &more);
  }
}

bcir_per_status bcir_per_length(bcir_per_reader *r, int64_t lb, int64_t ub, int has_ub,
                                uint64_t *out, int *more) {
  bcir_per_status st;
  uint64_t first = 0u;
  if (r == NULL || out == NULL || more == NULL) return BCIR_PER_INVALID;
  *more = 0;

  if (has_ub) {                                              /* 11.9.3.3 */
    int64_t value = 0;
    if (lb < 0 || ub < lb) return BCIR_PER_RANGE;
    st = bcir_per_constrained(r, lb, ub, &value);
    if (st != BCIR_PER_OK) return st;
    if (value < 0) return BCIR_PER_RANGE;
    *out = (uint64_t)value;
    return BCIR_PER_OK;
  }

  st = bcir_per_align(r);
  if (st != BCIR_PER_OK) return st;
  st = bcir_per_get_bits(r, 8u, &first);
  if (st != BCIR_PER_OK) return st;

  if ((first & 0x80u) == 0u) {                               /* 11.9.3.6: n <= 127 */
    *out = first;
    return BCIR_PER_OK;
  }
  if ((first & 0x40u) == 0u) {                               /* 11.9.3.7: n < 16K */
    uint64_t second = 0u;
    st = bcir_per_get_bits(r, 8u, &second);
    if (st != BCIR_PER_OK) return st;
    *out = ((first & 0x3Fu) << 8) | second;
    return BCIR_PER_OK;
  }
  {                                                          /* 11.9.3.8: a fragment */
    uint64_t blocks = first & 0x3Fu;
    if (blocks < 1u || blocks > BCIR_PER_FRAG_MAX_BLOCKS) return BCIR_PER_MALFORMED;
    *out = blocks * (uint64_t)BCIR_PER_FRAG_UNIT;
    *more = 1;
  }
  return BCIR_PER_OK;
}
