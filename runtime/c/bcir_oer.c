/*===- bcir_oer.c - freestanding X.696 OER decoding primitives --------------===
 *
 * See bcir_oer.h for the contract. Every function here is total over its inputs and
 * touches no memory outside the buffers the caller passed.
 *===----------------------------------------------------------------------===*/
#include "bcir_oer.h"

static bcir_oer_status fail(bcir_oer_diag *diag, bcir_oer_status status, size_t offset,
                            uint64_t needed) {
  if (diag != 0) {
    diag->status = status;
    diag->offset = offset;
    diag->needed = needed;
  }
  return status;
}

static void clear(bcir_oer_diag *diag) {
  if (diag != 0) {
    diag->status = BCIR_OER_OK;
    diag->offset = BCIR_OER_NO_OFFSET;
    diag->needed = 0;
  }
}

/* --- 8.6 the length determinant ----------------------------------------------------------- */

bcir_oer_status bcir_oer_length(const uint8_t *data, size_t len, size_t pos,
                                uint64_t *value, size_t *end, int *canonical,
                                bcir_oer_diag *diag) {
  uint8_t first;
  unsigned count;
  uint64_t out = 0;
  unsigned at;

  clear(diag);
  if (canonical != 0) *canonical = 1;
  if (value == 0 || end == 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  *value = 0;
  *end = pos;
  if (data == 0 && len != 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (pos > len) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (pos == len) return fail(diag, BCIR_OER_TRUNCATED, pos, 1);

  first = data[pos];
  if ((first & 0x80u) == 0) {                      /* 8.6.4 short form */
    *value = first;
    *end = pos + 1;
    return BCIR_OER_OK;
  }
  count = (unsigned)(first & 0x7Fu);
  if (count == 0) {
    /* 8.6.5: the long form's initial octet carries the COUNT of subsequent octets, and a
     * count of zero encodes nothing at all. Not a truncation -- the octets are all here
     * and they mean nothing, which is a different fault and gets a different status. */
    return fail(diag, BCIR_OER_MALFORMED, pos, 0);
  }
  if (count > 8) {
    /* A determinant wider than the target's size_t cannot address anything this decoder
     * could then read, so it is refused rather than truncated into range. */
    return fail(diag, BCIR_OER_RANGE, pos, count);
  }
  if (len - pos - 1 < count) return fail(diag, BCIR_OER_TRUNCATED, pos, count + 1u);
  for (at = 0; at < count; at++) out = (out << 8) | (uint64_t)data[pos + 1 + at];
  /* 31.2 requires the fewest octets; 3.7.12's NOTE lets BASIC-OER carry leading zeros.
   * Both are accepted and the caller is told which it read -- a decoder that silently
   * normalized would let a peer choose the digest by choosing a spelling. */
  if (canonical != 0 && (data[pos + 1] == 0 || out < 0x80u)) *canonical = 0;
  *value = out;
  *end = pos + 1 + count;
  return BCIR_OER_OK;
}

/* --- 10.3 / 10.4 integers ------------------------------------------------------------------- */

bcir_oer_status bcir_oer_integer(const uint8_t *data, size_t len, size_t pos,
                                 unsigned width, int is_signed, int64_t *value,
                                 size_t *end, bcir_oer_diag *diag) {
  uint64_t magnitude = 0;
  size_t start = pos;
  unsigned at;

  clear(diag);
  if (value == 0 || end == 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  *value = 0;
  *end = pos;
  if (data == 0 && len != 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (pos > len) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);

  if (width == 0) {
    /* 10.3 e) / 10.4 e): a length determinant, then that many contents octets. */
    uint64_t count = 0;
    bcir_oer_status st = bcir_oer_length(data, len, pos, &count, &start, 0, diag);
    if (st != BCIR_OER_OK) return st;
    if (count == 0) return fail(diag, BCIR_OER_MALFORMED, pos, 0);
    if (count > 8) return fail(diag, BCIR_OER_RANGE, pos, count);
    if (len - start < count) return fail(diag, BCIR_OER_TRUNCATED, start, count);
    for (at = 0; at < (unsigned)count; at++)
      magnitude = (magnitude << 8) | (uint64_t)data[start + at];
    if (is_signed && (data[start] & 0x80u) != 0) {
      /* Sign-extend by hand rather than by shifting a signed value: a right shift of a
       * negative int is implementation-defined, and this file is built at -O0 and -O3 and
       * compared. */
      unsigned bits = (unsigned)count * 8u;
      if (bits < 64) magnitude |= ~(uint64_t)0 << bits;
    }
    *value = (int64_t)magnitude;
    *end = start + (size_t)count;
    return BCIR_OER_OK;
  }

  if (width != 1 && width != 2 && width != 4 && width != 8)
    return fail(diag, BCIR_OER_INVALID, pos, 0);
  if (len - pos < width) return fail(diag, BCIR_OER_TRUNCATED, pos, width);
  for (at = 0; at < width; at++) magnitude = (magnitude << 8) | (uint64_t)data[pos + at];
  if (is_signed && width < 8 && (data[pos] & 0x80u) != 0)
    magnitude |= ~(uint64_t)0 << (width * 8u);
  if (!is_signed && width == 8 && (magnitude >> 63) != 0) {
    /* An unsigned 64-bit value above INT64_MAX has no int64_t to land in. Refused rather
     * than wrapped: a decoder that returned a negative number for a positive value would
     * be handing the caller a different value, not a lossy one. */
    return fail(diag, BCIR_OER_RANGE, pos, 0);
  }
  *value = (int64_t)magnitude;
  *end = pos + width;
  return BCIR_OER_OK;
}

/* --- 16.2 the SEQUENCE preamble --------------------------------------------------------------- */

bcir_oer_status bcir_oer_preamble(const uint8_t *data, size_t len, size_t pos,
                                  unsigned optional_count, uint64_t *present,
                                  size_t *end, int *canonical, bcir_oer_diag *diag) {
  size_t octets;
  uint64_t bits = 0;
  unsigned at;

  clear(diag);
  if (canonical != 0) *canonical = 1;
  if (present == 0 || end == 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  *present = 0;
  *end = pos;
  if (data == 0 && len != 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (pos > len) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (optional_count == 0) return BCIR_OER_OK;   /* no OPTIONAL/DEFAULT root: no preamble */
  if (optional_count > 64) return fail(diag, BCIR_OER_INVALID, pos, optional_count);

  octets = (optional_count + 7u) / 8u;
  if (len - pos < octets) return fail(diag, BCIR_OER_TRUNCATED, pos, octets);
  for (at = 0; at < optional_count; at++) {
    /* Most significant bit of the first octet is the FIRST optional component. */
    uint8_t octet = data[pos + at / 8u];
    if ((octet >> (7u - (at % 8u))) & 1u) bits |= (uint64_t)1 << at;
  }
  if (canonical != 0) {
    /* 16.2.2: the trailing padding bits shall be zero under CANONICAL-OER. */
    unsigned used = optional_count % 8u;
    if (used != 0 && (data[pos + octets - 1] & (uint8_t)((1u << (8u - used)) - 1u)) != 0)
      *canonical = 0;
  }
  *present = bits;
  *end = pos + octets;
  return BCIR_OER_OK;
}

/* --- the plan-driven SEQUENCE decode ------------------------------------------------------------ */

bcir_oer_status bcir_oer_decode_sequence(const uint8_t *data, size_t len, size_t pos,
                                         const bcir_oer_field *fields, size_t count,
                                         bcir_oer_value *out, size_t *end,
                                         int *canonical, bcir_oer_diag *diag) {
  unsigned optional_count = 0;
  unsigned optional_seen = 0;
  uint64_t present = 0;
  size_t at;
  bcir_oer_status st;

  clear(diag);
  if (canonical != 0) *canonical = 1;
  if (end == 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  *end = pos;
  if (fields == 0 && count != 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (out == 0 && count != 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (data == 0 && len != 0) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
  if (pos > len) return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);

  /* Check the whole plan BEFORE reading an octet. A plan this decoder cannot execute must
   * be a refusal, never a partial decode that stopped somewhere in the middle -- the
   * caller would otherwise have to distinguish "the input was short" from "the plan was
   * wrong" by inspecting how far it got. */
  for (at = 0; at < count; at++) {
    const bcir_oer_field *f = &fields[at];
    if (f->kind > BCIR_OER_VAR_OCTETS)
      return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
    if (f->kind == BCIR_OER_INTEGER && f->width != 0 && f->width != 1 && f->width != 2 &&
        f->width != 4 && f->width != 8)
      return fail(diag, BCIR_OER_INVALID, BCIR_OER_NO_OFFSET, 0);
    if (f->optional) optional_count++;
  }

  {
    size_t after = pos;
    int preamble_canonical = 1;
    st = bcir_oer_preamble(data, len, pos, optional_count, &present, &after,
                           &preamble_canonical, diag);
    if (st != BCIR_OER_OK) return st;
    if (canonical != 0 && !preamble_canonical) *canonical = 0;
    pos = after;
  }

  for (at = 0; at < count; at++) {
    const bcir_oer_field *f = &fields[at];
    bcir_oer_value *slot = &out[at];
    size_t after = pos;

    slot->present = 1;
    slot->integer = 0;
    slot->offset = pos;
    slot->length = 0;

    if (f->optional) {
      int here = (present >> optional_seen) & 1u;
      optional_seen++;
      if (!here) {
        /* 16.2: an absent OPTIONAL component occupies NO octets. The cursor does not
         * move, which is why absence has to be read from the preamble rather than
         * discovered from the contents. */
        slot->present = 0;
        continue;
      }
    }

    switch (f->kind) {
      case BCIR_OER_NULL:
        break;                                   /* clause 11: no octets */
      case BCIR_OER_BOOLEAN:
        if (len - pos < 1) return fail(diag, BCIR_OER_TRUNCATED, pos, 1);
        /* 12.1: FALSE is zero and ANY non-zero octet is TRUE on input. CANONICAL-OER
         * emits 0xFF, so a TRUE that is not 0xFF is legal BASIC-OER and non-canonical. */
        slot->integer = data[pos] != 0;
        if (canonical != 0 && data[pos] != 0 && data[pos] != 0xFF) *canonical = 0;
        after = pos + 1;
        break;
      case BCIR_OER_INTEGER:
        st = bcir_oer_integer(data, len, pos, f->width, f->is_signed, &slot->integer,
                              &after, diag);
        if (st != BCIR_OER_OK) return st;
        break;
      case BCIR_OER_FIXED_OCTETS:
        /* 14.1: a SIZE-fixed string carries no length determinant at all. */
        if (len - pos < f->fixed_len)
          return fail(diag, BCIR_OER_TRUNCATED, pos, f->fixed_len);
        slot->offset = pos;
        slot->length = f->fixed_len;
        after = pos + f->fixed_len;
        break;
      case BCIR_OER_VAR_OCTETS:
      default: {
        uint64_t length = 0;
        size_t body = pos;
        int length_canonical = 1;
        st = bcir_oer_length(data, len, pos, &length, &body, &length_canonical, diag);
        if (st != BCIR_OER_OK) return st;
        if (canonical != 0 && !length_canonical) *canonical = 0;
        if ((uint64_t)(len - body) < length)
          return fail(diag, BCIR_OER_TRUNCATED, body, length);
        slot->offset = body;
        slot->length = (size_t)length;
        after = body + (size_t)length;
        break;
      }
    }
    pos = after;
  }
  *end = pos;
  return BCIR_OER_OK;
}
