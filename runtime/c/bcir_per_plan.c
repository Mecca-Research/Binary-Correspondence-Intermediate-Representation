/* bcir_per_plan.c -- see bcir_per_plan.h for why a PER decoder must be plan-driven. */
#include "bcir_per_plan.h"

/* 18.2's preamble is BIT-aligned, unlike 16.2's OER one: "a bit-map, consisting of a single
 * bit for each of the optional or default components", with no padding to an octet. So it is
 * read one bit at a time and the reader is left wherever it lands. */
static bcir_per_status per_preamble(bcir_per_reader *r, const bcir_per_field *fields,
                                    size_t count, bcir_per_value *out) {
  size_t i;
  for (i = 0; i < count; ++i) {
    out[i].present = 1;
    out[i].integer = 0;
    out[i].offset = 0;
    out[i].length = 0;
  }
  for (i = 0; i < count; ++i) {
    unsigned bit;
    bcir_per_status st;
    if (!fields[i].optional) continue;
    st = bcir_per_get_bit(r, &bit);
    if (st != BCIR_PER_OK) return st;
    out[i].present = (int)bit;
  }
  return BCIR_PER_OK;
}

/* 13.2's three shapes. Which applies is a fact about the type's constraint and comes from the
 * plan; 6.2 forbids inferring it from the octets, and this never tries. */
static bcir_per_status per_integer(bcir_per_reader *r, const bcir_per_field *f, int64_t *out) {
  switch (f->bounds) {
    case BCIR_PER_B_CONSTRAINED:
      return bcir_per_constrained(r, f->lb, f->ub, out);
    case BCIR_PER_B_SEMI:
      return bcir_per_semi_constrained(r, f->lb, out);
    case BCIR_PER_B_UNCONSTRAINED:
      return bcir_per_unconstrained(r, out);
    default:
      return BCIR_PER_INVALID;
  }
}

/* 17's octet string. `fixed_len` names the SIZE-fixed count for the fixed form and the SIZE
 * upper bound for the variable one; zero there means an unbounded length determinant.
 *
 * ALIGNMENT. 15.6 and 16.6 align a string of more than two octets to an octet boundary in
 * ALIGNED PER and never in UNALIGNED, which is the whole of the difference between the two
 * variants at this point in the walk -- so it is one branch on the call's flag, not a second
 * decoder. */
static bcir_per_status per_octets(bcir_per_reader *r, const bcir_per_field *f, int aligned,
                                  size_t base_bit, bcir_per_value *out) {
  uint64_t octets;
  bcir_per_status st;
  int more = 0;

  if (f->kind == BCIR_PER_K_FIXED_OCTETS) {
    octets = f->fixed_len;
  } else {
    st = bcir_per_length(r, 0, (int64_t)f->fixed_len, f->fixed_len != 0, &octets, &more);
    if (st != BCIR_PER_OK) return st;
    /* 11.9.3.8's fragmentation moves in 16K blocks. A plan-driven fast path is for the
     * bounded fields a driver reads, so a fragmented string is refused by name rather than
     * half-assembled -- the caller can fall back to the general decoder. */
    if (more) return BCIR_PER_MALFORMED;
  }
  if (aligned && octets > 2) {
    st = bcir_per_align(r);
    if (st != BCIR_PER_OK) return st;
  }
  {
    size_t here = base_bit - bcir_per_bits_left(r);
    uint64_t bits = octets * 8u;
    if (bits > bcir_per_bits_left(r)) return BCIR_PER_TRUNCATED;
    if (here % 8u != 0u && aligned) return BCIR_PER_MALFORMED;
    out->offset = here / 8u;
    out->length = (size_t)octets;
    /* Skip the body without copying it: the caller owns the buffer, and a decoder that copied
     * would be choosing an allocation policy on its behalf. */
    while (bits > 0) {
      unsigned take = bits > 64u ? 64u : (unsigned)bits;
      uint64_t sink;
      st = bcir_per_get_bits(r, take, &sink);
      if (st != BCIR_PER_OK) return st;
      bits -= take;
    }
  }
  return BCIR_PER_OK;
}

bcir_per_status bcir_per_decode_sequence(const uint8_t *data, size_t len,
                                         const bcir_per_field *fields, size_t count,
                                         int aligned, int extensible,
                                         bcir_per_value *out, size_t *end_bit) {
  bcir_per_reader reader;
  bcir_per_status st;
  size_t base_bit;
  size_t i;

  if (data == 0 || fields == 0 || out == 0) return BCIR_PER_INVALID;

  st = bcir_per_reader_init(&reader, data, len, aligned);
  if (st != BCIR_PER_OK) return st;
  base_bit = bcir_per_bits_left(&reader);

  /* 18.1: "a single bit shall be added to the field-list in a bit-field of length one" when
   * the type is extensible. Zero says the value is from the root. */
  if (extensible) {
    unsigned bit;
    st = bcir_per_get_bit(&reader, &bit);
    if (st != BCIR_PER_OK) return st;
    if (bit) return BCIR_PER_MALFORMED;  /* see the header: 18.8's skip needs what a plan lacks */
  }

  st = per_preamble(&reader, fields, count, out);
  if (st != BCIR_PER_OK) return st;

  for (i = 0; i < count; ++i) {
    const bcir_per_field *f = &fields[i];
    if (!out[i].present) continue;
    switch (f->kind) {
      case BCIR_PER_K_INTEGER:
        st = per_integer(&reader, f, &out[i].integer);
        break;
      case BCIR_PER_K_BOOLEAN: {
        /* X.691 15: one bit, and no alignment in either variant. */
        unsigned bit;
        st = bcir_per_get_bit(&reader, &bit);
        out[i].integer = (int64_t)bit;
        break;
      }
      case BCIR_PER_K_NULL:
        /* X.691 19: "shall not contribute to the field-list". No bits, no action. */
        st = BCIR_PER_OK;
        break;
      case BCIR_PER_K_FIXED_OCTETS:
      case BCIR_PER_K_VAR_OCTETS:
        st = per_octets(&reader, f, aligned, base_bit, &out[i]);
        break;
      default:
        st = BCIR_PER_INVALID;
        break;
    }
    if (st != BCIR_PER_OK) return st;
  }

  if (end_bit != 0) *end_bit = base_bit - bcir_per_bits_left(&reader);
  return BCIR_PER_OK;
}
