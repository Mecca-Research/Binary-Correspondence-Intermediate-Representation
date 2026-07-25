/*===- bcir_binrec.c - bounds-checked ETL binary-record field decoder -----===*/
#include "bcir_binrec.h"

bcir_binrec_status bcir_binrec_field(const uint8_t *BCIR_RESTRICT data, size_t len,
                                     const bcir_binfield *BCIR_RESTRICT f,
                                     int big_endian, int64_t *out) {
  if (out) *out = 0;
  if (!data || !f || !out)
    return BCIR_BINREC_ERR_INVALID;
  /* Only byte-aligned fields (mirrors etl.binary._extract). */
  if ((f->offset_bits % 8u) != 0u || (f->width_bits % 8u) != 0u)
    return BCIR_BINREC_ERR_UNALIGNED;
  size_t nbytes = (size_t)(f->width_bits / 8u);
  if (nbytes == 0u || nbytes > 8u)
    return BCIR_BINREC_ERR_WIDTH;          /* must fit a 64-bit value */
  size_t start = (size_t)(f->offset_bits / 8u);
  /* Overflow-safe bounds: the field must lie wholly within [0, len). */
  if (start > len || nbytes > len - start)
    return BCIR_BINREC_ERR_OOB;

  uint64_t v = 0;
  if (big_endian) {
    for (size_t i = 0; i < nbytes; ++i)
      v = (v << 8) | (uint64_t)data[start + i];
  } else {
    for (size_t i = 0; i < nbytes; ++i)
      v |= (uint64_t)data[start + i] << (8u * i);
  }

  if (f->is_signed && nbytes < 8u) {
    /* Two's-complement sign-extend an nbytes-wide value (int.from_bytes signed=True). */
    uint64_t sign_bit = (uint64_t)1 << (nbytes * 8u - 1u);
    if (v & sign_bit)
      v |= ~((sign_bit << 1) - 1u);        /* set all bits above the field */
  }
  /* Reinterpret the two's-complement bit pattern WITHOUT the conversion from an
   * out-of-range uint64_t to int64_t, which C11/C17 leave implementation-defined (only
   * C23 mandates the wraparound every target we build for happens to implement). The
   * subtraction below is always in range before the signed cast, including the INT64_MIN
   * pattern -- the same idiom bcir_telemetry_frame.c's rd_i64 uses for the same reason,
   * so the two decoders no longer disagree about how carefully to spell this. */
  *out = (v <= (uint64_t)INT64_MAX) ? (int64_t)v
                                    : -INT64_C(1) - (int64_t)(UINT64_MAX - v);
  return BCIR_BINREC_OK;
}
