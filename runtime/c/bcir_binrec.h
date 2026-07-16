/*===- bcir_binrec.h - freestanding ETL binary-record field decoder -------===
 *
 * The C twin of bcir/etl/binary.py: decode a byte-aligned field out of a packed
 * binary record (an NVMe SQE header, a DDR5 timing word, an MMIO register map).
 * This is a *trust boundary* -- the record bytes are untrusted driver/device data --
 * so every extraction is bounds-checked: a malformed descriptor or short buffer
 * returns a status, never an out-of-bounds read (exercised by fuzz_binrec.c under
 * libFuzzer + ASan/UBSan, the same regime as the StreamPack decoder).
 *
 * Depends only on <stddef.h> + <stdint.h> (freestanding-safe). A Python<->C parity
 * test (bcir/tests/test_c23_runtime.py) gates the decode semantics against the oracle.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_BINREC_H
#define BCIR_BINREC_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_streampack.h"  /* BCIR_RESTRICT / BCIR_NODISCARD */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_binrec_status {
  BCIR_BINREC_OK = 0,
  BCIR_BINREC_ERR_UNALIGNED = 1,  /* offset_bits / width_bits not a multiple of 8 */
  BCIR_BINREC_ERR_WIDTH = 2,      /* width_bits 0 or > 64 (no int64 representation) */
  BCIR_BINREC_ERR_OOB = 3,        /* the field extends past the buffer */
  BCIR_BINREC_ERR_INVALID = 4     /* NULL descriptor, output, or data pointer */
} bcir_binrec_status;

/* A byte-aligned field descriptor (mirrors etl.binary.BinaryField). */
typedef struct bcir_binfield {
  uint16_t offset_bits;
  uint16_t width_bits;
  uint8_t  is_signed;  /* 0 = unsigned ('u'), nonzero = two's-complement signed ('s') */
} bcir_binfield;

/* Decode one field from data[0..len). On BCIR_BINREC_OK, *out holds the value
 * (sign-extended when is_signed); big_endian != 0 selects MSB-first (etl endianness
 * "big"/"be"), else little-endian. Bounds-checked end to end: any malformed field or
 * short buffer returns an error and never reads out of bounds.  On every
 * failure with a non-NULL output pointer, *out is reset to zero. */
BCIR_NODISCARD bcir_binrec_status bcir_binrec_field(const uint8_t *BCIR_RESTRICT data,
                                                    size_t len,
                                                    const bcir_binfield *BCIR_RESTRICT f,
                                                    int big_endian, int64_t *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_BINREC_H */
