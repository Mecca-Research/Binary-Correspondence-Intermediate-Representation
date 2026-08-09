/*===- bcir_per.h - freestanding X.691 PER decoding primitives --------------===
 *
 * The C twin of bcir/asn1/per.py's clause 11 machinery: Rec. ITU-T X.691 (02/2021) |
 * ISO/IEC 8825-2:2021, the bit-field reader and the whole-number and length-determinant
 * decoders that every PER type encoding is built from.
 *
 * WHY THIS LAYER. The X.690 twin can walk a whole encoding without a schema, because BER
 * is self-delimiting. PER is not (X.691 7.2: "without knowledge of the type of the value
 * encoded, it is not possible to determine the structure of the encoding"). So the
 * schema-free surface of PER is exactly clause 11 -- and that is also where the memory
 * safety risk lives, because these are the decoders that take an attacker-supplied width,
 * count or fragment header and use it to advance a cursor.
 *
 * FREESTANDING: depends only on <stddef.h> and <stdint.h>. No allocation, no libc, no
 * recursion. The reader is a cursor over caller-owned bytes and copies nothing.
 *
 * TRUST BOUNDARY. The contract is total: for ANY (data, len) and ANY arguments, every
 * entry point returns a bcir_per_status and never reads outside [data, data + len). A
 * width or count that would run past the end is BCIR_PER_TRUNCATED, not a read.
 *
 * RANGE. The Python rail is arbitrary-precision; this twin is int64/uint64. Bounds whose
 * range exceeds 64 bits are refused with BCIR_PER_RANGE rather than silently wrapped --
 * the wrap is the bug class that bit bcir_q8_model.c and the R24 I64Attr accessors, and a
 * decoder that wraps here would accept a value the encoder could never have produced.
 *
 * Parity: bcir/tests/test_c_per.py drives the same campaign through this decoder and the
 * Python rail and compares every field.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_PER_H
#define BCIR_PER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_per_status {
  BCIR_PER_OK = 0,
  /* The requested field runs past the end of the buffer. */
  BCIR_PER_TRUNCATED = 1,
  /* Bounds this twin cannot represent, or a decoded value outside them. */
  BCIR_PER_RANGE = 2,
  /* A structurally invalid encoding: a non-zero pad bit, a fragment count outside 1..4. */
  BCIR_PER_MALFORMED = 3,
  /* A NULL pointer or a nonsensical argument (a bit width above 64). */
  BCIR_PER_INVALID = 4
} bcir_per_status;

/* 11.9.3.8: fragmentation moves in multiples of 16K and the block count is 1..4. */
#define BCIR_PER_FRAG_UNIT (16u * 1024u)
#define BCIR_PER_FRAG_MAX_BLOCKS 4u

/* 7.7: the two variants. They do not interwork (7.8), so the reader carries which one it
 * is and `bcir_per_align` is a no-op for UNALIGNED. */
typedef enum bcir_per_variant {
  BCIR_PER_UNALIGNED = 0,
  BCIR_PER_ALIGNED = 1
} bcir_per_variant;

typedef struct bcir_per_reader {
  const uint8_t *data;   /* borrowed; never written, never freed */
  size_t bit_len;        /* len * 8, precomputed so every bound check is one compare */
  size_t pos;            /* current bit offset */
  bcir_per_variant variant;
} bcir_per_reader;

/* Initialise a reader over `len` bytes. `data` may be NULL only when `len` is zero. */
bcir_per_status bcir_per_reader_init(bcir_per_reader *r, const uint8_t *data, size_t len,
                                     bcir_per_variant variant);

/* Bits remaining. Never traps; returns 0 for a NULL or exhausted reader. */
size_t bcir_per_bits_left(const bcir_per_reader *r);

/* 11.3: a non-negative-binary-integer from a bit-field of `width` bits (width <= 64). */
bcir_per_status bcir_per_get_bits(bcir_per_reader *r, unsigned width, uint64_t *out);

/* One bit. */
bcir_per_status bcir_per_get_bit(bcir_per_reader *r, unsigned *out);

/* 11.1.4: in the ALIGNED variant advance to the next octet boundary, requiring the skipped
 * pad bits to be ZERO. A no-op in the UNALIGNED variant. A non-zero pad bit is
 * BCIR_PER_MALFORMED: it is a second spelling of one abstract value. */
bcir_per_status bcir_per_align(bcir_per_reader *r);

/* 11.5.6 NOTE: the minimum number of bits that can represent `range` distinct values,
 * i.e. ceil(log2(range)); 0 when range <= 1 (11.5.4's empty bit-field). */
unsigned bcir_per_bits_for_range(uint64_t range);

/* 11.5: a constrained whole number in [lb, ub]. Refuses a range wider than 64 bits. */
bcir_per_status bcir_per_constrained(bcir_per_reader *r, int64_t lb, int64_t ub,
                                     int64_t *out);

/* 11.7: a semi-constrained whole number with lower bound `lb`. */
bcir_per_status bcir_per_semi_constrained(bcir_per_reader *r, int64_t lb, int64_t *out);

/* 11.8: an unconstrained whole number (2's complement, minimum octets). */
bcir_per_status bcir_per_unconstrained(bcir_per_reader *r, int64_t *out);

/* 11.6: a normally small non-negative whole number. */
bcir_per_status bcir_per_normally_small(bcir_per_reader *r, uint64_t *out);

/* 11.9.3.4: a normally small LENGTH. Note the n-1 bias, which 11.6 does not have.
 *
 * CAVEAT for a future caller: on the 11.9.3.4 escape this delegates to bcir_per_length and
 * DISCARDS its `more` flag, so a count that named a 16K fragment rather than the final piece
 * is returned as though it were final. No decode path calls this today -- it exists for
 * primitive parity with per.py, driven only by test_per.c and fuzz_per.c -- and wiring it into
 * one without surfacing `more` would truncate a fragmented length silently, which is exactly
 * what 11.9.3.8.3's loop exists to prevent. Expose the flag before that first real caller. */
bcir_per_status bcir_per_normally_small_length(bcir_per_reader *r, uint64_t *out);

/* 11.9: one length determinant.
 *
 * `has_ub` selects 11.9.3.3's constrained form (ub < 64K) over the unconstrained forms of
 * 11.9.3.6 to 11.9.3.8. For the unconstrained forms `*more` is set when the count names a
 * 16K FRAGMENT rather than the final piece, so a caller loops until it reads zero -- which
 * is what stops a decoder from silently truncating at the last full block (11.9.3.8.3). */
bcir_per_status bcir_per_length(bcir_per_reader *r, int64_t lb, int64_t ub, int has_ub,
                                uint64_t *out, int *more);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_PER_H */
