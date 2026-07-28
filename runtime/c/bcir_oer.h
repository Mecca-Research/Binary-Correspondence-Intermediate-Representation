/*===- bcir_oer.h - freestanding X.696 OER decoding primitives --------------===
 *
 * The C twin of the decoding half of bcir/asn1/oer.py: Rec. ITU-T X.696 (02/2021) |
 * ISO/IEC 8825-7:2021.
 *
 * WHY OER GETS A NATIVE DECODER WHEN JER GOT A SCANNER. OER is the encoding rule in the
 * suite with the best decode cost: everything is octet-aligned, so a decoder never shifts
 * bits, and most fields are fixed-width words the target can load directly. That is why the
 * ASN.1 build-out roadmap pairs it with the driver-side and DMA-fed fast path -- this is the
 * rail a privileged consumer would actually want.
 *
 * AND WHY IT IS SCHEMA-DIRECTED, WHICH IS NOT A DESIGN CHOICE. X.696 6.2 is explicit:
 * "without knowledge of the type of the value encoded, it is not possible to determine the
 * structure of the encoding". There are no tags on the wire outside a CHOICE (8.7.1) and no
 * lengths except where a clause asks for one, so there IS no schema-free structural pass
 * over OER -- the same law X.691 7.2 states for PER. A decoder that guessed would not be a
 * lenient decoder, it would be a wrong one.
 *
 * The consequence for measurement is worth stating because it was got wrong once: OER's
 * absence from the native cost table was labelled "no C decoder exists yet", as though it
 * were an ordinary gap. It is the same LAW as PER's absence. What this header changes is
 * not that -- a schema-free OER walk is still impossible -- but that a schema-DIRECTED
 * decode can now be timed against another schema-directed decode, which is like work
 * compared with like.
 *
 * THE PLAN IS DATA, NOT CODE. `bcir_oer_field` is a flat table a caller fills in from a
 * compiled descriptor; there is no code generation and no callback into a type model. That
 * is the same contract jer_plan.py's 5.1 descriptor sets ("descriptors are data; they
 * contain no process pointers or executable callbacks when serialized"), and it is what
 * lets one freestanding decoder serve every schema.
 *
 * FREESTANDING: depends only on <stddef.h> and <stdint.h>. No allocation, no libc, no
 * recursion -- a SEQUENCE's components are walked iteratively over the caller's field table.
 *
 * TRUST BOUNDARY. Total over its inputs: for ANY octets and ANY plan, every entry point
 * returns a status and never reads outside [data, data + len) nor writes outside the
 * caller's output. A construct running past the end is a diagnosed refusal, not a read.
 *
 * Parity: bcir/tests/test_c_oer.py drives one campaign through this decoder and through
 * bcir/asn1/oer.py and compares values, offsets and refusal classes.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_OER_H
#define BCIR_OER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_oer_status {
  BCIR_OER_OK = 0,
  /* The construct runs past the end of the buffer. */
  BCIR_OER_TRUNCATED = 1,
  /* Well-formed octets that the clause forbids -- e.g. a long-form length determinant
   * with a zero subsequent-octet count (8.6.5). */
  BCIR_OER_MALFORMED = 2,
  /* The value does not fit the width the caller's plan declared. */
  BCIR_OER_RANGE = 3,
  /* A caller buffer was too small; `needed` carries the required capacity. */
  BCIR_OER_OVERFLOW = 4,
  /* A NULL pointer, a nonsensical argument, or a plan this decoder cannot execute. */
  BCIR_OER_INVALID = 5
} bcir_oer_status;

typedef struct bcir_oer_diag {
  bcir_oer_status status;
  size_t offset;        /* octet offset of the fault, or BCIR_OER_NO_OFFSET */
  uint64_t needed;      /* the capacity that would have sufficed, or 0 */
} bcir_oer_diag;

#define BCIR_OER_NO_OFFSET ((size_t)-1)

/* --- 8.6 the length determinant ----------------------------------------------------------
 *
 * Short form for 0..127 (one octet, bit 8 clear); long form otherwise, whose initial octet
 * carries bit 8 set and the COUNT of subsequent octets in bits 7..1.
 *
 * BASIC-OER permits redundant leading zero octets in the long form (3.7.12 NOTE) and
 * CANONICAL-OER does not (31.2). This decoder accepts both -- that is the whole of
 * "BASIC in, CANONICAL out" -- and `canonical` reports whether what it read was the
 * canonical spelling, so a caller digesting the input can refuse on its own terms rather
 * than needing a second parser to find out. */
bcir_oer_status bcir_oer_length(const uint8_t *data, size_t len, size_t pos,
                                uint64_t *value, size_t *end, int *canonical,
                                bcir_oer_diag *diag);

/* --- 10.3 / 10.4 integers ------------------------------------------------------------------
 *
 * `width` is the fixed word width the type's effective value constraint selected (1, 2, 4
 * or 8), or 0 for the length-prefixed variable-size forms 10.3 e) and 10.4 e). `is_signed`
 * comes from clause 10.2's split, which turns on whether a lower bound EXISTS and is
 * non-negative -- not on whether the bounds happen to be small. Both are the CALLER's
 * facts, from the schema; this file never infers them from the octets, because 6.2 says it
 * cannot. */
bcir_oer_status bcir_oer_integer(const uint8_t *data, size_t len, size_t pos,
                                 unsigned width, int is_signed, int64_t *value,
                                 size_t *end, bcir_oer_diag *diag);

/* --- 16.2 the SEQUENCE preamble -------------------------------------------------------------
 *
 * One presence bit per OPTIONAL-or-DEFAULT root component, most significant bit first,
 * padded to an octet boundary. `optional_count` is how many such components the type has;
 * a type with none has no preamble at all and this returns a zero-length read.
 *
 * 16.2.2's trailing padding bits "shall be zero" under CANONICAL-OER. `canonical` reports
 * whether they were, for the same reason the length determinant does. */
bcir_oer_status bcir_oer_preamble(const uint8_t *data, size_t len, size_t pos,
                                  unsigned optional_count, uint64_t *present,
                                  size_t *end, int *canonical, bcir_oer_diag *diag);

/* --- the plan -------------------------------------------------------------------------------
 *
 * A flat description of one SEQUENCE's root components, in order. This is deliberately a
 * narrow subset -- integers, fixed-size and length-prefixed octet/character strings,
 * BOOLEAN and NULL -- because it is the subset a driver-side fast path actually reads, and
 * a decoder that claimed the whole of X.696 without implementing it would be worse than one
 * that names its bounds. Anything outside it is BCIR_OER_INVALID at plan-check time, never
 * a silent misread. */
typedef enum bcir_oer_kind {
  BCIR_OER_INTEGER = 0,     /* width/is_signed from the plan (10.3, 10.4) */
  BCIR_OER_BOOLEAN = 1,     /* 12.1: one octet, zero is FALSE and any non-zero is TRUE */
  BCIR_OER_NULL = 2,        /* 11: no octets at all */
  BCIR_OER_FIXED_OCTETS = 3,/* 14.1: a SIZE-fixed OCTET STRING, no length determinant */
  BCIR_OER_VAR_OCTETS = 4   /* 14.2/27: a length determinant then that many octets */
} bcir_oer_kind;

typedef struct bcir_oer_field {
  bcir_oer_kind kind;
  uint8_t width;            /* integer word width, or 0 for the variable-size form */
  uint8_t is_signed;
  uint8_t optional;         /* participates in the 16.2 preamble */
  uint32_t fixed_len;       /* BCIR_OER_FIXED_OCTETS only */
} bcir_oer_field;

/* One decoded root component. `offset`/`length` bound a string INSIDE `data`; nothing is
 * copied, because the caller already owns the buffer and a decoder that copied would be
 * choosing an allocation policy on its behalf. */
typedef struct bcir_oer_value {
  int present;
  int64_t integer;          /* INTEGER and BOOLEAN */
  size_t offset;            /* octet/character string start, inside `data` */
  size_t length;
} bcir_oer_value;

/* Decode one SEQUENCE against `fields`, writing `count` results into `out`.
 *
 * `canonical` reports whether every construct read was in its CANONICAL-OER spelling, so a
 * caller that digests the input can refuse a BASIC-OER encoding of the same value without
 * a second pass. `end` receives the offset just past the SEQUENCE. */
bcir_oer_status bcir_oer_decode_sequence(const uint8_t *data, size_t len, size_t pos,
                                         const bcir_oer_field *fields, size_t count,
                                         bcir_oer_value *out, size_t *end,
                                         int *canonical, bcir_oer_diag *diag);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_OER_H */
