/* bcir_per_plan.h -- X.691 PER, decoded against a plan rather than discovered from octets.
 *
 * WHY THIS FILE EXISTS. X.691 6.2's NOTE is blunt about it: PER "does not include the
 * identifier of the type being encoded" and "the abstract syntax is required in order to
 * decode". There is no tag on the wire, no length except where a clause asks for one, and
 * a field may be an odd number of bits wide -- so there is no schema-free structural pass
 * over a PER encoding, not a slow one, none at all.
 *
 * 7.2 is therefore a statement about schema-FREE decoding and nothing more. A schema-directed
 * decode is exactly what a plan describes, which is what this file walks. bcir_oer.c is the
 * same idea one Recommendation over, and this deliberately mirrors its shape: a flat field
 * table the caller fills in, walked iteratively, writing into the caller's storage.
 *
 * WHAT PER ADDS OVER OER, and where each one lives:
 *
 *   - 18.2's presence bitmap is BIT-aligned, where 16.2's OER preamble is padded to an octet.
 *     It is read with one `get_bit` per optional component and no padding at all.
 *   - ALIGNED and UNALIGNED are two decoders over ONE field table. `bcir_per_align` is the
 *     only difference at a field boundary, but it is at every boundary, so `aligned` is a
 *     property of the DECODE CALL rather than of any field.
 *   - 18.1's extension marker is a leading bit when the type is extensible. Same reasoning:
 *     it describes the type, so it rides on the call.
 *
 * NOTHING IS COPIED. A string is reported as an offset and a length into the caller's buffer,
 * because the caller already owns it and a decoder that copied would be choosing an allocation
 * policy on its behalf. A construct running past the end is a diagnosed refusal, never a read.
 */
#ifndef BCIR_PER_PLAN_H
#define BCIR_PER_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_per.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The subset a driver-side fast path actually reads. Deliberately narrow and deliberately
 * named: a decoder that claimed the whole of X.691 without implementing it would be worse
 * than one that states its bounds. Anything outside this is refused at plan-check time, never
 * misread at decode time. */
typedef enum bcir_per_kind {
  BCIR_PER_K_INTEGER = 0,      /* 13: constrained, semi-constrained or unconstrained */
  BCIR_PER_K_BOOLEAN = 1,      /* 13.1 (X.691 15): exactly one bit, no alignment */
  BCIR_PER_K_NULL = 2,         /* 19: no bits at all */
  BCIR_PER_K_FIXED_OCTETS = 3, /* 17: a SIZE-fixed OCTET STRING, no length determinant */
  BCIR_PER_K_VAR_OCTETS = 4    /* 17/19: a length determinant, then that many octets */
} bcir_per_kind;

/* 13.2's three integer shapes. Which one applies is a fact about the TYPE's constraint, so it
 * comes from the plan; this file never infers it from the octets, because 6.2 says it cannot. */
typedef enum bcir_per_bounds {
  BCIR_PER_B_UNCONSTRAINED = 0, /* 13.2.4: a length determinant then a 2's-complement value */
  BCIR_PER_B_SEMI = 1,          /* 13.2.3: lower bound only */
  BCIR_PER_B_CONSTRAINED = 2    /* 13.2.2: both bounds, width from the range */
} bcir_per_bounds;

typedef struct bcir_per_field {
  bcir_per_kind kind;
  bcir_per_bounds bounds;
  int64_t lb;             /* SEMI and CONSTRAINED */
  int64_t ub;             /* CONSTRAINED only */
  uint32_t fixed_len;     /* BCIR_PER_K_FIXED_OCTETS: octets. VAR_OCTETS: SIZE upper bound,
                           * or 0 for an unbounded length determinant. */
  uint8_t optional;       /* participates in 18.2's preamble */
} bcir_per_field;

/* One decoded root component. `offset`/`length` bound a string INSIDE the caller's buffer. */
typedef struct bcir_per_value {
  int present;
  int64_t integer;        /* INTEGER and BOOLEAN */
  size_t offset;          /* octet-string start, inside `data` */
  size_t length;
} bcir_per_value;

/* Decode one SEQUENCE against `fields`, writing `count` results into `out`.
 *
 * `aligned` selects ALIGNED PER (X.691 clause 10's alignment before each length determinant
 * and each octet-aligned field) from UNALIGNED. `extensible` reads 18.1's leading bit; when it
 * is set and the bit says the value came from an extended type, decoding stops with
 * BCIR_PER_MALFORMED rather than guessing at 18.8's skip. That is a refusal about THIS PLAN
 * rather than about the octets -- an unknown extension is well-formed PER that the plan cannot
 * describe -- and refusing is the only honest answer, since 18.8's skip needs the extension's
 * own length determinant and a plan that named it would no longer be describing an unknown.
 *
 * `end_bit` receives the bit offset just past the SEQUENCE, so a caller can check that the
 * encoding it was handed contained nothing more. */
bcir_per_status bcir_per_decode_sequence(const uint8_t *data, size_t len,
                                         const bcir_per_field *fields, size_t count,
                                         int aligned, int extensible,
                                         bcir_per_value *out, size_t *end_bit);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_PER_PLAN_H */
