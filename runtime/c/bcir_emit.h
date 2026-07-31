/*===- bcir_emit.h - freestanding plan-driven ASN.1 encoder (E2) -----------===
 *
 * The C twin of bcir/asn1/emit.py. E1 built the write-side plan and proved four Python
 * emitters byte-identical to the oracle; this is the half that produces a NATIVE number,
 * because a Python timing wearing a `measured` label is what J6's refusal exists to prevent.
 *
 * WHY A PLAN AT ALL. #682 established, against the oracle's own encoders, that only X.690
 * can be encoded without a type: X.697 22.2 puts member IDENTIFIERS in a JER document and an
 * identifier exists only in the schema. A schema-free encode harness would therefore produce
 * a two-row table with JER missing. Every emitter here is schema-directed, so the work being
 * compared is the same work.
 *
 * THE DESCRIPTOR IS DATA. `bcir_emit_parse_plan` reads the canonical text form that
 * `EncodePlan.serialize()` writes -- no callbacks, no function pointers, matching 5.1's
 * "descriptors contain no process pointers or executable callbacks". The caller owns the
 * node and member arrays, so one parsed plan serves every value.
 *
 * THE VALUE IS A NEUTRAL STREAM, identical for every candidate. Handing DER its own octets
 * and JER a typed object would measure the adapters rather than the encodings, which is the
 * error 2 warns about one level up.
 *
 * NO ALLOCATION AND NO FLOATING POINT. Recursion is present and STATICALLY BOUNDED: the plan
 * compiler refuses a schema deeper than 32, every descent is checked against the caller's
 * `max_depth`, and a SEQUENCE OF's elements are iteration rather than depth. Claiming "no
 * recursion" and then recursing would be worse than recursing.
 *
 * DER PAYS FOR ITS LENGTHS, AND THAT COST IS DER'S. X.690 10.1 forbids the indefinite form,
 * so a DER encoder must know each constructed length before it writes the header: this one
 * measures into a caller-owned scratch array and then writes. BER may leave the length open
 * and close with an EOC (8.1.3.6) and therefore needs ONE pass and no scratch -- which is
 * precisely the difference between the two candidates, so measuring it is the point rather
 * than an artefact of this implementation. JER and CANONICAL-OER are single-pass for the
 * same reason: neither back-patches a length.
 *
 * The scratch is sized by the VALUE, not the plan -- a SEQUENCE OF's element count is not in
 * the descriptor -- which is why it is a caller parameter, the same conclusion J2 reached
 * when it reported that JER's static capacity is almost always unknown.
 *
 * Every entry point is bounds-checked against the caller's capacities and reports the
 * capacity it needed, so a short buffer is a retryable answer rather than a corrupt one.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_EMIT_H
#define BCIR_EMIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BCIR_EMIT_NODISCARD __attribute__((warn_unused_result))
#else
#define BCIR_EMIT_NODISCARD
#endif

/* The plan-format version this reader accepts. A descriptor naming any other version is
 * refused rather than read hopefully: the fields are positional, so a mismatched reader
 * would silently mis-assign them instead of failing.
 *
 * Version 3 adds the optional `constraint` line. It is a BUG FIX: version 2 dropped subtype
 * constraints, which is harmless for DER, BER and JER but wrong for OER, where X.696 10.3
 * gives a constrained INTEGER a fixed-width form with no length determinant. */
#define BCIR_EMIT_PLAN_VERSION 3

/* The deepest plan this reader will accept, matching the plan compiler's own limit. */
#define BCIR_EMIT_MAX_PLAN_DEPTH 32

/* The longest member name stored. A longer one is REFUSED, never truncated: a truncated
 * JER identifier is a document that decodes to a different value. */
#define BCIR_EMIT_NAME_MAX 48

/* The longest permitted alphabet stored, in octets. X.691 30.5 makes the alphabet decide
 * bits-per-character, so a truncated one encodes a different document rather than a
 * slightly wrong one -- hence REFUSED above this, matching the compiler's own limit. 128
 * covers IA5String's whole repertoire, the largest a known-multiplier type has. */
#define BCIR_EMIT_ALPHABET_MAX 128

typedef enum bcir_emit_status {
  BCIR_EMIT_OK = 0,
  BCIR_EMIT_PLAN_MALFORMED = 1,   /* the descriptor text is not a plan */
  BCIR_EMIT_PLAN_VERSION_BAD = 2, /* a plan version this reader will not guess at */
  BCIR_EMIT_PLAN_TOO_BIG = 3,     /* more nodes, members or name than the caller supplied */
  BCIR_EMIT_STREAM_SHORT = 4,     /* the value stream ended inside a value */
  BCIR_EMIT_STREAM_LONG = 5,      /* octets left over: plan and stream disagree */
  BCIR_EMIT_OUT_SHORT = 6,        /* the output buffer is too small; `needed` is set */
  BCIR_EMIT_SCRATCH_SHORT = 7,    /* the size scratch is too small; `needed` is set */
  BCIR_EMIT_TOO_DEEP = 8,         /* deeper than the caller's stack */
  BCIR_EMIT_UNSUPPORTED = 9       /* a plan kind or value this build does not emit */
} bcir_emit_status;

typedef enum bcir_emit_rules {
  BCIR_EMIT_DER = 0,
  BCIR_EMIT_BER = 1,
  BCIR_EMIT_JER = 2,
  BCIR_EMIT_COER = 3
} bcir_emit_rules;

typedef enum bcir_emit_kind {
  BCIR_EMIT_BOOLEAN = 0,
  BCIR_EMIT_INTEGER = 1,
  BCIR_EMIT_ENUMERATED = 2,
  BCIR_EMIT_NULL = 3,
  BCIR_EMIT_OCTETSTRING = 4,
  BCIR_EMIT_STRING = 5,
  BCIR_EMIT_OID = 6,
  BCIR_EMIT_SEQUENCE = 7,
  BCIR_EMIT_SEQUENCE_OF = 8,
  BCIR_EMIT_CHOICE = 9
} bcir_emit_kind;

typedef struct bcir_emit_member {
  char name[BCIR_EMIT_NAME_MAX];
  uint8_t name_len;
  uint8_t optional;
  uint8_t has_default;
  uint8_t explicit_tag;
  uint8_t tag_class; /* 0x00 universal, 0x40 application, 0x80 context, 0xC0 private */
  int32_t tag;       /* -1 when the component keeps its base tag */
  uint32_t node;     /* index into the node table */
} bcir_emit_member;

/* One constraint bound. Sign and magnitude rather than a signed word, because the range a
 * bound may take is a property of ASN.1 and not of a machine: X.696 10.3 d)'s widest fixed
 * word is eight octets UNSIGNED, so `INTEGER (0..18446744073709551615)` has an upper bound
 * that does not fit int64_t. Storing it as int64_t would make that type's width unreadable
 * -- which is the whole fact the constraint exists to carry. */
typedef struct bcir_emit_bound {
  uint8_t present;    /* 0 = no finite bound (ASN.1 MIN/MAX, or no constraint at all) */
  uint8_t negative;   /* sign of `magnitude`; never set when `magnitude` is 0 */
  uint64_t magnitude;
} bcir_emit_bound;

/* What the encoding rules read out of a subtype constraint. Four bound pairs, not two,
 * because OER and PER disagree about extensibility: X.696 8.2.2 g) makes an extensible
 * constraint invisible to OER, while X.691 13.1 emits one bit and then encodes against the
 * extension ROOT's bounds. Those are different facts about the same constraint and they
 * genuinely differ, so deriving one from the other would be a guess. */
typedef struct bcir_emit_constraint {
  bcir_emit_bound value_low, value_high;            /* X.696 8.2.7 -- what OER reads */
  bcir_emit_bound size_low, size_high;              /* X.696 8.2.8 */
  bcir_emit_bound root_value_low, root_value_high;  /* X.691 13.1 -- what PER reads */
  bcir_emit_bound root_size_low, root_size_high;    /* X.691 17.3 / 20.4 / 30.4 */
  uint8_t value_extensible;  /* X.691 13.1: whether an extension bit is emitted */
  uint8_t size_extensible;
  uint8_t alphabet_len;      /* octets used in `alphabet`; 0 = unrestricted */
  char alphabet[BCIR_EMIT_ALPHABET_MAX];  /* X.691 30.5, UTF-8, canonical order */
} bcir_emit_constraint;

typedef struct bcir_emit_node {
  uint8_t kind;      /* bcir_emit_kind */
  uint32_t universal; /* base universal tag number; unused where the kind has none */
  uint32_t first_member;
  uint32_t member_count;
  int32_t element;   /* index of the SEQUENCE OF element node, or -1 */
  int32_t constraint; /* index into the constraint table, or -1 when unconstrained */
} bcir_emit_node;

typedef struct bcir_emit_plan {
  bcir_emit_node *nodes;
  uint32_t node_count;
  bcir_emit_member *members;
  uint32_t member_count;
  bcir_emit_constraint *constraints;
  uint32_t constraint_count;
  uint32_t root;
} bcir_emit_plan;

typedef struct bcir_emit_diag {
  bcir_emit_status status;
  /* Octet offset into whichever input failed -- the plan text for a plan error, the value
   * stream otherwise. Zero when the status names no position. */
  size_t offset;
  /* The capacity that would have sufficed, for the two SHORT statuses. Reserved for those:
   * a `needed` on an error that is not about capacity invites a retry that cannot help. */
  size_t needed;
} bcir_emit_diag;

/* Parse the canonical descriptor text into caller-owned tables. The constraint table is
 * separate from the node table, and sized by the number of CONSTRAINED nodes rather than by
 * the node count: a constraint is two orders of magnitude larger than a node and almost
 * every schema has none, so folding it into the node would make the common plan pay for the
 * rare one. Passing zero capacity is correct for a schema with no constraint. */
BCIR_EMIT_NODISCARD bcir_emit_status bcir_emit_parse_plan(
    const char *text, size_t len, bcir_emit_node *nodes, uint32_t node_cap,
    bcir_emit_member *members, uint32_t member_cap, bcir_emit_constraint *constraints,
    uint32_t constraint_cap, bcir_emit_plan *out, bcir_emit_diag *diag);

/* Emit one value. `scratch` holds one content length per visited node and is sized by the
 * VALUE; only BCIR_EMIT_DER reads it, and passing zero capacity for the other three rules is
 * correct rather than merely tolerated. `*written` receives the octet count that WOULD be
 * produced even when the buffer was too small, so a caller can size and retry exactly. */
BCIR_EMIT_NODISCARD bcir_emit_status bcir_emit(
    const bcir_emit_plan *plan, bcir_emit_rules rules, const uint8_t *stream,
    size_t stream_len, uint8_t *out, size_t out_cap, size_t *written,
    uint32_t *scratch, size_t scratch_cap, uint32_t max_depth, bcir_emit_diag *diag);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_EMIT_H */
