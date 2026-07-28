/*===- bcir_jer.h - freestanding bounded X.697 JER reader -------------------===
 *
 * The C twin of bcir/asn1/jer_bounded.py: the half of a JER decode that runs BEFORE any
 * type is consulted, on octets an attacker chose.
 *
 * WHY THIS LAYER. J2 (bcir/asn1/jer_plan.py) established the constraint this header is
 * shaped by: X.697 7.2.2 l) hides an integer's value constraint from JER, 7.2.2 h) hides a
 * SIZE on an octet or character string, and 7.2.2 g) removes extensible constraints
 * entirely. So a schema pinning `INTEGER (0..255)` and `OCTET STRING (SIZE (4))` -- bounded
 * in every binary rail in this repository -- tells a JER reader NOTHING about width. A
 * binary rail sizes its buffers from the schema. A JER rail cannot, and must be told.
 *
 * That is why every entry point here takes its capacity from the caller and this file
 * allocates nothing: the parser's container stack and its decoded-string scratch both
 * belong to whoever called it. `bcir_jer_stack_bytes` exists precisely so a caller can
 * size the one buffer whose shape is not obvious from the limits it chose.
 *
 * THE THREE STAGES, IN THE ORDER 4.2 REQUIRES. The Python rail's `decode_bounded` runs
 * scan, then UTF-8, then the value graph, then the schema, then canonical bytes. The first
 * three are here and run in the same order for the same reason -- so a refusal at any stage
 * happens before anything a caller could act on exists:
 *
 *   1. bcir_jer_scan            -- 4.3's limits, in a single octet pass, no value graph
 *   2. bcir_jer_validate_utf8   -- 7.6.2's encoding, over the whole document
 *   3. bcir_jer_parse           -- the ECMA-404 grammar, driving a caller's event sink
 *
 * Stage 1 is a BOUNDING pass and not a parser: it decides how much work the input may cost
 * and refuses beyond that. It does not check that a comma separates two values or that a
 * colon follows a member name. Stage 3 does, because unlike the Python rail there is no
 * `json.loads` behind it to fall back on.
 *
 * WHAT IS NOT HERE, AND WHY. Canonical-byte validation (3.2) needs the encoder, and schema
 * legality needs the type model; both live on the Python rail and neither is reimplemented
 * here. A twin that re-derived canonicality from a rule list would be a second definition
 * free to drift from the encoder that is the actual definition. `BCIR_JER_NOT_CANONICAL`
 * and `BCIR_JER_SCHEMA` therefore have no counterpart in this file's status enum.
 *
 * NO FLOATING POINT. A number event hands back the raw token as it appeared in the input.
 * Nothing here calls strtod, consults a locale, or rounds: a freestanding reader that
 * parsed doubles would introduce a rail on which "the same document" means two different
 * values depending on libm. The consumer decides what a number means; this file decides
 * only that it is well formed and within 4.3's digit, exponent and length ceilings.
 *
 * FREESTANDING: depends only on <stddef.h> and <stdint.h>. No allocation, no libc, no
 * recursion -- the parser's nesting lives in a caller-owned stack, so depth is a memory
 * budget the caller sets rather than a property of the C stack.
 *
 * TRUST BOUNDARY. The contract is total: for ANY (data, len), every entry point returns a
 * bcir_jer_status and never reads outside [data, data + len) nor writes outside the buffers
 * the caller passed. A construct that runs past the end is a diagnosed refusal, not a read.
 *
 * Parity: bcir/tests/test_c_jer.py drives one campaign through this reader and through
 * `jer_bounded.scan` / `json.loads`, and compares the error code, the byte offset, the
 * "how much would have been enough" figure and the whole event trace.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_JER_H
#define BCIR_JER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Spelled the way bcir_telemetry_frame.h spells it, so the `bcir_crc32` declaration below
 * matches the one in bcir_runtime.h on every toolchain the repository builds with. */
#if defined(__cplusplus)
#  define BCIR_JER_RESTRICT __restrict
#else
#  define BCIR_JER_RESTRICT restrict
#endif

/* Mirrors `bcir.asn1.jer_bounded.JerErrorCode`, value for value, so a parity test can
 * compare error CLASSES rather than message strings. The three codes that rail owns and
 * this one cannot (NOT_CANONICAL, SCHEMA, and the frame pair when no frame is in play) are
 * absent rather than stubbed -- see the header comment. */
typedef enum bcir_jer_status {
  BCIR_JER_OK = 0,
  BCIR_JER_INPUT_TOO_LARGE = 1,
  BCIR_JER_DEPTH_EXCEEDED = 2,
  BCIR_JER_NODES_EXCEEDED = 3,
  BCIR_JER_MEMBERS_EXCEEDED = 4,
  BCIR_JER_ELEMENTS_EXCEEDED = 5,
  BCIR_JER_STRING_TOO_LONG = 6,
  BCIR_JER_NUMBER_TOO_LONG = 7,
  BCIR_JER_DIGITS_EXCEEDED = 8,
  BCIR_JER_EXPONENT_EXCEEDED = 9,
  BCIR_JER_WORK_EXCEEDED = 10,
  BCIR_JER_MALFORMED = 11,
  BCIR_JER_NOT_UTF8 = 12,
  BCIR_JER_TRAILING_INPUT = 13,
  BCIR_JER_FRAME_MALFORMED = 14,
  BCIR_JER_FRAME_INTEGRITY = 15,
  /* Below here: statuses with no Python counterpart, because the Python rail allocates and
   * this one does not. They are numbered above every mirrored code so a parity table can
   * assert that anything >= BCIR_JER_OVERFLOW is a C-rail concern. */
  BCIR_JER_OVERFLOW = 16,       /* a caller buffer was too small; `needed` says how small */
  BCIR_JER_SINK_REFUSED = 17,   /* the caller's sink returned nonzero; see diag->sink_code */
  BCIR_JER_INVALID = 18         /* a NULL pointer or a nonsensical argument */
} bcir_jer_status;

/* 4.2's diagnostic: a stable code, a byte offset, and a required capacity.
 *
 * `needed` answers "how much would have been enough", which is what lets a caller retry
 * with a raised ceiling instead of guessing. Zero means the question does not apply -- every
 * genuine answer is at least one -- and mirrors the Python rail's `needed=None`.
 *
 * The schema path 4.2 also asks for is a schema-layer field; a reader with no type model
 * cannot fill it, and inventing one from JSON member names would be a path into a document
 * rather than into a type. The sink sees every member name in order and can build it. */
typedef struct bcir_jer_diag {
  bcir_jer_status status;
  size_t offset;        /* octet offset of the fault; BCIR_JER_NO_OFFSET when unlocated */
  uint64_t needed;      /* the capacity that would have sufficed, or 0 */
  int sink_code;        /* the sink's own return value, when status is SINK_REFUSED */
} bcir_jer_diag;

#define BCIR_JER_NO_OFFSET ((size_t)-1)

/* 4.3's required maxima, mirroring `JerLimits` field for field with the same defaults.
 *
 * `bcir_jer_default_limits` fills these in; a caller may then TIGHTEN any of them.
 * `bcir_jer_limits_tightened` enforces the direction, because 4.3 says limits "may be
 * tightened by a caller, never silently expanded" and a struct assignment cannot say so. */
typedef struct bcir_jer_limits {
  uint64_t input_bytes;
  uint32_t depth;
  uint64_t nodes;
  uint64_t members;
  uint64_t elements;
  uint64_t string_bytes;
  uint64_t number_bytes;
  uint32_t integer_digits;
  uint64_t exponent_magnitude;
  /* One unit per octet examined plus one per structural event, so a pathological input
   * cannot buy unbounded work with few bytes. */
  uint64_t work;
} bcir_jer_limits;

void bcir_jer_default_limits(bcir_jer_limits *out);

/* The tiny profile `STRICT_LIMITS` names on the Python rail, for control messages and for
 * the corpus 8.1 asks to cover limit boundaries. */
void bcir_jer_strict_limits(bcir_jer_limits *out);

/* Copy `from` into `out`, refusing if any field of `from` is LOOSER than `base`. Returns
 * BCIR_JER_MALFORMED on a loosened field, matching the Python rail's refusal. */
bcir_jer_status bcir_jer_limits_tightened(const bcir_jer_limits *base,
                                          const bcir_jer_limits *from,
                                          bcir_jer_limits *out);

/* --- stage 1: the bounding pass ---------------------------------------------------------
 *
 * Walk the octets once, enforcing every 4.3 limit, and report the node count. This is the
 * twin of `jer_bounded.scan`, including its work accounting, and the parity test compares
 * its diagnostics octet for octet.
 *
 * `stack` holds one entry per open container, so `members` and `elements` are per-container
 * maxima rather than document-wide totals -- which is what 4.3's wording asks for. It must
 * have room for `limits->depth` entries; `bcir_jer_stack_bytes` sizes it.
 *
 * It still tracks strings and escapes exactly, because a `{` inside a string is not a
 * structural token and an input that hid its nesting inside quotes would otherwise walk
 * straight past the depth ceiling. */
typedef struct bcir_jer_level {
  uint64_t count;       /* commas seen in this container (scan) or values placed (parse) */
  uint8_t is_object;    /* 1 for `{`, 0 for `[` */
  uint8_t state;        /* the parser's position within the container; unused by the scan */
} bcir_jer_level;

/* The octets a `limits->depth`-deep document needs for its container stack. */
size_t bcir_jer_stack_bytes(const bcir_jer_limits *limits);

bcir_jer_status bcir_jer_scan(const uint8_t *data, size_t len,
                              const bcir_jer_limits *limits,
                              bcir_jer_level *stack, size_t stack_entries,
                              uint64_t *nodes, bcir_jer_diag *diag);

/* --- stage 2: the encoding ---------------------------------------------------------------
 *
 * 7.6.2 makes a JER document UTF-8. This is the whole-document check the Python rail gets
 * from `octets.decode("utf-8")`, and it reports the offset of the first octet of the first
 * invalid sequence -- the same offset `UnicodeDecodeError.start` carries, so the two rails
 * name the same octet.
 *
 * Overlong forms, surrogates and anything above U+10FFFF are all refused: two decoders that
 * disagree about what a byte sequence means is the classic validator/consumer split. */
bcir_jer_status bcir_jer_validate_utf8(const uint8_t *data, size_t len, bcir_jer_diag *diag);

/* Decode one UTF-8 scalar at `pos`. `*width` receives the octets consumed. */
bcir_jer_status bcir_jer_utf8_next(const uint8_t *data, size_t len, size_t pos,
                                   uint32_t *code, size_t *width);

/* --- stage 3: the grammar, as events ------------------------------------------------------
 *
 * ECMA-404's grammar, driving a caller's sink. There is no value graph: the sink sees the
 * document in order and decides what to build, which is what makes a typed or transactional
 * builder possible without this file knowing any types. */
typedef enum bcir_jer_event {
  BCIR_JER_EV_OBJECT_BEGIN = 0,
  BCIR_JER_EV_OBJECT_END = 1,
  BCIR_JER_EV_ARRAY_BEGIN = 2,
  BCIR_JER_EV_ARRAY_END = 3,
  /* `text` is the DECODED member name, in the caller's scratch. */
  BCIR_JER_EV_MEMBER_NAME = 4,
  /* `text` is the DECODED string, in the caller's scratch. */
  BCIR_JER_EV_STRING = 5,
  /* `text` bounds the RAW token inside `data` -- see the header's no-floating-point note. */
  BCIR_JER_EV_NUMBER = 6,
  BCIR_JER_EV_TRUE = 7,
  BCIR_JER_EV_FALSE = 8,
  BCIR_JER_EV_NULL = 9
} bcir_jer_event;

/* Return 0 to continue. Any other value stops the parse with BCIR_JER_SINK_REFUSED and is
 * reported verbatim in `diag->sink_code`, so a schema layer can refuse a document from
 * inside the walk and still get 4.2's structured diagnostic out.
 *
 * `offset` is the octet at which the event's construct STARTS, so a sink can build a
 * diagnostic that points into the input. `text`/`len` are meaningful only for the three
 * events that name them above; they are NULL/0 otherwise. The pointer is valid only for the
 * duration of the call: the scratch is reused by the next string. */
typedef int (*bcir_jer_sink)(void *ctx, bcir_jer_event event, size_t offset,
                             const uint8_t *text, size_t len);

/* Parse one complete JSON document, then require that only white-space follows it.
 *
 * `stack` must hold `limits->depth` entries. `scratch` receives decoded strings and must
 * hold `limits->string_bytes` octets for the limits to be the binding constraint -- pass
 * less and a long string is refused with BCIR_JER_OVERFLOW and the required capacity in
 * `diag->needed`, which is a legitimate way to run with a tighter budget than the limits.
 *
 * A sink of NULL is legal and makes this a validity check that builds nothing. */
bcir_jer_status bcir_jer_parse(const uint8_t *data, size_t len,
                               const bcir_jer_limits *limits,
                               bcir_jer_level *stack, size_t stack_entries,
                               uint8_t *scratch, size_t scratch_cap,
                               bcir_jer_sink sink, void *ctx,
                               bcir_jer_diag *diag);

/* Decode the contents of a JSON string literal -- from just past the opening quote to just
 * before the closing one -- into UTF-8.
 *
 * Exposed separately because it is the single densest piece of the reader and the one a
 * fuzzer should reach directly: it owns the six two-character escapes, the four hex digits
 * of `\uXXXX`, and the surrogate pairing that 7.6.2 makes mandatory. An unpaired surrogate
 * is BCIR_JER_NOT_UTF8, not BCIR_JER_MALFORMED, because the document is well-formed JSON
 * that denotes no UTF-8 text -- the same distinction the Python rail draws.
 *
 * `out` may be NULL when `cap` is zero, which measures: `*written` still receives the
 * required capacity and the status is BCIR_JER_OVERFLOW unless the result is empty. */
bcir_jer_status bcir_jer_unescape(const uint8_t *data, size_t len,
                                  uint8_t *out, size_t cap, size_t *written,
                                  bcir_jer_diag *diag);

/* --- 3.3 framing --------------------------------------------------------------------------
 *
 * A frame is the transaction boundary: "No claim or artifact becomes visible before the
 * complete frame passes lexical, schema, semantic, and integrity checks." Integrity is the
 * part this layer owns, and it is checked BEFORE the payload is handed back rather than
 * alongside it, so a truncated or corrupted frame never yields octets a caller might act on.
 *
 * The CRC-32 detects corruption and is NOT a signature: integrity is not authenticity, and
 * BCIR_JER_FRAME_INTEGRITY says so by name. */
#define BCIR_JER_FRAME_HEADER_SIZE 32u
#define BCIR_JER_FRAME_VERSION 1u

typedef struct bcir_jer_frame {
  uint32_t version;
  uint64_t sequence;
  uint64_t generation;
  const uint8_t *payload;
  size_t payload_len;
} bcir_jer_frame;

/* Verify magic, version, declared length and CRC-32, then bind the payload. Declared in
 * bcir_runtime.h and REUSED here, never reimplemented, so the C and Python (zlib.crc32)
 * rails agree by construction -- the same discipline bcir_telemetry_frame.h follows. */
uint32_t bcir_crc32(const uint8_t *BCIR_JER_RESTRICT data, size_t len);

bcir_jer_status bcir_jer_unframe(const uint8_t *data, size_t len,
                                 bcir_jer_frame *out, bcir_jer_diag *diag);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_JER_H */
