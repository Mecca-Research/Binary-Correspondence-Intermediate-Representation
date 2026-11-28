/*===- bcir_spm.h - SentencePiece-BPE tokenizer (the Llama family's tokenizer.model) -===
 *
 * The C twin of bcir/frontends/models/spm.py (PORT-2b, docs/research/
 * BCIR_NATIVE_PORT_AUDIT_2026-11-09.md N2): load a SentencePiece `tokenizer.model`, encode UTF-8
 * text to ids by score-based merges and decode ids back, id for id and octet for octet with the
 * oracle.
 *
 * THE WIRE READER. A ModelProto is read for its field 1, the repeated SentencePiece messages
 * {piece = 1: string, score = 2: float, type = 3: enum}; every other field is skipped by its wire
 * type. Varints are at most ten octets and at most 2^64 - 1 (a non-minimal spelling is read, as
 * protobuf reads it); a length that leaves its message, field number zero, and the group wire types
 * 3 and 4 (and 6, 7) are refused. Inside a piece, a field 1, 2 or 3 that repeats with its own wire
 * type is refused; a piece needs its text, which must be UTF-8 (no overlong form, no surrogate); its
 * score must be finite and its type in 1..6.
 *
 * THE LAWS, in the oracle's order: at least one piece; piece texts distinct (whatever their types);
 * a byte piece (type 6) spelled `<0xHH>` with two uppercase hex digits, each byte at most once; one
 * unknown piece (type 2); the byte alphabet complete or absent. Ids are the pieces' order.
 *
 * ENCODE: spaces become U+2581 and, unless the text already starts with one, one is prepended (the
 * dummy prefix); the characters then merge pairwise, the adjacent pair whose concatenation is a
 * normal, user-defined or unused piece with the highest score first, leftmost on a tie -- through a
 * heap, O(n log n) where the oracle rescans in O(n^2). A symbol that is no such piece falls back to
 * its UTF-8 bytes' byte pieces, or to the unknown piece when the model has no byte alphabet.
 * DECODE: a byte piece renders as its byte, a control or unknown piece as nothing, any other piece
 * as its text with U+2581 as a space; the octets then decode as UTF-8 with Python's errors='replace'
 * (each maximal ill-formed subpart becomes U+FFFD), and one leading space is dropped.
 *
 * HOSTED (hosted_tool): every allocation goes through the bcir_host_allocator lent to
 * bcir_spm_load; the tokenizer owns its tables until bcir_spm_free and is read-only after it loads,
 * so encode and decode may run on several threads at once. Model, text and ids are borrowed for
 * the call that receives them.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_SPM_H
#define BCIR_SPM_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_host_alloc.h"

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define BCIR_SPM_NODISCARD [[nodiscard]]
#else
#  define BCIR_SPM_NODISCARD
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_spm_status {
  BCIR_SPM_OK = 0,
  BCIR_SPM_ERR_WIRE = 1,       /* the protobuf wire format: see diag.law and diag.offset */
  BCIR_SPM_ERR_SHAPE = 2,      /* a piece or model law: see diag.law */
  BCIR_SPM_ERR_NOMEM = 3,      /* the allocator refused; nothing is kept */
  BCIR_SPM_ERR_UNKNOWN_ID = 4, /* decode: an id that names no piece */
  BCIR_SPM_ERR_NOT_UTF8 = 5,   /* encode: the text is not UTF-8 */
  BCIR_SPM_ERR_NOSPACE = 6,    /* a caller buffer is too small: the count says how large */
  BCIR_SPM_ERR_INVALID = 7     /* a NULL pointer, an unusable allocator or an oversized input */
} bcir_spm_status;

/* Which rule refused the model, for a diagnostic; not part of the parity. */
typedef enum bcir_spm_law {
  BCIR_SPM_LAW_NONE = 0,
  BCIR_SPM_LAW_VARINT,          /* a varint is truncated or exceeds 2^64 - 1 */
  BCIR_SPM_LAW_FIELD,           /* field number zero, a group wire type, or a length past its end */
  BCIR_SPM_LAW_PIECE_FIELDS,    /* a piece without text, or with a field 1, 2 or 3 twice */
  BCIR_SPM_LAW_PIECE_TEXT,      /* a piece text that is not UTF-8 */
  BCIR_SPM_LAW_SCORE,           /* a non-finite score */
  BCIR_SPM_LAW_TYPE,            /* a type outside 1..6 */
  BCIR_SPM_LAW_NO_PIECES,       /* the model holds no piece */
  BCIR_SPM_LAW_DUPLICATE_TEXT,  /* two pieces with one text */
  BCIR_SPM_LAW_BYTE_PIECE,      /* a byte piece not spelled <0xHH>, or a byte twice */
  BCIR_SPM_LAW_UNKNOWN_PIECE,   /* no unknown piece, or more than one */
  BCIR_SPM_LAW_BYTE_ALPHABET    /* some byte pieces, but not all 256 */
} bcir_spm_law;

typedef struct bcir_spm_diag {
  bcir_spm_status status;
  bcir_spm_law law;
  size_t offset; /* the octet the refusal points at, or (size_t)-1 */
} bcir_spm_diag;

#define BCIR_SPM_NO_OFFSET ((size_t)-1)

typedef struct bcir_spm bcir_spm; /* owned: released by bcir_spm_free */

/* Load `model[0..len)` (borrowed for the call; at most 2^32 - 1 octets). On BCIR_SPM_OK `*out` owns
 * the tokenizer; on any other status `*out` is NULL, nothing is kept and `diag` (optional) says
 * why. `allocator` NULL selects libc. */
BCIR_SPM_NODISCARD bcir_spm_status bcir_spm_load(const uint8_t *model, size_t len, const bcir_host_allocator *allocator,
                                                 bcir_spm **out, bcir_spm_diag *diag);

void bcir_spm_free(bcir_spm *tokenizer);

/* The SHA-256 of the loaded file's octets, the number of pieces, the unknown piece's id, whether
 * the model carries the byte alphabet, and the number of control and unknown pieces. */
void bcir_spm_digest(const bcir_spm *tokenizer, uint8_t sha256[32]);
size_t bcir_spm_piece_count(const bcir_spm *tokenizer);
uint32_t bcir_spm_unk_id(const bcir_spm *tokenizer);
int bcir_spm_has_byte_fallback(const bcir_spm *tokenizer);
size_t bcir_spm_special_count(const bcir_spm *tokenizer);

/* Encode UTF-8 `text[0..len)` into `ids[0..cap)`; `*count` receives the number of ids. A text of
 * `len` octets yields at most 3 * len + 3 ids, so that capacity always suffices; `ids` NULL with cap
 * 0 counts. `add_dummy_prefix` nonzero prepends U+2581 unless the text starts with one (or with a
 * space). NOT_UTF8 for text that is not UTF-8 (nothing is written), NOSPACE when cap is short
 * (*count then says how many were needed), NOMEM when the scratch cannot be had, INVALID for a text
 * of a quarter of 2^32 octets or more. */
BCIR_SPM_NODISCARD bcir_spm_status bcir_spm_encode(const bcir_spm *tokenizer, const uint8_t *text, size_t len,
                                                   int add_dummy_prefix, uint32_t *ids, size_t cap, size_t *count);

/* Decode `ids[0..n)` into `out[0..cap)`; `*written` receives the octets (always UTF-8). UNKNOWN_ID
 * for an id that names no piece; NOSPACE when cap is short (*written then says how many were
 * needed). Nothing is guaranteed in `out` on a refusal. */
BCIR_SPM_NODISCARD bcir_spm_status bcir_spm_decode(const bcir_spm *tokenizer, const uint32_t *ids, size_t n,
                                                   uint8_t *out, size_t cap, size_t *written);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_SPM_H */
