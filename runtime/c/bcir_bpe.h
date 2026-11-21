/*===- bcir_bpe.h - byte-level BPE tokenizer (the GPT-2 / Qwen tokenizer.json family) ===
 *
 * The C twin of bcir/frontends/models/tokenizer.py (PORT-2, docs/research/
 * BCIR_NATIVE_PORT_AUDIT_2026-11-09.md N2): load a HF-shaped `tokenizer.json` (model.type "BPE"),
 * encode UTF-8 text to token ids and decode ids back, id for id and octet for octet with the oracle.
 *
 * ONE JSON READER. The file is read by the bounded JER reader (bcir_jer.h) on this rail and by its
 * Python twin (`jer_bounded.scan`, then `json.loads`) on the oracle's, under one limits profile
 * (bcir_bpe_limits here, the JerLimits fields `TOKENIZER_LIMITS` there): a scan that bounds depth,
 * nodes, members, elements, string and number lengths and total work before anything is built,
 * then UTF-8 (no overlong form, no surrogate -- an escaped lone surrogate included), then
 * ECMA-404's grammar. A member name that repeats within one object, and the non-JSON NaN/Infinity
 * literals, are refused on both rails.
 *
 * THE LAWS, in the oracle's order: the root is an object; `model` an object whose `type` is "BPE";
 * `model.vocab` an object of nonempty tokens, each inside the byte alphabet, to distinct integer ids
 * in [0, 2^31-1]; all 256 base byte tokens present; `model.merges` an array of "a b" strings or
 * [a, b] pairs of nonempty tokens, each pair once, whose parts and concatenation are vocab tokens
 * (a merge's rank is its index); `added_tokens` an array of objects with nonempty string `content`
 * and an integer `id`, contents distinct, an id never naming two texts, a content in the vocab only
 * under its vocab id. Members the laws do not name are read (and held to the grammar) but not kept.
 *
 * ENCODE: specials first, longest match at each position, never split; the rest split into words
 * that keep their leading spaces (the reference pre-tokenizer); each word's bytes merged by lowest
 * rank, leftmost first -- through a heap, so a word of n bytes costs O(n log n) where the oracle's
 * loop costs O(n^2), the same ids either way. DECODE: specials render as their text, every other
 * token as its bytes, and each run between specials must be UTF-8, as the oracle decodes it.
 *
 * HOSTED (hosted_tool): every allocation goes through the bcir_host_allocator the caller lends to
 * bcir_bpe_load; the tokenizer owns its tables until bcir_bpe_free. Text and ids are borrowed for
 * the call that receives them.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_BPE_H
#define BCIR_BPE_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_host_alloc.h"
#include "bcir_jer.h"

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define BCIR_BPE_NODISCARD [[nodiscard]]
#else
#  define BCIR_BPE_NODISCARD
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_bpe_status {
  BCIR_BPE_OK = 0,
  BCIR_BPE_ERR_JSON = 1,     /* the bounded reader refused the file: see diag.jer */
  BCIR_BPE_ERR_SHAPE = 2,    /* a tokenizer law: see diag.law and diag.offset */
  BCIR_BPE_ERR_NOMEM = 3,    /* the allocator refused; nothing is kept */
  BCIR_BPE_ERR_UNKNOWN_ID = 4,
  BCIR_BPE_ERR_NOT_UTF8 = 5, /* text to encode, or a decoded run, is not UTF-8 */
  BCIR_BPE_ERR_NOSPACE = 6,  /* a caller buffer is too small: the count says how large */
  BCIR_BPE_ERR_INVALID = 7   /* a NULL pointer, an unusable allocator or an oversized text */
} bcir_bpe_status;

/* Which law refused the file (BCIR_BPE_ERR_SHAPE), for a diagnostic; not part of the parity. */
typedef enum bcir_bpe_law {
  BCIR_BPE_LAW_NONE = 0,
  BCIR_BPE_LAW_ROOT,          /* the root is not an object */
  BCIR_BPE_LAW_MODEL,         /* `model` is not an object, or its type is not "BPE" */
  BCIR_BPE_LAW_VOCAB,         /* `vocab` is not an object of byte-alphabet tokens to ids */
  BCIR_BPE_LAW_ID,            /* an id that is no integer in [0, 2^31-1], or names two tokens */
  BCIR_BPE_LAW_BASE_BYTES,    /* a base byte token is missing */
  BCIR_BPE_LAW_MERGES,        /* `merges` is not an array of well-formed, distinct, known pairs */
  BCIR_BPE_LAW_ADDED,         /* `added_tokens` is not an array of well-formed, consistent tokens */
  BCIR_BPE_LAW_DUPLICATE_KEY  /* a member name repeats within one object */
} bcir_bpe_law;

typedef struct bcir_bpe_diag {
  bcir_bpe_status status;
  bcir_bpe_law law;
  size_t offset;              /* the octet the refusal points at, or BCIR_JER_NO_OFFSET */
  bcir_jer_diag jer;          /* the reader's own diagnostic, for BCIR_BPE_ERR_JSON */
} bcir_bpe_diag;

/* The limits both rails read a tokenizer.json under (`TOKENIZER_LIMITS` on the oracle's): a named
 * profile, as the reader's strict one is, sized for the largest released vocabularies (a million
 * members in one object, two million elements in one array, 256 MiB in all). */
void bcir_bpe_limits(bcir_jer_limits *out);

typedef struct bcir_bpe bcir_bpe; /* owned: released by bcir_bpe_free */

/* Load `json[0..len)` (borrowed for the call). On BCIR_BPE_OK `*out` owns the tokenizer; on any
 * other status `*out` is NULL, nothing is kept and `diag` (optional) says why. `allocator` NULL
 * selects libc. */
BCIR_BPE_NODISCARD bcir_bpe_status bcir_bpe_load(const uint8_t *json, size_t len, const bcir_host_allocator *allocator,
                              bcir_bpe **out, bcir_bpe_diag *diag);

void bcir_bpe_free(bcir_bpe *tokenizer);

/* The SHA-256 of the loaded file's octets (the `digest` the oracle ties to a model manifest), and
 * the table sizes. */
void bcir_bpe_digest(const bcir_bpe *tokenizer, uint8_t sha256[32]);
size_t bcir_bpe_vocab_size(const bcir_bpe *tokenizer);
size_t bcir_bpe_merge_count(const bcir_bpe *tokenizer);
size_t bcir_bpe_special_count(const bcir_bpe *tokenizer);

/* Encode UTF-8 `text[0..len)` into `ids[0..cap)`; `*count` receives the number of ids. A text of
 * `len` octets yields at most `len` ids, so cap >= len always suffices; `ids` NULL with cap 0
 * counts. BCIR_BPE_ERR_NOT_UTF8 for text that is not UTF-8 (nothing is written), NOSPACE when cap
 * is short (*count then says how many were needed), NOMEM when the word scratch cannot be had, and
 * INVALID for a text of 2^32 - 1 octets or more (a symbol is indexed by a 32-bit octet offset). */
BCIR_BPE_NODISCARD bcir_bpe_status bcir_bpe_encode(const bcir_bpe *tokenizer, const uint8_t *text, size_t len,
                                uint32_t *ids, size_t cap, size_t *count);

/* Decode `ids[0..n)` into `out[0..cap)`; `*written` receives the octets. UNKNOWN_ID for an id no
 * token has; NOT_UTF8 when a run of byte tokens between specials is not UTF-8; NOSPACE when cap is
 * short (*written then says how many were needed). Nothing is guaranteed in `out` on a refusal. */
BCIR_BPE_NODISCARD bcir_bpe_status bcir_bpe_decode(const bcir_bpe *tokenizer, const uint32_t *ids, size_t n,
                                uint8_t *out, size_t cap, size_t *written);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_BPE_H */
