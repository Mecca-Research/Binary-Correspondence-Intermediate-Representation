/*===- bcir_xer.h - freestanding X.693 XER lexical primitives ---------------===
 *
 * The C twin of the lexical layer of bcir/asn1/xer.py: Rec. ITU-T X.693 (02/2021) |
 * ISO/IEC 8825-4:2021, the tag scanner and the `xmlcstring` escaper of X.680 clause 12.15.
 *
 * WHY THIS LAYER AND NOT MORE. XER is a text encoding, so unlike PER there is no bit
 * cursor to get wrong -- but there IS a byte cursor, and it is driven entirely by
 * attacker-supplied content. The schema-directed half of the decode (which element belongs
 * to which component) is a walk over a type this rail does not carry in C; the half that
 * runs before any type is consulted is the tag scanner and the escape decoder, and that is
 * exactly the half where a buffer overrun lives. So that is what is twinned.
 *
 * WHAT IS EXCLUDED IS AS IMPORTANT AS WHAT IS PARSED. X.693 8.1.2's NOTE says a conforming
 * BASIC-XER encoder never produces an XML comment, processing instruction, CDATA section
 * or document type declaration, and BASIC-XER produces no XML attribute either -- those
 * arrive with the EXTENDED-XER ATTRIBUTE (clause 20) and NAMESPACE (clause 29) encoding
 * instructions. A general XML parser accepts all of them silently. This scanner names each
 * one with a bcir_xer_excluded reason, so a C peer refuses them for the stated cause
 * instead of decoding something the sender did not mean.
 *
 * FREESTANDING: depends only on <stddef.h> and <stdint.h>. No allocation, no libc, no
 * recursion. Every entry point is a pure function over caller-owned buffers.
 *
 * TRUST BOUNDARY. The contract is total: for ANY (data, len) and ANY offset, every entry
 * point returns a bcir_xer_status and never reads outside [data, data + len) nor writes
 * outside the caller's output buffer. A construct that runs past the end is
 * BCIR_XER_TRUNCATED, not a read.
 *
 * Parity: bcir/tests/test_c_xer.py drives the same campaign through this scanner and the
 * Python rail's `_Reader` and compares every field.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_XER_H
#define BCIR_XER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_xer_status {
  BCIR_XER_OK = 0,
  /* The construct runs past the end of the buffer. */
  BCIR_XER_TRUNCATED = 1,
  /* Not a well-formed tag, escape sequence or UTF-8 sequence. */
  BCIR_XER_MALFORMED = 2,
  /* Well-formed XML that X.693 clause 8 excludes; see bcir_xer_tag.excluded. */
  BCIR_XER_EXCLUDED = 3,
  /* A character with no `xmlcstring` spelling at all (X.680 12.15.1, 41.10 NOTE). */
  BCIR_XER_UNREPRESENTABLE = 4,
  /* The result does not fit in the caller's output buffer. */
  BCIR_XER_OVERFLOW = 5,
  /* A NULL pointer or a nonsensical argument. */
  BCIR_XER_INVALID = 6
} bcir_xer_status;

/* Which excluded construct was found, when the status is BCIR_XER_EXCLUDED. Each names the
 * clause that excludes it rather than reporting a generic syntax error. */
typedef enum bcir_xer_excluded {
  BCIR_XER_EXCL_NONE = 0,
  BCIR_XER_EXCL_COMMENT = 1,    /* 8.1.2 NOTE */
  BCIR_XER_EXCL_PI = 2,         /* 8.1.2 NOTE; only the 8.2.1 prolog is permitted */
  BCIR_XER_EXCL_CDATA = 3,      /* 8.1.2 NOTE */
  BCIR_XER_EXCL_DOCTYPE = 4,    /* 8.1.2 NOTE */
  BCIR_XER_EXCL_ATTRIBUTE = 5,  /* EXTENDED-XER clause 20 */
  BCIR_XER_EXCL_NAMESPACE = 6,  /* EXTENDED-XER clause 29 */
  BCIR_XER_EXCL_NUMERIC = 7     /* 9.1.3 deletes X.680 12.15.8's numeric escapes */
} bcir_xer_excluded;

typedef enum bcir_xer_tag_kind {
  BCIR_XER_TAG_START = 0,   /* <name>  */
  BCIR_XER_TAG_END = 1,     /* </name> */
  BCIR_XER_TAG_EMPTY = 2    /* <name/> */
} bcir_xer_tag_kind;

typedef struct bcir_xer_tag {
  bcir_xer_tag_kind kind;
  size_t name_off;              /* offset of the first name character in the buffer */
  size_t name_len;
  size_t end;                   /* offset just past the closing ">" */
  bcir_xer_excluded excluded;   /* BCIR_XER_EXCL_NONE unless the status is EXCLUDED */
} bcir_xer_tag;

/* X.693 8.1.4: HORIZONTAL TABULATION, LINE FEED, CARRIAGE RETURN, SPACE. Deliberately
 * narrower than XML's own S production, so a document using some other space character is
 * not silently accepted. */
int bcir_xer_is_space(int c);

/* Advance past 8.1.4 white-space starting at `pos`; returns the new offset. Never traps. */
size_t bcir_xer_skip_space(const char *data, size_t len, size_t pos);

/* Scan the tag beginning at `pos` (which must address a "<"). On BCIR_XER_OK, `out` bounds
 * the element name and gives the offset just past the tag. On BCIR_XER_EXCLUDED, `out`
 * carries the reason and nothing else is meaningful. */
bcir_xer_status bcir_xer_scan_tag(const char *data, size_t len, size_t pos,
                                  bcir_xer_tag *out);

/* Escape `len` octets of UTF-8 as an `xmlcstring` (X.680 12.15.4 and 12.15.5).
 *
 * `out` may be NULL when `cap` is zero, which measures the result: the required length is
 * still written to `*written` and the status is BCIR_XER_OVERFLOW unless the result is
 * empty. Only `&amp;`/`&lt;`/`&gt;` and the Table 3 empty elements are produced -- never a
 * numeric escape, because X.693 9.1.3 forbids those under CXER. */
bcir_xer_status bcir_xer_escape(const uint8_t *utf8, size_t len, char *out, size_t cap,
                                size_t *written);

/* The inverse: decode an `xmlcstring` back to UTF-8. `allow_numeric` admits X.680 12.15.8's
 * "&#n;" and "&#xn;" forms, which BASIC-XER permits and 9.1.3 deletes from CXER; pass 0 to
 * read canonically. A bare ">" is rejected (12.15.2). */
bcir_xer_status bcir_xer_unescape(const char *data, size_t len, int allow_numeric,
                                  uint8_t *out, size_t cap, size_t *written);

/* Decode one UTF-8 scalar at `pos`, rejecting overlong forms, surrogates and anything above
 * U+10FFFF. `*width` receives the octet count consumed. */
bcir_xer_status bcir_xer_utf8_next(const uint8_t *data, size_t len, size_t pos,
                                   uint32_t *code, size_t *width);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_XER_H */
