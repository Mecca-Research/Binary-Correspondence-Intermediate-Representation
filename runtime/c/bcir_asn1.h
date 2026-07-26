/*===- bcir_asn1.h - freestanding X.690 BER/DER decoder ---------------------===
 *
 * The C twin of bcir/asn1/. Implements Rec. ITU-T X.690 (02/2021) | ISO/IEC
 * 8825-1:2021 clause 8 (structure), clause 10 and clause 11 (the DER restrictions a
 * schema-free walk can see), over the X.680 (02/2021) universal tag assignments.
 *
 * FREESTANDING: depends only on <stddef.h> and <stdint.h>. No allocation, no libc,
 * no recursion — the walk is an explicit cursor plus a caller-sized stack, so a
 * hostile encoding cannot exhaust the C stack and a driver can decode in place.
 *
 * TRUST BOUNDARY. Every byte handed to this decoder is untrusted: a BCIR artifact
 * arriving as DER has crossed a wire, and X.690's own structures (indefinite lengths,
 * nested constructed encodings, multi-octet lengths and tags) are exactly the shapes
 * that make hand-written parsers over-read. The contract is total: for ANY (data,
 * len), every entry point returns a bcir_asn1_status and never reads outside
 * [data, data+len).
 *
 * Parity: bcir/tests/test_c_asn1.py drives the same corpus through this decoder and
 * the Python rail and compares field for field.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_ASN1_H
#define BCIR_ASN1_H

#include <stddef.h>
#include <stdint.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define BCIR_ASN1_RESTRICT restrict
#else
#define BCIR_ASN1_RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* X.690 Table 1 - encoding of class of tag (identifier octet bits 8 and 7). */
typedef enum bcir_asn1_class {
  BCIR_ASN1_UNIVERSAL = 0,
  BCIR_ASN1_APPLICATION = 1,
  BCIR_ASN1_CONTEXT = 2,
  BCIR_ASN1_PRIVATE = 3
} bcir_asn1_class;

/* X.680 clause 8 Table 1 - the universal tag numbers this decoder names. */
enum {
  BCIR_ASN1_U_EOC = 0,
  BCIR_ASN1_U_BOOLEAN = 1,
  BCIR_ASN1_U_INTEGER = 2,
  BCIR_ASN1_U_BIT_STRING = 3,
  BCIR_ASN1_U_OCTET_STRING = 4,
  BCIR_ASN1_U_NULL = 5,
  BCIR_ASN1_U_OID = 6,
  BCIR_ASN1_U_REAL = 9,
  BCIR_ASN1_U_ENUMERATED = 10,
  BCIR_ASN1_U_OBJECT_DESCRIPTOR = 7,
  BCIR_ASN1_U_UTF8_STRING = 12,
  BCIR_ASN1_U_RELATIVE_OID = 13,
  BCIR_ASN1_U_SEQUENCE = 16,
  BCIR_ASN1_U_SET = 17,
  BCIR_ASN1_U_NUMERIC_STRING = 18,
  BCIR_ASN1_U_PRINTABLE_STRING = 19,
  BCIR_ASN1_U_TELETEX_STRING = 20,
  BCIR_ASN1_U_VIDEOTEX_STRING = 21,
  BCIR_ASN1_U_IA5_STRING = 22,
  BCIR_ASN1_U_UTC_TIME = 23,
  BCIR_ASN1_U_GENERALIZED_TIME = 24,
  BCIR_ASN1_U_GRAPHIC_STRING = 25,
  BCIR_ASN1_U_VISIBLE_STRING = 26,
  BCIR_ASN1_U_GENERAL_STRING = 27,
  BCIR_ASN1_U_UNIVERSAL_STRING = 28,
  BCIR_ASN1_U_BMP_STRING = 30
};

typedef enum bcir_asn1_status {
  BCIR_ASN1_OK = 0,
  /* Iteration finished: the parent has no further children. This is NOT an error
   * and must stay distinct from BCIR_ASN1_ERR_EOC. Sharing one code let a stray
   * end-of-contents identifier octet inside a definite-length body read as "no more
   * children", silently truncating the value -- a decoder that stops early where
   * another decoder keeps reading is exactly the parser differential an attacker
   * uses to smuggle components past one implementation. */
  BCIR_ASN1_END = 11,
  BCIR_ASN1_ERR_TRUNCATED = 1,     /* the encoding runs past the buffer            */
  BCIR_ASN1_ERR_TAG = 2,           /* malformed identifier octets (X.690 8.1.2)    */
  BCIR_ASN1_ERR_LENGTH = 3,        /* malformed length octets (X.690 8.1.3)        */
  BCIR_ASN1_ERR_EOC = 4,           /* misplaced/absent end-of-contents (8.1.5)     */
  BCIR_ASN1_ERR_FORM = 5,          /* primitive/constructed used illegally         */
  BCIR_ASN1_ERR_DEPTH = 6,         /* nesting beyond the caller's stack            */
  BCIR_ASN1_ERR_NOT_DER = 7,       /* a clause 10/11 restriction was broken        */
  BCIR_ASN1_ERR_VALUE = 8,         /* contents octets are invalid for the type     */
  BCIR_ASN1_ERR_TRAILING = 9,      /* bytes remain after a complete encoding       */
  BCIR_ASN1_ERR_INVALID = 10       /* a null argument / zero-sized caller buffer   */
} bcir_asn1_status;

/* One decoded encoding, as a VIEW into the caller's buffer: `content` points into
 * `data`, so decoding copies nothing and a driver can parse a DMA'd artifact in
 * place. The view is valid only while that buffer is. */
typedef struct bcir_asn1_tlv {
  bcir_asn1_class cls;
  uint32_t number;                 /* tag number (X.690 8.1.2.2 / 8.1.2.4)         */
  int constructed;                 /* identifier octet bit 6 (X.690 8.1.2.5)       */
  int indefinite;                  /* the sender used the indefinite form (8.1.3.6)*/
  int non_minimal_length;          /* more length octets than needed (10.1)        */
  const uint8_t *content;          /* contents octets (primitive), else the body   */
  size_t content_len;
  size_t header_len;               /* identifier + length octets                   */
  size_t total_len;                /* the whole encoding, EOC included             */
  size_t offset;                   /* where it started, for diagnostics            */
} bcir_asn1_tlv;

/* Decode ONE encoding at `data[offset]`. Structural only: for a constructed
 * encoding `content` spans the children, which the caller walks with
 * bcir_asn1_first_child / bcir_asn1_next. Never recurses. */
bcir_asn1_status bcir_asn1_decode(const uint8_t *BCIR_ASN1_RESTRICT data, size_t len,
                                  size_t offset, bcir_asn1_tlv *out);

/* Decode a buffer that must hold exactly one encoding: trailing octets are a fault
 * (X.690 12.1 - an encoding is self-delimiting). */
bcir_asn1_status bcir_asn1_decode_exact(const uint8_t *BCIR_ASN1_RESTRICT data,
                                        size_t len, bcir_asn1_tlv *out);

/* Iterate the children of a constructed encoding. `first_child` seeds `child` from
 * `parent`; `next` advances in place. Both return BCIR_ASN1_ERR_EOC when the parent
 * is exhausted, which is the loop's normal termination. */
bcir_asn1_status bcir_asn1_first_child(const uint8_t *BCIR_ASN1_RESTRICT data,
                                       size_t len, const bcir_asn1_tlv *parent,
                                       bcir_asn1_tlv *child);
bcir_asn1_status bcir_asn1_next(const uint8_t *BCIR_ASN1_RESTRICT data, size_t len,
                                const bcir_asn1_tlv *parent, bcir_asn1_tlv *child);

/* Walk an entire encoding, checking structure at every level without recursing.
 * `max_depth` bounds the nesting the walk will follow. */
bcir_asn1_status bcir_asn1_validate(const uint8_t *BCIR_ASN1_RESTRICT data, size_t len,
                                    unsigned max_depth);

/* As bcir_asn1_validate, and additionally enforce every clause 10/11 restriction
 * this rail can see: definite minimal lengths (10.1), no constructed strings (10.2),
 * boolean TRUE all-ones (11.1), zeroed unused bitstring bits (11.2.1), and ascending
 * set-of components (11.6). This is the entry point a trust boundary calls. */
bcir_asn1_status bcir_asn1_validate_der(const uint8_t *BCIR_ASN1_RESTRICT data,
                                        size_t len, unsigned max_depth);

/* The clause 10/11 restrictions visible in ONE encoding, without walking its
 * children. Exposed so a caller that already has a tlv can check it in place. */
bcir_asn1_status bcir_asn1_check_der_node(const bcir_asn1_tlv *tlv);

/* Contents-octet decoders for the types the BCIR-StreamPack module uses. Each
 * enforces the rules of its clause; `bcir_asn1_integer` refuses the padded forms
 * X.690 8.3.2 rules out, and refuses a value wider than int64_t rather than
 * truncating it. */
bcir_asn1_status bcir_asn1_boolean(const bcir_asn1_tlv *tlv, int der, int *out);
bcir_asn1_status bcir_asn1_integer(const bcir_asn1_tlv *tlv, int64_t *out);
bcir_asn1_status bcir_asn1_uinteger(const bcir_asn1_tlv *tlv, uint64_t *out);
bcir_asn1_status bcir_asn1_oid_arcs(const bcir_asn1_tlv *tlv, uint32_t *arcs,
                                    size_t cap, size_t *count);

/* A human-readable name for a status, for diagnostics. Never NULL. */
const char *bcir_asn1_status_name(bcir_asn1_status status);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_ASN1_H */
