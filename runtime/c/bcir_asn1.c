/*===- bcir_asn1.c - freestanding X.690 BER/DER decoder (see bcir_asn1.h) ---===*/
#include "bcir_asn1.h"

/* Bound a declared length before it is believed: a hostile encoding can claim a
 * 2^64-octet body in eight bytes, and a decoder that trusts it before comparing
 * against the buffer is a memory-exhaustion surface even without allocating. */
#define BCIR_ASN1_MAX_LENGTH_OCTETS 8u

/* --- 8.1.2 identifier octets ----------------------------------------------- */

static bcir_asn1_status decode_tag(const uint8_t *data, size_t len, size_t pos,
                                   bcir_asn1_tlv *out, size_t *next) {
  uint8_t lead;
  uint32_t number;

  if (pos >= len) return BCIR_ASN1_ERR_TRUNCATED;
  lead = data[pos];
  out->cls = (bcir_asn1_class)((lead >> 6) & 0x03u);
  out->constructed = (lead & 0x20u) != 0u;
  number = (uint32_t)(lead & 0x1Fu);
  pos++;

  if (number != 0x1Fu) {                      /* 8.1.2.2: the short form */
    out->number = number;
    *next = pos;
    return BCIR_ASN1_OK;
  }

  /* 8.1.2.4: the high-tag-number form. 8.1.2.4.2 c) forbids a first subsequent
   * octet whose bits 7 to 1 are all zero -- the tag-number analogue of a
   * non-minimal integer, and the only way this form can encode redundantly. */
  if (pos >= len) return BCIR_ASN1_ERR_TRUNCATED;
  if (data[pos] == 0x80u) return BCIR_ASN1_ERR_TAG;
  number = 0u;
  for (;;) {
    uint8_t octet;
    if (pos >= len) return BCIR_ASN1_ERR_TRUNCATED;
    octet = data[pos];
    if (number > (UINT32_MAX >> 7)) return BCIR_ASN1_ERR_TAG;   /* would overflow */
    number = (number << 7) | (uint32_t)(octet & 0x7Fu);
    pos++;
    if ((octet & 0x80u) == 0u) break;
  }
  if (number <= 30u) return BCIR_ASN1_ERR_TAG;  /* 8.1.2.2 covers 0..30 */
  out->number = number;
  *next = pos;
  return BCIR_ASN1_OK;
}

/* --- 8.1.3 length octets ---------------------------------------------------- */

static bcir_asn1_status decode_length(const uint8_t *data, size_t len, size_t pos,
                                      size_t *value, int *indefinite,
                                      int *non_minimal, size_t *next) {
  uint8_t initial;
  unsigned count;
  size_t accumulated = 0u;
  unsigned i;

  *indefinite = 0;
  *non_minimal = 0;
  if (pos >= len) return BCIR_ASN1_ERR_TRUNCATED;
  initial = data[pos];
  pos++;

  if (initial == 0x80u) {                     /* 8.1.3.6: the indefinite form */
    *indefinite = 1;
    *value = 0u;
    *next = pos;
    return BCIR_ASN1_OK;
  }
  if ((initial & 0x80u) == 0u) {              /* 8.1.3.4: the short form */
    *value = (size_t)initial;
    *next = pos;
    return BCIR_ASN1_OK;
  }
  if (initial == 0xFFu) return BCIR_ASN1_ERR_LENGTH;  /* 8.1.3.5 c): reserved */

  count = (unsigned)(initial & 0x7Fu);
  if (count > BCIR_ASN1_MAX_LENGTH_OCTETS) return BCIR_ASN1_ERR_LENGTH;
  if (count > len - pos) return BCIR_ASN1_ERR_TRUNCATED;
  for (i = 0; i < count; ++i) accumulated = (accumulated << 8) | (size_t)data[pos + i];
  /* 10.1 minimality: the short form would have sufficed, or a leading zero octet. */
  if (accumulated <= 127u || (count > 1u && data[pos] == 0x00u)) *non_minimal = 1;
  pos += count;
  *value = accumulated;
  *next = pos;
  return BCIR_ASN1_OK;
}

/* --- the structural decode -------------------------------------------------- */

/* Find the end-of-contents octets that close an indefinite-length encoding
 * starting at `pos`, without recursing: track how many indefinite encodings are
 * open and skip definite ones wholesale. */
static bcir_asn1_status scan_indefinite_end(const uint8_t *data, size_t len,
                                            size_t pos, size_t *end) {
  unsigned depth = 1u;
  while (depth > 0u) {
    bcir_asn1_tlv probe;
    size_t after_tag, after_len, body;
    int indefinite, non_minimal;
    bcir_asn1_status status;

    if (pos + 1u < len && data[pos] == 0x00u && data[pos + 1] == 0x00u) {
      pos += 2u;
      depth--;
      continue;
    }
    status = decode_tag(data, len, pos, &probe, &after_tag);
    if (status != BCIR_ASN1_OK) return status;
    status = decode_length(data, len, after_tag, &body, &indefinite, &non_minimal,
                           &after_len);
    if (status != BCIR_ASN1_OK) return status;
    if (indefinite) {
      if (!probe.constructed) return BCIR_ASN1_ERR_FORM;   /* 8.1.3.2 a) */
      if (depth >= 1024u) return BCIR_ASN1_ERR_DEPTH;
      depth++;
      pos = after_len;
    } else {
      if (body > len - after_len) return BCIR_ASN1_ERR_TRUNCATED;
      pos = after_len + body;
    }
  }
  *end = pos;
  return BCIR_ASN1_OK;
}

bcir_asn1_status bcir_asn1_decode(const uint8_t *BCIR_ASN1_RESTRICT data, size_t len,
                                  size_t offset, bcir_asn1_tlv *out) {
  size_t after_tag, after_len, body, end;
  int indefinite, non_minimal;
  bcir_asn1_status status;

  if (!data || !out) return BCIR_ASN1_ERR_INVALID;
  if (offset > len) return BCIR_ASN1_ERR_TRUNCATED;

  out->indefinite = 0;
  out->non_minimal_length = 0;
  out->content = data + offset;
  out->content_len = 0u;
  out->offset = offset;

  status = decode_tag(data, len, offset, out, &after_tag);
  if (status != BCIR_ASN1_OK) return status;
  /* 8.1.5: UNIVERSAL 0 is the end-of-contents marker, never a value in its own
   * right, so meeting one here means an EOC appeared with nothing open. */
  if (out->cls == BCIR_ASN1_UNIVERSAL && out->number == BCIR_ASN1_U_EOC)
    return BCIR_ASN1_ERR_EOC;

  status = decode_length(data, len, after_tag, &body, &indefinite, &non_minimal,
                         &after_len);
  if (status != BCIR_ASN1_OK) return status;
  out->non_minimal_length = non_minimal;
  out->header_len = after_len - offset;

  if (indefinite) {
    if (!out->constructed) return BCIR_ASN1_ERR_FORM;      /* 8.1.3.2 a) */
    status = scan_indefinite_end(data, len, after_len, &end);
    if (status != BCIR_ASN1_OK) return status;
    out->indefinite = 1;
    out->content = data + after_len;
    out->content_len = (end - after_len) - 2u;             /* less the EOC octets */
    out->total_len = end - offset;
    return BCIR_ASN1_OK;
  }

  if (body > len - after_len) return BCIR_ASN1_ERR_TRUNCATED;
  out->content = data + after_len;
  out->content_len = body;
  out->total_len = (after_len + body) - offset;
  return BCIR_ASN1_OK;
}

bcir_asn1_status bcir_asn1_decode_exact(const uint8_t *BCIR_ASN1_RESTRICT data,
                                        size_t len, bcir_asn1_tlv *out) {
  bcir_asn1_status status = bcir_asn1_decode(data, len, 0u, out);
  if (status != BCIR_ASN1_OK) return status;
  if (out->total_len != len) return BCIR_ASN1_ERR_TRAILING;
  return BCIR_ASN1_OK;
}

/* --- child iteration -------------------------------------------------------- */

/* A child is decoded against its PARENT'S body end, not against the whole buffer.
 *
 * This is the load-bearing bound in the whole decoder. If a child is decoded with
 * the outer buffer length, a constructed encoding's declared length stops being
 * authoritative over its contents: a sender can declare a short parent and let the
 * last child run past it, so a decoder that walks children sees octets a decoder
 * that trusts the parent length does not. That divergence is precisely how content
 * gets smuggled through one X.690 implementation and not another. Passing the body
 * end as the bound makes an overrunning child BCIR_ASN1_ERR_TRUNCATED, which is
 * what X.690 8.1.1's nesting requires and what the Python rail reports. */
bcir_asn1_status bcir_asn1_first_child(const uint8_t *BCIR_ASN1_RESTRICT data,
                                       size_t len, const bcir_asn1_tlv *parent,
                                       bcir_asn1_tlv *child) {
  size_t start, body_end;
  if (!data || !parent || !child) return BCIR_ASN1_ERR_INVALID;
  if (!parent->constructed) return BCIR_ASN1_ERR_FORM;
  if (parent->content_len == 0u) return BCIR_ASN1_END;       /* empty: done */
  start = (size_t)(parent->content - data);
  body_end = start + parent->content_len;
  if (body_end > len) return BCIR_ASN1_ERR_TRUNCATED;
  return bcir_asn1_decode(data, body_end, start, child);
}

bcir_asn1_status bcir_asn1_next(const uint8_t *BCIR_ASN1_RESTRICT data, size_t len,
                                const bcir_asn1_tlv *parent, bcir_asn1_tlv *child) {
  size_t body_start, body_end, next;
  if (!data || !parent || !child) return BCIR_ASN1_ERR_INVALID;
  body_start = (size_t)(parent->content - data);
  body_end = body_start + parent->content_len;
  if (body_end > len) return BCIR_ASN1_ERR_TRUNCATED;
  next = child->offset + child->total_len;
  if (next >= body_end) return BCIR_ASN1_END;                /* exhausted: done */
  return bcir_asn1_decode(data, body_end, next, child);
}

/* --- the walks -------------------------------------------------------------- */

/* Explicit stack rather than recursion: the depth is attacker-chosen, so the walk
 * must be bounded by a value the caller sets, not by the C stack. */
#define BCIR_ASN1_STACK 64u

static bcir_asn1_status walk(const uint8_t *data, size_t len, unsigned max_depth,
                             int der) {
  bcir_asn1_tlv stack_parent[BCIR_ASN1_STACK];
  bcir_asn1_tlv stack_child[BCIR_ASN1_STACK];
  /* The previous sibling's encoding, for the 11.6 set-of ordering comparison. */
  const uint8_t *prev_start[BCIR_ASN1_STACK];
  size_t prev_len[BCIR_ASN1_STACK];
  unsigned depth = 0u;
  bcir_asn1_tlv root;
  bcir_asn1_status status;

  if (!data) return BCIR_ASN1_ERR_INVALID;
  if (max_depth == 0u || max_depth > BCIR_ASN1_STACK) max_depth = BCIR_ASN1_STACK;

  status = bcir_asn1_decode_exact(data, len, &root);
  if (status != BCIR_ASN1_OK) return status;
  status = der ? bcir_asn1_check_der_node(&root) : BCIR_ASN1_OK;
  if (status != BCIR_ASN1_OK) return status;
  if (!root.constructed) return BCIR_ASN1_OK;

  stack_parent[0] = root;
  prev_start[0] = (const uint8_t *)0;
  prev_len[0] = 0u;
  status = bcir_asn1_first_child(data, len, &stack_parent[0], &stack_child[0]);
  if (status == BCIR_ASN1_END) return BCIR_ASN1_OK;
  if (status != BCIR_ASN1_OK) return status;

  for (;;) {
    bcir_asn1_tlv *cur = &stack_child[depth];
    if (der) {
      status = bcir_asn1_check_der_node(cur);
      if (status != BCIR_ASN1_OK) return status;
      /* 11.6: set-of components ascend as octet strings, the shorter padded with
       * trailing zero octets for the comparison only. */
      if (stack_parent[depth].cls == BCIR_ASN1_UNIVERSAL &&
          stack_parent[depth].number == BCIR_ASN1_U_SET && prev_start[depth]) {
        const uint8_t *a = prev_start[depth];
        const uint8_t *b = data + cur->offset;
        size_t na = prev_len[depth], nb = cur->total_len, i, n = na > nb ? na : nb;
        for (i = 0; i < n; ++i) {
          uint8_t x = i < na ? a[i] : 0u;
          uint8_t y = i < nb ? b[i] : 0u;
          if (x != y) { if (x > y) return BCIR_ASN1_ERR_NOT_DER; break; }
        }
      }
      prev_start[depth] = data + cur->offset;
      prev_len[depth] = cur->total_len;
    }

    if (cur->constructed) {                      /* descend */
      bcir_asn1_tlv first;
      if (depth + 1u >= max_depth) return BCIR_ASN1_ERR_DEPTH;
      status = bcir_asn1_first_child(data, len, cur, &first);
      if (status == BCIR_ASN1_OK) {
        stack_parent[depth + 1u] = *cur;
        stack_child[depth + 1u] = first;
        prev_start[depth + 1u] = (const uint8_t *)0;
        prev_len[depth + 1u] = 0u;
        depth++;
        continue;
      }
      if (status != BCIR_ASN1_END) return status;
    }
    for (;;) {                                   /* advance, unwinding as needed */
      status = bcir_asn1_next(data, len, &stack_parent[depth], &stack_child[depth]);
      if (status == BCIR_ASN1_OK) break;
      if (status != BCIR_ASN1_END) return status;
      if (depth == 0u) return BCIR_ASN1_OK;
      depth--;
    }
  }
}

bcir_asn1_status bcir_asn1_validate(const uint8_t *BCIR_ASN1_RESTRICT data, size_t len,
                                    unsigned max_depth) {
  return walk(data, len, max_depth, 0);
}

bcir_asn1_status bcir_asn1_validate_der(const uint8_t *BCIR_ASN1_RESTRICT data,
                                        size_t len, unsigned max_depth) {
  return walk(data, len, max_depth, 1);
}

/* --- clause 10 + 11, per node ----------------------------------------------- */

static int all_digits(const uint8_t *p, size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) if (p[i] < '0' || p[i] > '9') return 0;
  return 1;
}

/* 11.8: "YYMMDDHHMMSSZ" -- seconds always present (11.8.2), always "Z" (11.8.1). */
static int canonical_utctime(const uint8_t *p, size_t n) {
  return n == 13u && all_digits(p, 12u) && p[12] == 'Z';
}

/* 11.7: "YYYYMMDDHHMMSS[.f]Z" -- seconds present (11.7.2), "Z" terminated (11.7.1),
 * the point option "." if a fraction is present (11.7.4), and no trailing zero in
 * the fraction, nor a bare decimal point (11.7.3). */
static int canonical_generalizedtime(const uint8_t *p, size_t n) {
  size_t i;
  if (n < 15u || p[n - 1u] != 'Z') return 0;
  if (!all_digits(p, 14u)) return 0;
  if (n == 15u) return 1;                       /* no fractional part */
  if (p[14] != '.') return 0;                   /* 11.7.4: the point option */
  if (n < 17u) return 0;                        /* 11.7.3: no bare decimal point */
  for (i = 15u; i + 1u < n; ++i) if (p[i] < '0' || p[i] > '9') return 0;
  if (p[n - 2u] == '0') return 0;               /* 11.7.3: no trailing zeros */
  return 1;
}

bcir_asn1_status bcir_asn1_check_der_node(const bcir_asn1_tlv *tlv) {
  if (!tlv) return BCIR_ASN1_ERR_INVALID;
  if (tlv->indefinite) return BCIR_ASN1_ERR_NOT_DER;          /* 10.1 */
  if (tlv->non_minimal_length) return BCIR_ASN1_ERR_NOT_DER;  /* 10.1 */
  if (tlv->cls != BCIR_ASN1_UNIVERSAL) return BCIR_ASN1_OK;

  /* 10.2 applies to bitstring, octetstring, and EVERY restricted character string
   * type of X.680 41 -- the list must match the Python rail's exactly, or a peer's
   * artifact passes one rail's DER check and fails the other's. */
  switch (tlv->number) {
    case BCIR_ASN1_U_BIT_STRING:
    case BCIR_ASN1_U_OCTET_STRING:
    case BCIR_ASN1_U_OBJECT_DESCRIPTOR:
    case BCIR_ASN1_U_UTF8_STRING:
    case BCIR_ASN1_U_NUMERIC_STRING:
    case BCIR_ASN1_U_PRINTABLE_STRING:
    case BCIR_ASN1_U_TELETEX_STRING:
    case BCIR_ASN1_U_VIDEOTEX_STRING:
    case BCIR_ASN1_U_IA5_STRING:
    case BCIR_ASN1_U_GRAPHIC_STRING:
    case BCIR_ASN1_U_VISIBLE_STRING:
    case BCIR_ASN1_U_GENERAL_STRING:
    case BCIR_ASN1_U_UNIVERSAL_STRING:
    case BCIR_ASN1_U_BMP_STRING:
      if (tlv->constructed) return BCIR_ASN1_ERR_NOT_DER;     /* 10.2 */
      break;
    default:
      break;
  }
  /* Scope: clauses 10 and 11 ONLY. Whether the contents octets are a well-formed
   * value of the type is clause 8's question, and it is answered by the value
   * decoders below (bcir_asn1_boolean and friends) when a caller interprets the
   * value. Folding clause-8 checks in here would make this function reject
   * encodings the Python rail's der_violations() accepts -- the two rails must
   * partition the work identically or a peer's artifact passes one and fails the
   * other. A malformed BOOLEAN is not "not DER", it is not a BOOLEAN. */

  /* 11.1: boolean TRUE has all eight bits set. */
  if (tlv->number == BCIR_ASN1_U_BOOLEAN && !tlv->constructed &&
      tlv->content_len == 1u) {
    if (tlv->content[0] != 0x00u && tlv->content[0] != 0xFFu)
      return BCIR_ASN1_ERR_NOT_DER;
  }
  /* 11.2.1: every unused bit in a bitstring's final octet is zero. */
  if (tlv->number == BCIR_ASN1_U_BIT_STRING && !tlv->constructed &&
      tlv->content_len > 1u) {
    uint8_t unused = tlv->content[0];
    if (unused <= 7u && unused > 0u) {
      uint8_t mask = (uint8_t)((1u << unused) - 1u);
      if (tlv->content[tlv->content_len - 1u] & mask) return BCIR_ASN1_ERR_NOT_DER;
    }
  }
  /* 11.7 / 11.8: the canonical time spellings. A non-canonical GeneralizedTime is
   * still a legal BER GeneralizedTime, so this is a clause-11 restriction and not a
   * value fault -- which is why it lives here and not in a value decoder. */
  if (tlv->number == BCIR_ASN1_U_UTC_TIME && !tlv->constructed)
    return canonical_utctime(tlv->content, tlv->content_len)
        ? BCIR_ASN1_OK : BCIR_ASN1_ERR_NOT_DER;
  if (tlv->number == BCIR_ASN1_U_GENERALIZED_TIME && !tlv->constructed)
    return canonical_generalizedtime(tlv->content, tlv->content_len)
        ? BCIR_ASN1_OK : BCIR_ASN1_ERR_NOT_DER;
  return BCIR_ASN1_OK;
}

/* --- contents-octet decoders ------------------------------------------------ */

bcir_asn1_status bcir_asn1_boolean(const bcir_asn1_tlv *tlv, int der, int *out) {
  if (!tlv || !out) return BCIR_ASN1_ERR_INVALID;
  if (tlv->constructed) return BCIR_ASN1_ERR_FORM;            /* 8.2.1 */
  if (tlv->content_len != 1u) return BCIR_ASN1_ERR_VALUE;
  if (der && tlv->content[0] != 0x00u && tlv->content[0] != 0xFFu)
    return BCIR_ASN1_ERR_NOT_DER;                             /* 11.1 */
  *out = tlv->content[0] != 0u;
  return BCIR_ASN1_OK;
}

bcir_asn1_status bcir_asn1_integer(const bcir_asn1_tlv *tlv, int64_t *out) {
  size_t i;
  uint64_t bits;
  if (!tlv || !out) return BCIR_ASN1_ERR_INVALID;
  if (tlv->constructed) return BCIR_ASN1_ERR_FORM;            /* 8.3.1 */
  if (tlv->content_len == 0u) return BCIR_ASN1_ERR_VALUE;
  /* 8.3.2: the first octet and bit 8 of the second shall not all be ones nor all
   * zero -- the rule that makes the encoding minimal, and a BER rule, not a DER
   * one, so it is enforced on every rail. */
  if (tlv->content_len > 1u) {
    uint8_t a = tlv->content[0], b = tlv->content[1];
    if (a == 0x00u && (b & 0x80u) == 0u) return BCIR_ASN1_ERR_VALUE;
    if (a == 0xFFu && (b & 0x80u) != 0u) return BCIR_ASN1_ERR_VALUE;
  }
  /* Refuse rather than truncate: a 9-octet integer is a real value this API
   * cannot represent, and silently narrowing it would corrupt a plan. */
  if (tlv->content_len > 8u) return BCIR_ASN1_ERR_VALUE;

  bits = (tlv->content[0] & 0x80u) ? ~(uint64_t)0 : (uint64_t)0;  /* sign-extend */
  for (i = 0; i < tlv->content_len; ++i) bits = (bits << 8) | (uint64_t)tlv->content[i];
  /* Reinterpret the two's-complement pattern without the conversion C11/C17 leave
   * implementation-defined for an out-of-range uint64_t (see bcir_binrec.c). */
  *out = (bits <= (uint64_t)INT64_MAX) ? (int64_t)bits
                                       : -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
  return BCIR_ASN1_OK;
}

bcir_asn1_status bcir_asn1_uinteger(const bcir_asn1_tlv *tlv, uint64_t *out) {
  int64_t signed_value;
  bcir_asn1_status status = bcir_asn1_integer(tlv, &signed_value);
  if (status != BCIR_ASN1_OK) return status;
  if (signed_value < 0) return BCIR_ASN1_ERR_VALUE;
  *out = (uint64_t)signed_value;
  return BCIR_ASN1_OK;
}

bcir_asn1_status bcir_asn1_oid_arcs(const bcir_asn1_tlv *tlv, uint32_t *arcs,
                                    size_t cap, size_t *count) {
  size_t i, n = 0u;
  uint64_t value = 0u;
  int started = 0;

  if (!tlv || !arcs || !count || cap < 2u) return BCIR_ASN1_ERR_INVALID;
  if (tlv->constructed) return BCIR_ASN1_ERR_FORM;            /* 8.19.1 */
  if (tlv->content_len == 0u) return BCIR_ASN1_ERR_VALUE;
  *count = 0u;

  for (i = 0; i < tlv->content_len; ++i) {
    uint8_t octet = tlv->content[i];
    /* 8.19.2: "the leading octet of the subidentifier shall not have the value 80". */
    if (!started && octet == 0x80u) return BCIR_ASN1_ERR_VALUE;
    started = 1;
    if (value > (UINT64_MAX >> 7)) return BCIR_ASN1_ERR_VALUE;
    value = (value << 7) | (uint64_t)(octet & 0x7Fu);
    if ((octet & 0x80u) != 0u) continue;
    if (n == 0u) {                        /* 8.19.4: the first two components pack */
      uint64_t first = value < 40u ? 0u : (value < 80u ? 1u : 2u);
      uint64_t second = value - first * 40u;
      if (second > UINT32_MAX) return BCIR_ASN1_ERR_VALUE;
      arcs[0] = (uint32_t)first;
      arcs[1] = (uint32_t)second;
      n = 2u;
    } else {
      if (n >= cap) return BCIR_ASN1_ERR_INVALID;
      if (value > UINT32_MAX) return BCIR_ASN1_ERR_VALUE;
      arcs[n++] = (uint32_t)value;
    }
    value = 0u;
    started = 0;
  }
  if (started) return BCIR_ASN1_ERR_VALUE;   /* ended mid-subidentifier */
  *count = n;
  return BCIR_ASN1_OK;
}

const char *bcir_asn1_status_name(bcir_asn1_status status) {
  switch (status) {
    case BCIR_ASN1_OK: return "ok";
    case BCIR_ASN1_END: return "end";
    case BCIR_ASN1_ERR_TRUNCATED: return "truncated";
    case BCIR_ASN1_ERR_TAG: return "malformed-tag";
    case BCIR_ASN1_ERR_LENGTH: return "malformed-length";
    case BCIR_ASN1_ERR_EOC: return "end-of-contents";
    case BCIR_ASN1_ERR_FORM: return "illegal-form";
    case BCIR_ASN1_ERR_DEPTH: return "nesting-too-deep";
    case BCIR_ASN1_ERR_NOT_DER: return "not-der";
    case BCIR_ASN1_ERR_VALUE: return "invalid-value";
    case BCIR_ASN1_ERR_TRAILING: return "trailing-octets";
    case BCIR_ASN1_ERR_INVALID: return "invalid-argument";
    default: return "unknown";
  }
}
