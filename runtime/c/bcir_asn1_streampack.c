/*===- bcir_asn1_streampack.c - DER -> native StreamPack fast path --------===
 *
 * No libc: only <stddef.h>/<stdint.h> (via bcir_runtime.h). Mirrors, in the reverse
 * direction, bcir/asn1/streampack.py::value_to_pack composed with bcir/abi::encode.
 *
 * HOW THE PROJECTION IS READ. The BCIR-StreamPack module is an IMPLICIT TAGS module in
 * which EVERY component carries a context-specific tag, so a component is found by its
 * tag number rather than by its position. That is what makes a schema-free C decoder
 * possible here: there is no need for a type model at run time, only the tag -> field
 * table below. An absent component is not an error -- X.690 11.5 omits any component
 * equal to its DEFAULT, so absence MEANS the default, and the defaults are written out
 * explicitly because the native format has no notion of an omitted field.
 *
 * WHY TWO PASSES. The native header carries the record counts and the format version,
 * and the version is derived from CONTENT (v2 if any pipelining/double-buffering, v3 if
 * any non-default dispatch/channel) exactly as bcir/abi::encode derives it. The version
 * changes the BODY layout, so it has to be known before the first body octet is written.
 * Pass one counts and classifies; pass two writes. Both passes are bounded walks over
 * the caller's buffer -- no allocation, no recursion.
 *===----------------------------------------------------------------------===*/
#include "bcir_asn1_streampack.h"

#define SP_HDR_SIZE      ((size_t)BCIR_STREAMPACK_HEADER_SIZE)
#define SP_PIPELINE_OFF  ((size_t)36)   /* "<4sHHIIIIIII" -- the v2 u16 goes right after */

/* The native defaults the projection omits (X.690 11.5). Spelled here because the C rail
 * must reproduce the same octets the Python encoder writes unconditionally. */
static const char SP_CHANNEL_DEFAULT[] = "host";
static const char SP_HINT_DEFAULT[] = "T0";
static const char SP_PATTERN_DEFAULT[] = "linear";

/* --- bounded writer ------------------------------------------------------------- */

typedef struct {
  uint8_t *d;
  size_t cap;
  size_t pos;
  int err;
} wcur;

static int w_has(const wcur *w, size_t n) {
  return !w->err && w->pos <= w->cap && n <= w->cap - w->pos;
}
static void w_u8(wcur *w, uint8_t v) {
  if (!w_has(w, 1)) { w->err = 1; return; }
  w->d[w->pos++] = v;
}
static void w_u16(wcur *w, uint16_t v) {
  if (!w_has(w, 2)) { w->err = 1; return; }
  w->d[w->pos++] = (uint8_t)(v & 0xFFu);
  w->d[w->pos++] = (uint8_t)((v >> 8) & 0xFFu);
}
static void w_u32(wcur *w, uint32_t v) {
  if (!w_has(w, 4)) { w->err = 1; return; }
  for (int i = 0; i < 4; i++) w->d[w->pos++] = (uint8_t)((v >> (8 * i)) & 0xFFu);
}
static void w_u64(wcur *w, uint64_t v) {
  if (!w_has(w, 8)) { w->err = 1; return; }
  for (int i = 0; i < 8; i++) w->d[w->pos++] = (uint8_t)((v >> (8 * i)) & 0xFFu);
}
static void w_bytes(wcur *w, const uint8_t *p, size_t n) {
  if ((!p && n) || !w_has(w, n)) { w->err = 1; return; }
  for (size_t i = 0; i < n; i++) w->d[w->pos++] = p[i];
}
/* A native length-prefixed string: u16 count + octets. */
static void w_str(wcur *w, const uint8_t *p, size_t n) {
  if (n > 0xFFFFu) { w->err = 1; return; }
  w_u16(w, (uint16_t)n);
  w_bytes(w, p, n);
}
static void w_cstr(wcur *w, const char *s, size_t n) {
  w_str(w, (const uint8_t *)s, n);
}

/* --- projection navigation -------------------------------------------------------- */

/* Find the child of `parent` carrying context-specific tag `tag`.
 * Sets *found to 0 when the component is absent -- which is the DEFAULT case, not a
 * fault, so the caller supplies the default rather than failing. */
static bcir_asn1_status sp_find(const uint8_t *der, size_t len,
                                const bcir_asn1_tlv *parent, uint32_t tag,
                                bcir_asn1_tlv *out, int *found) {
  bcir_asn1_tlv child;
  bcir_asn1_status st = bcir_asn1_first_child(der, len, parent, &child);
  *found = 0;
  while (st == BCIR_ASN1_OK) {
    if (child.cls == BCIR_ASN1_CONTEXT && child.number == tag) {
      *out = child;
      *found = 1;
      return BCIR_ASN1_OK;
    }
    st = bcir_asn1_next(der, len, parent, &child);
  }
  return (st == BCIR_ASN1_END) ? BCIR_ASN1_OK : st;
}

/* Count the children of a constructed node. */
static bcir_asn1_status sp_count(const uint8_t *der, size_t len,
                                 const bcir_asn1_tlv *parent, uint32_t *out) {
  bcir_asn1_tlv child;
  uint32_t n = 0;
  bcir_asn1_status st = bcir_asn1_first_child(der, len, parent, &child);
  while (st == BCIR_ASN1_OK) {
    if (n == 0xFFFFFFFFu) return BCIR_ASN1_ERR_VALUE;
    n++;
    st = bcir_asn1_next(der, len, parent, &child);
  }
  if (st != BCIR_ASN1_END) return st;
  *out = n;
  return BCIR_ASN1_OK;
}

/* An unsigned component, or `fallback` when the component is absent (its DEFAULT). */
static bcir_asn1_status sp_uint_or(const uint8_t *der, size_t len,
                                   const bcir_asn1_tlv *parent, uint32_t tag,
                                   uint64_t fallback, uint64_t *out) {
  bcir_asn1_tlv node;
  int found;
  bcir_asn1_status st = sp_find(der, len, parent, tag, &node, &found);
  if (st != BCIR_ASN1_OK) return st;
  if (!found) { *out = fallback; return BCIR_ASN1_OK; }
  return bcir_asn1_uinteger(&node, out);
}

/* Map an ASN.1 status onto the runtime's.
 *
 * BCIR_ASN1_ERR_VALUE is kept distinct: this path raises it for a value the projection
 * states legally but the NATIVE ABI cannot represent (a claim count past 2^16, a width
 * past 2^32). That is an overflow of the target format, not a malformed input, and
 * reporting it as TRUNCATED would send a caller looking for a corrupt buffer that is in
 * fact perfectly well formed. Every other fault is a malformed projection. */
static bcir_status sp_status(bcir_asn1_status st) {
  switch (st) {
    case BCIR_ASN1_OK:        return BCIR_OK;
    case BCIR_ASN1_ERR_VALUE: return BCIR_ERR_OVERFLOW;
    default:                  return BCIR_ERR_TRUNCATED;
  }
}

size_t bcir_asn1_streampack_bound(size_t der_len) {
  /* Every native field is at most its DER form plus a fixed expansion: an omitted
   * DEFAULT costs zero DER octets but up to 8 native ones (a u64), and each record adds
   * the fixed-width fields the projection may omit entirely. Four times the projection
   * plus the header, the CRC, and a fixed floor covers every shape the module admits. */
  size_t bound = der_len * 4u + SP_HDR_SIZE + 4u + 256u;
  return (bound < der_len) ? (size_t)-1 : bound;         /* saturate rather than wrap */
}

/* --- pass one: version + counts ---------------------------------------------------- */

typedef struct {
  uint32_t n_segments, n_prefetches, n_blocks, n_trace;
  uint16_t version;
  uint16_t pipeline_depth;
  bcir_asn1_tlv segments, prefetches, blocks, trace;
  int has_segments, has_prefetches, has_blocks, has_trace;
} sp_shape;

static bcir_asn1_status sp_classify(const uint8_t *der, size_t len,
                                    const bcir_asn1_tlv *root, sp_shape *shape) {
  bcir_asn1_status st;
  uint64_t depth = 1;
  int needs_v2 = 0, needs_v3 = 0;

  st = sp_uint_or(der, len, root, 5, 1, &depth);          /* pipelineDepth [5] */
  if (st != BCIR_ASN1_OK) return st;
  if (depth > 0xFFFFu) return BCIR_ASN1_ERR_VALUE;
  shape->pipeline_depth = (uint16_t)depth;
  if (depth > 1) needs_v2 = 1;

  st = sp_find(der, len, root, 6, &shape->segments, &shape->has_segments);
  if (st != BCIR_ASN1_OK) return st;
  st = sp_find(der, len, root, 7, &shape->prefetches, &shape->has_prefetches);
  if (st != BCIR_ASN1_OK) return st;
  st = sp_find(der, len, root, 8, &shape->blocks, &shape->has_blocks);
  if (st != BCIR_ASN1_OK) return st;
  st = sp_find(der, len, root, 9, &shape->trace, &shape->has_trace);
  if (st != BCIR_ASN1_OK) return st;

  shape->n_segments = shape->n_prefetches = shape->n_blocks = shape->n_trace = 0;
  if (shape->has_segments) {
    st = sp_count(der, len, &shape->segments, &shape->n_segments);
    if (st != BCIR_ASN1_OK) return st;
  }
  if (shape->has_prefetches) {
    st = sp_count(der, len, &shape->prefetches, &shape->n_prefetches);
    if (st != BCIR_ASN1_OK) return st;
  }
  if (shape->has_blocks) {
    st = sp_count(der, len, &shape->blocks, &shape->n_blocks);
    if (st != BCIR_ASN1_OK) return st;
  }
  if (shape->has_trace) {
    st = sp_count(der, len, &shape->trace, &shape->n_trace);
    if (st != BCIR_ASN1_OK) return st;
  }

  /* v3 iff any segment carries a non-default dispatch or channel; v2 iff any prefetch
   * is double-buffered. Both mirror bcir/abi::encode's `needs_v2` / `needs_v3`. */
  if (shape->has_segments) {
    bcir_asn1_tlv seg;
    st = bcir_asn1_first_child(der, len, &shape->segments, &seg);
    while (st == BCIR_ASN1_OK) {
      bcir_asn1_tlv node;
      int found;
      uint64_t dispatch = 0;
      bcir_asn1_status inner = sp_uint_or(der, len, &seg, 11, 0, &dispatch);
      if (inner != BCIR_ASN1_OK) return inner;
      if (dispatch != 0) needs_v3 = 1;
      inner = sp_find(der, len, &seg, 12, &node, &found);  /* channel [12] */
      if (inner != BCIR_ASN1_OK) return inner;
      if (found) needs_v3 = 1;   /* present => it differs from the DEFAULT (X.690 11.5) */
      st = bcir_asn1_next(der, len, &shape->segments, &seg);
    }
    if (st != BCIR_ASN1_END) return st;
  }
  if (shape->has_prefetches) {
    bcir_asn1_tlv pf;
    st = bcir_asn1_first_child(der, len, &shape->prefetches, &pf);
    while (st == BCIR_ASN1_OK) {
      uint64_t buffers = 1;
      bcir_asn1_status inner = sp_uint_or(der, len, &pf, 5, 1, &buffers);
      if (inner != BCIR_ASN1_OK) return inner;
      if (buffers != 1) needs_v2 = 1;
      st = bcir_asn1_next(der, len, &shape->prefetches, &pf);
    }
    if (st != BCIR_ASN1_END) return st;
  }

  shape->version = needs_v3 ? 3u : (needs_v2 ? 2u : 1u);
  return BCIR_ASN1_OK;
}

/* --- pass two: write the native artifact -------------------------------------------- */

/* Write a component's octets as a native string, or its default when absent. */
static bcir_asn1_status sp_write_str_or(const uint8_t *der, size_t len,
                                        const bcir_asn1_tlv *parent, uint32_t tag,
                                        const char *fallback, size_t fallback_len,
                                        wcur *w) {
  bcir_asn1_tlv node;
  int found;
  bcir_asn1_status st = sp_find(der, len, parent, tag, &node, &found);
  if (st != BCIR_ASN1_OK) return st;
  if (found)
    w_str(w, node.content, node.content_len);
  else
    w_cstr(w, fallback, fallback_len);
  return BCIR_ASN1_OK;
}

/* Write a SEQUENCE OF INTEGER as a native u32 array (u16 count + count*u32). */
static bcir_asn1_status sp_write_u32arr(const uint8_t *der, size_t len,
                                        const bcir_asn1_tlv *parent, uint32_t tag,
                                        wcur *w) {
  bcir_asn1_tlv node, item;
  int found;
  uint32_t count = 0;
  bcir_asn1_status st = sp_find(der, len, parent, tag, &node, &found);
  if (st != BCIR_ASN1_OK) return st;
  if (!found) { w_u16(w, 0); return BCIR_ASN1_OK; }
  st = sp_count(der, len, &node, &count);
  if (st != BCIR_ASN1_OK) return st;
  if (count > 0xFFFFu) return BCIR_ASN1_ERR_VALUE;
  w_u16(w, (uint16_t)count);
  st = bcir_asn1_first_child(der, len, &node, &item);
  while (st == BCIR_ASN1_OK) {
    uint64_t value = 0;
    bcir_asn1_status inner = bcir_asn1_uinteger(&item, &value);
    if (inner != BCIR_ASN1_OK) return inner;
    if (value > 0xFFFFFFFFu) return BCIR_ASN1_ERR_VALUE;
    w_u32(w, (uint32_t)value);
    st = bcir_asn1_next(der, len, &node, &item);
  }
  return (st == BCIR_ASN1_END) ? BCIR_ASN1_OK : st;
}

/* Write a SEQUENCE OF INTEGER as a native u64 array, defaulting to `{ 1 }` when the
 * component is absent -- Block.strides is the one component whose DEFAULT is non-empty. */
static bcir_asn1_status sp_write_u64arr_or_one(const uint8_t *der, size_t len,
                                               const bcir_asn1_tlv *parent, uint32_t tag,
                                               wcur *w) {
  bcir_asn1_tlv node, item;
  int found;
  uint32_t count = 0;
  bcir_asn1_status st = sp_find(der, len, parent, tag, &node, &found);
  if (st != BCIR_ASN1_OK) return st;
  if (!found) { w_u16(w, 1); w_u64(w, 1); return BCIR_ASN1_OK; }
  st = sp_count(der, len, &node, &count);
  if (st != BCIR_ASN1_OK) return st;
  if (count > 0xFFFFu) return BCIR_ASN1_ERR_VALUE;
  w_u16(w, (uint16_t)count);
  st = bcir_asn1_first_child(der, len, &node, &item);
  while (st == BCIR_ASN1_OK) {
    uint64_t value = 0;
    bcir_asn1_status inner = bcir_asn1_uinteger(&item, &value);
    if (inner != BCIR_ASN1_OK) return inner;
    w_u64(w, value);
    st = bcir_asn1_next(der, len, &node, &item);
  }
  return (st == BCIR_ASN1_END) ? BCIR_ASN1_OK : st;
}

/* Write a SEQUENCE OF UTF8String as a native string array (u16 count + strings). */
static bcir_asn1_status sp_write_strarr(const uint8_t *der, size_t len,
                                        const bcir_asn1_tlv *parent, uint32_t tag,
                                        wcur *w) {
  bcir_asn1_tlv node, item;
  int found;
  uint32_t count = 0;
  bcir_asn1_status st = sp_find(der, len, parent, tag, &node, &found);
  if (st != BCIR_ASN1_OK) return st;
  if (!found) { w_u16(w, 0); return BCIR_ASN1_OK; }
  st = sp_count(der, len, &node, &count);
  if (st != BCIR_ASN1_OK) return st;
  if (count > 0xFFFFu) return BCIR_ASN1_ERR_VALUE;
  w_u16(w, (uint16_t)count);
  st = bcir_asn1_first_child(der, len, &node, &item);
  while (st == BCIR_ASN1_OK) {
    w_str(w, item.content, item.content_len);
    st = bcir_asn1_next(der, len, &node, &item);
  }
  return (st == BCIR_ASN1_END) ? BCIR_ASN1_OK : st;
}

/* A mandatory unsigned component -- absence is a malformed projection, not a default. */
static bcir_asn1_status sp_uint_req(const uint8_t *der, size_t len,
                                    const bcir_asn1_tlv *parent, uint32_t tag,
                                    uint64_t *out) {
  bcir_asn1_tlv node;
  int found;
  bcir_asn1_status st = sp_find(der, len, parent, tag, &node, &found);
  if (st != BCIR_ASN1_OK) return st;
  if (!found) return BCIR_ASN1_ERR_TRUNCATED;
  return bcir_asn1_uinteger(&node, out);
}

static bcir_asn1_status sp_write_body(const uint8_t *der, size_t len,
                                      const bcir_asn1_tlv *root, const sp_shape *shape,
                                      wcur *w) {
  bcir_asn1_status st;
  bcir_asn1_tlv node;
  int found;

  st = sp_find(der, len, root, 1, &node, &found);          /* sourcePlan [1], mandatory */
  if (st != BCIR_ASN1_OK) return st;
  if (!found) return BCIR_ASN1_ERR_TRUNCATED;
  w_str(w, node.content, node.content_len);

  if (shape->has_segments) {
    bcir_asn1_tlv seg;
    st = bcir_asn1_first_child(der, len, &shape->segments, &seg);
    while (st == BCIR_ASN1_OK) {
      uint64_t value = 0;
      st = sp_find(der, len, &seg, 0, &node, &found);      /* name [0] */
      if (st != BCIR_ASN1_OK) return st;
      if (!found) return BCIR_ASN1_ERR_TRUNCATED;
      w_str(w, node.content, node.content_len);

      st = sp_uint_req(der, len, &seg, 1, &value); if (st) return st;  /* claimId  */
      w_u64(w, value);
      st = sp_uint_req(der, len, &seg, 2, &value); if (st) return st;  /* phaseId  */
      if (value > 0xFFFFFFFFu) return BCIR_ASN1_ERR_VALUE;
      w_u32(w, (uint32_t)value);
      st = sp_uint_req(der, len, &seg, 3, &value); if (st) return st;  /* lane     */
      if (value > 0xFFu) return BCIR_ASN1_ERR_VALUE;
      w_u8(w, (uint8_t)value);
      st = sp_uint_req(der, len, &seg, 4, &value); if (st) return st;  /* width    */
      if (value > 0xFFFFFFFFu) return BCIR_ASN1_ERR_VALUE;
      w_u32(w, (uint32_t)value);
      /* stride_k: reserved, constant zero, and deliberately NOT projected -- the native
       * segment carries stride per claim (docs/BCIR_ASN1_X690_ABI.md 3). */
      w_u32(w, 0);

      st = sp_find(der, len, &seg, 5, &node, &found);      /* opcode [5] */
      if (st != BCIR_ASN1_OK) return st;
      if (!found) return BCIR_ASN1_ERR_TRUNCATED;
      w_str(w, node.content, node.content_len);

      st = sp_write_u32arr(der, len, &seg, 6, w); if (st) return st;   /* reads    */
      st = sp_write_u32arr(der, len, &seg, 7, w); if (st) return st;   /* writes   */
      st = sp_write_str_or(der, len, &seg, 8, "", 0, w); if (st) return st; /* prefetch */
      st = sp_write_strarr(der, len, &seg, 9, w); if (st) return st;   /* fenceBefore */
      st = sp_write_strarr(der, len, &seg, 10, w); if (st) return st;  /* fenceAfter  */
      if (shape->version >= 3) {
        st = sp_uint_or(der, len, &seg, 11, 0, &value); if (st) return st;  /* dispatch */
        if (value > 0xFFu) return BCIR_ASN1_ERR_VALUE;
        w_u8(w, (uint8_t)value);
        st = sp_write_str_or(der, len, &seg, 12, SP_CHANNEL_DEFAULT,
                             sizeof SP_CHANNEL_DEFAULT - 1, w);
        if (st) return st;
      }
      st = bcir_asn1_next(der, len, &shape->segments, &seg);
    }
    if (st != BCIR_ASN1_END) return st;
  }

  if (shape->has_prefetches) {
    bcir_asn1_tlv pf;
    st = bcir_asn1_first_child(der, len, &shape->prefetches, &pf);
    while (st == BCIR_ASN1_OK) {
      uint64_t value = 0;
      st = sp_find(der, len, &pf, 0, &node, &found);       /* name [0] */
      if (st != BCIR_ASN1_OK) return st;
      if (!found) return BCIR_ASN1_ERR_TRUNCATED;
      w_str(w, node.content, node.content_len);
      st = sp_uint_req(der, len, &pf, 1, &value); if (st) return st;   /* distance */
      if (value > 0xFFFFFFFFu) return BCIR_ASN1_ERR_VALUE;
      w_u32(w, (uint32_t)value);
      st = sp_write_u32arr(der, len, &pf, 2, w); if (st) return st;    /* targets  */
      st = sp_write_str_or(der, len, &pf, 3, SP_HINT_DEFAULT,
                           sizeof SP_HINT_DEFAULT - 1, w); if (st) return st;
      st = sp_write_str_or(der, len, &pf, 4, SP_PATTERN_DEFAULT,
                           sizeof SP_PATTERN_DEFAULT - 1, w); if (st) return st;
      if (shape->version >= 2) {
        st = sp_uint_or(der, len, &pf, 5, 1, &value); if (st) return st; /* buffers */
        if (value > 0xFFu) return BCIR_ASN1_ERR_VALUE;
        w_u8(w, (uint8_t)value);
      }
      st = bcir_asn1_next(der, len, &shape->prefetches, &pf);
    }
    if (st != BCIR_ASN1_END) return st;
  }

  if (shape->has_blocks) {
    bcir_asn1_tlv blk;
    st = bcir_asn1_first_child(der, len, &shape->blocks, &blk);
    while (st == BCIR_ASN1_OK) {
      uint64_t value = 0;
      st = sp_uint_req(der, len, &blk, 0, &value); if (st) return st;  /* base  */
      w_u64(w, value);
      st = sp_uint_req(der, len, &blk, 1, &value); if (st) return st;  /* count */
      w_u64(w, value);
      st = sp_write_u64arr_or_one(der, len, &blk, 2, w); if (st) return st;
      st = bcir_asn1_next(der, len, &shape->blocks, &blk);
    }
    if (st != BCIR_ASN1_END) return st;
  }

  if (shape->has_trace) {
    bcir_asn1_tlv tr;
    st = bcir_asn1_first_child(der, len, &shape->trace, &tr);
    while (st == BCIR_ASN1_OK) {
      uint64_t value = 0;
      st = sp_uint_req(der, len, &tr, 0, &value); if (st) return st;   /* claimId    */
      w_u64(w, value);
      st = sp_uint_or(der, len, &tr, 1, 0, &value); if (st) return st; /* srcHash    */
      w_u64(w, value);
      st = sp_uint_or(der, len, &tr, 2, 0, &value); if (st) return st; /* traceHash  */
      w_u64(w, value);
      st = bcir_asn1_next(der, len, &shape->trace, &tr);
    }
    if (st != BCIR_ASN1_END) return st;
  }
  return BCIR_ASN1_OK;
}

bcir_status bcir_asn1_to_streampack(const uint8_t *BCIR_RESTRICT der, size_t der_len,
                                    uint8_t *BCIR_RESTRICT out, size_t out_cap,
                                    size_t *BCIR_RESTRICT out_len) {
  bcir_asn1_tlv root;
  sp_shape shape;
  bcir_asn1_status ast;
  uint64_t topo = 1, mapg = 0, datag = 0;

  if (out_len) *out_len = 0;
  if (!der || !out) return BCIR_ERR_NOSPACE;

  /* The projection is DER, not merely BER: BCIR digests what it exchanges, so a peer
   * must not get to choose the octets. Reject a BER-only spelling here rather than
   * silently normalizing it -- normalizing would change the digest. */
  ast = bcir_asn1_validate_der(der, der_len, 64u);
  if (ast != BCIR_ASN1_OK) return sp_status(ast);
  ast = bcir_asn1_decode_exact(der, der_len, &root);
  if (ast != BCIR_ASN1_OK) return sp_status(ast);
  if (root.cls != BCIR_ASN1_UNIVERSAL ||
      root.number != BCIR_ASN1_U_SEQUENCE || !root.constructed)
    return BCIR_ERR_TRUNCATED;

  ast = sp_classify(der, der_len, &root, &shape);
  if (ast != BCIR_ASN1_OK) return sp_status(ast);

  ast = sp_uint_or(der, der_len, &root, 2, 1, &topo);      /* topoGen [2] DEFAULT 1 */
  if (ast != BCIR_ASN1_OK) return sp_status(ast);
  ast = sp_uint_or(der, der_len, &root, 3, 0, &mapg);      /* mapGen  [3] DEFAULT 0 */
  if (ast != BCIR_ASN1_OK) return sp_status(ast);
  ast = sp_uint_or(der, der_len, &root, 4, 0, &datag);     /* dataGen [4] DEFAULT 0 */
  if (ast != BCIR_ASN1_OK) return sp_status(ast);
  if (topo > 0xFFFFFFFFu || mapg > 0xFFFFFFFFu || datag > 0xFFFFFFFFu)
    return BCIR_ERR_NOSPACE;

  wcur w = {out, out_cap, 0, 0};

  /* Header: magic, version, flags, three generations, four counts; then the v2
   * pipeline_depth; then zero padding to the frozen 64-octet size. */
  w_bytes(&w, (const uint8_t *)"BSPK", 4);
  w_u16(&w, shape.version);
  w_u16(&w, 0);                                            /* flags */
  w_u32(&w, (uint32_t)topo);
  w_u32(&w, (uint32_t)mapg);
  w_u32(&w, (uint32_t)datag);
  w_u32(&w, shape.n_segments);
  w_u32(&w, shape.n_prefetches);
  w_u32(&w, shape.n_blocks);
  w_u32(&w, shape.n_trace);
  if (shape.version >= 2) {
    if (w.pos != SP_PIPELINE_OFF) return BCIR_ERR_NOSPACE;  /* layout guard */
    w_u16(&w, shape.pipeline_depth);
  }
  while (!w.err && w.pos < SP_HDR_SIZE) w_u8(&w, 0);
  if (w.err) return BCIR_ERR_NOSPACE;

  ast = sp_write_body(der, der_len, &root, &shape, &w);
  if (ast != BCIR_ASN1_OK) return sp_status(ast);
  if (w.err) return BCIR_ERR_NOSPACE;

  w_u32(&w, bcir_crc32(out, w.pos));
  if (w.err) return BCIR_ERR_NOSPACE;

  /* A reconstruction that is well-formed but not EXECUTABLE must not be blessed: the
   * same semantic gate the native decoder applies (R10 provenance, R11 range) runs over
   * the artifact before it is handed back, so this path cannot mint a pack that
   * `bcir_sp_execute_checked` would refuse. */
  bcir_status sem = bcir_sp_verify_semantic(out, w.pos, UINT32_MAX, UINT32_MAX);
  if (sem != BCIR_OK) return sem;

  if (out_len) *out_len = w.pos;
  return BCIR_OK;
}
