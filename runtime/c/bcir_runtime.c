/*===- bcir_runtime.c - freestanding BCIR StreamPack runtime --------------===
 *
 * No libc: only <stddef.h>/<stdint.h>. Little-endian wire decoding done byte by
 * byte (host-endian-independent). Matches bcir/abi/streampack_abi.py.
 *===----------------------------------------------------------------------===*/
#include "bcir_runtime.h"

/* --- zlib-compatible CRC-32 (bitwise; no table, freestanding) --- */
uint32_t bcir_crc32(const uint8_t *BCIR_RESTRICT data, size_t len) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++) {
    c ^= data[i];
    for (int k = 0; k < 8; k++)
      c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1u)));
  }
  return c ^ 0xFFFFFFFFu;
}

/* The default heterogeneous channel for a v1/v2 segment (no channel on the wire). The C
 * decoder defaults it here so a v1/v2 segment view reads "host" -- byte-identical to the
 * Python decoder's _CHANNEL_DEFAULT, closing the decode-rail asymmetry. */
static const char BCIR_DEFAULT_CHANNEL[] = "host";
#define BCIR_DEFAULT_CHANNEL_LEN 4u

/* --- little-endian readers (byte-wise; host-endian-independent) --- */
static uint16_t rd16(const uint8_t *BCIR_RESTRICT p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t rd32(const uint8_t *BCIR_RESTRICT p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *BCIR_RESTRICT p) {
  return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

/* --- bounded body cursor --- */
typedef struct {
  const uint8_t *d;
  size_t len;   /* logical end (body, i.e. total minus the 4-byte CRC) */
  size_t pos;
  int err;
} cur;

static int c_has(const cur *c, size_t n) {
  return !c->err && c->pos <= c->len && n <= c->len - c->pos;
}
static void c_skip(cur *c, size_t n) {
  if (!c_has(c, n)) c->err = 1; else c->pos += n;
}

static uint8_t c_u8(cur *c) {
  if (!c_has(c, 1)) { c->err = 1; return 0; }
  return c->d[c->pos++];
}
static uint16_t c_u16(cur *c) {
  if (!c_has(c, 2)) { c->err = 1; return 0; }
  uint16_t v = rd16(c->d + c->pos); c->pos += 2; return v;
}
static uint32_t c_u32(cur *c) {
  if (!c_has(c, 4)) { c->err = 1; return 0; }
  uint32_t v = rd32(c->d + c->pos); c->pos += 4; return v;
}
static uint64_t c_u64(cur *c) {
  if (!c_has(c, 8)) { c->err = 1; return 0; }
  uint64_t v = rd64(c->d + c->pos); c->pos += 8; return v;
}
static const char *c_str(cur *c, uint16_t *out_len) {
  uint16_t n = c_u16(c);
  if (!c_has(c, n)) { c->err = 1; *out_len = 0; return 0; }
  const char *p = (const char *)(c->d + c->pos);
  c->pos += n; *out_len = n; return p;
}
static const uint8_t *c_u32arr(cur *c, uint16_t *out_cnt) {
  uint16_t n = c_u16(c);
  size_t bytes = (size_t)n * 4u;
  if (!c_has(c, bytes)) { c->err = 1; *out_cnt = 0; return 0; }
  const uint8_t *p = c->d + c->pos;
  c->pos += bytes; *out_cnt = n; return p;
}
static void c_skip_str(cur *c) {
  uint16_t n = c_u16(c);
  c_skip(c, n);
}
static void c_skip_strarr(cur *c) {
  uint16_t n = c_u16(c);
  for (uint16_t i = 0; i < n && !c->err; i++) c_skip_str(c);
}

uint32_t bcir_seg_read_rid(const bcir_segment_view *seg, uint16_t i) {
  return (i < seg->n_reads) ? rd32(seg->reads + (size_t)i * 4u) : 0u;
}
uint32_t bcir_seg_write_rid(const bcir_segment_view *seg, uint16_t i) {
  return (i < seg->n_writes) ? rd32(seg->writes + (size_t)i * 4u) : 0u;
}

/* The range gate over a decoded segment view (the rail-symmetry fix): lane in [0,5],
 * width a nonzero power of two, dispatch in the legal set. Mirrors the Python decoder's
 * Lane(r.u8()) raise, the width power-of-two raise, and the unknown-dispatch-code raise.
 * Returns BCIR_OK or BCIR_ERR_*. */
static bcir_status seg_range_ok(const bcir_segment_view *v) {
  if (v->lane > (uint8_t)BCIR_LANE_H) return BCIR_ERR_LANE;   /* 6 valid lanes: 0..5 */
  if (v->width == 0u || (v->width & (v->width - 1u)) != 0u) return BCIR_ERR_WIDTH;
  if (v->dispatch > (uint8_t)BCIR_DISPATCH_MAX) return BCIR_ERR_DISPATCH;
  return BCIR_OK;
}

bcir_status bcir_sp_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                             bcir_streampack_header *BCIR_RESTRICT hdr) {
  if (!data) return BCIR_ERR_TRUNCATED;
  if (len < (size_t)BCIR_STREAMPACK_HEADER_SIZE + 4u) return BCIR_ERR_TRUNCATED;
  if (data[0] != 'B' || data[1] != 'S' || data[2] != 'P' || data[3] != 'K')
    return BCIR_ERR_MAGIC;
  uint16_t version = rd16(data + 4);
  if (version < BCIR_STREAMPACK_VERSION || version > BCIR_STREAMPACK_VERSION_MAX)
    return BCIR_ERR_VERSION;
  uint32_t stored = rd32(data + len - 4);
  if (bcir_crc32(data, len - 4) != stored) return BCIR_ERR_CRC;
  if (hdr) {
    for (int i = 0; i < 4; i++) hdr->magic[i] = data[i];
    hdr->version = version;
    hdr->flags = rd16(data + 6);
    hdr->topo_gen = rd32(data + 8);
    hdr->map_gen = rd32(data + 12);
    hdr->data_gen = rd32(data + 16);
    hdr->n_segments = rd32(data + 20);
    hdr->n_prefetches = rd32(data + 24);
    hdr->n_blocks = rd32(data + 28);
    hdr->n_trace = rd32(data + 32);
    /* v2 appended into the v1 pad; v1 packs are single-phase-in-flight. */
    hdr->pipeline_depth = (version >= 2) ? rd16(data + 36) : 1;
    if (hdr->pipeline_depth == 0) hdr->pipeline_depth = 1;
  }
  return BCIR_OK;
}

bcir_status bcir_sp_for_each_segment(const uint8_t *BCIR_RESTRICT data, size_t len,
                                     bcir_seg_fn fn, void *ctx) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);
  if (st != BCIR_OK) return st;

  cur c;
  c.d = data; c.len = len - 4u; c.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  c_skip_str(&c); /* source_plan */

  for (uint32_t i = 0; i < hdr.n_segments && !c.err; i++) {
    bcir_segment_view v;
    v.name = c_str(&c, &v.name_len);
    v.claim_id = c_u64(&c);
    v.phase_id = c_u32(&c);
    v.lane = c_u8(&c);
    v.width = c_u32(&c);
    v.stride_k = c_u32(&c);
    v.opcode = c_str(&c, &v.opcode_len);
    v.reads = c_u32arr(&c, &v.n_reads);
    v.writes = c_u32arr(&c, &v.n_writes);
    v.prefetch = c_str(&c, &v.prefetch_len);   /* "" (len 0) == no prefetch */
    c_skip_strarr(&c);  /* fence_before */
    c_skip_strarr(&c);  /* fence_after */
    /* v3 (append-only) segment-record tail: dispatch (u8) + channel (str). v1/v2 default. */
    if (hdr.version >= 3) {
      v.dispatch = c_u8(&c);
      v.channel = c_str(&c, &v.channel_len);
    } else {
      v.dispatch = (uint8_t)BCIR_DISPATCH_CORE;
      v.channel = BCIR_DEFAULT_CHANNEL; v.channel_len = BCIR_DEFAULT_CHANNEL_LEN;
    }
    if (c.err) return BCIR_ERR_TRUNCATED;
    /* Range gate: reject an out-of-range lane/width/dispatch (mirror the Python raise),
     * so a CRC-valid pack with a bad lane fails on BOTH rails identically. */
    bcir_status rg = seg_range_ok(&v);
    if (rg != BCIR_OK) return rg;
    if (fn && fn(&v, ctx)) break;
  }
  return c.err ? BCIR_ERR_TRUNCATED : BCIR_OK;
}

bcir_status bcir_sp_check_generation(const uint8_t *BCIR_RESTRICT data, size_t len,
                                     uint32_t expected_map_gen,
                                     uint32_t expected_data_gen) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);
  if (st != BCIR_OK) return st;
  /* R11: a generation mismatch is a STALE pack -- rehydrate, never execute. */
  if (expected_map_gen != 0xFFFFFFFFu && hdr.map_gen != expected_map_gen)
    return BCIR_ERR_STALE;
  if (expected_data_gen != 0xFFFFFFFFu && hdr.data_gen != expected_data_gen)
    return BCIR_ERR_STALE;
  return BCIR_OK;
}

/* --- R10 provenance walk (freestanding; second pass over the decoded body) ---
 * verify_pack's R10 resolves each segment's claim_id against the pack's trace records and
 * each segment's prefetch name against the declared prefetches. The body is a single forward
 * stream, so we make two cheap O(n*m) passes (no allocation, freestanding): first walk to
 * the trace/prefetch tables, then per segment confirm its claim_id is traced and its prefetch
 * resolves. Bounds-checked end to end via the same cursor. */

/* Skip one whole segment record (all versions), advancing the cursor. */
static void skip_segment(cur *c, uint16_t version) {
  c_skip_str(c);                /* name */
  c_skip(c, 8 + 4 + 1 + 4 + 4); /* claim_id,u64 phase_id,u32 lane,u8 width,u32 stride_k,u32 */
  if (c->err) return;
  c_skip_str(c);                /* opcode */
  { uint16_t n; (void)c_u32arr(c, &n); }  /* reads */
  { uint16_t n; (void)c_u32arr(c, &n); }  /* writes */
  c_skip_str(c);                /* prefetch */
  c_skip_strarr(c);             /* fence_before */
  c_skip_strarr(c);             /* fence_after */
  if (version >= 3) {
    c_skip(c, 1);               /* dispatch u8 */
    c_skip_str(c);              /* channel */
  }
}

/* A length-prefixed string equals (a,la) == (b,lb)? */
static int str_eq(const char *a, uint16_t la, const char *b, uint16_t lb) {
  if (la != lb) return 0;
  for (uint16_t i = 0; i < la; i++) if (a[i] != b[i]) return 0;
  return 1;
}

/* Does a trace record with claim_id == cid exist? Re-walks the trace stream (bounded). */
static int trace_has_claim(const uint8_t *data, size_t body_len, size_t trace_start,
                           uint32_t n_trace, uint64_t cid) {
  cur c; c.d = data; c.len = body_len; c.pos = trace_start; c.err = 0;
  for (uint32_t i = 0; i < n_trace && !c.err; i++) {
    uint64_t tc = c_u64(&c);    /* claim_id */
    c_skip(&c, 16);             /* src_hash u64 + trace_hash u64 */
    if (!c.err && tc == cid) return 1;
  }
  return 0;
}

/* Does a prefetch record with the given name exist? Re-walks the prefetch stream (bounded). */
static int prefetch_has_name(const uint8_t *data, size_t body_len, size_t pf_start,
                             uint32_t n_pf, uint16_t version,
                             const char *name, uint16_t name_len) {
  cur c; c.d = data; c.len = body_len; c.pos = pf_start; c.err = 0;
  for (uint32_t i = 0; i < n_pf && !c.err; i++) {
    uint16_t nl; const char *pn = c_str(&c, &nl);
    if (c.err) break;
    if (str_eq(pn, nl, name, name_len)) return 1;
    /* skip the rest of this prefetch record */
    c_skip(&c, 4);              /* distance */
    { uint16_t n; (void)c_u32arr(&c, &n); }  /* targets */
    c_skip_str(&c);             /* hint */
    c_skip_str(&c);             /* pattern */
    if (version >= 2) c_skip(&c, 1);
  }
  return 0;
}

/* R10 prefetch-coverage: a segment that DECLARES a prefetch must have at least one of its
 * read RIDs among that prefetch's targets (the prefetch feeds this segment's operands --
 * the contract hydrate establishes, `pf.targets == claim.rd`). Returns 1 if some read RID is
 * a target (covered), 0 otherwise. Re-walks the prefetch stream to the named record (bounded).
 * A segment with no reads is vacuously covered. Mirrors verify_pack's prefetch provenance. */
static int prefetch_covers_read(const uint8_t *data, size_t body_len, size_t pf_start,
                                uint32_t n_pf, uint16_t version,
                                const char *name, uint16_t name_len,
                                const uint8_t *reads, uint16_t n_reads) {
  if (n_reads == 0) return 1;
  cur c; c.d = data; c.len = body_len; c.pos = pf_start; c.err = 0;
  for (uint32_t i = 0; i < n_pf && !c.err; i++) {
    uint16_t nl; const char *pn = c_str(&c, &nl);
    if (c.err) break;
    int is_named = str_eq(pn, nl, name, name_len);
    c_skip(&c, 4);              /* distance */
    uint16_t nt; const uint8_t *targets = c_u32arr(&c, &nt);
    if (c.err) break;
    if (is_named) {
      for (uint16_t r = 0; r < n_reads; r++) {
        uint32_t rid = rd32(reads + (size_t)r * 4u);
        for (uint16_t t = 0; t < nt; t++)
          if (rd32(targets + (size_t)t * 4u) == rid) return 1;
      }
      return 0;                 /* the named prefetch covers no read RID -> redirected target */
    }
    c_skip_str(&c);             /* hint */
    c_skip_str(&c);             /* pattern */
    if (version >= 2) c_skip(&c, 1);
  }
  return 0;                     /* name not found -> caller's prefetch_has_name already failed */
}

bcir_status bcir_sp_verify_semantic(const uint8_t *BCIR_RESTRICT data, size_t len,
                                    uint32_t expected_map_gen,
                                    uint32_t expected_data_gen) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);   /* magic + version + CRC + bounds */
  if (st != BCIR_OK) return st;

  /* R11 (staleness): a generation mismatch is a stale pack -- reject, do not execute. */
  if (expected_map_gen != 0xFFFFFFFFu && hdr.map_gen != expected_map_gen)
    return BCIR_ERR_STALE;
  if (expected_data_gen != 0xFFFFFFFFu && hdr.data_gen != expected_data_gen)
    return BCIR_ERR_STALE;

  /* R10 well-formedness: pipeline_depth >= 1 (already normalized by validate); the per-
   * prefetch buffers (1|2) is checked as the prefetch stream is walked below. */
  if (hdr.pipeline_depth < 1) return BCIR_ERR_PROVENANCE;

  const size_t body_len = len - 4u;

  /* Pass 1: locate the prefetch + trace record streams by skipping the segment + block
   * streams (bounds-checked). The body is source_plan, segments, prefetches, blocks, trace. */
  cur c; c.d = data; c.len = body_len; c.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  c_skip_str(&c);   /* source_plan */
  for (uint32_t i = 0; i < hdr.n_segments && !c.err; i++) skip_segment(&c, hdr.version);
  if (c.err) return BCIR_ERR_TRUNCATED;
  size_t pf_start = c.pos;
  /* validate prefetch buffers (1|2) while skipping to the blocks */
  for (uint32_t i = 0; i < hdr.n_prefetches && !c.err; i++) {
    c_skip_str(&c);                /* name */
    c_skip(&c, 4);                 /* distance */
    { uint16_t n; (void)c_u32arr(&c, &n); }  /* targets */
    c_skip_str(&c);                /* hint */
    c_skip_str(&c);                /* pattern */
    if (hdr.version >= 2) {
      uint8_t buffers = c_u8(&c);
      if (!c.err && buffers != 1u && buffers != 2u) return BCIR_ERR_PROVENANCE;
    }
  }
  if (c.err) return BCIR_ERR_TRUNCATED;
  for (uint32_t i = 0; i < hdr.n_blocks && !c.err; i++) {
    c_skip(&c, 16);                /* base u64 + count u64 */
    { uint16_t n = c_u16(&c);      /* strides u64_array */
      c_skip(&c, (size_t)n * 8u); }
  }
  if (c.err) return BCIR_ERR_TRUNCATED;
  size_t trace_start = c.pos;

  /* Pass 2: per segment, confirm (R10) its claim_id is traced + its prefetch resolves. */
  cur s; s.d = data; s.len = body_len; s.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; s.err = 0;
  c_skip_str(&s);   /* source_plan */
  for (uint32_t i = 0; i < hdr.n_segments && !s.err; i++) {
    bcir_segment_view v;
    v.name = c_str(&s, &v.name_len);
    v.claim_id = c_u64(&s);
    v.phase_id = c_u32(&s);
    v.lane = c_u8(&s);
    v.width = c_u32(&s);
    v.stride_k = c_u32(&s);
    v.opcode = c_str(&s, &v.opcode_len);
    v.reads = c_u32arr(&s, &v.n_reads);
    v.writes = c_u32arr(&s, &v.n_writes);
    v.prefetch = c_str(&s, &v.prefetch_len);
    c_skip_strarr(&s);  /* fence_before */
    c_skip_strarr(&s);  /* fence_after */
    if (hdr.version >= 3) { v.dispatch = c_u8(&s); v.channel = c_str(&s, &v.channel_len); }
    else { v.dispatch = (uint8_t)BCIR_DISPATCH_CORE; v.channel = BCIR_DEFAULT_CHANNEL; v.channel_len = BCIR_DEFAULT_CHANNEL_LEN; }
    if (s.err) return BCIR_ERR_TRUNCATED;
    bcir_status rg = seg_range_ok(&v);
    if (rg != BCIR_OK) return rg;
    /* R10: every segment maps back to a live trace note (no dangling/redirected claim_id). */
    if (!trace_has_claim(data, body_len, trace_start, hdr.n_trace, v.claim_id))
      return BCIR_ERR_PROVENANCE;
    /* R10: a declared prefetch name must resolve to a prefetch record... */
    if (v.prefetch_len > 0) {
      if (!prefetch_has_name(data, body_len, pf_start, hdr.n_prefetches, hdr.version,
                             v.prefetch, v.prefetch_len))
        return BCIR_ERR_PROVENANCE;
      /* ...and that prefetch must actually feed this segment (cover >= 1 read RID), so a
       * swapped/redirected prefetch target (an undeclared RID) is caught. */
      if (!prefetch_covers_read(data, body_len, pf_start, hdr.n_prefetches, hdr.version,
                                v.prefetch, v.prefetch_len, v.reads, v.n_reads))
        return BCIR_ERR_PROVENANCE;
    }
  }
  return BCIR_OK;
}
