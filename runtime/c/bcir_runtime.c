/*===- bcir_runtime.c - freestanding BCIR StreamPack runtime --------------===
 *
 * No libc: only <stddef.h>/<stdint.h>. Little-endian wire decoding done byte by
 * byte (host-endian-independent). Matches bcir/abi/streampack_abi.py.
 *===----------------------------------------------------------------------===*/
#include "bcir_runtime.h"
#include "bcir_crc32.h"
#include "bcir_execution_plan.h"

/* --- zlib-compatible CRC-32 (freestanding) --- */
/* CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320; zlib's crc32) eight bytes a step: bcir_crc32.h's tables
 * and step, the ones every unit that checks a CRC-32 reads (CRC-SLICE; the byte table before it, one table step a
 * byte, replaced in turn the bitwise loop that was 81% of the native hydrate's instructions -- CXX4). Const data:
 * freestanding, no heap, no init. Held equal to zlib.crc32 over random buffers (test_c_runtime.py). */
uint32_t bcir_crc32(const uint8_t *BCIR_RESTRICT data, size_t len) {
  if (!data && len) return 0;
  return bcir_crc32_step(0xFFFFFFFFu, data, len) ^ 0xFFFFFFFFu;
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
  int err;      /* 0 clean, 1 truncated, 2 invalid UTF-8 */
} cur;

int bcir_utf8_valid(const uint8_t *BCIR_RESTRICT s, size_t n) {
  size_t i = 0;
  while (i < n) {
    uint8_t b0 = s[i++];
    uint32_t cp;
    size_t extra;
    if (b0 < 0x80u) continue;
    if (b0 >= 0xC2u && b0 <= 0xDFu) {
      cp = (uint32_t)(b0 & 0x1Fu); extra = 1u;
    } else if (b0 >= 0xE0u && b0 <= 0xEFu) {
      cp = (uint32_t)(b0 & 0x0Fu); extra = 2u;
    } else if (b0 >= 0xF0u && b0 <= 0xF4u) {
      cp = (uint32_t)(b0 & 0x07u); extra = 3u;
    } else {
      return 0;
    }
    if (extra > n - i) return 0;
    for (size_t j = 0; j < extra; j++) {
      uint8_t bx = s[i++];
      if ((bx & 0xC0u) != 0x80u) return 0;
      cp = (cp << 6) | (uint32_t)(bx & 0x3Fu);
    }
    if ((extra == 1u && cp < 0x80u) || (extra == 2u && cp < 0x800u) ||
        (extra == 3u && cp < 0x10000u) || cp > 0x10FFFFu ||
        (cp >= 0xD800u && cp <= 0xDFFFu)) return 0;
  }
  return 1;
}

static bcir_status c_status(const cur *c) {
  return c->err == 2 ? BCIR_ERR_UTF8 : BCIR_ERR_TRUNCATED;
}

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
  if (!bcir_utf8_valid(c->d + c->pos, n)) { c->err = 2; *out_len = 0; return 0; }
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
  uint16_t n;
  (void)c_str(c, &n);
}
static void c_skip_strarr(cur *c) {
  uint16_t n = c_u16(c);
  for (uint16_t i = 0; i < n && !c->err; i++) c_skip_str(c);
}

uint32_t bcir_seg_read_rid(const bcir_segment_view *seg, uint16_t i) {
  return (seg && seg->reads && i < seg->n_reads)
             ? rd32(seg->reads + (size_t)i * 4u) : 0u;
}
uint32_t bcir_seg_write_rid(const bcir_segment_view *seg, uint16_t i) {
  return (seg && seg->writes && i < seg->n_writes)
             ? rd32(seg->writes + (size_t)i * 4u) : 0u;
}

/* The range gate over a decoded segment view (the rail-symmetry fix): lane in [0,5],
 * width a nonzero power of two, dispatch in the legal set. Mirrors the Python decoder's
 * Lane(r.u8()) raise, the width power-of-two raise, and the unknown-dispatch-code raise.
 * Returns BCIR_OK or BCIR_ERR_*. */
static bcir_status seg_range_ok(const bcir_segment_view *v) {
  if (v->lane > (uint8_t)BCIR_LANE_H) return BCIR_ERR_LANE;   /* 6 valid lanes: 0..5 */
  if (v->width == 0u || (v->width & (v->width - 1u)) != 0u) return BCIR_ERR_WIDTH;
  if (v->stride_k != 0u) return BCIR_ERR_RESERVED;
  if (v->dispatch > (uint8_t)BCIR_DISPATCH_MAX) return BCIR_ERR_DISPATCH;
  return BCIR_OK;
}

bcir_status bcir_sp_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                             bcir_streampack_header *BCIR_RESTRICT hdr) {
  if (hdr) {
    for (int i = 0; i < 4; i++) hdr->magic[i] = 0;
    hdr->version = 0; hdr->flags = 0;
    hdr->topo_gen = 0; hdr->map_gen = 0; hdr->data_gen = 0;
    hdr->n_segments = 0; hdr->n_prefetches = 0;
    hdr->n_blocks = 0; hdr->n_trace = 0; hdr->pipeline_depth = 0;
    hdr->reserved0[0] = 0; hdr->reserved0[1] = 0; hdr->n_gens = 0;
    for (int i = 0; i < 20; i++) hdr->reserved[i] = 0;
  }
  if (!data) return BCIR_ERR_TRUNCATED;
  if (len < (size_t)BCIR_STREAMPACK_HEADER_SIZE + 4u) return BCIR_ERR_TRUNCATED;
  if (data[0] != 'B' || data[1] != 'S' || data[2] != 'P' || data[3] != 'K')
    return BCIR_ERR_MAGIC;
  uint16_t version = rd16(data + 4);
  if (version < BCIR_STREAMPACK_VERSION || version > BCIR_STREAMPACK_VERSION_MAX)
    return BCIR_ERR_VERSION;
  if (rd16(data + 6) != 0u) return BCIR_ERR_RESERVED;
  {
    /* v2 carved pipeline_depth at 36..37; v4 carved n_gens at 40..43. Everything else in
     * the pad is reserved and must be zero -- a reserved byte is never an extension point. */
    size_t reserved_start = version >= 2u ? 38u : 36u;
    for (size_t i = reserved_start; i < BCIR_STREAMPACK_HEADER_SIZE; i++) {
      if (version >= 4u && i >= 40u && i < 44u) continue;
      if (data[i] != 0u) return BCIR_ERR_RESERVED;
    }
  }
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
    /* v4 appended the generation-vector count; a v1-v3 pack carries no vector. */
    hdr->n_gens = (version >= 4) ? rd32(data + 40) : 0;
  }
  return BCIR_OK;
}

/* SP-ONCE: each walk is a body over the header bcir_sp_validate read from the same bytes. A plain walk
 * validates and runs its body; a checked walk (bcir_sp_checked_*) runs it over a pack bcir_sp_check
 * validated once -- every bound, every range gate and every verdict the same, the CRC not recomputed. */
static bcir_status sp_segments(const uint8_t *BCIR_RESTRICT data, size_t len,
                               const bcir_streampack_header *hdr, bcir_seg_fn fn, void *ctx) {
  cur c;
  c.d = data; c.len = len - 4u; c.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  int invoke_callback = 1;
  c_skip_str(&c); /* source_plan */

  for (uint32_t i = 0; i < hdr->n_segments && !c.err; i++) {
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
    if (hdr->version >= 3) {
      v.dispatch = c_u8(&c);
      v.channel = c_str(&c, &v.channel_len);
    } else {
      v.dispatch = (uint8_t)BCIR_DISPATCH_CORE;
      v.channel = BCIR_DEFAULT_CHANNEL; v.channel_len = BCIR_DEFAULT_CHANNEL_LEN;
    }
    if (c.err) return c_status(&c);
    /* Range gate: reject an out-of-range lane/width/dispatch (mirror the Python raise),
     * so a CRC-valid pack with a bad lane fails on BOTH rails identically. */
    bcir_status rg = seg_range_ok(&v);
    if (rg != BCIR_OK) return rg;
    /* Stop invoking the callback when requested, but still parse every declared
     * segment so a malformed later record cannot hide behind an early match. */
    if (invoke_callback && fn && fn(&v, ctx)) invoke_callback = 0;
  }
  return c.err ? c_status(&c) : BCIR_OK;
}

bcir_status bcir_sp_for_each_segment(const uint8_t *BCIR_RESTRICT data, size_t len,
                                     bcir_seg_fn fn, void *ctx) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);
  return st != BCIR_OK ? st : sp_segments(data, len, &hdr, fn, ctx);
}

/* R11: a generation mismatch is a STALE pack -- rehydrate, never execute. */
static bcir_status sp_generation(const bcir_streampack_header *hdr, uint32_t expected_map_gen,
                                 uint32_t expected_data_gen) {
  if (expected_map_gen != 0xFFFFFFFFu && hdr->map_gen != expected_map_gen)
    return BCIR_ERR_STALE;
  if (expected_data_gen != 0xFFFFFFFFu && hdr->data_gen != expected_data_gen)
    return BCIR_ERR_STALE;
  return BCIR_OK;
}

bcir_status bcir_sp_check_generation(const uint8_t *BCIR_RESTRICT data, size_t len,
                                     uint32_t expected_map_gen,
                                     uint32_t expected_data_gen) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);
  return st != BCIR_OK ? st : sp_generation(&hdr, expected_map_gen, expected_data_gen);
}

/* --- R10 provenance walk (freestanding; second pass over the decoded body) ---
 * verify_pack's R10 resolves each segment's claim_id against the pack's trace records and
 * each segment's prefetch name against the declared prefetches. The body is a single forward
 * stream, so we make two passes (no allocation, freestanding): first walk to the trace/
 * prefetch tables, then per segment confirm its claim_id is traced and its prefetch resolves.
 * Bounds-checked end to end via the same cursor.
 *
 * Every lookup is exact, and linear on the packs hydrate writes, whose segments, trace records
 * and prefetches name their claims in one order: a search starts where the previous one matched
 * and wraps around, and a uniqueness check rescans its prefix only for an id or a name that
 * does not sort after every earlier one. Any other order costs rescans, never a verdict. */

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

/* Skip one whole prefetch record (all versions), advancing the cursor. */
static void skip_prefetch(cur *c, uint16_t version) {
  c_skip_str(c);                /* name */
  c_skip(c, 4);                 /* distance */
  { uint16_t n; (void)c_u32arr(c, &n); }  /* targets */
  c_skip_str(c);                /* hint */
  c_skip_str(c);                /* pattern */
  if (version >= 2) c_skip(c, 1);         /* buffers */
}

static const uint8_t *c_u64arr(cur *c, uint16_t *out_cnt) {
  uint16_t n = c_u16(c);
  size_t bytes = (size_t)n * 8u;
  if (!c_has(c, bytes)) { c->err = 1; *out_cnt = 0; return 0; }
  const uint8_t *p = c->d + c->pos;
  c->pos += bytes; *out_cnt = n; return p;
}

uint64_t bcir_block_stride(const bcir_block_view *block, uint16_t i) {
  return (block && block->strides && i < block->n_strides)
             ? rd64(block->strides + (size_t)i * 8u) : 0u;
}

static bcir_status sp_blocks(const uint8_t *BCIR_RESTRICT data, size_t len,
                             const bcir_streampack_header *hdr, bcir_block_fn fn, void *ctx) {
  cur c;
  c.d = data; c.len = len - 4u; c.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  int invoke_callback = 1;
  c_skip_str(&c); /* source_plan */
  for (uint32_t i = 0; i < hdr->n_segments && !c.err; i++) skip_segment(&c, hdr->version);
  for (uint32_t i = 0; i < hdr->n_prefetches && !c.err; i++) skip_prefetch(&c, hdr->version);
  for (uint32_t i = 0; i < hdr->n_blocks && !c.err; i++) {
    bcir_block_view b;
    b.base = c_u64(&c);
    b.count = c_u64(&c);
    b.strides = c_u64arr(&c, &b.n_strides);
    if (c.err) return c_status(&c);
    /* As the segment walk: stop invoking on request, keep parsing every declared block. */
    if (invoke_callback && fn && fn(&b, ctx)) invoke_callback = 0;
  }
  return c.err ? c_status(&c) : BCIR_OK;
}

bcir_status bcir_sp_for_each_block(const uint8_t *BCIR_RESTRICT data, size_t len,
                                   bcir_block_fn fn, void *ctx) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);
  return st != BCIR_OK ? st : sp_blocks(data, len, &hdr, fn, ctx);
}

/* A StreamPack executes each claim at most once. Re-walk only the already-validated
 * prefix so duplicate segment ids can be refused without heap scratch. */
static int segment_claim_seen_before(const uint8_t *data, size_t body_len,
                                     uint16_t version, uint32_t before, uint64_t cid) {
  cur c; c.d = data; c.len = body_len; c.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  c_skip_str(&c);
  for (uint32_t i = 0; i < before && !c.err; i++) {
    size_t start = c.pos;
    c_skip_str(&c);
    uint64_t prior = c_u64(&c);
    if (c.err) return 0;
    if (!c.err && prior == cid) return 1;
    c.pos = start; c.err = 0;
    skip_segment(&c, version);
  }
  return 0;
}

/* A length-prefixed string equals (a,la) == (b,lb)? */
static int str_eq(const char *a, uint16_t la, const char *b, uint16_t lb) {
  if (la != lb) return 0;
  for (uint16_t i = 0; i < la; i++) if (a[i] != b[i]) return 0;
  return 1;
}

/* Does (a,la) sort strictly after (b,lb) in shortlex order (length, then octets)? A strict
 * total order on octet strings, so a name after the greatest earlier one repeats none. */
static int shortlex_after(const char *a, uint16_t la, const char *b, uint16_t lb) {
  if (la != lb) return la > lb;
  for (uint16_t i = 0; i < la; i++)
    if (a[i] != b[i]) return (unsigned char)a[i] > (unsigned char)b[i];
  return 0;
}

/* Does a trace record with claim_id == cid exist? The trace stream is fixed-width records
 * already walked to the CRC, so record k is read in place: from *next (where the previous
 * search matched) to the end, then from the start. A match leaves *next after it. */
static int trace_find_claim(const uint8_t *data, size_t trace_start, uint32_t n_trace,
                            uint64_t cid, uint32_t *next) {
  for (uint32_t k = *next; k < n_trace; k++)
    if (rd64(data + trace_start + (size_t)k * 24u) == cid) { *next = k + 1u; return 1; }
  for (uint32_t k = 0; k < *next && k < n_trace; k++)
    if (rd64(data + trace_start + (size_t)k * 24u) == cid) { *next = k + 1u; return 1; }
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

/* Where the previous prefetch search matched: the record after it, and that record's offset. */
typedef struct { size_t pos; uint32_t index; } pf_cursor;

/* Records [first, last) starting at offset `from`: the one named (name, name_len), its
 * targets, and the cursor after it. The stream was walked whole in pass 1, so the re-walk
 * cannot fail; a failure reads as "not found", a refusal, as before. */
static int prefetch_scan(const uint8_t *data, size_t body_len, uint16_t version, size_t from,
                         uint32_t first, uint32_t last, const char *name, uint16_t name_len,
                         const uint8_t **targets, uint16_t *n_targets, pf_cursor *at) {
  cur c; c.d = data; c.len = body_len; c.pos = from; c.err = 0;
  for (uint32_t i = first; i < last && !c.err; i++) {
    uint16_t nl; const char *pn = c_str(&c, &nl);
    c_skip(&c, 4);              /* distance */
    uint16_t nt; const uint8_t *tg = c_u32arr(&c, &nt);
    c_skip_str(&c);             /* hint */
    c_skip_str(&c);             /* pattern */
    if (version >= 2) c_skip(&c, 1);
    if (c.err) return 0;
    if (str_eq(pn, nl, name, name_len)) {
      *targets = tg; *n_targets = nt; at->pos = c.pos; at->index = i + 1u;
      return 1;
    }
  }
  return 0;
}

/* The prefetch record named (name, name_len), searched from the cursor to the end, then from
 * the first record. Names are unique by the time this runs (pass 1 refuses a repeat), so the
 * record found is THE named one, wherever the search starts. */
static int prefetch_find(const uint8_t *data, size_t body_len, uint16_t version,
                         size_t pf_start, uint32_t n_pf, pf_cursor *at,
                         const char *name, uint16_t name_len,
                         const uint8_t **targets, uint16_t *n_targets) {
  if (prefetch_scan(data, body_len, version, at->pos, at->index, n_pf, name, name_len,
                    targets, n_targets, at))
    return 1;
  return prefetch_scan(data, body_len, version, pf_start, 0, at->index, name, name_len,
                       targets, n_targets, at);
}

/* R10 prefetch-coverage: a segment that DECLARES a prefetch must have at least one of its
 * read RIDs among that prefetch's targets (the prefetch feeds this segment's operands --
 * the contract hydrate establishes, `pf.targets == claim.rd`). Returns 1 if some read RID is
 * a target (covered), 0 otherwise (a redirected target). A segment with no reads is vacuously
 * covered. Mirrors verify_pack's prefetch provenance. */
static int targets_cover_read(const uint8_t *targets, uint16_t n_targets,
                              const uint8_t *reads, uint16_t n_reads) {
  if (n_reads == 0) return 1;
  for (uint16_t r = 0; r < n_reads; r++) {
    uint32_t rid = rd32(reads + (size_t)r * 4u);
    for (uint16_t t = 0; t < n_targets; t++)
      if (rd32(targets + (size_t)t * 4u) == rid) return 1;
  }
  return 0;
}

static bcir_status sp_semantic(const uint8_t *BCIR_RESTRICT data, size_t len,
                               const bcir_streampack_header *hdr, uint32_t expected_map_gen,
                               uint32_t expected_data_gen) {
  /* R11 (staleness): a generation mismatch is a stale pack -- reject, do not execute. */
  bcir_status st = sp_generation(hdr, expected_map_gen, expected_data_gen);
  if (st != BCIR_OK) return st;

  /* R10 well-formedness: pipeline_depth >= 1; the per-
   * prefetch buffers (1|2) is checked as the prefetch stream is walked below. */
  if (hdr->pipeline_depth < 1) return BCIR_ERR_PROVENANCE;

  const size_t body_len = len - 4u;

  /* Pass 1: locate the prefetch + trace record streams by skipping the segment + block
   * streams (bounds-checked). The body is source_plan, segments, prefetches, blocks, trace. */
  cur c; c.d = data; c.len = body_len; c.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  c_skip_str(&c);   /* source_plan */
  for (uint32_t i = 0; i < hdr->n_segments && !c.err; i++) skip_segment(&c, hdr->version);
  if (c.err) return c_status(&c);
  size_t pf_start = c.pos;
  /* validate prefetch buffers (1|2) while skipping to the blocks */
  const char *pf_last = 0;         /* the shortlex-greatest name so far */
  uint16_t pf_last_len = 0;
  for (uint32_t i = 0; i < hdr->n_prefetches && !c.err; i++) {
    uint16_t name_len;
    const char *name = c_str(&c, &name_len);
    if (!c.err) {
      if (i == 0 || shortlex_after(name, name_len, pf_last, pf_last_len)) {
        pf_last = name; pf_last_len = name_len;  /* repeats no earlier name */
      } else if (prefetch_has_name(data, body_len, pf_start, i, hdr->version, name, name_len)) {
        return BCIR_ERR_PROVENANCE;  /* duplicate names make target binding ambiguous */
      }
    }
    c_skip(&c, 4);                 /* distance */
    { uint16_t n; (void)c_u32arr(&c, &n); }  /* targets */
    c_skip_str(&c);                /* hint */
    c_skip_str(&c);                /* pattern */
    if (hdr->version >= 2) {
      uint8_t buffers = c_u8(&c);
      if (!c.err && buffers != 1u && buffers != 2u) return BCIR_ERR_PROVENANCE;
    }
  }
  if (c.err) return c_status(&c);
  for (uint32_t i = 0; i < hdr->n_blocks && !c.err; i++) {
    c_skip(&c, 16);                /* base u64 + count u64 */
    { uint16_t n = c_u16(&c);      /* strides u64_array */
      c_skip(&c, (size_t)n * 8u); }
  }
  if (c.err) return c_status(&c);
  size_t trace_start = c.pos;
  /* The declared trace stream must itself fit and must end EXACTLY at the CRC trailer. CRC alone
   * authenticates bytes, not their schema membership: accepting undeclared suffix bytes gives two
   * byte-distinct artifacts the same decoded meaning and lets future parsers disagree. */
  uint64_t trace_max = 0;           /* the greatest claim id traced so far */
  for (uint32_t i = 0; i < hdr->n_trace && !c.err; i++) {
    uint64_t cid = c_u64(&c);
    if (c.err) break;
    if (i == 0 || cid > trace_max) {
      trace_max = cid;              /* repeats no earlier id */
    } else {
      for (uint32_t j = 0; j < i; j++)
        if (rd64(data + trace_start + (size_t)j * 24u) == cid)
          return BCIR_ERR_PROVENANCE; /* conflicting trace hashes cannot share an id */
    }
    c_skip(&c, 16);                 /* src_hash + trace_hash */
  }
  if (c.err) return c_status(&c);
  /* v4: the per-resource generation vector -- RIDs strictly ascending (sorted, unique) and
   * the header map_gen/data_gen exactly its maxima. The header tags a v1-v3 reader sees are
   * a SUMMARY of the vector; two sources of truth that may disagree are a second spelling
   * of a stale pack (the Python decoder applies the same predicate). */
  if (hdr->version >= 4) {
    uint32_t prev = 0, vmap = 0, vdata = 0;
    int have_prev = 0;
    for (uint32_t i = 0; i < hdr->n_gens && !c.err; i++) {
      uint32_t rid = c_u32(&c), mg = c_u32(&c), dg = c_u32(&c);
      if (c.err) break;
      if (have_prev && rid <= prev) return BCIR_ERR_GENERATION;
      prev = rid; have_prev = 1;
      if (mg > vmap) vmap = mg;
      if (dg > vdata) vdata = dg;
    }
    if (c.err) return c_status(&c);
    if (hdr->n_gens != 0 && (vmap != hdr->map_gen || vdata != hdr->data_gen))
      return BCIR_ERR_GENERATION;
  }
  if (c.pos != body_len) return BCIR_ERR_TRAILING;

  /* Pass 2: per segment, confirm (R10) its claim_id is traced + its prefetch resolves. */
  cur s; s.d = data; s.len = body_len; s.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; s.err = 0;
  c_skip_str(&s);   /* source_plan */
  uint64_t seg_max = 0;             /* the greatest segment claim id so far */
  uint32_t trace_next = 0;          /* the trace record after the previous match */
  pf_cursor pf_next; pf_next.pos = pf_start; pf_next.index = 0;
  for (uint32_t i = 0; i < hdr->n_segments && !s.err; i++) {
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
    if (hdr->version >= 3) { v.dispatch = c_u8(&s); v.channel = c_str(&s, &v.channel_len); }
    else { v.dispatch = (uint8_t)BCIR_DISPATCH_CORE; v.channel = BCIR_DEFAULT_CHANNEL; v.channel_len = BCIR_DEFAULT_CHANNEL_LEN; }
    if (s.err) return c_status(&s);
    bcir_status rg = seg_range_ok(&v);
    if (rg != BCIR_OK) return rg;
    if (i == 0 || v.claim_id > seg_max)
      seg_max = v.claim_id;         /* repeats no earlier segment's claim */
    else if (segment_claim_seen_before(data, body_len, hdr->version, i, v.claim_id))
      return BCIR_ERR_PROVENANCE;
    /* R10: every segment maps back to a live trace note (no dangling/redirected claim_id). */
    if (!trace_find_claim(data, trace_start, hdr->n_trace, v.claim_id, &trace_next))
      return BCIR_ERR_PROVENANCE;
    /* R10: a declared prefetch name must resolve to a prefetch record... */
    if (v.prefetch_len > 0) {
      const uint8_t *targets = 0;
      uint16_t n_targets = 0;
      if (!prefetch_find(data, body_len, hdr->version, pf_start, hdr->n_prefetches, &pf_next,
                         v.prefetch, v.prefetch_len, &targets, &n_targets))
        return BCIR_ERR_PROVENANCE;
      /* ...and that prefetch must actually feed this segment (cover >= 1 read RID), so a
       * swapped/redirected prefetch target (an undeclared RID) is caught. */
      if (!targets_cover_read(targets, n_targets, v.reads, v.n_reads))
        return BCIR_ERR_PROVENANCE;
    }
  }
  return BCIR_OK;
}

bcir_status bcir_sp_verify_semantic(const uint8_t *BCIR_RESTRICT data, size_t len,
                                    uint32_t expected_map_gen,
                                    uint32_t expected_data_gen) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);   /* magic + version + CRC + bounds */
  return st != BCIR_OK ? st : sp_semantic(data, len, &hdr, expected_map_gen, expected_data_gen);
}

/* --- SP-ONCE: a pack checked once, walked without recomputing its CRC -------------------- */

bcir_status bcir_sp_check(const uint8_t *BCIR_RESTRICT data, size_t len,
                          bcir_sp_checked *BCIR_RESTRICT out) {
  if (!out) return bcir_sp_validate(data, len, NULL);
  bcir_status st = bcir_sp_validate(data, len, &out->hdr);
  out->data = st == BCIR_OK ? data : NULL;   /* a refused pack is no token: its walks refuse */
  out->len = st == BCIR_OK ? len : 0u;
  return st;
}

/* What every body assumes of its bytes besides their CRC: a buffer, and one long enough for a
 * header and the trailer (the body's cursor ends 4 octets short). A token bcir_sp_check did
 * not fill -- or one it refused -- fails here, never reads. */
static int sp_checked_ok(const bcir_sp_checked *pack) {
  return pack && pack->data && pack->len >= (size_t)BCIR_STREAMPACK_HEADER_SIZE + 4u;
}

bcir_status bcir_sp_checked_verify_semantic(const bcir_sp_checked *BCIR_RESTRICT pack,
                                            uint32_t expected_map_gen,
                                            uint32_t expected_data_gen) {
  if (!sp_checked_ok(pack)) return BCIR_ERR_TRUNCATED;
  return sp_semantic(pack->data, pack->len, &pack->hdr, expected_map_gen, expected_data_gen);
}

bcir_status bcir_sp_checked_generation(const bcir_sp_checked *BCIR_RESTRICT pack,
                                       uint32_t expected_map_gen, uint32_t expected_data_gen) {
  if (!sp_checked_ok(pack)) return BCIR_ERR_TRUNCATED;
  return sp_generation(&pack->hdr, expected_map_gen, expected_data_gen);
}

bcir_status bcir_sp_checked_for_each_segment(const bcir_sp_checked *BCIR_RESTRICT pack,
                                             bcir_seg_fn fn, void *ctx) {
  if (!sp_checked_ok(pack)) return BCIR_ERR_TRUNCATED;
  return sp_segments(pack->data, pack->len, &pack->hdr, fn, ctx);
}

bcir_status bcir_sp_checked_for_each_block(const bcir_sp_checked *BCIR_RESTRICT pack,
                                           bcir_block_fn fn, void *ctx) {
  if (!sp_checked_ok(pack)) return BCIR_ERR_TRUNCATED;
  return sp_blocks(pack->data, pack->len, &pack->hdr, fn, ctx);
}

/* --- v4: the per-resource generation vector (R11) ----------------------------------- */

/* The vector is the body's tail (verified above to end exactly at the CRC), so after the
 * semantic pass it is located by arithmetic, not by a third walk. */
static bcir_status locate_generations(const uint8_t *BCIR_RESTRICT data, size_t len,
                                      bcir_streampack_header *hdr, size_t *start) {
  bcir_status st = bcir_sp_validate(data, len, hdr);
  if (st == BCIR_OK) st = sp_semantic(data, len, hdr, 0xFFFFFFFFu, 0xFFFFFFFFu);
  if (st != BCIR_OK) return st;
  *start = 0;
  if (hdr->version < 4 || hdr->n_gens == 0) return BCIR_OK;
  {
    size_t body_len = len - 4u;
    size_t room = body_len - (size_t)BCIR_STREAMPACK_HEADER_SIZE;
    if ((size_t)hdr->n_gens > room / BCIR_GENERATION_WIRE_SIZE) return BCIR_ERR_TRUNCATED;
    *start = body_len - (size_t)hdr->n_gens * BCIR_GENERATION_WIRE_SIZE;
  }
  return BCIR_OK;
}

static void read_generation(const uint8_t *p, bcir_generation_view *g) {
  g->rid = rd32(p); g->map_gen = rd32(p + 4); g->data_gen = rd32(p + 8);
}

bcir_status bcir_sp_for_each_generation(const uint8_t *BCIR_RESTRICT data, size_t len,
                                        bcir_gen_fn fn, void *ctx) {
  bcir_streampack_header hdr;
  size_t start;
  bcir_status st = locate_generations(data, len, &hdr, &start);
  if (st != BCIR_OK) return st;
  if (hdr.version < 4) return BCIR_OK;
  for (uint32_t i = 0; i < hdr.n_gens; i++) {
    bcir_generation_view g;
    read_generation(data + start + (size_t)i * BCIR_GENERATION_WIRE_SIZE, &g);
    if (fn && fn(&g, ctx)) break;
  }
  return BCIR_OK;
}

bcir_status bcir_sp_check_generation_vector(const uint8_t *BCIR_RESTRICT data, size_t len,
                                            const bcir_generation_view *BCIR_RESTRICT live,
                                            size_t n_live) {
  bcir_streampack_header hdr;
  size_t start;
  bcir_status st = locate_generations(data, len, &hdr, &start);
  if (st != BCIR_OK) return st;
  if (n_live && !live) return BCIR_ERR_NOSPACE;   /* a caller table without storage */
  {
    uint32_t n_gens = hdr.version >= 4 ? hdr.n_gens : 0;
    /* R11: a registry that declares resources needs the vector -- a v1-v3 artifact, or a
     * pack with the vector stripped, is stale by construction, never judged by its maxima. */
    if (n_gens == 0) return n_live ? BCIR_ERR_STALE : BCIR_OK;
    /* A vector against an empty live registry: every entry names a resource the registry
     * no longer declares. */
    if (n_live == 0) return BCIR_ERR_STALE;
    /* Every entry matches a live resource exactly (O(n*m), no allocation, no ordering
     * demanded of the caller's table). */
    for (uint32_t i = 0; i < n_gens; i++) {
      bcir_generation_view g;
      size_t j;
      read_generation(data + start + (size_t)i * BCIR_GENERATION_WIRE_SIZE, &g);
      for (j = 0; j < n_live; j++)
        if (live[j].rid == g.rid) break;
      if (j == n_live) return BCIR_ERR_STALE;                    /* an undeclared RID */
      if (live[j].map_gen != g.map_gen || live[j].data_gen != g.data_gen)
        return BCIR_ERR_STALE;                                    /* moved: repack/replan */
    }
    /* ...and every live resource has an entry (declared after hydration otherwise). */
    for (size_t j = 0; j < n_live; j++) {
      uint32_t i;
      for (i = 0; i < n_gens; i++)
        if (rd32(data + start + (size_t)i * BCIR_GENERATION_WIRE_SIZE) == live[j].rid) break;
      if (i == n_gens) return BCIR_ERR_STALE;
    }
  }
  return BCIR_OK;
}


/* --- ExecutionPlanV1 (G11, staged plan S1-C): the plan the pack was derived from -------
 * The same discipline as the StreamPack rail: byte-wise little-endian reads, a bounded
 * cursor, every record parsed even after an early callback stop, O(n^2) re-walks of the
 * validated prefix instead of heap scratch, and exact body consumption. The Python codec
 * (bcir/abi/execution_plan_abi.py) applies the same laws, so the two rails refuse the same
 * bytes. */

static void ep_zero_header(bcir_ep_header *h) {
  int i;
  for (i = 0; i < 4; i++) h->magic[i] = 0;
  h->version = 0; h->flags = 0; h->mode = 0; h->liveness = 0;
  for (i = 0; i < 2; i++) h->reserved0[i] = 0;
  h->streams = 0; h->knee = 0;
  h->n_steps = 0; h->n_lifetimes = 0; h->n_moves = 0; h->n_gens = 0;
  for (i = 0; i < 4; i++) h->reserved1[i] = 0;
  h->makespan = 0; h->module_hash = 0; h->target_hash = 0;
}

bcir_status bcir_ep_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                             bcir_ep_header *BCIR_RESTRICT hdr) {
  if (hdr) ep_zero_header(hdr);
  if (!data) return BCIR_ERR_TRUNCATED;
  if (len < (size_t)BCIR_EP_HEADER_SIZE + 4u) return BCIR_ERR_TRUNCATED;
  if (data[0] != 'B' || data[1] != 'P' || data[2] != 'L' || data[3] != 'N')
    return BCIR_ERR_MAGIC;
  {
    uint16_t version = rd16(data + 4);
    if (version < BCIR_EP_VERSION || version > BCIR_EP_VERSION_MAX) return BCIR_ERR_VERSION;
  }
  if ((rd16(data + 6) & ~BCIR_EP_FLAGS) != 0u) return BCIR_ERR_RESERVED;
  /* v2 carved the liveness byte out of offset 9; on a v1 plan it is reserved (0). */
  if ((rd16(data + 4) < 2u && data[9]) || data[10] || data[11] ||
      data[36] || data[37] || data[38] || data[39])
    return BCIR_ERR_RESERVED;
  if (data[8] > (uint8_t)BCIR_EP_MODE_MAX) return BCIR_ERR_PLAN;
  /* G19: an explicit placement is a phase-barriered (eft) placement. */
  if ((rd16(data + 6) & BCIR_EP_FLAG_EXPLICIT_PLACEMENT) != 0u &&
      data[8] != (uint8_t)BCIR_EP_MODE_EFT)
    return BCIR_ERR_PLAN;
  if (rd16(data + 4) >= 2u && data[9] > (uint8_t)BCIR_EP_LIVENESS_MAX) return BCIR_ERR_PLAN;
  if (bcir_crc32(data, len - 4) != rd32(data + len - 4)) return BCIR_ERR_CRC;
  if (hdr) {
    int i;
    for (i = 0; i < 4; i++) hdr->magic[i] = data[i];
    hdr->version = rd16(data + 4);
    hdr->flags = rd16(data + 6);
    hdr->mode = data[8];
    hdr->liveness = hdr->version >= 2u ? data[9] : (uint8_t)BCIR_EP_LIVENESS_PHASE;
    hdr->streams = rd32(data + 12);
    hdr->knee = rd32(data + 16);
    hdr->n_steps = rd32(data + 20);
    hdr->n_lifetimes = rd32(data + 24);
    hdr->n_moves = rd32(data + 28);
    hdr->n_gens = rd32(data + 32);
    hdr->makespan = rd64(data + 40);
    hdr->module_hash = rd64(data + 48);
    hdr->target_hash = rd64(data + 56);
  }
  return BCIR_OK;
}

/* v3: the source/spec binding trailer after the generation vector. */
#define EP_BINDING_SIZE 16

/* Two's complement i64 from the wire word (well-defined for every bit pattern, pre-C23 too). */
static int64_t ep_i64(uint64_t raw) {
  if (raw < (UINT64_C(1) << 63)) return (int64_t)raw;
  return -(int64_t)(~raw) - 1;
}

/* Skip one step record (bounded). */
static void ep_skip_step(cur *c) {
  c_skip(c, 8 + 4);                     /* claim_id, phase_id */
  c_skip_str(c);                        /* candidate */
  c_skip(c, 1 + 4 + 8 + 4 + 8 + 8);     /* lane, width, cost, stream, start, duration */
}

/* A plan steps each claim at most once. Re-walk only the validated prefix (no heap). */
static int ep_claim_seen_before(const uint8_t *data, size_t body_len, size_t steps_start,
                                uint32_t before, uint64_t cid) {
  cur c; c.d = data; c.len = body_len; c.pos = steps_start; c.err = 0;
  for (uint32_t i = 0; i < before && !c.err; i++) {
    size_t start = c.pos;
    uint64_t prior = c_u64(&c);
    if (c.err) return 0;
    if (prior == cid) return 1;
    c.pos = start;
    ep_skip_step(&c);
  }
  return 0;
}

/* Read one lifetime record (all versions) into a view; a v1 record takes the phase default. */
static void ep_read_lifetime(cur *c, uint16_t version, bcir_ep_lifetime_view *v) {
  v->rid = c_u32(c);
  v->bank = c_str(c, &v->bank_len);
  v->offset = c_u64(c);
  v->size = c_u64(c);
  v->alignment = c_u32(c);
  v->first_phase = c_u32(c);
  v->last_phase = c_u32(c);
  if (version >= 2u) {
    v->first_tick = c_u64(c);
    v->last_tick = c_u64(c);
  } else {
    v->first_tick = (uint64_t)v->first_phase;
    v->last_tick = (uint64_t)v->last_phase + 1u;
  }
}

/* A lifetime's own laws, in the walk's order: the record read whole, RIDs ascending, a bank and
 * bytes, a power-of-two alignment its offset keeps, no interval past the 64-bit line, phases and
 * ticks in order. */
static bcir_status ep_lifetime_own_laws(const cur *c, const bcir_ep_lifetime_view *v,
                                        int have_prev, uint32_t prev) {
  if (c->err) return c_status(c);
  if (have_prev && v->rid <= prev) return BCIR_ERR_PLAN;
  if (v->bank_len == 0u || v->size == 0u) return BCIR_ERR_PLAN;
  if (v->alignment == 0u || (v->alignment & (v->alignment - 1u)) != 0u) return BCIR_ERR_PLAN;
  if ((v->offset & ((uint64_t)v->alignment - 1u)) != 0u) return BCIR_ERR_PLAN;
  if (v->offset > UINT64_MAX - v->size) return BCIR_ERR_OVERFLOW;
  if (v->last_phase < v->first_phase) return BCIR_ERR_PLAN;
  if (v->last_tick <= v->first_tick) return BCIR_ERR_PLAN;
  return BCIR_OK;
}

/* The alias law by bytes: two lifetimes of one bank live at once (half-open ticks) on
 * overlapping addresses. */
static int ep_lifetimes_alias(const bcir_ep_lifetime_view *a, const bcir_ep_lifetime_view *b) {
  return str_eq(a->bank, a->bank_len, b->bank, b->bank_len) &&
         a->first_tick < b->last_tick && b->first_tick < a->last_tick &&
         a->offset < b->offset + b->size && b->offset < a->offset + a->size;
}

/* The first of the first `k` lifetimes that aliases an earlier one (k when none does), with no
 * scratch: each against every earlier one, re-walking the validated prefix -- quadratic. */
static uint32_t ep_first_alias_rewalk(const uint8_t *data, size_t body_len, size_t start,
                                      uint32_t k, uint16_t version) {
  cur c; c.d = data; c.len = body_len; c.pos = start; c.err = 0;
  for (uint32_t i = 0; i < k; i++) {
    bcir_ep_lifetime_view v;
    cur p; p.d = data; p.len = body_len; p.pos = start; p.err = 0;
    ep_read_lifetime(&c, version, &v);
    for (uint32_t j = 0; j < i; j++) {
      bcir_ep_lifetime_view prior;
      ep_read_lifetime(&p, version, &prior);
      if (ep_lifetimes_alias(&prior, &v)) return i;
    }
  }
  return k;
}

/* ALIAS-SWEEP: the same first record in O(n log n) over caller-lent scratch.
 *
 * A sweep decides whether a prefix [0, p) of the records holds an aliasing pair. It visits the
 * records' ticks in order -- at one tick the ends before the starts, the intervals being
 * half-open -- keeping the live records ranked by (bank, offset, index). Until a pair is found
 * the live records of one bank lie pairwise apart in bytes, so a record that starts aliases a
 * live one only if it aliases the live record ranked just below it or just above it: two looks a
 * start, each O(log n) in a Fenwick tree over the ranks. The prefixes are monotone (a pair in
 * [0, p) is one in [0, p + 1)), so the first aliasing record is a binary search of sweeps, run
 * only once the whole is known to hold a pair: a plan the law admits costs one sweep.
 *
 * The scratch is bytes (little-endian, any alignment): per lifetime 42 bytes of fields -- its
 * offset, end, first and last tick and the position and length of its bank's octets in the plan
 * -- and five u32 (three orders: by key, by first tick, by last tick; its rank; a node of the
 * tree), and one u32 more, the tree's unused node 0: BCIR_EP_SWEEP_BYTES_PER_LIFETIME a lifetime
 * and BCIR_EP_SWEEP_BYTES_FIXED. */
#define EP_SW_FIELDS 42u

size_t bcir_ep_sweep_bytes(uint32_t n_lifetimes) {
  /* a u32 fits a size_t, so the count is exact; the product overflows only where size_t has 32
   * bits (a count past (2^32 - 5) / 62), and then names no scratch */
  const size_t n = (size_t)n_lifetimes;
  if (n > (SIZE_MAX - BCIR_EP_SWEEP_BYTES_FIXED) / BCIR_EP_SWEEP_BYTES_PER_LIFETIME) return 0;
  return n * BCIR_EP_SWEEP_BYTES_PER_LIFETIME + BCIR_EP_SWEEP_BYTES_FIXED;
}

static void sw_put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void sw_put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void sw_put64(uint8_t *p, uint64_t v) {
  sw_put32(p, (uint32_t)v);
  sw_put32(p + 4, (uint32_t)(v >> 32));
}

typedef struct {
  const uint8_t *data;   /* the plan: the banks' octets */
  uint8_t *f;            /* the fields, EP_SW_FIELDS bytes a record */
  uint8_t *by_key, *by_first, *by_last, *rank;  /* u32 arrays */
  uint8_t *tree;         /* u32 nodes 0 .. n */
  uint32_t n;            /* the records the sweep reads: those before the first breaking its own law */
} ep_sweep;

static uint32_t sw_at(const uint8_t *arr, size_t i) { return rd32(arr + 4u * i); }
static void sw_set(uint8_t *arr, size_t i, uint32_t v) { sw_put32(arr + 4u * i, v); }
static const uint8_t *sw_rec(const ep_sweep *s, uint32_t i) { return s->f + (size_t)i * EP_SW_FIELDS; }
static uint64_t sw_off(const ep_sweep *s, uint32_t i) { return rd64(sw_rec(s, i)); }
static uint64_t sw_end(const ep_sweep *s, uint32_t i) { return rd64(sw_rec(s, i) + 8); }
static uint64_t sw_first(const ep_sweep *s, uint32_t i) { return rd64(sw_rec(s, i) + 16); }
static uint64_t sw_last(const ep_sweep *s, uint32_t i) { return rd64(sw_rec(s, i) + 24); }
static const char *sw_bank(const ep_sweep *s, uint32_t i, uint16_t *len) {
  *len = rd16(sw_rec(s, i) + 40);
  return (const char *)(s->data + (size_t)rd64(sw_rec(s, i) + 32));
}

/* Carve the lent scratch for `n_lifetimes` records; 0 when none is lent or it is short. */
static int ep_sweep_lend(ep_sweep *s, uint8_t *scratch, size_t scratch_len, const uint8_t *data,
                         uint32_t n_lifetimes) {
  size_t need = bcir_ep_sweep_bytes(n_lifetimes), n = n_lifetimes;
  if (!scratch || need == 0u || scratch_len < need) return 0;
  s->data = data;
  s->f = scratch;
  s->by_key = s->f + n * EP_SW_FIELDS;
  s->by_first = s->by_key + 4u * n;
  s->by_last = s->by_first + 4u * n;
  s->rank = s->by_last + 4u * n;
  s->tree = s->rank + 4u * n;
  s->n = 0;
  return 1;
}

/* Record `i`'s fields, as the walk read it (its own laws kept: its end does not wrap). */
static void ep_sweep_put(ep_sweep *s, uint32_t i, const bcir_ep_lifetime_view *v) {
  uint8_t *p = s->f + (size_t)i * EP_SW_FIELDS;
  sw_put64(p, v->offset);
  sw_put64(p + 8, v->offset + v->size);
  sw_put64(p + 16, v->first_tick);
  sw_put64(p + 24, v->last_tick);
  sw_put64(p + 32, (uint64_t)((const uint8_t *)v->bank - s->data));
  sw_put16(p + 40, v->bank_len);
}

/* The three orders, each total (the index last): -1, 0 or 1 as record a sorts before, with or
 * after b. 0 by key (the bank's octets in shortlex, then the offset), 1 by first tick, 2 by last. */
static int sw_cmp_u64(uint64_t a, uint64_t b) { return a < b ? -1 : a > b ? 1 : 0; }
static int sw_cmp(const ep_sweep *s, int order, uint32_t a, uint32_t b) {
  int r = 0;
  if (order == 0) {
    uint16_t la, lb;
    const char *ba = sw_bank(s, a, &la), *bb = sw_bank(s, b, &lb);
    if (la != lb) return la < lb ? -1 : 1;
    for (uint16_t i = 0; i < la && !r; i++)
      if (ba[i] != bb[i]) r = (uint8_t)ba[i] < (uint8_t)bb[i] ? -1 : 1;
    if (!r) r = sw_cmp_u64(sw_off(s, a), sw_off(s, b));
  } else {
    r = order == 1 ? sw_cmp_u64(sw_first(s, a), sw_first(s, b))
                   : sw_cmp_u64(sw_last(s, a), sw_last(s, b));
  }
  return r ? r : sw_cmp_u64(a, b);
}

/* Heapsort of the record indices 0 .. n by `order`: in place, no recursion, O(n log n) on every
 * input. */
static void sw_sift(const ep_sweep *s, uint8_t *arr, int order, size_t root, size_t n) {
  for (;;) {
    size_t child = 2u * root + 1u;
    uint32_t top;
    if (child >= n) return;
    if (child + 1u < n && sw_cmp(s, order, sw_at(arr, child), sw_at(arr, child + 1u)) < 0)
      child++;
    if (sw_cmp(s, order, sw_at(arr, root), sw_at(arr, child)) >= 0) return;
    top = sw_at(arr, root);
    sw_set(arr, root, sw_at(arr, child));
    sw_set(arr, child, top);
    root = child;
  }
}
static void sw_sort(const ep_sweep *s, uint8_t *arr, int order) {
  size_t n = s->n, i;
  for (i = 0; i < n; i++) sw_set(arr, i, (uint32_t)i);
  for (i = n / 2u; i-- > 0;) sw_sift(s, arr, order, i, n);
  for (i = n; i-- > 1;) {
    uint32_t top = sw_at(arr, 0);
    sw_set(arr, 0, sw_at(arr, i));
    sw_set(arr, i, top);
    sw_sift(s, arr, order, 0, i);
  }
}

/* The Fenwick tree over the ranks: node i (1 .. n) counts the live ranks (i - lowbit(i), i]. */
static void sw_tree_add(ep_sweep *s, uint32_t rank, uint32_t delta) {  /* delta 1 or UINT32_MAX */
  for (size_t i = (size_t)rank + 1u; i <= s->n; i += i & (~i + 1u))
    sw_set(s->tree, i, sw_at(s->tree, i) + delta);
}
static uint32_t sw_tree_below(const ep_sweep *s, uint32_t rank) {  /* the live ranks below `rank` */
  uint32_t c = 0;
  for (size_t i = rank; i > 0; i -= i & (~i + 1u)) c += sw_at(s->tree, i);
  return c;
}
static uint32_t sw_tree_kth(const ep_sweep *s, uint32_t k) {       /* the k-th live rank, 1-based */
  size_t pos = 0, step = 1;
  while (step <= s->n / 2u) step <<= 1;
  for (; step; step >>= 1)
    if (pos + step <= s->n && sw_at(s->tree, pos + step) < k) {
      pos += step;
      k -= sw_at(s->tree, pos);
    }
  return (uint32_t)pos;
}

/* Do records a and b, both live, alias: one bank and overlapping bytes. */
static int sw_alias(const ep_sweep *s, uint32_t a, uint32_t b) {
  uint16_t la, lb;
  const char *ba = sw_bank(s, a, &la), *bb = sw_bank(s, b, &lb);
  return str_eq(ba, la, bb, lb) && sw_off(s, a) < sw_end(s, b) && sw_off(s, b) < sw_end(s, a);
}

/* Do records [0, p) hold an aliasing pair? */
static int sw_prefix_aliases(ep_sweep *s, uint32_t p) {
  size_t at_start = 0, at_end = 0, i;
  uint32_t live = 0;
  for (i = 0; i <= s->n; i++) sw_set(s->tree, i, 0);
  for (;;) {
    uint32_t x, r, below;
    while (at_start < s->n && sw_at(s->by_first, at_start) >= p) at_start++;
    while (at_end < s->n && sw_at(s->by_last, at_end) >= p) at_end++;
    if (at_start == s->n) return 0;                 /* every record started: no pair */
    x = sw_at(s->by_first, at_start);
    if (at_end < s->n && sw_last(s, sw_at(s->by_last, at_end)) <= sw_first(s, x)) {
      sw_tree_add(s, sw_at(s->rank, sw_at(s->by_last, at_end)), UINT32_MAX);  /* an end first */
      live--;
      at_end++;
      continue;
    }
    r = sw_at(s->rank, x);
    below = sw_tree_below(s, r);
    if (below && sw_alias(s, x, sw_at(s->by_key, sw_tree_kth(s, below)))) return 1;
    if (below < live && sw_alias(s, x, sw_at(s->by_key, sw_tree_kth(s, below + 1u)))) return 1;
    sw_tree_add(s, r, 1u);
    live++;
    at_start++;
  }
}

/* The first of the sweep's n records that aliases an earlier one, n when none does. */
static uint32_t ep_sweep_first_alias(ep_sweep *s) {
  uint32_t lo, hi;
  size_t i;
  if (s->n < 2u) return s->n;
  sw_sort(s, s->by_key, 0);
  sw_sort(s, s->by_first, 1);
  sw_sort(s, s->by_last, 2);
  for (i = 0; i < s->n; i++) sw_set(s->rank, sw_at(s->by_key, i), (uint32_t)i);
  if (!sw_prefix_aliases(s, s->n)) return s->n;
  lo = 2u;                                          /* [0, lo - 1) holds no pair, [0, hi) one */
  hi = s->n;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2u;
    if (sw_prefix_aliases(s, mid)) hi = mid; else lo = mid + 1u;
  }
  return lo - 1u;
}

/* The placement slot of step `cid`: found in the validated step prefix (no heap). */
static int ep_step_slot(const uint8_t *data, size_t body_len, size_t steps_start,
                        uint32_t n_steps, uint64_t cid, uint64_t *start, uint64_t *finish) {
  cur c; c.d = data; c.len = body_len; c.pos = steps_start; c.err = 0;
  for (uint32_t i = 0; i < n_steps && !c.err; i++) {
    uint64_t claim = c_u64(&c);
    uint64_t s, d;
    c_skip(&c, 4);                      /* phase_id */
    c_skip_str(&c);                     /* candidate */
    c_skip(&c, 1 + 4 + 8 + 4);          /* lane, width, cost, stream */
    s = c_u64(&c);
    d = c_u64(&c);
    if (c.err) return 0;
    if (claim == cid) { *start = s; *finish = s + d; return 1; }
  }
  return 0;
}

/* The bank of resource `rid`'s lifetime, if the plan carries one (no heap). */
static int ep_lifetime_bank(const uint8_t *data, size_t body_len, size_t lifetimes_start,
                            uint32_t n_lifetimes, uint16_t version, uint32_t rid,
                            const char **bank, uint16_t *bank_len) {
  cur c; c.d = data; c.len = body_len; c.pos = lifetimes_start; c.err = 0;
  for (uint32_t i = 0; i < n_lifetimes && !c.err; i++) {
    bcir_ep_lifetime_view v;
    ep_read_lifetime(&c, version, &v);
    if (c.err) return 0;
    if (v.rid == rid) { *bank = v.bank; *bank_len = v.bank_len; return 1; }
  }
  return 0;
}

/* The v3 move laws over one edge (see bcir_execution_plan.h). */
static bcir_status ep_check_move_v3(const uint8_t *data, size_t body_len, size_t steps_start,
                                    size_t lifetimes_start, const bcir_ep_header *hdr,
                                    const bcir_ep_move_view *v) {
  uint64_t cs = 0, cf = 0, s, f;
  const int has_claim = (v->flags & BCIR_EP_MOVE_HAS_CLAIM) != 0u;
  if ((v->flags & ~BCIR_EP_MOVE_FLAGS) != 0u) return BCIR_ERR_PLAN;
  if (!(v->flags & BCIR_EP_MOVE_HAS_AFTER) && v->after_claim) return BCIR_ERR_PLAN;
  if (!(v->flags & BCIR_EP_MOVE_HAS_BEFORE) && v->before_claim) return BCIR_ERR_PLAN;
  if (!(v->flags & BCIR_EP_MOVE_HAS_PRODUCER) && v->producer) return BCIR_ERR_PLAN;
  if (!has_claim && v->claim) return BCIR_ERR_PLAN;
  if (((v->flags & BCIR_EP_MOVE_HAS_PRODUCER) != 0u) !=
      (v->kind == (uint8_t)BCIR_EP_MOVE_REMATERIALIZED))
    return BCIR_ERR_PLAN;
  if ((v->kind == (uint8_t)BCIR_EP_MOVE_COMPRESSED) != (v->bits >= 1u && v->bits <= 31u))
    return BCIR_ERR_PLAN;
  if (v->kind != (uint8_t)BCIR_EP_MOVE_COMPRESSED && v->bits) return BCIR_ERR_PLAN;
  if ((v->cert != 0u) != (v->kind == (uint8_t)BCIR_EP_MOVE_REMATERIALIZED ||
                          v->kind == (uint8_t)BCIR_EP_MOVE_COMPRESSED))
    return BCIR_ERR_PLAN;
  if (has_claim &&
      !ep_step_slot(data, body_len, steps_start, hdr->n_steps, v->claim, &cs, &cf))
    return BCIR_ERR_PLAN;
  if (v->flags & BCIR_EP_MOVE_HAS_PRODUCER) {
    if (!ep_step_slot(data, body_len, steps_start, hdr->n_steps, v->producer, &s, &f))
      return BCIR_ERR_PLAN;
    if (has_claim && v->producer == v->claim) return BCIR_ERR_PLAN;
  }
  if (v->flags & BCIR_EP_MOVE_HAS_AFTER) {
    if (!ep_step_slot(data, body_len, steps_start, hdr->n_steps, v->after_claim, &s, &f))
      return BCIR_ERR_PLAN;
    if (has_claim && f > cs) return BCIR_ERR_PLAN;        /* the source's writer finishes first */
  }
  if (v->flags & BCIR_EP_MOVE_HAS_BEFORE) {
    if (!ep_step_slot(data, body_len, steps_start, hdr->n_steps, v->before_claim, &s, &f))
      return BCIR_ERR_PLAN;
    if (has_claim && cf > s) return BCIR_ERR_PLAN;        /* ... and the first reader starts after */
  }
  if (v->coherence == (uint8_t)BCIR_EP_COHERENCE_WRITEBACK) {
    const char *home; uint16_t home_len;
    if (ep_lifetime_bank(data, body_len, lifetimes_start, hdr->n_lifetimes, hdr->version,
                         v->rid, &home, &home_len) &&
        !str_eq(home, home_len, v->dst_bank, v->dst_len))
      return BCIR_ERR_PLAN;                                /* a writeback lands home */
  }
  return BCIR_OK;
}

typedef struct ep_callbacks {
  bcir_ep_step_fn step; void *step_ctx;
  bcir_ep_lifetime_fn lifetime; void *lifetime_ctx;
  bcir_ep_move_fn move; void *move_ctx;
  bcir_gen_fn gen; void *gen_ctx;
  uint8_t *scratch; size_t scratch_len;  /* ALIAS-SWEEP: the alias law's sweep, when lent */
} ep_callbacks;

/* The one walk: validate, then every record family under the wire laws, invoking the
 * callbacks a caller set. Every declared record is parsed even after a callback stops. */
static bcir_status ep_walk(const uint8_t *BCIR_RESTRICT data, size_t len,
                           const ep_callbacks *cb) {
  bcir_ep_header hdr;
  bcir_status st = bcir_ep_validate(data, len, &hdr);
  if (st != BCIR_OK) return st;
  if (hdr.streams == 0u || hdr.knee == 0u || hdr.knee > hdr.streams) return BCIR_ERR_PLAN;
  {
    const size_t body_len = len - 4u;
    cur c; c.d = data; c.len = body_len; c.pos = (size_t)BCIR_EP_HEADER_SIZE; c.err = 0;
    int invoke_step = 1, invoke_mv = 1, invoke_gen = 1;
    size_t steps_start;
    uint64_t claim_max = 0;           /* the greatest step claim id so far */
    uint32_t i;
    c_skip_str(&c);                     /* source_plan */
    if (c.err) return c_status(&c);
    steps_start = c.pos;
    for (i = 0; i < hdr.n_steps; i++) {
      bcir_ep_step_view v;
      uint64_t raw;
      v.claim_id = c_u64(&c);
      v.phase_id = c_u32(&c);
      v.candidate = c_str(&c, &v.candidate_len);
      v.lane = c_u8(&c);
      v.width = c_u32(&c);
      raw = c_u64(&c);
      v.cost = ep_i64(raw);
      v.stream = c_u32(&c);
      v.start = c_u64(&c);
      v.duration = c_u64(&c);
      if (c.err) return c_status(&c);
      if (v.lane > (uint8_t)BCIR_LANE_H) return BCIR_ERR_LANE;
      if (v.width == 0u || (v.width & (v.width - 1u)) != 0u) return BCIR_ERR_WIDTH;
      if (v.stream != BCIR_EP_TAIL_STREAM && v.stream >= hdr.streams) return BCIR_ERR_PLAN;
      {
        uint64_t want = v.cost > 0 ? (uint64_t)v.cost : 0u;
        if (v.duration != want) return BCIR_ERR_PLAN;
      }
      if (v.start > UINT64_MAX - v.duration) return BCIR_ERR_OVERFLOW;
      if (v.start + v.duration > hdr.makespan) return BCIR_ERR_PLAN;
      if (i == 0 || v.claim_id > claim_max)
        claim_max = v.claim_id;         /* steps no earlier step's claim */
      else if (ep_claim_seen_before(data, body_len, steps_start, i, v.claim_id))
        return BCIR_ERR_PROVENANCE;
      if (invoke_step && cb && cb->step && cb->step(&v, cb->step_ctx)) invoke_step = 0;
    }
    size_t lifetimes_start = c.pos;
    {
      /* each record's own laws first, to the first that breaks one (`own` its status), then the
       * alias law over the records before it: the first that aliases an earlier one is refused
       * (BCIR_ERR_PLAN) ahead of any later record's own law -- the verdict of one walk checking
       * both record by record. The lifetime callback sees the records before that verdict, in
       * order, as such a walk hands them over. */
      uint32_t prev = 0, k, first;
      int have_prev = 0;
      bcir_status own = BCIR_OK;
      ep_sweep sw;
      const int lent = cb && ep_sweep_lend(&sw, cb->scratch, cb->scratch_len, data,
                                           hdr.n_lifetimes);
      for (k = 0; k < hdr.n_lifetimes; k++) {
        bcir_ep_lifetime_view v;
        ep_read_lifetime(&c, hdr.version, &v);
        own = ep_lifetime_own_laws(&c, &v, have_prev, prev);
        if (own != BCIR_OK) break;
        prev = v.rid; have_prev = 1;
        if (lent) ep_sweep_put(&sw, k, &v);
      }
      /* no two lifetimes of one bank live at once at overlapping addresses */
      if (lent) {
        sw.n = k;
        first = ep_sweep_first_alias(&sw);
      } else {
        first = ep_first_alias_rewalk(data, body_len, lifetimes_start, k, hdr.version);
      }
      if (cb && cb->lifetime) {
        cur r; r.d = data; r.len = body_len; r.pos = lifetimes_start; r.err = 0;
        for (i = 0; i < first; i++) {
          bcir_ep_lifetime_view v;
          ep_read_lifetime(&r, hdr.version, &v);
          if (cb->lifetime(&v, cb->lifetime_ctx)) break;
        }
      }
      if (first < k) return BCIR_ERR_PLAN;
      if (own != BCIR_OK) return own;
    }
    for (i = 0; i < hdr.n_moves; i++) {
      bcir_ep_move_view v;
      v.rid = c_u32(&c);
      v.src_bank = c_str(&c, &v.src_len);
      v.dst_bank = c_str(&c, &v.dst_len);
      v.offset = c_u64(&c);
      v.size = c_u64(&c);
      v.route = c_str(&c, &v.route_len);
      v.kind = c_u8(&c);
      v.coherence = c_u8(&c);
      v.map_gen = c_u32(&c);
      v.data_gen = c_u32(&c);
      v.after_claim = c_u64(&c);
      v.before_claim = c_u64(&c);
      v.claim = 0; v.version = 0; v.flags = 0; v.producer = 0; v.bits = 0; v.cert = 0;
      if (hdr.version >= 3u) {
        v.claim = c_u64(&c);
        v.version = c_u32(&c);
        v.flags = c_u8(&c);
        v.producer = c_u64(&c);
        v.bits = c_u8(&c);
        v.cert = c_u64(&c);
      }
      if (c.err) return c_status(&c);
      if (v.src_len == 0u || v.dst_len == 0u || v.size == 0u) return BCIR_ERR_PLAN;
      if (v.offset > UINT64_MAX - v.size) return BCIR_ERR_OVERFLOW;
      if (v.kind > (uint8_t)BCIR_EP_MOVE_KIND_MAX) return BCIR_ERR_PLAN;
      if (v.coherence > (uint8_t)BCIR_EP_COHERENCE_MAX) return BCIR_ERR_PLAN;
      /* a move moves between two banks; a remat replays its producer in place */
      if (str_eq(v.src_bank, v.src_len, v.dst_bank, v.dst_len) !=
          (v.kind == (uint8_t)BCIR_EP_MOVE_REMATERIALIZED))
        return BCIR_ERR_PLAN;
      if (v.coherence == (uint8_t)BCIR_EP_COHERENCE_WRITEBACK &&
          (v.kind == (uint8_t)BCIR_EP_MOVE_COMPRESSED ||
           v.kind == (uint8_t)BCIR_EP_MOVE_REMATERIALIZED))
        return BCIR_ERR_PLAN;
      if (hdr.version >= 3u) {
        bcir_status mst = ep_check_move_v3(data, body_len, steps_start, lifetimes_start, &hdr, &v);
        if (mst != BCIR_OK) return mst;
      }
      if (invoke_mv && cb && cb->move && cb->move(&v, cb->move_ctx)) invoke_mv = 0;
    }
    {
      uint32_t prev = 0; int have_prev = 0;
      for (i = 0; i < hdr.n_gens; i++) {
        bcir_generation_view g;
        g.rid = c_u32(&c); g.map_gen = c_u32(&c); g.data_gen = c_u32(&c);
        if (c.err) return c_status(&c);
        if (have_prev && g.rid <= prev) return BCIR_ERR_GENERATION;
        prev = g.rid; have_prev = 1;
        if (invoke_gen && cb && cb->gen && cb->gen(&g, cb->gen_ctx)) invoke_gen = 0;
      }
    }
    if (hdr.version >= 3u) {            /* the binding trailer: the source and the spec */
      uint64_t source_hash = c_u64(&c), spec_hash = c_u64(&c);
      if (c.err) return c_status(&c);
      if (source_hash == 0u || spec_hash == 0u) return BCIR_ERR_PLAN;
    }
    if (c.pos != body_len) return BCIR_ERR_TRAILING;
  }
  return BCIR_OK;
}

bcir_status bcir_ep_verify(const uint8_t *BCIR_RESTRICT data, size_t len) {
  return ep_walk(data, len, 0);
}

bcir_status bcir_ep_verify_lent(const uint8_t *BCIR_RESTRICT data, size_t len, void *scratch,
                                size_t scratch_len) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.scratch = (uint8_t *)scratch; cb.scratch_len = scratch_len;
  return ep_walk(data, len, &cb);
}

const char *bcir_ep_source_plan(const uint8_t *BCIR_RESTRICT data, size_t len,
                                uint16_t *BCIR_RESTRICT out_len) {
  cur c;
  const char *s;
  uint16_t n = 0;
  if (out_len) *out_len = 0;
  if (bcir_ep_verify(data, len) != BCIR_OK) return 0;
  c.d = data; c.len = len - 4u; c.pos = (size_t)BCIR_EP_HEADER_SIZE; c.err = 0;
  s = c_str(&c, &n);
  if (c.err) return 0;
  if (out_len) *out_len = n;
  return s;
}

bcir_status bcir_ep_for_each_step(const uint8_t *BCIR_RESTRICT data, size_t len,
                                  bcir_ep_step_fn fn, void *ctx) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.step = fn; cb.step_ctx = ctx;
  return ep_walk(data, len, &cb);
}

bcir_status bcir_ep_for_each_lifetime(const uint8_t *BCIR_RESTRICT data, size_t len,
                                      bcir_ep_lifetime_fn fn, void *ctx) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.lifetime = fn; cb.lifetime_ctx = ctx;
  return ep_walk(data, len, &cb);
}

bcir_status bcir_ep_for_each_move(const uint8_t *BCIR_RESTRICT data, size_t len,
                                  bcir_ep_move_fn fn, void *ctx) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.move = fn; cb.move_ctx = ctx;
  return ep_walk(data, len, &cb);
}

bcir_status bcir_ep_for_each_generation(const uint8_t *BCIR_RESTRICT data, size_t len,
                                        bcir_gen_fn fn, void *ctx) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.gen = fn; cb.gen_ctx = ctx;
  return ep_walk(data, len, &cb);
}

/* Where a verified plan's generation vector ends: the body's end (the CRC), before the v3
 * binding trailer. The one predicate every reader of the vector locates it by. */
static size_t ep_generations_end(size_t len, const bcir_ep_header *hdr) {
  return len - 4u - (hdr->version >= 3u ? (size_t)EP_BINDING_SIZE : 0u);
}

/* The vector is the body's tail (the walk verified the declared records end exactly at the
 * CRC, or at the v3 trailer), located by arithmetic after the walk, as the StreamPack's is. */
static bcir_status ep_locate_generations(const uint8_t *BCIR_RESTRICT data, size_t len,
                                         bcir_ep_header *hdr, size_t *start) {
  bcir_status st = bcir_ep_verify(data, len);
  if (st != BCIR_OK) return st;
  st = bcir_ep_validate(data, len, hdr);
  if (st != BCIR_OK) return st;
  *start = 0;
  if (hdr->n_gens == 0u) return BCIR_OK;
  {
    size_t body_len = ep_generations_end(len, hdr);
    size_t room = body_len - (size_t)BCIR_EP_HEADER_SIZE;
    if ((size_t)hdr->n_gens > room / BCIR_GENERATION_WIRE_SIZE) return BCIR_ERR_TRUNCATED;
    *start = body_len - (size_t)hdr->n_gens * BCIR_GENERATION_WIRE_SIZE;
  }
  return BCIR_OK;
}

bcir_status bcir_ep_binding(const uint8_t *BCIR_RESTRICT data, size_t len,
                            uint64_t *BCIR_RESTRICT source_hash,
                            uint64_t *BCIR_RESTRICT spec_hash) {
  bcir_ep_header hdr;
  bcir_status st;
  if (source_hash) *source_hash = 0;
  if (spec_hash) *spec_hash = 0;
  st = bcir_ep_verify(data, len);
  if (st != BCIR_OK) return st;
  st = bcir_ep_validate(data, len, &hdr);
  if (st != BCIR_OK) return st;
  if (hdr.version >= 3u) {              /* verified: the trailer ends exactly at the CRC */
    const uint8_t *tail = data + len - 4u - (size_t)EP_BINDING_SIZE;
    if (source_hash) *source_hash = rd64(tail);
    if (spec_hash) *spec_hash = rd64(tail + 8);
  }
  return BCIR_OK;
}

bcir_status bcir_ep_check_generation_vector(const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
                                            const bcir_generation_view *BCIR_RESTRICT live,
                                            size_t n_live) {
  bcir_ep_header hdr;
  size_t start;
  bcir_status st = ep_locate_generations(plan, plan_len, &hdr, &start);
  if (st != BCIR_OK) return st;
  if (n_live && !live) return BCIR_ERR_NOSPACE;
  if (hdr.n_gens == 0u) return n_live ? BCIR_ERR_STALE : BCIR_OK;
  if (n_live == 0) return BCIR_ERR_STALE;
  {
    uint32_t i;
    size_t j;
    for (i = 0; i < hdr.n_gens; i++) {
      bcir_generation_view g;
      read_generation(plan + start + (size_t)i * BCIR_GENERATION_WIRE_SIZE, &g);
      for (j = 0; j < n_live; j++)
        if (live[j].rid == g.rid) break;
      if (j == n_live) return BCIR_ERR_STALE;                    /* an undeclared RID */
      if (live[j].map_gen != g.map_gen || live[j].data_gen != g.data_gen)
        return BCIR_ERR_STALE;                                    /* moved: repack/replan */
    }
    for (j = 0; j < n_live; j++) {
      for (i = 0; i < hdr.n_gens; i++)
        if (rd32(plan + start + (size_t)i * BCIR_GENERATION_WIRE_SIZE) == live[j].rid) break;
      if (i == hdr.n_gens) return BCIR_ERR_STALE;                /* declared after the plan */
    }
  }
  return BCIR_OK;
}

static bcir_status ep_check_pack(const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
                                 const uint8_t *BCIR_RESTRICT pack, size_t pack_len,
                                 const ep_callbacks *cb) {
  bcir_ep_header ph;
  bcir_streampack_header sh;
  bcir_status st = ep_walk(plan, plan_len, cb);
  if (st != BCIR_OK) return st;
  st = bcir_sp_verify_semantic(pack, pack_len, 0xFFFFFFFFu, 0xFFFFFFFFu);
  if (st != BCIR_OK) return st;
  st = bcir_ep_validate(plan, plan_len, &ph);
  if (st != BCIR_OK) return st;
  st = bcir_sp_validate(pack, pack_len, &sh);
  if (st != BCIR_OK) return st;
  {
    cur a; cur b;
    uint16_t la, lb;
    const char *sa; const char *sb;
    uint32_t i;
    a.d = plan; a.len = plan_len - 4u; a.pos = (size_t)BCIR_EP_HEADER_SIZE; a.err = 0;
    b.d = pack; b.len = pack_len - 4u; b.pos = (size_t)BCIR_STREAMPACK_HEADER_SIZE; b.err = 0;
    sa = c_str(&a, &la);
    sb = c_str(&b, &lb);
    if (a.err) return c_status(&a);
    if (b.err) return c_status(&b);
    if (!str_eq(sa, la, sb, lb)) return BCIR_ERR_PROVENANCE;      /* another plan's pack */
    if (ph.n_steps != sh.n_segments) return BCIR_ERR_PROVENANCE;
    for (i = 0; i < ph.n_steps; i++) {
      uint64_t pc, sc; uint32_t pp, sp, pw, sw; uint8_t pl, sl;
      /* plan step head */
      pc = c_u64(&a); pp = c_u32(&a); c_skip_str(&a); pl = c_u8(&a); pw = c_u32(&a);
      c_skip(&a, 8 + 4 + 8 + 8);                                   /* cost, stream, start, duration */
      /* pack segment head (hydrate emits one segment per step, in step order) */
      c_skip_str(&b);                                              /* name */
      sc = c_u64(&b); sp = c_u32(&b); sl = c_u8(&b); sw = c_u32(&b);
      c_skip(&b, 4);                                               /* stride_k */
      c_skip_str(&b);                                              /* opcode */
      { uint16_t n; (void)c_u32arr(&b, &n); }                      /* reads */
      { uint16_t n; (void)c_u32arr(&b, &n); }                      /* writes */
      c_skip_str(&b);                                              /* prefetch */
      c_skip_strarr(&b);                                           /* fence_before */
      c_skip_strarr(&b);                                           /* fence_after */
      if (sh.version >= 3) { c_skip(&b, 1); c_skip_str(&b); }      /* dispatch, channel */
      if (a.err) return c_status(&a);
      if (b.err) return c_status(&b);
      if (pc != sc || pp != sp || pl != sl || pw != sw) return BCIR_ERR_PROVENANCE;
    }
  }
  {
    /* R11 across the artifacts: the pack must carry the vector the plan was placed under. */
    uint32_t pn = ph.n_gens;
    uint32_t sn = sh.version >= 4 ? sh.n_gens : 0u;
    size_t ps, ss, k;
    if (pn != sn) return BCIR_ERR_STALE;
    ps = ep_generations_end(plan_len, &ph) - (size_t)pn * BCIR_GENERATION_WIRE_SIZE;
    ss = (pack_len - 4u) - (size_t)sn * BCIR_GENERATION_WIRE_SIZE;
    for (k = 0; k < (size_t)pn * BCIR_GENERATION_WIRE_SIZE; k++)
      if (plan[ps + k] != pack[ss + k]) return BCIR_ERR_STALE;
  }
  return BCIR_OK;
}

bcir_status bcir_ep_check_pack(const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
                               const uint8_t *BCIR_RESTRICT pack, size_t pack_len) {
  return ep_check_pack(plan, plan_len, pack, pack_len, 0);
}

bcir_status bcir_ep_check_pack_lifetimes(const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
                                         const uint8_t *BCIR_RESTRICT pack, size_t pack_len,
                                         bcir_ep_lifetime_fn fn, void *ctx) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.lifetime = fn; cb.lifetime_ctx = ctx;
  return ep_check_pack(plan, plan_len, pack, pack_len, &cb);
}

bcir_status bcir_ep_check_pack_walk(const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
                                    const uint8_t *BCIR_RESTRICT pack, size_t pack_len,
                                    bcir_ep_step_fn step, bcir_ep_lifetime_fn lifetime,
                                    void *ctx, void *scratch, size_t scratch_len) {
  ep_callbacks cb = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cb.step = step; cb.step_ctx = ctx;
  cb.lifetime = lifetime; cb.lifetime_ctx = ctx;
  cb.scratch = (uint8_t *)scratch; cb.scratch_len = scratch_len;
  return ep_check_pack(plan, plan_len, pack, pack_len, &cb);
}
