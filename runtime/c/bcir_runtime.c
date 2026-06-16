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

static uint8_t c_u8(cur *c) {
  if (c->pos + 1 > c->len) { c->err = 1; return 0; }
  return c->d[c->pos++];
}
static uint16_t c_u16(cur *c) {
  if (c->pos + 2 > c->len) { c->err = 1; return 0; }
  uint16_t v = rd16(c->d + c->pos); c->pos += 2; return v;
}
static uint32_t c_u32(cur *c) {
  if (c->pos + 4 > c->len) { c->err = 1; return 0; }
  uint32_t v = rd32(c->d + c->pos); c->pos += 4; return v;
}
static uint64_t c_u64(cur *c) {
  if (c->pos + 8 > c->len) { c->err = 1; return 0; }
  uint64_t v = rd64(c->d + c->pos); c->pos += 8; return v;
}
static const char *c_str(cur *c, uint16_t *out_len) {
  uint16_t n = c_u16(c);
  if (c->err || c->pos + n > c->len) { c->err = 1; *out_len = 0; return 0; }
  const char *p = (const char *)(c->d + c->pos);
  c->pos += n; *out_len = n; return p;
}
static const uint8_t *c_u32arr(cur *c, uint16_t *out_cnt) {
  uint16_t n = c_u16(c);
  if (c->err || c->pos + (size_t)n * 4u > c->len) { c->err = 1; *out_cnt = 0; return 0; }
  const uint8_t *p = c->d + c->pos;
  c->pos += (size_t)n * 4u; *out_cnt = n; return p;
}
static void c_skip_str(cur *c) {
  uint16_t n = c_u16(c);
  if (c->pos + n > c->len) c->err = 1; else c->pos += n;
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

bcir_status bcir_sp_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                             bcir_streampack_header *BCIR_RESTRICT hdr) {
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
    c_skip_str(&c);     /* prefetch */
    c_skip_strarr(&c);  /* fence_before */
    c_skip_strarr(&c);  /* fence_after */
    if (c.err) return BCIR_ERR_TRUNCATED;
    if (fn && fn(&v, ctx)) break;
  }
  return c.err ? BCIR_ERR_TRUNCATED : BCIR_OK;
}
