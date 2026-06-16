/*===- bcir_encode.c - freestanding StreamPack encoder (write side) -------===
 *
 * No libc: only <stddef.h>/<stdint.h> (via bcir_runtime.h). Little-endian wire encoding
 * byte by byte (host-endian-independent), mirroring bcir/abi/streampack_abi.py.
 *===----------------------------------------------------------------------===*/
#include "bcir_encode.h"

/* --- bounded little-endian reader (over the input body, [0, len) excludes CRC) --- */
typedef struct {
  const uint8_t *d;
  size_t len;
  size_t pos;
  int err;
} rcur;

static uint8_t r_u8(rcur *c) {
  if (c->pos + 1 > c->len) { c->err = 1; return 0; }
  return c->d[c->pos++];
}
static uint16_t r_u16(rcur *c) {
  if (c->pos + 2 > c->len) { c->err = 1; return 0; }
  uint16_t v = (uint16_t)((uint16_t)c->d[c->pos] | ((uint16_t)c->d[c->pos + 1] << 8));
  c->pos += 2; return v;
}
/* Borrow `n` raw bytes from the input at the cursor (bounds-checked); advance. */
static const uint8_t *r_take(rcur *c, size_t n) {
  if (c->pos + n > c->len) { c->err = 1; return 0; }
  const uint8_t *p = c->d + c->pos; c->pos += n; return p;
}

/* --- bounded little-endian writer (into the caller's output buffer) --- */
typedef struct {
  uint8_t *d;
  size_t cap;
  size_t pos;
  int err;
} wcur;

static void w_u8(wcur *w, uint8_t v) {
  if (w->pos + 1 > w->cap) { w->err = 1; return; }
  w->d[w->pos++] = v;
}
static void w_u16(wcur *w, uint16_t v) {
  if (w->pos + 2 > w->cap) { w->err = 1; return; }
  w->d[w->pos++] = (uint8_t)(v & 0xFF);
  w->d[w->pos++] = (uint8_t)((v >> 8) & 0xFF);
}
static void w_u32(wcur *w, uint32_t v) {
  if (w->pos + 4 > w->cap) { w->err = 1; return; }
  for (int i = 0; i < 4; i++) w->d[w->pos++] = (uint8_t)((v >> (8 * i)) & 0xFF);
}
static void w_bytes(wcur *w, const uint8_t *p, size_t n) {
  if (!p || w->pos + n > w->cap) { w->err = 1; return; }
  for (size_t i = 0; i < n; i++) w->d[w->pos++] = p[i];
}

/* Copy a length-prefixed string (u16 len + bytes) input -> output. */
static void copy_str(rcur *r, wcur *w) {
  uint16_t n = r_u16(r);
  const uint8_t *p = r_take(r, n);
  w_u16(w, n);
  w_bytes(w, p, n);
}
/* Copy a u32 array (u16 count + count*u32). */
static void copy_u32arr(rcur *r, wcur *w) {
  uint16_t n = r_u16(r);
  const uint8_t *p = r_take(r, (size_t)n * 4u);
  w_u16(w, n);
  w_bytes(w, p, (size_t)n * 4u);
}
/* Copy a u64 array (u16 count + count*u64). */
static void copy_u64arr(rcur *r, wcur *w) {
  uint16_t n = r_u16(r);
  const uint8_t *p = r_take(r, (size_t)n * 8u);
  w_u16(w, n);
  w_bytes(w, p, (size_t)n * 8u);
}
/* Copy a string array (u16 count + count strings). */
static void copy_strarr(rcur *r, wcur *w) {
  uint16_t n = r_u16(r);
  w_u16(w, n);
  for (uint16_t i = 0; i < n && !r->err && !w->err; i++)
    copy_str(r, w);
}

bcir_status bcir_sp_reencode(const uint8_t *BCIR_RESTRICT in, size_t in_len,
                             uint8_t *BCIR_RESTRICT out, size_t out_cap,
                             size_t *BCIR_RESTRICT out_len) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(in, in_len, &hdr);
  if (st != BCIR_OK)
    return st;

  rcur r = {in, in_len - 4u, (size_t)BCIR_STREAMPACK_HEADER_SIZE, 0};
  wcur w = {out, out_cap, 0, 0};

  /* Header: 64 bytes, frozen and already correct -- copy verbatim. */
  w_bytes(&w, in, (size_t)BCIR_STREAMPACK_HEADER_SIZE);

  /* Body: source_plan, then segments / prefetches / blocks / trace. */
  copy_str(&r, &w);  /* source_plan */
  for (uint32_t i = 0; i < hdr.n_segments && !r.err && !w.err; i++) {
    copy_str(&r, &w);             /* name */
    w_bytes(&w, r_take(&r, 8), 8); /* claim_id (u64) */
    w_bytes(&w, r_take(&r, 4), 4); /* phase_id (u32) */
    w_u8(&w, r_u8(&r));            /* lane (u8) */
    w_bytes(&w, r_take(&r, 4), 4); /* width (u32) */
    w_bytes(&w, r_take(&r, 4), 4); /* stride_k (u32, reserved 0) */
    copy_str(&r, &w);             /* opcode */
    copy_u32arr(&r, &w);          /* reads */
    copy_u32arr(&r, &w);          /* writes */
    copy_str(&r, &w);             /* prefetch */
    copy_strarr(&r, &w);          /* fence_before */
    copy_strarr(&r, &w);          /* fence_after */
  }
  for (uint32_t i = 0; i < hdr.n_prefetches && !r.err && !w.err; i++) {
    copy_str(&r, &w);             /* name */
    w_bytes(&w, r_take(&r, 4), 4); /* distance (u32) */
    copy_u32arr(&r, &w);          /* targets */
    copy_str(&r, &w);             /* hint */
    copy_str(&r, &w);             /* pattern */
    if (hdr.version >= 2)
      w_u8(&w, r_u8(&r));         /* v2: buffers (u8) */
  }
  for (uint32_t i = 0; i < hdr.n_blocks && !r.err && !w.err; i++) {
    w_bytes(&w, r_take(&r, 8), 8); /* base (u64) */
    w_bytes(&w, r_take(&r, 8), 8); /* count (u64) */
    copy_u64arr(&r, &w);          /* strides */
  }
  for (uint32_t i = 0; i < hdr.n_trace && !r.err && !w.err; i++) {
    w_bytes(&w, r_take(&r, 8), 8); /* claim_id (u64) */
    w_bytes(&w, r_take(&r, 8), 8); /* src_hash (u64) */
    w_bytes(&w, r_take(&r, 8), 8); /* trace_hash (u64) */
  }

  if (r.err)
    return BCIR_ERR_TRUNCATED;
  if (w.err)
    return BCIR_ERR_NOSPACE;

  /* CRC-32 of every preceding byte, appended (zlib-compatible, same as the decoder). */
  uint32_t crc = bcir_crc32(out, w.pos);
  w_u32(&w, crc);
  if (w.err)
    return BCIR_ERR_NOSPACE;

  if (out_len)
    *out_len = w.pos;
  return BCIR_OK;
}
