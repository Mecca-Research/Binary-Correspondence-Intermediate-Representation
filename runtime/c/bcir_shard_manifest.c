/*===- bcir_shard_manifest.c - the manifest-of-shards (BSHM, version zero) ---===
 *
 * See bcir_shard_manifest.h. Every walk below runs over a pack bcir_sp_verify_semantic has
 * accepted, so the cursor's bounds checks are the second line, never the first. Every emitter
 * writes through one sink with three modes -- count, write, compare -- so the size of a
 * sub-pack, its bytes, and the check that a shard IS that sub-pack are one definition.
 *===----------------------------------------------------------------------===*/
#include "bcir_shard_manifest.h"

#include "bcir_control_plane.h" /* bcir_ctl_pack_registry_digest: the one registry binding */
#include "bcir_sha256.h"

static uint16_t rd16(const uint8_t *p) { return (uint16_t)((uint16_t)p[0] | (uint16_t)(p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static void wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void wr64(uint8_t *p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }
static void copy_bytes(uint8_t *d, const uint8_t *s, size_t n) {
  for (size_t i = 0; i < n; i++) d[i] = s[i];
}
static void zero_bytes(uint8_t *d, size_t n) {
  for (size_t i = 0; i < n; i++) d[i] = 0u;
}
static int bytes_eq(const uint8_t *a, const uint8_t *b, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i]) return 0;
  return 1;
}

/* --- a bounds-checked cursor over a pack body ------------------------------------------- */

typedef struct cur {
  const uint8_t *d;
  size_t len; /* the body: the CRC trailer is never read as a record */
  size_t pos;
  int err;
} cur;

static void c_skip(cur *c, size_t n) {
  if (c->err || c->pos > c->len || n > c->len - c->pos) { c->err = 1; return; }
  c->pos += n;
}
static const uint8_t *c_take(cur *c, size_t n) {
  const uint8_t *p = c->d + c->pos;
  c_skip(c, n);
  return c->err ? (const uint8_t *)0 : p;
}
static uint16_t c_u16(cur *c) {
  const uint8_t *p = c_take(c, 2u);
  return p ? rd16(p) : 0u;
}
static const uint8_t *c_str(cur *c, uint16_t *n) {
  *n = c_u16(c);
  return c_take(c, *n);
}
static void c_skip_str(cur *c) { c_skip(c, c_u16(c)); }
static void c_skip_arr(cur *c, size_t width) { c_skip(c, (size_t)c_u16(c) * width); }
static void c_skip_strarr(cur *c) {
  uint16_t n = c_u16(c);
  for (uint16_t i = 0; i < n && !c->err; i++) c_skip_str(c);
}

/* One v4 segment record; its claim id and prefetch name come back. */
static void c_segment(cur *c, uint64_t *claim, const uint8_t **pf, uint16_t *pf_len) {
  c_skip_str(c);                     /* name */
  const uint8_t *id = c_take(c, 8u); /* claim_id */
  *claim = id ? rd64(id) : 0u;
  c_skip(c, 4u + 1u + 4u + 4u);      /* phase_id, lane, width, stride_k */
  c_skip_str(c);                     /* opcode */
  c_skip_arr(c, 4u);                 /* reads */
  c_skip_arr(c, 4u);                 /* writes */
  *pf = c_str(c, pf_len);            /* prefetch ("" = none) */
  c_skip_strarr(c);                  /* fence_before */
  c_skip_strarr(c);                  /* fence_after */
  c_skip(c, 1u);                     /* v3: dispatch */
  c_skip_str(c);                     /* v3: channel */
}

/* One v2+ prefetch record; its name comes back. */
static const uint8_t *c_prefetch(cur *c, uint16_t *name_len) {
  const uint8_t *name = c_str(c, name_len);
  c_skip(c, 4u);        /* distance */
  c_skip_arr(c, 4u);    /* targets */
  c_skip_str(c);        /* hint */
  c_skip_str(c);        /* pattern */
  c_skip(c, 1u);        /* buffers */
  return name;
}

/* --- a verified pack's layout -------------------------------------------------------------- */

typedef struct pk {
  const uint8_t *d;
  size_t len;
  uint32_t n_seg, n_pf, n_blk, n_tr, n_gens;
  size_t plan_end; /* [64, plan_end) is the source plan */
  size_t seg_end;  /* [plan_end, seg_end) the segments */
  size_t pf_end;   /* [seg_end, pf_end) the prefetch records */
  size_t tr_start; /* [pf_end, tr_start) the blocks; then 24-byte trace records */
} pk;

/* Any pack bcir_sp_verify_semantic accepts, at version 4 with a vector. */
static bcir_status pk_open(const uint8_t *d, size_t len, pk *w) {
  cur c;
  uint64_t claim;
  const uint8_t *pf;
  uint16_t pf_len;
  if (bcir_sp_verify_semantic(d, len, 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_OK) return BCIR_ERR_SHARD;
  if (rd16(d + 4) != BCIR_SHM_PACK_VERSION) return BCIR_ERR_SHARD;
  w->d = d;
  w->len = len;
  w->n_seg = rd32(d + 20);
  w->n_pf = rd32(d + 24);
  w->n_blk = rd32(d + 28);
  w->n_tr = rd32(d + 32);
  w->n_gens = rd32(d + 40);
  if (w->n_gens == 0u) return BCIR_ERR_SHARD;
  c.d = d; c.len = len - 4u; c.pos = BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  c_skip_str(&c);
  w->plan_end = c.pos;
  for (uint32_t i = 0; i < w->n_seg && !c.err; i++) c_segment(&c, &claim, &pf, &pf_len);
  w->seg_end = c.pos;
  for (uint32_t i = 0; i < w->n_pf && !c.err; i++) (void)c_prefetch(&c, &pf_len);
  w->pf_end = c.pos;
  for (uint32_t i = 0; i < w->n_blk && !c.err; i++) { c_skip(&c, 16u); c_skip_arr(&c, 8u); }
  w->tr_start = c.pos;
  c_skip(&c, (size_t)w->n_tr * 24u);
  c_skip(&c, (size_t)w->n_gens * 12u);
  if (c.err || c.pos != len - 4u) return BCIR_ERR_SHARD;
  return BCIR_OK;
}

/* Advance a prefetch cursor to the record named `name`; its span comes back. 0 when the stream
 * ends first -- a name out of segment order, named twice, or undeclared. */
static int pf_find(cur *p, const uint8_t *name, uint16_t n, size_t *start, size_t *end) {
  while (!p->err && p->pos < p->len) {
    uint16_t got_len;
    size_t at = p->pos;
    const uint8_t *got = c_prefetch(p, &got_len);
    if (p->err) return 0;
    if (got_len == n && bytes_eq(got, name, n)) {
      *start = at;
      *end = p->pos;
      return 1;
    }
  }
  return 0;
}

/* The hydrated layout: one trace record per segment in segment order (S1); the prefetch records
 * the segments name appear in segment order, each named at most once (S2). */
static int pk_hydrated(const pk *w) {
  cur s, p;
  uint64_t claim;
  const uint8_t *name;
  uint16_t n;
  size_t a, b;
  if (w->n_tr != w->n_seg) return 0;
  s.d = w->d; s.len = w->seg_end; s.pos = w->plan_end; s.err = 0;
  p.d = w->d; p.len = w->pf_end; p.pos = w->seg_end; p.err = 0;
  for (uint32_t i = 0; i < w->n_seg; i++) {
    c_segment(&s, &claim, &name, &n);
    if (s.err || rd64(w->d + w->tr_start + (size_t)i * 24u) != claim) return 0;
    if (n != 0u && !pf_find(&p, name, n, &a, &b)) return 0;
  }
  return 1;
}

static bcir_status pk_whole(const uint8_t *d, size_t len, pk *w) {
  bcir_status st = pk_open(d, len, w);
  if (st != BCIR_OK) return st;
  return pk_hydrated(w) ? BCIR_OK : BCIR_ERR_SHARD;
}

/* --- one sink, three modes ------------------------------------------------------------------ */

typedef struct sink {
  uint8_t *out;       /* write mode */
  size_t cap;
  const uint8_t *cmp; /* compare mode */
  size_t cmp_len;
  size_t n;
  int err;            /* 1 no space, 2 mismatch */
} sink;

static void put(sink *s, const uint8_t *p, size_t n) {
  if (s->err) return;
  if (n > SIZE_MAX - s->n) { s->err = 1; return; }
  if (s->cmp) {
    if (s->n + n > s->cmp_len || !bytes_eq(s->cmp + s->n, p, n)) { s->err = 2; return; }
  } else if (s->out) {
    if (s->n + n > s->cap) { s->err = 1; return; }
    copy_bytes(s->out + s->n, p, n);
  }
  s->n += n;
}

/* The trailer: counted, written, or held to the compared bytes' own CRC. */
static void put_crc(sink *s) {
  uint8_t b[4];
  if (s->err) return;
  if (s->cmp) {
    if (s->cmp_len != s->n + 4u || rd32(s->cmp + s->n) != bcir_crc32(s->cmp, s->n)) s->err = 2;
    else s->n += 4u;
    return;
  }
  if (s->out) {
    if (s->n + 4u > s->cap) { s->err = 1; return; }
    wr32(b, bcir_crc32(s->out, s->n));
    copy_bytes(s->out + s->n, b, 4u);
  }
  s->n += 4u;
}

static void put_header(sink *s, const pk *w, uint32_t n_seg, uint32_t n_pf, uint32_t n_blk,
                       uint32_t n_tr) {
  uint8_t h[BCIR_STREAMPACK_HEADER_SIZE];
  copy_bytes(h, w->d, sizeof h);
  wr32(h + 20, n_seg);
  wr32(h + 24, n_pf);
  wr32(h + 28, n_blk);
  wr32(h + 32, n_tr);
  put(s, h, sizeof h);
}

/* The canonical sub-pack over [b, e) (a pass to count the named prefetch records, then one to
 * emit them). */
static void emit_sub(const pk *w, uint32_t b, uint32_t e, sink *s) {
  cur c, p;
  uint64_t claim;
  const uint8_t *name;
  uint16_t n;
  size_t seg_begin = w->plan_end, seg_stop, a, z;
  uint32_t n_pf = 0;
  c.d = w->d; c.len = w->seg_end; c.pos = w->plan_end; c.err = 0;
  for (uint32_t i = 0; i < e && !c.err; i++) {
    if (i == b) seg_begin = c.pos;
    c_segment(&c, &claim, &name, &n);
  }
  seg_stop = b == e ? seg_begin : c.pos;
  /* count the prefetch records the range names */
  c.pos = seg_begin;
  p.d = w->d; p.len = w->pf_end; p.pos = w->seg_end; p.err = 0;
  for (uint32_t i = b; i < e && !c.err; i++) {
    c_segment(&c, &claim, &name, &n);
    if (n != 0u) {
      if (!pf_find(&p, name, n, &a, &z)) { s->err = 2; return; }
      n_pf++;
    }
  }
  if (c.err) { s->err = 2; return; }
  put_header(s, w, e - b, n_pf, 0u, e - b);
  put(s, w->d + BCIR_STREAMPACK_HEADER_SIZE, w->plan_end - BCIR_STREAMPACK_HEADER_SIZE);
  put(s, w->d + seg_begin, seg_stop - seg_begin);
  c.pos = seg_begin;
  p.pos = w->seg_end;
  for (uint32_t i = b; i < e; i++) {
    c_segment(&c, &claim, &name, &n);
    if (n != 0u && pf_find(&p, name, n, &a, &z)) put(s, w->d + a, z - a);
  }
  put(s, w->d + w->tr_start + (size_t)b * 24u, (size_t)(e - b) * 24u);
  put(s, w->d + w->tr_start + (size_t)w->n_tr * 24u, (size_t)w->n_gens * 12u);
  put_crc(s);
}

/* The frame: the whole with no segments. */
static void emit_frame(const pk *w, sink *s) {
  put_header(s, w, 0u, w->n_pf, w->n_blk, w->n_tr);
  put(s, w->d + BCIR_STREAMPACK_HEADER_SIZE, w->plan_end - BCIR_STREAMPACK_HEADER_SIZE);
  put(s, w->d + w->seg_end, w->len - 4u - w->seg_end);
  put_crc(s);
}

typedef enum { EMIT_SUB, EMIT_FRAME } emit_kind;

static bcir_status emit_to(const uint8_t *whole, size_t len, emit_kind kind, uint32_t b, uint32_t e,
                           uint8_t *out, size_t cap, size_t *out_len) {
  pk w;
  sink s = {0, 0, 0, 0, 0, 0};
  bcir_status st;
  if (!out_len) return BCIR_ERR_NOSPACE;
  *out_len = 0;
  if (!whole) return BCIR_ERR_SHARD;
  st = pk_whole(whole, len, &w);
  if (st != BCIR_OK) return st;
  if (kind == EMIT_SUB && (b > e || e > w.n_seg)) return BCIR_ERR_SHARD;
  if (kind == EMIT_SUB) emit_sub(&w, b, e, &s); else emit_frame(&w, &s);
  if (s.err) return BCIR_ERR_SHARD;
  if (!out) {
    *out_len = s.n;
    return BCIR_OK;
  }
  if (cap < s.n) return BCIR_ERR_NOSPACE;
  {
    size_t need = s.n;
    sink wsink = {out, cap, 0, 0, 0, 0};
    if (kind == EMIT_SUB) emit_sub(&w, b, e, &wsink); else emit_frame(&w, &wsink);
    if (wsink.err || wsink.n != need) return BCIR_ERR_SHARD; /* unreachable: same definition */
    *out_len = need;
  }
  return BCIR_OK;
}

static int emitted_equals(const pk *w, emit_kind kind, uint32_t b, uint32_t e, const uint8_t *blob,
                          size_t blob_len) {
  sink s = {0, 0, blob, blob_len, 0, 0};
  if (kind == EMIT_SUB) emit_sub(w, b, e, &s); else emit_frame(w, &s);
  return !s.err && s.n == blob_len;
}

/* --- the public API --------------------------------------------------------------------- */

bcir_status bcir_shm_check_whole(const uint8_t *BCIR_RESTRICT whole, size_t len) {
  pk w;
  if (!whole) return BCIR_ERR_SHARD;
  return pk_whole(whole, len, &w);
}

bcir_status bcir_shm_sub_pack(const uint8_t *BCIR_RESTRICT whole, size_t len, uint32_t begin,
                              uint32_t end, uint8_t *BCIR_RESTRICT out, size_t cap,
                              size_t *BCIR_RESTRICT out_len) {
  return emit_to(whole, len, EMIT_SUB, begin, end, out, cap, out_len);
}

bcir_status bcir_shm_frame(const uint8_t *BCIR_RESTRICT whole, size_t len,
                           uint8_t *BCIR_RESTRICT out, size_t cap, size_t *BCIR_RESTRICT out_len) {
  return emit_to(whole, len, EMIT_FRAME, 0u, 0u, out, cap, out_len);
}

bcir_status bcir_shm_partition(uint32_t n_segments, uint32_t world, uint32_t *BCIR_RESTRICT begins,
                               uint32_t *BCIR_RESTRICT ends, uint32_t cap,
                               uint32_t *BCIR_RESTRICT count) {
  uint64_t w = world ? world : 1u, per, k;
  if (!count) return BCIR_ERR_NOSPACE;
  *count = 0;
  if (n_segments == 0u) {
    if (cap < 1u || !begins || !ends) return BCIR_ERR_NOSPACE;
    begins[0] = 0u;
    ends[0] = 0u;
    *count = 1u;
    return BCIR_OK;
  }
  per = ((uint64_t)n_segments + w - 1u) / w;
  k = ((uint64_t)n_segments + per - 1u) / per;
  if (k > BCIR_SHM_SHARDS_MAX) return BCIR_ERR_SHARD;
  if (k > cap || !begins || !ends) return BCIR_ERR_NOSPACE;
  for (uint64_t i = 0; i < k; i++) {
    uint64_t b = i * per, e = b + per < n_segments ? b + per : n_segments;
    begins[i] = (uint32_t)b;
    ends[i] = (uint32_t)e;
  }
  *count = (uint32_t)k;
  return BCIR_OK;
}

static void view_zero(bcir_shm_view *v) {
  v->data = 0; v->len = 0; v->n_shards = 0; v->n_segments = 0;
  v->whole_length = 0; v->frame_length = 0;
  v->map_gen = 0; v->data_gen = 0; v->topo_gen = 0; v->n_gens = 0;
  v->whole_sha256 = 0; v->frame_sha256 = 0; v->registry_digest = 0;
}

bcir_status bcir_shm_decode(const uint8_t *BCIR_RESTRICT data, size_t len,
                            bcir_shm_view *BCIR_RESTRICT out) {
  uint32_t n, n_seg, n_gens, expect = 0;
  uint64_t whole, frame, minimum;
  if (!out) return BCIR_ERR_TRUNCATED;
  view_zero(out);
  if (!data || len < BCIR_SHM_FIXED) return BCIR_ERR_TRUNCATED;
  if (data[0] != 'B' || data[1] != 'S' || data[2] != 'H' || data[3] != 'M') return BCIR_ERR_MAGIC;
  if (rd16(data + 4) != BCIR_SHM_VERSION) return BCIR_ERR_VERSION;
  if (rd32(data + len - 4u) != bcir_crc32(data, len - 4u)) return BCIR_ERR_CRC;
  if (rd16(data + 6) != 0u || rd16(data + 46) != 0u) return BCIR_ERR_RESERVED;
  for (size_t i = 52; i < 64; i++)
    if (data[i] != 0u) return BCIR_ERR_RESERVED;
  n = rd32(data + 8);
  if (n == 0u || n > BCIR_SHM_SHARDS_MAX) return BCIR_ERR_SHARD;
  if (len < BCIR_SHM_FIXED + (size_t)BCIR_SHM_ENTRY_SIZE * n) return BCIR_ERR_TRUNCATED;
  if (len > BCIR_SHM_FIXED + (size_t)BCIR_SHM_ENTRY_SIZE * n) return BCIR_ERR_TRAILING;
  n_gens = rd32(data + 48);
  if (rd16(data + 44) != BCIR_SHM_PACK_VERSION || n_gens == 0u) return BCIR_ERR_SHARD;
  n_seg = rd32(data + 12);
  whole = rd64(data + 16);
  frame = rd64(data + 24);
  minimum = (uint64_t)BCIR_STREAMPACK_HEADER_SIZE + 2u + 12u * (uint64_t)n_gens + 4u;
  if (whole > BCIR_SHM_BLOB_MAX || frame > BCIR_SHM_BLOB_MAX) return BCIR_ERR_SHARD;
  if (frame < minimum || whole < frame) return BCIR_ERR_SHARD;
  if ((n_seg == 0u) != (whole == frame)) return BCIR_ERR_SHARD;
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t *entry = data + BCIR_SHM_HEADER_SIZE + BCIR_SHM_DIGESTS_SIZE +
                           (size_t)BCIR_SHM_ENTRY_SIZE * i;
    uint32_t b = rd32(entry), e = rd32(entry + 4);
    uint64_t length = rd64(entry + 8);
    if (length < minimum || length > BCIR_SHM_BLOB_MAX) return BCIR_ERR_SHARD;
    if (b != expect) return BCIR_ERR_SHARD;
    if (e < b || (e == b && !(n_seg == 0u && n == 1u))) return BCIR_ERR_SHARD;
    expect = e;
  }
  if (expect != n_seg) return BCIR_ERR_SHARD;
  out->data = data;
  out->len = len;
  out->n_shards = n;
  out->n_segments = n_seg;
  out->whole_length = whole;
  out->frame_length = frame;
  out->map_gen = rd32(data + 32);
  out->data_gen = rd32(data + 36);
  out->topo_gen = rd32(data + 40);
  out->n_gens = n_gens;
  out->whole_sha256 = data + 64;
  out->frame_sha256 = data + 96;
  out->registry_digest = data + 128;
  return BCIR_OK;
}

bcir_status bcir_shm_entry_at(const bcir_shm_view *BCIR_RESTRICT m, uint32_t index,
                              bcir_shm_entry *BCIR_RESTRICT out) {
  const uint8_t *entry;
  if (!m || !out || !m->data || index >= m->n_shards) return BCIR_ERR_SHARD;
  entry = m->data + BCIR_SHM_HEADER_SIZE + BCIR_SHM_DIGESTS_SIZE +
          (size_t)BCIR_SHM_ENTRY_SIZE * index;
  out->seg_begin = rd32(entry);
  out->seg_end = rd32(entry + 4);
  out->length = rd64(entry + 8);
  copy_bytes(out->sha256, entry + 16, 32u);
  return BCIR_OK;
}

bcir_status bcir_shm_encode(const uint8_t *BCIR_RESTRICT whole, size_t whole_len,
                            const uint8_t *BCIR_RESTRICT frame, size_t frame_len,
                            const uint32_t *begins, const uint32_t *ends,
                            const uint8_t *const *shards, const size_t *shard_lens,
                            uint32_t n_shards, uint8_t *BCIR_RESTRICT out, size_t cap,
                            size_t *BCIR_RESTRICT out_len) {
  pk w;
  bcir_status st;
  bcir_shm_view check;
  size_t size;
  uint8_t *entry;
  if (!out_len) return BCIR_ERR_NOSPACE;
  *out_len = 0;
  if (!whole || !frame || (n_shards && (!begins || !ends || !shards || !shard_lens)))
    return BCIR_ERR_SHARD;
  if (n_shards == 0u || n_shards > BCIR_SHM_SHARDS_MAX) return BCIR_ERR_SHARD;
  st = pk_whole(whole, whole_len, &w);
  if (st != BCIR_OK) return st;
  if (!emitted_equals(&w, EMIT_FRAME, 0u, 0u, frame, frame_len)) return BCIR_ERR_SHARD;
  for (uint32_t i = 0; i < n_shards; i++) {
    if (!shards[i] || begins[i] > ends[i] || ends[i] > w.n_seg) return BCIR_ERR_SHARD;
    if (!emitted_equals(&w, EMIT_SUB, begins[i], ends[i], shards[i], shard_lens[i]))
      return BCIR_ERR_SHARD;
  }
  size = BCIR_SHM_FIXED + (size_t)BCIR_SHM_ENTRY_SIZE * n_shards;
  if (!out) {
    *out_len = size;
    return BCIR_OK;
  }
  if (cap < size) return BCIR_ERR_NOSPACE;
  zero_bytes(out, size);
  out[0] = 'B'; out[1] = 'S'; out[2] = 'H'; out[3] = 'M';
  wr16(out + 4, (uint16_t)BCIR_SHM_VERSION);
  wr32(out + 8, n_shards);
  wr32(out + 12, w.n_seg);
  wr64(out + 16, (uint64_t)whole_len);
  wr64(out + 24, (uint64_t)frame_len);
  wr32(out + 32, rd32(whole + 12)); /* map_gen */
  wr32(out + 36, rd32(whole + 16)); /* data_gen */
  wr32(out + 40, rd32(whole + 8));  /* topo_gen */
  wr16(out + 44, (uint16_t)BCIR_SHM_PACK_VERSION);
  wr32(out + 48, w.n_gens);
  bcir_sha256_digest(whole, whole_len, out + 64);
  bcir_sha256_digest(frame, frame_len, out + 96);
  st = bcir_ctl_pack_registry_digest(whole, whole_len, out + 128);
  if (st != BCIR_OK) { zero_bytes(out, size); return BCIR_ERR_SHARD; }
  for (uint32_t i = 0; i < n_shards; i++) {
    entry = out + BCIR_SHM_HEADER_SIZE + BCIR_SHM_DIGESTS_SIZE + (size_t)BCIR_SHM_ENTRY_SIZE * i;
    wr32(entry, begins[i]);
    wr32(entry + 4, ends[i]);
    wr64(entry + 8, (uint64_t)shard_lens[i]);
    bcir_sha256_digest(shards[i], shard_lens[i], entry + 16);
  }
  wr32(out + size - 4u, bcir_crc32(out, size - 4u));
  st = bcir_shm_decode(out, size, &check); /* the encoder refuses what the decoder refuses */
  if (st != BCIR_OK) { zero_bytes(out, size); return st; }
  *out_len = size;
  return BCIR_OK;
}

/* A blob of the declared length and digest, or 0. */
static const uint8_t *fetch_blob(bcir_shm_fetch_fn fetch, void *ctx, const uint8_t digest[32],
                                 uint64_t length) {
  const uint8_t *data = 0;
  size_t len = 0;
  uint8_t got[32];
  if (fetch(digest, &data, &len, ctx) != 0 || !data || (uint64_t)len != length) return 0;
  bcir_sha256_digest(data, len, got);
  return bcir_sha256_equal(got, digest, 32u) ? data : 0;
}

/* A shard blob's segment span, and that it is a v4 pack with exactly `count` segments. */
static int shard_span(const uint8_t *d, size_t len, uint32_t count, size_t *start, size_t *end) {
  cur c;
  uint64_t claim;
  const uint8_t *pf;
  uint16_t pf_len;
  if (bcir_sp_verify_semantic(d, len, 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_OK) return 0;
  if (rd16(d + 4) != BCIR_SHM_PACK_VERSION || rd32(d + 20) != count) return 0;
  c.d = d; c.len = len - 4u; c.pos = BCIR_STREAMPACK_HEADER_SIZE; c.err = 0;
  c_skip_str(&c);
  *start = c.pos;
  for (uint32_t i = 0; i < count && !c.err; i++) c_segment(&c, &claim, &pf, &pf_len);
  *end = c.pos;
  return !c.err;
}

bcir_status bcir_shm_reassemble(const uint8_t *BCIR_RESTRICT manifest, size_t manifest_len,
                                bcir_shm_fetch_fn fetch, void *ctx, uint8_t *BCIR_RESTRICT out,
                                size_t cap, size_t *BCIR_RESTRICT out_len) {
  bcir_shm_view m;
  bcir_shm_entry entry;
  const uint8_t *frame;
  size_t f_plan, f_seg, total, s_start, s_end;
  uint8_t digest[32];
  pk w;
  bcir_status st;
  if (!out_len) return BCIR_ERR_NOSPACE;
  *out_len = 0;
  st = bcir_shm_decode(manifest, manifest_len, &m);
  if (st != BCIR_OK) return st;
  if (!fetch) return BCIR_ERR_SHARD;
  frame = fetch_blob(fetch, ctx, m.frame_sha256, m.frame_length);
  if (!frame) return BCIR_ERR_SHARD;
  for (uint32_t i = 0; i < m.n_shards; i++) {
    (void)bcir_shm_entry_at(&m, i, &entry);
    if (!fetch_blob(fetch, ctx, entry.sha256, entry.length)) return BCIR_ERR_SHARD;
  }
  /* the frame: a v4 pack with no segments */
  if (!shard_span(frame, (size_t)m.frame_length, 0u, &f_plan, &f_seg)) return BCIR_ERR_SHARD;
  total = (size_t)m.frame_length;
  for (uint32_t i = 0; i < m.n_shards; i++) {
    const uint8_t *blob;
    (void)bcir_shm_entry_at(&m, i, &entry);
    blob = fetch_blob(fetch, ctx, entry.sha256, entry.length);
    if (!blob || !shard_span(blob, (size_t)entry.length, entry.seg_end - entry.seg_begin, &s_start,
                             &s_end))
      return BCIR_ERR_SHARD;
    if (s_end - s_start > (size_t)BCIR_SHM_BLOB_MAX - total) return BCIR_ERR_SHARD;
    total += s_end - s_start;
  }
  if ((uint64_t)total != m.whole_length) return BCIR_ERR_SHARD; /* the blobs prove the length */
  if (!out || cap < total) return BCIR_ERR_NOSPACE;
  {
    sink s = {out, cap, 0, 0, 0, 0};
    uint8_t h[BCIR_STREAMPACK_HEADER_SIZE];
    copy_bytes(h, frame, sizeof h);
    wr32(h + 20, m.n_segments);
    put(&s, h, sizeof h);
    put(&s, frame + BCIR_STREAMPACK_HEADER_SIZE, f_plan - BCIR_STREAMPACK_HEADER_SIZE);
    for (uint32_t i = 0; i < m.n_shards; i++) {
      const uint8_t *blob;
      (void)bcir_shm_entry_at(&m, i, &entry);
      blob = fetch_blob(fetch, ctx, entry.sha256, entry.length);
      if (!blob || !shard_span(blob, (size_t)entry.length, entry.seg_end - entry.seg_begin,
                               &s_start, &s_end)) {
        zero_bytes(out, s.n);
        return BCIR_ERR_SHARD; /* a store that answers differently the second time */
      }
      put(&s, blob + s_start, s_end - s_start);
    }
    put(&s, frame + f_seg, (size_t)m.frame_length - 4u - f_seg);
    put_crc(&s);
    if (s.err || s.n != total) { zero_bytes(out, s.n < cap ? s.n : cap); return BCIR_ERR_SHARD; }
  }
  bcir_sha256_digest(out, total, digest);
  st = BCIR_ERR_SHARD;
  if (bcir_sha256_equal(digest, m.whole_sha256, 32u) && pk_whole(out, total, &w) == BCIR_OK &&
      w.n_seg == m.n_segments && rd32(out + 12) == m.map_gen && rd32(out + 16) == m.data_gen &&
      rd32(out + 8) == m.topo_gen && w.n_gens == m.n_gens &&
      bcir_ctl_pack_registry_digest(out, total, digest) == BCIR_OK &&
      bcir_sha256_equal(digest, m.registry_digest, 32u) &&
      emitted_equals(&w, EMIT_FRAME, 0u, 0u, frame, (size_t)m.frame_length)) {
    st = BCIR_OK;
    for (uint32_t i = 0; i < m.n_shards && st == BCIR_OK; i++) {
      const uint8_t *blob;
      (void)bcir_shm_entry_at(&m, i, &entry);
      blob = fetch_blob(fetch, ctx, entry.sha256, entry.length);
      if (!blob ||
          !emitted_equals(&w, EMIT_SUB, entry.seg_begin, entry.seg_end, blob, (size_t)entry.length))
        st = BCIR_ERR_SHARD;
    }
  }
  if (st != BCIR_OK) {
    zero_bytes(out, total);
    return st;
  }
  *out_len = total;
  return BCIR_OK;
}
