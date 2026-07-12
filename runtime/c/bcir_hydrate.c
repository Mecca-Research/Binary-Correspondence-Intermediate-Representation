/*===- bcir_hydrate.c - claim graph + plan -> StreamPack bytes -------------===*/
#include "bcir_hydrate.h"

/* freestanding byte ops (no <string.h> dependency, matching bcir_runtime.c). */
static void cp(uint8_t *d, const void *s, size_t n) {
  const uint8_t *p = (const uint8_t *)s; for (size_t i = 0; i < n; i++) d[i] = p[i];
}
static size_t slen(const char *s) { size_t n = 0; while (s && s[n]) n++; return n; }

/* --- a bounds-checked little-endian byte writer --------------------------- */
typedef struct { uint8_t *p; size_t cap, n; int err; } W;

static void w_bytes(W *w, const void *src, size_t n) {
  if (w->err || w->n > w->cap || n > w->cap - w->n || (!src && n)) { w->err = 1; return; }
  cp(w->p + w->n, src, n); w->n += n;
}
static void w_u8(W *w, uint8_t v) { w_bytes(w, &v, 1); }
static void w_u16(W *w, uint16_t v) { uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; w_bytes(w, b, 2); }
static void w_u32(W *w, uint32_t v) {
  uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
  w_bytes(w, b, 4);
}
static void w_u64(W *w, uint64_t v) { w_u32(w, (uint32_t)v); w_u32(w, (uint32_t)(v >> 32)); }
static void w_str(W *w, const char *s) {           /* str := u16 len + bytes */
  size_t n = slen(s); if(n>UINT16_MAX){w->err=1;return;} w_u16(w, (uint16_t)n); w_bytes(w, s, n);
}
static void w_u32arr(W *w, const uint32_t *a, uint16_t count) {  /* u16 count + count*u32 */
  w_u16(w, count); for (uint16_t i = 0; i < count; i++) w_u32(w, a[i]);
}

bcir_status bcir_hydrate(const bcir_func *f, const bcir_plan *plan,
                         uint8_t *buf, size_t cap, size_t *out_len) {
  if(!f||!out_len||(!buf&&cap)||f->n_claims>UINT32_MAX)return BCIR_ERR_NOSPACE;
  W w = {buf, cap, 0, 0};

  /* control-flow markers (opcode NOP) are emit-only; they are not realizable segments. */
  uint32_t n_seg = 0;
  for (size_t i = 0; i < f->n_claims; i++) if (f->claims[i].opcode != BCIR_OP_NOP) n_seg++;

  /* header (64 bytes) */
  w_bytes(&w, BCIR_STREAMPACK_MAGIC, 4);
  w_u16(&w, BCIR_STREAMPACK_VERSION);
  w_u16(&w, 0);                       /* flags */
  w_u32(&w, 0); w_u32(&w, 0); w_u32(&w, 0);          /* topo/map/data gen */
  w_u32(&w, n_seg);                   /* n_segments (realizable claims only) */
  /* one trace note PER realizable segment (claim_id provenance), mirroring Python hydrate:
   * R10 requires every segment's claim_id to resolve to a decoded trace record, so the C
   * hydrate must emit them too -- else bcir_sp_verify_semantic / the checked executor would
   * reject the pack this very function produced. */
  w_u32(&w, 0); w_u32(&w, 0); w_u32(&w, n_seg);      /* n_prefetches / n_blocks / n_trace */
  w_u16(&w, 0);                       /* pipeline_depth (v1) */
  for (int i = 0; i < 26; i++) w_u8(&w, 0);          /* reserved -> 64 bytes */

  /* body */
  w_str(&w, "");                      /* source_plan (empty) */
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    if (cl->opcode == BCIR_OP_NOP) continue;          /* skip a control-flow marker */
    uint32_t width = (plan && i < plan->n) ? plan->steps[i].width : (cl->count ? cl->count : 1);
    w_str(&w, cl->op);                /* name */
    w_u64(&w, cl->id);                /* claim_id */
    w_u32(&w, 0);                     /* phase_id (the C subset is single-phase) */
    w_u8(&w, cl->lane);
    w_u32(&w, width);
    w_u32(&w, 0);                     /* stride_k (scalar) */
    w_str(&w, cl->op);                /* opcode label */
    w_u32arr(&w, cl->rd, cl->n_rd);   /* reads */
    w_u32arr(&w, cl->wr, cl->n_wr);   /* writes */
    w_str(&w, "");                    /* prefetch */
    w_u16(&w, 0);                     /* fence_before (str_array, count 0) */
    w_u16(&w, 0);                     /* fence_after */
  }
  /* prefetches[0] / blocks[0]: nothing to write (counts are 0). */
  /* trace[n_seg]: one note per realizable segment, claim_id only (src/trace hash 0). */
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    if (cl->opcode == BCIR_OP_NOP) continue;
    w_u64(&w, cl->id);                /* claim_id */
    w_u64(&w, 0);                     /* src_hash */
    w_u64(&w, 0);                     /* trace_hash */
  }

  if (w.err) return BCIR_ERR_NOSPACE;

  /* trailer: CRC-32 of every preceding byte */
  uint32_t crc = bcir_crc32(buf, w.n);
  w_u32(&w, crc);
  if (w.err) return BCIR_ERR_NOSPACE;

  *out_len = w.n;
  return BCIR_OK;
}
