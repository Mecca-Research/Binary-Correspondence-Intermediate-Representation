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
  if (w->p) cp(w->p + w->n, src, n);
  w->n += n;
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

static int label_ok(const char *s, size_t cap) {
  for (size_t i = 0; i < cap; i++) {
    unsigned char ch = (unsigned char)s[i];
    if (ch == 0) return 1;
    if (ch < 0x20u || ch >= 0x7fu) return 0;
  }
  return 0;
}

/* The v4 binding of a frozen step (NULL for the frozen v1 bytes bcir_hydrate writes). */
typedef struct binding {
  uint32_t topo_gen, map_gen, data_gen;
  const bcir_generation_view *gens;
  size_t n_gens;
} binding;

static void emit_pack(W *w, const bcir_func *f, const bcir_plan *plan, uint32_t n_seg,
                      const binding *v4) {
  /* header (64 bytes) */
  w_bytes(w, BCIR_STREAMPACK_MAGIC, 4);
  w_u16(w, v4 ? (uint16_t)4u : (uint16_t)BCIR_STREAMPACK_VERSION);
  w_u16(w, 0);                       /* flags */
  if (v4) { w_u32(w, v4->topo_gen); w_u32(w, v4->map_gen); w_u32(w, v4->data_gen); }
  else { w_u32(w, 0); w_u32(w, 0); w_u32(w, 0); }  /* topo/map/data gen */
  w_u32(w, n_seg);                   /* n_segments (realizable claims only) */
  w_u32(w, 0); w_u32(w, 0); w_u32(w, n_seg);      /* n_prefetches / n_blocks / n_trace */
  if (v4) {
    w_u16(w, 1);                     /* pipeline_depth (v2+): one phase in flight */
    w_u16(w, 0);                     /* reserved 38..39 */
    w_u32(w, (uint32_t)v4->n_gens);  /* n_gens (v4) */
    for (int i = 0; i < 20; i++) w_u8(w, 0);        /* reserved -> 64 bytes */
  } else {
    w_u16(w, 0);                     /* pipeline_depth (v1) */
    for (int i = 0; i < 26; i++) w_u8(w, 0);        /* reserved -> 64 bytes */
  }

  w_str(w, "");                      /* source_plan (empty) */
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    if (cl->opcode == BCIR_OP_NOP) continue;
    uint32_t width = plan ? plan->steps[i].width : 1u;   /* no plan: the scalar realization */
    w_str(w, cl->op);                /* name */
    w_u64(w, cl->id);                /* claim_id */
    w_u32(w, 0);                     /* phase_id (the C subset is single-phase) */
    w_u8(w, cl->lane);
    w_u32(w, width);
    w_u32(w, 0);                     /* stride_k (scalar) */
    w_str(w, cl->op);                /* opcode label */
    w_u32arr(w, cl->rd, cl->n_rd);   /* reads */
    w_u32arr(w, cl->wr, cl->n_wr);   /* writes */
    w_str(w, "");                    /* prefetch */
    w_u16(w, 0);                     /* fence_before (str_array, count 0) */
    w_u16(w, 0);                     /* fence_after */
    if (v4) {
      w_u8(w, (uint8_t)BCIR_DISPATCH_CORE);   /* v3 tail: dispatch core ... */
      w_str(w, "host");                       /* ... on the host channel */
    }
  }
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    if (cl->opcode == BCIR_OP_NOP) continue;
    w_u64(w, cl->id);                /* claim_id */
    w_u64(w, 0);                     /* src_hash */
    w_u64(w, 0);                     /* trace_hash */
  }
  if (v4)
    for (size_t i = 0; i < v4->n_gens; i++) {
      w_u32(w, v4->gens[i].rid);
      w_u32(w, v4->gens[i].map_gen);
      w_u32(w, v4->gens[i].data_gen);
    }
}

static bcir_status emit_checked(const bcir_func *f, const bcir_plan *plan, uint32_t n_seg,
                                const binding *v4, uint8_t *buf, size_t cap, size_t *out_len);

bcir_status bcir_hydrate(const bcir_func *f, const bcir_plan *plan,
                         uint8_t *buf, size_t cap, size_t *out_len) {
  if (out_len) *out_len = 0;
  if(!f||!out_len||(!buf&&cap)||(f->n_claims&&!f->claims)||
     f->n_claims>UINT32_MAX)return BCIR_ERR_NOSPACE;
  if (plan && (plan->n != f->n_claims || (plan->n && !plan->steps)))
    return BCIR_ERR_PROVENANCE;

  /* control-flow markers (opcode NOP) are emit-only; they are not realizable segments. */
  uint32_t n_seg = 0;
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    if (cl->n_rd > BCIR_CLAIM_MAX_RD || cl->n_wr > BCIR_CLAIM_MAX_WR ||
        !label_ok(cl->op, sizeof cl->op)) return BCIR_ERR_PROVENANCE;
    if (cl->opcode == BCIR_OP_NOP) continue;
    n_seg++;
    if (cl->lane > BCIR_LANE_H) return BCIR_ERR_LANE;
    uint32_t width = plan ? plan->steps[i].width : 1u;   /* no plan: the scalar realization */
    if (plan && plan->steps[i].claim_id != cl->id) return BCIR_ERR_PROVENANCE;
    if (!width || (width & (width - 1u))) return BCIR_ERR_WIDTH;
  }

  /* Count first.  Capacity and every derived size are proved before the caller's
   * output buffer is touched, so an error never leaves a partial StreamPack. */
  return emit_checked(f, plan, n_seg, NULL, buf, cap, out_len);
}

/* Count, prove the capacity, then write: an error never leaves a partial StreamPack. */
static bcir_status emit_checked(const bcir_func *f, const bcir_plan *plan, uint32_t n_seg,
                                const binding *v4, uint8_t *buf, size_t cap, size_t *out_len) {
  W count = {NULL, SIZE_MAX, 0, 0};
  emit_pack(&count, f, plan, n_seg, v4);
  if (count.err || count.n > SIZE_MAX - 4u) return BCIR_ERR_OVERFLOW;
  size_t needed = count.n + 4u;
  if (cap < needed || (!buf && needed)) return BCIR_ERR_NOSPACE;

  W w = {buf, cap, 0, 0};
  emit_pack(&w, f, plan, n_seg, v4);
  if (w.err) return BCIR_ERR_NOSPACE;  /* unreachable after identical count pass */

  /* trailer: CRC-32 of every preceding byte */
  uint32_t crc = bcir_crc32(buf, w.n);
  w_u32(&w, crc);
  if (w.err) return BCIR_ERR_NOSPACE;

  *out_len = w.n;
  return BCIR_OK;
}

static int declared(const bcir_generation_view *gens, size_t n, uint32_t rid) {
  size_t lo = 0, hi = n;  /* the vector is strictly ascending: binary search */
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2u;
    if (gens[mid].rid == rid) return 1;
    if (gens[mid].rid < rid) lo = mid + 1u; else hi = mid;
  }
  return 0;
}

bcir_status bcir_hydrate_generations(const bcir_func *f, const bcir_plan *plan, uint32_t topo_gen,
                                     const bcir_generation_view *gens, size_t n_gens,
                                     uint8_t *buf, size_t cap, size_t *out_len) {
  binding v4;
  if (out_len) *out_len = 0;
  if(!f||!out_len||(!buf&&cap)||(f->n_claims&&!f->claims)||
     f->n_claims>UINT32_MAX)return BCIR_ERR_NOSPACE;
  if (plan && (plan->n != f->n_claims || (plan->n && !plan->steps)))
    return BCIR_ERR_PROVENANCE;
  if (!gens || n_gens == 0u || n_gens > UINT32_MAX) return BCIR_ERR_GENERATION;
  v4.topo_gen = topo_gen; v4.map_gen = 0; v4.data_gen = 0; v4.gens = gens; v4.n_gens = n_gens;
  for (size_t i = 0; i < n_gens; i++) {
    if (i && gens[i].rid <= gens[i - 1u].rid) return BCIR_ERR_GENERATION;
    if (gens[i].map_gen > v4.map_gen) v4.map_gen = gens[i].map_gen;
    if (gens[i].data_gen > v4.data_gen) v4.data_gen = gens[i].data_gen;
  }

  uint32_t n_seg = 0;
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    if (cl->n_rd > BCIR_CLAIM_MAX_RD || cl->n_wr > BCIR_CLAIM_MAX_WR ||
        !label_ok(cl->op, sizeof cl->op)) return BCIR_ERR_PROVENANCE;
    if (i && cl->id <= f->claims[i - 1u].id) return BCIR_ERR_PROVENANCE;  /* ids ascend */
    if (cl->opcode == BCIR_OP_NOP) continue;
    n_seg++;
    if (cl->lane > BCIR_LANE_H) return BCIR_ERR_LANE;
    uint32_t width = plan ? plan->steps[i].width : 1u;
    if (plan && plan->steps[i].claim_id != cl->id) return BCIR_ERR_PROVENANCE;
    if (!width || (width & (width - 1u))) return BCIR_ERR_WIDTH;
    for (uint8_t r = 0; r < cl->n_rd; r++)
      if (!declared(gens, n_gens, cl->rd[r])) return BCIR_ERR_PROVENANCE;
    for (uint8_t r = 0; r < cl->n_wr; r++)
      if (!declared(gens, n_gens, cl->wr[r])) return BCIR_ERR_PROVENANCE;
  }
  return emit_checked(f, plan, n_seg, &v4, buf, cap, out_len);
}
