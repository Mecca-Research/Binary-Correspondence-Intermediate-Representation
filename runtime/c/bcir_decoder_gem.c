/*===- bcir_decoder_gem.c - a decoder program, executed through GEM ------===*/
#include "bcir_decoder_gem.h"

#include "bcir_runtime.h"

#include <string.h>

/* One claim, decoded from its segment and block and resolved against the model. */
struct bcir_dgem_claim {
  uint8_t op, seen, n_rd, n_wr, final_norm;
  uint32_t rd[3], wr[2];
  uint64_t offset, count;
  const bcir_q8_tensor *weight; /* a projection's matrix, a norm's gain, the head */
  double *src, *dst;            /* a projection's / norm's activation operands */
  uint32_t in, out, layer;
};

/* A bound on the claim table, so a declared segment count cannot size an allocation
 * unchecked: 2^20 claims is ~18k tokens of a four-layer program. */
#define DGEM_MAX_CLAIMS (1u << 20)
#define DGEM_NONE ((size_t)-1)

static const char *const OP_NAMES[BCIR_DGEM_N_OPS] = {
    "dec.embed",     "dec.rmsnorm",    "dec.matvec", "dec.rope", "dec.kv_append",
    "dec.attention", "dec.matvec_add", "dec.swiglu", "dec.head", "dec.argmax"};

const char *bcir_dgem_op_name(bcir_dgem_op op) {
  return (unsigned)op < (unsigned)BCIR_DGEM_N_OPS ? OP_NAMES[op] : "";
}

static void *alloc_zero(bcir_host_allocator *a, size_t count, size_t size) {
  size_t bytes;
  void *p;
  if (!bcir_size_mul(count ? count : 1u, size, &bytes)) return NULL;
  p = bcir_host_allocate(a, bytes);
  if (p) memset(p, 0, bytes);
  return p;
}

void bcir_dgem_free(bcir_dgem *g) {
  bcir_host_allocator a;
  if (!g) return;
  a = g->allocator;
  bcir_host_deallocate(&a, g->claims);
  bcir_host_deallocate(&a, g->scratch);
  bcir_host_deallocate(&a, g->phases);
  bcir_host_deallocate(&a, g->tape);
  bcir_host_deallocate(&a, g->kv_rows);
  bcir_host_deallocate(&a, g->final_row);
  bcir_llama_ws_free(&g->ws);
  memset(g, 0, sizeof *g);
}

size_t bcir_dgem_capacity(const bcir_dgem *g) { return g ? g->capacity : 0u; }
size_t bcir_dgem_prompt_len(const bcir_dgem *g) { return g ? g->prompt_len : 0u; }

void bcir_dgem_reset_stats(bcir_dgem *g) {
  if (g) memset(&g->stats, 0, sizeof g->stats);
}

/* --- load: the segment and block walks fill the claim table ------------------------------ */

typedef struct load_ctx {
  bcir_dgem *g;
  uint32_t *order; /* segment index -> claim id: blocks pair with segments by index */
  size_t index;
} load_ctx;

static int on_segment(const bcir_segment_view *s, void *vctx) {
  load_ctx *lc = (load_ctx *)vctx;
  bcir_dgem *g = lc->g;
  struct bcir_dgem_claim *c;
  unsigned op;
  uint16_t i;
  if (lc->index >= g->n_claims || s->claim_id >= g->n_claims) return 1;
  c = &g->claims[s->claim_id];
  if (c->seen) return 1; /* a claim executes once */
  for (op = 0; op < (unsigned)BCIR_DGEM_N_OPS; ++op) {
    size_t n = strlen(OP_NAMES[op]);
    if (s->opcode_len == n && memcmp(s->opcode, OP_NAMES[op], n) == 0) break;
  }
  if (op == (unsigned)BCIR_DGEM_N_OPS || s->n_reads > 3u || s->n_writes > 2u) return 1;
  c->op = (uint8_t)op;
  c->seen = 1u;
  c->n_rd = (uint8_t)s->n_reads;
  c->n_wr = (uint8_t)s->n_writes;
  for (i = 0; i < s->n_reads; ++i) c->rd[i] = bcir_seg_read_rid(s, i);
  for (i = 0; i < s->n_writes; ++i) c->wr[i] = bcir_seg_write_rid(s, i);
  lc->order[lc->index++] = (uint32_t)s->claim_id;
  return 0;
}

static int on_block(const bcir_block_view *b, void *vctx) {
  load_ctx *lc = (load_ctx *)vctx;
  struct bcir_dgem_claim *c;
  if (lc->index >= lc->g->n_claims) return 1;
  c = &lc->g->claims[lc->order[lc->index++]];
  c->offset = b->base;
  c->count = b->count;
  return 0;
}

/* A per-layer RID: its layer and slot, or 0 when it is not one of `n_layers` layers'. */
static int layer_slot(uint32_t rid, uint32_t n_layers, uint32_t *layer, uint32_t *slot) {
  if (rid < (uint32_t)BCIR_DGEM_LAYER_BASE) return 0;
  rid -= (uint32_t)BCIR_DGEM_LAYER_BASE;
  *layer = rid / (uint32_t)BCIR_DGEM_LAYER_STRIDE;
  *slot = rid % (uint32_t)BCIR_DGEM_LAYER_STRIDE;
  return *layer < n_layers;
}

static uint16_t slot_tensor(uint32_t slot) {
  switch (slot) {
  case BCIR_DGEM_SLOT_G_ATTN: return BCIR_Q8_TENSOR_G_ATTN;
  case BCIR_DGEM_SLOT_WQ: return BCIR_Q8_TENSOR_W_Q;
  case BCIR_DGEM_SLOT_WK: return BCIR_Q8_TENSOR_W_K;
  case BCIR_DGEM_SLOT_WV: return BCIR_Q8_TENSOR_W_V;
  case BCIR_DGEM_SLOT_WO: return BCIR_Q8_TENSOR_W_O;
  case BCIR_DGEM_SLOT_G_FF: return BCIR_Q8_TENSOR_G_FF;
  case BCIR_DGEM_SLOT_W_GATE: return BCIR_Q8_TENSOR_W_GATE;
  case BCIR_DGEM_SLOT_W_UP: return BCIR_Q8_TENSOR_W_UP;
  case BCIR_DGEM_SLOT_W_DOWN: return BCIR_Q8_TENSOR_W_DOWN;
  default: return 0xFFFFu;
  }
}

/* The activation buffer a RID names. FINAL is the interpreter's own row: the monolithic
 * runner reuses h for it, but a program may write H between the final norm and the head. */
static double *activation(bcir_dgem *g, uint32_t rid) {
  bcir_llama_ws *w = &g->ws;
  switch (rid) {
  case BCIR_DGEM_RID_X: return w->x;
  case BCIR_DGEM_RID_H: return w->h;
  case BCIR_DGEM_RID_Q: return w->q;
  case BCIR_DGEM_RID_K: return w->k;
  case BCIR_DGEM_RID_V: return w->v;
  case BCIR_DGEM_RID_QR: return w->q_rope;
  case BCIR_DGEM_RID_KR: return w->k_rope;
  case BCIR_DGEM_RID_CTX: return w->context;
  case BCIR_DGEM_RID_H2: return w->h2;
  case BCIR_DGEM_RID_GATE: return w->gate;
  case BCIR_DGEM_RID_UP: return w->up;
  case BCIR_DGEM_RID_FF: return w->ff;
  case BCIR_DGEM_RID_FINAL: return g->final_row;
  case BCIR_DGEM_RID_LOGITS: return w->logits;
  default: return NULL;
  }
}

static int rids2(const struct bcir_dgem_claim *c, uint32_t r0, uint32_t r1, uint32_t w0) {
  return c->n_rd == 2u && c->n_wr == 1u && c->rd[0] == r0 && c->rd[1] == r1 && c->wr[0] == w0;
}

/* Check one claim against the RID scheme and the model's shapes; resolve its tensor. The
 * activation pointers are bound later, once the workspace exists. */
static int check_claim(const bcir_q8_model *m, struct bcir_dgem_claim *c) {
  uint32_t d = m->d_model, dk = d / m->n_heads, kvd = m->n_kv_heads * dk;
  uint32_t layer = 0, slot = 0, layer2 = 0, slot2 = 0;
  switch ((bcir_dgem_op)c->op) {
  case BCIR_DGEM_EMBED:
    return rids2(c, BCIR_DGEM_RID_TOK, BCIR_DGEM_RID_EMBED, BCIR_DGEM_RID_X) && c->count == d;
  case BCIR_DGEM_RMSNORM:
    if (c->n_rd != 2u || c->n_wr != 1u || c->rd[0] != BCIR_DGEM_RID_X || c->count != d) return 0;
    if (c->rd[1] == BCIR_DGEM_RID_G_FINAL && c->wr[0] == BCIR_DGEM_RID_FINAL) {
      c->weight = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_G_FINAL, -1);
      c->final_norm = 1u;
    } else if (layer_slot(c->rd[1], m->n_layers, &layer, &slot) &&
               ((slot == BCIR_DGEM_SLOT_G_ATTN && c->wr[0] == BCIR_DGEM_RID_H) ||
                (slot == BCIR_DGEM_SLOT_G_FF && c->wr[0] == BCIR_DGEM_RID_H2))) {
      c->weight = bcir_q8_model_tensor(m, slot_tensor(slot), (int16_t)layer);
    } else {
      return 0;
    }
    return c->weight != NULL;
  case BCIR_DGEM_MATVEC:
  case BCIR_DGEM_MATVEC_ADD: {
    int add = c->op == BCIR_DGEM_MATVEC_ADD;
    if (c->n_rd != (add ? 3u : 2u) || c->n_wr != 1u) return 0;
    if (!layer_slot(c->rd[1], m->n_layers, &layer, &slot)) return 0;
    if (add) {
      if (c->rd[2] != BCIR_DGEM_RID_X || c->wr[0] != BCIR_DGEM_RID_X ||
          !((c->rd[0] == BCIR_DGEM_RID_CTX && slot == BCIR_DGEM_SLOT_WO) ||
            (c->rd[0] == BCIR_DGEM_RID_FF && slot == BCIR_DGEM_SLOT_W_DOWN)))
        return 0;
    } else if (!((c->rd[0] == BCIR_DGEM_RID_H &&
                  ((slot == BCIR_DGEM_SLOT_WQ && c->wr[0] == BCIR_DGEM_RID_Q) ||
                   (slot == BCIR_DGEM_SLOT_WK && c->wr[0] == BCIR_DGEM_RID_K) ||
                   (slot == BCIR_DGEM_SLOT_WV && c->wr[0] == BCIR_DGEM_RID_V))) ||
                 (c->rd[0] == BCIR_DGEM_RID_H2 &&
                  ((slot == BCIR_DGEM_SLOT_W_GATE && c->wr[0] == BCIR_DGEM_RID_GATE) ||
                   (slot == BCIR_DGEM_SLOT_W_UP && c->wr[0] == BCIR_DGEM_RID_UP))))) {
      return 0;
    }
    c->weight = bcir_q8_model_tensor(m, slot_tensor(slot), (int16_t)layer);
    if (!c->weight || c->weight->rank != 2u) return 0;
    c->in = c->weight->dim0;
    c->out = c->weight->dim1;
    return c->count == (uint64_t)c->in * c->out;
  }
  case BCIR_DGEM_ROPE:
    return c->n_rd == 2u && c->n_wr == 2u && c->rd[0] == BCIR_DGEM_RID_Q &&
           c->rd[1] == BCIR_DGEM_RID_K && c->wr[0] == BCIR_DGEM_RID_QR &&
           c->wr[1] == BCIR_DGEM_RID_KR && c->count == (uint64_t)d + kvd;
  case BCIR_DGEM_KV_APPEND:
    if (c->n_rd != 2u || c->n_wr != 2u || c->rd[0] != BCIR_DGEM_RID_KR ||
        c->rd[1] != BCIR_DGEM_RID_V || c->count != 2ull * kvd ||
        !layer_slot(c->wr[0], m->n_layers, &layer, &slot) ||
        !layer_slot(c->wr[1], m->n_layers, &layer2, &slot2))
      return 0;
    c->layer = layer;
    return slot == BCIR_DGEM_SLOT_K_CACHE && slot2 == BCIR_DGEM_SLOT_V_CACHE && layer2 == layer;
  case BCIR_DGEM_ATTENTION:
    if (c->n_rd != 3u || c->n_wr != 1u || c->rd[0] != BCIR_DGEM_RID_QR ||
        c->wr[0] != BCIR_DGEM_RID_CTX || !layer_slot(c->rd[1], m->n_layers, &layer, &slot) ||
        !layer_slot(c->rd[2], m->n_layers, &layer2, &slot2) || c->offset >= (1ull << 32))
      return 0;
    c->layer = layer;
    return slot == BCIR_DGEM_SLOT_K_CACHE && slot2 == BCIR_DGEM_SLOT_V_CACHE && layer2 == layer &&
           c->count == 2ull * d * (c->offset + 1u);
  case BCIR_DGEM_SWIGLU:
    return rids2(c, BCIR_DGEM_RID_GATE, BCIR_DGEM_RID_UP, BCIR_DGEM_RID_FF) &&
           c->count == m->d_ff;
  case BCIR_DGEM_HEAD:
    c->weight = bcir_llama_head_tensor(m);
    return rids2(c, BCIR_DGEM_RID_FINAL,
                 (m->flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS) ? (uint32_t)BCIR_DGEM_RID_EMBED
                                                           : (uint32_t)BCIR_DGEM_RID_HEAD,
                 BCIR_DGEM_RID_LOGITS) &&
           c->weight != NULL && c->count == (uint64_t)m->vocab_size * d;
  case BCIR_DGEM_ARGMAX:
    return c->n_rd == 1u && c->n_wr == 1u && c->rd[0] == BCIR_DGEM_RID_LOGITS &&
           c->wr[0] == BCIR_DGEM_RID_TOK && c->count == m->vocab_size;
  default:
    return 0;
  }
}

static int fail_load(bcir_dgem *g, bcir_host_allocator *a, uint32_t *order, int code) {
  bcir_host_deallocate(a, order);
  bcir_dgem_free(g);
  return code;
}

int bcir_dgem_load(bcir_dgem *g, const bcir_q8_model *model, const uint8_t *pack, size_t len,
                   const bcir_host_allocator *allocator) {
  bcir_streampack_header hdr;
  load_ctx lc;
  uint32_t *order = NULL;
  size_t i, n_embed = 0, n_argmax = 0, first_argmax = DGEM_NONE, last_argmax = 0;
  bcir_status st;
  if (!g) return BCIR_DGEM_E_ARG;
  memset(g, 0, sizeof *g);
  g->allocator = bcir_host_allocator_or_default(allocator);
  if (!model || !pack) return BCIR_DGEM_E_ARG;
  if (!bcir_llama_model_ready(model) || model->n_layers > (uint32_t)BCIR_DGEM_MAX_LAYERS ||
      model->vocab_size > INT32_MAX)
    return BCIR_DGEM_E_MODEL;
  g->model = model;
  g->pack = pack;
  g->pack_len = len;
  /* The semantic walk (R10 and the range gate) reads every declared record before anything is
   * sized from the header: a count the body does not carry is refused, not allocated for. */
  st = bcir_sp_validate(pack, len, &hdr);
  if (st == BCIR_OK) st = bcir_sp_verify_semantic(pack, len, 0xFFFFFFFFu, 0xFFFFFFFFu);
  if (st != BCIR_OK) {
    g->pack_status = st;
    return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PACK);
  }
  if (!hdr.n_segments || hdr.n_segments > DGEM_MAX_CLAIMS || hdr.n_blocks != hdr.n_segments)
    return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PROGRAM);
  g->n_claims = hdr.n_segments;
  g->claims = (struct bcir_dgem_claim *)alloc_zero(&g->allocator, g->n_claims,
                                                    sizeof *g->claims);
  order = (uint32_t *)alloc_zero(&g->allocator, g->n_claims, sizeof *order);
  if (!g->claims || !order) return fail_load(g, &g->allocator, order, BCIR_DGEM_E_MEMORY);
  lc.g = g;
  lc.order = order;
  lc.index = 0;
  st = bcir_sp_for_each_segment(pack, len, on_segment, &lc);
  if (st != BCIR_OK) {
    g->pack_status = st;
    return fail_load(g, &g->allocator, order, BCIR_DGEM_E_PACK);
  }
  if (lc.index != g->n_claims) return fail_load(g, &g->allocator, order, BCIR_DGEM_E_PROGRAM);
  lc.index = 0;
  st = bcir_sp_for_each_block(pack, len, on_block, &lc);
  if (st != BCIR_OK) {
    g->pack_status = st;
    return fail_load(g, &g->allocator, order, BCIR_DGEM_E_PACK);
  }
  bcir_host_deallocate(&g->allocator, order);
  order = NULL;
  if (lc.index != g->n_claims) return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PROGRAM);

  /* Every claim a decoder operation of this model; the tape's shape from the argmaxes. */
  for (i = 0; i < g->n_claims; ++i) {
    struct bcir_dgem_claim *c = &g->claims[i];
    if (!check_claim(model, c)) return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PROGRAM);
    if (c->op == BCIR_DGEM_EMBED) n_embed++;
    if (c->op == BCIR_DGEM_ARGMAX) {
      n_argmax++;
      if (first_argmax == DGEM_NONE || c->offset < first_argmax) first_argmax = (size_t)c->offset;
      if (c->offset > last_argmax) last_argmax = (size_t)c->offset;
    }
  }
  if (!n_argmax || first_argmax < 1u || last_argmax >= (size_t)INT32_MAX ||
      n_argmax != last_argmax + 1u - first_argmax || n_embed != last_argmax)
    return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PROGRAM);
  g->capacity = last_argmax + 1u;
  g->prompt_len = first_argmax;
  if (model->context_length && g->capacity - 1u > (size_t)model->context_length)
    return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PROGRAM);
  for (i = 0; i < g->n_claims; ++i) { /* every position a claim names lies on the tape */
    const struct bcir_dgem_claim *c = &g->claims[i];
    if (c->op != BCIR_DGEM_ARGMAX && c->offset >= g->capacity - 1u)
      return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_PROGRAM);
  }

  if (bcir_llama_ws_init(&g->ws, model, g->capacity, &g->allocator))
    return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_MEMORY);
  g->scratch = (bcir_exec_item *)alloc_zero(&g->allocator, g->n_claims, sizeof *g->scratch);
  g->phases = (bcir_phase_stat *)alloc_zero(&g->allocator, g->n_claims, sizeof *g->phases);
  g->tape = (int32_t *)alloc_zero(&g->allocator, g->capacity, sizeof *g->tape);
  g->kv_rows = (uint32_t *)alloc_zero(&g->allocator, model->n_layers, sizeof *g->kv_rows);
  g->final_row = (double *)alloc_zero(&g->allocator, model->d_model, sizeof *g->final_row);
  if (!g->scratch || !g->phases || !g->tape || !g->kv_rows || !g->final_row)
    return fail_load(g, &g->allocator, NULL, BCIR_DGEM_E_MEMORY);
  for (i = 0; i < g->n_claims; ++i) { /* bind the activation operands */
    struct bcir_dgem_claim *c = &g->claims[i];
    if (c->op == BCIR_DGEM_RMSNORM) {
      c->src = g->ws.x;
      c->dst = activation(g, c->wr[0]);
    } else if (c->op == BCIR_DGEM_MATVEC) {
      c->src = activation(g, c->rd[0]);
      c->dst = activation(g, c->wr[0]);
    } else if (c->op == BCIR_DGEM_MATVEC_ADD) {
      c->src = activation(g, c->rd[0]);
      c->dst = g->ws.attn; /* the projection, then the residual adds it to x */
    } else if (c->op == BCIR_DGEM_HEAD) {
      c->src = activation(g, c->rd[0]); /* FINAL */
    }
  }
  return 0;
}

/* --- run: GEM dispatches, each claim runs its operation ---------------------------------- */

static uint64_t q8_octets(const bcir_q8_tensor *t) {
  return (uint64_t)t->element_count + 2ull * t->group_count;
}

static uint64_t groups_touched(uint64_t start, uint64_t n, uint32_t group) {
  return (start + n - 1u) / group - start / group + 1u;
}

static int refuse(bcir_dgem *g, int code) {
  if (!g->error) g->error = code;
  return 1; /* stop the executor */
}

static int dgem_kernel(const bcir_exec_item *item, void *vctx) {
  bcir_dgem *g = (bcir_dgem *)vctx;
  const bcir_q8_model *m = g->model;
  struct bcir_dgem_claim *c;
  uint64_t t0 = 0, rd = 0, wr = 0;
  uint64_t d = m->d_model, kvd = (uint64_t)m->n_kv_heads * (m->d_model / m->n_heads);
  size_t pos = g->current;
  if (item->claim_id >= g->n_claims) return refuse(g, BCIR_DGEM_E_PROGRAM);
  c = &g->claims[item->claim_id];
  if (c->op != BCIR_DGEM_EMBED && pos == DGEM_NONE) return refuse(g, BCIR_DGEM_E_ORDER);
  if (g->clock) t0 = g->clock(g->clock_ctx);
  switch ((bcir_dgem_op)c->op) {
  case BCIR_DGEM_EMBED: {
    int32_t token;
    pos = (size_t)c->offset;
    if (pos != g->next_position || pos >= g->filled) return refuse(g, BCIR_DGEM_E_ORDER);
    token = g->tape[pos];
    if (bcir_llama_op_embed(m, &g->ws, token)) return refuse(g, BCIR_DGEM_E_KERNEL);
    g->current = pos;
    g->next_position = pos + 1u;
    rd = 4u + d + 2u * groups_touched((uint64_t)(uint32_t)token * d, d, m->group_size);
    wr = 8u * d;
    break;
  }
  case BCIR_DGEM_RMSNORM:
    bcir_llama_op_rmsnorm(m, &g->ws, c->src, c->weight, c->dst);
    if (c->final_norm) g->final_at = pos;
    rd = 8u * d + q8_octets(c->weight);
    wr = 8u * d;
    break;
  case BCIR_DGEM_MATVEC:
  case BCIR_DGEM_MATVEC_ADD:
    if (bcir_llama_op_matvec(m, c->src, c->weight, c->in, c->out, c->dst))
      return refuse(g, BCIR_DGEM_E_KERNEL);
    rd = 8ull * c->in + q8_octets(c->weight);
    wr = 8ull * c->out;
    if (c->op == BCIR_DGEM_MATVEC_ADD) {
      bcir_llama_op_residual(m, &g->ws);
      rd += 8ull * c->out;
    }
    break;
  case BCIR_DGEM_ROPE:
    if (c->offset != pos) return refuse(g, BCIR_DGEM_E_ORDER);
    bcir_llama_op_rope(m, &g->ws, pos);
    rd = wr = 8u * (d + kvd);
    break;
  case BCIR_DGEM_KV_APPEND:
    if (c->offset != pos || g->kv_rows[c->layer] != pos) return refuse(g, BCIR_DGEM_E_ORDER);
    bcir_llama_op_kv_append(m, &g->ws, c->layer, pos);
    g->kv_rows[c->layer] = (uint32_t)(pos + 1u);
    rd = wr = 16u * kvd;
    break;
  case BCIR_DGEM_ATTENTION:
    if (c->offset != pos || g->kv_rows[c->layer] != pos + 1u)
      return refuse(g, BCIR_DGEM_E_ORDER);
    if (bcir_llama_op_attention(m, &g->ws, c->layer, pos)) return refuse(g, BCIR_DGEM_E_KERNEL);
    rd = 8u * d + 16u * (uint64_t)(pos + 1u) * kvd;
    wr = 8u * d;
    break;
  case BCIR_DGEM_SWIGLU:
    bcir_llama_op_swiglu(m, &g->ws);
    rd = 16ull * m->d_ff;
    wr = 8ull * m->d_ff;
    break;
  case BCIR_DGEM_HEAD:
    if (g->final_at != pos) return refuse(g, BCIR_DGEM_E_ORDER);
    if (bcir_llama_op_head(m, c->src, g->ws.logits)) return refuse(g, BCIR_DGEM_E_KERNEL);
    g->head_at = pos;
    rd = 8u * d + q8_octets(c->weight);
    wr = 8ull * m->vocab_size;
    break;
  case BCIR_DGEM_ARGMAX: {
    int32_t token;
    if (g->head_at != pos || c->offset != pos + 1u || c->offset != g->filled ||
        c->offset >= g->capacity)
      return refuse(g, BCIR_DGEM_E_ORDER);
    token = bcir_llama_op_argmax(m, g->ws.logits);
    g->tape[g->filled++] = token;
    rd = 8ull * m->vocab_size;
    wr = 4u;
    break;
  }
  default:
    return refuse(g, BCIR_DGEM_E_PROGRAM);
  }
  g->stats.claims[c->op]++;
  g->stats.bytes_read[c->op] += rd;
  g->stats.bytes_written[c->op] += wr;
  if (g->clock) g->stats.ns[c->op] += g->clock(g->clock_ctx) - t0;
  if (c->op == BCIR_DGEM_ARGMAX && g->on_token)
    g->on_token(g->token_ctx, (size_t)c->offset, g->tape[c->offset]);
  return 0;
}

/* Zero every activation a program can name: a run reads only what it wrote, or zeros -- never
 * a value an earlier run left -- so a run is a function of the program, the prompt and the
 * model alone, whatever order a program's claims take. */
static void clear_activations(bcir_dgem *g) {
  const bcir_q8_model *m = g->model;
  size_t d = m->d_model, kvd = (size_t)m->n_kv_heads * (m->d_model / m->n_heads), ff = m->d_ff;
  double *const rows_d[] = {g->ws.x, g->ws.h, g->ws.h2, g->ws.q, g->ws.q_rope, g->ws.context,
                            g->final_row};
  double *const rows_kv[] = {g->ws.k, g->ws.k_rope, g->ws.v};
  double *const rows_ff[] = {g->ws.gate, g->ws.up, g->ws.ff};
  size_t i;
  for (i = 0; i < sizeof rows_d / sizeof rows_d[0]; ++i) memset(rows_d[i], 0, d * sizeof(double));
  for (i = 0; i < 3u; ++i) memset(rows_kv[i], 0, kvd * sizeof(double));
  for (i = 0; i < 3u; ++i) memset(rows_ff[i], 0, ff * sizeof(double));
  memset(g->ws.logits, 0, (size_t)m->vocab_size * sizeof(double));
}

int bcir_dgem_run(bcir_dgem *g, const int32_t *prompt, size_t prompt_len, int32_t *generated,
                  size_t max_new, double *final_logits) {
  bcir_exec_result res;
  bcir_status st;
  size_t i;
  if (!g || !g->claims || !prompt || prompt_len != g->prompt_len ||
      max_new != g->capacity - g->prompt_len || (max_new && !generated))
    return BCIR_DGEM_E_ARG;
  for (i = 0; i < prompt_len; ++i) {
    if (prompt[i] < 0 || (uint32_t)prompt[i] >= g->model->vocab_size) return BCIR_DGEM_E_ARG;
    g->tape[i] = prompt[i];
  }
  g->filled = prompt_len;
  g->next_position = 0;
  g->current = g->final_at = g->head_at = DGEM_NONE;
  g->error = 0;
  memset(g->kv_rows, 0, (size_t)g->model->n_layers * sizeof *g->kv_rows);
  clear_activations(g);
  st = bcir_sp_execute(g->pack, g->pack_len, g->scratch, g->n_claims, g->phases, g->n_claims,
                       dgem_kernel, g, &res);
  if (g->error) return g->error;
  if (st != BCIR_OK) {
    g->pack_status = st;
    return BCIR_DGEM_E_PACK;
  }
  if (res.executed != g->n_claims || g->filled != g->capacity) return BCIR_DGEM_E_ORDER;
  if (max_new) memcpy(generated, g->tape + prompt_len, max_new * sizeof *generated);
  if (final_logits)
    memcpy(final_logits, g->ws.logits, (size_t)g->model->vocab_size * sizeof *final_logits);
  return 0;
}
