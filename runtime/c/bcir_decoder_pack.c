/*===- bcir_decoder_pack.c - the native training step from a StreamPack (PX-2b) -------===
 *
 * bcir_decoder_pack.h states the contract. The lowering is three walks and a dispatch: the plan's
 * lifetimes into a table by RID (ascending on the wire, so a lookup is a binary search), the pack's
 * segments and blocks into a table by claim id, then bcir_sp_execute dispatching every claim in
 * GEM's order with a callback that writes its record. Nothing here allocates: the tables live in
 * the caller's `work`.
 *===----------------------------------------------------------------------===*/
#include "bcir_decoder_pack.h"

#include <string.h>

#include "bcir_exec.h"
#include "bcir_execution_plan.h"
#include "bcir_runtime.h"

#define DP_ONE 0x3f800000u /* 1.0f: the step's alpha, an accumulating beta */

/* The claims' names but the GEMM's, in native_pack.OPS's order. */
enum { OP_SCAN, OP_ZERO, OP_EMBED, OP_RMS, OP_ROPE, OP_ROPE_INV, OP_ATTN, OP_ATTN_VJP, OP_ADD,
  OP_MUL, OP_PSILU, OP_SILU, OP_SILU_P, OP_SWIGLU, OP_SWIGLU_P, OP_RMSV, OP_RMSV_A, OP_RMSV_F,
  OP_RMSV_AF, OP_LOSS, OP_SCATTER, OP_CHECK_F, OP_CHECK_R, OP_CHECK_L, OP_UPDATE, OP_ROPE_TABLE,
  OP_GEMM };
static const char *const dp_names[OP_GEMM] = {
  "train.scan", "train.zero", "train.embed", "train.rms", "train.rope", "train.rope.inverse",
  "train.attention", "train.attention.vjp", "train.add", "train.mul", "train.provider.silu",
  "train.silu_mul", "train.silu_mul.panels", "train.swiglu.vjp", "train.swiglu.vjp.panels",
  "train.rms.vjp", "train.rms.vjp.add", "train.rms.vjp.first", "train.rms.vjp.add.first",
  "train.loss", "train.scatter", "train.check.forward", "train.check.ran", "train.check.loss",
  "train.update", "train.rope.table"};
/* The banks, numbered as the operand spaces: the six arenas 1-6, the workspace 7, then the
 * micro-batch's tokens and targets (both space 8, told apart by the slot that takes them). */
enum { BANK_WORKSPACE = 7, BANK_TOKENS = 8, BANK_TARGETS = 9, DP_BANKS = 9 };
static const char *const dp_banks[DP_BANKS] = {"weights", "gradients", "moment1", "moment2",
  "activations", "scratch", "workspace", "tokens", "targets"};
/* An operand's use, and the kind of value its slot takes. */
enum { U_R = 1u, U_W = 2u, U_RW = 3u };
enum { SL_PLAIN, SL_TOKENS, SL_TARGETS, SL_WS, SL_RUN };

typedef struct dp_claim {
  const uint8_t *rd, *wr; /* the segment's RIDs, little-endian u32, in the pack */
  uint64_t base, count;
  uint16_t n_rd, n_wr;
  uint16_t instance, workers; /* PX-3b-2: the plan step's measured alternative (0: the default) */
  uint8_t op, form, seen;
} dp_claim;
typedef struct dp_life { uint64_t offset, size, first, last; uint32_t rid; unsigned bank; } dp_life;
typedef struct dp_slot { unsigned char slot, use, kind; } dp_slot;
typedef struct dp_ctx {
  dp_claim *claims;
  dp_life *lives;
  uint32_t *order; /* segment index -> claim id: blocks pair with segments by index */
  size_t n, nl, at;
  uint64_t *program; /* the records, once a dispatch has checked them all (`writing`) */
  uint64_t scratch[BCIR_DT_RECORD_WORDS];
  size_t k, rank, index, micro;
  uint32_t phase;
  int started, why, writing;
  /* the state's geometry and contract, as its records carry them */
  uint64_t rows, d, h, kh, hd, v, b, t, params, chunk, eps;
  unsigned gc, attn, attn_vjp, v2, reduce, provider;
  unsigned lent; /* ATT-CONTIG: the state lends workspace bytes, which the binary32 attention copies into */
} dp_ctx;

static int dp_fail(dp_ctx *c, int why) {
  if (!c->why) c->why = why;
  return 0;
}
static int dp_mul(uint64_t a, uint64_t b, uint64_t *out) {
  if (a && b > UINT64_MAX / a) return 0;
  *out = a * b;
  return 1;
}
/* q = a / (x * y) exactly, the divisor and q nonzero */
static int dp_div(uint64_t a, uint64_t x, uint64_t y, uint64_t *q) {
  uint64_t b;
  if (!dp_mul(x, y, &b) || !b || a % b) return 0;
  *q = a / b;
  return *q != 0;
}
static int dp_size_mul(size_t a, size_t b, size_t *out) {
  if (a && b > SIZE_MAX / a) return 0;
  *out = a * b;
  return 1;
}
static uint32_t dp_rid(const uint8_t *p, size_t i) {
  return (uint32_t)p[4 * i] | (uint32_t)p[4 * i + 1] << 8 | (uint32_t)p[4 * i + 2] << 16 |
         (uint32_t)p[4 * i + 3] << 24;
}

/* --- the vocabulary ------------------------------------------------------------------- */

static size_t dp_cat(char *out, size_t at, const char *s) {
  size_t m = strlen(s);
  memcpy(out + at, s, m);
  return at + m;
}
/* train.gemm.<role>.<orient>[.acc][.pa|.pb] as role | orient << 2 | acc << 4 | panel << 5 */
static int dp_gemm_name(const char *s, size_t n, unsigned *form) {
  static const char *const role[4] = {".alone", ".pair", ".chain", ".rest"};
  static const char *const orient[3] = {".nt", ".nn", ".tn"};
  static const char *const acc[2] = {"", ".acc"};
  static const char *const panel[3] = {"", ".pa", ".pb"};
  char name[40];
  unsigned r, o, a, p;
  if (n >= sizeof name) return 0;
  for (r = 0; r < 4u; r++)
    for (o = 0; o < 3u; o++)
      for (a = 0; a < 2u; a++)
        for (p = 0; p < 3u; p++) {
          size_t m = dp_cat(name, 0, "train.gemm");
          m = dp_cat(name, m, role[r]);
          m = dp_cat(name, m, orient[o]);
          m = dp_cat(name, m, acc[a]);
          m = dp_cat(name, m, panel[p]);
          if (m == n && !memcmp(name, s, n)) {
            *form = r | o << 2 | a << 4 | p << 5;
            return 1;
          }
        }
  return 0;
}
static int dp_name(const char *s, size_t n, unsigned *op, unsigned *form) {
  unsigned i;
  for (i = 0; i < (unsigned)OP_GEMM; i++)
    if (strlen(dp_names[i]) == n && !memcmp(dp_names[i], s, n)) {
      *op = i;
      *form = 0;
      return 1;
    }
  *op = OP_GEMM;
  return dp_gemm_name(s, n, form);
}

/* PX-3b-2: a measured realization's alternative, `i<bit>` (1, 2, 4 or 8) or `t<k>` (1 .. 32767,
 * ASCII digits, no leading zero) -- cost.measured_alternative_key's grammar -- as (instance,
 * workers); 0 for any other spelling. */
static int dp_alternative(const char *s, size_t n, uint16_t *instance, uint16_t *workers) {
  uint32_t v = 0;
  size_t i;
  if (n < 2u || n > 6u || (s[0] != 'i' && s[0] != 't') || s[1] == '0') return 0;
  for (i = 1; i < n; i++) {
    if (s[i] < '0' || s[i] > '9') return 0;
    v = v * 10u + (uint32_t)(s[i] - '0');
  }
  if (s[0] == 'i') {
    if (v != 1u && v != 2u && v != 4u && v != 8u) return 0;
    *instance = (uint16_t)v;
  } else {
    if (v > 32767u) return 0;
    *workers = (uint16_t)v;
  }
  return 1;
}

/* --- the walks ------------------------------------------------------------------------ */

/* each plan step, as the walk verifies it: a measured realization (`<name>@<alternative>`, the
 * plan's candidate string) names its claim's dispatch; any other runs the default */
static int dp_on_step(const bcir_ep_step_view *v, void *vc) {
  dp_ctx *c = (dp_ctx *)vc;
  dp_claim *cl;
  size_t at = 0;
  if (v->claim_id >= c->n) return !dp_fail(c, BCIR_DP_CLAIMS);
  while (at < v->candidate_len && v->candidate[at] != '@') at++;
  if (at == v->candidate_len) return 0;
  cl = &c->claims[v->claim_id];
  if (!dp_alternative(v->candidate + at + 1u, v->candidate_len - at - 1u, &cl->instance,
                      &cl->workers))
    return !dp_fail(c, BCIR_DP_ALTERNATIVE);
  return 0;
}

static int dp_on_life(const bcir_ep_lifetime_view *v, void *vc) {
  dp_ctx *c = (dp_ctx *)vc;
  dp_life *l;
  unsigned b;
  if (c->at >= c->nl) return !dp_fail(c, BCIR_DP_CLAIMS);
  for (b = 0; b < (unsigned)DP_BANKS; b++)
    if (strlen(dp_banks[b]) == v->bank_len && !memcmp(dp_banks[b], v->bank, v->bank_len)) break;
  if (b == (unsigned)DP_BANKS) return !dp_fail(c, BCIR_DP_BANK);
  l = &c->lives[c->at++];
  l->rid = v->rid;
  l->bank = b + 1u;
  l->offset = v->offset;
  l->size = v->size;
  l->first = v->first_tick; /* the alias law's own interval: half-open, in the phases' ranks */
  l->last = v->last_tick;
  return 0;
}
static int dp_on_seg(const bcir_segment_view *g, void *vc) {
  dp_ctx *c = (dp_ctx *)vc;
  dp_claim *cl;
  unsigned op = 0, form = 0;
  if (c->at >= c->n || g->claim_id >= c->n) return !dp_fail(c, BCIR_DP_CLAIMS);
  cl = &c->claims[g->claim_id];
  if (cl->seen) return !dp_fail(c, BCIR_DP_CLAIMS);
  if (!dp_name(g->opcode, g->opcode_len, &op, &form)) return !dp_fail(c, BCIR_DP_OP);
  cl->seen = 1;
  cl->op = (uint8_t)op;
  cl->form = (uint8_t)form;
  cl->rd = g->reads;
  cl->n_rd = g->n_reads;
  cl->wr = g->writes;
  cl->n_wr = g->n_writes;
  c->order[c->at++] = (uint32_t)g->claim_id;
  return 0;
}
static int dp_on_block(const bcir_block_view *b, void *vc) {
  dp_ctx *c = (dp_ctx *)vc;
  dp_claim *cl;
  if (c->at >= c->n) return !dp_fail(c, BCIR_DP_CLAIMS);
  cl = &c->claims[c->order[c->at++]];
  cl->base = b->base;
  cl->count = b->count;
  return 0;
}

/* --- one claim, one record --------------------------------------------------------------- */

static const dp_life *dp_find(const dp_ctx *c, uint32_t rid) {
  size_t lo = 0, hi = c->nl;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (c->lives[mid].rid < rid) lo = mid + 1;
    else hi = mid;
  }
  return lo < c->nl && c->lives[lo].rid == rid ? &c->lives[lo] : NULL;
}

/* Operand `sl`: `take` RIDs from rd (read), wr (written) or both alike (in place), each live at
 * the claim's rank, one contiguous span of one bank; its place into ref[0..2]. */
static int dp_operand(dp_ctx *c, const dp_claim *cl, const dp_slot *sl, size_t take, size_t *ir,
                      size_t *iw, uint64_t *ref) {
  uint64_t off = 0, size = 0;
  unsigned bank = 0;
  size_t j;
  if (!take || ((sl->use & U_R) && take > (size_t)cl->n_rd - *ir) ||
      ((sl->use & U_W) && take > (size_t)cl->n_wr - *iw))
    return dp_fail(c, BCIR_DP_SLOTS);
  for (j = 0; j < take; j++) {
    uint32_t rid = (sl->use & U_R) ? dp_rid(cl->rd, *ir + j) : dp_rid(cl->wr, *iw + j);
    const dp_life *l;
    if (sl->use == U_RW && dp_rid(cl->wr, *iw + j) != rid) return dp_fail(c, BCIR_DP_SLOTS);
    l = dp_find(c, rid);
    if (!l) return dp_fail(c, BCIR_DP_RID);
    if ((uint64_t)c->rank < l->first || (uint64_t)c->rank >= l->last)
      return dp_fail(c, BCIR_DP_LIVE);
    if (!j) {
      bank = l->bank;
      off = l->offset;
      size = l->size;
    } else {
      if (l->bank != bank || l->offset != off + size || l->size > UINT64_MAX - size)
        return dp_fail(c, BCIR_DP_RUN);
      size += l->size;
    }
  }
  if (sl->use & U_R) *ir += take;
  if (sl->use & U_W) *iw += take;
  switch (sl->kind) {
  case SL_TOKENS: if (bank != BANK_TOKENS) return dp_fail(c, BCIR_DP_BANK); bank = 8; break;
  case SL_TARGETS: if (bank != BANK_TARGETS) return dp_fail(c, BCIR_DP_BANK); bank = 8; break;
  case SL_WS: if (bank != BANK_WORKSPACE) return dp_fail(c, BCIR_DP_BANK); break;
  default: if (bank < 1u || bank > 6u) return dp_fail(c, BCIR_DP_BANK); break;
  }
  ref[0] = bank;
  ref[1] = off;
  ref[2] = size;
  return 1;
}

/* the call's work, as native_pack.work counts it from the record */
static int dp_work(const uint64_t *w, uint64_t *out) {
  uint64_t x, tri;
  switch (w[0]) {
  case BCIR_DT_REC_GEMM: return dp_mul(w[5], w[6], &x) && dp_mul(x, w[7], out);
  case BCIR_DT_REC_ATTN: case BCIR_DT_REC_ATTN_VJP:
    if (w[4] == UINT64_MAX || !dp_mul(w[4], w[4] + 1u, &tri)) return 0;
    return dp_mul(w[0] == BCIR_DT_REC_ATTN ? 2u : 4u, w[3], &x) && dp_mul(x, w[5], &x) &&
           dp_mul(x, w[7], &x) && dp_mul(x, tri / 2u, out);
  case BCIR_DT_REC_ROPE:
    return dp_mul(w[3], w[4], &x) && dp_mul(x, w[5], &x) && dp_mul(x, w[6], out);
  case BCIR_DT_REC_EMBED: case BCIR_DT_REC_SCATTER: case BCIR_DT_REC_RMS:
  case BCIR_DT_REC_RMS_VJP: case BCIR_DT_REC_LOSS: case BCIR_DT_REC_SILU_MUL_PANELS:
  case BCIR_DT_REC_SWIGLU_VJP_PANELS: case BCIR_DT_REC_ROPE_TABLE:
    return dp_mul(w[3], w[4], out);
  case BCIR_DT_REC_CHECK: *out = 1u; return 1;
  default: *out = w[3]; return 1;
  }
}

#define SLOT(q, u, k) \
  (sl[ns].slot = (unsigned char)(q), sl[ns].use = (unsigned char)(u), \
   sl[ns].kind = (unsigned char)(k), ns++)

/* claim `cl` at the dispatch's rank and index, as the record it runs: 1, or 0 with c->why */
static int dp_record(dp_ctx *c, const dp_claim *cl, uint64_t *w) {
  dp_slot sl[10];
  unsigned ns = 0, i, op = cl->op, form = cl->form, role = form & 3u, orient = form >> 2 & 3u,
           acc = form >> 4 & 1u, panel = form >> 5 & 3u, entry = 0, bf = 0;
  size_t ir = 0, iw = 0, take;
  uint64_t q, work, base;
  memset(w, 0, BCIR_DT_RECORD_WORDS * sizeof *w);
  w[1] = (uint64_t)c->rank + 1u;
  w[BCIR_DT_WORD_INSTANCE] = cl->instance; /* PX-3b-2: the plan's alternative (0: the default) */
  w[BCIR_DT_WORD_WORKERS] = cl->workers;
  switch (op) {
  case OP_SCAN: SLOT(0, U_R, SL_RUN); break;
  case OP_ZERO: SLOT(0, U_W, SL_RUN); break;
  case OP_EMBED: SLOT(0, U_R, SL_TOKENS); SLOT(1, U_R, SL_PLAIN); SLOT(2, U_W, SL_PLAIN); break;
  case OP_RMS: case OP_SILU:
    SLOT(0, U_R, SL_PLAIN); SLOT(1, U_R, SL_PLAIN); SLOT(2, U_W, SL_PLAIN); break;
  case OP_ROPE: case OP_ROPE_INV: SLOT(0, U_RW, SL_PLAIN); SLOT(1, U_R, SL_PLAIN); break;
  case OP_ROPE_TABLE: SLOT(0, U_W, SL_PLAIN); break;
  case OP_ATTN: case OP_ATTN_VJP:
    bf = c->attn_vjp == 256u;
    for (i = 0; i < 3u; i++) SLOT(i, U_R, SL_PLAIN);
    if (op == OP_ATTN) {
      SLOT(3, U_W, SL_PLAIN); SLOT(4, U_W, SL_PLAIN);
    } else {
      SLOT(3, bf ? U_R : U_RW, SL_PLAIN); SLOT(4, U_R, SL_PLAIN);
      for (i = 5; i < 8u; i++) SLOT(i, U_W, SL_PLAIN);
    }
    if (bf || c->lent) SLOT(8, U_W, SL_WS);
    break;
  case OP_ADD: case OP_MUL: SLOT(0, U_R, SL_PLAIN); SLOT(1, U_RW, SL_PLAIN); break;
  case OP_PSILU: SLOT(0, U_R, SL_PLAIN); SLOT(1, U_W, SL_PLAIN); break;
  case OP_SILU_P:
    SLOT(0, U_R, SL_PLAIN); SLOT(1, U_R, SL_PLAIN); SLOT(2, U_W, SL_PLAIN); SLOT(3, U_W, SL_PLAIN);
    break;
  case OP_SWIGLU: case OP_SWIGLU_P:
    for (i = 0; i < 3u; i++) SLOT(i, U_R, SL_PLAIN);
    for (i = 3; i < (op == OP_SWIGLU ? 5u : 7u); i++) SLOT(i, U_W, SL_PLAIN);
    break;
  case OP_RMSV: case OP_RMSV_A: case OP_RMSV_F: case OP_RMSV_AF:
    for (i = 0; i < 3u; i++) SLOT(i, U_R, SL_PLAIN);
    SLOT(3, (op == OP_RMSV_A || op == OP_RMSV_AF) ? U_RW : U_W, SL_PLAIN);
    SLOT(4, (op == OP_RMSV_F || op == OP_RMSV_AF) ? U_W : U_RW, SL_PLAIN);
    SLOT(5, U_W, SL_PLAIN);
    break;
  case OP_LOSS:
    SLOT(0, U_R, SL_PLAIN); SLOT(1, U_R, SL_TARGETS);
    SLOT(2, U_W, SL_PLAIN); SLOT(3, U_W, SL_PLAIN);
    break;
  case OP_SCATTER: SLOT(0, U_R, SL_TOKENS); SLOT(1, U_R, SL_PLAIN); SLOT(2, U_RW, SL_PLAIN); break;
  case OP_CHECK_F: case OP_CHECK_R: case OP_CHECK_L: break;
  case OP_UPDATE: SLOT(0, U_R, SL_RUN); for (i = 1; i < 4u; i++) SLOT(i, U_RW, SL_RUN); break;
  default: /* OP_GEMM: the entry the job's role takes under the state */
    if (c->index > 255u) return dp_fail(c, BCIR_DP_SLOTS);
    entry = role == 0u ? (c->provider ? 7u : c->gc) : role == 1u ? 4u : c->gc == 3u ? 6u : 5u;
    SLOT(0, U_R, SL_PLAIN); SLOT(1, U_R, SL_PLAIN); SLOT(2, acc ? U_RW : U_W, SL_PLAIN);
    if (entry == 3u || entry == 6u) SLOT(3, U_W, SL_WS);
    break;
  }
  for (i = 0; i < ns; i++) {
    take = 1;
    if (sl[i].kind == SL_RUN) {
      if (op == OP_SCAN) take = cl->n_rd;
      else if (op == OP_ZERO) take = cl->n_wr;
      else {
        take = cl->n_wr / 3u;
        if (cl->n_wr % 3u || (size_t)cl->n_rd != 4u * take) return dp_fail(c, BCIR_DP_SLOTS);
      }
    }
    if (!dp_operand(c, cl, &sl[i], take, &ir, &iw, w + 11 + 3u * sl[i].slot)) return 0;
  }
  if (ir != cl->n_rd || iw != cl->n_wr) return dp_fail(c, BCIR_DP_SLOTS);
  switch (op) {
  case OP_SCAN: case OP_ZERO: case OP_ADD: case OP_MUL: case OP_PSILU: case OP_SILU:
  case OP_SWIGLU:
    w[0] = op == OP_SCAN ? BCIR_DT_REC_SCAN : op == OP_ZERO ? BCIR_DT_REC_ZERO :
           op == OP_ADD ? BCIR_DT_REC_ADD : op == OP_MUL ? BCIR_DT_REC_MUL :
           op == OP_PSILU ? BCIR_DT_REC_PROVIDER_SILU : op == OP_SILU ? BCIR_DT_REC_SILU_MUL :
           BCIR_DT_REC_SWIGLU_VJP;
    if (op == OP_SILU || op == OP_SWIGLU) w[2] = c->v2;
    if (!dp_div(w[13], 4u, 1u, &w[3])) return dp_fail(c, BCIR_DP_SHAPE);
    break;
  case OP_EMBED: case OP_SCATTER: case OP_LOSS:
    w[0] = op == OP_EMBED ? BCIR_DT_REC_EMBED : op == OP_LOSS ? BCIR_DT_REC_LOSS
                                                               : BCIR_DT_REC_SCATTER;
    w[3] = c->rows;
    w[4] = op == OP_LOSS ? c->v : c->d;
    break;
  case OP_RMS: case OP_RMSV: case OP_RMSV_A: case OP_RMSV_F: case OP_RMSV_AF:
    w[0] = op == OP_RMS ? BCIR_DT_REC_RMS : BCIR_DT_REC_RMS_VJP;
    if (op == OP_RMSV_A || op == OP_RMSV_AF) w[2] |= BCIR_TENSOR_RMSB_ADD;
    if (op == OP_RMSV_F || op == OP_RMSV_AF) w[2] |= BCIR_TENSOR_RMSB_FIRST;
    w[3] = c->rows;
    w[4] = c->d;
    w[5] = c->eps;
    break;
  case OP_ROPE: case OP_ROPE_INV:
    w[0] = BCIR_DT_REC_ROPE;
    if (!dp_mul(4u, c->rows, &q) || !dp_div(w[13], q, c->hd, &w[5]))
      return dp_fail(c, BCIR_DP_SHAPE);
    w[3] = c->b;
    w[4] = c->t;
    w[6] = c->hd;
    w[7] = op == OP_ROPE_INV;
    /* ROPE-TABLE: the table a rotation reads is time x head-dim floats */
    if (!dp_mul(4u, c->t, &q) || !dp_mul(q, c->hd, &q) || w[16] != q)
      return dp_fail(c, BCIR_DP_SHAPE);
    break;
  case OP_ROPE_TABLE:
    w[0] = BCIR_DT_REC_ROPE_TABLE;
    w[3] = c->t;
    w[4] = c->hd;
    if (!dp_mul(4u, c->t, &q) || !dp_mul(q, c->hd, &q) || w[13] != q)
      return dp_fail(c, BCIR_DP_SHAPE);
    break;
  case OP_ATTN: case OP_ATTN_VJP:
    w[0] = op == OP_ATTN ? BCIR_DT_REC_ATTN : BCIR_DT_REC_ATTN_VJP;
    w[2] = op == OP_ATTN ? c->attn : c->attn_vjp;
    w[3] = c->b;
    w[4] = c->t;
    w[5] = c->h;
    w[6] = c->kh;
    w[7] = c->hd;
    break;
  case OP_SILU_P: case OP_SWIGLU_P:
    w[0] = op == OP_SILU_P ? BCIR_DT_REC_SILU_MUL_PANELS : BCIR_DT_REC_SWIGLU_VJP_PANELS;
    w[2] = c->v2;
    w[3] = c->rows;
    if (!dp_div(w[13], 4u, c->rows, &w[4])) return dp_fail(c, BCIR_DP_SHAPE);
    break;
  case OP_CHECK_F: case OP_CHECK_R: case OP_CHECK_L:
    w[0] = BCIR_DT_REC_CHECK;
    w[2] = op == OP_CHECK_F ? BCIR_DT_CHECK_FORWARD : op == OP_CHECK_R ? BCIR_DT_CHECK_RAN
                                                                        : BCIR_DT_CHECK_LOSS;
    break;
  case OP_UPDATE:
    w[0] = BCIR_DT_REC_UPDATE;
    w[2] = c->reduce;
    w[3] = c->params;
    w[4] = c->chunk;
    break;
  default: { /* GEMM: m, n, k from the operands, the token rows on the orientation's row side */
    uint64_t ea = panel == 1u ? 2u : 4u, eb = panel == 2u ? 2u : 4u, m, n, k;
    unsigned nib = role == 0u ? 0u : role == 1u ? (unsigned)c->index + 1u : role == 2u ? 1u : 2u;
    unsigned field = entry < 4u ? entry : entry == 7u ? 15u : c->gc;
    if (orient == 3u) return dp_fail(c, BCIR_DP_OP);
    if (orient == 2u) { /* tn: dW = dy^T x */
      k = c->rows;
      if (!dp_div(w[13], ea, k, &m) || !dp_div(w[16], eb, k, &n))
        return dp_fail(c, BCIR_DP_SHAPE);
    } else { /* nt: y = x W^T; nn: dx = dy W */
      m = c->rows;
      if (!dp_div(w[13], ea, m, &k) || !dp_div(w[19], 4u, m, &n))
        return dp_fail(c, BCIR_DP_SHAPE);
    }
    w[0] = BCIR_DT_REC_GEMM;
    w[2] = (uint64_t)entry | (uint64_t)nib << 4 | (uint64_t)c->index << 8 |
           (uint64_t)field << 16 | (uint64_t)(panel == 1u) << 24 | (uint64_t)(panel == 2u) << 25;
    w[3] = orient == 2u;
    w[4] = orient == 0u;
    w[5] = m;
    w[6] = n;
    w[7] = k;
    w[8] = DP_ONE;
    w[9] = acc ? DP_ONE : 0u;
    break;
  }
  }
  base = (op == OP_EMBED || op == OP_LOSS || op == OP_SCATTER) ? (uint64_t)c->micro : 0u;
  if (!dp_work(w, &work)) return dp_fail(c, BCIR_DP_SHAPE);
  if (cl->base != base || cl->count != work) return dp_fail(c, BCIR_DP_BLOCK);
  if (op == OP_CHECK_L) c->micro++;
  return 1;
}
#undef SLOT

/* the dispatch: each claim GEM's executor issues becomes the next record -- in a checking pass a
 * record of the context's own, in the writing pass the program's */
static int dp_on_item(const bcir_exec_item *it, void *vc) {
  dp_ctx *c = (dp_ctx *)vc;
  if (!c->started || it->phase_id != c->phase) {
    c->rank = c->started ? c->rank + 1u : 0u;
    c->started = 1;
    c->phase = it->phase_id;
    c->index = 0;
  } else {
    c->index++;
  }
  if (it->claim_id >= c->n || c->k >= c->n) return !dp_fail(c, BCIR_DP_CLAIMS);
  if (!dp_record(c, &c->claims[it->claim_id],
                 c->writing ? c->program + c->k * BCIR_DT_RECORD_WORDS : c->scratch))
    return 1;
  c->k++;
  return 0;
}

/* --- the entry points -------------------------------------------------------------------- */

static size_t dp_round(size_t x) { return (x + 7u) & ~(size_t)7u; }

/* the claim table, the lifetime table, the segment order and the executor's two buffers -- the
 * last two lent first to the plan walk's alias law (ALIAS-SWEEP: its sweep's scratch), which is
 * done with them before the executor takes them, so they hold the larger of the two */
static int dp_parts(size_t n, size_t nl, size_t part[5], size_t *total) {
  size_t i, sweep;
  if (!dp_size_mul(n, sizeof(dp_claim), &part[0]) ||
      !dp_size_mul(nl, sizeof(dp_life), &part[1]) ||
      !dp_size_mul(n, sizeof(uint32_t), &part[2]) ||
      !dp_size_mul(n, sizeof(bcir_exec_item), &part[3]) ||
      !dp_size_mul(n, sizeof(bcir_phase_stat), &part[4]))
    return 0;
  sweep = bcir_ep_sweep_bytes((uint32_t)nl);  /* nl: the plan header's u32 lifetime count */
  if (sweep == 0u) return 0;
  if (sweep > part[3] && sweep - part[3] > part[4]) part[4] = sweep - part[3];
  *total = 8u;
  for (i = 0; i < 5u; i++) {
    if (part[i] > SIZE_MAX - 7u) return 0;
    part[i] = dp_round(part[i]);
    if (*total > SIZE_MAX - part[i]) return 0;
    *total += part[i];
  }
  return 1;
}

static int dp_sizes(const uint8_t *pack, size_t pack_len, const uint8_t *plan, size_t plan_len,
                    size_t *n, size_t *nl, size_t part[5], size_t *total, int *why) {
  bcir_streampack_header ph;
  bcir_ep_header eh;
  bcir_status st = bcir_sp_validate(pack, pack_len, &ph);
  if (st == BCIR_OK) st = bcir_sp_verify_semantic(pack, pack_len, 0xFFFFFFFFu, 0xFFFFFFFFu);
  if (st == BCIR_OK) st = bcir_ep_validate(plan, plan_len, &eh);
  /* a lifetime is at least 34 bytes on the wire: a count past that is not one the bytes hold */
  if (st == BCIR_OK && eh.n_lifetimes > plan_len / 34u) st = BCIR_ERR_TRUNCATED;
  if (st != BCIR_OK) {
    *why = (int)st;
    return 0;
  }
  *n = ph.n_segments;
  *nl = eh.n_lifetimes;
  if (!dp_parts(*n, *nl, part, total)) {
    *why = BCIR_ERR_OVERFLOW;
    return 0;
  }
  return 1;
}

int bcir_decoder_pack_sizes(const uint8_t *pack, size_t pack_len, const uint8_t *plan,
                            size_t plan_len, size_t *records, size_t *work_bytes, int *why) {
  size_t n, nl, part[5], total;
  int none = 0;
  if (!why) why = &none;
  *why = 0;
  if (!pack || !plan || !records || !work_bytes) return BCIR_DT_INVALID;
  if (!dp_sizes(pack, pack_len, plan, plan_len, &n, &nl, part, &total, why))
    return BCIR_DT_INVALID;
  *records = n;
  *work_bytes = total;
  return BCIR_DT_OK;
}

/* a refusal: the first reason a walk recorded, else the walk's status, else a claim it missed;
 * the `written` records a writing dispatch began zeroed again (none: the checking pass refuses
 * whatever the writing pass could) */
static int dp_refuse(dp_ctx *c, int *why, bcir_status st, size_t written) {
  if (c->program && written)
    memset(c->program, 0, written * BCIR_DT_RECORD_WORDS * sizeof(uint64_t));
  *why = c->why ? c->why : st != BCIR_OK ? (int)st : BCIR_DP_CLAIMS;
  return BCIR_DT_INVALID;
}

int bcir_decoder_lower_pack(const bcir_decoder_state *s, const uint8_t *pack, size_t pack_len,
                            const uint8_t *plan, size_t plan_len, void *work, size_t work_bytes,
                            uint64_t *program, size_t capacity, size_t *records, int *why) {
  dp_ctx c;
  size_t n = 0, nl = 0, part[5], total = 0;
  int none = 0;
  bcir_status st;
  bcir_ep_header eh;
  bcir_streampack_header ph;
  bcir_exec_result res;
  unsigned char *at;
  const bcir_decoder_spec *sp;
  float eps;
  uint32_t bits;
  if (!why) why = &none;
  *why = 0;
  if (!records) return BCIR_DT_INVALID;
  *records = 0;
  if (!s || !pack || !plan) return BCIR_DT_INVALID;
  sp = &s->plan.spec;
  if (!sp->batch || !sp->time || !sp->width || !sp->heads || !sp->kvheads || !sp->vocab ||
      sp->width % sp->heads)
    return BCIR_DT_INVALID;
  if (!dp_sizes(pack, pack_len, plan, plan_len, &n, &nl, part, &total, why))
    return BCIR_DT_INVALID;
  if (!work || ((uintptr_t)work & 7u) || (n && !program)) return BCIR_DT_INVALID;
  if (work_bytes < total || capacity < n) return BCIR_DT_CAPACITY;
  memset(&c, 0, sizeof c);
  at = (unsigned char *)work;
  c.claims = (dp_claim *)(void *)at;
  at += part[0];
  c.lives = (dp_life *)(void *)at;
  at += part[1];
  c.order = (uint32_t *)(void *)at;
  at += part[2];
  memset(c.claims, 0, part[0]);
  c.n = n;
  c.nl = nl;
  c.program = program;
  c.rows = (uint64_t)sp->batch * sp->time;
  c.d = sp->width;
  c.h = sp->heads;
  c.kh = sp->kvheads;
  c.hd = sp->width / sp->heads;
  c.v = sp->vocab;
  c.b = sp->batch;
  c.t = sp->time;
  c.params = s->plan.parameters;
  c.chunk = bcir_decoder_reduce_chunk(s->plan.parameters);
  eps = (float)sp->rms_eps;
  memcpy(&bits, &eps, sizeof bits);
  c.eps = bits;
  c.gc = (s->contract & BCIR_DT_CONTRACT_BF16)     ? 3u
         : (s->contract & BCIR_DT_CONTRACT_KBLOCK) ? 2u
         : (s->contract & BCIR_DT_CONTRACT_FUSED)  ? 1u
                                                   : 0u;
  c.attn = ((s->contract & BCIR_DT_CONTRACT_EXP) ? BCIR_TENSOR_ATTENTION_EXP : 0u) |
           ((s->contract & BCIR_DT_CONTRACT_ATTN) ? BCIR_TENSOR_ATTENTION_FUSED : 0u);
  /* the binary32 backward under P-BANDS, the layout its stored forward's word names */
  c.attn_vjp =
      (s->contract & BCIR_DT_CONTRACT_ATTNBF16) ? 256u : c.attn | BCIR_TENSOR_ATTENTION_BANDS;
  /* a program's forward is a stored one: the bf16 entry's flagged 256, the binary32 entry's under
   * P-LOWER and P-BANDS (bcir_decoder_train.c's attention_forward_contract) */
  c.attn |= (s->contract & BCIR_DT_CONTRACT_ATTNBF16)
                ? 256u
                : BCIR_TENSOR_ATTENTION_LOWER | BCIR_TENSOR_ATTENTION_BANDS;
  c.v2 = (s->contract & BCIR_DT_CONTRACT_EXP) != 0;
  c.reduce = (s->contract & BCIR_DT_CONTRACT_REDUCE) != 0;
  c.provider = s->provider.mm != NULL;
  c.lent = s->workspace && s->workspace_bytes;
  /* the two verified by their own laws, the plan's walk handing over its steps (PX-3b-2: their
   * measured alternatives) and its lifetimes as it verifies them: one walk of the plan's laws
   * where a check and two walks would take three, its alias law swept over the executor's bytes
   * (ALIAS-SWEEP), lent before the executor takes them. A status refuses ahead of any reason the
   * steps or the lifetimes gave. */
  st = bcir_ep_check_pack_walk(plan, plan_len, pack, pack_len, dp_on_step, dp_on_life, &c, at,
                               part[3] + part[4]);
  if (st != BCIR_OK) {
    *why = (int)st;
    return BCIR_DT_INVALID;
  }
  if (bcir_ep_validate(plan, plan_len, &eh) != BCIR_OK ||
      bcir_sp_validate(pack, pack_len, &ph) != BCIR_OK)
    return BCIR_DT_INVALID; /* unreachable: both just verified */
  if (eh.liveness != BCIR_EP_LIVENESS_PHASE) {
    *why = BCIR_DP_LIVENESS;
    return BCIR_DT_INVALID;
  }
  if (ph.n_blocks != ph.n_segments) {
    *why = BCIR_DP_CLAIMS;
    return BCIR_DT_INVALID;
  }
  if (c.why || c.at != nl) return dp_refuse(&c, why, BCIR_OK, 0);
  c.at = 0;
  st = bcir_sp_for_each_segment(pack, pack_len, dp_on_seg, &c);
  if (st != BCIR_OK || c.why || c.at != n) return dp_refuse(&c, why, st, 0);
  c.at = 0;
  st = bcir_sp_for_each_block(pack, pack_len, dp_on_block, &c);
  if (st != BCIR_OK || c.why || c.at != n) return dp_refuse(&c, why, st, 0);
  /* GEM's dispatch twice: the first checks every claim against its own record, the second, which
   * runs only once nothing was refused, writes the program -- so a refused pack writes nothing */
  for (c.writing = 0; c.writing < 2; c.writing++) {
    c.k = c.rank = c.index = c.micro = 0;
    c.started = 0;
    st = bcir_sp_execute(pack, pack_len, (bcir_exec_item *)(void *)at, n,
                         (bcir_phase_stat *)(void *)(at + part[3]), n, dp_on_item, &c, &res);
    if (c.why || st != BCIR_OK || res.executed != n || c.k != n)
      return dp_refuse(&c, why, st, c.writing ? (c.k < n ? c.k + 1u : n) : 0u);
  }
  *records = n;
  return BCIR_DT_OK;
}
