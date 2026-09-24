/*===- test_handoff.c - the G16 C harness (pack table, freeze, manifest-of-shards) ===
 *
 * Drives the freestanding hand-off (bcir_handoff.h), the per-step freeze
 * (bcir_hydrate_generations) and the manifest-of-shards (bcir_shard_manifest.h) so every G16 row
 * runs on the C rail from one harness; the Python rails are bcir/gem/handoff.py and
 * bcir/abi/shard_manifest.py, graded by bcir/tests/handoff_fixtures.py.
 *
 *   --script <file>  scenarios of table and plane operations; one trace line per operation:
 *       "<scenario> <step> <verdict> <refusal> <status|-> h=<index:epoch|-> g=<generation>
 *       c=<claims digest|-> t=<table digest> p=<plane digest>" -- identical, line for line, to
 *       the Python rail's (handoff_fixtures.trace_line).
 *   --freeze <file>  per case: the step's claims planned (bcir_plan_func) and frozen
 *       (bcir_hydrate_generations): "OK <pack hex>" or "ERR <status>".
 *   --split <file>   per case (a whole pack and its ranges): "OK <manifest hex> <frame hex>
 *       <shard hex>..." or "ERR <status>" (bcir_shm_frame / _sub_pack / _encode).
 *   --decode <file>  per manifest: "OK <n_shards> <n_segments> <whole> <frame> <map> <data>
 *       <topo> <n_gens>" or "ERR <status>".
 *   --reassemble <file>  per case (a manifest and its store): "OK <sha256 of the whole>" or
 *       "ERR <status>".
 *   --partition <file>   per (n_segments, world): "OK b:e,..." or "ERR <status>".
 *   --api            the API's own fail-closed laws no script reaches: prints "OK <n>".
 *   --stage3 <file>  the Stage 3 exit flow (test_stage3.h) over this rail's pack table.
 *
 * Inputs are little-endian and u32-length framed (see handoff_fixtures.py). The exit status is 0
 * when the input was well-formed (the verdicts are data, graded by the caller) and --api held;
 * 2 on a usage or input error; 1 when --api found a violation. The harness uses libc (it is a
 * test); the units stay freestanding.
 *===----------------------------------------------------------------------===*/
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_control_plane.h"
#include "bcir_handoff.h"
#include "bcir_hydrate.h"
#include "bcir_plan.h"
#include "bcir_sha256.h"
#include "bcir_shard_manifest.h"
#include "test_stage3.h"

static const char *status_name(bcir_status s) {
  switch (s) {
    case BCIR_OK:              return "BCIR_OK";
    case BCIR_ERR_TRUNCATED:   return "BCIR_ERR_TRUNCATED";
    case BCIR_ERR_MAGIC:       return "BCIR_ERR_MAGIC";
    case BCIR_ERR_VERSION:     return "BCIR_ERR_VERSION";
    case BCIR_ERR_CRC:         return "BCIR_ERR_CRC";
    case BCIR_ERR_NOSPACE:     return "BCIR_ERR_NOSPACE";
    case BCIR_ERR_LANE:        return "BCIR_ERR_LANE";
    case BCIR_ERR_WIDTH:       return "BCIR_ERR_WIDTH";
    case BCIR_ERR_DISPATCH:    return "BCIR_ERR_DISPATCH";
    case BCIR_ERR_PROVENANCE:  return "BCIR_ERR_PROVENANCE";
    case BCIR_ERR_STALE:       return "BCIR_ERR_STALE";
    case BCIR_ERR_OVERFLOW:    return "BCIR_ERR_OVERFLOW";
    case BCIR_ERR_TRAILING:    return "BCIR_ERR_TRAILING";
    case BCIR_ERR_RESERVED:    return "BCIR_ERR_RESERVED";
    case BCIR_ERR_UTF8:        return "BCIR_ERR_UTF8";
    case BCIR_ERR_GENERATION:  return "BCIR_ERR_GENERATION";
    case BCIR_ERR_PLAN:        return "BCIR_ERR_PLAN";
    case BCIR_ERR_CONTROL:     return "BCIR_ERR_CONTROL";
    case BCIR_ERR_MAC:         return "BCIR_ERR_MAC";
    case BCIR_ERR_TELEMETRY:   return "BCIR_ERR_TELEMETRY";
    case BCIR_ERR_RING:        return "BCIR_ERR_RING";
    case BCIR_ERR_FULL:        return "BCIR_ERR_FULL";
    case BCIR_ERR_BUSY:        return "BCIR_ERR_BUSY";
    case BCIR_ERR_LIFETIME:    return "BCIR_ERR_LIFETIME";
    case BCIR_ERR_SHARD:       return "BCIR_ERR_SHARD";
  }
  return "BCIR_ERR_UNKNOWN";
}

/* --- input ------------------------------------------------------------------------------- */

typedef struct rd {
  const uint8_t *d;
  size_t len, pos;
  int err;
} rd;

static const uint8_t *take(rd *r, size_t n) {
  const uint8_t *p = r->d + r->pos;
  if (r->err || n > r->len - r->pos) { r->err = 1; return NULL; }
  r->pos += n;
  return p;
}
static uint8_t u8(rd *r) { const uint8_t *p = take(r, 1); return p ? p[0] : 0u; }
static uint32_t u32(rd *r) {
  const uint8_t *p = take(r, 4);
  return p ? (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24)
           : 0u;
}
static uint64_t u64(rd *r) { uint64_t lo = u32(r); return lo | ((uint64_t)u32(r) << 32); }
static const uint8_t *blob(rd *r, size_t *n) { *n = u32(r); return take(r, *n); }

static uint8_t *slurp(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  uint8_t *buf = NULL;
  size_t cap = 0, n = 0;
  if (!f) return NULL;
  for (;;) {
    if (n == cap) {
      uint8_t *grown = realloc(buf, cap ? cap * 2u : 65536u);
      if (!grown) { free(buf); fclose(f); return NULL; }
      buf = grown;
      cap = cap ? cap * 2u : 65536u;
    }
    size_t got = fread(buf + n, 1, cap - n, f);
    n += got;
    if (got == 0) break;
  }
  fclose(f);
  *len = n;
  return buf;
}

static void hex(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
}

/* --- a step's claim graph ------------------------------------------------------------------ */

typedef struct graph {
  bcir_func f;
  bcir_claim *claims;
  bcir_plan_step *steps;
  bcir_generation_view *gens;
  uint32_t n_gens, topo;
} graph;

/* u32 topo, u32 n_gens, n x (rid, map, data), u32 n_claims, per claim: u32 id, u8 opcode,
 * u8 lane, u8 domain, u32 count, u8 n_rd, u32 x n_rd, u8 n_wr, u32 x n_wr, u8 label_len, label */
static int read_graph(rd *r, graph *g) {
  memset(g, 0, sizeof *g);
  g->topo = u32(r);
  g->n_gens = u32(r);
  if (r->err || g->n_gens > 1000000u) return 0;
  g->gens = calloc(g->n_gens ? g->n_gens : 1u, sizeof *g->gens);
  for (uint32_t i = 0; i < g->n_gens && g->gens; i++) {
    g->gens[i].rid = u32(r);
    g->gens[i].map_gen = u32(r);
    g->gens[i].data_gen = u32(r);
  }
  uint32_t n = u32(r);
  if (r->err || n > 1000000u || !g->gens) return 0;
  g->claims = calloc(n ? n : 1u, sizeof *g->claims);
  g->steps = calloc(n ? n : 1u, sizeof *g->steps);
  if (!g->claims || !g->steps) return 0;
  for (uint32_t i = 0; i < n; i++) {
    bcir_claim *c = &g->claims[i];
    c->id = u32(r);
    c->opcode = (bcir_opcode)u8(r);
    c->lane = u8(r);
    c->domain = (bcir_domain)u8(r);
    c->count = u32(r);
    c->n_rd = u8(r);
    for (uint8_t k = 0; k < c->n_rd; k++) {
      uint32_t v = u32(r);
      if (k < BCIR_CLAIM_MAX_RD) c->rd[k] = v;
    }
    c->n_wr = u8(r);
    for (uint8_t k = 0; k < c->n_wr; k++) {
      uint32_t v = u32(r);
      if (k < BCIR_CLAIM_MAX_WR) c->wr[k] = v;
    }
    size_t ll = u8(r);
    const uint8_t *label = take(r, ll);
    if (!label) return 0;
    memcpy(c->op, label, ll < sizeof c->op ? ll : sizeof c->op);
    if (ll < sizeof c->op) c->op[ll] = '\0';  /* 32 bytes and up: no NUL, label_ok refuses */
  }
  g->f.claims = g->claims;
  g->f.n_claims = n;
  g->f.cap_claims = n;
  return !r->err;
}

static void free_graph(graph *g) {
  free(g->claims);
  free(g->steps);
  free(g->gens);
}

/* plan + freeze into (buf, cap) -- the builder's step, as the C++ seam performs it. */
static bcir_status freeze_graph(graph *g, uint8_t *buf, size_t cap, size_t *len) {
  bcir_plan plan;
  *len = 0;
  bcir_status st = bcir_plan_func(&g->f, g->steps, g->f.n_claims, &plan);
  if (st != BCIR_OK) return st;
  return bcir_hydrate_generations(&g->f, &plan, g->topo, g->gens, g->n_gens, buf, cap, len);
}

/* --- --script ----------------------------------------------------------------------------- */

enum {
  OP_SUBMIT = 1, OP_ENTER = 2, OP_LEAVE = 3, OP_ADVANCE = 4, OP_RESERVE = 5, OP_WRITE = 6,
  OP_COMMIT = 7, OP_ABORT = 8, OP_BORROW = 9, OP_GIVE_BACK = 10, OP_RELEASE = 11, OP_ADMIT = 12,
  OP_DISPATCH = 13, OP_ADMIT_MANIFEST = 14, OP_FREEZE = 15, OP_FORGE = 16, OP_POKE_EPOCH = 17,
  OP_POKE_PINS = 18, OP_COPY_VIEW = 19, OP_RESTART = 20
};
#define REGS 16u

typedef struct claims_ctx {
  bcir_sha256 h;
  uint32_t n;
} claims_ctx;

static int collect(const bcir_segment_view *seg, void *ctx) {
  claims_ctx *c = (claims_ctx *)ctx;
  uint8_t b[8];
  for (int i = 0; i < 8; i++) b[i] = (uint8_t)(seg->claim_id >> (8 * i));
  bcir_sha256_update(&c->h, b, sizeof b);
  c->n++;
  return 0;
}

static void trace(uint32_t scn, uint32_t step, bcir_ho_outcome o, int op, const uint8_t *claims,
                  const bcir_ho_table *t, const bcir_ctl_state *plane) {
  uint8_t td[32], pd[32];
  int hide = (op == OP_ADMIT || op == OP_DISPATCH) &&
             (o.refusal == BCIR_HO_REFUSAL_MALFORMED || o.refusal == BCIR_HO_REFUSAL_WALK);
  bcir_ho_state_digest(t, td);
  bcir_ctl_state_digest(plane, pd);
  printf("%u %u %u %u %s ", scn, step, o.verdict, o.refusal, hide ? "-" : status_name(o.status));
  if (o.handle.epoch) printf("h=%u:%u ", o.handle.index, o.handle.epoch); else printf("h=- ");
  printf("g=%u c=", o.generation);
  if (claims) hex(claims, 16); else printf("-");
  printf(" t=");
  hex(td, 32);
  printf(" p=");
  hex(pd, 32);
  printf("\n");
}

static bcir_ho_outcome plane_step(bcir_ctl_outcome c) {
  /* A plane operation in a table trace: applied/none -> applied, deferred -> applied (the
   * switch is held), refused -> refused. The plane digest carries the rest. */
  bcir_ho_outcome o;
  memset(&o, 0, sizeof o);
  o.verdict = c.verdict == BCIR_CTL_REFUSED ? BCIR_HO_REFUSED : BCIR_HO_APPLIED;
  o.refusal = 0;
  o.generation = c.generation;
  o.status = c.status;
  return o;
}

static int run_script(const char *path) {
  size_t len;
  uint8_t *data = slurp(path, &len);
  rd r = {data, len, 0, 0};
  if (!data) return 2;
  const uint8_t *magic = take(&r, 4);
  if (!magic || memcmp(magic, "BHOS", 4) != 0) { free(data); return 2; }
  uint32_t n_scn = u32(&r);
  for (uint32_t scn = 0; scn < n_scn && !r.err; scn++) {
    uint32_t n_slots = u32(&r), capacity = u32(&r);
    size_t key_len;
    const uint8_t *key = blob(&r, &key_len);
    uint8_t scope = u8(&r);
    uint64_t subject = u64(&r);
    uint32_t n_steps = u32(&r);
    if (r.err || n_slots == 0u || n_slots > 4096u || capacity == 0u || capacity > (1u << 24)) break;
    bcir_ho_slot *slots = calloc(n_slots, sizeof *slots);
    uint8_t *arena = calloc(n_slots, capacity);
    bcir_ho_table table;
    bcir_ctl_state plane;
    bcir_ho_handle regs[REGS];
    bcir_ho_view views[REGS];
    memset(regs, 0, sizeof regs);
    memset(views, 0, sizeof views);
    if (!slots || !arena || bcir_ho_init(&table, slots, n_slots, arena, (size_t)n_slots * capacity,
                                         capacity) != BCIR_OK ||
        bcir_ctl_init(&plane, key, key_len, scope, subject) != BCIR_OK) {
      free(slots); free(arena); free(data);
      return 2;
    }
    for (uint32_t step = 0; step < n_steps && !r.err; step++) {
      uint8_t op = u8(&r), reg = u8(&r), vreg = u8(&r);
      size_t dlen;
      const uint8_t *d = blob(&r, &dlen);
      bcir_ho_outcome o;
      uint8_t claims_digest[32];
      const uint8_t *claims = NULL;
      if (r.err || reg >= REGS || vreg >= REGS) { r.err = 1; break; }
      memset(&o, 0, sizeof o);
      switch (op) {
        case OP_SUBMIT: o = plane_step(bcir_ctl_submit(&plane, d, dlen)); break;
        case OP_ENTER: o = plane_step(bcir_ctl_enter(&plane)); break;
        case OP_LEAVE: o = plane_step(bcir_ctl_leave(&plane)); break;
        case OP_ADVANCE: o = plane_step(bcir_ctl_advance(&plane)); break;
        case OP_RESTART: /* the plane restarts: generation 0, no registry, the same key */
          if (bcir_ctl_init(&plane, key, key_len, scope, subject) != BCIR_OK) { r.err = 1; break; }
          o.verdict = BCIR_HO_APPLIED;
          o.generation = plane.generation;
          o.status = BCIR_OK;
          break;
        case OP_RESERVE:
          o = bcir_ho_reserve(&table);
          if (o.verdict == BCIR_HO_APPLIED) regs[reg] = o.handle;
          break;
        case OP_WRITE: {
          uint8_t *region;
          size_t cap;
          rd w = {d, dlen, 0, 0};
          uint32_t offset = u32(&w);
          const uint8_t *bytes = take(&w, dlen >= 4u ? dlen - 4u : 0u);
          if (w.err) { r.err = 1; break; }
          if (bcir_ho_reserved(&table, regs[reg], &region, &cap) != BCIR_OK) {
            o.verdict = BCIR_HO_REFUSED; o.refusal = BCIR_HO_REFUSAL_LIFETIME;
            o.status = BCIR_ERR_LIFETIME;
          } else if ((size_t)offset + (dlen - 4u) > cap) {
            o.verdict = BCIR_HO_REFUSED; o.refusal = BCIR_HO_REFUSAL_CAPACITY;
            o.status = BCIR_ERR_NOSPACE;
          } else {
            memcpy(region + offset, bytes, dlen - 4u);
            o.verdict = BCIR_HO_APPLIED;
            o.handle = regs[reg];
            o.status = BCIR_OK;
          }
          break;
        }
        case OP_COMMIT: {
          rd w = {d, dlen, 0, 0};
          uint64_t length = u64(&w);
          if (w.err) { r.err = 1; break; }
          o = bcir_ho_commit(&table, regs[reg], (size_t)length);
          break;
        }
        case OP_ABORT: o = bcir_ho_abort(&table, regs[reg]); break;
        case OP_BORROW: o = bcir_ho_borrow(&table, regs[reg], &views[vreg]); break;
        case OP_GIVE_BACK:
          o = bcir_ho_give_back(&table, &views[vreg]);
          if (o.verdict == BCIR_HO_APPLIED) memset(&views[vreg], 0, sizeof views[vreg]); /* returned */
          break;
        case OP_RELEASE: o = bcir_ho_release(&table, regs[reg]); break;
        case OP_ADMIT: o = bcir_ho_admit(&table, regs[reg], &plane); break;
        case OP_DISPATCH: {
          claims_ctx cc;
          bcir_sha256_init(&cc.h);
          cc.n = 0;
          o = bcir_ho_dispatch(&table, regs[reg], &plane, collect, &cc);
          bcir_sha256_final(&cc.h, claims_digest);
          if (o.verdict == BCIR_HO_APPLIED) claims = claims_digest;
          break;
        }
        case OP_ADMIT_MANIFEST: o = bcir_ho_admit_manifest(&plane, d, dlen); break;
        case OP_FREEZE: {
          graph g;
          rd w = {d, dlen, 0, 0};
          int ok = read_graph(&w, &g);
          if (!ok) { free_graph(&g); r.err = 1; break; }
          o = bcir_ho_reserve(&table);
          if (o.verdict == BCIR_HO_APPLIED) {
            bcir_ho_handle h = o.handle;
            uint8_t *region;
            size_t cap, got;
            (void)bcir_ho_reserved(&table, h, &region, &cap);
            bcir_status st = freeze_graph(&g, region, cap, &got);
            if (st != BCIR_OK) {
              (void)bcir_ho_abort(&table, h);
              memset(&o, 0, sizeof o);
              o.verdict = BCIR_HO_REFUSED;
              o.refusal = st == BCIR_ERR_NOSPACE ? BCIR_HO_REFUSAL_CAPACITY : BCIR_HO_REFUSAL_MALFORMED;
              o.status = st;
            } else {
              o = bcir_ho_commit(&table, h, got);
              if (o.verdict == BCIR_HO_APPLIED) regs[reg] = h;
            }
          }
          free_graph(&g);
          break;
        }
        case OP_FORGE: {
          rd w = {d, dlen, 0, 0};
          regs[reg].index = u32(&w);
          regs[reg].epoch = u32(&w);
          o.verdict = BCIR_HO_APPLIED;
          break;
        }
        case OP_POKE_EPOCH: {
          rd w = {d, dlen, 0, 0};
          uint32_t index = u32(&w), epoch = u32(&w);
          if (index < table.n_slots) table.slots[index].epoch = epoch;
          o.verdict = BCIR_HO_APPLIED;
          break;
        }
        case OP_POKE_PINS: {
          rd w = {d, dlen, 0, 0};
          uint32_t index = u32(&w), pins = u32(&w);
          if (index < table.n_slots) table.slots[index].pins = pins;
          o.verdict = BCIR_HO_APPLIED;
          break;
        }
        case OP_COPY_VIEW: views[vreg] = views[reg]; o.verdict = BCIR_HO_APPLIED; break;
        default: r.err = 1; break;
      }
      if (r.err) break;
      trace(scn, step, o, op, claims, &table, &plane);
    }
    free(slots);
    free(arena);
  }
  free(data);
  return r.err ? 2 : 0;
}

/* --- --freeze / --split / --decode / --reassemble / --partition -------------------------- */

static int run_freeze(const char *path) {
  size_t len;
  uint8_t *data = slurp(path, &len);
  rd r = {data, len, 0, 0};
  if (!data) return 2;
  uint32_t n = u32(&r);
  for (uint32_t i = 0; i < n && !r.err; i++) {
    size_t glen;
    const uint8_t *gd = blob(&r, &glen);
    uint32_t cap = u32(&r);
    graph g;
    rd w = {gd, glen, 0, 0};
    if (r.err || !read_graph(&w, &g)) { free_graph(&g); r.err = 1; break; }
    uint8_t *out = malloc(cap ? cap : 1u);
    size_t got;
    bcir_status st = freeze_graph(&g, out, cap, &got);
    if (st == BCIR_OK) { printf("OK "); hex(out, got); printf("\n"); }
    else printf("ERR %s\n", status_name(st));
    free(out);
    free_graph(&g);
  }
  free(data);
  return r.err ? 2 : 0;
}

static int run_split(const char *path) {
  size_t len;
  uint8_t *data = slurp(path, &len);
  rd r = {data, len, 0, 0};
  if (!data) return 2;
  uint32_t n = u32(&r);
  for (uint32_t i = 0; i < n && !r.err; i++) {
    size_t wlen;
    const uint8_t *whole = blob(&r, &wlen);
    uint32_t k = u32(&r);
    if (r.err || k > BCIR_SHM_SHARDS_MAX) { r.err = 1; break; }
    uint32_t *b = calloc(k ? k : 1u, 4), *e = calloc(k ? k : 1u, 4);
    uint8_t **shards = calloc(k ? k : 1u, sizeof *shards);
    size_t *lens = calloc(k ? k : 1u, sizeof *lens);
    for (uint32_t j = 0; j < k; j++) { b[j] = u32(&r); e[j] = u32(&r); }
    bcir_status st = BCIR_OK;
    size_t flen = 0, mlen = 0;
    uint8_t *frame = NULL, *man = NULL;
    st = bcir_shm_frame(whole, wlen, NULL, 0, &flen);
    if (st == BCIR_OK) {
      frame = malloc(flen);
      st = bcir_shm_frame(whole, wlen, frame, flen, &flen);
    }
    for (uint32_t j = 0; j < k && st == BCIR_OK; j++) {
      size_t slen = 0;
      st = bcir_shm_sub_pack(whole, wlen, b[j], e[j], NULL, 0, &slen);
      if (st != BCIR_OK) break;
      shards[j] = malloc(slen);
      st = bcir_shm_sub_pack(whole, wlen, b[j], e[j], shards[j], slen, &lens[j]);
    }
    if (st == BCIR_OK) {
      st = bcir_shm_encode(whole, wlen, frame, flen, b, e, (const uint8_t *const *)shards, lens, k,
                           NULL, 0, &mlen);
      if (st == BCIR_OK) {
        man = malloc(mlen);
        st = bcir_shm_encode(whole, wlen, frame, flen, b, e, (const uint8_t *const *)shards, lens,
                             k, man, mlen, &mlen);
      }
    }
    if (st == BCIR_OK) {
      printf("OK ");
      hex(man, mlen);
      printf(" ");
      hex(frame, flen);
      for (uint32_t j = 0; j < k; j++) { printf(" "); hex(shards[j], lens[j]); }
      printf("\n");
    } else {
      printf("ERR %s\n", status_name(st));
    }
    for (uint32_t j = 0; j < k; j++) free(shards[j]);
    free(shards); free(lens); free(b); free(e); free(frame); free(man);
  }
  free(data);
  return r.err ? 2 : 0;
}

static int run_decode(const char *path) {
  size_t len;
  uint8_t *data = slurp(path, &len);
  rd r = {data, len, 0, 0};
  if (!data) return 2;
  uint32_t n = u32(&r);
  for (uint32_t i = 0; i < n && !r.err; i++) {
    size_t mlen;
    const uint8_t *m = blob(&r, &mlen);
    bcir_shm_view v;
    if (r.err) break;
    bcir_status st = bcir_shm_decode(m, mlen, &v);
    if (st != BCIR_OK) { printf("ERR %s\n", status_name(st)); continue; }
    printf("OK %u %u %" PRIu64 " %" PRIu64 " %u %u %u %u\n", v.n_shards, v.n_segments,
           v.whole_length, v.frame_length, v.map_gen, v.data_gen, v.topo_gen, v.n_gens);
  }
  free(data);
  return r.err ? 2 : 0;
}

typedef struct store {
  const uint8_t **blobs;
  size_t *lens;
  uint8_t (*digests)[32];
  uint32_t n;
} store;

static int fetch(const uint8_t digest[32], const uint8_t **data, size_t *len, void *ctx) {
  store *s = (store *)ctx;
  for (uint32_t i = 0; i < s->n; i++)
    if (memcmp(s->digests[i], digest, 32) == 0) { *data = s->blobs[i]; *len = s->lens[i]; return 0; }
  return 1;
}

static int run_reassemble(const char *path) {
  size_t len;
  uint8_t *data = slurp(path, &len);
  rd r = {data, len, 0, 0};
  if (!data) return 2;
  uint32_t n = u32(&r);
  for (uint32_t i = 0; i < n && !r.err; i++) {
    size_t mlen;
    const uint8_t *m = blob(&r, &mlen);
    uint32_t cap = u32(&r);
    store s;
    s.n = u32(&r);
    if (r.err || s.n > 100000u) { r.err = 1; break; }
    s.blobs = calloc(s.n ? s.n : 1u, sizeof *s.blobs);
    s.lens = calloc(s.n ? s.n : 1u, sizeof *s.lens);
    s.digests = calloc(s.n ? s.n : 1u, sizeof *s.digests);
    for (uint32_t j = 0; j < s.n; j++) {
      s.blobs[j] = blob(&r, &s.lens[j]);
      if (s.blobs[j]) bcir_sha256_digest(s.blobs[j], s.lens[j], s.digests[j]);
    }
    if (r.err) { free(s.blobs); free(s.lens); free(s.digests); break; }
    uint8_t *out = malloc(cap ? cap : 1u);
    size_t got = 0;
    bcir_status st = bcir_shm_reassemble(m, mlen, fetch, &s, out, cap, &got);
    if (st == BCIR_OK) {
      uint8_t d[32];
      bcir_sha256_digest(out, got, d);
      printf("OK ");
      hex(d, 32);
      printf("\n");
    } else {
      printf("ERR %s\n", status_name(st));
    }
    free(out); free(s.blobs); free(s.lens); free(s.digests);
  }
  free(data);
  return r.err ? 2 : 0;
}

static int run_partition(const char *path) {
  size_t len;
  uint8_t *data = slurp(path, &len);
  rd r = {data, len, 0, 0};
  if (!data) return 2;
  uint32_t n = u32(&r);
  uint32_t *b = calloc(BCIR_SHM_SHARDS_MAX, 4), *e = calloc(BCIR_SHM_SHARDS_MAX, 4);
  for (uint32_t i = 0; i < n && !r.err; i++) {
    uint32_t segs = u32(&r), world = u32(&r), count = 0;
    if (r.err) break;
    bcir_status st = bcir_shm_partition(segs, world, b, e, BCIR_SHM_SHARDS_MAX, &count);
    if (st != BCIR_OK) { printf("ERR %s\n", status_name(st)); continue; }
    printf("OK ");
    for (uint32_t j = 0; j < count; j++) printf("%s%u:%u", j ? "," : "", b[j], e[j]);
    printf("\n");
  }
  free(b); free(e); free(data);
  return r.err ? 2 : 0;
}

/* --- --api: the fail-closed laws no script reaches --------------------------------------- */

static int checks, failures;
#define CHECK(cond, what)                                   \
  do {                                                      \
    checks++;                                               \
    if (!(cond)) { failures++; printf("FAIL %s\n", what); } \
  } while (0)

static int run_api(void) {
  bcir_ho_table t;
  bcir_ho_slot slots[4];
  uint8_t arena[4 * 256];
  bcir_ho_view v;
  bcir_ho_handle h, zero = {0, 0};
  uint8_t *p;
  size_t cap, n;
  uint32_t b[8], e[8], count;
  bcir_shm_view mv;

  /* init: every bad shape refused, the table left zeroed */
  CHECK(bcir_ho_init(NULL, slots, 4, arena, sizeof arena, 256) == BCIR_ERR_NOSPACE, "init NULL table");
  CHECK(bcir_ho_init(&t, NULL, 4, arena, sizeof arena, 256) == BCIR_ERR_NOSPACE && t.slots == NULL,
        "init NULL slots");
  CHECK(bcir_ho_init(&t, slots, 0, arena, sizeof arena, 256) == BCIR_ERR_NOSPACE, "init no slots");
  CHECK(bcir_ho_init(&t, slots, 65536u, arena, sizeof arena, 256) == BCIR_ERR_NOSPACE,
        "init past the slot bound");
  CHECK(bcir_ho_init(&t, slots, 4, arena, sizeof arena, 0) == BCIR_ERR_NOSPACE, "init capacity 0");
  CHECK(bcir_ho_init(&t, slots, 4, arena, sizeof arena - 1u, 256) == BCIR_ERR_NOSPACE,
        "init short arena");
  CHECK(bcir_ho_init(&t, slots, 4, arena, SIZE_MAX, SIZE_MAX / 2u) == BCIR_ERR_NOSPACE,
        "init n * capacity overflow");
  CHECK(bcir_ho_reserve(&t).status == BCIR_ERR_FULL, "reserve on a refused table");
  CHECK(bcir_ho_init(&t, slots, 4, arena, sizeof arena, 256) == BCIR_OK, "init");
  /* the zero handle is never live; out-of-range indices are refused */
  CHECK(bcir_ho_borrow(&t, zero, &v).status == BCIR_ERR_LIFETIME, "zero handle");
  h.index = 99; h.epoch = 1;
  CHECK(bcir_ho_release(&t, h).status == BCIR_ERR_LIFETIME, "out-of-range handle");
  CHECK(bcir_ho_reserved(&t, h, &p, &cap) == BCIR_ERR_LIFETIME && p == NULL && cap == 0,
        "reserved on a dead handle");
  h = bcir_ho_reserve(&t).handle;
  CHECK(bcir_ho_reserved(&t, h, NULL, &cap) == BCIR_ERR_LIFETIME, "reserved NULL out");
  CHECK(bcir_ho_borrow(&t, h, NULL).status == BCIR_ERR_LIFETIME, "borrow NULL view");
  CHECK(bcir_ho_borrow(&t, h, &v).status == BCIR_ERR_LIFETIME, "borrow a building slot");
  CHECK(bcir_ho_give_back(&t, NULL).status == BCIR_ERR_LIFETIME, "give back NULL");
  CHECK(bcir_ho_commit(&t, h, 257).status == BCIR_ERR_NOSPACE, "commit past capacity");
  CHECK(bcir_ho_commit(&t, h, 0).status == BCIR_ERR_NOSPACE, "commit nothing");
  CHECK(bcir_ho_commit(&t, h, 8).verdict == BCIR_HO_APPLIED, "commit");
  CHECK(bcir_ho_admit(&t, h, NULL).status == BCIR_ERR_STALE, "admit without a plane");
  CHECK(bcir_ho_dispatch(&t, h, NULL, NULL, NULL).refusal == BCIR_HO_REFUSAL_UNADMITTED,
        "dispatch without a plane");
  CHECK(bcir_ho_admit_manifest(NULL, NULL, 0).status == BCIR_ERR_TRUNCATED, "manifest NULL");

  /* partition */
  CHECK(bcir_shm_partition(10, 3, b, e, 8, NULL) == BCIR_ERR_NOSPACE, "partition NULL count");
  CHECK(bcir_shm_partition(10, 0, b, e, 8, &count) == BCIR_OK && count == 1 && e[0] == 10,
        "partition world 0 counts as 1");
  CHECK(bcir_shm_partition(0, 5, b, e, 8, &count) == BCIR_OK && count == 1 && b[0] == 0 && e[0] == 0,
        "partition an empty pack");
  CHECK(bcir_shm_partition(10, 4, b, e, 2, &count) == BCIR_ERR_NOSPACE && count == 0,
        "partition short capacity");
  CHECK(bcir_shm_partition(5000, 5000, b, e, 8, &count) == BCIR_ERR_SHARD,
        "partition past the shard bound");

  /* manifest API */
  CHECK(bcir_shm_decode(NULL, 0, &mv) == BCIR_ERR_TRUNCATED && mv.data == NULL, "decode NULL");
  CHECK(bcir_shm_decode(arena, 10, NULL) == BCIR_ERR_TRUNCATED, "decode NULL out");
  CHECK(bcir_shm_entry_at(NULL, 0, NULL) == BCIR_ERR_SHARD, "entry_at NULL");
  CHECK(bcir_shm_sub_pack(arena, 8, 0, 0, NULL, 0, NULL) == BCIR_ERR_NOSPACE, "sub_pack NULL len");
  CHECK(bcir_shm_sub_pack(NULL, 0, 0, 0, NULL, 0, &n) == BCIR_ERR_SHARD && n == 0, "sub_pack NULL");
  CHECK(bcir_shm_frame(arena, 8, NULL, 0, &n) == BCIR_ERR_SHARD, "frame of a non-pack");
  CHECK(bcir_shm_encode(NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, 1, NULL, 0, &n) == BCIR_ERR_SHARD,
        "encode NULL");
  CHECK(bcir_shm_reassemble(NULL, 0, NULL, NULL, NULL, 0, NULL) == BCIR_ERR_NOSPACE,
        "reassemble NULL len");
  CHECK(bcir_shm_check_whole(NULL, 0) == BCIR_ERR_SHARD, "check_whole NULL");

  /* freeze: the vector laws, the id and RID laws, capacity -- nothing written on failure */
  {
    bcir_claim cl[2];
    bcir_func f;
    bcir_generation_view g[2] = {{1, 0, 0}, {2, 0, 0}};
    bcir_generation_view bad[2] = {{2, 0, 0}, {1, 0, 0}};
    uint8_t out[512];
    memset(cl, 0, sizeof cl);
    memset(&f, 0, sizeof f);
    cl[0].id = 1; cl[0].opcode = BCIR_OP_ADD; cl[0].n_rd = 1; cl[0].rd[0] = 1;
    memcpy(cl[0].op, "c.add", 6);
    cl[1] = cl[0]; cl[1].id = 2;
    f.claims = cl; f.n_claims = 2;
    memset(out, 0xAB, sizeof out);
    CHECK(bcir_hydrate_generations(&f, NULL, 1, NULL, 0, out, sizeof out, &n) == BCIR_ERR_GENERATION &&
              n == 0 && out[0] == 0xAB,
          "freeze with no vector");
    CHECK(bcir_hydrate_generations(&f, NULL, 1, bad, 2, out, sizeof out, &n) == BCIR_ERR_GENERATION,
          "freeze with an unsorted vector");
    cl[1].id = 1;
    CHECK(bcir_hydrate_generations(&f, NULL, 1, g, 2, out, sizeof out, &n) == BCIR_ERR_PROVENANCE,
          "freeze with ids that do not ascend");
    cl[1].id = 2; cl[1].rd[0] = 9;
    CHECK(bcir_hydrate_generations(&f, NULL, 1, g, 2, out, sizeof out, &n) == BCIR_ERR_PROVENANCE,
          "freeze touching an undeclared RID");
    cl[1].rd[0] = 2;
    CHECK(bcir_hydrate_generations(&f, NULL, 1, g, 2, out, 16, &n) == BCIR_ERR_NOSPACE && n == 0 &&
              out[0] == 0xAB,
          "freeze into a short buffer writes nothing");
    CHECK(bcir_hydrate_generations(&f, NULL, 1, g, 2, out, sizeof out, &n) == BCIR_OK && n > 0 &&
              bcir_sp_verify_semantic(out, n, 0xFFFFFFFFu, 0xFFFFFFFFu) == BCIR_OK,
          "freeze");
  }
  printf("%s %d\n", failures ? "FAILED" : "OK", checks);
  return failures ? 1 : 0;
}

/* --- --stage3: the Stage 3 exit flow, the data boundary through the C pack table --------- */

typedef struct s3_table {
  bcir_ho_table t;
  bcir_ho_slot slots[4];
  uint8_t arena[4u * 4096u];
  bcir_ho_handle h[2];
} s3_table;

static bcir_ho_outcome s3c_store(void *ctx, int which, const uint8_t *data, size_t len) {
  s3_table *c = (s3_table *)ctx;
  bcir_ho_outcome o = bcir_ho_reserve(&c->t);
  uint8_t *region;
  size_t cap;
  if (o.verdict != BCIR_HO_APPLIED) return o;
  if (bcir_ho_reserved(&c->t, o.handle, &region, &cap) != BCIR_OK || len > cap) {
    memset(&o, 0, sizeof o);
    o.verdict = BCIR_HO_REFUSED;
    o.refusal = BCIR_HO_REFUSAL_CAPACITY;
    o.status = BCIR_ERR_NOSPACE;
    return o;
  }
  memcpy(region, data, len);
  o = bcir_ho_commit(&c->t, o.handle, len);
  if (o.verdict == BCIR_HO_APPLIED) c->h[which] = o.handle;
  return o;
}

static bcir_ho_outcome s3c_admit(void *ctx, int which, bcir_ctl_state *plane) {
  s3_table *c = (s3_table *)ctx;
  return bcir_ho_admit(&c->t, c->h[which], plane);
}

static bcir_ho_outcome s3c_dispatch(void *ctx, int which, bcir_ctl_state *plane, bcir_seg_fn fn,
                                    void *fn_ctx) {
  s3_table *c = (s3_table *)ctx;
  return bcir_ho_dispatch(&c->t, c->h[which], plane, fn, fn_ctx);
}

static bcir_ho_outcome s3c_release(void *ctx, int which) {
  s3_table *c = (s3_table *)ctx;
  return bcir_ho_release(&c->t, c->h[which]);
}

static void s3c_digest(void *ctx, uint8_t out[32]) {
  bcir_ho_state_digest(&((s3_table *)ctx)->t, out);
}

static int run_stage3(const char *path) {
  static s3_table c;
  s3_ops ops;
  size_t len;
  uint8_t *data = slurp(path, &len);
  int rc;
  if (!data) return 2;
  memset(&c, 0, sizeof c);
  if (bcir_ho_init(&c.t, c.slots, 4u, c.arena, sizeof c.arena, 4096u) != BCIR_OK) {
    free(data);
    return 2;
  }
  ops.ctx = &c;
  ops.store = s3c_store;
  ops.admit = s3c_admit;
  ops.dispatch = s3c_dispatch;
  ops.release = s3c_release;
  ops.digest = s3c_digest;
  ops.status = status_name;
  rc = s3_run(data, len, &ops);
  free(data);
  return rc;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--api") == 0) return run_api();
  if (argc != 3) {
    fprintf(stderr, "usage: test_handoff --script|--freeze|--split|--decode|--reassemble|"
                    "--partition|--stage3 <file> | --api\n");
    return 2;
  }
  if (strcmp(argv[1], "--script") == 0) return run_script(argv[2]);
  if (strcmp(argv[1], "--freeze") == 0) return run_freeze(argv[2]);
  if (strcmp(argv[1], "--split") == 0) return run_split(argv[2]);
  if (strcmp(argv[1], "--decode") == 0) return run_decode(argv[2]);
  if (strcmp(argv[1], "--reassemble") == 0) return run_reassemble(argv[2]);
  if (strcmp(argv[1], "--partition") == 0) return run_partition(argv[2]);
  if (strcmp(argv[1], "--stage3") == 0) return run_stage3(argv[2]);
  fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
