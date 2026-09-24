/*===- fuzz_handoff.c - libFuzzer: the data-plane hand-off (G16) ----------------===
 *
 * Four trust boundaries in one harness (the first input byte picks one):
 *   0. the shard-manifest decoder over raw bytes (bcir_shm_decode): total on arbitrary bytes; a
 *      manifest it accepts has every entry readable and its ranges partition [0, n_segments);
 *   1. a SHARD SET cut from the input's pack and then tampered with: a byte of the manifest, the
 *      frame or a shard flipped -- optionally RESEALED (the blob's CRC, its digest and length in
 *      the manifest and the manifest's CRC repaired, so the mutation reaches past every seal).
 *      The honest set must reassemble to the whole; a tampered one may only ever reassemble to
 *      exactly the whole (a set that yields any other bytes is a finding);
 *   2. the PACK TABLE driven by the input as a script -- reserve, write, commit, abort, borrow,
 *      give back, release, forged handles -- with the table's laws asserted after every
 *      operation: every slot's pin count is exactly the borrows the script holds on it, a free or
 *      building slot has none, epochs never go back, and a borrowed view's bytes never change
 *      until it is given back (no API writes a frozen or retired slot);
 *   3. the per-step FREEZE (bcir_plan_func + bcir_hydrate_generations) over a claim graph and a
 *      vector built from the input: a refusal leaves nothing written; a frozen step verifies, is
 *      v4, carries exactly the vector (its registry digest is the vector's) and may shard.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_control_plane.h"
#include "bcir_handoff.h"
#include "bcir_hydrate.h"
#include "bcir_plan.h"
#include "bcir_sha256.h"
#include "bcir_shard_manifest.h"

#define BLOB_MAX 16384u
#define SHARDS 8u

typedef struct cursor {
  const uint8_t *data;
  size_t size, at;
} cursor;

static uint8_t take(cursor *c) { return c->at < c->size ? c->data[c->at++] : 0u; }
static uint32_t take32(cursor *c) {
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) v |= (uint32_t)take(c) << (8 * i);
  return v;
}

static void wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void wr64(uint8_t *p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }

/* --- mode 0 --------------------------------------------------------------------------------- */

static void fuzz_decode(const uint8_t *data, size_t size) {
  bcir_shm_view m;
  bcir_shm_entry e;
  uint32_t expect = 0;
  if (bcir_shm_decode(data, size, &m) != BCIR_OK) {
    if (m.data != NULL || m.n_shards != 0u) abort(); /* a refusal leaves the view zeroed */
    return;
  }
  for (uint32_t i = 0; i < m.n_shards; i++) {
    if (bcir_shm_entry_at(&m, i, &e) != BCIR_OK || e.seg_begin != expect || e.seg_end < e.seg_begin)
      abort();
    expect = e.seg_end;
  }
  if (expect != m.n_segments || bcir_shm_entry_at(&m, m.n_shards, &e) == BCIR_OK) abort();
}

/* --- mode 1 --------------------------------------------------------------------------------- */

static uint8_t frame_buf[BLOB_MAX], shard_buf[SHARDS][BLOB_MAX], manifest_buf[BCIR_SHM_FIXED + 48u * SHARDS];
static uint8_t whole_out[BLOB_MAX + 64u];
static size_t frame_len, shard_len[SHARDS], manifest_len;

typedef struct store {
  const uint8_t *blob[SHARDS + 1u];
  size_t len[SHARDS + 1u];
  uint32_t n;
} store;

static int fetch(const uint8_t digest[32], const uint8_t **data, size_t *len, void *ctx) {
  store *s = (store *)ctx;
  uint8_t d[32];
  for (uint32_t i = 0; i < s->n; i++) {
    bcir_sha256_digest(s->blob[i], s->len[i], d);
    if (memcmp(d, digest, 32) == 0) {
      *data = s->blob[i];
      *len = s->len[i];
      return 0;
    }
  }
  return 1;
}

static void reseal_blob(uint8_t *blob, size_t len) { /* a pack's CRC trailer */
  if (len >= 4u) wr32(blob + len - 4u, bcir_crc32(blob, len - 4u));
}

static void reseal_manifest(uint32_t entry, const uint8_t *blob, size_t len) {
  uint8_t *e = manifest_buf + BCIR_SHM_HEADER_SIZE + BCIR_SHM_DIGESTS_SIZE + 48u * entry;
  wr64(e + 8, (uint64_t)len);
  bcir_sha256_digest(blob, len, e + 16);
  wr32(manifest_buf + manifest_len - 4u, bcir_crc32(manifest_buf, manifest_len - 4u));
}

static void fuzz_shards(cursor *c) {
  uint32_t world = (take(c) % SHARDS) + 1u, target = take(c), flags = take(c);
  uint32_t pos = take32(c);
  uint8_t flip = (uint8_t)(take(c) | 1u);
  const uint8_t *whole = c->data + c->at;
  size_t whole_len = c->size - c->at, got = 0;
  uint32_t begins[SHARDS], ends[SHARDS], n = 0;
  const uint8_t *ptrs[SHARDS];
  store st;
  if (whole_len > BLOB_MAX || bcir_shm_check_whole(whole, whole_len) != BCIR_OK) return;
  {
    bcir_streampack_header hdr;
    if (bcir_sp_validate(whole, whole_len, &hdr) != BCIR_OK) abort(); /* check_whole passed */
    if (bcir_shm_partition(hdr.n_segments, world, begins, ends, SHARDS, &n) != BCIR_OK) return;
  }
  if (bcir_shm_frame(whole, whole_len, frame_buf, sizeof frame_buf, &frame_len) != BCIR_OK) abort();
  for (uint32_t i = 0; i < n; i++) {
    if (bcir_shm_sub_pack(whole, whole_len, begins[i], ends[i], shard_buf[i], BLOB_MAX, &shard_len[i]) !=
        BCIR_OK)
      abort(); /* a sub-pack is never larger than its whole */
    ptrs[i] = shard_buf[i];
  }
  if (bcir_shm_encode(whole, whole_len, frame_buf, frame_len, begins, ends, ptrs, shard_len, n,
                      manifest_buf, sizeof manifest_buf, &manifest_len) != BCIR_OK)
    abort();
  st.n = n + 1u;
  st.blob[0] = frame_buf;
  st.len[0] = frame_len;
  for (uint32_t i = 0; i < n; i++) { st.blob[i + 1u] = shard_buf[i]; st.len[i + 1u] = shard_len[i]; }
  /* the honest set reassembles to the whole, exactly */
  if (bcir_shm_reassemble(manifest_buf, manifest_len, fetch, &st, whole_out, sizeof whole_out, &got) !=
          BCIR_OK ||
      got != whole_len || memcmp(whole_out, whole, whole_len) != 0)
    abort();
  /* tamper: the manifest (target 0), the frame (1) or a shard (2..) */
  target %= n + 2u;
  if (target == 0u) {
    manifest_buf[pos % manifest_len] ^= flip;
    if (flags & 1u) wr32(manifest_buf + manifest_len - 4u, bcir_crc32(manifest_buf, manifest_len - 4u));
  } else {
    uint8_t *blob = target == 1u ? frame_buf : shard_buf[target - 2u];
    size_t len = target == 1u ? frame_len : shard_len[target - 2u];
    blob[pos % len] ^= flip;
    if (flags & 1u) reseal_blob(blob, len);
    if (flags & 2u) {
      if (target == 1u) {
        uint8_t *m = manifest_buf + 96; /* frame_sha256 */
        bcir_sha256_digest(blob, len, m);
        wr32(manifest_buf + manifest_len - 4u, bcir_crc32(manifest_buf, manifest_len - 4u));
      } else {
        reseal_manifest(target - 2u, blob, len);
      }
    }
  }
  got = 0;
  if (bcir_shm_reassemble(manifest_buf, manifest_len, fetch, &st, whole_out, sizeof whole_out, &got) ==
      BCIR_OK) {
    /* whatever passed every law must be the declared whole -- and that is the input's whole */
    if (got != whole_len || memcmp(whole_out, whole, whole_len) != 0) abort();
  } else if (got != 0u) {
    abort(); /* a refusal returns no length */
  }
}

/* --- mode 2 --------------------------------------------------------------------------------- */

#define SLOTS 4u
#define CAP 256u
static bcir_ho_slot slots[SLOTS];
static uint8_t arena[SLOTS * CAP];

typedef struct held {
  bcir_ho_view view;
  uint8_t copy[CAP];
  int live;
} held;

static void check_table(const bcir_ho_table *t, const held *views, size_t n_views,
                        const uint32_t *epochs) {
  for (uint32_t s = 0; s < SLOTS; s++) {
    uint32_t pins = 0;
    for (size_t v = 0; v < n_views; v++)
      if (views[v].live && views[v].view.index == s) pins++;
    const bcir_ho_slot *sl = &t->slots[s];
    if (sl->pins != pins) abort();                              /* exact pin accounting */
    if ((sl->state == BCIR_HO_FREE || sl->state == BCIR_HO_BUILDING) && sl->pins) abort();
    if (sl->state == BCIR_HO_RETIRED && sl->pins == 0u) abort();  /* retired = draining */
    if (sl->epoch < epochs[s]) abort();                          /* epochs never go back */
    if (sl->length > CAP) abort();
  }
  for (size_t v = 0; v < n_views; v++)                           /* borrowed bytes never change */
    if (views[v].live && memcmp(views[v].view.data, views[v].copy, views[v].view.len) != 0) abort();
}

static void fuzz_table(cursor *c) {
  bcir_ho_table t;
  bcir_ho_handle regs[4];
  held views[16];
  uint32_t epochs[SLOTS];
  if (bcir_ho_init(&t, slots, SLOTS, arena, sizeof arena, CAP) != BCIR_OK) abort();
  memset(regs, 0, sizeof regs);
  memset(views, 0, sizeof views);
  for (uint32_t s = 0; s < SLOTS; s++) epochs[s] = t.slots[s].epoch;
  for (int step = 0; step < 96 && c->at < c->size; step++) {
    uint8_t op = take(c), r = take(c) & 3u, v = take(c) & 15u;
    bcir_ho_outcome o;
    switch (op % 8u) {
      case 0:
        o = bcir_ho_reserve(&t);
        if (o.verdict == BCIR_HO_APPLIED) {
          if (t.slots[o.handle.index].state != BCIR_HO_BUILDING) abort();
          regs[r] = o.handle;
        }
        break;
      case 1: { /* the producer writes its reservation */
        uint8_t *p;
        size_t cap;
        if (bcir_ho_reserved(&t, regs[r], &p, &cap) == BCIR_OK) {
          size_t n = take(c) % (cap + 1u);
          for (size_t i = 0; i < n; i++) p[i] = take(c);
        }
        break;
      }
      case 2: (void)bcir_ho_commit(&t, regs[r], take(c) % (CAP + 2u)); break;
      case 3: (void)bcir_ho_abort(&t, regs[r]); break;
      case 4:
        if (!views[v].live) {
          bcir_ho_handle h = regs[r];
          int frozen = h.index < SLOTS && t.slots[h.index].state == BCIR_HO_FROZEN &&
                       t.slots[h.index].epoch == h.epoch;
          o = bcir_ho_borrow(&t, h, &views[v].view);
          if ((o.verdict == BCIR_HO_APPLIED) != frozen) abort(); /* borrow iff frozen at epoch */
          if (o.verdict == BCIR_HO_APPLIED) {
            views[v].live = 1;
            memcpy(views[v].copy, views[v].view.data, views[v].view.len);
          }
        }
        break;
      case 5:
        if (views[v].live) {
          if (bcir_ho_give_back(&t, &views[v].view).verdict != BCIR_HO_APPLIED) abort();
          views[v].live = 0;
        }
        break;
      case 6: (void)bcir_ho_release(&t, regs[r]); break;
      default: /* a forged handle: any index, any epoch */
        regs[r].index = take(c) % (SLOTS + 2u);
        regs[r].epoch = take(c) % 4u;
        break;
    }
    check_table(&t, views, 16, epochs);
    for (uint32_t s = 0; s < SLOTS; s++) epochs[s] = t.slots[s].epoch;
  }
}

/* --- mode 3 --------------------------------------------------------------------------------- */

static uint8_t freeze_out[4096];

static void fuzz_freeze(cursor *c) {
  static const char *labels[] = {"c.add", "c.load", "c.store", "", "x\tbad", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"};
  bcir_generation_view gens[8];
  bcir_claim claims[48];
  bcir_plan_step steps[48];
  bcir_func f;
  bcir_plan plan;
  size_t n_gens = take(c) % 9u, n_claims = take(c) % 49u, len = 1;
  uint32_t topo = take(c) % 3u, rid = 0;
  for (size_t i = 0; i < n_gens; i++) {
    uint8_t step = take(c);
    rid = (step & 0x80u) ? rid : rid + 1u + (step & 7u); /* mostly ascending, sometimes repeated */
    gens[i].rid = rid;
    gens[i].map_gen = take(c) % 4u;
    gens[i].data_gen = take(c) % 4u;
  }
  memset(claims, 0, sizeof claims);
  for (size_t i = 0; i < n_claims; i++) {
    bcir_claim *cl = &claims[i];
    cl->id = (uint32_t)(i + 1u) + (take(c) == 0u ? 0u : (uint32_t)i); /* rarely non-ascending */
    cl->opcode = (bcir_opcode)(take(c) % 18u);
    cl->lane = take(c) % 7u;
    cl->domain = (bcir_domain)(take(c) % 6u);
    cl->count = take(c) == 255u ? 0xFFFFFFFFu : take(c);
    cl->n_rd = take(c) % 8u;
    for (uint8_t k = 0; k < cl->n_rd && k < BCIR_CLAIM_MAX_RD; k++)
      cl->rd[k] = n_gens ? gens[take(c) % n_gens].rid + (take(c) == 7u ? 1000u : 0u) : 1u;
    cl->n_wr = take(c) % 4u;
    for (uint8_t k = 0; k < cl->n_wr && k < BCIR_CLAIM_MAX_WR; k++)
      cl->wr[k] = n_gens ? gens[take(c) % n_gens].rid : 1u;
    const char *label = labels[take(c) % 6u];
    memcpy(cl->op, label, strlen(label) + 1u);
  }
  memset(&f, 0, sizeof f);
  f.claims = claims;
  f.n_claims = n_claims;
  memset(freeze_out, 0xA5, sizeof freeze_out);
  bcir_status st = bcir_plan_func(&f, steps, 48u, &plan);
  int hydrated = st == BCIR_OK; /* a plan the planner refused never reaches the hydrator */
  if (hydrated)
    st = bcir_hydrate_generations(&f, &plan, topo, n_gens ? gens : NULL, n_gens, freeze_out,
                                  take(c) == 0u ? 64u : sizeof freeze_out, &len);
  if (st != BCIR_OK) {
    if (hydrated && len != 0u) abort(); /* the hydrator zeroes the length it refuses */
    for (size_t i = 0; i < sizeof freeze_out; i++)
      if (freeze_out[i] != 0xA5u) abort(); /* a refused freeze writes nothing */
    return;
  }
  {
    uint8_t want[32], got[32];
    bcir_streampack_header hdr;
    if (bcir_sp_verify_semantic(freeze_out, len, 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_OK) abort();
    if (bcir_sp_validate(freeze_out, len, &hdr) != BCIR_OK || hdr.version != 4u ||
        hdr.n_gens != n_gens || hdr.topo_gen != topo)
      abort();
    if (bcir_ctl_registry_digest(gens, n_gens, want) != BCIR_OK ||
        bcir_ctl_pack_registry_digest(freeze_out, len, got) != BCIR_OK || memcmp(want, got, 32) != 0)
      abort();
    if (bcir_shm_check_whole(freeze_out, len) != BCIR_OK) abort(); /* every frozen step shards */
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  cursor c = {data, size, 1u};
  if (size == 0u) return 0;
  switch (data[0] % 4u) {
    case 0: fuzz_decode(data + 1, size - 1u); break;
    case 1: fuzz_shards(&c); break;
    case 2: fuzz_table(&c); break;
    default: fuzz_freeze(&c); break;
  }
  return 0;
}
