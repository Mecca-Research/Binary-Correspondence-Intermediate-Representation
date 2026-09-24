/*===- bcir_handoff.c - the data-plane hand-off: a pack table with explicit lifetime ===
 *
 * See bcir_handoff.h. The mirror of bcir/gem/handoff.py::PackTable decision for decision.
 *===----------------------------------------------------------------------===*/
#include "bcir_handoff.h"

#include "bcir_shard_manifest.h"

static const uint8_t STATE_TAG[] = "BHOF/state/v0";  /* sizeof counts the NUL: the tag's 0x00 */
static const uint8_t ZERO32[32] = {0};

static void wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void wr64(uint8_t *p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }
static void copy32(uint8_t *d, const uint8_t *s) {
  for (size_t i = 0; i < 32u; i++) d[i] = s[i];
}
static int eq32(const uint8_t *a, const uint8_t *b) {
  uint8_t diff = 0;
  for (size_t i = 0; i < 32u; i++) diff = (uint8_t)(diff | (a[i] ^ b[i]));
  return diff == 0u;
}

static bcir_ho_outcome made(uint8_t verdict, uint8_t refusal, bcir_status status,
                            uint32_t generation, bcir_ho_handle handle) {
  bcir_ho_outcome o;
  o.verdict = verdict;
  o.refusal = refusal;
  o.reserved = 0;
  o.generation = generation;
  o.handle = handle;
  o.status = status;
  return o;
}
static const bcir_ho_handle NO_HANDLE = {0u, 0u};
static bcir_ho_outcome applied(uint32_t generation, bcir_ho_handle handle) {
  return made(BCIR_HO_APPLIED, BCIR_HO_REFUSAL_NONE, BCIR_OK, generation, handle);
}
static bcir_ho_outcome refused(uint8_t refusal, bcir_status status, uint32_t generation) {
  return made(BCIR_HO_REFUSED, refusal, status, generation, NO_HANDLE);
}
static bcir_ho_outcome lifetime(void) {
  return refused(BCIR_HO_REFUSAL_LIFETIME, BCIR_ERR_LIFETIME, 0u);
}

/* The slot the handle names, when it is in `state` at the handle's epoch. */
static bcir_ho_slot *live(const bcir_ho_table *t, bcir_ho_handle h, uint8_t state) {
  bcir_ho_slot *s;
  if (!t || !t->slots || h.index >= t->n_slots) return (bcir_ho_slot *)0;
  s = &t->slots[h.index];
  return s->state == state && s->epoch == h.epoch ? s : (bcir_ho_slot *)0;
}

bcir_status bcir_ho_init(bcir_ho_table *BCIR_RESTRICT table, bcir_ho_slot *BCIR_RESTRICT slots,
                         uint32_t n_slots, uint8_t *BCIR_RESTRICT arena, size_t arena_len,
                         size_t slot_capacity) {
  if (!table) return BCIR_ERR_NOSPACE;
  table->slots = (bcir_ho_slot *)0;
  table->n_slots = 0;
  table->capacity = 0;
  table->arena = (uint8_t *)0;
  if (!slots || !arena || n_slots == 0u || n_slots > BCIR_HO_SLOTS_MAX || slot_capacity == 0u ||
      (uint64_t)slot_capacity > 0xFFFFFFFFu || slot_capacity > SIZE_MAX / n_slots ||
      arena_len < slot_capacity * n_slots)
    return BCIR_ERR_NOSPACE;
  for (uint32_t i = 0; i < n_slots; i++) {
    bcir_ho_slot *s = &slots[i];
    s->epoch = BCIR_HO_EPOCH_FIRST;
    s->state = BCIR_HO_FREE;
    s->admitted = 0;
    s->reserved = 0;
    s->pins = 0;
    s->admitted_generation = 0;
    copy32(s->admitted_registry, ZERO32);
    s->length = 0;
  }
  table->slots = slots;
  table->n_slots = n_slots;
  table->capacity = slot_capacity;
  table->arena = arena;
  return BCIR_OK;
}

bcir_ho_outcome bcir_ho_reserve(bcir_ho_table *table) {
  if (!table || !table->slots) return refused(BCIR_HO_REFUSAL_FULL, BCIR_ERR_FULL, 0u);
  for (uint32_t i = 0; i < table->n_slots; i++) {
    bcir_ho_slot *s = &table->slots[i];
    if (s->state == BCIR_HO_FREE) {
      bcir_ho_handle h;
      s->state = BCIR_HO_BUILDING;
      s->length = 0;
      h.index = i;
      h.epoch = s->epoch;
      return applied(0u, h);
    }
  }
  return refused(BCIR_HO_REFUSAL_FULL, BCIR_ERR_FULL, 0u);
}

bcir_status bcir_ho_reserved(const bcir_ho_table *BCIR_RESTRICT table, bcir_ho_handle handle,
                             uint8_t **BCIR_RESTRICT data, size_t *BCIR_RESTRICT capacity) {
  if (data) *data = (uint8_t *)0;
  if (capacity) *capacity = 0;
  if (!data || !capacity || !live(table, handle, BCIR_HO_BUILDING)) return BCIR_ERR_LIFETIME;
  *data = table->arena + (size_t)handle.index * table->capacity;
  *capacity = table->capacity;
  return BCIR_OK;
}

bcir_ho_outcome bcir_ho_commit(bcir_ho_table *table, bcir_ho_handle handle, size_t length) {
  bcir_ho_slot *s = live(table, handle, BCIR_HO_BUILDING);
  if (!s) return lifetime();
  if (length == 0u || length > table->capacity)
    return refused(BCIR_HO_REFUSAL_CAPACITY, BCIR_ERR_NOSPACE, 0u);
  s->state = BCIR_HO_FROZEN;
  s->length = length;
  s->admitted = 0;
  return applied(0u, handle);
}

/* The end of an incarnation: every handle and view of it dies here. */
static bcir_ho_outcome end(bcir_ho_slot *s) {
  s->admitted = 0;
  s->admitted_generation = 0;
  copy32(s->admitted_registry, ZERO32);
  if (s->epoch == BCIR_HO_EPOCH_MAX) {  /* an epoch never wraps: out of service for good */
    s->state = BCIR_HO_EXHAUSTED;
  } else {
    s->epoch++;
    s->state = s->pins == 0u ? BCIR_HO_FREE : BCIR_HO_RETIRED;
  }
  if (s->pins == 0u) s->length = 0;
  return applied(0u, NO_HANDLE);
}

bcir_ho_outcome bcir_ho_abort(bcir_ho_table *table, bcir_ho_handle handle) {
  bcir_ho_slot *s = live(table, handle, BCIR_HO_BUILDING);
  return s ? end(s) : lifetime();
}

bcir_ho_outcome bcir_ho_borrow(bcir_ho_table *BCIR_RESTRICT table, bcir_ho_handle handle,
                               bcir_ho_view *BCIR_RESTRICT view) {
  bcir_ho_slot *s = live(table, handle, BCIR_HO_FROZEN);
  if (view) {
    view->index = 0;
    view->epoch = 0;
    view->data = (const uint8_t *)0;
    view->len = 0;
  }
  if (!s || !view) return lifetime();
  if (s->pins == BCIR_HO_PINS_MAX) return refused(BCIR_HO_REFUSAL_EXHAUSTED, BCIR_ERR_OVERFLOW, 0u);
  s->pins++;
  view->index = handle.index;
  view->epoch = handle.epoch;
  view->data = table->arena + (size_t)handle.index * table->capacity;
  view->len = s->length;
  return applied(0u, handle);
}

bcir_ho_outcome bcir_ho_give_back(bcir_ho_table *BCIR_RESTRICT table,
                                  const bcir_ho_view *BCIR_RESTRICT view) {
  bcir_ho_slot *s;
  int current, retired, exhausted;
  /* epoch 0 is never issued: a view a refused borrow zeroed names nothing, and must not match a
   * slot retired from epoch 1 (it would return another borrower's pin) */
  if (!table || !table->slots || !view || view->epoch == 0u || view->index >= table->n_slots)
    return lifetime();
  s = &table->slots[view->index];
  current = s->state == BCIR_HO_FROZEN && s->epoch == view->epoch;
  retired = s->state == BCIR_HO_RETIRED && view->epoch != BCIR_HO_EPOCH_MAX &&
            s->epoch == view->epoch + 1u;
  exhausted = s->state == BCIR_HO_EXHAUSTED && s->epoch == view->epoch;
  if (s->pins == 0u || !(current || retired || exhausted)) return lifetime();
  s->pins--;
  if (s->pins == 0u && (s->state == BCIR_HO_RETIRED || s->state == BCIR_HO_EXHAUSTED)) {
    if (s->state == BCIR_HO_RETIRED) s->state = BCIR_HO_FREE;
    s->length = 0;
  }
  return applied(0u, NO_HANDLE);
}

bcir_ho_outcome bcir_ho_release(bcir_ho_table *table, bcir_ho_handle handle) {
  bcir_ho_slot *s = live(table, handle, BCIR_HO_FROZEN);
  return s ? end(s) : lifetime();
}

bcir_ho_outcome bcir_ho_admit(bcir_ho_table *BCIR_RESTRICT table, bcir_ho_handle handle,
                              bcir_ctl_state *BCIR_RESTRICT plane) {
  bcir_ho_view v;
  bcir_ho_outcome o = bcir_ho_borrow(table, handle, &v);
  bcir_ctl_outcome c;
  bcir_ho_slot *s;
  if (o.verdict != BCIR_HO_APPLIED) return o;
  if (!plane) {
    (void)bcir_ho_give_back(table, &v);
    return refused(BCIR_HO_REFUSAL_STALE, BCIR_ERR_STALE, 0u);  /* no plane: nothing proves it current */
  }
  c = bcir_ctl_admit_pack(plane, v.data, v.len);
  (void)bcir_ho_give_back(table, &v);
  s = &table->slots[handle.index];
  if (c.verdict == BCIR_CTL_APPLIED) {
    s->admitted = 1;
    s->admitted_generation = plane->generation;
    copy32(s->admitted_registry, plane->reg_digest);
    return applied(plane->generation, handle);
  }
  s->admitted = 0;
  s->admitted_generation = 0;
  copy32(s->admitted_registry, ZERO32);
  if (c.refusal == BCIR_CTL_REFUSAL_STALE)
    return refused(BCIR_HO_REFUSAL_STALE, BCIR_ERR_STALE, plane->generation);
  return refused(BCIR_HO_REFUSAL_MALFORMED, c.status, plane->generation);
}

bcir_ho_outcome bcir_ho_dispatch(bcir_ho_table *BCIR_RESTRICT table, bcir_ho_handle handle,
                                 bcir_ctl_state *BCIR_RESTRICT plane, bcir_seg_fn fn, void *ctx) {
  bcir_ho_view v;
  bcir_ho_outcome o = bcir_ho_borrow(table, handle, &v);
  bcir_ctl_outcome entered;
  bcir_ho_slot *s;
  bcir_status st;
  uint32_t gen;
  if (o.verdict != BCIR_HO_APPLIED) return o;
  s = &table->slots[handle.index];
  gen = plane ? plane->generation : 0u;
  if (!s->admitted || !plane) {
    (void)bcir_ho_give_back(table, &v);
    return refused(BCIR_HO_REFUSAL_UNADMITTED, BCIR_ERR_STALE, gen);
  }
  /* an admission lives within one resident generation of one registry */
  if (s->admitted_generation != gen ||
      !eq32(s->admitted_registry, plane->has_registry ? plane->reg_digest : ZERO32)) {
    (void)bcir_ho_give_back(table, &v);
    return refused(BCIR_HO_REFUSAL_STALE, BCIR_ERR_STALE, gen);
  }
  entered = bcir_ctl_enter(plane);
  if (entered.verdict == BCIR_CTL_REFUSED) {
    (void)bcir_ho_give_back(table, &v);
    if (entered.refusal == BCIR_CTL_REFUSAL_DRAINING)
      return refused(BCIR_HO_REFUSAL_DRAINING, BCIR_ERR_CONTROL, plane->generation);
    return refused(BCIR_HO_REFUSAL_EXHAUSTED, BCIR_ERR_OVERFLOW, plane->generation);
  }
  st = bcir_sp_for_each_segment(v.data, v.len, fn, ctx);
  (void)bcir_ctl_leave(plane);  /* the boundary: a switch requested mid-dispatch lands here */
  (void)bcir_ho_give_back(table, &v);
  if (st != BCIR_OK) return refused(BCIR_HO_REFUSAL_WALK, st, plane->generation);
  return applied(plane->generation, handle);
}

bcir_ho_outcome bcir_ho_admit_manifest(const bcir_ctl_state *BCIR_RESTRICT plane,
                                       const uint8_t *BCIR_RESTRICT manifest, size_t len) {
  bcir_shm_view m;
  bcir_status st = bcir_shm_decode(manifest, len, &m);
  uint32_t gen = plane ? plane->generation : 0u;
  if (st != BCIR_OK) return refused(BCIR_HO_REFUSAL_MALFORMED, st, gen);
  if (!plane || !plane->has_registry || m.map_gen != plane->reg_map_gen ||
      m.data_gen != plane->reg_data_gen || m.topo_gen != plane->reg_topo_gen ||
      !eq32(m.registry_digest, plane->reg_digest))
    return refused(BCIR_HO_REFUSAL_STALE, BCIR_ERR_STALE, gen);
  return applied(gen, NO_HANDLE);
}

void bcir_ho_state_digest(const bcir_ho_table *BCIR_RESTRICT table, uint8_t out[32]) {
  bcir_sha256 h;
  uint8_t b[22];
  uint8_t held_digest[32];
  bcir_sha256_init(&h);
  bcir_sha256_update(&h, STATE_TAG, sizeof(STATE_TAG));
  wr32(b, table ? table->n_slots : 0u);
  wr64(b + 4, table ? (uint64_t)table->capacity : 0u);
  bcir_sha256_update(&h, b, 12u);
  for (uint32_t i = 0; table && table->slots && i < table->n_slots; i++) {
    const bcir_ho_slot *s = &table->slots[i];
    int held = s->state == BCIR_HO_FROZEN || s->state == BCIR_HO_RETIRED ||
               (s->state == BCIR_HO_EXHAUSTED && s->pins > 0u);
    wr32(b, s->epoch);
    b[4] = s->state;
    b[5] = s->admitted;
    wr32(b + 6, s->pins);
    wr32(b + 10, s->admitted_generation);
    wr64(b + 14, held ? (uint64_t)s->length : 0u);
    bcir_sha256_update(&h, b, sizeof b);
    bcir_sha256_update(&h, s->admitted_registry, 32u);
    if (held) {
      bcir_sha256_digest(table->arena + (size_t)i * table->capacity, s->length, held_digest);
      bcir_sha256_update(&h, held_digest, 32u);
    } else {
      bcir_sha256_update(&h, ZERO32, 32u);
    }
  }
  bcir_sha256_final(&h, out);
}
