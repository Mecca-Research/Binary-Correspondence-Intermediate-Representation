/*===- bcir_ring.c - BCIR live SPSC ring, version zero ----------------------===
 *
 * The C twin of bcir/gem/ring.py: the same algorithm over the same bytes, operation for
 * operation (see bcir_ring.h for the region and the contract, docs/kernel/BCIR_LIVE_RING_ABI.md
 * for the laws). Memory ordering, in the terms of the C11 model:
 *
 *   publication   the producer marks the slot odd (relaxed) and issues a release fence, writes
 *                 pos, epoch/length and the payload words (relaxed), marks the slot even with a
 *                 release store and releases the head. The consumer loads the slot's sequence
 *                 with acquire, the words relaxed, issues an acquire fence and reloads the
 *                 sequence: a copy that raced a writer sees a different sequence (Boehm's
 *                 seqlock), so a torn record is never delivered -- in OVERWRITE it is counted
 *                 LOST, in BACKPRESSURE (where no writer may touch an unconsumed slot) it is a
 *                 protocol violation.
 *   accounting    the consumer writes the inactive copy of its accounting after a release fence
 *                 and flips c_commit with a release store; a reader loads c_commit (acquire),
 *                 the copy, fences (acquire) and reloads c_commit -- the same pattern, so a
 *                 snapshot is never torn, and a consumer that dies mid-update leaves the active
 *                 copy intact.
 *   ownership     an attach claims the owner word with a compare-and-swap (acq_rel), fences,
 *                 writes the epoch origin and releases the owner word; the consumer reads
 *                 (owner, origin, owner) around an acquire fence.
 *
 * Every payload word is copied with relaxed atomic loads and stores, so the OVERWRITE race a
 * seqlock tolerates is not a data race in the C11 sense (and ThreadSanitizer, which models the
 * atomics, reports none).
 *===----------------------------------------------------------------------===*/
#include "bcir_ring.h"

#include <stdatomic.h>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "the version-zero ring's shared words are little-endian: a big-endian host is outside it"
#endif

_Static_assert(sizeof(unsigned long long) == 8, "a ring word is 64 bits");
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2,
               "the ring's shared words need lock-free (address-free) 64-bit atomics");

typedef _Atomic unsigned long long ring_word;

#define HALF (UINT64_C(1) << 63)
#define M32 UINT64_C(0xFFFFFFFF)

static ring_word *word_at(uint8_t *region, size_t off) {
  return (ring_word *)(void *)(region + off);
}

static uint64_t ld_acq(uint8_t *r, size_t off) {
  return (uint64_t)atomic_load_explicit(word_at(r, off), memory_order_acquire);
}

static uint64_t ld_rlx(uint8_t *r, size_t off) {
  return (uint64_t)atomic_load_explicit(word_at(r, off), memory_order_relaxed);
}

static void st_rel(uint8_t *r, size_t off, uint64_t v) {
  atomic_store_explicit(word_at(r, off), (unsigned long long)v, memory_order_release);
}

static void st_rlx(uint8_t *r, size_t off, uint64_t v) {
  atomic_store_explicit(word_at(r, off), (unsigned long long)v, memory_order_relaxed);
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)(p[1] << 8)); }

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }

static void wr16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}

static void wr32(uint8_t *p, uint32_t v) {
  for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static void wr64(uint8_t *p, uint64_t v) {
  for (unsigned i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static uint64_t owner_word(uint32_t epoch, uint32_t state) {
  return (uint64_t)epoch << 32 | state;
}

static int aligned(const uint8_t *region) { return ((uintptr_t)(const void *)region & 7u) == 0; }

static bcir_ring_outcome outcome(bcir_ring_verdict verdict, bcir_status status, uint64_t position,
                                 uint64_t count, uint32_t epoch) {
  bcir_ring_outcome o;
  o.verdict = verdict;
  o.status = status;
  o.position = position;
  o.count = count;
  o.epoch = epoch;
  o.length = 0;
  return o;
}

static bcir_ring_outcome refused(bcir_status status, uint64_t position) {
  return outcome(BCIR_RING_REFUSED, status, position, 0, 0);
}

bcir_status bcir_ring_geometry_validate(const bcir_ring_geometry *g) {
  if (!g) return BCIR_ERR_RING;
  if (g->policy != BCIR_RING_BACKPRESSURE && g->policy != BCIR_RING_OVERWRITE) return BCIR_ERR_RING;
  if (g->payload != BCIR_RING_TELEMETRY && g->payload != BCIR_RING_CONTROL) return BCIR_ERR_RING;
  if (g->slot_size < BCIR_RING_SLOT_SIZE_MIN || g->slot_size > BCIR_RING_SLOT_SIZE_MAX ||
      g->slot_size % 64u != 0u)
    return BCIR_ERR_RING;
  if (g->slot_count < BCIR_RING_SLOTS_MIN || g->slot_count > BCIR_RING_SLOTS_MAX ||
      (g->slot_count & (g->slot_count - 1u)) != 0u)
    return BCIR_ERR_RING;
  if (g->ring_id == 0u) return BCIR_ERR_RING;
  if (g->payload == BCIR_RING_CONTROL && g->policy != BCIR_RING_BACKPRESSURE) return BCIR_ERR_RING;
  if (g->payload == BCIR_RING_CONTROL && g->slot_size < BCIR_RING_CONTROL_SLOT_MIN) return BCIR_ERR_RING;
  return BCIR_OK;
}

uint64_t bcir_ring_region_size(const bcir_ring_geometry *g) {
  if (bcir_ring_geometry_validate(g) != BCIR_OK) return 0u;
  return (uint64_t)BCIR_RING_HEADER_SIZE + (uint64_t)g->slot_count * g->slot_size;
}

static void encode_geometry(uint8_t *line, const bcir_ring_geometry *g, uint64_t region_size) {
  for (unsigned i = 0; i < BCIR_RING_GEOMETRY_SIZE; i++) line[i] = 0;
  for (unsigned i = 0; i < 4; i++) line[i] = (uint8_t)BCIR_RING_MAGIC[i];
  wr16(line + 4, (uint16_t)BCIR_RING_VERSION);
  line[6] = g->policy;
  line[7] = g->payload;
  wr32(line + 8, g->slot_size);
  wr32(line + 12, g->slot_count);
  wr64(line + 16, g->ring_id);
  wr64(line + 24, region_size);
  wr64(line + 32, g->origin);
  wr32(line + 60, bcir_crc32(line, 60));
}

bcir_status bcir_ring_format(uint8_t *region, size_t len, const bcir_ring_geometry *g) {
  bcir_status st = bcir_ring_geometry_validate(g);
  if (st != BCIR_OK) return st;
  if (!region || !aligned(region)) return BCIR_ERR_RING;
  uint64_t size = bcir_ring_region_size(g);
  if ((uint64_t)len < size) return BCIR_ERR_NOSPACE;
  for (uint64_t i = 0; i < size; i++) region[i] = 0;
  encode_geometry(region, g, size);
  st_rlx(region, BCIR_RING_OFF_HEAD, g->origin);
  st_rlx(region, BCIR_RING_OFF_ORIGIN, g->origin);
  st_rlx(region, BCIR_RING_OFF_TAIL, g->origin);
  st_rlx(region, BCIR_RING_OFF_ACCT0, g->origin);
  atomic_thread_fence(memory_order_release);
  return BCIR_OK;
}

bcir_status bcir_ring_decode_geometry(const uint8_t *region, size_t len, bcir_ring_geometry *out) {
  if (!region || len < BCIR_RING_HEADER_SIZE) return BCIR_ERR_TRUNCATED;
  for (unsigned i = 0; i < 4; i++)
    if (region[i] != (uint8_t)BCIR_RING_MAGIC[i]) return BCIR_ERR_MAGIC;
  if (rd16(region + 4) != BCIR_RING_VERSION) return BCIR_ERR_VERSION;
  if (rd32(region + 60) != bcir_crc32(region, 60)) return BCIR_ERR_CRC;
  for (unsigned i = 40; i < 60; i++)
    if (region[i] != 0u) return BCIR_ERR_RESERVED;
  bcir_ring_geometry g;
  g.policy = region[6];
  g.payload = region[7];
  if (g.policy != BCIR_RING_BACKPRESSURE && g.policy != BCIR_RING_OVERWRITE) return BCIR_ERR_RING;
  if (g.payload != BCIR_RING_TELEMETRY && g.payload != BCIR_RING_CONTROL) return BCIR_ERR_RING;
  g.slot_size = rd32(region + 8);
  g.slot_count = rd32(region + 12);
  g.ring_id = rd64(region + 16);
  g.origin = rd64(region + 32);
  bcir_status st = bcir_ring_geometry_validate(&g);
  if (st != BCIR_OK) return st;
  g.region_size = bcir_ring_region_size(&g);
  if (rd64(region + 24) != g.region_size) return BCIR_ERR_RING;
  if (g.region_size > (uint64_t)len) return BCIR_ERR_TRUNCATED;
  if (out) *out = g;
  return BCIR_OK;
}

bcir_status bcir_ring_accounting_of(const uint8_t *region, size_t len, bcir_ring_accounting *out) {
  bcir_ring_geometry g;
  bcir_status st = bcir_ring_decode_geometry(region, len, &g);
  if (st != BCIR_OK) return st;
  if (!aligned(region) || !out) return BCIR_ERR_RING;
  /* The atomics are loads; the pointer is shared memory the caller may not write. */
  uint8_t *r = (uint8_t *)(uintptr_t)region;
  for (unsigned tries = 0; tries < 64u; tries++) {
    uint64_t c1 = ld_acq(r, BCIR_RING_OFF_C_COMMIT);
    size_t base = (c1 & 1u) ? BCIR_RING_OFF_ACCT1 : BCIR_RING_OFF_ACCT0;
    uint64_t tail = ld_rlx(r, base), lost = ld_rlx(r, base + 8), stale = ld_rlx(r, base + 16);
    uint64_t word = ld_rlx(r, base + 24);
    atomic_thread_fence(memory_order_acquire);
    if (ld_rlx(r, BCIR_RING_OFF_C_COMMIT) != c1) continue;
    out->tail = tail;
    out->lost = lost;
    out->stale = stale;
    out->last_epoch = (uint32_t)(word & M32);
    out->delivered = tail - g.origin - lost - stale;
    return BCIR_OK;
  }
  return BCIR_ERR_BUSY;
}

static size_t slot_base(const bcir_ring_geometry *g, uint64_t position) {
  return (size_t)BCIR_RING_HEADER_SIZE +
         (size_t)(position & (uint64_t)(g->slot_count - 1u)) * (size_t)g->slot_size;
}

/* ---- the producer ---------------------------------------------------------------------- */

void bcir_ring_producer_init(bcir_ring_producer *p, uint8_t *region, size_t len) {
  if (!p) return;
  bcir_ring_producer zero = {0};
  *p = zero;
  p->region = region;
  p->region_len = len;
}

static int producer_held(const bcir_ring_producer *p) {
  return p->epoch != 0u &&
         ld_acq(p->region, BCIR_RING_OFF_P_OWNER) == owner_word(p->epoch, BCIR_RING_ATTACHED);
}

/* The consumer's published progress: one word on its own line, released after each commit, so
 * it never runs ahead of the accounting (a lagging value only makes the producer conservative). */
static uint64_t snapshot_tail(const bcir_ring_producer *p) {
  return ld_acq(p->region, BCIR_RING_OFF_TAIL);
}

/* The attach laws shared by both ends: the region, the geometry, then the owner word. */
static bcir_status attach_checks(uint8_t *region, size_t len, size_t owner_off, uint32_t takeover,
                                 bcir_ring_geometry *g, uint64_t *word) {
  if (!region || !aligned(region)) return BCIR_ERR_RING;
  bcir_status st = bcir_ring_decode_geometry(region, len, g);
  if (st != BCIR_OK) return st;
  *word = ld_acq(region, owner_off);
  uint32_t epoch = (uint32_t)(*word >> 32), state = (uint32_t)(*word & M32);
  if (state > BCIR_RING_ATTACHED || state == BCIR_RING_ATTACHING) return BCIR_ERR_RING;
  if (takeover == 0u) {
    if (state != BCIR_RING_DETACHED) return BCIR_ERR_BUSY;
  } else if (state != BCIR_RING_ATTACHED || epoch != takeover) {
    return BCIR_ERR_STALE;
  }
  if (epoch >= BCIR_RING_EPOCH_MAX) return BCIR_ERR_RING;
  return BCIR_OK;
}

bcir_ring_outcome bcir_ring_producer_attach(bcir_ring_producer *p, uint32_t takeover) {
  if (!p) return refused(BCIR_ERR_RING, 0);
  bcir_ring_geometry g;
  uint64_t word = 0;
  bcir_status st = attach_checks(p->region, p->region_len, BCIR_RING_OFF_P_OWNER, takeover, &g, &word);
  if (st != BCIR_OK) return refused(st, 0);
  uint8_t *r = p->region;
  uint32_t epoch = (uint32_t)(word >> 32);
  unsigned long long expected = (unsigned long long)word;
  if (!atomic_compare_exchange_strong_explicit(
          word_at(r, BCIR_RING_OFF_P_OWNER), &expected,
          (unsigned long long)owner_word(epoch, BCIR_RING_ATTACHING), memory_order_acq_rel,
          memory_order_acquire))
    return refused(BCIR_ERR_BUSY, 0);
  atomic_thread_fence(memory_order_release);
  uint64_t head = ld_acq(r, BCIR_RING_OFF_HEAD);
  p->g = g;
  if (takeover != 0u) {
    size_t base = slot_base(&g, head);
    uint64_t seq = ld_acq(r, base);
    if (seq & 1u) { /* the dead producer's half-written slot: retire it */
      st_rlx(r, base + 8, head);
      st_rlx(r, base + 16, 0u);
      st_rel(r, base, seq + 1u);
    }
  }
  st_rlx(r, BCIR_RING_OFF_ORIGIN, head);
  st_rel(r, BCIR_RING_OFF_P_OWNER, owner_word(epoch + 1u, BCIR_RING_ATTACHED));
  p->epoch = epoch + 1u;
  p->is_open = 0;
  p->cached_tail = 0;
  p->cached_tail = snapshot_tail(p);
  return outcome(BCIR_RING_OK, BCIR_OK, head, 0, p->epoch);
}

bcir_ring_outcome bcir_ring_producer_detach(bcir_ring_producer *p) {
  if (!p || !p->region || !producer_held(p)) return refused(BCIR_ERR_STALE, 0);
  if (p->is_open) return refused(BCIR_ERR_RING, 0);
  st_rel(p->region, BCIR_RING_OFF_P_OWNER, owner_word(p->epoch, BCIR_RING_DETACHED));
  return outcome(BCIR_RING_OK, BCIR_OK, 0, 0, p->epoch);
}

bcir_ring_outcome bcir_ring_write_open(bcir_ring_producer *p, size_t length) {
  if (!p || !p->region || !producer_held(p)) return refused(BCIR_ERR_STALE, 0);
  if (p->is_open) return refused(BCIR_ERR_RING, 0);
  const bcir_ring_geometry *g = &p->g;
  if (length > (size_t)(g->slot_size - BCIR_RING_SLOT_HEADER)) return refused(BCIR_ERR_NOSPACE, 0);
  uint8_t *r = p->region;
  uint64_t head = ld_rlx(r, BCIR_RING_OFF_HEAD);
  if (g->policy == BCIR_RING_BACKPRESSURE) {
    if (head - p->cached_tail >= g->slot_count) p->cached_tail = snapshot_tail(p);
    uint64_t depth = head - p->cached_tail;
    if (depth >= HALF || depth > g->slot_count) return refused(BCIR_ERR_RING, head);
    if (depth == g->slot_count) {
      st_rlx(r, BCIR_RING_OFF_REFUSED, ld_rlx(r, BCIR_RING_OFF_REFUSED) + 1u);
      return refused(BCIR_ERR_FULL, head);
    }
  }
  size_t base = slot_base(g, head);
  uint64_t seq = ld_rlx(r, base);
  if (seq & 1u) return refused(BCIR_ERR_RING, head);
  st_rlx(r, base, seq + 1u);
  atomic_thread_fence(memory_order_release);
  st_rlx(r, base + 8, head);
  st_rlx(r, base + 16, (uint64_t)p->epoch | (uint64_t)length << 32);
  p->is_open = 1;
  p->open_head = head;
  p->open_base = base;
  p->open_seq = seq;
  p->open_length = (uint32_t)length;
  return outcome(BCIR_RING_OPEN, BCIR_OK, head, 0, p->epoch);
}

bcir_ring_outcome bcir_ring_write_fill(bcir_ring_producer *p, size_t offset, const uint8_t *data,
                                       size_t n) {
  if (!p || !p->is_open) return refused(BCIR_ERR_RING, 0);
  if (n != 0u && !data) return refused(BCIR_ERR_RING, p->open_head);
  if (offset % 8u != 0u || offset > p->open_length || n > p->open_length - offset)
    return refused(BCIR_ERR_NOSPACE, p->open_head);
  size_t payload = p->open_base + BCIR_RING_SLOT_HEADER + offset;
  for (size_t i = 0; i < n; i += 8u) {
    uint64_t w = 0;
    for (size_t b = 0; b < 8u && i + b < n; b++) w |= (uint64_t)data[i + b] << (8u * b);
    st_rlx(p->region, payload + i, w);
  }
  return outcome(BCIR_RING_OPEN, BCIR_OK, p->open_head, 0, p->epoch);
}

bcir_ring_outcome bcir_ring_write_close(bcir_ring_producer *p) {
  if (!p || !p->is_open) return refused(BCIR_ERR_RING, 0);
  uint8_t *r = p->region;
  st_rel(r, p->open_base, p->open_seq + 2u);
  st_rel(r, BCIR_RING_OFF_HEAD, p->open_head + 1u);
  st_rlx(r, BCIR_RING_OFF_P_BEAT, ld_rlx(r, BCIR_RING_OFF_P_BEAT) + 1u);
  p->is_open = 0;
  return outcome(BCIR_RING_OK, BCIR_OK, p->open_head, 1, p->epoch);
}

bcir_ring_outcome bcir_ring_publish(bcir_ring_producer *p, const uint8_t *data, size_t n) {
  bcir_ring_outcome o = bcir_ring_write_open(p, n);
  if (o.verdict != BCIR_RING_OPEN) return o;
  o = bcir_ring_write_fill(p, 0, data, n);
  if (o.verdict != BCIR_RING_OPEN) return o;
  return bcir_ring_write_close(p);
}

bcir_ring_outcome bcir_ring_producer_beat(bcir_ring_producer *p) {
  if (!p || !p->region || !producer_held(p)) return refused(BCIR_ERR_STALE, 0);
  st_rlx(p->region, BCIR_RING_OFF_P_BEAT, ld_rlx(p->region, BCIR_RING_OFF_P_BEAT) + 1u);
  return outcome(BCIR_RING_OK, BCIR_OK, 0, 0, p->epoch);
}

/* ---- the consumer ---------------------------------------------------------------------- */

void bcir_ring_consumer_init(bcir_ring_consumer *c, uint8_t *region, size_t len) {
  if (!c) return;
  bcir_ring_consumer zero = {0};
  *c = zero;
  c->region = region;
  c->region_len = len;
}

static int consumer_held(const bcir_ring_consumer *c) {
  return c->epoch != 0u &&
         ld_acq(c->region, BCIR_RING_OFF_C_OWNER) == owner_word(c->epoch, BCIR_RING_ATTACHED);
}

bcir_ring_outcome bcir_ring_consumer_attach(bcir_ring_consumer *c, uint32_t takeover) {
  if (!c) return refused(BCIR_ERR_RING, 0);
  bcir_ring_geometry g;
  uint64_t word = 0;
  bcir_status st = attach_checks(c->region, c->region_len, BCIR_RING_OFF_C_OWNER, takeover, &g, &word);
  if (st != BCIR_OK) return refused(st, 0);
  uint8_t *r = c->region;
  uint64_t commit = ld_acq(r, BCIR_RING_OFF_C_COMMIT);
  size_t base = (commit & 1u) ? BCIR_RING_OFF_ACCT1 : BCIR_RING_OFF_ACCT0;
  uint64_t tail = ld_rlx(r, base), lost = ld_rlx(r, base + 8), stale = ld_rlx(r, base + 16);
  uint64_t last = ld_rlx(r, base + 24);
  uint64_t head = ld_acq(r, BCIR_RING_OFF_HEAD);
  if (head - tail >= HALF) return refused(BCIR_ERR_RING, tail);
  uint32_t epoch = (uint32_t)(word >> 32);
  unsigned long long expected = (unsigned long long)word;
  if (!atomic_compare_exchange_strong_explicit(
          word_at(r, BCIR_RING_OFF_C_OWNER), &expected,
          (unsigned long long)owner_word(epoch + 1u, BCIR_RING_ATTACHED), memory_order_acq_rel,
          memory_order_acquire))
    return refused(BCIR_ERR_BUSY, tail);
  st_rel(r, BCIR_RING_OFF_TAIL, tail); /* republish: a dead consumer may have died before it */
  c->g = g;
  c->epoch = epoch + 1u;
  c->commit = commit;
  c->tail = tail;
  c->lost = lost;
  c->stale = stale;
  c->last_epoch = (uint32_t)(last & M32);
  c->cached_head = tail;
  c->has_cursor = 0;
  c->copied = 0;
  return outcome(BCIR_RING_OK, BCIR_OK, tail, 0, c->epoch);
}

bcir_ring_outcome bcir_ring_consumer_detach(bcir_ring_consumer *c) {
  if (!c || !c->region || !consumer_held(c)) return refused(BCIR_ERR_STALE, 0);
  if (c->has_cursor) return refused(BCIR_ERR_RING, 0);
  st_rel(c->region, BCIR_RING_OFF_C_OWNER, owner_word(c->epoch, BCIR_RING_DETACHED));
  return outcome(BCIR_RING_OK, BCIR_OK, 0, 0, c->epoch);
}

static void commit_accounting(bcir_ring_consumer *c, uint64_t tail, uint64_t lost, uint64_t stale,
                              uint32_t last_epoch) {
  uint8_t *r = c->region;
  uint64_t next = c->commit + 1u;
  size_t base = (next & 1u) ? BCIR_RING_OFF_ACCT1 : BCIR_RING_OFF_ACCT0;
  atomic_thread_fence(memory_order_release);
  st_rlx(r, base, tail);
  st_rlx(r, base + 8, lost);
  st_rlx(r, base + 16, stale);
  st_rlx(r, base + 24, last_epoch);
  st_rel(r, BCIR_RING_OFF_C_COMMIT, next);
  st_rlx(r, BCIR_RING_OFF_C_BEAT, ld_rlx(r, BCIR_RING_OFF_C_BEAT) + 1u);
  st_rel(r, BCIR_RING_OFF_TAIL, tail);
  c->commit = next;
  c->tail = tail;
  c->lost = lost;
  c->stale = stale;
  c->last_epoch = last_epoch;
}

static bcir_ring_outcome lose(bcir_ring_consumer *c, uint64_t position, uint64_t count) {
  commit_accounting(c, position + count, c->lost + count, c->stale, c->last_epoch);
  return outcome(BCIR_RING_LOST, BCIR_OK, position, count, 0);
}

bcir_ring_outcome bcir_ring_read_open(bcir_ring_consumer *c) {
  if (!c || !c->region || !consumer_held(c)) return refused(BCIR_ERR_STALE, 0);
  if (c->has_cursor) return refused(BCIR_ERR_RING, 0);
  const bcir_ring_geometry *g = &c->g;
  uint8_t *r = c->region;
  uint64_t q = c->tail;
  if (g->policy == BCIR_RING_OVERWRITE || q == c->cached_head)
    c->cached_head = ld_acq(r, BCIR_RING_OFF_HEAD);
  uint64_t depth = c->cached_head - q;
  if (depth == 0u) return outcome(BCIR_RING_EMPTY, BCIR_OK, q, 0, 0);
  if (depth >= HALF) return refused(BCIR_ERR_RING, q);
  if (depth > g->slot_count) {
    if (g->policy == BCIR_RING_BACKPRESSURE) return refused(BCIR_ERR_RING, q);
    return lose(c, q, depth - g->slot_count);
  }
  size_t base = slot_base(g, q);
  uint64_t seq = ld_acq(r, base);
  if (seq & 1u) {
    if (g->policy == BCIR_RING_BACKPRESSURE) return refused(BCIR_ERR_RING, q);
    return lose(c, q, 1);
  }
  uint64_t word = ld_rlx(r, base + 16);
  c->cur_pos = ld_rlx(r, base + 8);
  c->cur_epoch = (uint32_t)(word & M32);
  c->cur_length = (uint32_t)(word >> 32);
  c->cur_position = q;
  c->cur_base = base;
  c->cur_seq = seq;
  c->has_cursor = 1;
  c->copied = 0;
  return outcome(BCIR_RING_OPEN, BCIR_OK, q, 0, 0);
}

bcir_ring_outcome bcir_ring_read_copy(bcir_ring_consumer *c, uint8_t *out, size_t cap) {
  if (!c || !c->has_cursor) return refused(BCIR_ERR_RING, 0);
  size_t capacity = (size_t)(c->g.slot_size - BCIR_RING_SLOT_HEADER);
  size_t n = c->cur_length < capacity ? (size_t)c->cur_length : capacity;
  if (!out) cap = 0;
  if (n > cap) n = cap;
  size_t payload = c->cur_base + BCIR_RING_SLOT_HEADER;
  for (size_t i = 0; i < n; i += 8u) {
    uint64_t w = ld_rlx(c->region, payload + i);
    for (size_t b = 0; b < 8u && i + b < n; b++) out[i + b] = (uint8_t)(w >> (8u * b));
  }
  c->copied = 1;
  return outcome(BCIR_RING_OPEN, BCIR_OK, c->cur_position, 0, 0);
}

bcir_ring_outcome bcir_ring_read_close(bcir_ring_consumer *c, uint8_t *out, size_t cap) {
  if (!c || !c->has_cursor) return refused(BCIR_ERR_RING, 0);
  if (!c->copied) (void)bcir_ring_read_copy(c, out, cap);
  const bcir_ring_geometry *g = &c->g;
  uint8_t *r = c->region;
  uint64_t q = c->cur_position;
  int overwrite = g->policy == BCIR_RING_OVERWRITE;
  atomic_thread_fence(memory_order_acquire);
  uint64_t seq = ld_rlx(r, c->cur_base);
  c->has_cursor = 0;
  if (seq != c->cur_seq) return overwrite ? lose(c, q, 1) : refused(BCIR_ERR_RING, q);
  if (c->cur_pos != q) {
    uint64_t ahead = c->cur_pos - q;
    if (overwrite && ahead < HALF && ahead % g->slot_count == 0u) return lose(c, q, 1);
    return refused(BCIR_ERR_RING, q);
  }
  if (c->cur_length > g->slot_size - BCIR_RING_SLOT_HEADER) return refused(BCIR_ERR_RING, q);
  uint64_t first = 0, origin = 0;
  unsigned tries = 0;
  for (; tries < 64u; tries++) {
    first = ld_acq(r, BCIR_RING_OFF_P_OWNER);
    origin = ld_rlx(r, BCIR_RING_OFF_ORIGIN);
    atomic_thread_fence(memory_order_acquire);
    if (ld_rlx(r, BCIR_RING_OFF_P_OWNER) == first) break;
  }
  if (tries == 64u) return refused(BCIR_ERR_BUSY, q); /* an attach in flight: read again */
  uint32_t current = (uint32_t)(first >> 32), state = (uint32_t)(first & M32);
  uint32_t epoch = c->cur_epoch;
  if (state > BCIR_RING_ATTACHED || epoch == 0u || epoch > current) return refused(BCIR_ERR_RING, q);
  int after_origin = (q - origin) < HALF;
  if ((after_origin && epoch != current) || (!after_origin && epoch < c->last_epoch)) {
    commit_accounting(c, q + 1u, c->lost, c->stale + 1u, c->last_epoch);
    return outcome(BCIR_RING_STALE, BCIR_OK, q, 1, epoch);
  }
  if ((size_t)c->cur_length > cap || (c->cur_length != 0u && !out))
    return refused(BCIR_ERR_NOSPACE, q); /* not consumed: read again with room for it */
  commit_accounting(c, q + 1u, c->lost, c->stale, epoch);
  bcir_ring_outcome o = outcome(BCIR_RING_DELIVERED, BCIR_OK, q, 1, epoch);
  o.length = c->cur_length;
  return o;
}

bcir_ring_outcome bcir_ring_consume(bcir_ring_consumer *c, uint8_t *out, size_t cap) {
  bcir_ring_outcome o = bcir_ring_read_open(c);
  if (o.verdict != BCIR_RING_OPEN) return o;
  (void)bcir_ring_read_copy(c, out, cap);
  return bcir_ring_read_close(c, out, cap);
}

bcir_ring_outcome bcir_ring_consumer_beat(bcir_ring_consumer *c) {
  if (!c || !c->region || !consumer_held(c)) return refused(BCIR_ERR_STALE, 0);
  st_rlx(c->region, BCIR_RING_OFF_C_BEAT, ld_rlx(c->region, BCIR_RING_OFF_C_BEAT) + 1u);
  return outcome(BCIR_RING_OK, BCIR_OK, 0, 0, c->epoch);
}
