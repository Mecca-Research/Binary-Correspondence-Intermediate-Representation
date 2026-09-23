/*===- fuzz_ring.c - libFuzzer: the live SPSC ring + TelemetryEnvelopeV0 --------===
 *
 * Three trust boundaries in one harness (the first input byte picks one):
 *   1. the envelope decoder over the raw input (bcir_tev_decode) -- total on arbitrary bytes,
 *      and canonical: a decoded envelope re-encodes to exactly the input (one spelling), and
 *      the intake decides it without reading past it;
 *   2. a HOSTILE region: the input becomes the shared region's bytes -- the geometry line
 *      repaired to a valid one (so the fuzzer lands past the CRC) while the owners, head, tail,
 *      accounting and slots stay whatever the input says. Both endpoints attach (taking over a
 *      peer the input claims is attached) and run a bounded number of operations: every access
 *      must stay inside the region (ASan) and every call must return;
 *   3. an HONEST region driven by the input as a SCRIPT: writes whose bytes are a function of
 *      their position, reads, two-phase writes and reads interleaved, endpoints that die and are
 *      taken over, detach and re-attach. After every operation the ring's laws must hold, or the
 *      harness aborts (a finding):
 *        - the committed accounting is exact: tail - origin == delivered + lost + stale, and
 *          head - origin == the positions the producers were told they published;
 *        - a delivered record is byte for byte what was published at its position;
 *        - a backpressure ring never holds more than slot_count records and never loses one;
 *        - nothing the harness wrote is ever reported STALE (no zombie writes in this mode).
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_ring.h"
#include "bcir_telemetry_envelope.h"

#define REGION_MAX 16384u
static _Alignas(64) uint8_t region[REGION_MAX];
static uint8_t scratch[BCIR_RING_SLOT_SIZE_MAX];

typedef struct cursor {
  const uint8_t *data;
  size_t size, at;
} cursor;

static uint8_t take(cursor *c) { return c->at < c->size ? c->data[c->at++] : 0u; }

static uint64_t get64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = v << 8 | p[i];
  return v;
}

/* ---- 1. the envelope ---- */
static void fuzz_envelope(const uint8_t *data, size_t size) {
  bcir_tev e;
  if (bcir_tev_decode(data, size, &e) == BCIR_OK) {
    uint8_t again[BCIR_TEV_MAX];
    size_t written = 0;
    if (bcir_tev_encode(&e, again, sizeof again, &written) != BCIR_OK) abort();
    if (written != size || memcmp(again, data, size) != 0) abort(); /* a second spelling */
  }
  static bcir_tev_intake intake;
  bcir_tev_intake_init(&intake, 3);
  (void)bcir_tev_intake_admit(&intake, data, size, NULL);
}

/* A valid geometry chosen from the input, fitting REGION_MAX. */
static bcir_ring_geometry pick_geometry(cursor *c) {
  bcir_ring_geometry g = {0};
  g.payload = (take(c) & 1u) ? BCIR_RING_CONTROL : BCIR_RING_TELEMETRY;
  g.policy = g.payload == BCIR_RING_CONTROL || (take(c) & 1u) ? BCIR_RING_BACKPRESSURE
                                                               : BCIR_RING_OVERWRITE;
  g.slot_size = 64u * (1u + take(c) % 4u);
  if (g.payload == BCIR_RING_CONTROL) g.slot_size = BCIR_RING_CONTROL_SLOT_MIN;
  g.slot_count = 2u << (take(c) % 4u); /* 2..16 */
  g.ring_id = 1u + take(c);
  uint8_t o = take(c);
  g.origin = o & 1u ? UINT64_MAX - (uint64_t)(o >> 1) : (uint64_t)o;
  return g;
}

/* ---- 2. a hostile region ---- */
static void fuzz_hostile(cursor *c) {
  bcir_ring_geometry g = pick_geometry(c);
  uint64_t size = bcir_ring_region_size(&g);
  if (size == 0u || size > REGION_MAX) return;
  if (bcir_ring_format(region, (size_t)size, &g) != BCIR_OK) abort();
  /* everything after the geometry line comes from the input */
  for (size_t i = BCIR_RING_GEOMETRY_SIZE; i < size; i++) region[i] = take(c);
  bcir_ring_producer p;
  bcir_ring_consumer k;
  bcir_ring_producer_init(&p, region, (size_t)size);
  bcir_ring_consumer_init(&k, region, (size_t)size);
  uint32_t pe = (uint32_t)(get64(region + BCIR_RING_OFF_P_OWNER) >> 32);
  uint32_t ce = (uint32_t)(get64(region + BCIR_RING_OFF_C_OWNER) >> 32);
  (void)bcir_ring_producer_attach(&p, take(c) & 1u ? pe : 0u);
  (void)bcir_ring_consumer_attach(&k, take(c) & 1u ? ce : 0u);
  for (int step = 0; step < 64; step++) {
    uint8_t op = take(c);
    if (op & 1u) {
      size_t n = (size_t)take(c) % (g.slot_size + 8u);
      (void)bcir_ring_publish(&p, scratch, n);
    } else {
      bcir_ring_outcome o = bcir_ring_consume(&k, scratch, sizeof scratch);
      if (o.verdict == BCIR_RING_DELIVERED && o.length > g.slot_size - BCIR_RING_SLOT_HEADER) abort();
    }
  }
  bcir_ring_accounting a;
  (void)bcir_ring_accounting_of(region, (size_t)size, &a);
}

/* ---- 3. an honest region, scripted ---- */
#define SHADOW 64u
typedef struct shadow_entry {
  uint64_t position;
  uint32_t length;
  uint8_t seed;
  int valid;
} shadow_entry;

static void content(uint64_t position, uint8_t seed, uint32_t length, uint8_t *out) {
  for (uint32_t i = 0; i < length; i++) out[i] = (uint8_t)(position * 31u + seed * 7u + i);
}

static void fuzz_script(cursor *c) {
  bcir_ring_geometry g = pick_geometry(c);
  uint64_t size = bcir_ring_region_size(&g);
  if (size == 0u || size > REGION_MAX) return;
  if (bcir_ring_format(region, (size_t)size, &g) != BCIR_OK) abort();
  static shadow_entry shadow[SHADOW];
  memset(shadow, 0, sizeof shadow);
  uint32_t capacity = g.slot_size - BCIR_RING_SLOT_HEADER;
  bcir_ring_producer p;
  bcir_ring_consumer k;
  bcir_ring_producer_init(&p, region, (size_t)size);
  bcir_ring_consumer_init(&k, region, (size_t)size);
  if (bcir_ring_producer_attach(&p, 0).verdict != BCIR_RING_OK) abort();
  if (bcir_ring_consumer_attach(&k, 0).verdict != BCIR_RING_OK) abort();
  uint64_t published = 0, delivered = 0, lost = 0;
  uint8_t open_seed = 0;
  uint32_t open_length = 0;
  uint8_t buf[BCIR_RING_SLOT_SIZE_MAX], want[BCIR_RING_SLOT_SIZE_MAX];
  for (int step = 0; step < 256 && c->at < c->size; step++) {
    uint8_t op = take(c) % 10u;
    bcir_ring_outcome o;
    switch (op) {
      case 0: case 1: { /* write */
        uint32_t n = take(c) % (capacity + 1u);
        uint8_t seed = take(c);
        uint64_t head = get64(region + BCIR_RING_OFF_HEAD);
        content(head, seed, n, buf);
        o = bcir_ring_publish(&p, buf, n);
        if (o.verdict == BCIR_RING_OK) {
          if (o.position != head) abort();
          shadow[head % SHADOW] = (shadow_entry){head, n, seed, 1};
          published++;
        } else if (o.status != BCIR_ERR_FULL && o.status != BCIR_ERR_RING) {
          abort();
        }
        break;
      }
      case 2: { /* open + a partial fill, left open */
        if (p.is_open) break;
        open_length = take(c) % (capacity + 1u);
        open_seed = take(c);
        o = bcir_ring_write_open(&p, open_length);
        if (o.verdict == BCIR_RING_OPEN) {
          content(o.position, open_seed, open_length, buf);
          uint32_t part = (open_length / 16u) * 8u;
          (void)bcir_ring_write_fill(&p, 0, buf, part);
        }
        break;
      }
      case 3: { /* complete an open write */
        if (!p.is_open) break;
        uint64_t head = p.open_head;
        content(head, open_seed, open_length, buf);
        (void)bcir_ring_write_fill(&p, 0, buf, open_length);
        o = bcir_ring_write_close(&p);
        if (o.verdict != BCIR_RING_OK || o.position != head) abort();
        shadow[head % SHADOW] = (shadow_entry){head, open_length, open_seed, 1};
        published++;
        break;
      }
      case 4: case 5: { /* read */
        o = bcir_ring_consume(&k, buf, sizeof buf);
        if (o.verdict == BCIR_RING_DELIVERED) {
          shadow_entry *s = &shadow[o.position % SHADOW];
          if (!s->valid || s->position != o.position || s->length != o.length) abort();
          content(o.position, s->seed, s->length, want);
          if (memcmp(want, buf, s->length) != 0) abort(); /* torn */
          delivered++;
        } else if (o.verdict == BCIR_RING_LOST) {
          if (g.policy == BCIR_RING_BACKPRESSURE) abort(); /* backpressure never loses */
          lost += o.count;
        } else if (o.verdict == BCIR_RING_STALE) {
          abort(); /* the harness writes nothing a deposed producer could have */
        } else if (o.verdict == BCIR_RING_REFUSED) {
          abort(); /* an honest ring is never corrupt */
        }
        break;
      }
      case 6: { /* the producer dies (possibly mid-write); a successor takes over */
        uint32_t dead = p.epoch;
        bcir_ring_producer_init(&p, region, (size_t)size);
        if (bcir_ring_producer_attach(&p, dead).verdict != BCIR_RING_OK) abort();
        break;
      }
      case 7: { /* the consumer dies (possibly mid-read); a successor takes over */
        uint32_t dead = k.epoch;
        bcir_ring_consumer_init(&k, region, (size_t)size);
        if (bcir_ring_consumer_attach(&k, dead).verdict != BCIR_RING_OK) abort();
        break;
      }
      case 8: { /* a two-phase read, the producer writing in between */
        o = bcir_ring_read_open(&k);
        if (o.verdict != BCIR_RING_OPEN) {
          if (o.verdict == BCIR_RING_LOST) lost += o.count;
          break;
        }
        uint32_t n = take(c) % (capacity + 1u);
        uint64_t head = get64(region + BCIR_RING_OFF_HEAD);
        content(head, 9, n, buf);
        if (!p.is_open && bcir_ring_publish(&p, buf, n).verdict == BCIR_RING_OK) {
          shadow[head % SHADOW] = (shadow_entry){head, n, 9, 1};
          published++;
        }
        o = bcir_ring_read_close(&k, buf, sizeof buf);
        if (o.verdict == BCIR_RING_DELIVERED) {
          shadow_entry *s = &shadow[o.position % SHADOW];
          if (!s->valid || s->position != o.position || s->length != o.length) abort();
          content(o.position, s->seed, s->length, want);
          if (memcmp(want, buf, s->length) != 0) abort();
          delivered++;
        } else if (o.verdict == BCIR_RING_LOST) {
          if (g.policy == BCIR_RING_BACKPRESSURE) abort();
          lost += o.count;
        } else {
          abort();
        }
        break;
      }
      default: { /* detach and re-attach an endpoint (never while open) */
        if (take(c) & 1u) {
          if (!p.is_open && bcir_ring_producer_detach(&p).verdict == BCIR_RING_OK &&
              bcir_ring_producer_attach(&p, 0).verdict != BCIR_RING_OK)
            abort();
        } else if (!k.has_cursor && bcir_ring_consumer_detach(&k).verdict == BCIR_RING_OK &&
                   bcir_ring_consumer_attach(&k, 0).verdict != BCIR_RING_OK) {
          abort();
        }
        break;
      }
    }
    /* the laws, after every operation */
    bcir_ring_accounting a;
    if (bcir_ring_accounting_of(region, (size_t)size, &a) != BCIR_OK) abort();
    uint64_t head = get64(region + BCIR_RING_OFF_HEAD);
    if (head - g.origin != published) abort();
    if (a.tail - g.origin != a.delivered + a.lost + a.stale) abort();
    if (a.delivered != delivered || a.lost != lost || a.stale != 0u) abort();
    if (g.policy == BCIR_RING_BACKPRESSURE && head - a.tail > g.slot_count) abort();
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size == 0u) return 0;
  cursor c = {data + 1, size - 1u, 0};
  switch (data[0] % 3u) {
    case 0: fuzz_envelope(data + 1, size - 1u); break;
    case 1: fuzz_hostile(&c); break;
    default: fuzz_script(&c); break;
  }
  return 0;
}
