/*===- fuzz_control_plane.c - libFuzzer: ControlRecordV1 decoder + resident plane -===
 *
 * Two trust boundaries in one harness:
 *   1. the keyless decoder over the raw input (bcir_ctl_validate / _decode / _check_mac /
 *      _rollback_token) -- total on arbitrary bytes, never an out-of-bounds read;
 *   2. the resident plane, driven by the input read as a SCRIPT of operations. A record the
 *      fuzzer writes would die at the MAC wall long before it reached a lease table or the
 *      pending slot, so a "sealed" operation shapes the record's fields from the input,
 *      MACs it under the key the plane holds (or the lease's) and fixes the CRC -- the
 *      BCIRQ8 harness's checksum repair, applied to authority -- while a "raw" operation
 *      keeps every refusal path covered.
 *
 * After every operation the plane's invariants must hold, or the harness aborts (a finding):
 *   - a refused operation changes nothing: the state is byte-identical (a boundary that
 *     refuses a due pending switch is the one exception -- it crossed, and reports it);
 *   - the generation never decreases, and a deferral never moves it;
 *   - a switch is applied only while nothing is in flight;
 *   - the lease table holds at most eight leases, strictly ascending by id, each holding exactly
 *     its derived key (an unused slot holds nothing), and the pending
 *     slot is empty or holds a verified switch.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_control_plane.h"

static const uint8_t ROOT[32] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
                                 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};
static const uint8_t REASONS[7] = {0, 1, 4, 4, 3, 4, 3};
static const uint32_t BODY[7] = {0, 40, 48, 8, 64, 64, 16};

typedef struct cursor {
  const uint8_t *data;
  size_t size, at;
} cursor;

static uint8_t take(cursor *c) { return c->at < c->size ? c->data[c->at++] : 0u; }

static void put32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void put64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static int is_switch(uint8_t kind) {
  return kind == BCIR_CTL_GENERATION || kind == BCIR_CTL_ACTIVATE || kind == BCIR_CTL_ROLLBACK;
}

/* One of three distinct nonzero digests, or (when `zero_ok`) all-zero. */
static void pick_digest(cursor *c, uint8_t out[32], int zero_ok) {
  uint8_t which = (uint8_t)(take(c) % (zero_ok ? 4u : 3u));
  memset(out, 0, 32);
  if (which < 3u) memset(out, 0x11 * (which + 1), 32);
}

/* Shape a record from the input around the plane's state, then MAC and CRC-seal it. */
static size_t sealed(cursor *c, const bcir_ctl_state *s, uint8_t out[BCIR_CTL_RECORD_MAX]) {
  uint8_t kind = (uint8_t)(1u + take(c) % 6u);
  uint8_t *b = out + BCIR_CTL_HEADER_SIZE;
  uint64_t lease = 0, sequence;
  uint32_t expect;
  memset(out, 0, BCIR_CTL_RECORD_MAX);
  memcpy(out, "BCTL", 4);
  out[4] = 1;
  out[8] = kind;
  out[9] = take(c) % 16u == 0u ? (uint8_t)(take(c) % 5u) : s->scope;
  out[10] = (uint8_t)(take(c) % REASONS[kind]);
  put32(out + 12, BODY[kind]);
  switch (take(c) % 4u) {  /* mostly the resident generation: the interesting decisions */
    case 0: expect = s->generation ? s->generation - 1u : 0u; break;
    case 1: expect = s->generation + 1u; break;
    default: expect = s->generation; break;
  }
  put32(out + 16, is_switch(kind) ? expect + 1u : expect);
  put32(out + 20, expect);
  put64(out + 24, BCIR_CTL_CAP(kind));
  put64(out + 32, s->boundary + (take(c) % 4u == 0u ? take(c) % 4u : 0u));
  if (kind != BCIR_CTL_LEASE) {
    uint8_t pick = take(c);
    lease = s->n_leases && pick % 8u ? s->leases[pick % s->n_leases].lease_id : pick % 12u + 1u;
  }
  {
    const bcir_ctl_lease_entry *entry = 0;
    for (uint32_t i = 0; i < s->n_leases; ++i)
      if (s->leases[i].lease_id == lease) entry = &s->leases[i];
    uint64_t last = kind == BCIR_CTL_LEASE ? s->root_sequence : entry ? entry->last_sequence : 0u;
    sequence = take(c) % 8u == 0u ? take(c) : last + 1u + take(c) % 2u;
  }
  put64(out + 48, lease);
  put64(out + 56, take(c) % 16u == 0u ? take(c) : s->subject);
  switch (kind) {
    case BCIR_CTL_LEASE: {
      uint64_t issued = s->boundary > 2u ? s->boundary - take(c) % 3u : take(c) % 2u;
      uint64_t granted = (uint64_t)take(c) & BCIR_CTL_CAP_GRANTABLE;
      put64(b, take(c) % 4u == 0u ? take(c) % 12u : s->last_lease_id + 1u);
      put64(b + 8, granted ? granted : BCIR_CTL_CAP_GRANTABLE);
      put64(b + 16, issued);
      put64(b + 24, issued + 1u + take(c) % 8u);
      put64(b + 32, 42);
      break;
    }
    case BCIR_CTL_GENERATION:
      put32(b, take(c) % 4u);
      put32(b + 4, take(c) % 4u);
      put32(b + 8, take(c) % 4u);
      pick_digest(c, b + 16, 0);
      break;
    case BCIR_CTL_QUIESCE:
      put64(b, s->boundary + take(c) % 4u);
      break;
    case BCIR_CTL_ACTIVATE:
      pick_digest(c, b, 0);
      if (take(c) % 4u) memcpy(b + 32, s->artifact, 32);
      else pick_digest(c, b + 32, 1);
      if (memcmp(b, b + 32, 32) == 0) b[0] ^= 0x5a;  /* an activation changes the artifact */
      break;
    case BCIR_CTL_ROLLBACK: {
      static const uint8_t zero[32] = {0};
      if (take(c) % 4u && memcmp(s->previous, zero, 32) != 0) memcpy(b, s->previous, 32);
      else pick_digest(c, b, 0);
      if (take(c) % 4u && memcmp(s->token, zero, 32) != 0) memcpy(b + 32, s->token, 32);
      else pick_digest(c, b + 32, 0);
      break;
    }
    default: { /* cancel: a range around the pending switch's sequence */
      bcir_ctl_record p;
      uint64_t around = s->pending_len && bcir_ctl_decode(s->pending, s->pending_len, &p) == BCIR_OK
                            ? p.hdr.sequence
                            : 1u + take(c) % 4u;
      uint64_t first = around > 1u ? around - take(c) % 2u : 1u;
      uint64_t last = around + take(c) % 2u;
      put64(b, first);
      put64(b + 8, last);
      if (sequence <= last) sequence = last + 1u;
      break;
    }
  }
  put64(out + 40, sequence);
  {
    size_t len = BCIR_CTL_MIN_BYTES + BODY[kind];
    size_t signed_len = BCIR_CTL_HEADER_SIZE + BODY[kind];
    uint8_t key[32];
    if (lease == 0u) {
      bcir_hmac_sha256(ROOT, sizeof(ROOT), out, signed_len, out + signed_len);
    } else {
      if (bcir_ctl_lease_key(ROOT, sizeof(ROOT), lease, key) != BCIR_OK) abort();
      bcir_hmac_sha256(key, sizeof(key), out, signed_len, out + signed_len);
    }
    put32(out + len - 4u, bcir_crc32(out, len - 4u));
    return len;
  }
}

/* The state has no padding (every field is naturally aligned and the sizes sum to the
 * struct's), so a byte comparison of two snapshots is exact -- and far cheaper than the
 * state digest, which the harness exercises once per input instead. */
BCIR_STATIC_ASSERT(sizeof(bcir_ctl_state) == 64 + 4 + 4 + 8 + 4 * 4 + 32 * 4 + 8 * 5 + 4 + 4 +
                                                 80 * BCIR_CTL_LEASE_CAPACITY + BCIR_CTL_RECORD_MAX,
                   "bcir_ctl_state has no padding");

static void check(const bcir_ctl_state *s, const bcir_ctl_outcome *o, const bcir_ctl_state *before,
                  int crossed_boundary) {
  uint32_t generation_before = before->generation;
  if (o->verdict == BCIR_CTL_REFUSED && !(crossed_boundary && o->kind != 0u) &&
      memcmp(before, s, sizeof(*s)) != 0)
    abort();  /* a refusal changed the resident state */
  if (s->generation < generation_before) abort();
  if (o->verdict == BCIR_CTL_DEFERRED && (s->generation != generation_before || !s->pending_len))
    abort();
  if (o->verdict == BCIR_CTL_APPLIED && is_switch(o->kind) && s->in_flight != 0u) abort();
  if (o->refusal > BCIR_CTL_REFUSAL_MAX || o->verdict > BCIR_CTL_NONE) abort();
  if (s->n_leases > BCIR_CTL_LEASE_CAPACITY) abort();
  for (uint32_t i = 1; i < s->n_leases; ++i)
    if (s->leases[i].lease_id <= s->leases[i - 1].lease_id) abort();
  /* the key cache (G15/S3-B) is exact: every live lease holds exactly its derived key, and a
   * slot the table does not use holds nothing -- a lease that left took its key with it */
  for (uint32_t i = 0; i < BCIR_CTL_LEASE_CAPACITY; ++i) {
    uint8_t want[32] = {0};
    if (i < s->n_leases &&
        bcir_ctl_lease_key(ROOT, sizeof(ROOT), s->leases[i].lease_id, want) != BCIR_OK)
      abort();
    if (i >= s->n_leases) {
      static const bcir_ctl_lease_entry empty;
      if (memcmp(&s->leases[i], &empty, sizeof(empty)) != 0) abort();
    } else if (memcmp(s->leases[i].key, want, sizeof(want)) != 0) {
      abort();
    }
  }
  if (s->pending_len) {
    bcir_ctl_record p;
    if (s->pending_len > BCIR_CTL_RECORD_MAX ||
        bcir_ctl_decode(s->pending, s->pending_len, &p) != BCIR_OK || !is_switch(p.hdr.kind))
      abort();
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  /* 1. the keyless decoder, on the raw bytes */
  {
    bcir_ctl_header h;
    bcir_ctl_record r;
    uint8_t token[32];
    (void)bcir_ctl_validate(data, size, &h);
    if (bcir_ctl_decode(data, size, &r) == BCIR_OK) {
      (void)bcir_ctl_check_mac(data, size, ROOT, sizeof(ROOT));
      (void)bcir_ctl_rollback_token(data, size, token);
    }
  }
  /* 2. the plane, driven by the bytes as a script */
  static bcir_ctl_state state, before;
  uint8_t last_sealed[BCIR_CTL_RECORD_MAX], digest[32];
  size_t last_len = 0;
  cursor c = {data, size, 0};
  if (bcir_ctl_init(&state, ROOT, sizeof(ROOT), BCIR_CTL_SCOPE_MODULE, 7) != BCIR_OK) abort();
  for (unsigned ops = 0; ops < 256u && c.at < c.size; ++ops) {
    uint8_t op = (uint8_t)(take(&c) % 8u);
    int crossed = op == 3u || op == 4u;
    bcir_ctl_outcome o;
    memcpy(&before, &state, sizeof(state));
    switch (op) {
      case 0: { /* raw bytes, in an exact-size buffer so ASan sees any overread */
        size_t n = take(&c) % (BCIR_CTL_RECORD_MAX + 9u);
        if (n > c.size - c.at) n = c.size - c.at;
        uint8_t *raw = (uint8_t *)malloc(n ? n : 1u);
        if (!raw) return 0;
        memcpy(raw, c.data + c.at, n);
        c.at += n;
        o = bcir_ctl_submit(&state, n ? raw : NULL, n);
        free(raw);
        break;
      }
      case 1:
        last_len = sealed(&c, &state, last_sealed);
        o = bcir_ctl_submit(&state, last_sealed, last_len);
        break;
      case 2: o = bcir_ctl_enter(&state); break;
      case 3: o = bcir_ctl_leave(&state); break;
      case 4: o = bcir_ctl_advance(&state); break;
      case 5:
      case 6: {
        size_t n = (size_t)take(&c) | ((size_t)take(&c) << 8);
        if (n > c.size - c.at) n = c.size - c.at;
        o = op == 5u ? bcir_ctl_admit_pack(&state, c.data + c.at, n)
                     : bcir_ctl_admit_plan(&state, c.data + c.at, n);
        c.at += n;
        break;
      }
      default: /* replay the last sealed record */
        if (!last_len) continue;
        o = bcir_ctl_submit(&state, last_sealed, last_len);
        break;
    }
    check(&state, &o, &before, crossed);
  }
  bcir_ctl_state_digest(&state, digest);  /* the digest over whatever state the script reached */
  return 0;
}
