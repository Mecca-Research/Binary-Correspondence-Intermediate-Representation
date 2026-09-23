/*===- bcir_control_plane.c - ControlRecordV1: decoder, verifier, resident plane -===
 *
 * The freestanding C twin of bcir/abi/control_abi.py (the wire laws, in the same order) and
 * bcir/gem/control.py (the plane, decision for decision). bcir_control_plane.h is the
 * normative C view; docs/kernel/BCIR_CONTROL_PLANE_ABI.md the prose spec.
 *
 * No heap and no libc: SHA-256 / HMAC-SHA256 come from bcir_sha256.c, the CRC and the
 * StreamPack / ExecutionPlanV1 verifiers from bcir_runtime.c. Every read of caller bytes is
 * bounded by the framing laws before a field is trusted.
 *===----------------------------------------------------------------------===*/
#include "bcir_control_plane.h"

#include "bcir_execution_plan.h"

/* The domain-separation tags (each ends with its NUL: sizeof counts it, as the spec's
 * `|| 0x00` does). */
static const uint8_t LEASE_TAG[] = "BCTL/lease/v1";
static const uint8_t REGISTRY_TAG[] = "BCTL/registry/v1";
static const uint8_t TOKEN_TAG[] = "BCTL/token/v1";
static const uint8_t STATE_TAG[] = "BCTL/state/v1";

/* The append-only (version, kind) body table and each kind's closed reason set. */
static const uint32_t BODY_BYTES_V1[BCIR_CTL_KIND_MAX + 1] = {0, 40, 48, 8, 64, 64, 16};
static const uint8_t REASON_COUNT[BCIR_CTL_KIND_MAX + 1] = {0, 1, 4, 4, 3, 4, 3};

static const uint8_t ZERO32[32] = {0};

/* --- little-endian primitives ------------------------------------------------------------ */

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)(p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static void wr32(uint8_t *p, uint32_t v) {
  for (unsigned i = 0; i < 4u; ++i) p[i] = (uint8_t)(v >> (8u * i));
}
static void wr64(uint8_t *p, uint64_t v) {
  for (unsigned i = 0; i < 8u; ++i) p[i] = (uint8_t)(v >> (8u * i));
}
static void copy_bytes(uint8_t *dst, const uint8_t *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}
static void wipe(void *pointer, size_t n) {  /* volatile: a key copy must not survive */
  volatile uint8_t *p = (volatile uint8_t *)pointer;
  for (size_t i = 0; i < n; ++i) p[i] = 0;
}
static int is_zero32(const uint8_t *p) {
  uint8_t acc = 0;
  for (unsigned i = 0; i < 32u; ++i) acc |= p[i];
  return acc == 0;
}
static int eq32(const uint8_t *a, const uint8_t *b) { return bcir_sha256_equal(a, b, 32u); }

static int is_switch(uint8_t kind) {
  return kind == BCIR_CTL_GENERATION || kind == BCIR_CTL_ACTIVATE || kind == BCIR_CTL_ROLLBACK;
}

/* The one staleness predicate (bcir/gem/control.py::is_stale): a record minted against the
 * resident generation `witnessed` is stale once the resident has moved past it. */
static int is_stale(uint32_t witnessed, uint32_t resident) { return witnessed < resident; }

/* --- the wire laws ------------------------------------------------------------------------- */

uint32_t bcir_ctl_body_bytes(uint16_t version, uint8_t kind) {
  if (version < BCIR_CTL_VERSION || version > BCIR_CTL_VERSION_MAX) return 0;
  if (kind < 1u || kind > BCIR_CTL_KIND_MAX) return 0;
  return BODY_BYTES_V1[kind];
}

static void read_header(const uint8_t *d, bcir_ctl_header *h) {
  for (unsigned i = 0; i < 4u; ++i) h->magic[i] = d[i];
  h->version = rd16(d + 4);
  h->flags = rd16(d + 6);
  h->kind = d[8];
  h->scope = d[9];
  h->reason = d[10];
  h->reserved0 = d[11];
  h->body_len = rd32(d + 12);
  h->generation = rd32(d + 16);
  h->expect = rd32(d + 20);
  h->capability = rd64(d + 24);
  h->boundary = rd64(d + 32);
  h->sequence = rd64(d + 40);
  h->lease = rd64(d + 48);
  h->subject = rd64(d + 56);
}

bcir_status bcir_ctl_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                              bcir_ctl_header *BCIR_RESTRICT hdr) {
  bcir_ctl_header h;
  /* 1. at least a header and a trailer */
  if (!data || len < BCIR_CTL_MIN_BYTES) return BCIR_ERR_TRUNCATED;
  read_header(data, &h);
  /* 2-4. magic, version, reserved header bytes */
  if (h.magic[0] != 'B' || h.magic[1] != 'C' || h.magic[2] != 'T' || h.magic[3] != 'L')
    return BCIR_ERR_MAGIC;
  if (h.version < BCIR_CTL_VERSION || h.version > BCIR_CTL_VERSION_MAX) return BCIR_ERR_VERSION;
  if (h.flags != 0u || h.reserved0 != 0u) return BCIR_ERR_RESERVED;
  /* 5-6. the kind and its body length (redundant on purpose: bounded before trusted) */
  if (h.kind < 1u || h.kind > BCIR_CTL_KIND_MAX) return BCIR_ERR_CONTROL;
  if (h.body_len != bcir_ctl_body_bytes(h.version, h.kind)) return BCIR_ERR_CONTROL;
  /* 7. exactly one record: never an extension point */
  {
    size_t total = (size_t)BCIR_CTL_MIN_BYTES + (size_t)h.body_len;  /* <= 164: no overflow */
    if (len < total) return BCIR_ERR_TRUNCATED;
    if (len > total) return BCIR_ERR_TRAILING;
    /* 8. the CRC over every preceding byte */
    if (bcir_crc32(data, total - 4u) != rd32(data + total - 4u)) return BCIR_ERR_CRC;
  }
  if (hdr) *hdr = h;
  return BCIR_OK;
}

bcir_status bcir_ctl_decode(const uint8_t *BCIR_RESTRICT data, size_t len,
                            bcir_ctl_record *BCIR_RESTRICT out) {
  bcir_ctl_record r;
  bcir_status st = bcir_ctl_validate(data, len, &r.hdr);
  if (st != BCIR_OK) return st;
  const bcir_ctl_header *h = &r.hdr;
  const uint8_t *b = data + BCIR_CTL_HEADER_SIZE;

  /* 9. the header field laws */
  if (h->scope > BCIR_CTL_SCOPE_MAX) return BCIR_ERR_CONTROL;
  if (h->reason >= REASON_COUNT[h->kind]) return BCIR_ERR_CONTROL;
  if (h->capability != BCIR_CTL_CAP(h->kind)) return BCIR_ERR_CONTROL;
  if (h->sequence == 0u) return BCIR_ERR_CONTROL;
  if ((h->lease == 0u) != (h->kind == BCIR_CTL_LEASE)) return BCIR_ERR_CONTROL;
  if (is_switch(h->kind)) {
    /* expect + 1 without wrapping: a stale tag cannot come back to life */
    if (h->expect == UINT32_MAX || h->generation != h->expect + 1u) return BCIR_ERR_CONTROL;
  } else if (h->generation != h->expect) {
    return BCIR_ERR_CONTROL;
  }

  /* the body, in wire order */
  switch (h->kind) {
    case BCIR_CTL_LEASE:
      r.body.lease.lease_id = rd64(b);
      r.body.lease.granted = rd64(b + 8);
      r.body.lease.issued_epoch = rd64(b + 16);
      r.body.lease.expiry_epoch = rd64(b + 24);
      r.body.lease.holder = rd64(b + 32);
      break;
    case BCIR_CTL_GENERATION:
      r.body.generation.map_gen = rd32(b);
      r.body.generation.data_gen = rd32(b + 4);
      r.body.generation.topo_gen = rd32(b + 8);
      r.body.generation.reserved = rd32(b + 12);
      copy_bytes(r.body.generation.registry_digest, b + 16, 32u);
      break;
    case BCIR_CTL_QUIESCE:
      r.body.quiesce.drain_deadline = rd64(b);
      break;
    case BCIR_CTL_ACTIVATE:
      copy_bytes(r.body.activate.artifact_sha256, b, 32u);
      copy_bytes(r.body.activate.previous_sha256, b + 32, 32u);
      break;
    case BCIR_CTL_ROLLBACK:
      copy_bytes(r.body.rollback.restore_sha256, b, 32u);
      copy_bytes(r.body.rollback.rollback_token, b + 32, 32u);
      break;
    default: /* BCIR_CTL_CANCEL (the kind was range-checked by the framing laws) */
      r.body.cancel.first_sequence = rd64(b);
      r.body.cancel.last_sequence = rd64(b + 8);
      break;
  }

  /* 10. the generation body's reserved word, then the body laws */
  if (h->kind == BCIR_CTL_GENERATION && r.body.generation.reserved != 0u) return BCIR_ERR_RESERVED;
  switch (h->kind) {
    case BCIR_CTL_LEASE: {
      const bcir_ctl_lease_body *l = &r.body.lease;
      if (l->lease_id == 0u) return BCIR_ERR_CONTROL;
      if (l->granted == 0u || (l->granted & ~BCIR_CTL_CAP_GRANTABLE) != 0u) return BCIR_ERR_CONTROL;
      if (l->expiry_epoch <= l->issued_epoch) return BCIR_ERR_CONTROL;
      if (l->holder == 0u) return BCIR_ERR_CONTROL;
      break;
    }
    case BCIR_CTL_GENERATION:
      if (is_zero32(r.body.generation.registry_digest)) return BCIR_ERR_CONTROL;
      break;
    case BCIR_CTL_QUIESCE:
      if (r.body.quiesce.drain_deadline < h->boundary) return BCIR_ERR_CONTROL;
      break;
    case BCIR_CTL_ACTIVATE:
      if (is_zero32(r.body.activate.artifact_sha256)) return BCIR_ERR_CONTROL;
      if (eq32(r.body.activate.artifact_sha256, r.body.activate.previous_sha256))
        return BCIR_ERR_CONTROL;
      break;
    case BCIR_CTL_ROLLBACK:
      if (is_zero32(r.body.rollback.restore_sha256)) return BCIR_ERR_CONTROL;
      if (is_zero32(r.body.rollback.rollback_token)) return BCIR_ERR_CONTROL;
      break;
    default: { /* cancel: 1 <= first <= last < its own sequence (only earlier records) */
      const bcir_ctl_cancel_body *c = &r.body.cancel;
      if (c->first_sequence < 1u || c->first_sequence > c->last_sequence ||
          c->last_sequence >= h->sequence)
        return BCIR_ERR_CONTROL;
      break;
    }
  }

  /* 11. a MAC is present (whether it is the right one needs the key) */
  copy_bytes(r.mac, data + len - BCIR_CTL_TRAILER_SIZE, 32u);
  if (is_zero32(r.mac)) return BCIR_ERR_MAC;
  if (out) *out = r;
  return BCIR_OK;
}

bcir_status bcir_ctl_verify(const uint8_t *BCIR_RESTRICT data, size_t len) {
  return bcir_ctl_decode(data, len, (bcir_ctl_record *)0);
}

/* The MAC of an already-framed record of `len` bytes, compared in constant time. */
static int mac_matches(const uint8_t *data, size_t len, const uint8_t *key, size_t key_len) {
  uint8_t mac[32];
  bcir_hmac_sha256(key, key_len, data, len - BCIR_CTL_TRAILER_SIZE, mac);
  int ok = bcir_sha256_equal(mac, data + len - BCIR_CTL_TRAILER_SIZE, 32u);
  wipe(mac, sizeof(mac));
  return ok;
}

bcir_status bcir_ctl_check_mac(const uint8_t *BCIR_RESTRICT data, size_t len,
                               const uint8_t *BCIR_RESTRICT key, size_t key_len) {
  bcir_status st = bcir_ctl_verify(data, len);
  if (st != BCIR_OK) return st;
  /* an invalid (NULL, nonzero) key is refused, never read as the empty key */
  if (!key && key_len != 0u) return BCIR_ERR_MAC;
  return mac_matches(data, len, key, key_len) ? BCIR_OK : BCIR_ERR_MAC;
}

bcir_status bcir_ctl_lease_key(const uint8_t *BCIR_RESTRICT root, size_t root_len,
                               uint64_t lease_id, uint8_t out[32]) {
  bcir_hmac_sha256_ctx ctx;
  uint8_t id[8];
  if (!root && root_len != 0u) {  /* never read a missing key as the empty one */
    wipe(out, 32u);
    return BCIR_ERR_CONTROL;
  }
  wr64(id, lease_id);
  bcir_hmac_sha256_init(&ctx, root, root_len);
  bcir_hmac_sha256_update(&ctx, LEASE_TAG, sizeof(LEASE_TAG));
  bcir_hmac_sha256_update(&ctx, id, sizeof(id));
  bcir_hmac_sha256_final(&ctx, out);
  wipe(&ctx, sizeof(ctx));
  return BCIR_OK;
}

static int feed_generation(const bcir_generation_view *g, void *ctx) {
  uint8_t b[BCIR_GENERATION_WIRE_SIZE];
  wr32(b, g->rid);
  wr32(b + 4, g->map_gen);
  wr32(b + 8, g->data_gen);
  bcir_sha256_update((bcir_sha256 *)ctx, b, sizeof(b));
  return 0;
}

bcir_status bcir_ctl_registry_digest(const bcir_generation_view *BCIR_RESTRICT vector, size_t n,
                                     uint8_t out[32]) {
  bcir_sha256 h;
  if (!vector && n != 0u) return BCIR_ERR_GENERATION;
  for (size_t i = 1; i < n; ++i)
    if (vector[i].rid <= vector[i - 1].rid) return BCIR_ERR_GENERATION;
  bcir_sha256_init(&h);
  bcir_sha256_update(&h, REGISTRY_TAG, sizeof(REGISTRY_TAG));
  for (size_t i = 0; i < n; ++i) (void)feed_generation(&vector[i], &h);
  bcir_sha256_final(&h, out);
  return BCIR_OK;
}

static void token_of(const uint8_t *data, size_t len, uint8_t out[32]) {
  bcir_sha256 h;
  bcir_sha256_init(&h);
  bcir_sha256_update(&h, TOKEN_TAG, sizeof(TOKEN_TAG));
  bcir_sha256_update(&h, data, len - BCIR_CTL_TRAILER_SIZE);
  bcir_sha256_final(&h, out);
}

bcir_status bcir_ctl_rollback_token(const uint8_t *BCIR_RESTRICT data, size_t len,
                                    uint8_t out[32]) {
  bcir_ctl_record r;
  bcir_status st = bcir_ctl_decode(data, len, &r);
  if (st != BCIR_OK) return st;
  if (r.hdr.kind != BCIR_CTL_ACTIVATE) return BCIR_ERR_CONTROL;
  token_of(data, len, out);
  return BCIR_OK;
}

/* --- the resident plane --------------------------------------------------------------------- */

static bcir_ctl_outcome outcome(const bcir_ctl_state *s, uint8_t verdict, uint8_t refusal,
                                const bcir_ctl_record *r, bcir_status status) {
  bcir_ctl_outcome o;
  o.verdict = verdict;
  o.refusal = refusal;
  o.kind = r ? r->hdr.kind : 0u;
  o.reserved = 0u;
  o.generation = s->generation;
  o.sequence = r ? r->hdr.sequence : 0u;
  o.status = status;
  return o;
}

bcir_status bcir_ctl_init(bcir_ctl_state *BCIR_RESTRICT state, const uint8_t *BCIR_RESTRICT key,
                          size_t key_len, uint8_t scope, uint64_t subject) {
  if (!state) return BCIR_ERR_CONTROL;
  wipe(state, sizeof(*state));
  if (!key || key_len < BCIR_CTL_KEY_MIN || key_len > BCIR_CTL_KEY_MAX || scope > BCIR_CTL_SCOPE_MAX)
    return BCIR_ERR_CONTROL;
  copy_bytes(state->key, key, key_len);
  state->key_len = (uint32_t)key_len;
  state->scope = scope;
  state->subject = subject;
  return BCIR_OK;
}

static bcir_ctl_lease_entry *find_lease(bcir_ctl_state *s, uint64_t lease_id) {
  for (uint32_t i = 0; i < s->n_leases && i < BCIR_CTL_LEASE_CAPACITY; ++i)
    if (s->leases[i].lease_id == lease_id) return &s->leases[i];
  return (bcir_ctl_lease_entry *)0;
}

static int valid_at(const bcir_ctl_lease_entry *e, uint64_t boundary) {
  return e->issued <= boundary && boundary < e->expiry;
}

/* The leases a grant would keep: lapsed ones are dropped (ids never recur, so a dropped
 * lease's sequence space cannot come back). */
static uint32_t live_leases(const bcir_ctl_state *s) {
  uint32_t live = 0;
  for (uint32_t i = 0; i < s->n_leases && i < BCIR_CTL_LEASE_CAPACITY; ++i)
    live += s->leases[i].expiry > s->boundary;
  return live;
}

/* The state laws a switch must meet -- at submission, and again (totally) when a deferred
 * switch comes due. */
static uint8_t switch_law(const bcir_ctl_state *s, const bcir_ctl_record *r) {
  if (r->hdr.kind == BCIR_CTL_ACTIVATE)
    return eq32(r->body.activate.previous_sha256, s->artifact) ? BCIR_CTL_REFUSAL_NONE
                                                                : BCIR_CTL_REFUSAL_MISMATCH;
  if (r->hdr.kind == BCIR_CTL_ROLLBACK) {
    if (is_zero32(s->previous)) return BCIR_CTL_REFUSAL_NOTHING;
    if (!eq32(r->body.rollback.restore_sha256, s->previous) ||
        !eq32(r->body.rollback.rollback_token, s->token))
      return BCIR_CTL_REFUSAL_MISMATCH;
    return BCIR_CTL_REFUSAL_NONE;
  }
  /* generation: a registry identical to the installed one would invalidate every admitted
   * pack for nothing */
  if (s->has_registry && r->body.generation.map_gen == s->reg_map_gen &&
      r->body.generation.data_gen == s->reg_data_gen &&
      r->body.generation.topo_gen == s->reg_topo_gen &&
      eq32(r->body.generation.registry_digest, s->reg_digest))
    return BCIR_CTL_REFUSAL_MISMATCH;
  return BCIR_CTL_REFUSAL_NONE;
}

/* The pending switch, decoded from its own bytes (verified when it was deferred). */
static int pending_record(const bcir_ctl_state *s, bcir_ctl_record *p) {
  return s->pending_len != 0u && s->pending_len <= BCIR_CTL_RECORD_MAX &&
         bcir_ctl_decode(s->pending, s->pending_len, p) == BCIR_OK;
}

static uint8_t kind_law(const bcir_ctl_state *s, const bcir_ctl_record *r) {
  switch (r->hdr.kind) {
    case BCIR_CTL_LEASE:
      if (r->body.lease.lease_id <= s->last_lease_id) return BCIR_CTL_REFUSAL_DUPLICATE;
      if (r->body.lease.expiry_epoch <= s->boundary) return BCIR_CTL_REFUSAL_EXPIRED;
      if (live_leases(s) >= BCIR_CTL_LEASE_CAPACITY) return BCIR_CTL_REFUSAL_FULL;
      return BCIR_CTL_REFUSAL_NONE;
    case BCIR_CTL_QUIESCE:
      return r->body.quiesce.drain_deadline < s->boundary ? BCIR_CTL_REFUSAL_EXPIRED
                                                          : BCIR_CTL_REFUSAL_NONE;
    case BCIR_CTL_CANCEL: {
      bcir_ctl_record p;
      if (!pending_record(s, &p) || p.hdr.lease != r->hdr.lease ||
          p.hdr.sequence < r->body.cancel.first_sequence ||
          p.hdr.sequence > r->body.cancel.last_sequence)
        return BCIR_CTL_REFUSAL_NOTHING;
      return BCIR_CTL_REFUSAL_NONE;
    }
    default: /* a switch: one pending slot, then the state laws */
      if (s->pending_len != 0u) return BCIR_CTL_REFUSAL_PENDING;
      return switch_law(s, r);
  }
}

static void consume(bcir_ctl_state *s, const bcir_ctl_record *r, bcir_ctl_lease_entry *entry) {
  if (entry)
    entry->last_sequence = r->hdr.sequence;
  else
    s->root_sequence = r->hdr.sequence;
}

static void apply(bcir_ctl_state *s, const bcir_ctl_record *r, const uint8_t *data, size_t len) {
  switch (r->hdr.kind) {
    case BCIR_CTL_LEASE: {
      uint32_t kept = 0, before = s->n_leases < BCIR_CTL_LEASE_CAPACITY ? s->n_leases
                                                                         : BCIR_CTL_LEASE_CAPACITY;
      for (uint32_t i = 0; i < before; ++i)
        if (s->leases[i].expiry > s->boundary) s->leases[kept++] = s->leases[i];
      for (uint32_t i = kept; i < before; ++i) /* a lease that left takes its key with it */
        wipe((uint8_t *)&s->leases[i], sizeof(s->leases[i]));
      s->n_leases = kept;
      if (kept < BCIR_CTL_LEASE_CAPACITY) {  /* held by the `full` law; checked totally */
        bcir_ctl_lease_entry *e = &s->leases[kept];
        e->lease_id = r->body.lease.lease_id;
        e->granted = r->body.lease.granted;
        e->issued = r->body.lease.issued_epoch;
        e->expiry = r->body.lease.expiry_epoch;
        e->holder = r->body.lease.holder;
        e->last_sequence = 0u;
        /* the key the holder MACs with, derived once; a failure leaves it zero, which no MAC
         * matches (fail closed) */
        if (bcir_ctl_lease_key(s->key, s->key_len, e->lease_id, e->key) != BCIR_OK)
          wipe(e->key, sizeof(e->key));
        s->n_leases = kept + 1u;
      }
      s->last_lease_id = r->body.lease.lease_id;
      return;
    }
    case BCIR_CTL_QUIESCE:
      s->draining = 1u;
      s->drain_deadline = r->body.quiesce.drain_deadline;
      return;
    case BCIR_CTL_CANCEL:
      wipe(s->pending, sizeof(s->pending));
      s->pending_len = 0u;
      return;
    case BCIR_CTL_GENERATION:
      s->reg_map_gen = r->body.generation.map_gen;
      s->reg_data_gen = r->body.generation.data_gen;
      s->reg_topo_gen = r->body.generation.topo_gen;
      copy_bytes(s->reg_digest, r->body.generation.registry_digest, 32u);
      s->has_registry = 1u;
      break;
    case BCIR_CTL_ACTIVATE:
      copy_bytes(s->previous, s->artifact, 32u);
      copy_bytes(s->artifact, r->body.activate.artifact_sha256, 32u);
      token_of(data, len, s->token);
      break;
    default: /* rollback */
      copy_bytes(s->artifact, s->previous, 32u);
      copy_bytes(s->previous, ZERO32, 32u);
      copy_bytes(s->token, ZERO32, 32u);
      break;
  }
  s->generation = r->hdr.generation;
  s->draining = 0u;  /* the switch a drain prepared has landed */
}

bcir_ctl_outcome bcir_ctl_submit(bcir_ctl_state *BCIR_RESTRICT state,
                                 const uint8_t *BCIR_RESTRICT data, size_t len) {
  bcir_ctl_state *s = state;
  bcir_ctl_record r;
  bcir_ctl_lease_entry *entry = (bcir_ctl_lease_entry *)0;
  uint8_t refusal;
  int authentic;
  bcir_status st = bcir_ctl_decode(data, len, &r);
  if (st != BCIR_OK) return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MALFORMED, 0, st);
  /* an uninitialized plane holds no key, and no key verifies anything (fail closed) */
  if (s->key_len < BCIR_CTL_KEY_MIN || s->key_len > BCIR_CTL_KEY_MAX)
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MAC, &r, BCIR_ERR_MAC);
  if (r.hdr.lease != 0u) {
    entry = find_lease(s, r.hdr.lease);
    if (!entry) return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_LEASE, &r, BCIR_OK);
    authentic = mac_matches(data, len, entry->key, sizeof(entry->key));
  } else {
    authentic = mac_matches(data, len, s->key, s->key_len);
  }
  if (!authentic) return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MAC, &r, BCIR_ERR_MAC);
  if (r.hdr.scope != s->scope || r.hdr.subject != s->subject)
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_SUBJECT, &r, BCIR_OK);
  if (entry) {
    if (!valid_at(entry, s->boundary))
      return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_EXPIRED, &r, BCIR_OK);
    if ((entry->granted & r.hdr.capability) == 0u)
      return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_CAPABILITY, &r, BCIR_OK);
  }
  if (r.hdr.sequence <= (entry ? entry->last_sequence : s->root_sequence))
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_REPLAY, &r, BCIR_OK);
  if (is_stale(r.hdr.expect, s->generation))
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, &r, BCIR_OK);
  if (r.hdr.expect > s->generation)
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_AHEAD, &r, BCIR_OK);
  refusal = kind_law(s, &r);
  if (refusal != BCIR_CTL_REFUSAL_NONE) return outcome(s, BCIR_CTL_REFUSED, refusal, &r, BCIR_OK);
  if (!is_switch(r.hdr.kind)) {
    if (s->boundary < r.hdr.boundary)
      return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_EARLY, &r, BCIR_OK);
  } else if (s->in_flight != 0u || s->boundary < r.hdr.boundary) {
    /* mid-phase, or before its boundary: held, never applied early */
    consume(s, &r, entry);
    copy_bytes(s->pending, data, len);
    s->pending_len = (uint32_t)len;
    return outcome(s, BCIR_CTL_DEFERRED, BCIR_CTL_REFUSAL_NONE, &r, BCIR_OK);
  }
  consume(s, &r, entry);
  apply(s, &r, data, len);
  return outcome(s, BCIR_CTL_APPLIED, BCIR_CTL_REFUSAL_NONE, &r, BCIR_OK);
}

/* Crossing a boundary: the counter advances, a lapsed drain releases, and a pending switch
 * that is due reaches exactly one reported decision. */
static bcir_ctl_outcome cross(bcir_ctl_state *s) {
  bcir_ctl_record p;
  uint8_t data[BCIR_CTL_RECORD_MAX];
  size_t len;
  uint8_t refusal;
  bcir_ctl_lease_entry *entry;
  s->boundary += 1u;
  if (s->draining && s->boundary > s->drain_deadline) s->draining = 0u;
  if (s->pending_len == 0u) return outcome(s, BCIR_CTL_NONE, BCIR_CTL_REFUSAL_NONE, 0, BCIR_OK);
  if (!pending_record(s, &p)) {  /* unreachable through this API: the slot holds verified bytes */
    bcir_status st = s->pending_len <= BCIR_CTL_RECORD_MAX ? bcir_ctl_verify(s->pending, s->pending_len)
                                                           : BCIR_ERR_TRAILING;
    wipe(s->pending, sizeof(s->pending));
    s->pending_len = 0u;
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MALFORMED, 0, st);
  }
  if (s->boundary < p.hdr.boundary)
    return outcome(s, BCIR_CTL_NONE, BCIR_CTL_REFUSAL_NONE, 0, BCIR_OK);
  len = s->pending_len;
  copy_bytes(data, s->pending, len);
  wipe(s->pending, sizeof(s->pending));
  s->pending_len = 0u;
  entry = find_lease(s, p.hdr.lease);
  if (!entry || !valid_at(entry, s->boundary))  /* its lease lapsed while it waited */
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_EXPIRED, &p, BCIR_OK);
  if (p.hdr.expect != s->generation)  /* held true by the one-slot rule; checked totally */
    return outcome(s, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, &p, BCIR_OK);
  refusal = switch_law(s, &p);
  if (refusal != BCIR_CTL_REFUSAL_NONE) return outcome(s, BCIR_CTL_REFUSED, refusal, &p, BCIR_OK);
  apply(s, &p, data, len);
  return outcome(s, BCIR_CTL_APPLIED, BCIR_CTL_REFUSAL_NONE, &p, BCIR_OK);
}

bcir_ctl_outcome bcir_ctl_enter(bcir_ctl_state *state) {
  if (state->draining)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_DRAINING, 0, BCIR_OK);
  if (state->in_flight == UINT64_MAX)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_EXHAUSTED, 0, BCIR_OK);
  state->in_flight += 1u;
  return outcome(state, BCIR_CTL_NONE, BCIR_CTL_REFUSAL_NONE, 0, BCIR_OK);
}

bcir_ctl_outcome bcir_ctl_leave(bcir_ctl_state *state) {
  if (state->in_flight == 0u)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_IDLE, 0, BCIR_OK);
  if (state->in_flight == 1u && state->boundary == UINT64_MAX)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_EXHAUSTED, 0, BCIR_OK);
  state->in_flight -= 1u;
  if (state->in_flight != 0u) return outcome(state, BCIR_CTL_NONE, BCIR_CTL_REFUSAL_NONE, 0, BCIR_OK);
  return cross(state);
}

bcir_ctl_outcome bcir_ctl_advance(bcir_ctl_state *state) {
  if (state->in_flight != 0u)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_BUSY, 0, BCIR_OK);
  if (state->boundary == UINT64_MAX)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_EXHAUSTED, 0, BCIR_OK);
  return cross(state);
}

/* --- the data plane, by bytes ----------------------------------------------------------------- */

bcir_ctl_outcome bcir_ctl_admit_pack(bcir_ctl_state *BCIR_RESTRICT state,
                                     const uint8_t *BCIR_RESTRICT data, size_t len) {
  bcir_streampack_header hdr;
  bcir_sha256 h;
  uint8_t digest[32];
  bcir_status st;
  if (!state->has_registry)  /* nothing to prove the pack current against */
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, 0, BCIR_ERR_STALE);
  st = bcir_sp_verify_semantic(data, len, 0xFFFFFFFFu, 0xFFFFFFFFu);
  if (st == BCIR_OK) st = bcir_sp_validate(data, len, &hdr);
  if (st != BCIR_OK) return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MALFORMED, 0, st);
  if (hdr.map_gen != state->reg_map_gen || hdr.data_gen != state->reg_data_gen ||
      hdr.topo_gen != state->reg_topo_gen)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, 0, BCIR_ERR_STALE);
  bcir_sha256_init(&h);
  bcir_sha256_update(&h, REGISTRY_TAG, sizeof(REGISTRY_TAG));
  st = bcir_sp_for_each_generation(data, len, feed_generation, &h);
  if (st != BCIR_OK) return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MALFORMED, 0, st);
  bcir_sha256_final(&h, digest);
  if (!eq32(digest, state->reg_digest))  /* a resource moved under unchanged maxima, or no vector */
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, 0, BCIR_ERR_STALE);
  return outcome(state, BCIR_CTL_APPLIED, BCIR_CTL_REFUSAL_NONE, 0, BCIR_OK);
}

bcir_ctl_outcome bcir_ctl_admit_plan(bcir_ctl_state *BCIR_RESTRICT state,
                                     const uint8_t *BCIR_RESTRICT data, size_t len) {
  bcir_sha256 h;
  uint8_t digest[32];
  bcir_status st;
  if (!state->has_registry)
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, 0, BCIR_ERR_STALE);
  st = bcir_ep_verify(data, len);
  if (st != BCIR_OK) return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MALFORMED, 0, st);
  bcir_sha256_init(&h);
  bcir_sha256_update(&h, REGISTRY_TAG, sizeof(REGISTRY_TAG));
  st = bcir_ep_for_each_generation(data, len, feed_generation, &h);
  if (st != BCIR_OK) return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_MALFORMED, 0, st);
  bcir_sha256_final(&h, digest);
  if (!eq32(digest, state->reg_digest))
    return outcome(state, BCIR_CTL_REFUSED, BCIR_CTL_REFUSAL_STALE, 0, BCIR_ERR_STALE);
  return outcome(state, BCIR_CTL_APPLIED, BCIR_CTL_REFUSAL_NONE, 0, BCIR_OK);
}

/* --- the resident state digest ------------------------------------------------------------ */

static void put8(bcir_sha256 *h, uint8_t v) { bcir_sha256_update(h, &v, 1u); }
static void put32(bcir_sha256 *h, uint32_t v) {
  uint8_t b[4];
  wr32(b, v);
  bcir_sha256_update(h, b, sizeof(b));
}
static void put64(bcir_sha256 *h, uint64_t v) {
  uint8_t b[8];
  wr64(b, v);
  bcir_sha256_update(h, b, sizeof(b));
}

void bcir_ctl_state_digest(const bcir_ctl_state *BCIR_RESTRICT state, uint8_t out[32]) {
  const bcir_ctl_state *s = state;
  uint32_t n_leases = s->n_leases <= BCIR_CTL_LEASE_CAPACITY ? s->n_leases : BCIR_CTL_LEASE_CAPACITY;
  uint32_t pending = s->pending_len <= BCIR_CTL_RECORD_MAX ? s->pending_len : BCIR_CTL_RECORD_MAX;
  bcir_sha256 h;
  bcir_sha256_init(&h);
  bcir_sha256_update(&h, STATE_TAG, sizeof(STATE_TAG));
  put32(&h, s->generation);
  put64(&h, s->boundary);
  put64(&h, s->in_flight);
  put8(&h, s->draining ? 1u : 0u);
  put64(&h, s->drain_deadline);
  bcir_sha256_update(&h, s->artifact, 32u);
  bcir_sha256_update(&h, s->previous, 32u);
  bcir_sha256_update(&h, s->token, 32u);
  put8(&h, s->has_registry ? 1u : 0u);
  put32(&h, s->has_registry ? s->reg_map_gen : 0u);
  put32(&h, s->has_registry ? s->reg_data_gen : 0u);
  put32(&h, s->has_registry ? s->reg_topo_gen : 0u);
  bcir_sha256_update(&h, s->has_registry ? s->reg_digest : ZERO32, 32u);
  put64(&h, s->root_sequence);
  put64(&h, s->last_lease_id);
  put8(&h, s->scope);
  put64(&h, s->subject);
  put32(&h, n_leases);
  for (uint32_t i = 0; i < n_leases; ++i) {
    const bcir_ctl_lease_entry *e = &s->leases[i];
    put64(&h, e->lease_id);
    put64(&h, e->granted);
    put64(&h, e->issued);
    put64(&h, e->expiry);
    put64(&h, e->holder);
    put64(&h, e->last_sequence);
  }
  put32(&h, pending);
  bcir_sha256_update(&h, s->pending, pending);
  bcir_sha256_final(&h, out);
}
