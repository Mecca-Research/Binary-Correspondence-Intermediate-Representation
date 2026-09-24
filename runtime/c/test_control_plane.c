/*===- test_control_plane.c - ControlRecordV1 C harness (parity + plane traces) -===
 *
 * Drives the freestanding control plane (bcir_control_plane.h) three ways, so every G14 gate
 * runs on the C rail from one harness:
 *
 *   --dump <records> --key <hex>   decode each u32le-length-framed record and print it field by
 *       field ("record kind=... mac=<hex> macok=0|1"), or "refused status=<BCIR_*>" -- the
 *       Python side rebuilds each record from the dump and must re-encode it byte for byte
 *       (Python encode -> C decode -> Python re-encode), and grades each malformed variant's
 *       status against the one the specification declares. `macok` is the keyed check under
 *       the root key (a grant) or the lease's key (bcir_ctl_lease_key).
 *   --script <script> [--via-ring]  run scenarios of plane operations and print one trace line
 *       per operation: "<scenario> <op> <verdict> <refusal> k=<kind> s=<sequence>
 *       g=<generation> <status|-> <state digest>" -- identical, line for line, to the Python
 *       rail's (bcir/tests/control_fixtures.py::trace_line). With --via-ring every submitted
 *       record first crosses a live control ring (bcir_ring.h: BACKPRESSURE, 256-byte slots):
 *       written by a producer endpoint, read back by a consumer endpoint, then submitted -- the
 *       trace must not change (G15, the ring is a transport, never a decision). A record the
 *       ring cannot carry prints "RINGFAIL", which no Python trace line matches.
 *   --api                          the API's own fail-closed laws, which no record can reach:
 *       prints OK when every one holds.
 *
 * Script format (little-endian): "BCTS" u32 n_scenarios; per scenario u8 key_len, key,
 * u8 scope, u64 subject, u32 n_ops; per op u8 code, u32 length, data. Codes: 1 submit,
 * 2 enter, 3 leave, 4 advance, 5 admit_pack, 6 admit_plan; 7 / 8 set the boundary counter /
 * the in-flight count to a u64 (harness-only, so the `exhausted` refusals have a witness).
 *
 * The exit status is 0 when the input was well-formed (the verdicts are data, graded by the
 * caller) and --api held; 2 on a usage or input error; 1 when --api found a violation.
 * The harness uses libc (it is a test); the plane stays freestanding.
 *===----------------------------------------------------------------------===*/
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_control_plane.h"
#include "bcir_ring.h"

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
    default:                   return "BCIR_ERR_UNKNOWN";
  }
}

static void hex(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", (unsigned)p[i]);
}

static uint8_t *read_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  uint8_t *buf = NULL;
  size_t cap = 0, len = 0;
  if (!f) return NULL;
  for (;;) {
    if (len == cap) {
      size_t next = cap ? cap * 2 : 4096;
      uint8_t *grown = (uint8_t *)realloc(buf, next);
      if (!grown) { free(buf); fclose(f); return NULL; }
      buf = grown; cap = next;
    }
    size_t got = fread(buf + len, 1, cap - len, f);
    len += got;
    if (got == 0) break;
  }
  fclose(f);
  *out_len = len;
  return buf;
}

static uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t le64(const uint8_t *p) { return (uint64_t)le32(p) | ((uint64_t)le32(p + 4) << 32); }

/* --- --dump ----------------------------------------------------------------------------- */

static void dump_record(const uint8_t *data, size_t len, const uint8_t *root, size_t root_len) {
  bcir_ctl_record r;
  bcir_status st = bcir_ctl_decode(data, len, &r);
  if (st != BCIR_OK) { printf("refused status=%s\n", status_name(st)); return; }
  const bcir_ctl_header *h = &r.hdr;
  printf("record kind=%u scope=%u reason=%u generation=%" PRIu32 " expect=%" PRIu32
         " capability=%" PRIu64 " boundary=%" PRIu64 " sequence=%" PRIu64 " lease=%" PRIu64
         " subject=%" PRIu64,
         (unsigned)h->kind, (unsigned)h->scope, (unsigned)h->reason, h->generation, h->expect,
         h->capability, h->boundary, h->sequence, h->lease, h->subject);
  switch (h->kind) {
    case BCIR_CTL_LEASE:
      printf(" lease_id=%" PRIu64 " granted=%" PRIu64 " issued=%" PRIu64 " expiry=%" PRIu64
             " holder=%" PRIu64,
             r.body.lease.lease_id, r.body.lease.granted, r.body.lease.issued_epoch,
             r.body.lease.expiry_epoch, r.body.lease.holder);
      break;
    case BCIR_CTL_GENERATION:
      printf(" map_gen=%" PRIu32 " data_gen=%" PRIu32 " topo_gen=%" PRIu32 " registry=",
             r.body.generation.map_gen, r.body.generation.data_gen, r.body.generation.topo_gen);
      hex(r.body.generation.registry_digest, 32);
      break;
    case BCIR_CTL_QUIESCE:
      printf(" deadline=%" PRIu64, r.body.quiesce.drain_deadline);
      break;
    case BCIR_CTL_ACTIVATE:
      printf(" artifact="); hex(r.body.activate.artifact_sha256, 32);
      printf(" previous="); hex(r.body.activate.previous_sha256, 32);
      break;
    case BCIR_CTL_ROLLBACK:
      printf(" restore="); hex(r.body.rollback.restore_sha256, 32);
      printf(" token="); hex(r.body.rollback.rollback_token, 32);
      break;
    default:
      printf(" first=%" PRIu64 " last=%" PRIu64, r.body.cancel.first_sequence,
             r.body.cancel.last_sequence);
      break;
  }
  printf(" mac="); hex(r.mac, 32);
  {
    uint8_t lease_key[32];
    bcir_status keyed;
    if (h->lease == 0u) {
      keyed = bcir_ctl_check_mac(data, len, root, root_len);
    } else if (bcir_ctl_lease_key(root, root_len, h->lease, lease_key) == BCIR_OK) {
      keyed = bcir_ctl_check_mac(data, len, lease_key, sizeof(lease_key));
    } else {
      keyed = BCIR_ERR_MAC;
    }
    printf(" macok=%d\n", keyed == BCIR_OK);
  }
}

static int run_dump(const char *path, const char *key_hex) {
  uint8_t root[BCIR_CTL_KEY_MAX];
  size_t root_len = strlen(key_hex) / 2u, len = 0, at = 0;
  if (strlen(key_hex) % 2u != 0u || root_len > sizeof(root)) {
    fprintf(stderr, "bad --key\n");
    return 2;
  }
  for (size_t i = 0; i < root_len; ++i) {
    unsigned v;
    if (sscanf(key_hex + 2u * i, "%2x", &v) != 1) { fprintf(stderr, "bad --key\n"); return 2; }
    root[i] = (uint8_t)v;
  }
  uint8_t *buf = read_file(path, &len);
  if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 2; }
  while (at < len) {
    if (len - at < 4u) { fprintf(stderr, "truncated frame\n"); free(buf); return 2; }
    size_t n = le32(buf + at);
    at += 4u;
    if (n > len - at) { fprintf(stderr, "truncated frame\n"); free(buf); return 2; }
    /* each record in a buffer of its own exact size, so an out-of-bounds read is a real one
     * (ASan sees it) rather than a read of the next frame */
    uint8_t *one = (uint8_t *)malloc(n ? n : 1u);
    if (!one) { free(buf); return 2; }
    if (n) memcpy(one, buf + at, n);
    dump_record(n ? one : NULL, n, root, root_len);
    free(one);
    at += n;
  }
  free(buf);
  return 0;
}

/* --- --script ---------------------------------------------------------------------------- */

enum { OP_SUBMIT = 1, OP_ENTER, OP_LEAVE, OP_ADVANCE, OP_ADMIT_PACK, OP_ADMIT_PLAN,
       OP_POKE_BOUNDARY, OP_POKE_IN_FLIGHT };

/* The control ring a --via-ring scenario's records cross (8-byte aligned, as the ring needs). A
 * control ring's slot carries every legal record, the declared bound included. */
_Static_assert(BCIR_RING_CONTROL_SLOT_MIN - BCIR_RING_SLOT_HEADER >= BCIR_CTL_RECORD_MAX,
               "a control ring's slot must carry every legal ControlRecordV1");
_Static_assert(BCIR_RING_CONTROL_SLOT_MIN - 64u - BCIR_RING_SLOT_HEADER < BCIR_CTL_RECORD_MAX,
               "BCIR_RING_CONTROL_SLOT_MIN is the smallest cache-line multiple that carries it");
#define VIA_RING_SLOT  BCIR_RING_CONTROL_SLOT_MIN
#define VIA_RING_SLOTS 4u
static _Alignas(64) uint8_t via_region[BCIR_RING_HEADER_SIZE + VIA_RING_SLOTS * VIA_RING_SLOT];
static bcir_ring_producer via_producer;
static bcir_ring_consumer via_consumer;

static int via_ring_open(uint32_t scenario) {
  bcir_ring_geometry g = {0};
  g.policy = BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_CONTROL;
  g.slot_size = VIA_RING_SLOT;
  g.slot_count = VIA_RING_SLOTS;
  g.ring_id = (uint64_t)scenario + 1u;
  if (bcir_ring_format(via_region, sizeof via_region, &g) != BCIR_OK) return 0;
  bcir_ring_producer_init(&via_producer, via_region, sizeof via_region);
  bcir_ring_consumer_init(&via_consumer, via_region, sizeof via_region);
  return bcir_ring_producer_attach(&via_producer, 0).verdict == BCIR_RING_OK &&
         bcir_ring_consumer_attach(&via_consumer, 0).verdict == BCIR_RING_OK;
}

/* Carry one record across the ring: write it, read it back into `out`; 0 if the ring refused. */
static int via_ring_carry(const uint8_t *data, size_t n, uint8_t *out, size_t *out_len) {
  bcir_ring_outcome w = bcir_ring_publish(&via_producer, data, n);
  if (w.verdict != BCIR_RING_OK) return 0;
  bcir_ring_outcome r = bcir_ring_consume(&via_consumer, out, VIA_RING_SLOT);
  if (r.verdict != BCIR_RING_DELIVERED || r.position != w.position) return 0;
  *out_len = r.length;
  return 1;
}

static int run_script(const char *path, int via_ring) {
  size_t len = 0, at = 8;
  uint8_t *buf = read_file(path, &len);
  if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 2; }
  if (len < 8u || memcmp(buf, "BCTS", 4) != 0) { fprintf(stderr, "not a script\n"); free(buf); return 2; }
  uint32_t n_scenarios = le32(buf + 4);
  static bcir_ctl_state state;  /* large: keep it off the stack */
  for (uint32_t scn = 0; scn < n_scenarios; ++scn) {
    if (len - at < 1u) goto bad;
    size_t key_len = buf[at++];
    if (len - at < key_len + 13u) goto bad;
    const uint8_t *key = buf + at;
    at += key_len;
    uint8_t scope = buf[at++];
    uint64_t subject = le64(buf + at);
    at += 8u;
    uint32_t n_ops = le32(buf + at);
    at += 4u;
    if (bcir_ctl_init(&state, key, key_len, scope, subject) != BCIR_OK) {
      fprintf(stderr, "scenario %u: bcir_ctl_init refused its key/scope\n", scn);
      free(buf);
      return 2;
    }
    if (via_ring && !via_ring_open(scn)) {
      fprintf(stderr, "scenario %u: the control ring did not open\n", scn);
      free(buf);
      return 2;
    }
    for (uint32_t op = 0; op < n_ops; ++op) {
      if (len - at < 5u) goto bad;
      uint8_t code = buf[at];
      size_t n = le32(buf + at + 1);
      at += 5u;
      if (n > len - at) goto bad;
      uint8_t *data = (uint8_t *)malloc(n ? n : 1u);  /* exact-size copy, as in --dump */
      if (!data) { free(buf); return 2; }
      if (n) memcpy(data, buf + at, n);
      at += n;
      bcir_ctl_outcome o;
      switch (code) {
        case OP_SUBMIT:
          if (via_ring) {
            uint8_t carried[VIA_RING_SLOT];
            size_t carried_len = 0;
            if (!via_ring_carry(data, n, carried, &carried_len)) {
              printf("%u %u RINGFAIL\n", scn, op);
              free(data);
              continue;
            }
            o = bcir_ctl_submit(&state, carried_len ? carried : NULL, carried_len);
          } else {
            o = bcir_ctl_submit(&state, n ? data : NULL, n);
          }
          break;
        case OP_ENTER: o = bcir_ctl_enter(&state); break;
        case OP_LEAVE: o = bcir_ctl_leave(&state); break;
        case OP_ADVANCE: o = bcir_ctl_advance(&state); break;
        case OP_ADMIT_PACK: o = bcir_ctl_admit_pack(&state, n ? data : NULL, n); break;
        case OP_ADMIT_PLAN: o = bcir_ctl_admit_plan(&state, n ? data : NULL, n); break;
        case OP_POKE_BOUNDARY:
        case OP_POKE_IN_FLIGHT:
          if (n != 8u) { free(data); goto bad; }
          if (code == OP_POKE_BOUNDARY) state.boundary = le64(data);
          else state.in_flight = le64(data);
          o.verdict = BCIR_CTL_NONE; o.refusal = BCIR_CTL_REFUSAL_NONE; o.kind = 0;
          o.reserved = 0; o.generation = state.generation; o.sequence = 0; o.status = BCIR_OK;
          break;
        default:
          free(data);
          goto bad;
      }
      free(data);
      uint8_t digest[32];
      bcir_ctl_state_digest(&state, digest);
      printf("%u %u %u %u k=%u s=%" PRIu64 " g=%" PRIu32 " %s ", scn, op, (unsigned)o.verdict,
             (unsigned)o.refusal, (unsigned)o.kind, o.sequence, o.generation,
             code == OP_SUBMIT ? status_name(o.status) : "-");
      hex(digest, 32);
      printf("\n");
    }
  }
  if (at != len) goto bad;
  free(buf);
  return 0;
bad:
  fprintf(stderr, "malformed script\n");
  free(buf);
  return 2;
}

/* --- --api: the fail-closed laws no record can reach ----------------------------------------- */

static int failures = 0;
#define CHECK(cond, name) do { if (!(cond)) { printf("FAIL %s\n", name); ++failures; } } while (0)

static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }

/* A lease grant (140 bytes) for plane (module, 7), MACed under `key` and CRC-sealed. */
static void grant_record(uint8_t out[140], const uint8_t *key, size_t key_len) {
  memset(out, 0, 140);
  memcpy(out, "BCTL", 4);
  out[4] = 1;                    /* version */
  out[8] = BCIR_CTL_LEASE;       /* kind; scope 0 (module), reason 0 */
  put32(out + 12, 40);           /* body_len */
  put64(out + 24, BCIR_CTL_CAP(BCIR_CTL_LEASE));
  put64(out + 40, 1);            /* sequence */
  put64(out + 56, 7);            /* subject */
  put64(out + 64, 1);            /* lease_id */
  put64(out + 72, BCIR_CTL_CAP_GRANTABLE);
  put64(out + 88, 100);          /* expiry */
  put64(out + 96, 42);           /* holder */
  bcir_hmac_sha256(key, key_len, out, 104, out + 104);
  put32(out + 136, bcir_crc32(out, 136));
}

static int run_api(void) {
  static bcir_ctl_state state;
  const uint8_t root[32] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
                            0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
                            0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};
  uint8_t out[32], again[32], record[140], empty_key_record[140];
  bcir_ctl_outcome o;

  /* the body table: exactly (1, 1..6) */
  CHECK(bcir_ctl_body_bytes(1, 4) == 64u && bcir_ctl_body_bytes(1, 1) == 40u, "body table");
  CHECK(bcir_ctl_body_bytes(0, 1) == 0u && bcir_ctl_body_bytes(2, 1) == 0u, "body table version");
  CHECK(bcir_ctl_body_bytes(1, 0) == 0u && bcir_ctl_body_bytes(1, 7) == 0u, "body table kind");

  /* init refuses a short, a long or a missing key and an unknown scope, leaving the state zeroed */
  CHECK(bcir_ctl_init(&state, root, 15, 0, 7) == BCIR_ERR_CONTROL, "init short key");
  CHECK(bcir_ctl_init(&state, root, 65, 0, 7) == BCIR_ERR_CONTROL, "init long key");
  CHECK(bcir_ctl_init(&state, NULL, 16, 0, 7) == BCIR_ERR_CONTROL, "init missing key");
  CHECK(bcir_ctl_init(&state, root, 32, 5, 7) == BCIR_ERR_CONTROL, "init unknown scope");
  CHECK(state.key_len == 0u && state.subject == 0u, "refused init leaves the state zeroed");

  /* a plane whose init was refused (or never ran) holds no key: a record MACed under the empty
   * key -- the key a zeroed state would otherwise hold -- is refused, not applied */
  grant_record(empty_key_record, NULL, 0);
  o = bcir_ctl_submit(&state, empty_key_record, sizeof(empty_key_record));
  CHECK(o.verdict == BCIR_CTL_REFUSED && o.refusal == BCIR_CTL_REFUSAL_MAC, "zeroed plane fails closed");
  CHECK(state.root_sequence == 0u && state.n_leases == 0u, "zeroed plane unchanged");

  /* an initialized plane: the empty-key record is refused, the root's is applied */
  CHECK(bcir_ctl_init(&state, root, 32, 0, 7) == BCIR_OK, "init");
  o = bcir_ctl_submit(&state, empty_key_record, sizeof(empty_key_record));
  CHECK(o.verdict == BCIR_CTL_REFUSED && o.refusal == BCIR_CTL_REFUSAL_MAC &&
            o.status == BCIR_ERR_MAC, "foreign (empty) key refused");
  grant_record(record, root, 32);
  o = bcir_ctl_submit(&state, record, sizeof(record));
  CHECK(o.verdict == BCIR_CTL_APPLIED && state.n_leases == 1u, "root grant applied");

  /* the keyed check: an invalid (NULL, nonzero) key is refused without being read */
  CHECK(bcir_ctl_check_mac(record, sizeof(record), root, 32) == BCIR_OK, "check_mac root");
  CHECK(bcir_ctl_check_mac(record, sizeof(record), NULL, 32) == BCIR_ERR_MAC, "check_mac NULL key");
  CHECK(bcir_ctl_check_mac(record, sizeof(record) - 1u, root, 32) == BCIR_ERR_TRUNCATED,
        "check_mac verifies first");

  /* lease keys: deterministic, id-scoped; a missing root is refused and the output zeroed */
  CHECK(bcir_ctl_lease_key(root, 32, 1, out) == BCIR_OK, "lease key");
  CHECK(bcir_ctl_lease_key(root, 32, 1, again) == BCIR_OK && memcmp(out, again, 32) == 0,
        "lease key deterministic");
  CHECK(bcir_ctl_lease_key(root, 32, 2, again) == BCIR_OK && memcmp(out, again, 32) != 0,
        "lease key scoped by id");
  memset(out, 0xAA, sizeof(out));
  CHECK(bcir_ctl_lease_key(NULL, 32, 1, out) == BCIR_ERR_CONTROL, "lease key NULL root");
  {
    uint8_t acc = 0;
    for (int i = 0; i < 32; ++i) acc |= out[i];
    CHECK(acc == 0, "lease key NULL root zeroes the output");
  }

  /* the registry digest: strictly ascending RIDs; the empty vector is the tag's digest */
  {
    const bcir_generation_view ok[2] = {{1, 2, 3}, {4, 5, 6}};
    const bcir_generation_view dup[2] = {{4, 2, 3}, {4, 5, 6}};
    const bcir_generation_view down[2] = {{5, 2, 3}, {4, 5, 6}};
    static const uint8_t tag[] = "BCTL/registry/v1";
    CHECK(bcir_ctl_registry_digest(ok, 2, out) == BCIR_OK, "registry digest");
    CHECK(bcir_ctl_registry_digest(dup, 2, out) == BCIR_ERR_GENERATION, "registry duplicate rid");
    CHECK(bcir_ctl_registry_digest(down, 2, out) == BCIR_ERR_GENERATION, "registry descending");
    CHECK(bcir_ctl_registry_digest(NULL, 2, out) == BCIR_ERR_GENERATION, "registry NULL vector");
    CHECK(bcir_ctl_registry_digest(NULL, 0, out) == BCIR_OK, "registry empty");
    bcir_sha256_digest(tag, sizeof(tag), again);
    CHECK(memcmp(out, again, 32) == 0, "empty registry is the tag's digest");
  }

  /* the rollback token exists only for an activation */
  CHECK(bcir_ctl_rollback_token(record, sizeof(record), out) == BCIR_ERR_CONTROL,
        "token of a non-activation");

  /* the boundaries on a fresh plane */
  CHECK(bcir_ctl_init(&state, root, 32, 0, 7) == BCIR_OK, "re-init");
  o = bcir_ctl_leave(&state);
  CHECK(o.verdict == BCIR_CTL_REFUSED && o.refusal == BCIR_CTL_REFUSAL_IDLE, "leave idle");
  o = bcir_ctl_admit_pack(&state, record, sizeof(record));
  CHECK(o.verdict == BCIR_CTL_REFUSED && o.status == BCIR_ERR_STALE, "admit without registry");

  if (failures) return 1;
  printf("OK\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 5 && strcmp(argv[1], "--dump") == 0 && strcmp(argv[3], "--key") == 0)
    return run_dump(argv[2], argv[4]);
  if (argc == 3 && strcmp(argv[1], "--script") == 0) return run_script(argv[2], 0);
  if (argc == 4 && strcmp(argv[1], "--script") == 0 && strcmp(argv[3], "--via-ring") == 0)
    return run_script(argv[2], 1);
  if (argc == 2 && strcmp(argv[1], "--api") == 0) return run_api();
  fprintf(stderr,
          "usage: test_control_plane --dump <records> --key <hex> | --script <script> [--via-ring]"
          " | --api\n");
  return 2;
}
