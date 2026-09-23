/*===- test_ring.c - the live SPSC ring's C harness (parity, stress, peer death) -===
 *
 * Drives the freestanding ring (bcir_ring.h), the telemetry envelope and its intake
 * (bcir_telemetry_envelope.h) and the generated signal table, so every G15 gate runs on the C
 * rail from one harness:
 *
 *   --script <script>   run scripted scenarios and print one trace line per operation --
 *                       identical, line for line, to bcir/tests/ring_fixtures.py::run_python.
 *   --signals           every generated table row as hex (the Python table's encode_row).
 *   --envelopes <file>  decode each u32le-framed envelope: its fields and the C re-encoding, or
 *                       "refused status=<BCIR_*>" (Python encode -> C decode -> both re-encode).
 *   --api               the API's fail-closed laws no script reaches; prints OK.
 *   --stress <bp|ow> <records> <seed>
 *                       one producer thread and one consumer thread over a shared region;
 *                       every delivered record is decoded and checked against the payload its
 *                       position was written with, the accounting is reconciled exactly and the
 *                       continuity reported by the intake is checked against the ring's count.
 *   --procs <kill-producer|kill-consumer> <bp|ow> <records> <seed>
 *                       the same across two PROCESSES over a MAP_SHARED mapping; the supervisor
 *                       SIGKILLs the named peer mid-stream, reaps it (death proved) and starts a
 *                       successor that takes the end over; nothing torn, nothing unaccounted.
 *   --bench <records> <slot_size> <slots> <repeats>
 *                       records per second through a backpressure ring (two threads) against a
 *                       memcpy floor of the same bytes (one thread) -- the G15 ratio row.
 *
 * Script format (little-endian): "BRTS" u32 n_scenarios; per scenario u8 policy, u8 payload,
 * u32 slot_size, u32 slot_count, u64 ring_id, u64 origin, u32 live_generation, u32 n_ops; per op
 * u8 code, u8 actor, u64 arg, u32 length, data. The harness formats the region, then runs the
 * ops (codes below). Exit status: 0 when the input was well-formed and every mode's own laws
 * held (verdicts are data, graded by the caller); 1 when --api/--stress/--procs found a
 * violation; 2 on a usage or input error; 3 when a mode is unavailable on this host (no POSIX
 * threads/processes). The harness uses libc and POSIX (it is a test); the ring stays
 * freestanding.
 *===----------------------------------------------------------------------===*/
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1
#endif

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_ring.h"
#include "bcir_telemetry_envelope.h"

#if defined(__unix__) || defined(__APPLE__)
#define RING_HAS_POSIX 1
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#else
#define RING_HAS_POSIX 0
#endif

static const char *status_name(bcir_status s) {
  switch (s) {
    case BCIR_OK: return "BCIR_OK";
    case BCIR_ERR_TRUNCATED: return "BCIR_ERR_TRUNCATED";
    case BCIR_ERR_MAGIC: return "BCIR_ERR_MAGIC";
    case BCIR_ERR_VERSION: return "BCIR_ERR_VERSION";
    case BCIR_ERR_CRC: return "BCIR_ERR_CRC";
    case BCIR_ERR_NOSPACE: return "BCIR_ERR_NOSPACE";
    case BCIR_ERR_LANE: return "BCIR_ERR_LANE";
    case BCIR_ERR_WIDTH: return "BCIR_ERR_WIDTH";
    case BCIR_ERR_DISPATCH: return "BCIR_ERR_DISPATCH";
    case BCIR_ERR_PROVENANCE: return "BCIR_ERR_PROVENANCE";
    case BCIR_ERR_STALE: return "BCIR_ERR_STALE";
    case BCIR_ERR_OVERFLOW: return "BCIR_ERR_OVERFLOW";
    case BCIR_ERR_TRAILING: return "BCIR_ERR_TRAILING";
    case BCIR_ERR_RESERVED: return "BCIR_ERR_RESERVED";
    case BCIR_ERR_UTF8: return "BCIR_ERR_UTF8";
    case BCIR_ERR_GENERATION: return "BCIR_ERR_GENERATION";
    case BCIR_ERR_PLAN: return "BCIR_ERR_PLAN";
    case BCIR_ERR_CONTROL: return "BCIR_ERR_CONTROL";
    case BCIR_ERR_MAC: return "BCIR_ERR_MAC";
    case BCIR_ERR_TELEMETRY: return "BCIR_ERR_TELEMETRY";
    case BCIR_ERR_RING: return "BCIR_ERR_RING";
    case BCIR_ERR_FULL: return "BCIR_ERR_FULL";
    case BCIR_ERR_BUSY: return "BCIR_ERR_BUSY";
  }
  return "BCIR_ERR_UNKNOWN";
}

static const char *const VERDICTS[] = {"ok", "open", "delivered", "empty", "lost", "stale", "refused"};
static const char *const INTAKE_VERDICTS[] = {"accepted", "skipped", "refused"};
static const char *const INTAKE_REASONS[] = {"none", "malformed", "streams", "unknown", "stale", "ahead"};
static const char *const CLASSES[] = {"none", "first", "next", "gap", "duplicate", "reorder"};

static uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t le64(const uint8_t *p) { return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32; }

static uint8_t *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long size = ftell(f);
  if (size < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
  uint8_t *buf = (uint8_t *)malloc((size_t)size + 1u);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(buf); return NULL; }
  fclose(f);
  *len = (size_t)size;
  return buf;
}

static void *aligned_region(size_t size) {
  size_t rounded = (size + 63u) & ~(size_t)63u;
  void *p = NULL;
#if RING_HAS_POSIX
  if (posix_memalign(&p, 64, rounded ? rounded : 64u) != 0) return NULL;
#else
  p = aligned_alloc(64, rounded ? rounded : 64u);
#endif
  return p;
}

/* --- --script ------------------------------------------------------------------------------ */

enum {
  OP_P_ATTACH = 1, OP_P_DETACH, OP_P_WRITE, OP_P_OPEN, OP_P_FILL, OP_P_CLOSE, OP_P_BEAT, OP_P_DROP,
  OP_C_ATTACH, OP_C_DETACH, OP_C_READ, OP_C_OPEN, OP_C_COPY, OP_C_CLOSE, OP_C_BEAT, OP_C_DROP,
  OP_POKE, OP_INTAKE, OP_LIVE, OP_CHECK
};
#define ACTORS 4u
#define PAYLOAD_MAX BCIR_RING_SLOT_SIZE_MAX

static uint8_t last_payload[ACTORS][PAYLOAD_MAX];
static size_t last_length[ACTORS];
static uint8_t read_buffer[PAYLOAD_MAX];

static void print_ring(uint32_t scn, uint32_t op, bcir_ring_outcome o, const uint8_t *payload,
                       const uint8_t *region, uint64_t region_size) {
  uint32_t pcrc = 0;
  size_t plen = 0;
  if (o.verdict == BCIR_RING_DELIVERED) {
    plen = o.length;
    pcrc = bcir_crc32(payload, plen);
  }
  printf("%u %u R %s %s %" PRIu64 " %" PRIu64 " %u %zu %08x %08x\n", scn, op, VERDICTS[o.verdict],
         status_name(o.status), o.position, o.count, o.epoch, plen, pcrc,
         bcir_crc32(region, (size_t)region_size));
}

static int run_script(const char *path) {
  size_t len = 0, at = 8;
  uint8_t *buf = read_file(path, &len);
  if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 2; }
  if (len < 8u || memcmp(buf, "BRTS", 4) != 0) { fprintf(stderr, "not a script\n"); free(buf); return 2; }
  uint32_t n_scenarios = le32(buf + 4);
  static bcir_ring_producer producers[ACTORS];
  static bcir_ring_consumer consumers[ACTORS];
  static bcir_tev_intake intake;
  for (uint32_t scn = 0; scn < n_scenarios; ++scn) {
    if (len - at < 34u) goto bad;
    bcir_ring_geometry g = {0};
    g.policy = buf[at];
    g.payload = buf[at + 1];
    g.slot_size = le32(buf + at + 2);
    g.slot_count = le32(buf + at + 6);
    g.ring_id = le64(buf + at + 10);
    g.origin = le64(buf + at + 18);
    uint32_t live = le32(buf + at + 26);
    uint32_t n_ops = le32(buf + at + 30);
    at += 34u;
    uint64_t region_size = bcir_ring_region_size(&g);
    if (region_size == 0u || region_size > (uint64_t)1 << 28) goto bad;
    uint8_t *region = (uint8_t *)aligned_region((size_t)region_size);
    if (!region) { free(buf); return 2; }
    if (bcir_ring_format(region, (size_t)region_size, &g) != BCIR_OK) { free(region); goto bad; }
    for (unsigned a = 0; a < ACTORS; a++) {
      bcir_ring_producer_init(&producers[a], region, (size_t)region_size);
      bcir_ring_consumer_init(&consumers[a], region, (size_t)region_size);
      last_length[a] = 0;
    }
    bcir_tev_intake_init(&intake, live);
    for (uint32_t op = 0; op < n_ops; ++op) {
      if (len - at < 14u) { free(region); goto bad; }
      uint8_t code = buf[at];
      uint8_t actor = buf[at + 1];
      uint64_t arg = le64(buf + at + 2);
      size_t n = le32(buf + at + 10);
      at += 14u;
      if (n > len - at || actor >= ACTORS) { free(region); goto bad; }
      const uint8_t *data = buf + at;
      at += n;
      bcir_ring_producer *p = &producers[actor];
      bcir_ring_consumer *c = &consumers[actor];
      bcir_ring_outcome o;
      int ring_op = 1;
      switch (code) {
        case OP_P_ATTACH: o = bcir_ring_producer_attach(p, (uint32_t)arg); break;
        case OP_P_DETACH: o = bcir_ring_producer_detach(p); break;
        case OP_P_WRITE: o = bcir_ring_publish(p, n ? data : NULL, n); break;
        case OP_P_OPEN: o = bcir_ring_write_open(p, (size_t)arg); break;
        case OP_P_FILL: o = bcir_ring_write_fill(p, (size_t)arg, n ? data : NULL, n); break;
        case OP_P_CLOSE: o = bcir_ring_write_close(p); break;
        case OP_P_BEAT: o = bcir_ring_producer_beat(p); break;
        case OP_P_DROP:
          bcir_ring_producer_init(p, region, (size_t)region_size);
          o.verdict = BCIR_RING_OK; o.status = BCIR_OK; o.position = 0; o.count = 0; o.epoch = 0;
          o.length = 0;
          break;
        case OP_C_ATTACH: o = bcir_ring_consumer_attach(c, (uint32_t)arg); break;
        case OP_C_DETACH: o = bcir_ring_consumer_detach(c); break;
        case OP_C_READ: o = bcir_ring_consume(c, read_buffer, sizeof read_buffer); break;
        case OP_C_OPEN: o = bcir_ring_read_open(c); break;
        case OP_C_COPY: o = bcir_ring_read_copy(c, read_buffer, sizeof read_buffer); break;
        case OP_C_CLOSE: o = bcir_ring_read_close(c, read_buffer, sizeof read_buffer); break;
        case OP_C_BEAT: o = bcir_ring_consumer_beat(c); break;
        case OP_C_DROP:
          bcir_ring_consumer_init(c, region, (size_t)region_size);
          o.verdict = BCIR_RING_OK; o.status = BCIR_OK; o.position = 0; o.count = 0; o.epoch = 0;
          o.length = 0;
          break;
        default: ring_op = 0; o.verdict = BCIR_RING_OK; o.status = BCIR_OK; break;
      }
      if (ring_op) {
        if (o.verdict == BCIR_RING_DELIVERED) {
          memcpy(last_payload[actor], read_buffer, o.length);
          last_length[actor] = o.length;
        }
        print_ring(scn, op, o, read_buffer, region, region_size);
        continue;
      }
      if (code == OP_POKE) {
        if (arg > region_size || n > region_size - arg) { free(region); goto bad; }
        memcpy(region + arg, data, n);
        printf("%u %u P %08x\n", scn, op, bcir_crc32(region, (size_t)region_size));
      } else if (code == OP_INTAKE) {
        const uint8_t *record = n ? data : last_payload[actor];
        size_t record_len = n ? n : last_length[actor];
        bcir_tev_outcome t = bcir_tev_intake_admit(&intake, record, record_len, NULL);
        printf("%u %u I %s %s %s %s\n", scn, op, INTAKE_VERDICTS[t.verdict],
               INTAKE_REASONS[t.reason], status_name(t.status), CLASSES[t.classification]);
      } else if (code == OP_LIVE) {
        bcir_tev_intake_set_live(&intake, (uint32_t)arg);
        printf("%u %u L %u\n", scn, op, (uint32_t)arg);
      } else if (code == OP_CHECK) {
        bcir_ring_accounting a;
        bcir_status st = bcir_ring_accounting_of(region, (size_t)region_size, &a);
        if (st != BCIR_OK) {
          printf("%u %u A ERR %s\n", scn, op, status_name(st));
        } else {
          bcir_tev_report r;
          bcir_tev_intake_report(&intake, &r);
          printf("%u %u A %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %" PRIu64 " %" PRIu64
                 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
                 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %u\n",
                 scn, op, a.tail, a.delivered, a.lost, a.stale, a.last_epoch,
                 le64(region + BCIR_RING_OFF_HEAD), le64(region + BCIR_RING_OFF_REFUSED),
                 le64(region + BCIR_RING_OFF_P_OWNER), le64(region + BCIR_RING_OFF_C_OWNER),
                 le64(region + BCIR_RING_OFF_P_BEAT), le64(region + BCIR_RING_OFF_C_BEAT),
                 r.accepted, r.skipped, r.refused, r.missing, r.reordered, r.duplicated,
                 r.reported_lost, r.streams);
        }
      } else {
        free(region);
        goto bad;
      }
    }
    free(region);
  }
  if (at != len) goto bad;
  free(buf);
  return 0;
bad:
  fprintf(stderr, "malformed script\n");
  free(buf);
  return 2;
}

/* --- --envelopes ---------------------------------------------------------------------------- */

static const char *const KINDS[] = {"", "sample", "datadna"};
static const char *const CLOCK_NAMES[] = {"none", "monotonic", "realtime", "boot", "cycles"};
static const char *const UNIT_NAMES[] = {"none", "ns", "us", "cycles"};

/* Decode each u32le-length-framed envelope: its fields and the C rail's own re-encoding
 * ("env kind=... hex=<C re-encode>"), or "refused status=<BCIR_*>". */
static int run_envelopes(const char *path) {
  size_t len = 0, at = 0;
  uint8_t *buf = read_file(path, &len);
  if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 2; }
  while (at < len) {
    if (len - at < 4u) { free(buf); return 2; }
    size_t n = le32(buf + at);
    at += 4u;
    if (n > len - at) { free(buf); return 2; }
    uint8_t *record = (uint8_t *)malloc(n ? n : 1u); /* exact-size copy: an overread is caught */
    if (!record) { free(buf); return 2; }
    if (n) memcpy(record, buf + at, n);
    at += n;
    bcir_tev e;
    bcir_status st = bcir_tev_decode(n ? record : NULL, n, &e);
    free(record);
    if (st != BCIR_OK) {
      printf("refused status=%s\n", status_name(st));
      continue;
    }
    uint8_t again[BCIR_TEV_MAX];
    size_t written = 0;
    if (bcir_tev_encode(&e, again, sizeof again, &written) != BCIR_OK) {
      printf("refused status=REENCODE\n");
      continue;
    }
    printf("env kind=%s clock=%s unit=%s required=%u source=%" PRIu64 " session=%" PRIu64
           " generation=%u signal=%u seq=%u lost=%u timestamp=%" PRIu64 " value=%" PRId64 " record=",
           KINDS[e.kind], CLOCK_NAMES[e.clock], UNIT_NAMES[e.unit], (unsigned)e.required, e.source,
           e.session, e.generation, e.signal, e.seq, e.lost, e.timestamp, e.value);
    for (unsigned i = 0; i < 7; i++) printf("%s%" PRId64, i ? "," : "", e.record[i]);
    printf(" hex=");
    for (size_t i = 0; i < written; i++) printf("%02x", again[i]);
    printf("\n");
  }
  free(buf);
  return 0;
}

/* --- --signals ------------------------------------------------------------------------------ */

static int run_signals(void) {
  for (unsigned i = 0; i < BCIR_SIGNAL_COUNT; i++) {
    uint8_t row[BCIR_SIGNAL_ROW_SIZE];
    bcir_signal_row_encode(&BCIR_SIGNAL_TABLE[i], row);
    for (unsigned b = 0; b < BCIR_SIGNAL_ROW_SIZE; b++) printf("%02x", row[b]);
    printf("\n");
  }
  return 0;
}

/* --- --api: the fail-closed laws no script reaches ------------------------------------------ */

static int failures = 0;
#define CHECK(cond, name) do { if (!(cond)) { printf("FAIL %s\n", name); ++failures; } } while (0)

static int run_api(void) {
  bcir_ring_geometry g = {0};
  g.policy = BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_TELEMETRY;
  g.slot_size = 192;
  g.slot_count = 4;
  g.ring_id = 7;
  uint64_t size = bcir_ring_region_size(&g);
  CHECK(size == BCIR_RING_HEADER_SIZE + 4u * 192u, "region size");
  uint8_t *region = (uint8_t *)aligned_region((size_t)size + 64u);
  if (!region) return 2;
  CHECK(bcir_ring_format(NULL, (size_t)size, &g) == BCIR_ERR_RING, "format NULL region");
  CHECK(bcir_ring_format(region + 1, (size_t)size, &g) == BCIR_ERR_RING, "format misaligned");
  CHECK(bcir_ring_format(region, (size_t)size - 1u, &g) == BCIR_ERR_NOSPACE, "format short");
  CHECK(bcir_ring_format(region, (size_t)size, NULL) == BCIR_ERR_RING, "format NULL geometry");
  bcir_ring_geometry bad = g;
  bad.payload = BCIR_RING_CONTROL;
  bad.policy = BCIR_RING_OVERWRITE;
  CHECK(bcir_ring_format(region, (size_t)size, &bad) == BCIR_ERR_RING, "control ring overwrite");
  bad.policy = BCIR_RING_BACKPRESSURE; /* a 192-byte slot cannot carry the control bound */
  CHECK(bcir_ring_format(region, (size_t)size, &bad) == BCIR_ERR_RING, "control ring slot too small");
  CHECK(bcir_ring_region_size(&bad) == 0u, "invalid geometry has no size");
  CHECK(bcir_ring_format(region, (size_t)size, &g) == BCIR_OK, "format");
  CHECK(bcir_ring_decode_geometry(NULL, 0, NULL) == BCIR_ERR_TRUNCATED, "decode NULL");
  bcir_ring_accounting acct;
  CHECK(bcir_ring_accounting_of(region, (size_t)size, NULL) == BCIR_ERR_RING, "accounting NULL out");
  CHECK(bcir_ring_accounting_of(region, (size_t)size, &acct) == BCIR_OK && acct.tail == 0u,
        "accounting");

  bcir_ring_producer p;
  bcir_ring_consumer c;
  bcir_ring_producer_init(&p, region + 8, (size_t)size); /* misaligned for a word: 8 is fine */
  bcir_ring_producer_init(&p, region + 4, (size_t)size);
  CHECK(bcir_ring_producer_attach(&p, 0).status == BCIR_ERR_RING, "attach misaligned");
  CHECK(bcir_ring_producer_attach(NULL, 0).status == BCIR_ERR_RING, "attach NULL");
  bcir_ring_producer_init(&p, region, (size_t)size);
  CHECK(bcir_ring_publish(&p, (const uint8_t *)"x", 1).status == BCIR_ERR_STALE, "write unattached");
  CHECK(bcir_ring_write_close(&p).status == BCIR_ERR_RING, "close without open");
  CHECK(bcir_ring_write_fill(&p, 0, (const uint8_t *)"x", 1).status == BCIR_ERR_RING,
        "fill without open");
  CHECK(bcir_ring_producer_attach(&p, 0).verdict == BCIR_RING_OK, "attach");
  CHECK(bcir_ring_write_open(&p, 169).status == BCIR_ERR_NOSPACE, "open beyond capacity");
  CHECK(bcir_ring_write_open(&p, 16).verdict == BCIR_RING_OPEN, "open");
  CHECK(bcir_ring_write_open(&p, 16).status == BCIR_ERR_RING, "open twice");
  CHECK(bcir_ring_producer_detach(&p).status == BCIR_ERR_RING, "detach while open");
  CHECK(bcir_ring_write_fill(&p, 4, (const uint8_t *)"abcd", 4).status == BCIR_ERR_NOSPACE,
        "fill misaligned offset");
  CHECK(bcir_ring_write_fill(&p, 8, (const uint8_t *)"abcdefghij", 10).status == BCIR_ERR_NOSPACE,
        "fill beyond length");
  CHECK(bcir_ring_write_fill(&p, 0, NULL, 4).status == BCIR_ERR_RING, "fill NULL data");
  CHECK(bcir_ring_write_fill(&p, 0, (const uint8_t *)"abcdefghijklmnop", 16).verdict ==
            BCIR_RING_OPEN, "fill");
  CHECK(bcir_ring_write_close(&p).verdict == BCIR_RING_OK, "close");

  bcir_ring_consumer_init(&c, region, (size_t)size);
  CHECK(bcir_ring_consume(&c, read_buffer, sizeof read_buffer).status == BCIR_ERR_STALE,
        "read unattached");
  CHECK(bcir_ring_consumer_attach(&c, 0).verdict == BCIR_RING_OK, "consumer attach");
  CHECK(bcir_ring_consumer_attach(&c, 0).status == BCIR_ERR_BUSY, "second consumer attach");
  CHECK(bcir_ring_consumer_attach(&c, 9).status == BCIR_ERR_STALE, "takeover of a wrong epoch");
  uint8_t small[8];
  bcir_ring_outcome o = bcir_ring_consume(&c, small, sizeof small);
  CHECK(o.verdict == BCIR_RING_REFUSED && o.status == BCIR_ERR_NOSPACE, "read into a short buffer");
  o = bcir_ring_consume(&c, NULL, 0);
  CHECK(o.verdict == BCIR_RING_REFUSED && o.status == BCIR_ERR_NOSPACE, "read into no buffer");
  o = bcir_ring_consume(&c, read_buffer, sizeof read_buffer);
  CHECK(o.verdict == BCIR_RING_DELIVERED && o.length == 16u &&
            memcmp(read_buffer, "abcdefghijklmnop", 16) == 0, "read after a refusal");
  CHECK(bcir_ring_read_close(&c, read_buffer, sizeof read_buffer).status == BCIR_ERR_RING,
        "close without open");
  CHECK(bcir_ring_read_copy(&c, read_buffer, sizeof read_buffer).status == BCIR_ERR_RING,
        "copy without open");
  CHECK(bcir_ring_read_open(&c).verdict == BCIR_RING_EMPTY, "empty");
  CHECK(bcir_ring_accounting_of(region, (size_t)size, &acct) == BCIR_OK && acct.delivered == 1u,
        "accounting after a read");

  /* the envelope codec's own API laws */
  bcir_tev e;
  memset(&e, 0, sizeof e);
  CHECK(bcir_tev_encode(NULL, read_buffer, sizeof read_buffer, NULL) == BCIR_ERR_TELEMETRY,
        "encode NULL");
  e.kind = BCIR_TEV_SAMPLE;
  e.source = 1;
  e.session = 1;
  e.signal = 1;
  size_t written = 0;
  CHECK(bcir_tev_encode(&e, read_buffer, 10, &written) == BCIR_ERR_NOSPACE, "encode short");
  CHECK(bcir_tev_encode(&e, read_buffer, sizeof read_buffer, &written) == BCIR_OK &&
            written == BCIR_TEV_SAMPLE_SIZE, "encode");
  CHECK(bcir_tev_decode(NULL, 0, NULL) == BCIR_ERR_TRUNCATED, "decode NULL");
  CHECK(bcir_signal_find(0) == NULL && bcir_signal_find(1) != NULL &&
            bcir_signal_find(BCIR_SIGNAL_COUNT) != NULL &&
            bcir_signal_find(BCIR_SIGNAL_COUNT + 1u) == NULL, "signal lookup");
  bcir_tev_outcome t = bcir_tev_intake_admit(NULL, read_buffer, written, NULL);
  CHECK(t.verdict == BCIR_TEV_REFUSED, "intake NULL");
  bcir_seq_tracker s;
  bcir_seq_init(&s);
  CHECK(bcir_seq_observe(&s, 0xFFFFFFFFu) == BCIR_SEQ_FIRST && bcir_seq_observe(&s, 0) == BCIR_SEQ_NEXT &&
            bcir_seq_observe(&s, 0) == BCIR_SEQ_DUPLICATE && bcir_seq_observe(&s, 0xFFFFFFFEu) ==
            BCIR_SEQ_REORDER && bcir_seq_observe(&s, 3) == BCIR_SEQ_GAP && s.missing == 2u,
        "sequence classes across the u32 wrap");
  free(region);
  if (failures) return 1;
  printf("OK\n");
  return 0;
}

#if RING_HAS_POSIX
/* --- shared helpers for --stress / --procs / --bench --------------------------------------- */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static uint64_t rng_next(uint64_t *state) { /* splitmix64 */
  uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}

/* A spin-wait hint: the loop that polls a peer's word should not starve that peer's core. */
static void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ volatile("yield");
#endif
}

static void spin(uint64_t n) {
  for (volatile uint64_t i = 0; i < n; i++) {
  }
}

/* The record a producer writes for sequence `seq` of `session`: a DataDNA envelope whose every
 * field is a function of (session, seq), so a consumer can check any record it is handed. */
static size_t make_record(uint64_t session, uint32_t seq, uint32_t lost, uint8_t *out) {
  bcir_tev e;
  memset(&e, 0, sizeof e);
  e.kind = BCIR_TEV_DATADNA;
  e.source = 1;
  e.session = session;
  e.generation = 1;
  e.seq = seq;
  e.lost = lost;
  e.clock = BCIR_TEV_CLOCK_MONOTONIC;
  e.unit = BCIR_TEV_UNIT_NS;
  e.timestamp = (uint64_t)seq * 1000u + 1u;
  e.record[0] = (int64_t)seq;
  e.record[1] = (int64_t)seq * 7;
  e.record[2] = (int64_t)(seq * 13u + (uint32_t)session);
  e.record[3] = (int64_t)(seq % 101u);
  e.record[4] = (int64_t)(seq % 97u);
  e.record[5] = (int64_t)(seq % 89u);
  e.record[6] = (int64_t)(seq % 83u);
  size_t written = 0;
  if (bcir_tev_encode(&e, out, BCIR_TEV_MAX, &written) != BCIR_OK) return 0;
  return written;
}

/* Check a delivered record: it is byte for byte the record its (session, sequence) was written
 * as (the producers here never report a drop, so every field is determined). */
static int record_ok(const uint8_t *data, size_t len, uint64_t session, uint32_t seq) {
  uint8_t want[BCIR_TEV_MAX];
  size_t n = make_record(session, seq, 0, want);
  return n != 0u && n == len && memcmp(data, want, n) == 0;
}

/* --- --stress: two threads ------------------------------------------------------------------ */

typedef struct stress_ctx {
  bcir_ring_producer producer;
  bcir_ring_consumer consumer;
  uint64_t records;
  uint64_t seed;
  int overwrite;
  /* consumer results; `leading` = positions lost before the first delivered record, which no
   * continuity tracker can see (the first record it observes is its baseline) */
  uint64_t delivered, lost, stale, torn, refusals, leading;
  bcir_tev_report report;
  /* producer results */
  uint64_t written, full_retries;
} stress_ctx;

static void *stress_producer(void *arg) {
  stress_ctx *x = (stress_ctx *)arg;
  uint64_t rng = x->seed ^ 0xA5A5u;
  uint8_t rec[BCIR_TEV_MAX];
  uint64_t deadline = now_ns() + UINT64_C(20) * 1000000000u;
  for (uint64_t i = 0; i < x->records; i++) {
    size_t n = make_record(1, (uint32_t)i, 0, rec);
    for (;;) {
      bcir_ring_outcome o = bcir_ring_publish(&x->producer, rec, n);
      if (o.verdict == BCIR_RING_OK) break;
      if (o.status != BCIR_ERR_FULL || now_ns() > deadline) return NULL; /* the consumer sees the shortfall */
      x->full_retries++;
    }
    x->written++;
    if ((rng_next(&rng) & 1023u) == 0u) spin(rng_next(&rng) & 4095u);
  }
  return NULL;
}

static void *stress_consumer(void *arg) {
  stress_ctx *x = (stress_ctx *)arg;
  uint64_t rng = x->seed ^ 0x5A5Au;
  bcir_tev_intake intake;
  bcir_tev_intake_init(&intake, 1);
  uint8_t buf[256];
  uint64_t accounted = 0;
  uint64_t origin = x->consumer.g.origin;
  uint64_t deadline = now_ns() + UINT64_C(20) * 1000000000u;
  while (accounted < x->records) {
    bcir_ring_outcome o = bcir_ring_consume(&x->consumer, buf, sizeof buf);
    if (o.verdict == BCIR_RING_DELIVERED) {
      uint32_t seq = (uint32_t)(o.position - origin);
      if (!record_ok(buf, o.length, 1, seq)) x->torn++;
      (void)bcir_tev_intake_admit(&intake, buf, o.length, NULL);
      x->delivered++;
      accounted++;
    } else if (o.verdict == BCIR_RING_LOST) {
      x->lost += o.count;
      if (x->delivered == 0u) x->leading += o.count;
      accounted += o.count;
    } else if (o.verdict == BCIR_RING_STALE) {
      x->stale++;
      accounted++;
    } else if (o.verdict == BCIR_RING_REFUSED) {
      x->refusals++;
      break;
    } else if (now_ns() > deadline) {
      x->refusals++;
      break;
    }
    if (x->overwrite && (rng_next(&rng) & 255u) == 0u) spin(rng_next(&rng) & 65535u);
  }
  bcir_tev_intake_report(&intake, &x->report);
  return NULL;
}

static int run_stress(const char *mode, uint64_t records, uint64_t seed) {
  int overwrite = strcmp(mode, "ow") == 0;
  if (!overwrite && strcmp(mode, "bp") != 0) return 2;
  bcir_ring_geometry g = {0};
  g.policy = overwrite ? BCIR_RING_OVERWRITE : BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_TELEMETRY;
  g.slot_size = 192;
  g.slot_count = 64;
  g.ring_id = 11;
  g.origin = UINT64_MAX - 1000u; /* the positions wrap past 2^64 mid-run */
  uint64_t size = bcir_ring_region_size(&g);
  uint8_t *region = (uint8_t *)aligned_region((size_t)size);
  if (!region || bcir_ring_format(region, (size_t)size, &g) != BCIR_OK) return 2;
  static stress_ctx x;
  memset(&x, 0, sizeof x);
  x.records = records;
  x.seed = seed;
  x.overwrite = overwrite;
  bcir_ring_producer_init(&x.producer, region, (size_t)size);
  bcir_ring_consumer_init(&x.consumer, region, (size_t)size);
  if (bcir_ring_producer_attach(&x.producer, 0).verdict != BCIR_RING_OK ||
      bcir_ring_consumer_attach(&x.consumer, 0).verdict != BCIR_RING_OK)
    return 2;
  pthread_t tp, tc;
  if (pthread_create(&tc, NULL, stress_consumer, &x) != 0) return 2;
  if (pthread_create(&tp, NULL, stress_producer, &x) != 0) return 2;
  pthread_join(tp, NULL);
  pthread_join(tc, NULL);
  bcir_ring_accounting a;
  int acct_ok = bcir_ring_accounting_of(region, (size_t)size, &a) == BCIR_OK;
  uint64_t published = le64(region + BCIR_RING_OFF_HEAD) - g.origin;
  uint64_t unaccounted = 0;
  if (!acct_ok || published != x.written || a.delivered != x.delivered || a.lost != x.lost ||
      a.stale != x.stale || a.delivered + a.lost + a.stale != published)
    unaccounted++;
  if (!overwrite && (x.lost != 0u || le64(region + BCIR_RING_OFF_REFUSED) != x.full_retries))
    unaccounted++;
  uint64_t continuity = 0; /* the intake sees exactly the ring's loss after its baseline, as gaps */
  if (x.report.missing != x.lost - x.leading || x.report.reordered != 0u || x.report.duplicated != 0u ||
      x.report.accepted != x.delivered)
    continuity++;
  uint64_t violations = x.torn + unaccounted + continuity + x.refusals + x.stale;
  printf("stress %s records=%" PRIu64 " published=%" PRIu64 " delivered=%" PRIu64 " lost=%" PRIu64
         " leading=%" PRIu64 " missing=%" PRIu64 " stale=%" PRIu64 " refused=%" PRIu64
         " torn=%" PRIu64 " unaccounted=%" PRIu64 " continuity=%" PRIu64 " violations=%" PRIu64 "\n",
         mode, records, published, x.delivered, x.lost, x.leading, x.report.missing, x.stale,
         le64(region + BCIR_RING_OFF_REFUSED), x.torn, unaccounted, continuity, violations);
  free(region);
  return violations ? 1 : 0;
}

/* --- --procs: two processes, a peer SIGKILLed mid-stream ------------------------------------ */

typedef struct procs_shared {
  volatile uint64_t stop;       /* the supervisor asks the consumer to drain and finish */
  volatile uint64_t torn;       /* records a consumer was handed that are not what was written */
  volatile uint64_t refusals;   /* a consumer or producer met a refusal it should not */
  volatile uint64_t lost;       /* LOST counts a consumer observed */
  volatile uint64_t stale;
  volatile uint64_t delivered;  /* DELIVERED verdicts consumers observed (after the commit) */
  volatile uint64_t session_origin[3]; /* the position each producer session began at */
  volatile uint64_t published_target;
  uint8_t log[];                /* per position (relative to origin): how many times logged */
} procs_shared;

static void procs_consumer(uint8_t *region, size_t size, procs_shared *sh, uint32_t takeover,
                           uint64_t seed, uint64_t records) {
  bcir_ring_consumer c;
  bcir_ring_consumer_init(&c, region, size);
  if (bcir_ring_consumer_attach(&c, takeover).verdict != BCIR_RING_OK) {
    sh->refusals++;
    _exit(3);
  }
  uint64_t rng = seed ^ 0xC0FFEEu;
  uint64_t origin = c.g.origin;
  uint8_t buf[256];
  uint64_t deadline = now_ns() + UINT64_C(20) * 1000000000u;
  for (;;) {
    bcir_ring_outcome o = bcir_ring_consume(&c, buf, sizeof buf);
    if (o.verdict == BCIR_RING_DELIVERED) {
      uint64_t rel = o.position - origin;
      uint64_t session = rel >= sh->session_origin[2] - origin && sh->session_origin[2] ? 2u : 1u;
      uint64_t start = session == 2u ? sh->session_origin[2] - origin : 0u;
      if (!record_ok(buf, o.length, session, (uint32_t)(rel - start))) sh->torn++;
      if (rel < records) sh->log[rel]++;
      sh->delivered++;
    } else if (o.verdict == BCIR_RING_LOST) {
      sh->lost += o.count;
    } else if (o.verdict == BCIR_RING_STALE) {
      sh->stale++;
    } else if (o.verdict == BCIR_RING_REFUSED) {
      if (o.status != BCIR_ERR_BUSY) { sh->refusals++; break; }
    } else if (o.verdict == BCIR_RING_EMPTY) {
      if (sh->stop) break;
      if (now_ns() > deadline) { sh->refusals++; break; }
    }
    if ((rng_next(&rng) & 127u) == 0u) spin(rng_next(&rng) & 16383u);
  }
  _exit(0);
}

static void procs_producer(uint8_t *region, size_t size, procs_shared *sh, uint32_t takeover,
                           uint64_t session, uint64_t count, uint64_t seed) {
  bcir_ring_producer p;
  bcir_ring_producer_init(&p, region, size);
  bcir_ring_outcome a = bcir_ring_producer_attach(&p, takeover);
  if (a.verdict != BCIR_RING_OK) {
    sh->refusals++;
    _exit(3);
  }
  sh->session_origin[session] = a.position;
  uint64_t rng = seed ^ (session * 0x1234567u);
  uint8_t rec[BCIR_TEV_MAX];
  uint64_t deadline = now_ns() + UINT64_C(20) * 1000000000u;
  for (uint64_t i = 0; i < count; i++) {
    size_t n = make_record(session, (uint32_t)i, 0, rec);
    for (;;) {
      bcir_ring_outcome o = bcir_ring_publish(&p, rec, n);
      if (o.verdict == BCIR_RING_OK) break;
      if (o.status != BCIR_ERR_FULL || now_ns() > deadline) { sh->refusals++; _exit(4); }
    }
    if ((rng_next(&rng) & 255u) == 0u) spin(rng_next(&rng) & 8191u);
  }
  (void)bcir_ring_producer_detach(&p);
  _exit(0);
}

/* Wait (bounded) until the producer's head, or the consumer's committed tail, reaches `target`
 * positions past the origin. */
static int wait_until(uint8_t *region, size_t size, uint64_t origin, uint64_t target, int consumer) {
  uint64_t deadline = now_ns() + UINT64_C(30) * 1000000000u;
  for (;;) {
    uint64_t at;
    if (consumer) {
      bcir_ring_accounting a;
      if (bcir_ring_accounting_of(region, size, &a) != BCIR_OK) return 0;
      at = a.tail - origin;
    } else {
      at = le64(region + BCIR_RING_OFF_HEAD) - origin;
    }
    if (at >= target) return 1;
    if (now_ns() > deadline) return 0;
    struct timespec pause = {0, 20000};
    nanosleep(&pause, NULL);
  }
}

static int run_procs(const char *which, const char *mode, uint64_t records, uint64_t seed) {
  int kill_producer = strcmp(which, "kill-producer") == 0;
  if (!kill_producer && strcmp(which, "kill-consumer") != 0) return 2;
  int overwrite = strcmp(mode, "ow") == 0;
  if (!overwrite && strcmp(mode, "bp") != 0) return 2;
  bcir_ring_geometry g = {0};
  g.policy = overwrite ? BCIR_RING_OVERWRITE : BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_TELEMETRY;
  g.slot_size = 192;
  g.slot_count = 32;
  g.ring_id = 13;
  g.origin = 1000;
  size_t size = (size_t)bcir_ring_region_size(&g);
  uint8_t *region = (uint8_t *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  size_t shared_size = sizeof(procs_shared) + records;
  procs_shared *sh = (procs_shared *)mmap(NULL, shared_size, PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (region == MAP_FAILED || sh == MAP_FAILED) return 2;
  memset(sh, 0, shared_size);
  if (bcir_ring_format(region, size, &g) != BCIR_OK) return 2;
  sh->published_target = records;
  uint64_t killed_at = 0;
  pid_t consumer = -1, producer = -1;
  consumer = fork();
  if (consumer == 0) procs_consumer(region, size, sh, 0, seed, records);
  producer = fork();
  if (producer == 0) procs_producer(region, size, sh, 0, 1, records, seed);
  int status = 0;
  if (kill_producer) {
    if (!wait_until(region, size, g.origin, records / 2u, 0)) sh->refusals++;
    kill(producer, SIGKILL);
    waitpid(producer, &status, 0); /* death proved: reaped */
    uint64_t owner = le64(region + BCIR_RING_OFF_P_OWNER);
    killed_at = le64(region + BCIR_RING_OFF_HEAD) - g.origin;
    uint64_t remaining = killed_at < records ? records - killed_at : 0u;
    pid_t successor = fork();
    if (successor == 0) procs_producer(region, size, sh, (uint32_t)(owner >> 32), 2, remaining, seed);
    waitpid(successor, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) sh->refusals++;
    sh->stop = 1;
    waitpid(consumer, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) sh->refusals++;
  } else {
    if (!wait_until(region, size, g.origin, records / 3u, 1)) sh->refusals++;
    kill(consumer, SIGKILL);
    waitpid(consumer, &status, 0);
    uint64_t owner = le64(region + BCIR_RING_OFF_C_OWNER);
    bcir_ring_accounting before;
    (void)bcir_ring_accounting_of(region, size, &before);
    killed_at = before.tail - g.origin;
    pid_t successor = fork();
    if (successor == 0) procs_consumer(region, size, sh, (uint32_t)(owner >> 32), seed + 1u, records);
    waitpid(producer, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) sh->refusals++;
    sh->stop = 1;
    waitpid(successor, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) sh->refusals++;
  }
  bcir_ring_accounting a;
  int acct_ok = bcir_ring_accounting_of(region, size, &a) == BCIR_OK;
  uint64_t published = le64(region + BCIR_RING_OFF_HEAD) - g.origin;
  uint64_t unaccounted = 0, logged = 0, twice = 0;
  for (uint64_t i = 0; i < records; i++) {
    logged += sh->log[i];
    twice += sh->log[i] > 1u;
  }
  if (!acct_ok || a.tail - g.origin != published || a.delivered + a.lost + a.stale != published ||
      published != records)
    unaccounted++;
  if (!overwrite && a.lost != 0u) unaccounted++;
  /* The ring commits every position exactly once. A consumer SIGKILLed after its commit but
   * before its own log line loses that one record to its application (at most once). */
  uint64_t slack = kill_producer ? 0u : 1u;
  if (twice != 0u || logged > a.delivered || a.delivered - logged > slack) unaccounted++;
  if (kill_producer && sh->delivered != a.delivered) unaccounted++;
  uint64_t violations = sh->torn + unaccounted + sh->refusals + a.stale;
  printf("procs %s %s records=%" PRIu64 " published=%" PRIu64 " killed_at=%" PRIu64
         " delivered=%" PRIu64 " lost=%" PRIu64 " stale=%" PRIu64 " logged=%" PRIu64
         " torn=%" PRIu64 " unaccounted=%" PRIu64 " violations=%" PRIu64 "\n",
         which, mode, records, published, killed_at, a.delivered, a.lost, a.stale, logged,
         (uint64_t)sh->torn, unaccounted, violations);
  munmap(region, size);
  munmap(sh, shared_size);
  return violations ? 1 : 0;
}

/* --- --bench: records per second against a memcpy floor ----------------------------------- */

typedef struct bench_ctx {
  bcir_ring_consumer consumer;
  uint64_t records;
  uint64_t checksum;
} bench_ctx;

static void *bench_consumer(void *arg) {
  bench_ctx *x = (bench_ctx *)arg;
  uint8_t buf[256];
  uint64_t got = 0, sum = 0;
  while (got < x->records) {
    bcir_ring_outcome o = bcir_ring_consume(&x->consumer, buf, sizeof buf);
    if (o.verdict == BCIR_RING_DELIVERED) {
      sum += buf[40];
      got++;
    } else {
      cpu_relax();
    }
  }
  x->checksum = sum;
  return NULL;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : x > y;
}

static int run_bench(uint64_t records, uint32_t slot_size, uint32_t slots, unsigned repeats) {
  enum { DISTINCT = 1024 };
  if (repeats == 0u || repeats > 64u || records == 0u) return 2;
  static uint8_t src[DISTINCT][BCIR_TEV_MAX], dst[DISTINCT][BCIR_TEV_MAX];
  size_t n = 0;
  for (unsigned i = 0; i < DISTINCT; i++) n = make_record(1, i, 0, src[i]);
  bcir_ring_geometry g = {0};
  g.policy = BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_TELEMETRY;
  g.slot_size = slot_size;
  g.slot_count = slots;
  g.ring_id = 17;
  uint64_t size = bcir_ring_region_size(&g);
  if (size == 0u) return 2;
  uint8_t *region = (uint8_t *)aligned_region((size_t)size);
  if (!region) return 2;
  uint64_t ring_ns[64], copy_ns[64], sink = 0;
  for (unsigned r = 0; r < repeats; r++) {
    if (bcir_ring_format(region, (size_t)size, &g) != BCIR_OK) return 2;
    static bench_ctx x;
    memset(&x, 0, sizeof x);
    x.records = records;
    bcir_ring_producer p;
    bcir_ring_producer_init(&p, region, (size_t)size);
    bcir_ring_consumer_init(&x.consumer, region, (size_t)size);
    if (bcir_ring_producer_attach(&p, 0).verdict != BCIR_RING_OK ||
        bcir_ring_consumer_attach(&x.consumer, 0).verdict != BCIR_RING_OK)
      return 2;
    pthread_t tc;
    uint64_t t0 = now_ns();
    if (pthread_create(&tc, NULL, bench_consumer, &x) != 0) return 2;
    for (uint64_t i = 0; i < records; i++) {
      while (bcir_ring_publish(&p, src[i % DISTINCT], n).verdict != BCIR_RING_OK) cpu_relax();
    }
    pthread_join(tc, NULL);
    ring_ns[r] = now_ns() - t0;
    sink += x.checksum;
    t0 = now_ns();
    for (uint64_t i = 0; i < records; i++) {
      memcpy(dst[i % DISTINCT], src[i % DISTINCT], n);
      __asm__ volatile("" : : "r"(dst[i % DISTINCT]) : "memory");
    }
    copy_ns[r] = now_ns() - t0;
    sink += dst[records % DISTINCT][40];
  }
  qsort(ring_ns, repeats, sizeof ring_ns[0], cmp_u64);
  qsort(copy_ns, repeats, sizeof copy_ns[0], cmp_u64);
  double ring = (double)ring_ns[repeats / 2u] / (double)records;
  double copy = (double)copy_ns[repeats / 2u] / (double)records;
  printf("bench records=%" PRIu64 " bytes=%zu slot=%u slots=%u ring_ns=%.3f memcpy_ns=%.3f ratio=%.3f "
         "records_per_s=%.0f sink=%" PRIu64 "\n",
         records, n, slot_size, slots, ring, copy, copy > 0.0 ? ring / copy : 0.0,
         ring > 0.0 ? 1e9 / ring : 0.0, sink & 1u);
  free(region);
  return 0;
}
#endif /* RING_HAS_POSIX */

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--script") == 0) return run_script(argv[2]);
  if (argc == 2 && strcmp(argv[1], "--signals") == 0) return run_signals();
  if (argc == 3 && strcmp(argv[1], "--envelopes") == 0) return run_envelopes(argv[2]);
  if (argc == 2 && strcmp(argv[1], "--api") == 0) return run_api();
#if RING_HAS_POSIX
  if (argc == 5 && strcmp(argv[1], "--stress") == 0)
    return run_stress(argv[2], strtoull(argv[3], NULL, 10), strtoull(argv[4], NULL, 10));
  if (argc == 6 && strcmp(argv[1], "--procs") == 0)
    return run_procs(argv[2], argv[3], strtoull(argv[4], NULL, 10), strtoull(argv[5], NULL, 10));
  if (argc == 6 && strcmp(argv[1], "--bench") == 0)
    return run_bench(strtoull(argv[2], NULL, 10), (uint32_t)strtoul(argv[3], NULL, 10),
                     (uint32_t)strtoul(argv[4], NULL, 10), (unsigned)strtoul(argv[5], NULL, 10));
#else
  if (argc >= 2 && (strcmp(argv[1], "--stress") == 0 || strcmp(argv[1], "--procs") == 0 ||
                    strcmp(argv[1], "--bench") == 0)) {
    printf("UNAVAILABLE no POSIX threads/processes on this host\n");
    return 3;
  }
#endif
  fprintf(stderr,
          "usage: test_ring --script <script> | --signals | --envelopes <records> | --api | --stress <bp|ow> <records> <seed>"
          " | --procs <kill-producer|kill-consumer> <bp|ow> <records> <seed>"
          " | --bench <records> <slot_size> <slots> <repeats>\n");
  return 2;
}
