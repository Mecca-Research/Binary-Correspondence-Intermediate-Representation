/*===- bcir_ring.h - BCIR live SPSC ring, version zero ----------------------===
 *
 * The live single-producer / single-consumer ring of the GEM+ roadmap's G15 (staged plan S3-B):
 * head and tail, acquire/release publication, a per-slot sequence, a declared overwrite or
 * backpressure policy and exact loss accounting, in a region two peers share (two threads, two
 * processes over a MAP_SHARED mapping, a driver and its host). It carries TelemetryEnvelopeV0
 * records first and ControlRecordV1 records second. The Python oracle is bcir/gem/ring.py and
 * docs/kernel/BCIR_LIVE_RING_ABI.md the prose spec; the rails agree operation for operation
 * (every scripted scenario's verdicts, statuses, positions, counts, epochs, payloads and the
 * region's bytes after every step), held by the G15 rows (tools/perf/gemplus_baseline.py --group
 * ring) and tools/c/check_runtime.sh.
 *
 * Region (little-endian; offsets in bytes; `*` = a word the peers share, read and written with
 * C11 atomics -- so the region must be 8-byte aligned, and the host little-endian):
 *   line 0   geometry, immutable after bcir_ring_format, CRC-sealed (bytes 0..63)
 *   line 1   producer ownership: p_owner* = epoch << 32 | state @64, epoch_origin* @72
 *   line 2   producer progress:  head* @128, p_heartbeat* @136   (the consumer polls head)
 *   line 3   producer counters:  refused* @192                    (the producer's alone)
 *   line 4   consumer progress:  tail* @256                       (the producer polls it)
 *   line 5   consumer: c_owner* @320, c_heartbeat* @328, c_commit* @336, acct[0] @352
 *   line 6   acct[1] @384        (acct = tail*, lost*, stale*, last_epoch* -- the valid one is
 *                                 acct[c_commit & 1]: a consumer that dies mid-update leaves the
 *                                 other intact; the progress tail is published after it)
 *   slots    @448, stride slot_size: seq* (odd while written) @0, pos* @8,
 *            epoch | length << 32 * @16, payload @24 (slot_size - 24 bytes, word-copied)
 *   Each line has one writer; a side's per-operation writes never share a line with what the
 *   other side polls.
 *
 * Every operation is O(1) and non-blocking: a full backpressure ring refuses (BCIR_ERR_FULL and
 * the `refused` count), an empty ring answers EMPTY, and a consumer the producer lapped is told
 * the exact count it lost (LOST) and never handed a torn record. Every position the producer
 * publishes is committed exactly once as delivered, lost or stale.
 *
 * The contract: ONE live producer and ONE live consumer. A takeover (attach with the epoch of a
 * dead peer) requires the embedding to have proved the peer dead (a reaped process, a watchdog
 * reset); a deposed peer that is still running refuses itself at its next operation (the owner
 * word is re-read by every open), and the consumer refuses a record stamped with a deposed epoch
 * at or after the takeover position (STALE) -- but two live producers are outside SPSC, and the
 * ring then reports BCIR_ERR_RING rather than deliver what it cannot vouch for. Version zero:
 * the layout carries no compatibility promise until UART and virtio-blk traces revise it.
 *
 * Freestanding: <stdatomic.h> (lock-free 64-bit atomics required), <stddef.h>, <stdint.h>; no
 * heap, no libc. The geometry CRC is bcir_crc32 (bcir_runtime.c).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_RING_H
#define BCIR_RING_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_RING_MAGIC         "BRNG"
#define BCIR_RING_VERSION       0u
#define BCIR_RING_HEADER_SIZE   448u
#define BCIR_RING_GEOMETRY_SIZE 64u
#define BCIR_RING_SLOT_HEADER   24u
#define BCIR_RING_SLOT_SIZE_MIN 64u
#define BCIR_RING_SLOT_SIZE_MAX 65536u
#define BCIR_RING_SLOTS_MIN     2u
#define BCIR_RING_SLOTS_MAX     (1u << 20)
#define BCIR_RING_EPOCH_MAX     0xFFFFFFFFu
/* A control ring carries every legal ControlRecordV1: a slot's payload holds the record ABI's
 * declared bound (BCIR_CTL_RECORD_MAX, 192 bytes), so the slot is that plus the slot header,
 * rounded up to a cache line (test_control_plane.c asserts the relation where both meet). */
#define BCIR_RING_CONTROL_SLOT_MIN 256u

/* Shared-word offsets. */
#define BCIR_RING_OFF_P_OWNER  64u
#define BCIR_RING_OFF_ORIGIN   72u
#define BCIR_RING_OFF_HEAD     128u
#define BCIR_RING_OFF_P_BEAT   136u
#define BCIR_RING_OFF_REFUSED  192u
#define BCIR_RING_OFF_TAIL     256u
#define BCIR_RING_OFF_C_OWNER  320u
#define BCIR_RING_OFF_C_BEAT   328u
#define BCIR_RING_OFF_C_COMMIT 336u
#define BCIR_RING_OFF_ACCT0    352u
#define BCIR_RING_OFF_ACCT1    384u

typedef enum bcir_ring_policy {
  BCIR_RING_BACKPRESSURE = 1, /* a full ring refuses; the producer never overwrites */
  BCIR_RING_OVERWRITE    = 2  /* the producer never waits; the consumer counts what it lost */
} bcir_ring_policy;

typedef enum bcir_ring_payload {
  BCIR_RING_TELEMETRY = 1, /* TelemetryEnvelopeV0 records */
  BCIR_RING_CONTROL   = 2  /* ControlRecordV1 records: always BACKPRESSURE */
} bcir_ring_payload;

typedef enum bcir_ring_state {
  BCIR_RING_DETACHED  = 0,
  BCIR_RING_ATTACHING = 1, /* a producer claiming the ring: its origin is being written */
  BCIR_RING_ATTACHED  = 2
} bcir_ring_state;

typedef enum bcir_ring_verdict {
  BCIR_RING_OK        = 0, /* an attach/detach/beat/publish that took effect */
  BCIR_RING_OPEN      = 1, /* a two-phase step in progress */
  BCIR_RING_DELIVERED = 2, /* a read delivered the record at `position` */
  BCIR_RING_EMPTY     = 3, /* nothing is published at the consumer's position */
  BCIR_RING_LOST      = 4, /* `count` positions were overwritten before they were read */
  BCIR_RING_STALE     = 5, /* the record carries a deposed producer's epoch: refused, counted */
  BCIR_RING_REFUSED   = 6  /* the operation was refused: `status` says why */
} bcir_ring_verdict;

typedef struct bcir_ring_geometry {
  uint8_t policy;
  uint8_t payload;
  uint32_t slot_size;
  uint32_t slot_count;
  uint64_t ring_id;
  uint64_t origin;
  uint64_t region_size;
} bcir_ring_geometry;

typedef struct bcir_ring_outcome {
  bcir_ring_verdict verdict;
  bcir_status status;
  uint64_t position;
  uint64_t count;
  uint32_t epoch;
  size_t length; /* a DELIVERED record's payload bytes */
} bcir_ring_outcome;

typedef struct bcir_ring_accounting {
  uint64_t tail;
  uint64_t delivered;
  uint64_t lost;
  uint64_t stale;
  uint32_t last_epoch;
} bcir_ring_accounting;

/* A producer endpoint: private state the caller owns (the region is shared). */
typedef struct bcir_ring_producer {
  uint8_t *region;
  size_t region_len;
  bcir_ring_geometry g;
  uint32_t epoch;
  uint64_t cached_tail;
  int is_open;
  uint64_t open_head;
  size_t open_base;
  uint64_t open_seq;
  uint32_t open_length;
} bcir_ring_producer;

/* A consumer endpoint. */
typedef struct bcir_ring_consumer {
  uint8_t *region;
  size_t region_len;
  bcir_ring_geometry g;
  uint32_t epoch;
  uint64_t commit;
  uint64_t tail;
  uint64_t lost;
  uint64_t stale;
  uint32_t last_epoch;
  uint64_t cached_head;
  int has_cursor;
  int copied;
  uint64_t cur_position;
  size_t cur_base;
  uint64_t cur_seq;
  uint64_t cur_pos;
  uint32_t cur_epoch;
  uint32_t cur_length;
} bcir_ring_consumer;

/* The geometry's value laws (BCIR_ERR_RING), without a region. */
BCIR_NODISCARD bcir_status bcir_ring_geometry_validate(const bcir_ring_geometry *g);

/* region_size for a geometry (header + slots); 0 when the geometry is not valid. */
BCIR_NODISCARD uint64_t bcir_ring_region_size(const bcir_ring_geometry *g);

/* Lay an empty ring into `region` (8-byte aligned, at least region_size bytes). */
BCIR_NODISCARD bcir_status bcir_ring_format(uint8_t *region, size_t len,
                                            const bcir_ring_geometry *g);

/* The geometry line's laws, in the specification's order, against the region's length:
 * TRUNCATED, MAGIC, VERSION, CRC, RESERVED, RING (policy, payload, slot size, slot count,
 * ring id, region size, a control ring that is not backpressure), TRUNCATED (region). */
BCIR_NODISCARD bcir_status bcir_ring_decode_geometry(const uint8_t *region, size_t len,
                                                     bcir_ring_geometry *out);

/* The consumer's committed accounting as any peer reads it (a consistent double-buffer
 * snapshot); delivered = tail - origin - lost - stale. */
BCIR_NODISCARD bcir_status bcir_ring_accounting_of(const uint8_t *region, size_t len,
                                                   bcir_ring_accounting *out);

/* ---- the producer ---- */
void bcir_ring_producer_init(bcir_ring_producer *p, uint8_t *region, size_t len);
/* takeover == 0: attach a DETACHED ring (else BUSY). takeover == E: take the end over from the
 * dead producer of epoch E (else STALE); its half-written slot at the head is retired. */
bcir_ring_outcome bcir_ring_producer_attach(bcir_ring_producer *p, uint32_t takeover);
bcir_ring_outcome bcir_ring_producer_detach(bcir_ring_producer *p);
bcir_ring_outcome bcir_ring_write_open(bcir_ring_producer *p, size_t length);
/* offset % 8 == 0 and offset + n <= the opened length; a partial last word is zero-padded. */
bcir_ring_outcome bcir_ring_write_fill(bcir_ring_producer *p, size_t offset,
                                       const uint8_t *data, size_t n);
bcir_ring_outcome bcir_ring_write_close(bcir_ring_producer *p);
/* One record in one call: write_open + write_fill + write_close. */
bcir_ring_outcome bcir_ring_publish(bcir_ring_producer *p, const uint8_t *data, size_t n);
bcir_ring_outcome bcir_ring_producer_beat(bcir_ring_producer *p);

/* ---- the consumer ---- */
void bcir_ring_consumer_init(bcir_ring_consumer *c, uint8_t *region, size_t len);
bcir_ring_outcome bcir_ring_consumer_attach(bcir_ring_consumer *c, uint32_t takeover);
bcir_ring_outcome bcir_ring_consumer_detach(bcir_ring_consumer *c);
bcir_ring_outcome bcir_ring_read_open(bcir_ring_consumer *c);
/* Copies the open record's payload into `out` (at most `cap` bytes; a record longer than `cap`
 * is refused NOSPACE at close, never truncated silently). */
bcir_ring_outcome bcir_ring_read_copy(bcir_ring_consumer *c, uint8_t *out, size_t cap);
bcir_ring_outcome bcir_ring_read_close(bcir_ring_consumer *c, uint8_t *out, size_t cap);
/* One position in one call: read_open + read_copy + read_close. */
bcir_ring_outcome bcir_ring_consume(bcir_ring_consumer *c, uint8_t *out, size_t cap);
bcir_ring_outcome bcir_ring_consumer_beat(bcir_ring_consumer *c);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_RING_H */
