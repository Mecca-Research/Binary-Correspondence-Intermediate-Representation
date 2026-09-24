/*===- bcir_telemetry_envelope.h - TelemetryEnvelopeV0 + the host intake ---===
 *
 * The C twin of bcir/abi/telemetry_envelope.py (the codec), bcir/telemetry_intake.py (the
 * intake) and bcir.telemetry.SequenceTracker (the continuity predicate), for G15 (staged plan
 * S3-B). docs/kernel/TELEMETRY_ENVELOPE_ABI.md is the prose spec. The rails agree record for
 * record (Python encode -> C decode -> the same fields) and decision for decision (every intake
 * scenario's verdicts, reasons, statuses and classifications).
 *
 * Wire format (little-endian; fixed-width; one CRC-sealed record):
 *   header(64): magic[4]="BTEV" version:u16=0 flags:u16 (bit 0 REQUIRED) kind:u8 schema:u8
 *               clock:u8 unit:u8 size:u16 reserved:u16 source:u64 session:u64 generation:u32
 *               signal:u32 seq:u32 lost:u32 timestamp:u64 reserved:u64
 *   payload:    sample  = value:i64 (8)          -> record 76 bytes
 *               datadna = the frozen <7q> DataDNA (56) -> record 124 bytes
 *   crc32:      bcir_crc32 (zlib-compatible) of every preceding byte
 *
 * Freestanding: <stddef.h> + <stdint.h> (through bcir_runtime.h), no heap, no libc. The intake
 * is a fixed-size value the caller owns (sixteen streams). Version zero: no compatibility
 * promise until UART and virtio-blk traces revise the field set.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_TELEMETRY_ENVELOPE_H
#define BCIR_TELEMETRY_ENVELOPE_H

#include "bcir_runtime.h"
#include "bcir_signal_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_TEV_MAGIC        "BTEV"
#define BCIR_TEV_VERSION      0u
#define BCIR_TEV_HEADER       64u
#define BCIR_TEV_CRC          4u
#define BCIR_TEV_SAMPLE_SIZE  76u
#define BCIR_TEV_DATADNA_SIZE 124u
#define BCIR_TEV_MAX          124u
#define BCIR_TEV_FLAG_REQUIRED 0x0001u
#define BCIR_TEV_SIGNAL_RESERVED 0xFFFFFFFFu
#define BCIR_TEV_STREAMS      16u

typedef enum bcir_tev_kind { BCIR_TEV_SAMPLE = 1, BCIR_TEV_DATADNA = 2 } bcir_tev_kind;

typedef enum bcir_tev_clock {
  BCIR_TEV_CLOCK_NONE = 0,
  BCIR_TEV_CLOCK_MONOTONIC = 1,
  BCIR_TEV_CLOCK_REALTIME = 2,
  BCIR_TEV_CLOCK_BOOT = 3,
  BCIR_TEV_CLOCK_CYCLES = 4
} bcir_tev_clock;
#define BCIR_TEV_CLOCK_MAX 4u

typedef enum bcir_tev_unit {
  BCIR_TEV_UNIT_NONE = 0,
  BCIR_TEV_UNIT_NS = 1,
  BCIR_TEV_UNIT_US = 2,
  BCIR_TEV_UNIT_CYCLES = 3
} bcir_tev_unit;
#define BCIR_TEV_UNIT_MAX 3u

/* A decoded envelope (the wire's fields; `record` for datadna, `value` for sample). */
typedef struct bcir_tev {
  uint8_t kind;
  uint8_t clock;
  uint8_t unit;
  uint8_t required;
  uint64_t source;
  uint64_t session;
  uint32_t generation;
  uint32_t signal;
  uint32_t seq;
  uint32_t lost;
  uint64_t timestamp;
  int64_t value;
  int64_t record[7];
} bcir_tev;

/* The wire laws, in the specification's order: TRUNCATED (< header), MAGIC, VERSION, RESERVED
 * (flag bits), TELEMETRY (kind, size), TRUNCATED/TRAILING, CRC, RESERVED (fields), TELEMETRY
 * (schema, clock, unit, clock/unit pairing, a timestamp with no clock, source 0, session 0, a
 * sample's signal 0 or reserved, a DataDNA record's signal, REQUIRED or generation 0). */
BCIR_NODISCARD bcir_status bcir_tev_decode(const uint8_t *data, size_t len, bcir_tev *out);

/* The encoder refuses exactly what the decoder refuses; `*written` is the record's size. */
BCIR_NODISCARD bcir_status bcir_tev_encode(const bcir_tev *env, uint8_t *out, size_t cap,
                                           size_t *written);

/* ---- continuity: the frame ABI's u32 predicate (bcir.telemetry.SequenceTracker) ---- */
typedef enum bcir_seq_class {
  BCIR_SEQ_NONE = 0,
  BCIR_SEQ_FIRST = 1,
  BCIR_SEQ_NEXT = 2,
  BCIR_SEQ_GAP = 3,
  BCIR_SEQ_DUPLICATE = 4,
  BCIR_SEQ_REORDER = 5
} bcir_seq_class;

typedef struct bcir_seq_tracker {
  uint32_t watermark;
  int has_watermark;
  uint64_t missing;
  uint64_t reordered;
  uint64_t duplicated;
  uint64_t observed;
} bcir_seq_tracker;

void bcir_seq_init(bcir_seq_tracker *t);
bcir_seq_class bcir_seq_observe(bcir_seq_tracker *t, uint32_t seq);

/* ---- the generated signal table ---- */
/* The row for `id`, or NULL (binary search over BCIR_SIGNAL_TABLE). */
const bcir_signal_def *bcir_signal_find(uint32_t id);
/* A row's 64 bytes (the Python table's encode_row). */
void bcir_signal_row_encode(const bcir_signal_def *def, uint8_t out[BCIR_SIGNAL_ROW_SIZE]);

/* ---- the intake ---- */
typedef enum bcir_tev_verdict {
  BCIR_TEV_ACCEPTED = 0,
  BCIR_TEV_SKIPPED = 1,
  BCIR_TEV_REFUSED = 2
} bcir_tev_verdict;

typedef enum bcir_tev_reason {
  BCIR_TEV_REASON_NONE = 0,
  BCIR_TEV_REASON_MALFORMED = 1,
  BCIR_TEV_REASON_STREAMS = 2,
  BCIR_TEV_REASON_UNKNOWN = 3,
  BCIR_TEV_REASON_STALE = 4,
  BCIR_TEV_REASON_AHEAD = 5
} bcir_tev_reason;

typedef struct bcir_tev_outcome {
  bcir_tev_verdict verdict;
  bcir_tev_reason reason;
  bcir_status status;
  bcir_seq_class classification;
} bcir_tev_outcome;

typedef struct bcir_tev_stream {
  uint64_t source;
  uint64_t session;
  bcir_seq_tracker tracker;
} bcir_tev_stream;

typedef struct bcir_tev_intake {
  uint32_t live_generation;
  uint32_t n_streams;
  bcir_tev_stream streams[BCIR_TEV_STREAMS];
  uint64_t accepted;
  uint64_t skipped;
  uint64_t refused;
  uint64_t malformed;
  uint64_t unknown;
  uint64_t stale;
  uint64_t ahead;
  uint64_t streams_full;
  uint64_t reported_lost;
} bcir_tev_intake;

typedef struct bcir_tev_report {
  uint64_t accepted, skipped, refused, malformed, unknown, stale, ahead, streams_full;
  uint64_t reported_lost, missing, reordered, duplicated;
  uint32_t streams;
} bcir_tev_report;

void bcir_tev_intake_init(bcir_tev_intake *in, uint32_t live_generation);
void bcir_tev_intake_set_live(bcir_tev_intake *in, uint32_t live_generation);
/* Decide one record: the wire laws, the stream table, continuity (before any semantic refusal),
 * the signal (unknown REQUIRED refused, unknown optional skipped), the generation (stale/ahead
 * refused; 0 unbound), accepted. `env` (may be NULL) receives the decoded record. */
bcir_tev_outcome bcir_tev_intake_admit(bcir_tev_intake *in, const uint8_t *data, size_t len,
                                       bcir_tev *env);
void bcir_tev_intake_report(const bcir_tev_intake *in, bcir_tev_report *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_TELEMETRY_ENVELOPE_H */
