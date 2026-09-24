/*===- bcir_telemetry_envelope.c - TelemetryEnvelopeV0 + the host intake ---===
 *
 * The C twin of bcir/abi/telemetry_envelope.py, bcir/telemetry_intake.py and
 * bcir.telemetry.SequenceTracker (see bcir_telemetry_envelope.h). Every field is read and
 * written byte by byte, little-endian, so the rails agree on any host.
 *===----------------------------------------------------------------------===*/
#include "bcir_telemetry_envelope.h"

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

static size_t kind_size(uint8_t kind) {
  return kind == BCIR_TEV_SAMPLE ? BCIR_TEV_SAMPLE_SIZE : BCIR_TEV_DATADNA_SIZE;
}

/* The clock/unit pairing: no clock counts in no unit; the calendar-free clocks count in ns or
 * us; a cycle counter counts cycles. */
static int unit_fits_clock(uint8_t clock, uint8_t unit) {
  switch (clock) {
    case BCIR_TEV_CLOCK_NONE: return unit == BCIR_TEV_UNIT_NONE;
    case BCIR_TEV_CLOCK_MONOTONIC:
    case BCIR_TEV_CLOCK_REALTIME:
    case BCIR_TEV_CLOCK_BOOT: return unit == BCIR_TEV_UNIT_NS || unit == BCIR_TEV_UNIT_US;
    case BCIR_TEV_CLOCK_CYCLES: return unit == BCIR_TEV_UNIT_CYCLES;
    default: return 0;
  }
}

/* The field laws both the decoder and the encoder apply (all BCIR_ERR_TELEMETRY). */
static bcir_status field_laws(const bcir_tev *e) {
  if (e->kind != BCIR_TEV_SAMPLE && e->kind != BCIR_TEV_DATADNA) return BCIR_ERR_TELEMETRY;
  if (e->required > 1u) return BCIR_ERR_TELEMETRY;
  if (e->clock > BCIR_TEV_CLOCK_MAX || e->unit > BCIR_TEV_UNIT_MAX) return BCIR_ERR_TELEMETRY;
  if (!unit_fits_clock(e->clock, e->unit)) return BCIR_ERR_TELEMETRY;
  if (e->clock == BCIR_TEV_CLOCK_NONE && e->timestamp != 0u) return BCIR_ERR_TELEMETRY;
  if (e->source == 0u || e->session == 0u) return BCIR_ERR_TELEMETRY;
  if (e->kind == BCIR_TEV_SAMPLE) {
    if (e->signal == 0u || e->signal == BCIR_TEV_SIGNAL_RESERVED) return BCIR_ERR_TELEMETRY;
  } else {
    if (e->signal != 0u || e->required || e->generation == 0u || e->value != 0)
      return BCIR_ERR_TELEMETRY;
  }
  return BCIR_OK;
}

bcir_status bcir_tev_decode(const uint8_t *data, size_t len, bcir_tev *out) {
  if (!data || len < BCIR_TEV_HEADER) return BCIR_ERR_TRUNCATED;
  for (unsigned i = 0; i < 4; i++)
    if (data[i] != (uint8_t)BCIR_TEV_MAGIC[i]) return BCIR_ERR_MAGIC;
  if (rd16(data + 4) != BCIR_TEV_VERSION) return BCIR_ERR_VERSION;
  uint16_t flags = rd16(data + 6);
  if ((flags & (uint16_t)~BCIR_TEV_FLAG_REQUIRED) != 0u) return BCIR_ERR_RESERVED;
  uint8_t kind = data[8];
  if (kind != BCIR_TEV_SAMPLE && kind != BCIR_TEV_DATADNA) return BCIR_ERR_TELEMETRY;
  size_t size = kind_size(kind);
  if (rd16(data + 12) != size) return BCIR_ERR_TELEMETRY;
  if (len < size) return BCIR_ERR_TRUNCATED;
  if (len > size) return BCIR_ERR_TRAILING;
  if (rd32(data + size - BCIR_TEV_CRC) != bcir_crc32(data, size - BCIR_TEV_CRC)) return BCIR_ERR_CRC;
  if (rd16(data + 14) != 0u || rd64(data + 56) != 0u) return BCIR_ERR_RESERVED;
  if (data[9] != 0u) return BCIR_ERR_TELEMETRY; /* schema */
  bcir_tev e;
  e.kind = kind;
  e.clock = data[10];
  e.unit = data[11];
  e.required = (uint8_t)(flags & BCIR_TEV_FLAG_REQUIRED);
  e.source = rd64(data + 16);
  e.session = rd64(data + 24);
  e.generation = rd32(data + 32);
  e.signal = rd32(data + 36);
  e.seq = rd32(data + 40);
  e.lost = rd32(data + 44);
  e.timestamp = rd64(data + 48);
  e.value = 0;
  for (unsigned i = 0; i < 7; i++) e.record[i] = 0;
  if (kind == BCIR_TEV_SAMPLE) {
    e.value = (int64_t)rd64(data + BCIR_TEV_HEADER);
  } else {
    for (unsigned i = 0; i < 7; i++) e.record[i] = (int64_t)rd64(data + BCIR_TEV_HEADER + 8u * i);
  }
  bcir_status st = field_laws(&e);
  if (st != BCIR_OK) return st;
  if (out) *out = e;
  return BCIR_OK;
}

bcir_status bcir_tev_encode(const bcir_tev *env, uint8_t *out, size_t cap, size_t *written) {
  if (!env) return BCIR_ERR_TELEMETRY;
  bcir_status st = field_laws(env);
  if (st != BCIR_OK) return st;
  size_t size = kind_size(env->kind);
  if (!out || cap < size) return BCIR_ERR_NOSPACE;
  for (size_t i = 0; i < size; i++) out[i] = 0;
  for (unsigned i = 0; i < 4; i++) out[i] = (uint8_t)BCIR_TEV_MAGIC[i];
  wr16(out + 4, (uint16_t)BCIR_TEV_VERSION);
  wr16(out + 6, env->required ? (uint16_t)BCIR_TEV_FLAG_REQUIRED : (uint16_t)0u);
  out[8] = env->kind;
  out[10] = env->clock;
  out[11] = env->unit;
  wr16(out + 12, (uint16_t)size);
  wr64(out + 16, env->source);
  wr64(out + 24, env->session);
  wr32(out + 32, env->generation);
  wr32(out + 36, env->signal);
  wr32(out + 40, env->seq);
  wr32(out + 44, env->lost);
  wr64(out + 48, env->timestamp);
  if (env->kind == BCIR_TEV_SAMPLE) {
    wr64(out + BCIR_TEV_HEADER, (uint64_t)env->value);
  } else {
    for (unsigned i = 0; i < 7; i++) wr64(out + BCIR_TEV_HEADER + 8u * i, (uint64_t)env->record[i]);
  }
  wr32(out + size - BCIR_TEV_CRC, bcir_crc32(out, size - BCIR_TEV_CRC));
  if (written) *written = size;
  return BCIR_OK;
}

/* ---- continuity ------------------------------------------------------------------------ */

void bcir_seq_init(bcir_seq_tracker *t) {
  if (!t) return;
  bcir_seq_tracker zero = {0};
  *t = zero;
}

bcir_seq_class bcir_seq_observe(bcir_seq_tracker *t, uint32_t seq) {
  if (!t) return BCIR_SEQ_NONE;
  t->observed++;
  if (!t->has_watermark) {
    t->has_watermark = 1;
    t->watermark = seq;
    return BCIR_SEQ_FIRST;
  }
  uint32_t delta = seq - t->watermark; /* modulo 2^32: 0xffffffff -> 0 is continuous */
  if (delta == 0u) {
    t->duplicated++;
    return BCIR_SEQ_DUPLICATE;
  }
  if (delta < 0x80000000u) {
    t->missing += (uint64_t)delta - 1u;
    t->watermark = seq;
    return delta == 1u ? BCIR_SEQ_NEXT : BCIR_SEQ_GAP;
  }
  t->reordered++;
  return BCIR_SEQ_REORDER;
}

/* ---- the generated signal table ----------------------------------------------------------- */

const bcir_signal_def *bcir_signal_find(uint32_t id) {
  size_t lo = 0, hi = BCIR_SIGNAL_COUNT;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2u;
    uint32_t at = BCIR_SIGNAL_TABLE[mid].id;
    if (at == id) return &BCIR_SIGNAL_TABLE[mid];
    if (at < id) lo = mid + 1u;
    else hi = mid;
  }
  return 0;
}

void bcir_signal_row_encode(const bcir_signal_def *def, uint8_t out[BCIR_SIGNAL_ROW_SIZE]) {
  for (unsigned i = 0; i < BCIR_SIGNAL_ROW_SIZE; i++) out[i] = 0;
  if (!def) return;
  wr32(out, def->id);
  out[4] = def->unit;
  out[5] = def->kind;
  out[6] = def->temporality;
  out[7] = def->flags;
  out[8] = def->sampling;
  out[9] = def->provenance;
  out[10] = def->cost_dim;
  wr64(out + 16, def->min_interval_ns);
  for (unsigned i = 0; i < BCIR_SIGNAL_NAME_WIDTH && def->name[i] != '\0'; i++)
    out[24u + i] = (uint8_t)def->name[i];
}

/* ---- the intake ------------------------------------------------------------------------- */

void bcir_tev_intake_init(bcir_tev_intake *in, uint32_t live_generation) {
  if (!in) return;
  bcir_tev_intake zero = {0};
  *in = zero;
  in->live_generation = live_generation;
}

void bcir_tev_intake_set_live(bcir_tev_intake *in, uint32_t live_generation) {
  if (in) in->live_generation = live_generation;
}

static bcir_tev_outcome decided(bcir_tev_verdict verdict, bcir_tev_reason reason,
                                bcir_status status, bcir_seq_class classification) {
  bcir_tev_outcome o;
  o.verdict = verdict;
  o.reason = reason;
  o.status = status;
  o.classification = classification;
  return o;
}

bcir_tev_outcome bcir_tev_intake_admit(bcir_tev_intake *in, const uint8_t *data, size_t len,
                                       bcir_tev *env) {
  if (!in) return decided(BCIR_TEV_REFUSED, BCIR_TEV_REASON_MALFORMED, BCIR_ERR_TELEMETRY, BCIR_SEQ_NONE);
  bcir_tev e;
  bcir_status st = bcir_tev_decode(data, len, &e);
  if (st != BCIR_OK) {
    in->malformed++;
    in->refused++;
    return decided(BCIR_TEV_REFUSED, BCIR_TEV_REASON_MALFORMED, st, BCIR_SEQ_NONE);
  }
  if (env) *env = e;
  bcir_tev_stream *stream = 0;
  for (uint32_t i = 0; i < in->n_streams; i++) {
    if (in->streams[i].source == e.source && in->streams[i].session == e.session) {
      stream = &in->streams[i];
      break;
    }
  }
  if (!stream) {
    if (in->n_streams >= BCIR_TEV_STREAMS) {
      in->streams_full++;
      in->refused++;
      return decided(BCIR_TEV_REFUSED, BCIR_TEV_REASON_STREAMS, BCIR_ERR_NOSPACE, BCIR_SEQ_NONE);
    }
    stream = &in->streams[in->n_streams++];
    stream->source = e.source;
    stream->session = e.session;
    bcir_seq_init(&stream->tracker);
  }
  bcir_seq_class cls = bcir_seq_observe(&stream->tracker, e.seq);
  in->reported_lost += e.lost;
  if (e.kind == BCIR_TEV_SAMPLE && !bcir_signal_find(e.signal)) {
    if (e.required) {
      in->unknown++;
      in->refused++;
      return decided(BCIR_TEV_REFUSED, BCIR_TEV_REASON_UNKNOWN, BCIR_ERR_TELEMETRY, cls);
    }
    in->skipped++;
    return decided(BCIR_TEV_SKIPPED, BCIR_TEV_REASON_UNKNOWN, BCIR_OK, cls);
  }
  if (e.generation != 0u) {
    if (e.generation < in->live_generation) { /* is_stale: minted against a generation left */
      in->stale++;
      in->refused++;
      return decided(BCIR_TEV_REFUSED, BCIR_TEV_REASON_STALE, BCIR_ERR_STALE, cls);
    }
    if (e.generation > in->live_generation) {
      in->ahead++;
      in->refused++;
      return decided(BCIR_TEV_REFUSED, BCIR_TEV_REASON_AHEAD, BCIR_ERR_STALE, cls);
    }
  }
  in->accepted++;
  return decided(BCIR_TEV_ACCEPTED, BCIR_TEV_REASON_NONE, BCIR_OK, cls);
}

void bcir_tev_intake_report(const bcir_tev_intake *in, bcir_tev_report *out) {
  if (!in || !out) return;
  bcir_tev_report r = {0};
  r.accepted = in->accepted;
  r.skipped = in->skipped;
  r.refused = in->refused;
  r.malformed = in->malformed;
  r.unknown = in->unknown;
  r.stale = in->stale;
  r.ahead = in->ahead;
  r.streams_full = in->streams_full;
  r.reported_lost = in->reported_lost;
  r.streams = in->n_streams;
  for (uint32_t i = 0; i < in->n_streams; i++) {
    r.missing += in->streams[i].tracker.missing;
    r.reordered += in->streams[i].tracker.reordered;
    r.duplicated += in->streams[i].tracker.duplicated;
  }
  *out = r;
}
