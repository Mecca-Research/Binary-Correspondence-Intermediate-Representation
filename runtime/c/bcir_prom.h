/*===- bcir_prom.h - the telemetry export's Prometheus text exposition --------------------------------------===
 *
 * N13 of the native port audit (docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md): the C twin of
 * bcir/telemetry_export.py's `to_prometheus`, the body a native telemetry or ledger service returns to
 * `GET /metrics` -- octet for octet the oracle's for the same snapshot, so a scrape cannot tell the two rails
 * apart.
 *
 * What it writes, in the oracle's order: the availability block (`bcir_signal_up`, one series a signal, 1 when it
 * reads and 0 when it does not, labelled from its definition when one is known); then for each signal that reads
 * a HELP line (its description, newlines folded and Python's whitespace stripped, or its definition's name), a
 * TYPE line (`counter` for a counter definition, else `gauge`) and its value; then the sensitivity ranking's
 * gauges; then the integrity witness -- the record counters, the frame counters and the frame-order gauge when
 * any frame counter is nonzero, and the `blind` gauge. Metric names are mapped to Prometheus's charset as
 * `sanitize_metric_name` maps them, one `_` a code point outside it; label values escape a backslash, a quote
 * and a newline.
 *
 * Integers only. A native exporter carries what native sources measure -- counters, levels, Q-free ratios in
 * milli-units -- and the oracle's integer rendering is `str(int)`; a float's rendering is Python's shortest
 * round-trip `repr`, which this file does not reimplement, so a float series, and the T3 derived metrics, stay
 * the oracle's. The signals come in the oracle's order: names strictly ascending by octet (code point order),
 * each valid UTF-8; anything else is refused (C only).
 *
 * Freestanding: <stddef.h> and <stdint.h>, no heap, no libc, no recursion. The caller owns the output; a short one
 * says how long the text is and writes nothing.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_PROM_H
#define BCIR_PROM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  BCIR_PROM_OK = 0,
  BCIR_PROM_ERR_ARG = 1,   /* a NULL, a reading without its definition */
  BCIR_PROM_ERR_ORDER = 2, /* the signals' names not strictly ascending (C only) */
  BCIR_PROM_ERR_UTF8 = 3,  /* a text that is not UTF-8 (C only) */
  BCIR_PROM_ERR_SPACE = 4  /* the output is short: *len is the text's length, nothing written */
};

typedef struct bcir_prom_text {
  const uint8_t *p;
  size_t n;
} bcir_prom_text;

/* One signal of the snapshot: its name (the snapshot's key), whether it read, and its definition. */
typedef struct bcir_prom_series {
  bcir_prom_text name;
  int available;      /* a reading exists: value and provenance are the reading's */
  int has_definition; /* the definition is known (always when available) */
  int counter;        /* metric_kind == counter */
  bcir_prom_text definition_name, description, cost_dim, provenance, unit; /* cost_dim empty: "none" */
  int64_t value;
} bcir_prom_series;

typedef struct bcir_prom_sensitivity {
  bcir_prom_text signal;
  int64_t abs_delta;
} bcir_prom_sensitivity;

/* The integrity witness's counters; frame order and blindness follow from them, as the oracle derives them. */
typedef struct bcir_prom_witness {
  uint64_t accepted, rejected, dropped;
  uint64_t frames_accepted, frames_rejected, frames_missing, frames_reordered, frames_duplicated;
} bcir_prom_witness;

/* The exposition of series[0 .. n) for `channel`, with the sensitivity rows and the witness (either may be
 * absent: n_sensitivity 0, witness NULL), into out[0 .. cap); its length in *len. */
int bcir_prom_write(const bcir_prom_series *series, size_t n, bcir_prom_text channel,
                    const bcir_prom_sensitivity *sensitivity, size_t n_sensitivity, const bcir_prom_witness *witness,
                    uint8_t *out, size_t cap, size_t *len);

#ifdef __cplusplus
}
#endif

#endif
