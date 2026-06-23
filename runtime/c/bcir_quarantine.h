#ifndef BCIR_QUARANTINE_H
#define BCIR_QUARANTINE_H
/* Bounds-quarantine runtime ABI (§5.12). The C frontend promotes a known-extent local/static array access
 * to `masked` (runtime-bounds-checked); the emitted C guards the index and, on an out-of-bounds value,
 * calls `bcir_bounds_quarantine(rid, index, extent, site)`. The WEAK default here records the access
 * provenance into a fixed ring (the debugger reads it via `bcir_quarantine_report`) and aborts -- fail-fast.
 * The ML-layer / debugger OVERRIDES the weak symbol with a STRONG definition to consult the graded-truth
 * quarantine and recover / log / apply policy (a follow-on; the only sanctioned two-truth crossing is a
 * recorded `decide`). In-bounds accesses never call it, so a promoted access is behaviour-identical to the
 * raw `a[i]` for any defined input. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* The out-of-bounds handler. `rid` is the numeric resource provenance (stable per translation unit) and
 * `site` is a static "<function>:<array>" source-site handle the debugger / ML-layer resolves -- the
 * site->source table realized inline as a string literal the frontend emits at the guard. */
void bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site);

/* The checked index used by the emitted C: `i` if in-bounds, else run the handler (default: record + abort)
 * and fall back to 0 if it returns. A single evaluation of `i` is assumed (the frontend feeds a temp). */
#define BCIR_CHK(rid, i, n, site) \
    ((uint64_t)(i) < (uint64_t)(n) ? (size_t)(i) \
     : (bcir_bounds_quarantine((uint64_t)(rid), (uint64_t)(i), (uint64_t)(n), (site)), (size_t)0))

/* A record of one out-of-bounds access, for the debugger / ML-layer to inspect. `site` points at the
 * emitted "<function>:<array>" string literal (static storage; valid for the program lifetime). */
typedef struct { uint64_t rid, index, extent; const char *site; } bcir_oob_record;
#define BCIR_OOB_RING 64

/* The debugger trace surface: the fixed ring of the most recent OOB events and the running total, plus a
 * reader that prints them (oldest-retained first) to `f`. The handler may be overridden, but these stay the
 * observability layer -- reading the ring never crosses the two-truth line into a legality verdict. */
extern bcir_oob_record bcir_oob_ring[BCIR_OOB_RING];
extern volatile unsigned long bcir_oob_count;
void bcir_quarantine_report(FILE *f);

/* Append one event to the ring (the weak default uses it; an override may reuse it before applying policy,
 * so the trace stays populated regardless of which handler is linked). */
void bcir_oob_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site);

#endif /* BCIR_QUARANTINE_H */
