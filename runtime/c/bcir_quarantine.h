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
 * site->source table realized inline as a string literal the frontend emits at the guard. RETURNS the
 * index the access should use: the WEAK default records + aborts (it never returns); a STRONG override may
 * RECOVER by returning an in-bounds index -- but only through a recorded `decide` (see bcir_decision). */
size_t bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site);

/* The checked index used by the emitted C: `i` if in-bounds, else run the handler and use the index it
 * RETURNS (the weak default aborts and never returns; a recovery override returns a clamped in-bounds
 * index). A single evaluation of `i` is assumed (the frontend feeds a temp). */
#define BCIR_CHK(rid, i, n, site) \
    ((uint64_t)(i) < (uint64_t)(n) ? (size_t)(i) \
     : bcir_bounds_quarantine((uint64_t)(rid), (uint64_t)(i), (uint64_t)(n), (site)))

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

/* --- the two-truth crossing: a RECORDED `decide` (§5.12 + LANGREF §14) -------------------------------
 * The weak default aborts; a recovery override turns an OOB access into a CLASSICAL action (abort / clamp).
 * That action is a graded->classical crossing -- the ML-layer's recovery confidence (graded truth) must
 * never silently become the access. The ONLY sanctioned crossing is a `decide`: collapse a graded
 * proposition `(action, confidence)` at a FROZEN threshold into a classical value, and RECORD it. This ring
 * is the C twin of `kbcir.twotruth.Decision` -- the auditable trail R13 (`verify.verify_quarantine`) speaks
 * of, so the crossing is never silent. Reading it never decides legality; it is pure observation. */
typedef enum { BCIR_RECOVER_ABORT = 0, BCIR_RECOVER_CLAMP = 1 } bcir_recover_action;

typedef struct {
    uint64_t rid, index, extent;     /* the OOB access this decision is about */
    const char *site;                /* the "<func>:<array>" handle */
    uint32_t confidence_milli;       /* the graded proposition's confidence, Q-milli [0,1000] */
    uint32_t threshold_milli;        /* the FROZEN decide threshold, Q-milli [0,1000] */
    int admitted;                    /* confidence >= threshold: the graded truth cleared the threshold */
    int action;                      /* the classical bcir_recover_action applied (abort / clamp) */
    size_t recovered_index;          /* the in-bounds index the access used (0 if it aborted) */
} bcir_decision;
#define BCIR_DECIDE_RING 64

/* The decide-audit surface: the ring of the most recent recovery crossings + the running total, and a
 * reader (oldest-retained first). An override appends via bcir_decide_record_event; the weak default never
 * crosses (it aborts unconditionally, with no graded input), so it records nothing here. */
extern bcir_decision bcir_decide_ring[BCIR_DECIDE_RING];
extern volatile unsigned long bcir_decide_count;
void bcir_decide_report(FILE *f);
void bcir_decide_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site,
                              uint32_t confidence_milli, uint32_t threshold_milli,
                              int admitted, int action, size_t recovered_index);

#endif /* BCIR_QUARANTINE_H */
