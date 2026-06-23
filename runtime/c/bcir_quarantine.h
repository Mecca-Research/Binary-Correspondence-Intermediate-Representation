#ifndef BCIR_QUARANTINE_H
#define BCIR_QUARANTINE_H
/* Bounds-quarantine runtime ABI (§5.12). The C frontend promotes a known-extent local/static array access
 * to `masked` (runtime-bounds-checked); the emitted C guards the index and, on an out-of-bounds value,
 * calls `bcir_bounds_quarantine(rid, index, extent)`. The WEAK default here records the access provenance
 * into a fixed ring (the debugger reads it) and aborts -- fail-fast. The ML-layer / debugger OVERRIDES the
 * weak symbol with a STRONG definition to consult the graded-truth quarantine and recover / log / apply
 * policy (a follow-on; the only sanctioned two-truth crossing is a recorded `decide`). In-bounds accesses
 * never call it, so a promoted access is behaviour-identical to the raw `a[i]` for any defined input. */
#include <stdint.h>
#include <stddef.h>

void bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent);

/* The checked index used by the emitted C: `i` if in-bounds, else run the handler (default: record + abort)
 * and fall back to 0 if it returns. A single evaluation of `i` is assumed (the frontend feeds a temp). */
#define BCIR_CHK(rid, i, n) \
    ((uint64_t)(i) < (uint64_t)(n) ? (size_t)(i) \
     : (bcir_bounds_quarantine((uint64_t)(rid), (uint64_t)(i), (uint64_t)(n)), (size_t)0))

/* A record of one out-of-bounds access, for the debugger / ML-layer to inspect. */
typedef struct { uint64_t rid, index, extent; } bcir_oob_record;
#define BCIR_OOB_RING 64

#endif /* BCIR_QUARANTINE_H */
