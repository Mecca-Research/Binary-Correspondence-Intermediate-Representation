/* The WEAK default bounds-quarantine handler (§5.12). Records the out-of-bounds access provenance into a
 * fixed ring (so a debugger can read the last BCIR_OOB_RING events) and aborts -- fail-fast safety. The
 * ML-layer / debugger overrides `bcir_bounds_quarantine` with a STRONG definition to recover / apply a
 * graded-truth policy through a recorded `decide` instead of aborting. */
#include "bcir_quarantine.h"
#include <stdio.h>
#include <stdlib.h>

bcir_oob_record bcir_oob_ring[BCIR_OOB_RING];
volatile unsigned long bcir_oob_count;          /* total OOB events; the ring index is count % BCIR_OOB_RING */

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent) {
    unsigned slot = (unsigned)(bcir_oob_count++ % BCIR_OOB_RING);
    bcir_oob_ring[slot].rid = rid;
    bcir_oob_ring[slot].index = index;
    bcir_oob_ring[slot].extent = extent;
    fprintf(stderr, "BCIR bounds-quarantine: resource %llu index %llu out of [0, %llu)\n",
            (unsigned long long)rid, (unsigned long long)index, (unsigned long long)extent);
    abort();
}
