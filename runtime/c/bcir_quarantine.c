/* The WEAK default bounds-quarantine handler (§5.12). Records the out-of-bounds access provenance into a
 * fixed ring (so a debugger can read the last BCIR_OOB_RING events via bcir_quarantine_report) and aborts
 * -- fail-fast safety. The ML-layer / debugger overrides `bcir_bounds_quarantine` with a STRONG definition
 * to recover / apply a graded-truth policy through a recorded `decide` instead of aborting; the ring +
 * reader below stay the observability layer regardless of which handler is linked. */
#include "bcir_quarantine.h"
#include <stdio.h>
#include <stdlib.h>

bcir_oob_record bcir_oob_ring[BCIR_OOB_RING];
volatile unsigned long bcir_oob_count;          /* total OOB events; the ring index is count % BCIR_OOB_RING */

/* Record one event into the ring (shared by the weak default and any override that wants the trace). */
void bcir_oob_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    unsigned slot = (unsigned)(bcir_oob_count++ % BCIR_OOB_RING);
    bcir_oob_ring[slot].rid = rid;
    bcir_oob_ring[slot].index = index;
    bcir_oob_ring[slot].extent = extent;
    bcir_oob_ring[slot].site = site;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    bcir_oob_record_event(rid, index, extent, site);
    fprintf(stderr, "BCIR bounds-quarantine: %s resource %llu index %llu out of [0, %llu)\n",
            site ? site : "?", (unsigned long long)rid,
            (unsigned long long)index, (unsigned long long)extent);
    abort();
}

/* Debugger trace reader: print the retained OOB events (oldest-first within the ring window) to `f`. The
 * ring keeps the most recent BCIR_OOB_RING events; the running total `bcir_oob_count` is reported even when
 * older events have scrolled out. Pure observation -- it never decides legality. */
void bcir_quarantine_report(FILE *f) {
    unsigned long total = bcir_oob_count;
    unsigned long shown = total < BCIR_OOB_RING ? total : BCIR_OOB_RING;
    unsigned long start = total < BCIR_OOB_RING ? 0UL : total % BCIR_OOB_RING;   /* oldest retained slot */
    fprintf(f, "BCIR quarantine report: %lu out-of-bounds event(s)\n", total);
    for (unsigned long k = 0; k < shown; k++) {
        const bcir_oob_record *r = &bcir_oob_ring[(start + k) % BCIR_OOB_RING];
        fprintf(f, "  [%lu] %s resource %llu index %llu out of [0, %llu)\n",
                total - shown + k, r->site ? r->site : "?", (unsigned long long)r->rid,
                (unsigned long long)r->index, (unsigned long long)r->extent);
    }
}
