/* The WEAK default bounds-quarantine handler (§5.12). Records the out-of-bounds access provenance into a
 * fixed ring (so a debugger can read the last BCIR_OOB_RING events via bcir_quarantine_report) and aborts
 * -- fail-fast safety. The ML-layer / debugger overrides `bcir_bounds_quarantine` with a STRONG definition
 * to recover / apply a graded-truth policy through a recorded `decide` instead of aborting; the ring +
 * reader below stay the observability layer regardless of which handler is linked. */
#include "bcir_quarantine.h"
#include <stdio.h>
#include <stdlib.h>

bcir_oob_record bcir_oob_ring[BCIR_OOB_RING];
bcir_oob_counter bcir_oob_count;                /* total OOB events; the ring index is count % BCIR_OOB_RING */

/* The atomic counter alone does not make the payload slots race-free: a reporter can otherwise read a
 * slot while a writer is updating its non-atomic fields, which is undefined behaviour in the C memory
 * model.  Serialize each short publish/snapshot critical section on hosted atomic builds.  The documented
 * freestanding fallback is single-threaded, so its lock is a no-op. */
#if BCIR_OOB_COUNTER_ATOMIC
static atomic_flag bcir_oob_ring_lock = ATOMIC_FLAG_INIT;
static atomic_flag bcir_decide_ring_lock = ATOMIC_FLAG_INIT;
static void ring_lock(atomic_flag *lock) {
    while (atomic_flag_test_and_set_explicit(lock, memory_order_acquire)) { }
}
static void ring_unlock(atomic_flag *lock) {
    atomic_flag_clear_explicit(lock, memory_order_release);
}
#define OOB_LOCK() ring_lock(&bcir_oob_ring_lock)
#define OOB_UNLOCK() ring_unlock(&bcir_oob_ring_lock)
#define DECIDE_LOCK() ring_lock(&bcir_decide_ring_lock)
#define DECIDE_UNLOCK() ring_unlock(&bcir_decide_ring_lock)
#else
#define OOB_LOCK() ((void)0)
#define OOB_UNLOCK() ((void)0)
#define DECIDE_LOCK() ((void)0)
#define DECIDE_UNLOCK() ((void)0)
#endif

/* Record one event into the ring (shared by the weak default and any override that wants the trace). The
 * slot is claimed atomically (bcir_counter_next returns the PRE-increment total), so two concurrent OOB
 * events land in DISTINCT slots and neither audit record is lost -- the non-atomic `count++` raced here. */
void bcir_oob_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    OOB_LOCK();
    unsigned slot = (unsigned)(bcir_counter_next(&bcir_oob_count) % BCIR_OOB_RING);
    bcir_oob_ring[slot].rid = rid;
    bcir_oob_ring[slot].index = index;
    bcir_oob_ring[slot].extent = extent;
    bcir_oob_ring[slot].site = site;
    OOB_UNLOCK();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
size_t bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    bcir_oob_record_event(rid, index, extent, site);
    fprintf(stderr, "BCIR bounds-quarantine: %s resource %llu index %llu out of [0, %llu)\n",
            site ? site : "?", (unsigned long long)rid,
            (unsigned long long)index, (unsigned long long)extent);
    abort();
    return 0;                                       /* unreachable (abort is noreturn); satisfies the return type */
}

/* The WEAK default WRITE handler. An OOB store is NEVER recoverable by clamping -- a clamped store would
 * silently corrupt a[extent-1] (a valid element the program owns), which is data corruption, not recovery.
 * So this handler is `noreturn`: it records the provenance + aborts (fail-fast). A strong override may
 * record a richer `decide` first (e.g. a rejected clamp proposal), but it too must not let the store land
 * on a redirected element -- the only legal classical write outcomes are abort / reject, both terminating. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak, noreturn))     /* noreturn on the DEFINITION too: a future edit that adds a returning
                                     * (recovering) path to a WRITE handler is then a compile error -- the
                                     * write-never-clamps invariant is enforced where it can break. */
#endif
void bcir_bounds_quarantine_write(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    bcir_oob_record_event(rid, index, extent, site);
    fprintf(stderr, "BCIR bounds-quarantine (WRITE): %s resource %llu index %llu out of [0, %llu) -- "
            "an out-of-bounds store cannot be clamped (would corrupt a valid element); aborting\n",
            site ? site : "?", (unsigned long long)rid,
            (unsigned long long)index, (unsigned long long)extent);
    abort();
}

/* --- the decide-audit ring: the recorded two-truth crossings (§5.12 + LANGREF §14) --- */
bcir_decision bcir_decide_ring[BCIR_DECIDE_RING];
bcir_oob_counter bcir_decide_count;                 /* total crossings; ring slot is count % BCIR_DECIDE_RING */

/* Record one recovery `decide` -- a graded->classical collapse an override applied. Keeping it auditable is
 * the whole point: the crossing is never silent. (The weak default never calls this; it aborts outright.) */
void bcir_decide_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site,
                              uint32_t confidence_milli, uint32_t threshold_milli,
                              int admitted, int action, size_t recovered_index) {
    confidence_milli = confidence_milli > 1000u ? 1000u : confidence_milli;
    threshold_milli = threshold_milli > 1000u ? 1000u : threshold_milli;
    admitted = admitted ? 1 : 0;
    if (!admitted || action != BCIR_RECOVER_CLAMP) {
        action = BCIR_RECOVER_ABORT;
        recovered_index = 0;
    }
    DECIDE_LOCK();
    unsigned slot = (unsigned)(bcir_counter_next(&bcir_decide_count) % BCIR_DECIDE_RING);
    bcir_decision *d = &bcir_decide_ring[slot];
    d->rid = rid; d->index = index; d->extent = extent; d->site = site;
    d->confidence_milli = confidence_milli; d->threshold_milli = threshold_milli;
    d->admitted = admitted; d->action = action; d->recovered_index = recovered_index;
    DECIDE_UNLOCK();
}

/* Audit reader: print the retained recovery decisions (oldest-first in the ring window). Each line names the
 * graded confidence, the frozen threshold, whether it was admitted, and the classical action taken -- the
 * trail that witnesses every graded->classical crossing. Pure observation; never decides legality. */
void bcir_decide_report(FILE *f) {
    bcir_decision snapshot[BCIR_DECIDE_RING];
    if (!f) return;
    DECIDE_LOCK();
    unsigned long total = bcir_counter_load(&bcir_decide_count);
    unsigned long shown = total < BCIR_DECIDE_RING ? total : BCIR_DECIDE_RING;
    unsigned long start = total < BCIR_DECIDE_RING ? 0UL : total % BCIR_DECIDE_RING;
    for (unsigned long k = 0; k < shown; k++) {
        snapshot[k] = bcir_decide_ring[(start + k) % BCIR_DECIDE_RING];
    }
    DECIDE_UNLOCK();
    fprintf(f, "BCIR decide report: %lu recovery crossing(s)\n", total);
    for (unsigned long k = 0; k < shown; k++) {
        const bcir_decision *d = &snapshot[k];
        fprintf(f, "  [%lu] %s index %llu of [0, %llu): confidence %u/1000 vs threshold %u/1000 -> %s, %s",
                total - shown + k, d->site ? d->site : "?", (unsigned long long)d->index,
                (unsigned long long)d->extent, d->confidence_milli, d->threshold_milli,
                d->admitted ? "admitted" : "rejected",
                d->action == BCIR_RECOVER_CLAMP ? "clamp" : "abort");
        if (d->action == BCIR_RECOVER_CLAMP)
            fprintf(f, " to index %llu", (unsigned long long)d->recovered_index);
        fprintf(f, "\n");
    }
}

/* Debugger trace reader: print the retained OOB events (oldest-first within the ring window) to `f`. The
 * ring keeps the most recent BCIR_OOB_RING events; the running total `bcir_oob_count` is reported even when
 * older events have scrolled out. Pure observation -- it never decides legality. */
void bcir_quarantine_report(FILE *f) {
    bcir_oob_record snapshot[BCIR_OOB_RING];
    if (!f) return;
    OOB_LOCK();
    unsigned long total = bcir_counter_load(&bcir_oob_count);
    unsigned long shown = total < BCIR_OOB_RING ? total : BCIR_OOB_RING;
    unsigned long start = total < BCIR_OOB_RING ? 0UL : total % BCIR_OOB_RING;   /* oldest retained slot */
    for (unsigned long k = 0; k < shown; k++) {
        snapshot[k] = bcir_oob_ring[(start + k) % BCIR_OOB_RING];
    }
    OOB_UNLOCK();
    fprintf(f, "BCIR quarantine report: %lu out-of-bounds event(s)\n", total);
    for (unsigned long k = 0; k < shown; k++) {
        const bcir_oob_record *r = &snapshot[k];
        fprintf(f, "  [%lu] %s resource %llu index %llu out of [0, %llu)\n",
                total - shown + k, r->site ? r->site : "?", (unsigned long long)r->rid,
                (unsigned long long)r->index, (unsigned long long)r->extent);
    }
}
