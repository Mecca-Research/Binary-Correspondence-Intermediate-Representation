/* The REFERENCE recovery override of bcir_bounds_quarantine (§5.12) -- the ML-layer / debugger seam made
 * concrete. It turns an out-of-bounds access into a CLASSICAL action through a recorded `decide`, the only
 * sanctioned two-truth crossing (LANGREF §14): the ML-layer's recovery confidence is GRADED truth and may
 * never silently become the access; it crosses only as the classical value of an audited collapse.
 *
 * On an OOB access:
 *   1. trace it (bcir_oob_record_event), so the debugger surface stays populated;
 *   2. consult the FROZEN policy -> a graded proposition `(action, confidence)` for the site (a table
 *      lookup, NOT live inference -- L1 frozen tables, respecting the L0 hot-path prohibition);
 *   3. `decide`: admitted := confidence >= frozen threshold; the classical action is the proposed one iff
 *      admitted AND it is a clamp, else abort (not confident enough to recover -> fail-fast);
 *   4. RECORD the crossing (bcir_decide_record_event) -- never silent;
 *   5. apply: clamp -> return an in-bounds index; abort -> fail-fast. */
#include "bcir_quarantine_recover.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const bcir_recover_rule *g_rules;            /* the frozen policy (by reference; static lifetime) */
static int g_nrules;
static uint32_t g_threshold_milli;                  /* the frozen decide threshold */

#if BCIR_OOB_COUNTER_ATOMIC
static atomic_flag g_policy_lock = ATOMIC_FLAG_INIT;
static void policy_lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_policy_lock, memory_order_acquire)) { }
}
static void policy_unlock(void) {
    atomic_flag_clear_explicit(&g_policy_lock, memory_order_release);
}
#else
static void policy_lock(void) { }
static void policy_unlock(void) { }
#endif

void bcir_recover_set_policy(const bcir_recover_rule *rules, int n, uint32_t threshold_milli) {
    policy_lock();
    g_rules = (rules && n > 0) ? rules : NULL;
    g_nrules = (rules && n > 0) ? n : 0;
    g_threshold_milli = threshold_milli > 1000u ? 1000u : threshold_milli;
    policy_unlock();
}

/* The frozen graded proposition for a site: its rule's (action, confidence), or (abort, 0) if no rule. */
static int policy_lookup(const char *site, uint32_t *confidence_milli,
                         uint32_t *threshold_milli) {
    int action = BCIR_RECOVER_ABORT;
    uint32_t confidence = 0;
    policy_lock();
    if (site) {
        for (int k = 0; k < g_nrules; k++)
            if (g_rules[k].site && !strcmp(g_rules[k].site, site)) {
                confidence = g_rules[k].confidence_milli > 1000u ? 1000u
                                                                  : g_rules[k].confidence_milli;
                action = g_rules[k].action == BCIR_RECOVER_CLAMP ? BCIR_RECOVER_CLAMP
                                                                  : BCIR_RECOVER_ABORT;
                break;
            }
    }
    *threshold_milli = g_threshold_milli;
    policy_unlock();
    *confidence_milli = confidence;                  /* unknown site: no confidence, fail closed */
    return action;
}

size_t bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    bcir_oob_record_event(rid, index, extent, site);          /* (1) trace */

    uint32_t confidence_milli, threshold_milli;               /* (2) frozen graded proposition */
    int proposed = policy_lookup(site, &confidence_milli, &threshold_milli);

    int admitted = confidence_milli >= threshold_milli;       /* (3) decide: collapse at the frozen threshold */
    /* A clamp needs a valid in-bounds target: with extent==0 (e.g. a zero-length array or a recovered
     * extent that came out 0) there is NO in-bounds index, so clamping to 0 would itself be out of bounds.
     * Force ABORT in that case rather than returning the OOB index 0. */
    int action = (admitted && proposed == BCIR_RECOVER_CLAMP && extent) ? BCIR_RECOVER_CLAMP : BCIR_RECOVER_ABORT;
    size_t recovered = (action == BCIR_RECOVER_CLAMP) ? (size_t)(extent - 1) : 0;

    bcir_decide_record_event(rid, index, extent, site,        /* (4) RECORD the crossing -- never silent */
                             confidence_milli, threshold_milli, admitted, action, recovered);

    if (action == BCIR_RECOVER_ABORT) {                       /* (5) apply */
        fprintf(stderr, "BCIR bounds-quarantine: %s index %llu of [0, %llu) -- recovery rejected "
                "(confidence %u/1000 < threshold %u/1000); aborting\n",
                site ? site : "?", (unsigned long long)index, (unsigned long long)extent,
                confidence_milli, threshold_milli);
        abort();
        return 0;                                             /* unreachable */
    }
    return recovered;                                         /* clamp: the access lands on a valid element */
}

/* The WRITE override (§5.12). A store is governed by the SAME frozen policy, but a clamp is NEVER applied to
 * a write: redirecting an OOB store onto a[extent-1] would silently corrupt a live element. So even when the
 * policy's graded proposition clears the threshold and proposes a clamp, the classical write outcome is
 * ABORT -- the crossing is RECORDED (so the audit trail still witnesses that a clamp was proposed AND
 * refused for a write), then we fail-fast. This handler is `noreturn`: an OOB store always terminates. The
 * only difference from the read path is that a cleared-threshold clamp becomes a recorded abort, not a
 * redirect. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))           /* noreturn on the DEFINITION too: a write override that adds a clamp
                                     * path (returning a redirected index) is then a compile error. */
#endif
void bcir_bounds_quarantine_write(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    bcir_oob_record_event(rid, index, extent, site);          /* (1) trace */

    uint32_t confidence_milli, threshold_milli;               /* (2) frozen graded proposition */
    int proposed = policy_lookup(site, &confidence_milli, &threshold_milli);

    int admitted = confidence_milli >= threshold_milli;       /* (3) decide: collapse at the frozen threshold */
    /* (4) RECORD the crossing -- a clamp is NEVER the write's classical action, so record abort regardless;
     * but preserve `admitted`/`proposed` so the trail shows a clamp WAS proposed and REFUSED for the write. */
    int clamp_proposed = admitted && proposed == BCIR_RECOVER_CLAMP;
    bcir_decide_record_event(rid, index, extent, site,
                             confidence_milli, threshold_milli, admitted, BCIR_RECOVER_ABORT, 0);

    fprintf(stderr, "BCIR bounds-quarantine (WRITE): %s index %llu of [0, %llu) -- %s; an out-of-bounds "
            "store cannot be clamped (would corrupt a valid element); aborting\n",
            site ? site : "?", (unsigned long long)index, (unsigned long long)extent,
            clamp_proposed ? "clamp proposed but REFUSED for a write"
                           : "recovery rejected (under confident)");
    abort();
}
