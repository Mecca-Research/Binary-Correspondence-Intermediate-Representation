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

void bcir_recover_set_policy(const bcir_recover_rule *rules, int n, uint32_t threshold_milli) {
    g_rules = rules;
    g_nrules = n;
    g_threshold_milli = threshold_milli > 1000u ? 1000u : threshold_milli;
}

/* The frozen graded proposition for a site: its rule's (action, confidence), or (abort, 0) if no rule. */
static int policy_lookup(const char *site, uint32_t *confidence_milli) {
    if (site) {
        for (int k = 0; k < g_nrules; k++)
            if (g_rules[k].site && !strcmp(g_rules[k].site, site)) {
                *confidence_milli = g_rules[k].confidence_milli > 1000u ? 1000u : g_rules[k].confidence_milli;
                return g_rules[k].action;
            }
    }
    *confidence_milli = 0;                           /* unknown site: a graded proposition of no confidence */
    return BCIR_RECOVER_ABORT;
}

size_t bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site) {
    bcir_oob_record_event(rid, index, extent, site);          /* (1) trace */

    uint32_t confidence_milli;                                /* (2) frozen graded proposition */
    int proposed = policy_lookup(site, &confidence_milli);

    int admitted = confidence_milli >= g_threshold_milli;     /* (3) decide: collapse at the frozen threshold */
    int action = (admitted && proposed == BCIR_RECOVER_CLAMP) ? BCIR_RECOVER_CLAMP : BCIR_RECOVER_ABORT;
    size_t recovered = (action == BCIR_RECOVER_CLAMP && extent) ? (size_t)(extent - 1) : 0;

    bcir_decide_record_event(rid, index, extent, site,        /* (4) RECORD the crossing -- never silent */
                             confidence_milli, g_threshold_milli, admitted, action, recovered);

    if (action == BCIR_RECOVER_ABORT) {                       /* (5) apply */
        fprintf(stderr, "BCIR bounds-quarantine: %s index %llu of [0, %llu) -- recovery rejected "
                "(confidence %u/1000 < threshold %u/1000); aborting\n",
                site ? site : "?", (unsigned long long)index, (unsigned long long)extent,
                confidence_milli, g_threshold_milli);
        abort();
        return 0;                                             /* unreachable */
    }
    return recovered;                                         /* clamp: the access lands on a valid element */
}
