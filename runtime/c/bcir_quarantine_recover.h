#ifndef BCIR_QUARANTINE_RECOVER_H
#define BCIR_QUARANTINE_RECOVER_H
/* A REFERENCE strong override of `bcir_bounds_quarantine` (§5.12) -- the ML-layer / debugger seam. Linking
 * this beside bcir_quarantine.c REPLACES the weak abort-on-OOB default with a recovery handler that consults
 * a FROZEN per-site policy and applies a CLASSICAL action (abort / clamp) through a recorded `decide` (the
 * only sanctioned two-truth crossing; the crossing is appended to the bcir_decide_ring audit surface).
 *
 * The policy is a frozen table the ML-layer bakes OFFLINE (L1: frozen tables only -- no learned inference on
 * the hot path, the L0 prohibition). At an OOB access this handler does a table lookup, NOT live inference:
 * the graded proposition's confidence is read from the frozen table, collapsed at the frozen threshold, and
 * the resulting classical action is recorded + applied. The real ML-layer supplies its own override + table;
 * this one is the canonical reference (and the test vehicle). */
#include "bcir_quarantine.h"

/* One frozen policy rule: for accesses whose `site` matches, propose `action` with `confidence_milli`. */
typedef struct {
    const char *site;            /* exact "<func>:<array>" handle this rule governs */
    int action;                  /* the proposed bcir_recover_action (BCIR_RECOVER_CLAMP / _ABORT) */
    uint32_t confidence_milli;   /* the graded confidence for this proposal, Q-milli [0,1000] */
} bcir_recover_rule;

/* Install the frozen policy: `rules` (kept by reference -- static-lifetime storage) governs which sites may
 * recover, and `threshold_milli` is the FROZEN decide threshold every crossing is collapsed at. A site with
 * no matching rule defaults to abort (confidence 0). With no policy installed, every OOB aborts (the same
 * fail-fast as the weak default, but routed through a recorded `decide`). */
void bcir_recover_set_policy(const bcir_recover_rule *rules, int n, uint32_t threshold_milli);

#endif /* BCIR_QUARANTINE_RECOVER_H */
