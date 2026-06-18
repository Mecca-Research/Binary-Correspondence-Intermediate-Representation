/*===- bcir_plan.h - a minimal K_BCIR plan over the claim graph ------------===
 *
 * Turns a claim graph (bcir_cir.h) into a *plan*: a per-claim realization (the lane
 * width to use) + an integer cost, and the plan total. This is the production C seam of
 * the K_BCIR optimizer; the full cost model / min-plus / RCSP live on the MLIR law rail
 * (roadmap 5.8) -- this is the compact scalar planner a driver-embedded C compiler needs
 * to drive hydration (bcir_hydrate.h) end to end.
 *
 * Freestanding (only the IR + <stdint.h>): no libc. Deterministic integer arithmetic.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_PLAN_H
#define BCIR_PLAN_H

#include "bcir_cir.h"
#include "bcir_runtime.h"   /* bcir_status */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcir_plan_step {
  uint32_t claim_id;
  uint32_t width;   /* the realized lane width (scalar == 1) */
  uint32_t cost;    /* integer K_BCIR cost = base(opcode) * width */
} bcir_plan_step;

typedef struct bcir_plan {
  const bcir_plan_step *steps;
  size_t n;
  uint64_t total_cost;
} bcir_plan;

/* Plan every claim in `f` into caller-owned `steps[0..cap)`; fills `out`. The plan points
 * into `steps`. Returns BCIR_ERR_NOSPACE if `cap < f->n_claims`. */
BCIR_NODISCARD bcir_status bcir_plan_func(const bcir_func *f, bcir_plan_step *steps, size_t cap,
                                          bcir_plan *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_PLAN_H */
