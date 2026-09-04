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
  uint32_t width;   /* the realized lane width: a power of two; 1 == the scalar realization,
                     * the only one this planner emits (never the claim's element count) */
  uint32_t cost;    /* integer K_BCIR cost = bcir_plan_base_cost(claim) * ceil(n / width),
                     * n = max(count, 1): the issues the realization needs, priced per issue */
} bcir_plan_step;

typedef struct bcir_plan {
  const bcir_plan_step *steps;
  size_t n;
  uint64_t total_cost;
} bcir_plan;

/* Plan every claim in `f` into caller-owned `steps[0..cap)`; fills `out`. The plan points
 * into `steps`. Returns BCIR_ERR_NOSPACE if `cap < f->n_claims`, or BCIR_ERR_OVERFLOW
 * when a per-step/total cost cannot be represented (never wraps a cost silently). `out`
 * is zeroed on every failure and arithmetic is preflighted before any step is written. */
BCIR_NODISCARD bcir_status bcir_plan_func(const bcir_func *f, bcir_plan_step *steps, size_t cap,
                                          bcir_plan *out);

/* The per-issue base cost of a claim's opcode/domain (an integer K_BCIR-flavoured metric:
 * memory + MMIO are dear, multiply dearer than add, atomics dearest; deterministic). The ONE
 * predicate the planner prices with and R9 (bcir_verify_plan) re-derives every step against --
 * a header inline so no build that links the verifier needs a new object for it. A plan whose
 * costs the verifier cannot reproduce from the claims and this function is not a plan this
 * planner could have produced, whatever it says its total is. */
static inline uint32_t bcir_plan_base_cost(const bcir_claim *cl) {
  uint32_t c;
  switch (cl->opcode) {
    case BCIR_OP_LOAD: case BCIR_OP_STORE: c = 4; break;
    case BCIR_OP_MUL:  c = 3; break;
    case BCIR_OP_ATOMIC_ADD: case BCIR_OP_ATOMIC_SUB: case BCIR_OP_ATOMIC_XOR:
    case BCIR_OP_CMPXCHG: c = 8; break;
    case BCIR_OP_GEM_DISPATCH: c = 2; break;     /* a call boundary */
    case BCIR_OP_NOP: c = 0; break;
    default: c = 1; break;                        /* ADD/SUB and the ALU family */
  }
  if (cl->domain == BCIR_DOM_MMIO) c += 4;        /* ordered device access is dearer */
  return c;
}

#ifdef __cplusplus
}
#endif

#endif /* BCIR_PLAN_H */
