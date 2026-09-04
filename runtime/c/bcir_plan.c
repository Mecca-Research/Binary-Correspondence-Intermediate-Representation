/*===- bcir_plan.c - a minimal K_BCIR plan over the claim graph ------------===*/
#include "bcir_plan.h"

bcir_status bcir_plan_func(const bcir_func *f, bcir_plan_step *steps, size_t cap, bcir_plan *out) {
  if (out) { out->steps = NULL; out->n = 0; out->total_cost = 0; }
  if(!f||!out||(f->n_claims&&!f->claims)||(!steps&&f->n_claims))return BCIR_ERR_NOSPACE;
  if (cap < f->n_claims) return BCIR_ERR_NOSPACE;

  /* Preflight every arithmetic operation before writing caller-owned steps.  A late
   * overflow must not leave a plausible-looking partial plan in the output buffer. */
  /* The scalar realization: width 1, one issue per element. The element count used to be
   * written INTO the width field, so a claim over 3 elements planned at "width 3" -- a lane
   * width no hardware issues and one hydration rightly refuses (power of two only): the
   * planner and the hydrator could not compose on any non-power-of-two count. */
  uint64_t total = 0;
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    uint64_t n = cl->count ? cl->count : 1;
    uint64_t cost64 = (uint64_t)bcir_plan_base_cost(cl) * n;
    if(cost64>UINT32_MAX||total>UINT64_MAX-cost64)return BCIR_ERR_OVERFLOW;
    total += cost64;
  }
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    uint64_t n = cl->count ? cl->count : 1;
    steps[i].claim_id = cl->id;
    steps[i].width = 1u;
    steps[i].cost = (uint32_t)((uint64_t)bcir_plan_base_cost(cl) * n); /* proved above */
  }
  out->steps = steps;
  out->n = f->n_claims;
  out->total_cost = total;
  return BCIR_OK;
}
