/*===- bcir_plan.c - a minimal K_BCIR plan over the claim graph ------------===*/
#include "bcir_plan.h"

/* Per-opcode base cost (an integer K_BCIR-flavoured metric): memory + MMIO are dear,
 * multiply dearer than add, atomics dearest. Deterministic. */
static uint32_t base_cost(const bcir_claim *cl) {
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

bcir_status bcir_plan_func(const bcir_func *f, bcir_plan_step *steps, size_t cap, bcir_plan *out) {
  if(!f||!out||(!steps&&f->n_claims))return BCIR_ERR_NOSPACE;
  if (cap < f->n_claims) return BCIR_ERR_NOSPACE;
  uint64_t total = 0;
  for (size_t i = 0; i < f->n_claims; i++) {
    const bcir_claim *cl = &f->claims[i];
    uint32_t width = cl->count ? cl->count : 1;   /* scalar realization (the C subset is width-1) */
    uint64_t cost64 = (uint64_t)base_cost(cl) * width;
    if(cost64>UINT32_MAX||total>UINT64_MAX-cost64)return BCIR_ERR_OVERFLOW;
    uint32_t cost = (uint32_t)cost64;
    steps[i].claim_id = cl->id;
    steps[i].width = width;
    steps[i].cost = cost;
    total += cost;
  }
  out->steps = steps;
  out->n = f->n_claims;
  out->total_cost = total;
  return BCIR_OK;
}
