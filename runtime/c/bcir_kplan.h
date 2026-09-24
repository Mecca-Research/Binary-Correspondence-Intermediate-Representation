/*===- bcir_kplan.h - the native K_BCIR planner (BKPI -> BKPR, version zero) -===
 *
 * The C twin of bcir/kbcir/realize.py's planner and of bcir/abi/planner_abi.py (GEM+ roadmap
 * G17, staged plan S4-A); the prose spec is docs/kernel/BCIR_PLANNER_ABI.md. It reads the
 * planner's whole input as one record (BKPI: the phases, the claims, their operands, the ops,
 * the target's cost constants, Theta and the policy) and writes the plan as one record (BKPR:
 * per step the claim, the phase, the realization with its coupled base cost and its realized
 * cost, and the score). A native planner may own a certificate only after it reproduces the
 * Python plan byte for byte over a generated corpus; `planner.parity` is that gate.
 *
 * The algorithm is the oracle's, step for step:
 *   - the phases in the canonical dependency-first order (model.topological_phase_ids);
 *   - per claim the realization enumeration (realize._offer_rows) with the tier of its
 *     primary resource, and the intra-phase data-flow discounts (realize.fused_offer): CSE by
 *     value-numbered semantic identity over eligible claims, else producer->consumer
 *     deforestation unless a barrier fences it;
 *   - the min-plus path over the layered realization DAG, an edge weighing the realization's
 *     base cost coupled (x0.75 memory after a vector predecessor sharing a read, x1.25
 *     thermal/power for wide SIMD on a hot machine) and scalarized under the policy's weights
 *     folded with Theta; ties keep the first predecessor in node order.
 * Every intermediate is exact: the declared domain (the laws below) bounds each product under
 * 2**96 and every path weight under 2**127, so the arithmetic is 128-bit and never wraps. A plan
 * is refused only when a value it must carry exceeds 2**63 - 1 (BCIR_ERR_OVERFLOW), exactly when
 * the Python encoder refuses the same plan.
 *
 * Freestanding: <stddef.h> + <stdint.h> (through bcir_runtime.h), no heap, no libc. The caller
 * supplies the scratch (bcir_kp_scratch_size) and the output (bcir_kp_realization_size); every
 * failure zeroes the output. Version zero: no compatibility promise; not the BCIR UAPI.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_KPLAN_H
#define BCIR_KPLAN_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_KP_INPUT_MAGIC             "BKPI"
#define BCIR_KP_REALIZATION_MAGIC       "BKPR"
#define BCIR_KP_VERSION                 0u
#define BCIR_KP_INPUT_HEADER_SIZE       64u
#define BCIR_KP_SCOPE_SIZE              168u
#define BCIR_KP_CLAIM_SIZE              48u
#define BCIR_KP_RESOURCE_SIZE           4u
#define BCIR_KP_REALIZATION_HEADER_SIZE 32u
#define BCIR_KP_STEP_SIZE               120u
#define BCIR_KP_CRC_SIZE                4u

/* The declared domain (docs/kernel/BCIR_PLANNER_ABI.md works the bounds). */
#define BCIR_KP_CLAIMS_MAX    (1u << 24) /* claims, phases and ops */
#define BCIR_KP_REFS_MAX      (1u << 26) /* operand references, dependencies and resources */
#define BCIR_KP_OPERANDS_MAX  255u       /* reads + writes of one claim */
#define BCIR_KP_WIDTHS_MAX    16u
#define BCIR_KP_PARAM_MAX     (1u << 16) /* target constants, lane widths, tier factors, weights */
#define BCIR_KP_THETA_MAX     100u
#define BCIR_KP_OP_BYTES_MAX  (1u << 24)
#define BCIR_KP_NO_RESOURCE   0xFFFFFFFFu

/* Realization name codes (BKPR `name`). */
enum {
  BCIR_KP_NAME_NOOP = 0,
  BCIR_KP_NAME_BARRIER = 1,
  BCIR_KP_NAME_ATOMIC = 2,
  BCIR_KP_NAME_BLOCKED = 3,
  BCIR_KP_NAME_GATHER = 4,
  BCIR_KP_NAME_SCALAR = 5,
  BCIR_KP_NAME_VEC = 6, /* "vec<width>" */
  BCIR_KP_NAME_STRIDED = 7,
  BCIR_KP_NAME_UX_BUCKET = 8,
  BCIR_KP_NAME_TILE = 9,
  BCIR_KP_NAME_COUNT = 10
};

/* A validated BKPI record, read in place: the section offsets into `data`. */
typedef struct bcir_kp_input {
  const uint8_t *data;
  size_t len;
  uint32_t n_phases, n_deps, n_claims, n_refs, n_resources, n_ops, op_bytes, n_widths;
  size_t off_widths, off_phases, off_deps, off_claims, off_refs, off_resources, off_op_lens,
      off_ops;
} bcir_kp_input;

/* A validated BKPR record, read in place. */
typedef struct bcir_kp_realization {
  const uint8_t *data;
  size_t len;
  uint32_t n_steps;
  uint64_t score;
} bcir_kp_realization;

/* Decode and validate a BKPI record, the laws in the specification's order (the Python
 * decoder, bcir.abi.planner_abi.decode_input, names the same first violation):
 * TRUNCATED, MAGIC, VERSION, CRC, RESERVED (flags, header pad); PLANNER for a section count
 * over its bound; TRUNCATED/TRAILING for the size; PLANNER for the scope, the widths, the
 * phases, the resources (RESERVED for a pad), the op table (UTF8 for an op that is not UTF-8),
 * each claim (RESERVED for a pad) and the totals. `out` is zeroed on every failure. */
BCIR_NODISCARD bcir_status bcir_kp_decode_input(const uint8_t *data, size_t len,
                                                bcir_kp_input *out);

/* The scratch bytes bcir_kp_plan needs for `in` (BCIR_ERR_OVERFLOW if it cannot be a size_t).
 * `in` is a record bcir_kp_decode_input accepted; this call and bcir_kp_plan re-check only the
 * lane-width count their geometry indexes by (BCIR_ERR_PLANNER outside 1..16). */
BCIR_NODISCARD bcir_status bcir_kp_scratch_size(const bcir_kp_input *in, size_t *bytes);

/* The exact size of the realization of `in`: 32 + 120 * n_claims + 4. */
BCIR_NODISCARD bcir_status bcir_kp_realization_size(const bcir_kp_input *in, size_t *bytes);

/* Plan `in` into `out[0..cap)`, using `scratch[0..scratch_len)` (any alignment). BCIR_ERR_NOSPACE
 * when the scratch or `cap` is short (checked before any work), BCIR_ERR_OVERFLOW when a value
 * the realization carries exceeds 2**63 - 1. `*out_len` is the realization's size on success
 * and 0 on failure, and the output is zeroed on failure. */
BCIR_NODISCARD bcir_status bcir_kp_plan(const bcir_kp_input *in, void *scratch,
                                        size_t scratch_len, uint8_t *out, size_t cap,
                                        size_t *out_len);

/* Decode and validate a BKPR record (the Python decode_realization's laws, in order):
 * TRUNCATED, MAGIC, VERSION, CRC, RESERVED; TRUNCATED/TRAILING for the size; OVERFLOW for the
 * score; per step RESERVED, OVERFLOW, PLANNER (vocabulary, a name's width); PLANNER when the
 * score is not the sum of the step costs. `out` is zeroed on failure. */
BCIR_NODISCARD bcir_status bcir_kp_decode_realization(const uint8_t *data, size_t len,
                                                      bcir_kp_realization *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_KPLAN_H */
