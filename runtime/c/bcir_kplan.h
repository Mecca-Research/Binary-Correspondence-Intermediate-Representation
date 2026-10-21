/*===- bcir_kplan.h - the native K_BCIR planner (BKPI -> BKPR, version zero) -===
 *
 * The C twin of bcir/kbcir/realize.py's planner and of bcir/abi/planner_abi.py (GEM+ roadmap
 * G17, staged plan S4-A); the prose spec is docs/kernel/BCIR_PLANNER_ABI.md. It reads the
 * planner's whole input as one record (BKPI: the phases, the claims, their operands, the ops,
 * the target's cost constants, Theta and the policy) and writes the plan as one record (BKPR:
 * per step the claim, the phase, the realization with its coupled base cost and its realized
 * cost, and the score). A native planner may own a certificate only after it reproduces the
 * Python plan byte for byte over a generated corpus; `planner.parity` is that gate. Its plan
 * then becomes the pack that runs natively too: bcir_kp_hydrate (below, CXX4) is the C twin of
 * bcir/gem/streampack.py's hydrate, held to it by `hydrate.native.parity`.
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
 * PX-3b-2: version 1 of both records carries the target's measured table (BKPI: n_measured u32 at
 * @40 and a section after the op bytes, n_measured x (op u32, count u32, code u16, q8 u16); BKPR:
 * a step's reserved u16 its measured alternative's code). A claim whose (op, count) an entry names
 * is offered each of its realizations run that alternative, its compute and memory axes scaled by
 * q8/256 after the intra-phase discount (realize._offer_rows). A record without a measured entry
 * (BKPI) or step (BKPR) is version zero, byte for byte, and a version-1 record without one is
 * refused: one spelling per input.
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
#define BCIR_KP_VERSION_MEASURED        1u /* PX-3b-2: a measured table / a measured step */
#define BCIR_KP_MEASURED_SIZE           12u
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
/* PX-3b-2: the measured table's bounds -- at most 2**24 entries, at most 7 sharing an (op,
 * count), so a claim's rows stay under 128 (16 widths x (1 + 7)) and a predecessor fits a u8 --
 * and a code's worker flag (t<k> is 0x8000 | k; an instance is its bit, 1, 2, 4 or 8). */
#define BCIR_KP_MEASURED_MAX          (1u << 24)
#define BCIR_KP_MEASURED_PER_KEY_MAX  7u
#define BCIR_KP_MEASURED_WORKERS      0x8000u
#define BCIR_KP_ROWS_MAX              (BCIR_KP_WIDTHS_MAX * (1u + BCIR_KP_MEASURED_PER_KEY_MAX))

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
  uint32_t n_measured; /* PX-3b-2: 0 in a version-zero record */
  size_t off_measured;
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
 * TRUNCATED, MAGIC, VERSION (0 or 1), CRC, RESERVED (flags, header pad: version 1's after
 * n_measured); PLANNER for a section count over its bound (version 1: 1 .. 2**24 measured
 * entries); TRUNCATED/TRAILING for the size; PLANNER for the scope, the widths, the phases, the
 * resources (RESERVED for a pad), the op table (UTF8 for an op that is not UTF-8), the measured
 * table (an entry's op an index naming a non-empty op, its count positive, its code an
 * alternative's, its price 1 .. 255, ascending by (op, count, code), at most 7 per op and count),
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
 * TRUNCATED, MAGIC, VERSION (0 or 1), CRC, RESERVED; TRUNCATED/TRAILING for the size; OVERFLOW
 * for the score; per step RESERVED (version 0's reserved u16), OVERFLOW, PLANNER (vocabulary, a
 * name's width, version 1's code an alternative's or 0); PLANNER when the score is not the sum
 * of the step costs, and when a version-1 record runs no measured alternative. `out` is zeroed
 * on failure. */
BCIR_NODISCARD bcir_status bcir_kp_decode_realization(const uint8_t *data, size_t len,
                                                      bcir_kp_realization *out);

/* ---- the native hydrate (CXX4): BKPI + BKPR + BKPB -> the oracle's StreamPack bytes --------
 *
 * The C twin of bcir/gem/streampack.py::hydrate followed by bcir/abi/streampack_abi.py::encode:
 * the planner's own input and realization, with the binding below, become the StreamPack the
 * oracle writes for the same module and plan -- byte for byte (`hydrate.native.parity`). BKPI
 * carries the claims and their operands as indices into a resource table; what it does not carry
 * is the binding (BKPB, version zero, docs/kernel/BCIR_PLANNER_ABI.md):
 *
 *     header (32): magic "BKPB"  version:u16=0  flags:u16=0  n_resources:u32 @8  n_gens:u32 @12
 *       topo_gen:u32 @16  plan_len:u32 @20  reserved:u64 @24
 *     rids n_resources x u32 -- the RID of each BKPI resource index
 *     gens n_gens x (rid, map_gen, data_gen):u32 -- the registry's generation vector, RID order
 *     plan plan_len bytes (UTF-8) -- the pack's source_plan | crc32:u32
 *
 * The laws, in order: TRUNCATED, MAGIC, VERSION, CRC, RESERVED; PLANNER for a count over its
 * bound (REFS_MAX) or a plan over 65535 bytes; TRUNCATED/TRAILING for the size; UTF8 for the
 * plan; GENERATION for RIDs not strictly ascending. Against the input (bcir_kp_hydrate): PLANNER
 * when the resource counts differ, two indices name one RID, or a resource is declared without a
 * generation (or undeclared with one). Then the oracle hydrate's own laws, each PROVENANCE: a
 * step naming no claim or a claim twice, a step whose phase is not its claim's, a step out of
 * the canonical phase order, a claim no step plans. Then the encoder's: WIDTH for a width that
 * is not a power of two, OVERFLOW for a negative offset or stride (a block field is a u64). */

#define BCIR_KP_BINDING_MAGIC       "BKPB"
#define BCIR_KP_BINDING_HEADER_SIZE 32u
#define BCIR_KP_PLAN_NAME_MAX       65535u

typedef struct bcir_kp_binding {
  const uint8_t *data;
  size_t len;
  uint32_t n_resources, n_gens, topo_gen, plan_len;
  size_t off_rids, off_gens, off_plan;
} bcir_kp_binding;

/* Decode and validate a BKPB record (bcir.abi.planner_abi.decode_binding's laws, in order).
 * `out` is zeroed on every failure. */
BCIR_NODISCARD bcir_status bcir_kp_decode_binding(const uint8_t *data, size_t len,
                                                  bcir_kp_binding *out);

/* The scratch bytes bcir_kp_hydrate needs for `in` and `b`. */
BCIR_NODISCARD bcir_status bcir_kp_hydrate_scratch_size(const bcir_kp_input *in,
                                                        const bcir_kp_binding *b, size_t *bytes);

/* Check every law and size the StreamPack bcir_kp_hydrate would write (0 on failure). */
BCIR_NODISCARD bcir_status bcir_kp_hydrate_size(const bcir_kp_input *in,
                                                const bcir_kp_realization *plan,
                                                const bcir_kp_binding *b, void *scratch,
                                                size_t scratch_len, size_t *bytes);

/* Write the StreamPack of `in` planned by `plan` under `b` into `out[0..cap)`. Every law is
 * checked before the first byte is written; BCIR_ERR_NOSPACE when the scratch or `cap` is short
 * (bcir_kp_hydrate_size gives the size the pack needs).
 * `*out_len` is the pack's size on success and 0 on failure, and the output is zeroed on
 * failure. */
BCIR_NODISCARD bcir_status bcir_kp_hydrate(const bcir_kp_input *in,
                                           const bcir_kp_realization *plan,
                                           const bcir_kp_binding *b, void *scratch,
                                           size_t scratch_len, uint8_t *out, size_t cap,
                                           size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_KPLAN_H */
