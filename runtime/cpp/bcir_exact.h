/*===- bcir_exact.h - the bounded exact scheduler, behind a C ABI -----------===
 *
 * PORT-5: the first solver of the hosted C++ solver library (the native port audit's N5, CXX2), at parity with the
 * oracle's `bcir/gem/exact.py` -- `exact_schedule`'s branch-and-bound over each phase's active schedules, the
 * lower-bound stack and the budget the phases share, expansion for expansion. The library is C++23 behind this C ABI:
 * every call is noexcept and returns a status (no exception crosses the façade), every allocation goes through the
 * caller's `bcir_host_allocator` (a pmr arena over it, one per phase), and nothing outlives the call.
 *
 * The heuristic placement each phase starts from is the artifact's (`schedule_eft`), computed by the caller -- the
 * oracle, or the C and MLIR twins that reproduce it -- and passed in, so the solver certifies the plan every rail
 * reads rather than a second copy of it. A search's resumable state (`SearchState`) stays the oracle's.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_EXACT_H
#define BCIR_EXACT_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_host_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_exact_status {
  BCIR_EXACT_OK = 0,
  BCIR_EXACT_ERR_INVALID = 1,   /* a NULL with a nonzero count; a claim with a negative duration, a repeated id, a
                                 * predecessor not in its phase (or itself), no stream, a stream past the tail, or
                                 * both the tail and a wave stream; a phase whose work leaves 64 bits; a heuristic
                                 * placement that is not a schedule of the phase spanning the heuristic */
  BCIR_EXACT_ERR_NOMEM = 2,     /* the allocator refused: nothing kept */
  BCIR_EXACT_ERR_CYCLIC = 3,    /* a phase's dependency graph is cyclic (the oracle's ValueError) */
  BCIR_EXACT_ERR_CAPACITY = 4,  /* a result's slot buffer smaller than its phase */
  BCIR_EXACT_ERR_INTERNAL = 5   /* an invariant the solver holds broke: a defect, reported, never thrown */
} bcir_exact_status;

/* A placed claim: `stream` is its affinity domain, or -1 for the decoupled tail stream (the oracle's TAIL_STREAM). */
typedef struct bcir_exact_slot { uint32_t claim; int32_t stream; int64_t start, finish; } bcir_exact_slot;

/* A claim of a phase, as the oracle's `_PhaseSearch` holds it: its duration (non-negative), its predecessors within
 * the phase in the dispatch's order, and the streams it may run on in the dispatch's order (`domains` the tail). */
typedef struct bcir_exact_claim {
  uint32_t id;
  int64_t duration;
  const uint32_t *preds; size_t n_preds;
  const uint32_t *streams; size_t n_streams;
} bcir_exact_claim;

typedef struct bcir_exact_phase {
  uint32_t phase_id;
  const bcir_exact_claim *claims; size_t n_claims;   /* the dispatch's order (claim ids, ascending) */
  uint32_t domains, knee;                            /* streams 0..domains-1 the wave, `domains` the tail */
  int64_t heuristic;                                 /* the artifact's span of the phase, from t0 = 0 */
  const bcir_exact_slot *heuristic_slots; size_t n_heuristic_slots;
} bcir_exact_phase;

/* The bound stack, in the oracle's order: critical-path, work/capacity, bandwidth/knee, tail-serial. */
#define BCIR_EXACT_BOUNDS 4

typedef struct bcir_exact_phase_result {
  uint32_t phase_id;
  int64_t heuristic, incumbent, lower_bound;
  int optimal;                                       /* the phase's gap closed ("optimal"), else "budget" */
  uint64_t expansions;
  int64_t bounds[BCIR_EXACT_BOUNDS];
  bcir_exact_slot *slots; size_t cap_slots, n_slots; /* the incumbent's placement, relative to the phase start */
} bcir_exact_phase_result;

typedef struct bcir_exact_result {
  int64_t heuristic, incumbent, lower_bound;
  int optimal;
  uint64_t expansions, budget;
} bcir_exact_result;

/* `exact_schedule`: the phases in topological order, the module's budget in node expansions consumed phase by
 * phase -- a phase that closes hands the rest to the next; once one stops on the budget, every later phase is
 * pending (the heuristic stands, the root stack its bound). `results` holds one entry per phase, its `slots` a buffer
 * the caller sized (`cap_slots`). The oracle's numbers, every one. */
bcir_exact_status bcir_exact_schedule(const bcir_exact_phase *phases, size_t n_phases, uint64_t budget,
                                      const bcir_host_allocator *allocator, bcir_exact_phase_result *results,
                                      bcir_exact_result *total);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_EXACT_H */
