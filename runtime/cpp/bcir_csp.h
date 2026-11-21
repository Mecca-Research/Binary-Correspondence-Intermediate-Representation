/*===- bcir_csp.h - the exact finite-domain solver (integer CSP / ILP), behind a C ABI ---===
 *
 * PORT-6: the hosted C++ solver library's second solver (the native port audit's N5, CXX3), at parity with the
 * oracle's `bcir/kbcir/csp.py` (G28) -- `solve`'s depth-first branch and bound, node for node: bounds consistency on
 * each linear row, the compulsory-part timetable on each cumulative, the disjunction on each rectangle pair, the
 * incumbent's objective as a cut, the narrowest open domain of the lowest priority split at its floor midpoint, the
 * lower half first, a fully fixed node a solution only where the constraints' definitions agree, and a budget of
 * nodes. Every call is noexcept and returns a status, every allocation goes through the caller's
 * `bcir_host_allocator`, and nothing outlives the call.
 *
 * The model is the oracle's `Model`: a linear row's terms and the objective's are merged here as `Model` merges
 * them (one term per variable, the coefficients summed, zeros dropped, in variable order), and a rectangle's group
 * is explicit (the oracle's default for a rectangle without one is `-1 - index`). Integers are 64 bits where the
 * oracle's are unbounded, so a model whose arithmetic could leave 62 bits is refused (BCIR_CSP_ERR_RANGE), never
 * wrapped: every variable's bounds, offset, duration and height within 2^60, and each row's and the objective's
 * coefficient-weighted bound sum within 2^62.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CSP_H
#define BCIR_CSP_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_host_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_csp_status {
  BCIR_CSP_OK = 0,
  BCIR_CSP_ERR_INVALID = 1,    /* a NULL with a nonzero count; an empty domain; a variable out of range; an unknown
                                * relation or constraint kind; a negative duration; a capacity or a height below 1;
                                * rectangles of one group with different y variables or heights */
  BCIR_CSP_ERR_NOMEM = 2,      /* the allocator refused: nothing kept */
  BCIR_CSP_ERR_RANGE = 3,      /* arithmetic the search would form leaves 62 bits (see above) */
  BCIR_CSP_ERR_INCUMBENT = 4,  /* the incumbent handed in does not satisfy the model (the oracle's CspError) */
  BCIR_CSP_ERR_CAPACITY = 5,   /* the result's values buffer smaller than the model */
  BCIR_CSP_ERR_INTERNAL = 6    /* propagation accepted a non-solution: a defect, reported, never thrown */
} bcir_csp_status;

typedef enum bcir_csp_relation { BCIR_CSP_LE = 0, BCIR_CSP_GE = 1, BCIR_CSP_EQ = 2 } bcir_csp_relation;
typedef enum bcir_csp_kind { BCIR_CSP_LINEAR = 0, BCIR_CSP_CUMULATIVE = 1, BCIR_CSP_NO_OVERLAP_2D = 2 } bcir_csp_kind;
typedef enum bcir_csp_stop { BCIR_CSP_OPTIMAL = 0, BCIR_CSP_INFEASIBLE = 1, BCIR_CSP_BUDGET = 2 } bcir_csp_stop;

/* A variable over [lo, hi]; the search branches on the lowest `priority` first, then the narrowest domain. */
typedef struct bcir_csp_var { int64_t lo, hi, priority; } bcir_csp_var;
typedef struct bcir_csp_term { int64_t coef; uint32_t var; } bcir_csp_term;
/* A task of unit demand: it runs over [start, start + duration). */
typedef struct bcir_csp_task { uint32_t start; int64_t duration; } bcir_csp_task;
/* The rectangle [x0 + a, x1 + b) x [y, y + h); rectangles of one group are parts of one object and never constrain
 * each other. */
typedef struct bcir_csp_rect {
  uint32_t x0; int64_t a;
  uint32_t x1; int64_t b;
  uint32_t y; int64_t h;
  int64_t group;
} bcir_csp_rect;

/* One constraint: `kind` names which fields it reads. LINEAR: sum(coef * x) <relation> rhs. CUMULATIVE: at no
 * instant do more than `capacity` tasks run. NO_OVERLAP_2D: the rectangles pairwise disjoint. */
typedef struct bcir_csp_constraint {
  int kind;
  const bcir_csp_term *terms; size_t n_terms; int relation; int64_t rhs;
  const bcir_csp_task *tasks; size_t n_tasks; int64_t capacity;
  const bcir_csp_rect *rects; size_t n_rects;
} bcir_csp_constraint;

/* Minimize offset + sum(objective) subject to the constraints, in their declared order. */
typedef struct bcir_csp_model {
  const bcir_csp_var *vars; size_t n_vars;
  const bcir_csp_constraint *constraints; size_t n_constraints;
  const bcir_csp_term *objective; size_t n_objective; int64_t offset;
} bcir_csp_model;

/* The solver's answer, the oracle's `CspResult`: the best assignment found (in `values`, a buffer the caller sized,
 * when `has_values`), its objective, a lower bound valid over the whole model (`has_lower`), why it stopped and the
 * nodes it expanded. */
typedef struct bcir_csp_result {
  int stop;
  int has_values; int64_t objective;
  int has_lower; int64_t lower_bound;
  uint64_t nodes;
  int64_t *values; size_t cap_values;
} bcir_csp_result;

/* `solve(model, budget, incumbent)`: `incumbent`, NULL or one value per variable, must satisfy the model and seeds
 * the search. */
bcir_csp_status bcir_csp_solve(const bcir_csp_model *model, uint64_t budget, const int64_t *incumbent,
                               const bcir_host_allocator *allocator, bcir_csp_result *result);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CSP_H */
