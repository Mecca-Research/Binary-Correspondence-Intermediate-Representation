/*===- bcir_exec.h - freestanding deterministic StreamPack executor -------===
 *
 * The C twin of bcir/gem/execute.py: a deterministic, phase-sliced executor over a
 * decoded StreamPack. It dispatches the pack's claims in the **GEM execution order** --
 * topological phase order (which is the first-appearance order of phase ids in the pack,
 * since hydrate flattens topologically), then ascending claim_id within a phase -- and
 * invokes an optional per-claim kernel callback so the same scheduler drives real work.
 * It collects per-phase telemetry (scheduled / executed) and leaves the dispatch order
 * in the caller's work buffer.
 *
 * This turns the StreamPack into a **no-Python hot artifact**: a driver decodes the pack
 * (bcir_runtime.c) and runs it end to end here, with no libc dependency. Builds on the
 * bounds-checked decoder, so a malformed pack returns a status, never an out-of-bounds
 * read (exercised by fuzz_exec.c under libFuzzer + ASan/UBSan). A Python<->C parity gate
 * (bcir/tests/test_c_executor.py) pins the dispatch order + telemetry against the oracle.
 *
 * Memory model: caller-owned. The caller sizes `scratch` (>= n_segments items) and
 * `phases` (>= the number of distinct phases) from the validated header -- no malloc.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_EXEC_H
#define BCIR_EXEC_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One scheduled claim: the executor's work item + the caller-visible record. After a
 * successful run, scratch[0 .. result.executed) holds these in dispatch order. */
typedef struct bcir_exec_item {
  uint64_t claim_id;
  uint32_t phase_id;
  uint32_t width;
  uint8_t  lane;     /* bcir_lane */
} bcir_exec_item;

/* Per-phase telemetry (mirrors gem.execute.PhaseStat). */
typedef struct bcir_phase_stat {
  uint32_t phase_id;
  uint32_t scheduled;  /* claims in the phase */
  uint32_t executed;   /* claims dispatched (== scheduled unless a kernel aborts) */
} bcir_phase_stat;

/* Per-claim dispatch callback (the "kernel"). Return nonzero to abort the run early
 * (the remaining claims are left undispatched, mirroring an executor stop). */
typedef int (*bcir_exec_fn)(const bcir_exec_item *item, void *ctx);

typedef struct bcir_exec_result {
  size_t executed;    /* claims dispatched */
  size_t n_phases;    /* distinct phases */
  size_t n_segments;  /* segments in the pack (== total scheduled) */
} bcir_exec_result;

/* Execute a StreamPack in deterministic GEM order, invoking `fn` (optional) per claim.
 * `scratch` must hold >= the pack's n_segments items; `phases` >= the distinct-phase
 * count. On BCIR_OK, scratch[0..out->executed) is the dispatch order and phases[0..
 * out->n_phases) is the per-phase telemetry. Returns BCIR_ERR_* on a malformed pack and
 * BCIR_ERR_NOSPACE if a buffer is too small. */
BCIR_NODISCARD bcir_status bcir_sp_execute(const uint8_t *BCIR_RESTRICT data, size_t len,
                                           bcir_exec_item *BCIR_RESTRICT scratch,
                                           size_t scratch_cap,
                                           bcir_phase_stat *BCIR_RESTRICT phases,
                                           size_t phases_cap, bcir_exec_fn fn, void *ctx,
                                           bcir_exec_result *BCIR_RESTRICT out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_EXEC_H */
