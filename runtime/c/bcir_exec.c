/*===- bcir_exec.c - deterministic StreamPack executor (gem/execute.py in C) ===*/
#include "bcir_exec.h"

/* Collect segment views into the caller's work buffer (preserving pack order). Stops
 * (returns nonzero) if the buffer overflows -- but the caller pre-checks scratch_cap >=
 * n_segments, so overflow cannot happen on a valid pack. */
typedef struct {
  bcir_exec_item *items;
  size_t cap;
  size_t n;
  int overflow;
} collect_ctx;

static int collect_segment(const bcir_segment_view *seg, void *vctx) {
  collect_ctx *cc = (collect_ctx *)vctx;
  if (cc->n >= cc->cap) { cc->overflow = 1; return 1; }
  cc->items[cc->n].claim_id = seg->claim_id;
  cc->items[cc->n].phase_id = seg->phase_id;
  cc->items[cc->n].width = seg->width;
  cc->items[cc->n].lane = seg->lane;
  cc->n++;
  return 0;
}

/* The first-appearance rank of `pid` among `phases[0..np)` (its topological position). */
static size_t phase_rank(const bcir_phase_stat *phases, size_t np, uint32_t pid) {
  for (size_t i = 0; i < np; i++)
    if (phases[i].phase_id == pid)
      return i;
  return np;  /* unreachable: every item's phase is registered before sorting */
}

bcir_status bcir_sp_execute(const uint8_t *BCIR_RESTRICT data, size_t len,
                            bcir_exec_item *BCIR_RESTRICT scratch, size_t scratch_cap,
                            bcir_phase_stat *BCIR_RESTRICT phases, size_t phases_cap,
                            bcir_exec_fn fn, void *ctx,
                            bcir_exec_result *BCIR_RESTRICT out) {
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(data, len, &hdr);
  if (st != BCIR_OK)
    return st;
  /* Trust boundary (R10 + lane/width/dispatch range): a CRC-valid but semantically-corrupt
   * pack -- a dangling/redirected claim_id, an unresolved prefetch, an out-of-range lane/
   * width/dispatch -- is REJECTED here rather than executed silently. R11 (generation/
   * staleness) needs a caller-supplied expected gen, so it is enforced by the
   * bcir_sp_execute_checked wrapper (and by bcir_sp_check_generation at the driver). */
  st = bcir_sp_verify_semantic(data, len, 0xFFFFFFFFu, 0xFFFFFFFFu);
  if (st != BCIR_OK)
    return st;
  if (scratch_cap < (size_t)hdr.n_segments)
    return BCIR_ERR_NOSPACE;

  /* 1) Decode the segment stream into the work buffer (bounds-checked by the decoder). */
  collect_ctx cc = {scratch, scratch_cap, 0, 0};
  st = bcir_sp_for_each_segment(data, len, collect_segment, &cc);
  if (st != BCIR_OK)
    return st;
  if (cc.overflow)
    return BCIR_ERR_NOSPACE;
  size_t n = cc.n;

  /* 2) Register phases in first-appearance (topological) order + count scheduled. */
  size_t np = 0;
  for (size_t i = 0; i < n; i++) {
    size_t r = phase_rank(phases, np, scratch[i].phase_id);
    if (r == np) {
      if (np >= phases_cap)
        return BCIR_ERR_NOSPACE;
      phases[np].phase_id = scratch[i].phase_id;
      phases[np].scheduled = 0;
      phases[np].executed = 0;
      r = np++;
    }
    phases[r].scheduled++;
  }

  /* 3) Stable insertion sort by (phase rank, claim_id) -- the GEM deterministic order. */
  for (size_t i = 1; i < n; i++) {
    bcir_exec_item key = scratch[i];
    size_t krank = phase_rank(phases, np, key.phase_id);
    size_t j = i;
    while (j > 0) {
      size_t prank = phase_rank(phases, np, scratch[j - 1].phase_id);
      if (prank < krank || (prank == krank && scratch[j - 1].claim_id <= key.claim_id))
        break;
      scratch[j] = scratch[j - 1];
      --j;
    }
    scratch[j] = key;
  }

  /* 4) Dispatch in order, invoking the kernel and accruing per-phase telemetry. */
  size_t executed = 0;
  for (size_t i = 0; i < n; i++) {
    if (fn && fn(&scratch[i], ctx))
      break;  /* a kernel aborted the run */
    phases[phase_rank(phases, np, scratch[i].phase_id)].executed++;
    executed++;
  }

  if (out) {
    out->executed = executed;
    out->n_phases = np;
    out->n_segments = n;
  }
  return BCIR_OK;
}

bcir_status bcir_sp_execute_checked(const uint8_t *BCIR_RESTRICT data, size_t len,
                                    uint32_t expected_map_gen, uint32_t expected_data_gen,
                                    bcir_exec_item *BCIR_RESTRICT scratch, size_t scratch_cap,
                                    bcir_phase_stat *BCIR_RESTRICT phases, size_t phases_cap,
                                    bcir_exec_fn fn, void *ctx,
                                    bcir_exec_result *BCIR_RESTRICT out) {
  /* R11 (generation/staleness): reject a STALE pack before any execution. R10 + the range
   * gate are then re-enforced inside bcir_sp_execute (cheap; both passes are bounds-checked). */
  bcir_status st = bcir_sp_check_generation(data, len, expected_map_gen, expected_data_gen);
  if (st != BCIR_OK)
    return st;
  return bcir_sp_execute(data, len, scratch, scratch_cap, phases, phases_cap, fn, ctx, out);
}
