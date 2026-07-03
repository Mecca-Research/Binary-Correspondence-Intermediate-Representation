/*===- bcir_train.h - the D1 numeric stage kernels (the train step COMPUTES in C) ---------===
 *
 * D1 step 4 (ML/AI roadmap §8.2): the six train-step stage kernels of the planned claim
 * graph (bcir/kbcir/train_graph.py `_step_kernels` -- the M3 logistic step: sigmoid + BCE,
 * exact closed-form gradient) as C twins behind the executor's per-claim callback
 * (`bcir_exec_fn`). Steps 1-3 made a training step a planned, hydrated, binary-executable
 * artifact whose DISPATCH runs with no Python; this makes the dispatched step COMPUTE:
 * `bcir_sp_execute(pack, ..., bcir_train_kernel, &state)` runs real training end to end in
 * C, and the loss curve is the differential gate against the oracle's `train_planned`
 * (bcir/tests/test_train_c_kernels.py).
 *
 * The arithmetic mirrors the oracle kernel-for-kernel, in the same accumulation order
 * (ascending-index sums; the guarded two-branch sigmoid; the eps-clamped BCE) -- on a shared
 * libm the two rails agree to the last bit, and the gate allows only float round-off.
 * Host-tool posture (libm for exp/log), like bcir_cc; the executor underneath stays
 * freestanding. Memory is CALLER-OWNED (no malloc here; the harness sizes the buffers).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_TRAIN_H
#define BCIR_TRAIN_H

#include "bcir_exec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The train-step state -- the resources of the claim graph, materialized (the C twin of the
 * oracle's shared `st` dict). `w` and `grad` have nf+1 entries (the bias rides last). */
typedef struct bcir_train_state {
  int nf;            /* n_features */
  int b;             /* batch size */
  double lr;
  double *X;         /* b x nf, row-major (the current batch) */
  double *y;         /* b */
  double *w;         /* nf+1 */
  double *z;         /* b */
  double *act;       /* b */
  double *lossv;     /* b */
  double loss;       /* the reduced mean BCE of the current step */
  double *grad;      /* nf+1 */
} bcir_train_state;

/* The per-claim kernel callback (`bcir_exec_fn` shape; ctx = bcir_train_state*): dispatches
 * the pack's claim ids to the six stage kernels exactly as the oracle keys `_step_kernels`
 *   1 forward (z = X@w + bias)   2 activation (guarded sigmoid)   3 per-example BCE
 *   4 reduce (mean loss)         5 backward (exact BCE+sigmoid gradient)   6 sgd update
 * An id outside 1..6 is a no-op returning 0 (the oracle's kernels.get semantics). */
int bcir_train_kernel(const bcir_exec_item *item, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_TRAIN_H */
