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
 * An id outside 1..6 is a no-op returning 0 (the oracle's kernels.get semantics).
 * A known id with an invalid state shape/pointer returns nonzero before mutation. */
int bcir_train_kernel(const bcir_exec_item *item, void *ctx);

/* --- D1 step 7: the STREAMED step (kbcir/train_graph.py train_stream_module) ------------- */

/* The streamed-step state (the C twin of train_streamed's `st`): `streams` micro-batch lanes
 * of mb rows each over per-stream buffers, ONE shared weight vector and ONE combined
 * gradient. Flat row-major layouts: X is streams x mb x nf; z/act/lossv are streams x mb;
 * loss is streams; grad is streams x (nf+1); w and gradc are nf+1. Caller-owned. */
typedef struct bcir_stream_state {
  int nf;            /* n_features */
  int mb;            /* micro-batch rows per stream (batch / streams) */
  int streams;
  double lr;
  double *X;         /* streams x mb x nf */
  double *y;         /* streams x mb */
  double *w;         /* nf+1 (SHARED) */
  double *z;         /* streams x mb */
  double *act;       /* streams x mb */
  double *lossv;     /* streams x mb */
  double *loss;      /* streams (each stream's mean BCE) */
  double *grad;      /* streams x (nf+1) (per-stream mean gradients) */
  double *gradc;     /* nf+1 (the combined mean-of-means gradient) */
} bcir_stream_state;

/* The streamed per-claim callback (`bcir_exec_fn`; ctx = bcir_stream_state*): claim ids are
 * train_stream_module's bands -- stream s stage k = s*10 + k (k in 1..5), the gradient
 * combine = streams*10 + 1, the SINGLE sgd update = streams*10 + 2. Same arithmetic order
 * as the oracle's _stream_kernels (ascending-index sums; mean of the stream means). An id
 * outside the bands is a no-op returning 0. Invalid state geometry/pointers return
 * nonzero before mutation so the executor stops the dispatch. */
int bcir_stream_kernel(const bcir_exec_item *item, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_TRAIN_H */
