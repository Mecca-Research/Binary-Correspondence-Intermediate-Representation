/* Complete dense Llama/SwiGLU training in C. No allocation in plan, forward,
 * backward, optimizer or multi-step run. FP32 weights/activations; FP64 loss/norm.
 * HF row-major projections and half-split RoPE (ingest permutes only at export).
 * The checked state owns no memory; arenas must be disjoint and float aligned.
 * Caller serializes access; tensor providers must honor bcir_tensor.h contracts.
 * The plan's three counts are the arena sums of the step program in
 * bcir/hosted/models/native_program.py (NDT-GEM), held to it by train.plan.mismatch. */
#ifndef BCIR_DECODER_TRAIN_H
#define BCIR_DECODER_TRAIN_H
#include "bcir_tensor.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct bcir_decoder_spec {
  uint32_t vocab,width,heads,kvheads,layers,ff,batch,time,tied;
  double rope_base,rms_eps;
} bcir_decoder_spec;
typedef struct bcir_decoder_plan {
  bcir_decoder_spec spec;
  size_t parameters,activations,scratch;
} bcir_decoder_plan;
typedef struct bcir_decoder_state {
  bcir_decoder_plan plan;
  float *weights,*gradients,*moment1,*moment2,*activations,*scratch;
  size_t weight_capacity,gradient_capacity,moment1_capacity,moment2_capacity;
  size_t activation_capacity,scratch_capacity;
  uint64_t step;
  double beta1_power,beta2_power;
  bcir_tensor_provider provider;
} bcir_decoder_state;
typedef struct bcir_decoder_adamw {
  double lr,beta1,beta2,epsilon,weight_decay,grad_clip;
} bcir_decoder_adamw;
typedef struct bcir_decoder_event { double loss,grad_norm; uint64_t step; } bcir_decoder_event;
enum { BCIR_DT_OK=0, BCIR_DT_INVALID=1, BCIR_DT_CAPACITY=2, BCIR_DT_NUMERIC=3 };
BCIR_DT_API uint32_t bcir_decoder_abi_version(void);
/* Checked shape/products; failure leaves *plan untouched. Parameter order:
 * embedding; each layer: attn_norm,q,k,v,o,ff_norm,gate,up,down;
 * final_norm; untied head. Retain caches; reuse adjoints across all layers.
 * Capacities are float element counts, not bytes. */
BCIR_DT_API int bcir_decoder_make_plan(const bcir_decoder_spec *spec,bcir_decoder_plan *plan);
/* Forward stores logits at the end of the activation arena. Backward computes mean
 * causal-token CE and every parameter gradient including tied-head accumulation; it consumes
 * the forward cache it recomputes, leaving each layer's attention probabilities replaced by
 * their score adjoint. IDs: batch*time, strictly [0,vocab), rejected before arena mutation.
 * accumulate=1 adds another mean batch gradient. */
BCIR_DT_API int bcir_decoder_forward(bcir_decoder_state *s,const uint32_t *tokens,size_t count);
BCIR_DT_API int bcir_decoder_loss_backward(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t count,int accumulate,double *loss);
/* Atomic preflight: invalid/non-finite inputs or overflowing updates leave weights,
 * moments and step unchanged. Scale before global-L2 clipping. */
BCIR_DT_API int bcir_decoder_update(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    double gradient_scale,double *grad_norm);
/* Native loop, optional per-update rates; validate all IDs/outputs before any update.
 * A later numeric failure preserves completed steps and reports completed count. */
BCIR_DT_API int bcir_decoder_run(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    const uint32_t *tokens,const uint32_t *targets,size_t token_count,
    const double *rates,size_t steps,bcir_decoder_event *events,
    size_t event_capacity,size_t *completed);
/* Average accumulation consecutive micro-batches; one clip and update per step. */
BCIR_DT_API int bcir_decoder_run_accum(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    const uint32_t *tokens,const uint32_t *targets,size_t token_count,
    const double *rates,size_t steps,size_t accumulation,bcir_decoder_event *events,
    size_t event_capacity,size_t *completed);
#ifdef __cplusplus
}
#endif
#endif
