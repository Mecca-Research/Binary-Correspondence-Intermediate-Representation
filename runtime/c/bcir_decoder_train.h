/* Complete dense Llama/SwiGLU training in C. No allocation in plan, forward,
 * backward, optimizer or multi-step run. FP32 weights/activations; FP64 loss/norm.
 * HF row-major projections and half-split RoPE (ingest permutes only at export).
 * The checked state owns no memory; arenas must be disjoint and float aligned.
 * Caller serializes access; tensor providers must honor bcir_tensor.h contracts.
 * The state ends in its thread pool (ABI 2, NDT-THREADS): NULL or a bcir_tensor_pool the
 * caller owns, run by every BCIR kernel the provider does not replace. Each kernel's
 * partition is fixed by the shapes, so the state after any number of steps is bit-identical
 * at every thread count and to the serial state; a provider threads its own kernels.
 * `contract` (ABI 3, NDT-FMA) names the arithmetic contract BCIR's GEMM keeps: 0 is v1 (one
 * rounded multiply and one rounded add per term), BCIR_DT_CONTRACT_FUSED is v2 (one fmaf per
 * term, k ascending -- bit-identical wherever fmaf is an IEEE fusedMultiplyAdd). A provider's
 * GEMM keeps its own. BCIR_DT_CONTRACT_EXP (NDT-EXP) puts SiLU, the SwiGLU VJP and the attention
 * softmax under their v2 contracts (the BCIR exponential, no libm); a provider's SiLU keeps its
 * own. BCIR_DT_CONTRACT_REDUCE (NDT-REDUCE) puts the clipping norm under a fixed-shape reduction:
 * the gradient in chunks of bcir_decoder_reduce_chunk elements (a function of the count alone),
 * each chunk's squares summed in index order in double from zero, the chunk sums added in chunk
 * order from zero -- so the update's one pass runs on the pool; without the bit the norm is the one
 * index-order chain, serial. BCIR_DT_CONTRACT_BF16 (ABI 4, NDT-BF16) puts BCIR's GEMMs under the
 * bf16 contract (bcir_tensor.h: operands rounded to bfloat16, accumulated as the AMX unit's
 * TDPBF16PS accumulates them, every instance bit for bit); the state then lends `workspace`, at
 * least bcir_decoder_workspace_bytes bytes disjoint from the arenas and the state, for the packed
 * operands (BCIR_DT_CAPACITY below that, BCIR_DT_INVALID without one or overlapping), and a
 * provider's GEMM, as under every contract, keeps its own. One GEMM contract at a time: FUSED,
 * KBLOCK and BF16 exclude one another. BCIR_DT_CONTRACT_ATTN (ATTN-FMA) puts the attention's
 * products under the fused contract (bcir_tensor.h's BCIR_TENSOR_ATTENTION_FUSED: every
 * multiply-add of the scores, the context and the backward's products one fmaf, the same terms in
 * the same order), with any GEMM contract or none. Other bits are refused. A state trained under one contract
 * is not another's: the checkpoint names it. Whatever the contract, the loss's rows, the RMSNorm gamma
 * adjoints and the embedding scatter run on the pool in the serial order (NDT-REDUCE), so every
 * loss and state is the serial one, bit for bit, at every thread count.
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
  bcir_tensor_pool *pool;
  uint32_t contract;
  void *workspace;          /* ABI 4 (NDT-BF16): the bf16 contract's packs; ignored otherwise */
  size_t workspace_bytes;
} bcir_decoder_state;
typedef struct bcir_decoder_adamw {
  double lr,beta1,beta2,epsilon,weight_decay,grad_clip;
} bcir_decoder_adamw;
typedef struct bcir_decoder_event { double loss,grad_norm; uint64_t step; } bcir_decoder_event;
enum { BCIR_DT_OK=0, BCIR_DT_INVALID=1, BCIR_DT_CAPACITY=2, BCIR_DT_NUMERIC=3 };
enum { BCIR_DT_CONTRACT_FUSED=1u, BCIR_DT_CONTRACT_EXP=2u, BCIR_DT_CONTRACT_KBLOCK=4u,
  BCIR_DT_CONTRACT_REDUCE=8u, BCIR_DT_CONTRACT_BF16=16u, BCIR_DT_CONTRACT_ATTN=32u };
/* 1: the state through its provider; 2: the state ends in its pool (NDT-THREADS); 3: and in its
 * contract (NDT-FMA, NDT-EXP, NDT-KBLOCK, NDT-REDUCE, ATTN-FMA: the same field, more bits); 4: and in the
 * bf16 contract's workspace (NDT-BF16). */
BCIR_DT_API uint32_t bcir_decoder_abi_version(void);
/* The bytes of workspace a state under `contract` lends its GEMMs: 0 unless the contract names
 * BF16; else the largest of the step's GEMM groups' packs, each call's packed whole
 * (bcir_tensor_mm_bf16_workspace summed over the group: an operand the calls share is packed once,
 * so the group needs no more), so no group of the step runs in chunks. SIZE_MAX for a plan
 * bcir_decoder_make_plan refuses. */
BCIR_DT_API size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan, uint32_t contract);
/* The reduce contract's chunk for `count` elements: 1024 doubled until the chunks number at most
 * 1024 (so 1024 up to 2^20 elements, 16384 for the 9.84M decoder). The update's first pass runs
 * one task per chunk under every contract; only under BCIR_DT_CONTRACT_REDUCE do the chunks reach
 * the norm. */
BCIR_DT_API size_t bcir_decoder_reduce_chunk(size_t count);
/* The clipping norm's square as the update's first pass computes it under the state's contract
 * -- the scaled gradient's squares in one index-order chain, or the reduce contract's chunk sums
 * in chunk order -- with the pass's verdict: BCIR_DT_NUMERIC on a non-finite element of the
 * gradient, weights or moments (a negative second moment included), nothing written then; the
 * update reports its square root. For the oracle's emulation, which compares the sum before the
 * root rounds a difference away. */
BCIR_DT_API int bcir_decoder_gradient_norm(bcir_decoder_state *s,double gradient_scale,
    double *squared);
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
 * A later numeric failure preserves completed steps and reports completed count.
 * A call's first step scans the weights before its forward; the steps after it skip that scan,
 * the weights being the call's own last commit -- every candidate admitted, finite -- with no
 * caller code run since (FACTS-RUN): one call of several steps scans once. */
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
