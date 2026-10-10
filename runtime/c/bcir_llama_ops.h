/*===- bcir_llama_ops.h - the decoder's operations, one function each ----===
 *
 * The kernels of a Llama/SwiGLU decoder over BCIRQ8 v1, one function per operation, on a
 * caller-owned workspace. Two dispatchers call them:
 *
 *   - bcir_llama.c's monolithic step, in its fixed order (bcir_llama_generate_greedy);
 *   - bcir_decoder_gem.c's StreamPack interpreter, as a planned program's claims dispatch
 *     through GEM (bcir_sp_execute).
 *
 * One set of kernels and two dispatchers: a GEM-executed decoder program computes what the
 * monolithic runner computes, operation for operation and bit for bit, because it runs the
 * same functions on the same values in the same order the program's data flow fixes.
 *
 * Contracts. Every operation assumes bcir_llama_model_ready(model) and a workspace that
 * bcir_llama_ws_init built for that model; the model is borrowed and must outlive the
 * workspace. Positions and layers are bounded by the caller (both dispatchers check them
 * against the workspace's capacity before calling). The workspace's buffers are owned by the
 * workspace, allocated through its bcir_host_allocator and released by bcir_llama_ws_free.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_LLAMA_OPS_H
#define BCIR_LLAMA_OPS_H

#include "bcir_host_alloc.h"
#include "bcir_q8_model.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The decoder's state between operations: activations in double, the K/V caches of every
 * layer (layer x capacity x kv_dim), and the scratch the kernels share. */
typedef struct bcir_llama_ws {
  double *x, *h, *h2, *q, *q_rope, *k, *k_rope, *v;
  double *context, *attn, *gate, *up, *ff, *gamma, *scores, *logits;
  double *k_cache, *v_cache;
  size_t capacity;
  bcir_host_allocator allocator;
} bcir_llama_ws;

/* Nonzero when `model` is a complete, well-formed Llama/SwiGLU BCIRQ8 decoder. */
int bcir_llama_model_ready(const bcir_q8_model *model);

/* Build a workspace for `capacity` positions (the whole request is budgeted against
 * BCIR_LLAMA_MAX_WORKSPACE_BYTES before any allocation). 0 on success; on failure nothing
 * stays allocated. The allocator is copied; NULL selects the default. */
int bcir_llama_ws_init(bcir_llama_ws *w, const bcir_q8_model *model, size_t capacity,
                       const bcir_host_allocator *allocator);
/* Release every buffer the workspace owns and zero it. Idempotent on a zeroed workspace. */
void bcir_llama_ws_free(bcir_llama_ws *w);

/* The logits head: the embedding table when tied, else lm_head. */
const bcir_q8_tensor *bcir_llama_head_tensor(const bcir_q8_model *model);

/* x = embedding[token]. -1 on a token outside the vocabulary. */
int bcir_llama_op_embed(const bcir_q8_model *model, bcir_llama_ws *w, int32_t token);
/* dst = rmsnorm(src) * gamma (gamma dequantized into the workspace's scratch). */
void bcir_llama_op_rmsnorm(const bcir_q8_model *model, bcir_llama_ws *w, const double *src,
                           const bcir_q8_tensor *gamma, double *dst);
/* y[out] = x[in] @ weight[in x out], the ascending-k order of matmul_reference. */
int bcir_llama_op_matvec(const bcir_q8_model *model, const double *x,
                         const bcir_q8_tensor *weight, uint32_t in, uint32_t out, double *y);
/* q_rope, k_rope = rope(q, k) at position `pos`, head by head. */
void bcir_llama_op_rope(const bcir_q8_model *model, bcir_llama_ws *w, size_t pos);
/* Layer `layer`'s caches at row `pos` = k_rope, v. */
void bcir_llama_op_kv_append(const bcir_q8_model *model, bcir_llama_ws *w, uint32_t layer,
                             size_t pos);
/* context = causal GQA attention of q_rope over layer `layer`'s rows 0..pos. */
int bcir_llama_op_attention(const bcir_q8_model *model, bcir_llama_ws *w, uint32_t layer,
                            size_t pos);
/* x += attn (the residual of the output and down projections). */
void bcir_llama_op_residual(const bcir_q8_model *model, bcir_llama_ws *w);
/* ff = silu(gate) * up, elementwise, with the guarded two-branch sigmoid. */
void bcir_llama_op_swiglu(const bcir_q8_model *model, bcir_llama_ws *w);
/* logits[vocab] = head . row. */
int bcir_llama_op_head(const bcir_q8_model *model, const double *row, double *logits);
/* The greedy pick: the largest logit, the lowest id on a tie. */
int32_t bcir_llama_op_argmax(const bcir_q8_model *model, const double *logits);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_LLAMA_OPS_H */
