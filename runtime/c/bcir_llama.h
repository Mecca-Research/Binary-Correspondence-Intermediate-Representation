/*===- bcir_llama.h - standalone greedy inference over BCIRQ8 v1 --------===*/
#ifndef BCIR_LLAMA_H
#define BCIR_LLAMA_H

#include "bcir_q8_model.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* This is a bounded reference runtime, not an unbounded serving allocator.
 * Requests whose complete double-precision scratch/KV workspace would exceed
 * this ceiling fail before the first allocation or output write. */
#define BCIR_LLAMA_MAX_WORKSPACE_BYTES (128u * 1024u * 1024u)

/* DECODE-THREADS: a runner the caller lends the decoder. run(self, task, ctx, tasks) runs task(ctx, 0) to
 * task(ctx, tasks - 1), each once, on threads of its own choosing, and returns when every one has finished;
 * `workers` is how many it runs at once, at most BCIR_LLAMA_MAX_PARTS. The decoder hands it the parts of one
 * operation's outputs: a matvec of at least split_min weights in ranges of its columns (multiples of eight), and
 * the head over its pack of at least head_split_min weights in ranges of its rows (whole 32-row passes), `workers`
 * parts each, fewer where the outputs are few. Every output is computed by one part with its chain the whole
 * call's, so every token and every logit is the same at every worker count, at every threshold and with no runner.
 * The thresholds are what a part costs the runner set against what it shares. BCIR_LLAMA_SPLIT_MIN and
 * BCIR_LLAMA_HEAD_SPLIT_MIN were measured with the trainer's deterministic pool (bcir_tensor_pool_run, which
 * bcir-llama --threads lends) on the host the decoder was measured on: below 64 Ki weights a matvec's parts cost more
 * than they share, and the head, once a token, splits from 1 Mi weights, where it pays for waking a pool the matvecs
 * left idle; 0 splits every one. The decoder calls a runner from one thread at a time. */
typedef void (*bcir_llama_task_fn)(void *ctx, size_t task);
typedef struct bcir_llama_runner {
  void (*run)(void *self, bcir_llama_task_fn task, void *ctx, size_t tasks);
  void *self;
  unsigned workers;
  size_t split_min, head_split_min;
} bcir_llama_runner;
#define BCIR_LLAMA_MAX_PARTS 64u
#define BCIR_LLAMA_SPLIT_MIN ((size_t)1 << 16)
#define BCIR_LLAMA_HEAD_SPLIT_MIN ((size_t)1 << 20)

/* Greedy, lowest-id-tie-breaking Llama/SwiGLU decode from verified token ids.
 * `generated_ids` has max_new_tokens entries; `final_logits`, when non-NULL, has
 * model->vocab_size entries and receives the scores that selected the last id.
 * Returns 0 on success and a negative value on invalid input, context/workspace
 * limit, or allocation failure. */
int bcir_llama_generate_greedy(const bcir_q8_model *model,
                               const int32_t *prompt_ids, size_t prompt_count,
                               size_t max_new_tokens,
                               int32_t *generated_ids,
                               double *final_logits);

/* Allocator-injected form. The allocator is borrowed for the call; all scratch
 * is released before return, including every failure path. */
int bcir_llama_generate_greedy_with_allocator(
    const bcir_q8_model *model, const int32_t *prompt_ids,
    size_t prompt_count, size_t max_new_tokens, int32_t *generated_ids,
    double *final_logits, const bcir_host_allocator *allocator);

/* DECODE-THREADS: the allocator-injected form with its workspace lent `runner` (NULL: none) -- every token and every
 * final logit the same. A runner of no function with more than one worker, or of more than BCIR_LLAMA_MAX_PARTS
 * workers, fails the call before any allocation. */
int bcir_llama_generate_greedy_with_runner(
    const bcir_q8_model *model, const int32_t *prompt_ids,
    size_t prompt_count, size_t max_new_tokens, int32_t *generated_ids,
    double *final_logits, const bcir_host_allocator *allocator,
    const bcir_llama_runner *runner);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_LLAMA_H */
