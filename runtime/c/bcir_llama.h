/*===- bcir_llama.h - standalone greedy inference over BCIRQ8 v1 --------===*/
#ifndef BCIR_LLAMA_H
#define BCIR_LLAMA_H

#include "bcir_q8_model.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Greedy, lowest-id-tie-breaking Llama/SwiGLU decode from verified token ids.
 * `generated_ids` has max_new_tokens entries; `final_logits`, when non-NULL, has
 * model->vocab_size entries and receives the scores that selected the last id.
 * Returns 0 on success and a negative value on invalid input/allocation failure. */
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

#ifdef __cplusplus
}
#endif

#endif /* BCIR_LLAMA_H */
