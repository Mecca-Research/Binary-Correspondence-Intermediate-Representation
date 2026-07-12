/*===- bcir_llama.c - standalone greedy inference over BCIRQ8 v1 --------===*/
#include "bcir_llama.h"

#include "bcir_decode.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct workspace {
  double *x, *h, *h2, *q, *q_rope, *k, *k_rope, *v;
  double *context, *attn, *gate, *up, *ff, *gamma, *scores, *logits;
  double *k_cache, *v_cache;
  size_t capacity;
  bcir_host_allocator allocator;
} workspace;

static int mul_size(size_t a, size_t b, size_t *out) {
  if (b && a > SIZE_MAX / b) return -1;
  *out = a * b;
  return 0;
}

static double *new_doubles(workspace *w,size_t count) {
  size_t bytes;
  double *result;
  if (!bcir_size_mul(count ? count : 1u,sizeof(double),&bytes)) return NULL;
  result=(double *)bcir_host_allocate(&w->allocator,bytes);
  if(result)memset(result,0,bytes);
  return result;
}

static void workspace_free(workspace *w) {
  bcir_host_allocator allocator=w->allocator;
  bcir_host_deallocate(&allocator,w->x); bcir_host_deallocate(&allocator,w->h);
  bcir_host_deallocate(&allocator,w->h2); bcir_host_deallocate(&allocator,w->q);
  bcir_host_deallocate(&allocator,w->q_rope); bcir_host_deallocate(&allocator,w->k);
  bcir_host_deallocate(&allocator,w->k_rope); bcir_host_deallocate(&allocator,w->v);
  bcir_host_deallocate(&allocator,w->context); bcir_host_deallocate(&allocator,w->attn);
  bcir_host_deallocate(&allocator,w->gate); bcir_host_deallocate(&allocator,w->up);
  bcir_host_deallocate(&allocator,w->ff); bcir_host_deallocate(&allocator,w->gamma);
  bcir_host_deallocate(&allocator,w->scores); bcir_host_deallocate(&allocator,w->logits);
  bcir_host_deallocate(&allocator,w->k_cache); bcir_host_deallocate(&allocator,w->v_cache);
  memset(w, 0, sizeof *w);
}

static int workspace_init(workspace *w, const bcir_q8_model *m, size_t capacity,
                          const bcir_host_allocator *allocator) {
  size_t cache_rows, cache_values;
  size_t d = m->d_model, kvd = (size_t)m->n_kv_heads * (m->d_model / m->n_heads);
  memset(w, 0, sizeof *w); w->capacity = capacity;
  w->allocator=bcir_host_allocator_or_default(allocator);
  if (mul_size(m->n_layers, capacity, &cache_rows) ||
      mul_size(cache_rows, kvd, &cache_values)) return -1;
  w->x = new_doubles(w,d); w->h = new_doubles(w,d); w->h2 = new_doubles(w,d);
  w->q = new_doubles(w,d); w->q_rope = new_doubles(w,d);
  w->k = new_doubles(w,kvd); w->k_rope = new_doubles(w,kvd); w->v = new_doubles(w,kvd);
  w->context = new_doubles(w,d); w->attn = new_doubles(w,d);
  w->gate = new_doubles(w,m->d_ff); w->up = new_doubles(w,m->d_ff);
  w->ff = new_doubles(w,m->d_ff); w->gamma = new_doubles(w,d);
  w->scores = new_doubles(w,capacity); w->logits = new_doubles(w,m->vocab_size);
  w->k_cache = new_doubles(w,cache_values); w->v_cache = new_doubles(w,cache_values);
  if (!w->x || !w->h || !w->h2 || !w->q || !w->q_rope || !w->k || !w->k_rope ||
      !w->v || !w->context || !w->attn || !w->gate || !w->up || !w->ff ||
      !w->gamma || !w->scores || !w->logits || !w->k_cache || !w->v_cache) {
    workspace_free(w);
    return -1;
  }
  return 0;
}

static void dequant_vector(const bcir_q8_model *m, const bcir_q8_tensor *t,
                           double *out) {
  uint32_t i;
  for (i = 0; i < t->element_count; ++i) out[i] = bcir_q8_tensor_value(m, t, i);
}

/* x[1 x in] @ weight[in x out], matching matmul_reference's i/j/k loop order. */
static void matvec(const bcir_q8_model *m, const double *x,
                   const bcir_q8_tensor *weight, uint32_t in, uint32_t out,
                   double *result) {
  uint32_t j, k;
  for (j = 0; j < out; ++j) {
    double sum = 0.0;
    for (k = 0; k < in; ++k)
      sum += x[k] * bcir_q8_tensor_value(m, weight, k * out + j);
    result[j] = sum;
  }
}

static double sigmoid_guarded(double x) {
  if (x >= 0.0) {
    double z = exp(-x);
    return 1.0 / (1.0 + z);
  }
  {
    double z = exp(x);
    return z / (1.0 + z);
  }
}

static int step_token(const bcir_q8_model *m, workspace *w, int32_t token, size_t pos,
                      double *final_row) {
  const bcir_q8_tensor *embedding;
  uint32_t d = m->d_model, dk = d / m->n_heads;
  uint32_t kvd = m->n_kv_heads * dk;
  uint32_t layer, i, head;
  if (token < 0 || (uint32_t)token >= m->vocab_size || pos >= w->capacity) return -1;
  embedding = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_EMBEDDING, -1);
  for (i = 0; i < d; ++i)
    w->x[i] = bcir_q8_tensor_value(m, embedding, (uint32_t)token * d + i);

  for (layer = 0; layer < m->n_layers; ++layer) {
    int16_t li = (int16_t)layer;
    const bcir_q8_tensor *g_attn = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_G_ATTN, li);
    const bcir_q8_tensor *wq = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_Q, li);
    const bcir_q8_tensor *wk = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_K, li);
    const bcir_q8_tensor *wv = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_V, li);
    const bcir_q8_tensor *wo = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_O, li);
    const bcir_q8_tensor *g_ff = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_G_FF, li);
    const bcir_q8_tensor *wg = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_GATE, li);
    const bcir_q8_tensor *wu = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_UP, li);
    const bcir_q8_tensor *wd = bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_W_DOWN, li);
    size_t layer_base = (size_t)layer * w->capacity * kvd;
    double *k_layer = w->k_cache + layer_base;
    double *v_layer = w->v_cache + layer_base;

    dequant_vector(m, g_attn, w->gamma);
    bcir_rmsnorm(w->x, 1, (int)d, w->gamma, m->rms_norm_eps, w->h);
    matvec(m, w->h, wq, d, d, w->q);
    matvec(m, w->h, wk, d, kvd, w->k);
    matvec(m, w->h, wv, d, kvd, w->v);
    for (head = 0; head < m->n_heads; ++head)
      bcir_rope(w->q + (size_t)head * dk, 1, (int)dk, m->rope_base, (int)pos,
                w->q_rope + (size_t)head * dk);
    for (head = 0; head < m->n_kv_heads; ++head)
      bcir_rope(w->k + (size_t)head * dk, 1, (int)dk, m->rope_base, (int)pos,
                w->k_rope + (size_t)head * dk);
    memcpy(k_layer + pos * kvd, w->k_rope, (size_t)kvd * sizeof(double));
    memcpy(v_layer + pos * kvd, w->v, (size_t)kvd * sizeof(double));
    if (bcir_gqa_attention_row(w->q_rope, k_layer, v_layer, (int)pos + 1,
                               (int)m->n_heads, (int)m->n_kv_heads, (int)dk,
                               w->scores, w->context)) return -1;
    matvec(m, w->context, wo, d, d, w->attn);
    for (i = 0; i < d; ++i) w->x[i] += w->attn[i];
    dequant_vector(m, g_ff, w->gamma);
    bcir_rmsnorm(w->x, 1, (int)d, w->gamma, m->rms_norm_eps, w->h2);
    matvec(m, w->h2, wg, d, m->d_ff, w->gate);
    matvec(m, w->h2, wu, d, m->d_ff, w->up);
    for (i = 0; i < m->d_ff; ++i)
      w->ff[i] = w->gate[i] * sigmoid_guarded(w->gate[i]) * w->up[i];
    matvec(m, w->ff, wd, m->d_ff, d, w->attn);
    for (i = 0; i < d; ++i) w->x[i] += w->attn[i];
  }
  {
    const bcir_q8_tensor *g_final =
        bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_G_FINAL, -1);
    dequant_vector(m, g_final, w->gamma);
    bcir_rmsnorm(w->x, 1, (int)d, w->gamma, m->rms_norm_eps, final_row);
  }
  return 0;
}

static int logits_and_argmax(const bcir_q8_model *m, const double *row,
                             double *logits) {
  const bcir_q8_tensor *head =
      (m->flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS)
          ? bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_EMBEDDING, -1)
          : bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_LM_HEAD, -1);
  uint32_t token, k, best = 0;
  for (token = 0; token < m->vocab_size; ++token) {
    double sum = 0.0;
    for (k = 0; k < m->d_model; ++k)
      sum += row[k] * bcir_q8_tensor_value(m, head, token * m->d_model + k);
    logits[token] = sum;
    if (token && logits[token] > logits[best]) best = token;
  }
  return (int)best;
}

int bcir_llama_generate_greedy_with_allocator(
    const bcir_q8_model *model,const int32_t *prompt_ids,size_t prompt_count,
    size_t max_new_tokens,int32_t *generated_ids,double *final_logits,
    const bcir_host_allocator *allocator) {
  workspace w;
  size_t capacity, pos, step;
  int next;
  if (!model || !prompt_ids || !prompt_count ||
      (max_new_tokens && !generated_ids)) return -1;
  if (model->vocab_size > INT32_MAX || model->d_model > INT32_MAX ||
      model->n_heads > INT32_MAX || model->n_kv_heads > INT32_MAX ||
      model->n_layers > INT16_MAX || model->d_ff > INT32_MAX) return -1;
  if (!max_new_tokens) return 0;
  if (prompt_count > SIZE_MAX - max_new_tokens) return -1;
  capacity = prompt_count + max_new_tokens;
  if (model->context_length && capacity > (size_t)model->context_length + 1u) return -2;
  if (capacity > (size_t)INT32_MAX || workspace_init(&w, model, capacity,allocator)) return -3;
  for (pos = 0; pos < prompt_count; ++pos)
    if (step_token(model, &w, prompt_ids[pos], pos, w.h)) {
      workspace_free(&w); return -1;
    }
  for (step = 0; step < max_new_tokens; ++step) {
    next = logits_and_argmax(model, w.h, w.logits);
    generated_ids[step] = (int32_t)next;
    if (step + 1 < max_new_tokens &&
        step_token(model, &w, (int32_t)next, prompt_count + step, w.h)) {
      workspace_free(&w); return -1;
    }
  }
  if (final_logits)
    memcpy(final_logits, w.logits, (size_t)model->vocab_size * sizeof(double));
  workspace_free(&w);
  return 0;
}

int bcir_llama_generate_greedy(const bcir_q8_model *model,
                               const int32_t *prompt_ids,size_t prompt_count,
                               size_t max_new_tokens,int32_t *generated_ids,
                               double *final_logits) {
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_llama_generate_greedy_with_allocator(model,prompt_ids,prompt_count,
                                                    max_new_tokens,generated_ids,
                                                    final_logits,&allocator);
}
