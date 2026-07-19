/*===- bcir_llama.c - standalone greedy inference over BCIRQ8 v1 --------===*/
#include "bcir_llama.h"

#include "bcir_ai_kernels.h"
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

static int tensor_ready(const bcir_q8_model *m, uint16_t id, int16_t layer,
                        uint8_t rank, uint32_t d0, uint32_t d1) {
  const bcir_q8_tensor *t=bcir_q8_model_tensor(m,id,layer);
  uint64_t count,groups,exponent_bytes;
  if(!t||t->rank!=rank||t->dim0!=d0||t->dim1!=d1||!d0||!d1) return 0;
  count=rank==1u?(uint64_t)d0:(uint64_t)d0*d1;
  if(count>UINT32_MAX) return 0;
  groups=(count+m->group_size-1u)/m->group_size;
  exponent_bytes=groups*2u;
  if(count!=t->element_count||groups!=t->group_count||
     t->exponent_offset>m->storage_size||exponent_bytes>m->storage_size-t->exponent_offset||
     t->code_offset>m->storage_size||count>m->storage_size-t->code_offset) return 0;
  /* bcir_q8_model_load already validated every immutable exponent/code byte and
   * CRC.  Re-scanning the complete model on every generation request would turn
   * request setup into O(parameter_count); retain the structural/stale-pointer
   * checks here and use the prevalidated kernels below. */
  return t->exponents == m->storage + t->exponent_offset &&
         t->codes == m->storage + t->code_offset;
}

static int model_ready(const bcir_q8_model *m) {
  uint32_t layer,d,dk,kvd,expected;
  if(!m||m->_owner_tag!=BCIR_Q8_MODEL_OWNER_TAG||
     !bcir_host_allocator_valid(&m->_allocator)||!m->storage||!m->tensors||
     !m->group_size||m->bits!=8u||!m->vocab_size||!m->d_model||!m->n_heads||!m->n_kv_heads||
     !m->n_layers||!m->d_ff||m->n_layers>(uint32_t)INT16_MAX||
     m->d_model%m->n_heads||m->n_heads%m->n_kv_heads||
     (m->d_model/m->n_heads)%2u||!isfinite(m->rope_base)||m->rope_base<=0.0||
     !isfinite(m->rms_norm_eps)||m->rms_norm_eps<=0.0||
     (m->flags&~BCIR_Q8_FLAG_TIED_EMBEDDINGS)) return 0;
  if ((m->bos_token_id != -1 &&
       (m->bos_token_id < 0 || (uint32_t)m->bos_token_id >= m->vocab_size)) ||
      (m->eos_token_id != -1 &&
       (m->eos_token_id < 0 || (uint32_t)m->eos_token_id >= m->vocab_size)) ||
      (m->pad_token_id != -1 &&
       (m->pad_token_id < 0 || (uint32_t)m->pad_token_id >= m->vocab_size))) return 0;
  d=m->d_model;dk=d/m->n_heads;kvd=m->n_kv_heads*dk;
  expected=2u+9u*m->n_layers+
           ((m->flags&BCIR_Q8_FLAG_TIED_EMBEDDINGS)?0u:1u);
  if(m->tensor_count!=expected||
     !tensor_ready(m,BCIR_Q8_TENSOR_EMBEDDING,-1,2,m->vocab_size,d)) return 0;
  for(layer=0;layer<m->n_layers;layer++){
    int16_t li=(int16_t)layer;
    if(!tensor_ready(m,BCIR_Q8_TENSOR_G_ATTN,li,1,d,1)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_Q,li,2,d,d)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_K,li,2,d,kvd)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_V,li,2,d,kvd)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_O,li,2,d,d)||
       !tensor_ready(m,BCIR_Q8_TENSOR_G_FF,li,1,d,1)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_GATE,li,2,d,m->d_ff)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_UP,li,2,d,m->d_ff)||
       !tensor_ready(m,BCIR_Q8_TENSOR_W_DOWN,li,2,m->d_ff,d)) return 0;
  }
  if(!tensor_ready(m,BCIR_Q8_TENSOR_G_FINAL,-1,1,d,1)) return 0;
  return (m->flags&BCIR_Q8_FLAG_TIED_EMBEDDINGS)||
         tensor_ready(m,BCIR_Q8_TENSOR_LM_HEAD,-1,2,m->vocab_size,d);
}

/* The public accessor defends against foreign/stale tensor pointers.  The whole
 * model has already passed model_ready() on this hot inference path, so avoid
 * repeating those ownership/span checks for every weight element. */
static double tensor_value_unchecked(const bcir_q8_model *m,
                                     const bcir_q8_tensor *t,uint32_t index) {
  const unsigned char *ep=t->exponents+2u*(index/m->group_size);
  int exponent=(int)(int16_t)((uint16_t)ep[0]|((uint16_t)ep[1]<<8));
  int code=(int)t->codes[index];
  if(code>=128)code-=256;
  return ldexp((double)code,exponent);
}

static int workspace_init(workspace *w, const bcir_q8_model *m, size_t capacity,
                          const bcir_host_allocator *allocator) {
  size_t cache_rows, cache_values, total_values = 0u, term, workspace_bytes;
  size_t d = m->d_model, kvd = (size_t)m->n_kv_heads * (m->d_model / m->n_heads);
  memset(w, 0, sizeof *w); w->capacity = capacity;
  w->allocator=bcir_host_allocator_or_default(allocator);
  if (mul_size(m->n_layers, capacity, &cache_rows) ||
      mul_size(cache_rows, kvd, &cache_values)) return -1;
  /* Eight d-sized vectors; three kvd vectors; gate/up/ff; scores; logits;
   * and two complete layer x position x kvd caches. Budget the whole operation
   * before making any allocator call so refusal is failure-atomic. */
  if (mul_size(8u, d, &term) || !bcir_size_add(total_values, term, &total_values) ||
      mul_size(3u, kvd, &term) || !bcir_size_add(total_values, term, &total_values) ||
      mul_size(3u, (size_t)m->d_ff, &term) ||
          !bcir_size_add(total_values, term, &total_values) ||
      !bcir_size_add(total_values, capacity, &total_values) ||
      !bcir_size_add(total_values, (size_t)m->vocab_size, &total_values) ||
      mul_size(2u, cache_values, &term) ||
          !bcir_size_add(total_values, term, &total_values) ||
      mul_size(total_values, sizeof(double), &workspace_bytes) ||
      workspace_bytes > (size_t)BCIR_LLAMA_MAX_WORKSPACE_BYTES) return -1;
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
  for (i = 0; i < t->element_count; ++i) out[i] = tensor_value_unchecked(m, t, i);
}

/* x[1 x in] @ weight[in x out], matching matmul_reference's i/j/k loop order. */
static int matvec(const bcir_q8_model *m, const double *x,
                  const bcir_q8_tensor *weight, uint32_t in, uint32_t out,
                  double *result) {
  return bcir_ai_q8_matvec_f64_prevalidated(
      x, in, weight->codes, weight->element_count, weight->exponents,
      weight->group_count, out, m->group_size, result);
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
    w->x[i] = tensor_value_unchecked(m, embedding, (uint32_t)token * d + i);

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
    if (matvec(m, w->h, wq, d, d, w->q) ||
        matvec(m, w->h, wk, d, kvd, w->k) ||
        matvec(m, w->h, wv, d, kvd, w->v)) return -1;
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
    if (matvec(m, w->context, wo, d, d, w->attn)) return -1;
    for (i = 0; i < d; ++i) w->x[i] += w->attn[i];
    dequant_vector(m, g_ff, w->gamma);
    bcir_rmsnorm(w->x, 1, (int)d, w->gamma, m->rms_norm_eps, w->h2);
    if (matvec(m, w->h2, wg, d, m->d_ff, w->gate) ||
        matvec(m, w->h2, wu, d, m->d_ff, w->up)) return -1;
    for (i = 0; i < m->d_ff; ++i)
      w->ff[i] = w->gate[i] * sigmoid_guarded(w->gate[i]) * w->up[i];
    if (matvec(m, w->ff, wd, m->d_ff, d, w->attn)) return -1;
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
                             double *logits, int *best_out) {
  const bcir_q8_tensor *head =
      (m->flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS)
          ? bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_EMBEDDING, -1)
          : bcir_q8_model_tensor(m, BCIR_Q8_TENSOR_LM_HEAD, -1);
  uint32_t token, best = 0;
  if (!best_out || bcir_ai_q8_rows_dot_f64_prevalidated(
          row, m->d_model, head->codes, head->element_count, head->exponents,
          head->group_count, m->vocab_size, m->group_size, logits)) return -1;
  for (token = 1; token < m->vocab_size; ++token)
    if (logits[token] > logits[best]) best = token;
  *best_out = (int)best;
  return 0;
}

int bcir_llama_generate_greedy_with_allocator(
    const bcir_q8_model *model,const int32_t *prompt_ids,size_t prompt_count,
    size_t max_new_tokens,int32_t *generated_ids,double *final_logits,
    const bcir_host_allocator *allocator) {
  workspace w;
  size_t capacity, pos, step;
  int next;
  if (!model || !prompt_ids || !prompt_count || !model_ready(model) ||
      (max_new_tokens && !generated_ids)) return -1;
  if (model->vocab_size > INT32_MAX || model->d_model > INT32_MAX ||
      model->n_heads > INT32_MAX || model->n_kv_heads > INT32_MAX ||
      model->n_layers > INT16_MAX || model->d_ff > INT32_MAX) return -1;
  for(pos=0;pos<prompt_count;pos++)
    if(prompt_ids[pos]<0||(uint32_t)prompt_ids[pos]>=model->vocab_size) return -1;
  if (!max_new_tokens) return 0;
  if (prompt_count > SIZE_MAX - max_new_tokens) return -1;
  capacity = prompt_count + max_new_tokens;
  if (model->context_length && capacity-1u > (size_t)model->context_length) return -2;
  if (capacity > (size_t)INT32_MAX || workspace_init(&w, model, capacity,allocator)) return -3;
  for (pos = 0; pos < prompt_count; ++pos)
    if (step_token(model, &w, prompt_ids[pos], pos, w.h)) {
      workspace_free(&w); return -1;
    }
  for (step = 0; step < max_new_tokens; ++step) {
    if (logits_and_argmax(model, w.h, w.logits, &next)) {
      workspace_free(&w); return -1;
    }
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
