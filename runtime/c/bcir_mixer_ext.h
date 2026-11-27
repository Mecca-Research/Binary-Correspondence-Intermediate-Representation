/*===- bcir_mixer_ext.h - Production Pluggable Sequence-Mixer ABI (ABI 6) ---===
 * Native C11 sequence-mixer training & evaluation runtime built on bcir_tensor.h
 * and bcir_precision.h. Preserves ABI 5 (bcir_decoder_train.h) untouched while
 * adding per-layer pluggable sequence mixers and 2025-2026 frontier upgrades:
 *
 * Mixers (layer_mixer[l]):
 *   - BCIR_MIXER_GQA      (0): Grouped-Query Attention (materialized or TRAIN-OPT-5 online softmax)
 *   - BCIR_MIXER_SWA      (1): Sliding Window Attention (O(T*W) cache)
 *   - BCIR_MIXER_DELTANET (2): Arch-1 Gated DeltaNet (scalar alpha/beta, FP64 chunkwise WY)
 *   - BCIR_MIXER_KDA      (3): Kimi Delta Attention (arXiv:2510.26692; channel-wise Diag(alpha),
 *                              low-rank decay bottleneck, per-head RMSNorm, FP64 chunkwise WY)
 *   - BCIR_MIXER_MLA      (4): Arch-4 Multi-Head Latent Attention (decoupled RoPE or NoPE,
 *                              kv_norm, absorbed decode equivalence, optional online softmax)
 *   - BCIR_MIXER_GDN2     (5): Gated DeltaNet-2 (arXiv:2605.22791v2; channel-wise Diag(alpha)
 *                              + decoupled channel-wise erase b_t & write w_t, FP64 WY + exact AD)
 *
 * Orthogonal architectural upgrades (default 0 for full backward compatibility):
 *   - gate_mode      (0=off, 1=headwise G1 SDPA sigmoid gate, 2=elementwise low-rank G1; arXiv:2505.06708)
 *   - qk_norm        (0=off, 1=per-head Q/K RMSNorm on GQA/SWA and Q RMSNorm on MLA)
 *   - canon_mode     (0=off, 1=Canon-A pre-mixer, 2=Canon-A+C causal conv1d k=4; arXiv:2512.17351v2)
 *   - mhc_streams    (0/1=standard residual, 2 or 4=DeepSeek mHC Birkhoff streams; arXiv:2512.24880v2)
 *   - online_softmax (0=materialized O(T^2), 1=TRAIN-OPT-5 O(T) LSE tiled attention for GQA & MLA)
 *
 * Pure ISO C11, zero hot-path mallocs, -ffp-contract=off, 0-bit-drift 1T..NT parity.
 * Python binding: bcir/hosted/models/mixer_ext.py.
 *===------------------------------------------------------------------------===*/
#ifndef BCIR_MIXER_EXT_H
#define BCIR_MIXER_EXT_H

#include "bcir_precision.h"
#include "bcir_tensor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_EXT_MAX_LAYERS 64u
#define BCIR_EXT_IGNORE     0xFFFFFFFFu
#define BCIR_EXT_CANON_K    4u
#define BCIR_EXT_MHC_MAX_C  4u

enum {
  BCIR_MIXER_GQA      = 0,
  BCIR_MIXER_SWA      = 1,
  BCIR_MIXER_DELTANET = 2,
  BCIR_MIXER_KDA      = 3,
  BCIR_MIXER_MLA      = 4,
  BCIR_MIXER_GDN2     = 5,
  BCIR_MIXER_KIND_COUNT = 6
};

enum {
  BCIR_GATE_NONE        = 0,
  BCIR_GATE_G1_HEADWISE = 1,
  BCIR_GATE_G1_LOWRANK  = 2
};

enum {
  BCIR_CANON_OFF = 0,
  BCIR_CANON_A   = 1,
  BCIR_CANON_AC  = 2
};

BCIR_DT_API uint32_t bcir_mixer_ext_abi_version(void); /* returns 6 */

typedef struct bcir_ext_spec {
  uint32_t vocab, width, heads, kvheads, layers, ff, batch, time, tied;
  double rope_base, rms_eps;
  uint32_t layer_mixer[BCIR_EXT_MAX_LAYERS];
  uint32_t layer_use_rope[BCIR_EXT_MAX_LAYERS];
  uint32_t swa_window;
  uint32_t delta_chunk;
  uint32_t mla_dc;
  uint32_t mla_dr;
  int init_policy;
  uint32_t mup_d_base;
  uint64_t seed;
  uint32_t threads;
  int skip_init;
  /* ABI 6 extensions (zero-initialized defaults preserve exact Exp 4 behavior). */
  uint32_t gate_mode;      /* BCIR_GATE_* */
  uint32_t qk_norm;        /* 0 or 1 */
  uint32_t canon_mode;     /* BCIR_CANON_* */
  uint32_t mhc_streams;    /* 0/1 = off, 2 or 4 = Sinkhorn-Knopp Birkhoff streams */
  uint32_t online_softmax; /* 0 = O(T^2) materialized, 1 = TRAIN-OPT-5 O(T) LSE tiled */
  uint32_t gate_rank;      /* 0 = default max(16, dh) bottleneck for KDA/GDN2/G1_LOWRANK */
} bcir_ext_spec;

typedef struct bcir_ext_mem_stats {
  size_t n_params, attn_params, mlp_params, nonlin_params;
  size_t weight_bytes, opt_only_bytes, persistent_bytes;
  size_t activation_bytes, mixer_prob_bytes, scratch_bytes, total_bytes;
  size_t infer_kv_bytes_per_tok, infer_rec_state_bytes;
  double init_ms;
} bcir_ext_mem_stats;

typedef struct bcir_ext_step_metrics {
  double loss, grad_norm, step_ms;
  int clipped, status;
  uint64_t step, supervised_tokens, fp64_promotions, nonfinite_events;
} bcir_ext_step_metrics;

typedef struct bcir_ext_eval_result {
  double ce, accuracy;
  uint64_t tokens, nonfinite;
  double eval_ms;
} bcir_ext_eval_result;

typedef struct bcir_ext_model bcir_ext_model;

BCIR_DT_API bcir_ext_model *bcir_ext_create(const bcir_ext_spec *spec, const bcir_prec_policy *pol);
BCIR_DT_API void bcir_ext_destroy(bcir_ext_model *m);
BCIR_DT_API void bcir_ext_get_mem_stats(const bcir_ext_model *m, bcir_ext_mem_stats *out);
BCIR_DT_API void bcir_ext_get_telemetry(const bcir_ext_model *m, bcir_prec_telemetry *out);
BCIR_DT_API void bcir_ext_reset_telemetry(bcir_ext_model *m);
BCIR_DT_API void bcir_ext_reset_opt(bcir_ext_model *m);
BCIR_DT_API size_t bcir_ext_opt_state_bytes(const bcir_ext_model *m);
BCIR_DT_API int bcir_ext_save_opt_state(const bcir_ext_model *m, void *dst, size_t dst_bytes);
BCIR_DT_API int bcir_ext_load_opt_state(bcir_ext_model *m, const void *src, size_t src_bytes);
BCIR_DT_API float *bcir_ext_weights_ptr(bcir_ext_model *m);
BCIR_DT_API float *bcir_ext_grads_fp32_ptr(bcir_ext_model *m);
BCIR_DT_API size_t bcir_ext_param_count(const bcir_ext_model *m);

typedef struct bcir_ext_tensor_info {
  size_t offset, rows, cols;
  int layer, region;
  char name[48];
} bcir_ext_tensor_info;

BCIR_DT_API size_t bcir_ext_tensor_count(const bcir_ext_model *m);
BCIR_DT_API int bcir_ext_tensor_info_at(const bcir_ext_model *m, size_t idx, bcir_ext_tensor_info *out);

BCIR_DT_API int bcir_ext_forward(bcir_ext_model *m, const uint32_t *tokens, uint32_t active_time,
                                 float *out_logits);
BCIR_DT_API int bcir_ext_evaluate(bcir_ext_model *m, const uint32_t *tokens, const uint32_t *targets,
                                  size_t n_batches, uint32_t active_time, bcir_ext_eval_result *out);
BCIR_DT_API int bcir_ext_loss_backward(bcir_ext_model *m, const uint32_t *tokens, const uint32_t *targets,
                                       uint32_t active_time, int accumulate,
                                       double *out_loss, uint64_t *out_sup_tokens);
BCIR_DT_API int bcir_ext_train_loop(bcir_ext_model *m, const uint32_t *tokens, const uint32_t *targets,
                                    const uint32_t *active_times_or_null, const double *lrs,
                                    size_t steps, const bcir_adamw_params *opt_cfg,
                                    bcir_ext_step_metrics *events);

/* Gate 0 mathematical identity verifiers. */
BCIR_DT_API int bcir_deltanet_verify_wy_vs_recurrent(uint32_t B, uint32_t T, uint32_t H, uint32_t dh,
                                                     uint32_t C, double *out_max_abs_diff);
BCIR_DT_API int bcir_kda_verify_wy_vs_recurrent(uint32_t B, uint32_t T, uint32_t H, uint32_t dh,
                                                uint32_t C, double *out_max_abs_diff);
BCIR_DT_API int bcir_gdn2_verify_wy_vs_recurrent(uint32_t B, uint32_t T, uint32_t H, uint32_t dh,
                                                 uint32_t C, double *out_max_abs_diff);
BCIR_DT_API int bcir_mla_verify_absorption(uint32_t T, uint32_t H, uint32_t dh, uint32_t dc, uint32_t dr,
                                           double *out_max_abs_diff);
BCIR_DT_API int bcir_mhc_verify_birkhoff(uint32_t C, double *out_max_marginal_err,
                                         double *out_max_grad_err);

#ifdef __cplusplus
}
#endif
#endif
