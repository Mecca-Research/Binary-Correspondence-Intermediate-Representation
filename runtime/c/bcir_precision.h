/*===- bcir_precision.h - Executable BCIR Precision Policy (P1-P6) Engine ---===
 * Pure ISO C11 precision policy, quantized/BF16 GEMM dispatch, deterministic
 * Ziggurat/mup_v1 weight initialization, and low-memory optimizers:
 *   - BCIR_OPT_FP32                 (16.00 B/p: FP32 AdamW reference)
 *   - BCIR_OPT_Q8_MOMENTS           (10.05 B/p: Q8 M1/M2 + FP32 grad)
 *   - BCIR_OPT_Q8_MOMENTS_BF16_GRAD ( 8.05 B/p: Q8 M1/M2 + BF16 grad)
 *   - BCIR_OPT_ADAFACTOR_M2         (12.01 B/p: FP32 M1 + factored M2 + FP32 grad)
 *   - BCIR_OPT_ADAFACTOR_BF16_GRAD  (10.01 B/p: FP32 M1 + factored M2 + BF16 grad)
 *   - BCIR_OPT_MUON_FP32            (12.00 B/p: 5-step NS Muon on 2D + AdamW on 1D/emb)
 *   - BCIR_OPT_Q8_MUON_BF16_GRAD    ( 7.05 B/p: Muon-8L/AdamW-32 + BF16 grad)
 *
 * All reduction and GEMM kernels keep deterministic partitions across thread
 * counts (0 bit drift between 1T and NT) and zero heap allocations during
 * bcir_opt_step / bcir_prec_mm_pool when workspace is pre-provisioned.
 * Python binding: bcir/hosted/models/mixer_ext.py.
 *===------------------------------------------------------------------------===*/
#ifndef BCIR_PRECISION_H
#define BCIR_PRECISION_H

#include "bcir_tensor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  BCIR_GEMM_FP32_KBLOCK    = 0,
  BCIR_GEMM_BF16_NDT       = 1,
  BCIR_GEMM_BF16_VDP       = 2,
  BCIR_GEMM_FP8_E4M3_BLK32 = 3,
  BCIR_GEMM_FP8_E5M2_BLK32 = 4,
  BCIR_GEMM_FP4_MX_BLK32   = 5,
  BCIR_GEMM_Q8_BLK32       = 6,
  BCIR_GEMM_KIND_COUNT     = 7
};

enum {
  BCIR_OPT_FP32                 = 0,
  BCIR_OPT_Q8_MOMENTS           = 1,
  BCIR_OPT_Q8_MOMENTS_BF16_GRAD = 2,
  BCIR_OPT_ADAFACTOR_M2         = 3,
  BCIR_OPT_ADAFACTOR_BF16_GRAD  = 4,
  BCIR_OPT_MUON_FP32            = 5,
  BCIR_OPT_Q8_MUON_BF16_GRAD    = 6,
  BCIR_OPT_KIND_COUNT           = 7
};

enum {
  BCIR_INIT_LEGACY_002      = 0,
  BCIR_INIT_RESIDUAL_SCALED = 1,
  BCIR_INIT_MUP_V1          = 2
};

enum {
  BCIR_FP64_GUARD_OFF         = 0,
  BCIR_FP64_GUARD_CONDITIONAL = 1,
  BCIR_FP64_GUARD_ALWAYS      = 2
};

/* Parameter roles for init, weight decay, mup_v1 LR scaling, and Muon 2D eligibility. */
enum {
  BCIR_ROLE_EMBED      = 0, /* embedding / lm_head: AdamW even under Muon */
  BCIR_ROLE_NORM       = 1, /* RMSNorm gamma: init 1.0, no WD, AdamW */
  BCIR_ROLE_LINEAR_IN  = 2, /* 2D hidden projection: mup_v1 + Muon 2D eligible */
  BCIR_ROLE_LINEAR_OUT = 3, /* 2D residual output (wo, wdown): resid-scaled + Muon 2D */
  BCIR_ROLE_ZERO_NOWD  = 4, /* zero-init bias / Canon conv1d tap: no WD, AdamW */
  BCIR_ROLE_CONST_NOWD = 5  /* const-init bias / mHC logits: no WD, AdamW */
};

typedef struct bcir_num_contract_desc {
  int kind;
  const char *name;
  const char *operand_fmt;
  const char *exponent_policy;
  const char *rounding_mode;
  const char *accumulator_type;
  const char *reduction_order;
  const char *subnormal_handling;
  const char *overflow_behavior;
  double max_rel_err_bound;
  int block_size;
  double bits_per_elem;
} bcir_num_contract_desc;

typedef struct bcir_prec_policy {
  int attn_fwd_gemm;
  int attn_bwd_gemm;
  int mlp_fwd_gemm;
  int mlp_bwd_gemm;
  int opt_kind;
  int fp64_guard_mode;
  double rmsnorm_cond_thresh;
  double softmax_logit_thresh;
  double wy_cond_thresh;
  int use_vdp_portable;
} bcir_prec_policy;

typedef struct bcir_prec_telemetry {
  uint64_t fp64_norm_promotions;
  uint64_t fp64_softmax_promotions;
  uint64_t fp64_wy_solves;
  uint64_t nonfinite_events;
  uint64_t grad_clip_hits;
  uint64_t total_steps;
  uint64_t q8_m2_floor_clamps;
  uint64_t gemm_calls_attn;
  uint64_t gemm_calls_mlp;
  uint64_t muon_ns_calls;
  double max_grad_norm;
  double max_softmax_logit;
  double min_rms_observed;
  double max_wy_offdiag;
  double max_abs_act;
  double first_tok_attn_mean;
} bcir_prec_telemetry;

/* Portable ISO C11 64-byte aligned zeroed allocator (Linux/macOS/Windows/ARM64). */
BCIR_DT_API void *bcir_aligned_calloc64(size_t count, size_t elem_size);
BCIR_DT_API void bcir_aligned_free64(void *ptr);

BCIR_DT_API const bcir_num_contract_desc *bcir_prec_gemm_desc(int gemm_kind);
BCIR_DT_API const char *bcir_prec_opt_name(int opt_kind);
BCIR_DT_API void bcir_prec_policy_default(bcir_prec_policy *pol);
BCIR_DT_API void bcir_prec_telemetry_reset(bcir_prec_telemetry *t);

BCIR_DT_API int bcir_mm_bf16_vdp_variants(void);
BCIR_DT_API float bcir_quant_fp8_e4m3_scalar(float x);
BCIR_DT_API float bcir_quant_fp8_e5m2_scalar(float x);
BCIR_DT_API float bcir_quant_fp4_e2m1_scalar(float x);
BCIR_DT_API void bcir_quant_dequant_blk32_pool(bcir_tensor_pool *pool, int gemm_kind,
                                               const float *src, float *dst, size_t n);

BCIR_DT_API void bcir_quant_q8_g32_encode(const float *src, int8_t *q_out, int8_t *exp_out, size_t n);
BCIR_DT_API void bcir_quant_q8_g32_decode(const int8_t *q_in, const int8_t *exp_in, float *dst, size_t n);

BCIR_DT_API int bcir_mm_bf16_vdp_variant(bcir_tensor_pool *pool, int variant,
                                         int ta, int tb, size_t m, size_t n, size_t k,
                                         float alpha, const float *a, const float *b,
                                         float beta, float *c, void *scratch, size_t scratch_bytes);
BCIR_DT_API size_t bcir_mm_bf16_vdp_scratch_bytes(size_t m, size_t n, size_t k);

BCIR_DT_API int bcir_prec_mm_pool(bcir_tensor_pool *pool, int gemm_kind, int force_portable,
                                  int ta, int tb, size_t m, size_t n, size_t k,
                                  float alpha, const float *a, const float *b,
                                  float beta, float *c, void *scratch, size_t scratch_bytes);
BCIR_DT_API size_t bcir_prec_mm_scratch_bytes(int gemm_kind, size_t m, size_t n, size_t k);

#define BCIR_OPT_Q8_BLOCK 256u

typedef struct bcir_opt_tensor_shape {
  size_t offset;
  size_t rows;
  size_t cols;
  double lr_mult;
  int apply_wd;
  int is_muon_2d; /* 1 for 2D hidden linear matrices eligible for Muon NS orthogonalization */
} bcir_opt_tensor_shape;

typedef struct bcir_opt_state {
  int opt_kind;
  size_t n_params;
  size_t n_blocks;
  uint64_t step;
  double beta1_power;
  double beta2_power;
  float *grad_fp32;
  uint16_t *grad_bf16;
  float *m1_fp32;
  float *m2_fp32;
  int8_t *m1_q8;
  float *m1_scale;
  uint8_t *m2_q8;
  float *m2_log_min;
  float *m2_log_step;
  size_t n_shapes;
  bcir_opt_tensor_shape *shapes;
  float **af_row;
  float **af_col;
  /* Muon / Q8-Muon per-tensor compact state + pre-allocated NS scratch (zero hot malloc). */
  float **muon_m1_fp32;
  int8_t **muon_m1_q8;
  float **muon_m1_scale;
  float **adam_m1_fp32;
  float **adam_m2_fp32;
  float *muon_ns_x;
  float *muon_ns_y;
  float *muon_ns_a;
  float *muon_ns_b;
  float *muon_ns_a2;
  size_t muon_ns_scratch_bytes;
  /* Pre-allocated reduction buffers for norm & clamps (zero hot malloc). */
  double *norm_partials;
  int *norm_bads;
  size_t norm_tasks_cap;
  uint64_t *q8_clamps_buf;
  size_t q8_tasks_cap;
  double *af_col_scratch;
  size_t af_col_cap;
  size_t persistent_bytes;
  size_t opt_only_bytes;
  size_t scratch_bytes;
  /* Upgrade 1 (opt-in, 0-defaulted): per-tensor AdaGC EMA clipping, FP32 Kahan
   * compensated rounding, and per-tensor cos(g_t, m_{t-1}) telemetry. */
  int adagc_enable;
  double adagc_beta;
  double adagc_lambda;
  double adagc_abs_floor;
  uint64_t adagc_warmup_steps;
  int tensor_tel_enabled;
  size_t kahan_active_tensors;
  size_t kahan_bytes;
  double *adagc_ema;
  double *tensor_grad_norm;
  double *tensor_adagc_scale;
  double *tensor_grad_mom_cos;
  uint64_t *tensor_clip_hits;
  uint8_t *kahan_mask;
  float **kahan_comp;
} bcir_opt_state;

typedef struct bcir_adamw_params {
  double lr, beta1, beta2, epsilon, weight_decay, grad_clip;
} bcir_adamw_params;

typedef struct bcir_adagc_config {
  int enable;            /* 0 = off (bit-identical default), 1 = per-tensor AdaGC EMA clip */
  double beta;           /* EMA smoothing in (0, 1), e.g., 0.98 (arXiv:2502.11034) */
  double lambda_mult;    /* spike threshold multiplier > 0, e.g., 1.05 */
  double abs_floor;      /* minimum absolute clip threshold >= 0 */
  uint64_t warmup_steps; /* steps before AdaGC clipping activates (0 -> default 1) */
} bcir_adagc_config;

typedef struct bcir_tensor_opt_telemetry {
  double last_grad_norm;
  double ema_grad_norm;
  double last_adagc_scale;
  double grad_mom_cos;   /* cos(g_t, m_{t-1}) before update; 0.0 if unavailable */
  double lr_mult;
  uint64_t adagc_clip_hits;
  int kahan_enabled;
} bcir_tensor_opt_telemetry;

BCIR_DT_API void bcir_adagc_config_default(bcir_adagc_config *cfg);
BCIR_DT_API int bcir_opt_configure_adagc(bcir_opt_state *opt, const bcir_adagc_config *cfg);
BCIR_DT_API int bcir_opt_enable_kahan_tensor(bcir_opt_state *opt, size_t shape_idx, int enable);
BCIR_DT_API int bcir_opt_set_tensor_lr_mult(bcir_opt_state *opt, size_t shape_idx, double lr_mult);
BCIR_DT_API int bcir_opt_enable_tensor_telemetry(bcir_opt_state *opt, int enable);
BCIR_DT_API int bcir_opt_get_tensor_telemetry(const bcir_opt_state *opt, size_t shape_idx,
                                              bcir_tensor_opt_telemetry *out);

BCIR_DT_API bcir_opt_state *bcir_opt_create(int opt_kind, size_t n_params,
                                            const bcir_opt_tensor_shape *shapes, size_t n_shapes);
BCIR_DT_API void bcir_opt_destroy(bcir_opt_state *opt);
BCIR_DT_API void bcir_opt_reset(bcir_opt_state *opt);
BCIR_DT_API void bcir_opt_zero_grad(bcir_opt_state *opt);
BCIR_DT_API void bcir_opt_commit_grad_fp32(bcir_opt_state *opt, const float *grad_work, int accumulate);
BCIR_DT_API void bcir_opt_commit_grad_span_fp32(bcir_opt_state *opt, size_t offset, size_t count,
                                                const float *grad_buf, int accumulate);
BCIR_DT_API int bcir_opt_step(bcir_tensor_pool *pool, bcir_opt_state *opt, float *weights,
                              const float *grad_work_or_null, const bcir_adamw_params *cfg,
                              double grad_scale, double *out_grad_norm, int *out_clipped,
                              bcir_prec_telemetry *tel);

/* Exact persistent optimizer state serialization (0-bit-drift checkpoint resume). */
BCIR_DT_API size_t bcir_opt_state_bytes(const bcir_opt_state *opt);
BCIR_DT_API int bcir_opt_save_state(const bcir_opt_state *opt, void *dst, size_t dst_bytes);
BCIR_DT_API int bcir_opt_load_state(bcir_opt_state *opt, const void *src, size_t src_bytes);

/* Gate 0 mathematical verifier for deterministic 5-step quintic Newton-Schulz polar factor. */
BCIR_DT_API int bcir_newton_schulz_verify(bcir_tensor_pool *pool, size_t rows, size_t cols,
                                          double *out_ortho_err);

typedef struct bcir_init_tensor_desc {
  size_t offset, rows, cols;
  int role;
  float const_val;
} bcir_init_tensor_desc;

BCIR_DT_API void bcir_init_weights_c11(float *weights, size_t n_params,
                                       const bcir_init_tensor_desc *descs, size_t n_descs,
                                       uint32_t n_layers, uint32_t d_model, uint32_t d_base,
                                       uint64_t seed, int init_policy,
                                       bcir_opt_tensor_shape *out_opt_shapes);

#ifdef __cplusplus
}
#endif
#endif
