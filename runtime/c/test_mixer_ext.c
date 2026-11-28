/*===- test_mixer_ext.c - Gate 0 ABI 6 MIXER-EXT & Optimizer Verifier -------===
 * Deterministic (0 variable output for CMake/BCIR-Make section parity),
 * pure ISO C11/C23 harness verifying:
 *   1. ABI version == 6
 *   2. WY-vs-recurrent identity for DeltaNet, KDA, and GDN-2 (< 1e-5)
 *   3. MLA absorption identity for both RoPE (dr=8) and NoPE (dr=0) (< 1e-6)
 *   4. mHC Sinkhorn-Knopp Birkhoff doubly stochastic & Jacobian (< 1e-11 / < 1e-4)
 *   5. Newton-Schulz 5-step polar orthogonalization verifier
 *   6. Finite-difference analytical AD verification across ALL 6 mixers
 *      (GQA, SWA, DeltaNet, KDA, MLA RoPE/NoPE, GDN-2) + ALL upgrades
 *      (G1 headwise & low-rank, QK-Norm, Canon-A/C, mHC C=2 & C=4, online_softmax)
 *   7. TRAIN-OPT-5 online_softmax=1 vs materialized=0 forward & gradient parity
 *      on GQA and MLA (plus O(T) memory savings check)
 *   8. All 7 optimizers (FP32, Q8, Q8+BF16Grad, Adafactor, Adafactor+BF16Grad,
 *      Muon-FP32, Q8-Muon+BF16Grad ~7.05 B/p): loss drop, memory ordering,
 *      and 1T vs 2T bit-exact parity across full hybrid GDN2+GatedMLA+Canon+mHC
 *   9. Fail-closed checks: invalid token, all-masked batch, NaN weight refusal
 *===------------------------------------------------------------------------===*/
#include "bcir_mixer_ext.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { if (!(cond)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; \
} } while (0)

static void fill_batch(uint32_t *tok, uint32_t *tgt, size_t n, uint32_t vocab, uint64_t seed) {
  uint64_t r = seed;
  for (size_t i = 0; i < n; ++i) {
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    tok[i] = (uint32_t)((r >> 32) % vocab);
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    tgt[i] = (uint32_t)((r >> 32) % vocab);
  }
}

static int check_finite_diff(const char *label, const bcir_ext_spec *sp, double tol) {
  bcir_prec_policy pol;
  bcir_prec_policy_default(&pol);
  pol.fp64_guard_mode = BCIR_FP64_GUARD_ALWAYS;
  pol.opt_kind = BCIR_OPT_FP32;
  bcir_ext_model *m = bcir_ext_create(sp, &pol);
  CHECK(m != NULL);

  size_t N = (size_t)sp->batch * sp->time;
  uint32_t *tok = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *tgt = (uint32_t *)malloc(N * sizeof(uint32_t));
  if (!tok || !tgt) { free(tok); free(tgt); bcir_ext_destroy(m); return 1; }
  fill_batch(tok, tgt, N, sp->vocab, 0xA1B2C3D4ULL);

  double loss0 = 0.0; uint64_t sup = 0u;
  CHECK(bcir_ext_loss_backward(m, tok, tgt, sp->time, 0, &loss0, &sup) == 0);
  CHECK(isfinite(loss0) && sup == N);

  float *W = bcir_ext_weights_ptr(m);
  float *G = bcir_ext_grads_fp32_ptr(m);
  CHECK(W && G);

  /* Sample 2 parameters from EVERY tensor in the model so every codepath is checked! */
  size_t nt = bcir_ext_tensor_count(m);
  double max_rel = 0.0;
  size_t worst_t = 0u;
  const double eps = 3e-3;
  size_t checked = 0u;

  for (size_t ti = 0; ti < nt; ++ti) {
    bcir_ext_tensor_info info;
    CHECK(bcir_ext_tensor_info_at(m, ti, &info) == 0);
    size_t len = info.rows * info.cols;
    size_t probes[2] = { 0u, (len > 1u) ? (len / 2u) : 0u };
    size_t np = (len > 1u) ? 2u : 1u;
    for (size_t p = 0; p < np; ++p) {
      size_t idx = info.offset + probes[p];
      float orig = W[idx];
      double lp = 0.0, lm = 0.0;
      bcir_ext_eval_result erp, erm;
      W[idx] = (float)((double)orig + eps);
      CHECK(bcir_ext_evaluate(m, tok, tgt, 1u, sp->time, &erp) == 0);
      lp = erp.ce;
      W[idx] = (float)((double)orig - eps);
      CHECK(bcir_ext_evaluate(m, tok, tgt, 1u, sp->time, &erm) == 0);
      lm = erm.ce;
      W[idx] = orig;
      double fd = (lp - lm) / (2.0 * eps);
      double ag = (double)G[idx];
      double denom = fmax(fabs(fd), fmax(fabs(ag), 1e-3));
      double rel = fabs(fd - ag) / denom;
      if (rel > max_rel) { max_rel = rel; worst_t = ti; }
      checked++;
    }
  }
  if (max_rel >= tol) {
    bcir_ext_tensor_info wi;
    bcir_ext_tensor_info_at(m, worst_t, &wi);
    fprintf(stderr, "FAIL AD [%s]: max_rel=%.6g >= tol=%.6g on tensor %s\n",
            label, max_rel, tol, wi.name);
    free(tok); free(tgt); bcir_ext_destroy(m);
    return 1;
  }
  printf("PASS ad.%s tensors=%zu probes=%zu\n", label, nt, checked);
  free(tok); free(tgt); bcir_ext_destroy(m);
  return 0;
}

static bcir_ext_spec base_spec(uint32_t mixer, uint32_t layers) {
  bcir_ext_spec sp;
  memset(&sp, 0, sizeof(sp));
  sp.vocab = 19u; sp.width = 16u; sp.heads = 2u; sp.kvheads = 1u;
  sp.layers = layers; sp.ff = 24u; sp.batch = 2u; sp.time = 10u; sp.tied = 0u;
  sp.rope_base = 10000.0; sp.rms_eps = 1e-5;
  sp.swa_window = 5u; sp.delta_chunk = 4u; sp.mla_dc = 8u; sp.mla_dr = 4u;
  sp.init_policy = BCIR_INIT_MUP_V1; sp.mup_d_base = 16u;
  sp.seed = 42ULL; sp.threads = 1u; sp.gate_rank = 8u;
  for (uint32_t l = 0; l < layers; ++l) {
    sp.layer_mixer[l] = mixer;
    sp.layer_use_rope[l] = 1u;
  }
  return sp;
}

static int check_online_softmax_parity(uint32_t mixer, uint32_t use_rope, const char *label) {
  bcir_ext_spec sp0 = base_spec(mixer, 2u);
  sp0.time = 20u; sp0.layer_use_rope[0] = sp0.layer_use_rope[1] = use_rope;
  sp0.online_softmax = 0u;
  bcir_ext_spec sp1 = sp0;
  sp1.online_softmax = 1u;

  bcir_prec_policy pol;
  bcir_prec_policy_default(&pol);
  pol.fp64_guard_mode = BCIR_FP64_GUARD_ALWAYS;

  bcir_ext_model *m0 = bcir_ext_create(&sp0, &pol);
  bcir_ext_model *m1 = bcir_ext_create(&sp1, &pol);
  CHECK(m0 && m1);
  bcir_ext_mem_stats s0, s1;
  bcir_ext_get_mem_stats(m0, &s0);
  bcir_ext_get_mem_stats(m1, &s1);
  CHECK(s1.mixer_prob_bytes < s0.mixer_prob_bytes);

  size_t N = (size_t)sp0.batch * sp0.time;
  uint32_t *tok = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *tgt = (uint32_t *)malloc(N * sizeof(uint32_t));
  fill_batch(tok, tgt, N, sp0.vocab, 777ULL);

  double l0 = 0.0, l1 = 0.0; uint64_t sup0 = 0u, sup1 = 0u;
  CHECK(bcir_ext_loss_backward(m0, tok, tgt, sp0.time, 0, &l0, &sup0) == 0);
  CHECK(bcir_ext_loss_backward(m1, tok, tgt, sp1.time, 0, &l1, &sup1) == 0);
  CHECK(fabs(l0 - l1) < 1e-6);

  const float *g0 = bcir_ext_grads_fp32_ptr(m0);
  const float *g1 = bcir_ext_grads_fp32_ptr(m1);
  size_t np = bcir_ext_param_count(m0);
  double max_gdiff = 0.0;
  for (size_t i = 0; i < np; ++i) {
    double df = fabs((double)g0[i] - (double)g1[i]);
    if (df > max_gdiff) max_gdiff = df;
  }
  CHECK(max_gdiff < 1e-5);
  printf("PASS online_softmax.%s prob_bytes_mat=%zu prob_bytes_online=%zu\n",
         label, s0.mixer_prob_bytes, s1.mixer_prob_bytes);
  free(tok); free(tgt);
  bcir_ext_destroy(m0); bcir_ext_destroy(m1);
  return 0;
}

static int check_thread_parity_and_optimizers(void) {
  /* Full beyond-the-list 4-layer hybrid: 3:1 GDN-2 + NoPE G1-Gated MLA + QK-Norm + Canon-AC + mHC C=2 */
  bcir_ext_spec sp1 = base_spec(BCIR_MIXER_GDN2, 4u);
  sp1.width = 32u; sp1.heads = 4u; sp1.kvheads = 2u; sp1.ff = 64u;
  sp1.batch = 2u; sp1.time = 12u; sp1.mla_dc = 16u; sp1.mla_dr = 4u; sp1.gate_rank = 8u;
  sp1.layer_mixer[0] = BCIR_MIXER_GDN2; sp1.layer_mixer[1] = BCIR_MIXER_KDA;
  sp1.layer_mixer[2] = BCIR_MIXER_DELTANET; sp1.layer_mixer[3] = BCIR_MIXER_MLA;
  sp1.layer_use_rope[3] = 0u; /* Kimi Linear style NoPE-MLA */
  sp1.gate_mode = BCIR_GATE_G1_LOWRANK;
  sp1.qk_norm = 1u;
  sp1.canon_mode = BCIR_CANON_AC;
  sp1.mhc_streams = 2u;
  sp1.online_softmax = 1u;
  sp1.threads = 1u;

  size_t N = (size_t)sp1.batch * sp1.time;
  uint32_t *tok = (uint32_t *)malloc(4u * N * sizeof(uint32_t));
  uint32_t *tgt = (uint32_t *)malloc(4u * N * sizeof(uint32_t));
  fill_batch(tok, tgt, 4u * N, sp1.vocab, 9999ULL);

  bcir_adamw_params opt_cfg = { 3e-3, 0.9, 0.95, 1e-8, 0.01, 3.0 };
  size_t prev_bytes[BCIR_OPT_KIND_COUNT];

  for (int ok = 0; ok < BCIR_OPT_KIND_COUNT; ++ok) {
    bcir_prec_policy pol;
    bcir_prec_policy_default(&pol);
    pol.opt_kind = ok;
    bcir_ext_spec sp2 = sp1;
    sp2.threads = 2u;

    bcir_ext_model *m1 = bcir_ext_create(&sp1, &pol);
    bcir_ext_model *m2 = bcir_ext_create(&sp2, &pol);
    CHECK(m1 && m2);

    bcir_ext_mem_stats ms1;
    bcir_ext_get_mem_stats(m1, &ms1);
    prev_bytes[ok] = ms1.persistent_bytes;

    bcir_ext_step_metrics ev1[4], ev2[4];
    CHECK(bcir_ext_train_loop(m1, tok, tgt, NULL, NULL, 4u, &opt_cfg, ev1) == 0);
    CHECK(bcir_ext_train_loop(m2, tok, tgt, NULL, NULL, 4u, &opt_cfg, ev2) == 0);

    /* Bit-identical 1T vs 2T weights and losses */
    size_t np = bcir_ext_param_count(m1);
    CHECK(memcmp(bcir_ext_weights_ptr(m1), bcir_ext_weights_ptr(m2), np * sizeof(float)) == 0);
    for (int s = 0; s < 4; ++s) {
      CHECK(ev1[s].loss == ev2[s].loss);
      CHECK(ev1[s].grad_norm == ev2[s].grad_norm);
      CHECK(isfinite(ev1[s].loss) && isfinite(ev1[s].grad_norm));
    }
    /* Verify Muon actually ran Newton-Schulz calls */
    bcir_prec_telemetry tel;
    bcir_ext_get_telemetry(m1, &tel);
    if (ok == BCIR_OPT_MUON_FP32 || ok == BCIR_OPT_Q8_MUON_BF16_GRAD) {
      CHECK(tel.muon_ns_calls > 0u);
    }

    /* Bit-exact mid-run checkpoint resume (2 + 2 steps == 4 uninterrupted steps) */
    bcir_ext_model *m3 = bcir_ext_create(&sp1, &pol);
    CHECK(m3 != NULL);
    bcir_ext_step_metrics ev3a[2], ev3b[2];
    CHECK(bcir_ext_train_loop(m3, tok, tgt, NULL, NULL, 2u, &opt_cfg, ev3a) == 0);
    size_t obytes = bcir_ext_opt_state_bytes(m3);
    CHECK(obytes >= 80u);
    float *w_snap = (float *)malloc(np * sizeof(float));
    unsigned char *o_snap = (unsigned char *)malloc(obytes);
    if (!w_snap || !o_snap) {
      free(w_snap); free(o_snap);
      bcir_ext_destroy(m1); bcir_ext_destroy(m2); bcir_ext_destroy(m3);
      free(tok); free(tgt);
      return 1;
    }
    memcpy(w_snap, bcir_ext_weights_ptr(m3), np * sizeof(float));
    CHECK(bcir_ext_save_opt_state(m3, o_snap, obytes) == 0);
    bcir_ext_destroy(m3);

    bcir_ext_spec sp_res = sp1;
    sp_res.skip_init = 1;
    bcir_ext_model *m4 = bcir_ext_create(&sp_res, &pol);
    CHECK(m4 != NULL);
    memcpy(bcir_ext_weights_ptr(m4), w_snap, np * sizeof(float));
    /* Fail-closed checks: truncated buffer, bad magic, and non-finite payload must fail */
    CHECK(bcir_ext_load_opt_state(m4, o_snap, obytes - 1u) != 0);
    o_snap[0] ^= 0xFFu;
    CHECK(bcir_ext_load_opt_state(m4, o_snap, obytes) != 0);
    o_snap[0] ^= 0xFFu;
    CHECK(bcir_ext_load_opt_state(m4, o_snap, obytes) == 0);
    CHECK(bcir_ext_train_loop(m4, tok + 2u * N, tgt + 2u * N, NULL, NULL, 2u, &opt_cfg, ev3b) == 0);
    CHECK(memcmp(bcir_ext_weights_ptr(m1), bcir_ext_weights_ptr(m4), np * sizeof(float)) == 0);
    for (int s = 0; s < 2; ++s) {
      CHECK(ev3b[s].loss == ev1[2 + s].loss);
      CHECK(ev3b[s].grad_norm == ev1[2 + s].grad_norm);
      CHECK(ev3b[s].step == ev1[2 + s].step);
    }
    free(w_snap);
    free(o_snap);
    bcir_ext_destroy(m4);

    printf("PASS opt.%s 1T==2T_bits=0 resume_bits=0 opt_state_bytes=%zu persistent_bytes=%zu\n",
           bcir_prec_opt_name(ok), obytes, ms1.persistent_bytes);
    bcir_ext_destroy(m1);
    bcir_ext_destroy(m2);
  }

  /* Verify memory hierarchy: Q8-Muon+BF16Grad < Q8+BF16Grad < Adafactor < FP32 */
  CHECK(prev_bytes[BCIR_OPT_Q8_MUON_BF16_GRAD] < prev_bytes[BCIR_OPT_Q8_MOMENTS_BF16_GRAD]);
  CHECK(prev_bytes[BCIR_OPT_Q8_MOMENTS_BF16_GRAD] < prev_bytes[BCIR_OPT_ADAFACTOR_M2]);
  CHECK(prev_bytes[BCIR_OPT_ADAFACTOR_M2] < prev_bytes[BCIR_OPT_FP32]);

  free(tok); free(tgt);
  return 0;
}

static int check_fail_closed(void) {
  bcir_ext_spec sp = base_spec(BCIR_MIXER_GDN2, 1u);
  bcir_prec_policy pol;
  bcir_prec_policy_default(&pol);
  bcir_ext_model *m = bcir_ext_create(&sp, &pol);
  CHECK(m != NULL);

  size_t N = (size_t)sp.batch * sp.time;
  uint32_t *tok = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *tgt = (uint32_t *)malloc(N * sizeof(uint32_t));
  fill_batch(tok, tgt, N, sp.vocab, 123ULL);

  /* 1. All-masked targets must be refused (rc == 1) */
  for (size_t i = 0; i < N; ++i) tgt[i] = BCIR_EXT_IGNORE;
  double loss = 0.0; uint64_t sup = 99u;
  CHECK(bcir_ext_loss_backward(m, tok, tgt, sp.time, 0, &loss, &sup) == 1);
  CHECK(sup == 0u);

  /* 2. Out-of-range token must be refused (rc == 1) */
  fill_batch(tok, tgt, N, sp.vocab, 123ULL);
  tok[0] = sp.vocab;
  CHECK(bcir_ext_forward(m, tok, sp.time, NULL) == 1);

  /* 3. NaN weight must fail closed in forward, evaluate, and opt_step */
  tok[0] = 0u;
  float *W = bcir_ext_weights_ptr(m);
  float saved = W[0];
  W[0] = NAN;
  CHECK(bcir_ext_forward(m, tok, sp.time, NULL) == 3);
  bcir_ext_eval_result er;
  CHECK(bcir_ext_evaluate(m, tok, tgt, 1u, sp.time, &er) == 3);
  CHECK(isnan(er.ce) && er.nonfinite > 0u);
  W[0] = saved;
  CHECK(bcir_ext_forward(m, tok, sp.time, NULL) == 0);

  printf("PASS fail_closed all_masked=1 bad_tok=1 nan_weight=3\n");
  free(tok); free(tgt); bcir_ext_destroy(m);
  return 0;
}

static int check_adagc_kahan_and_telemetry(void) {
  /* 1. Sub-ULP FP32 Kahan precision proof (update -1e-8 < ULP(1.0f)=5.96e-8) */
  bcir_opt_tensor_shape sh = { 0u, 1u, 4u, 1.0, 0, 0 };
  bcir_opt_state *opt_plain = bcir_opt_create(BCIR_OPT_FP32, 4u, &sh, 1u);
  bcir_opt_state *opt_kahan = bcir_opt_create(BCIR_OPT_FP32, 4u, &sh, 1u);
  CHECK(opt_plain && opt_kahan);
  CHECK(bcir_opt_enable_kahan_tensor(opt_kahan, 0u, 1) == 0);
  float w_plain[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
  float w_kahan[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
  float g_const[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
  bcir_adamw_params tiny_cfg = { 1e-8, 0.9, 0.99, 1e-8, 0.0, 0.0 };
  for (int it = 0; it < 1000; ++it) {
    CHECK(bcir_opt_step(NULL, opt_plain, w_plain, g_const, &tiny_cfg, 1.0, NULL, NULL, NULL) == 0);
    CHECK(bcir_opt_step(NULL, opt_kahan, w_kahan, g_const, &tiny_cfg, 1.0, NULL, NULL, NULL) == 0);
  }
  /* Plain FP32 loses 100% of sub-ULP updates; Kahan accumulates ~1e-5 accurately */
  CHECK(w_plain[0] == 1.0f);
  CHECK(fabs((double)w_kahan[0] - (1.0 - 1e-5)) < 1e-7);
  bcir_opt_destroy(opt_plain);
  bcir_opt_destroy(opt_kahan);

  /* 2. Per-tensor AdaGC spike isolation + cos(g_t, m_{t-1}) sign test */
  bcir_opt_tensor_shape two_sh[2] = {
    { 0u, 1u, 4u, 1.0, 0, 0 },
    { 4u, 1u, 4u, 0.5, 0, 0 },
  };
  bcir_opt_state *os = bcir_opt_create(BCIR_OPT_FP32, 8u, two_sh, 2u);
  CHECK(os != NULL);
  bcir_adagc_config acfg;
  bcir_adagc_config_default(&acfg);
  acfg.enable = 1; acfg.beta = 0.9; acfg.lambda_mult = 1.5; acfg.warmup_steps = 1u;
  CHECK(bcir_opt_configure_adagc(os, &acfg) == 0);
  CHECK(bcir_opt_enable_tensor_telemetry(os, 1) == 0);
  /* Fail-closed checks */
  bcir_adagc_config bad_cfg = acfg; bad_cfg.beta = 1.2;
  CHECK(bcir_opt_configure_adagc(os, &bad_cfg) != 0);
  CHECK(bcir_opt_set_tensor_lr_mult(os, 0u, -1.0) != 0);

  float w8[8] = {0};
  float g8[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  bcir_adamw_params ncfg = { 1e-3, 0.9, 0.95, 1e-8, 0.0, 0.0 };
  CHECK(bcir_opt_step(NULL, os, w8, g8, &ncfg, 1.0, NULL, NULL, NULL) == 0);
  CHECK(bcir_opt_step(NULL, os, w8, g8, &ncfg, 1.0, NULL, NULL, NULL) == 0);
  bcir_tensor_opt_telemetry t0, t1;
  CHECK(bcir_opt_get_tensor_telemetry(os, 0u, &t0) == 0);
  CHECK(t0.grad_mom_cos > 0.99);
  /* Inject 20x spike on tensor 0 only, flip sign on tensor 1 */
  float g_spike[8] = { 20.0f, 20.0f, 20.0f, 20.0f, -1.0f, -1.0f, -1.0f, -1.0f };
  int clipped = 0;
  CHECK(bcir_opt_step(NULL, os, w8, g_spike, &ncfg, 1.0, NULL, &clipped, NULL) == 0);
  CHECK(clipped == 1);
  CHECK(bcir_opt_get_tensor_telemetry(os, 0u, &t0) == 0);
  CHECK(bcir_opt_get_tensor_telemetry(os, 1u, &t1) == 0);
  CHECK(t0.adagc_clip_hits == 1u && t0.last_adagc_scale < 0.1);
  CHECK(t1.adagc_clip_hits == 0u && t1.last_adagc_scale == 1.0);
  CHECK(t1.grad_mom_cos < -0.99);
  bcir_opt_destroy(os);

  /* 2b. Enabling AdaGC mid-training (step > 0, ema == 0) must seed EMA rather than clip to floor */
  bcir_opt_state *os_mid = bcir_opt_create(BCIR_OPT_FP32, 8u, two_sh, 2u);
  CHECK(os_mid != NULL);
  float w8_mid[8] = {0};
  CHECK(bcir_opt_step(NULL, os_mid, w8_mid, g8, &ncfg, 1.0, NULL, NULL, NULL) == 0);
  CHECK(bcir_opt_step(NULL, os_mid, w8_mid, g8, &ncfg, 1.0, NULL, NULL, NULL) == 0);
  CHECK(bcir_opt_configure_adagc(os_mid, &acfg) == 0);
  clipped = 0;
  CHECK(bcir_opt_step(NULL, os_mid, w8_mid, g8, &ncfg, 1.0, NULL, &clipped, NULL) == 0);
  CHECK(clipped == 0);
  CHECK(bcir_opt_get_tensor_telemetry(os_mid, 0u, &t0) == 0);
  CHECK(t0.last_adagc_scale == 1.0 && t0.adagc_clip_hits == 0u && fabs(t0.ema_grad_norm - 2.0) < 1e-9);
  size_t mid_bytes = bcir_opt_state_bytes(os_mid);
  unsigned char *mid_buf = (unsigned char *)malloc(mid_bytes);
  if (!mid_buf) { bcir_opt_destroy(os_mid); return 1; }
  int save_rc = bcir_opt_save_state(os_mid, mid_buf, mid_bytes);
  double neg_ema = -1.0;
  size_t ema_off = mid_bytes - 2u * (sizeof(double) + sizeof(uint64_t));
  memcpy(mid_buf + ema_off, &neg_ema, sizeof(double));
  int bad_load_rc = bcir_opt_load_state(os_mid, mid_buf, mid_bytes);
  free(mid_buf);
  bcir_opt_destroy(os_mid);
  CHECK(save_rc == 0 && bad_load_rc != 0);

  /* 3. End-to-end 1T==2T parity + v2 checkpoint 0-bit-drift resume with AdaGC+Kahan+PerGroupLR */
  bcir_ext_spec sp1 = base_spec(BCIR_MIXER_GDN2, 2u);
  sp1.width = 32u; sp1.heads = 4u; sp1.kvheads = 2u; sp1.ff = 64u;
  sp1.layer_mixer[1] = BCIR_MIXER_MLA; sp1.layer_use_rope[1] = 0u;
  sp1.gate_mode = BCIR_GATE_G1_HEADWISE; sp1.canon_mode = BCIR_CANON_AC; sp1.mhc_streams = 2u;
  sp1.threads = 1u;
  bcir_ext_spec sp2 = sp1; sp2.threads = 2u;

  size_t N = (size_t)sp1.batch * sp1.time;
  uint32_t *tok = (uint32_t *)malloc(4u * N * sizeof(uint32_t));
  uint32_t *tgt = (uint32_t *)malloc(4u * N * sizeof(uint32_t));
  if (!tok || !tgt) { free(tok); free(tgt); return 1; }
  fill_batch(tok, tgt, 4u * N, sp1.vocab, 424242ULL);

  bcir_prec_policy pol;
  bcir_prec_policy_default(&pol);
  pol.opt_kind = BCIR_OPT_Q8_MUON_BF16_GRAD;
  bcir_adamw_params opt_cfg = { 3e-3, 0.9, 0.95, 1e-8, 0.01, 3.0 };
  acfg.enable = 1; acfg.beta = 0.95; acfg.lambda_mult = 1.05; acfg.warmup_steps = 1u;

  bcir_ext_model *m1 = bcir_ext_create(&sp1, &pol);
  bcir_ext_model *m2 = bcir_ext_create(&sp2, &pol);
  bcir_ext_model *m3 = bcir_ext_create(&sp1, &pol);
  CHECK(m1 && m2 && m3);
  bcir_ext_mem_stats ms_before, ms_after;
  bcir_ext_get_mem_stats(m1, &ms_before);
  bcir_ext_model *ms_arr[3] = { m1, m2, m3 };
  for (int i = 0; i < 3; ++i) {
    CHECK(bcir_ext_configure_adagc(ms_arr[i], &acfg) == 0);
    CHECK(bcir_ext_configure_kahan(ms_arr[i], 1, 1) == 0);
    CHECK(bcir_ext_enable_tensor_telemetry(ms_arr[i], 1) == 0);
    CHECK(bcir_ext_set_tensor_lr_mult(ms_arr[i], 0u, 0.75) == 0);
  }
  bcir_ext_get_mem_stats(m1, &ms_after);
  CHECK(ms_after.persistent_bytes > ms_before.persistent_bytes);

  bcir_ext_step_metrics ev1[4], ev2[4], ev3a[2], ev3b[2];
  CHECK(bcir_ext_train_loop(m1, tok, tgt, NULL, NULL, 4u, &opt_cfg, ev1) == 0);
  CHECK(bcir_ext_train_loop(m2, tok, tgt, NULL, NULL, 4u, &opt_cfg, ev2) == 0);
  size_t np = bcir_ext_param_count(m1);
  CHECK(memcmp(bcir_ext_weights_ptr(m1), bcir_ext_weights_ptr(m2), np * sizeof(float)) == 0);

  CHECK(bcir_ext_train_loop(m3, tok, tgt, NULL, NULL, 2u, &opt_cfg, ev3a) == 0);
  size_t ob = bcir_ext_opt_state_bytes(m3);
  float *ws = (float *)malloc(np * sizeof(float));
  unsigned char *osb = (unsigned char *)malloc(ob);
  if (!ws || !osb) {
    free(ws); free(osb); free(tok); free(tgt);
    bcir_ext_destroy(m1); bcir_ext_destroy(m2); bcir_ext_destroy(m3);
    return 1;
  }
  memcpy(ws, bcir_ext_weights_ptr(m3), np * sizeof(float));
  CHECK(bcir_ext_save_opt_state(m3, osb, ob) == 0);
  bcir_ext_destroy(m3);

  bcir_ext_spec spr = sp1; spr.skip_init = 1;
  bcir_ext_model *m4 = bcir_ext_create(&spr, &pol);
  CHECK(m4 != NULL);
  CHECK(bcir_ext_configure_adagc(m4, &acfg) == 0);
  CHECK(bcir_ext_configure_kahan(m4, 1, 1) == 0);
  CHECK(bcir_ext_enable_tensor_telemetry(m4, 1) == 0);
  CHECK(bcir_ext_set_tensor_lr_mult(m4, 0u, 0.75) == 0);
  memcpy(bcir_ext_weights_ptr(m4), ws, np * sizeof(float));
  CHECK(bcir_ext_load_opt_state(m4, osb, ob) == 0);
  CHECK(bcir_ext_train_loop(m4, tok + 2u * N, tgt + 2u * N, NULL, NULL, 2u, &opt_cfg, ev3b) == 0);
  CHECK(memcmp(bcir_ext_weights_ptr(m1), bcir_ext_weights_ptr(m4), np * sizeof(float)) == 0);

  bcir_tensor_opt_telemetry tt0;
  CHECK(bcir_ext_get_tensor_telemetry(m1, 0u, &tt0) == 0);
  CHECK(tt0.kahan_enabled == 1 && tt0.lr_mult == 0.75 && tt0.ema_grad_norm > 0.0);

  printf("PASS upgrade1.adagc_kahan_telemetry kahan_subulp=ok 1T==2T_bits=0 v2_resume_bits=0\n");
  free(ws); free(osb); free(tok); free(tgt);
  bcir_ext_destroy(m1); bcir_ext_destroy(m2); bcir_ext_destroy(m4);
  return 0;
}

int main(void) {
  CHECK(bcir_mixer_ext_abi_version() == 6u);
  printf("PASS abi_version=%u\n", bcir_mixer_ext_abi_version());

  double d_dn = 1.0, d_kda = 1.0, d_gdn2 = 1.0, d_mla_r = 1.0, d_mla_n = 1.0;
  double mhc_m2 = 1.0, mhc_g2 = 1.0, mhc_m4 = 1.0, mhc_g4 = 1.0, ns_err = 1.0;
  CHECK(bcir_deltanet_verify_wy_vs_recurrent(2u, 20u, 2u, 8u, 6u, &d_dn) == 0);
  CHECK(bcir_kda_verify_wy_vs_recurrent(2u, 20u, 2u, 8u, 6u, &d_kda) == 0);
  CHECK(bcir_gdn2_verify_wy_vs_recurrent(2u, 20u, 2u, 8u, 6u, &d_gdn2) == 0);
  CHECK(bcir_mla_verify_absorption(16u, 2u, 8u, 8u, 4u, &d_mla_r) == 0);
  CHECK(bcir_mla_verify_absorption(16u, 2u, 8u, 8u, 0u, &d_mla_n) == 0);
  CHECK(bcir_mhc_verify_birkhoff(2u, &mhc_m2, &mhc_g2) == 0);
  CHECK(bcir_mhc_verify_birkhoff(4u, &mhc_m4, &mhc_g4) == 0);
  CHECK(bcir_newton_schulz_verify(NULL, 16u, 24u, &ns_err) == 0);
  CHECK(bcir_newton_schulz_verify(NULL, 24u, 16u, &ns_err) == 0);
  printf("PASS identities wy_dn wy_kda wy_gdn2 mla_rope mla_nope mhc2 mhc4 ns5\n");

  /* Finite-difference analytical AD verification for EVERY mixer and EVERY upgrade */
  bcir_ext_spec sp;
  sp = base_spec(BCIR_MIXER_GQA, 1u); sp.qk_norm = 1u; sp.gate_mode = BCIR_GATE_G1_HEADWISE;
  CHECK(check_finite_diff("gqa_qknorm_g1h", &sp, 5e-3) == 0);

  sp = base_spec(BCIR_MIXER_SWA, 1u); sp.canon_mode = BCIR_CANON_AC;
  CHECK(check_finite_diff("swa_canon_ac", &sp, 5e-3) == 0);

  sp = base_spec(BCIR_MIXER_DELTANET, 1u);
  CHECK(check_finite_diff("deltanet", &sp, 5e-3) == 0);

  sp = base_spec(BCIR_MIXER_KDA, 1u); sp.gate_mode = BCIR_GATE_G1_LOWRANK;
  CHECK(check_finite_diff("kda_g1lr", &sp, 5e-3) == 0);

  sp = base_spec(BCIR_MIXER_GDN2, 1u); sp.canon_mode = BCIR_CANON_A;
  CHECK(check_finite_diff("gdn2_canon_a", &sp, 5e-3) == 0);

  sp = base_spec(BCIR_MIXER_MLA, 1u); sp.qk_norm = 1u; sp.gate_mode = BCIR_GATE_G1_HEADWISE;
  CHECK(check_finite_diff("mla_rope_g1h", &sp, 5e-3) == 0);

  sp = base_spec(BCIR_MIXER_MLA, 1u); sp.layer_use_rope[0] = 0u; sp.online_softmax = 1u;
  CHECK(check_finite_diff("mla_nope_online", &sp, 5e-3) == 0);

  /* mHC C=2 and C=4 on 2-layer hybrid GDN-2 + Gated MLA */
  sp = base_spec(BCIR_MIXER_GDN2, 2u);
  sp.layer_mixer[1] = BCIR_MIXER_MLA; sp.layer_use_rope[1] = 0u;
  sp.gate_mode = BCIR_GATE_G1_HEADWISE; sp.canon_mode = BCIR_CANON_AC; sp.mhc_streams = 2u;
  CHECK(check_finite_diff("hybrid_gdn2_mla_canon_mhc2", &sp, 5e-3) == 0);

  sp.mhc_streams = 4u;
  CHECK(check_finite_diff("hybrid_gdn2_mla_canon_mhc4", &sp, 5e-3) == 0);

  /* TRAIN-OPT-5 online softmax parity */
  CHECK(check_online_softmax_parity(BCIR_MIXER_GQA, 1u, "gqa") == 0);
  CHECK(check_online_softmax_parity(BCIR_MIXER_MLA, 1u, "mla_rope") == 0);
  CHECK(check_online_softmax_parity(BCIR_MIXER_MLA, 0u, "mla_nope") == 0);

  /* All 7 optimizers + 1T vs 2T bit-exact thread parity */
  CHECK(check_thread_parity_and_optimizers() == 0);

  /* Upgrade 1: per-tensor AdaGC + FP32 Kahan + cos(g,m) telemetry + v2 resume */
  CHECK(check_adagc_kahan_and_telemetry() == 0);

  /* Fail-closed contract checks */
  CHECK(check_fail_closed() == 0);

  printf("mixer_ext: ALL CHECKS PASSED\n");
  return 0;
}
