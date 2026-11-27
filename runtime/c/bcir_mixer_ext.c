/*===- bcir_mixer_ext.c - Production Pluggable Sequence-Mixer ABI (ABI 6) ---===
 * Pure ISO C11 implementation (zero POSIX-only APIs, zero hot-path mallocs,
 * -ffp-contract=off, 0-bit-drift across 1T..NT):
 *   - Mixers: GQA, SWA, Arch-1 DeltaNet (FP64 WY), KDA (arXiv:2510.26692),
 *     Arch-4 MLA (RoPE or NoPE + absorbed decode), GDN-2 (arXiv:2605.22791v2).
 *   - Upgrades: G1 Gated Attention (arXiv:2505.06708), QK-Norm, Canon-A/C
 *     (arXiv:2512.17351v2), DeepSeek mHC Birkhoff streams (arXiv:2512.24880v2),
 *     TRAIN-OPT-5 online-softmax O(T) tiled attention for GQA & MLA.
 *===------------------------------------------------------------------------===*/
#include "bcir_mixer_ext.h"
#include "bcir_host_alloc.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *ext_zalloc(size_t count, size_t elem) {
  size_t bytes = 0u;
  if (!bcir_size_mul(count, elem, &bytes)) return NULL;
  bcir_host_allocator a = bcir_host_allocator_default();
  return bcir_host_allocate_zeroed(&a, bytes);
}

static void ext_dealloc(void *ptr) {
  bcir_host_allocator a = bcir_host_allocator_default();
  bcir_host_deallocate(&a, ptr);
}

uint32_t bcir_mixer_ext_abi_version(void) { return 6u; }

static double now_ms(void) {
  struct timespec ts;
  if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static inline float sigmoid_f(float x) {
  if (x >= 0.0f) { float e = expf(-x); return 1.0f / (1.0f + e); }
  float e = expf(x); return e / (1.0f + e);
}

static inline double sigmoid_d(double x) {
  if (x >= 0.0) { double e = exp(-x); return 1.0 / (1.0 + e); }
  double e = exp(x); return e / (1.0 + e);
}

static size_t max_sz(size_t a, size_t b) { return a > b ? a : b; }

typedef struct layer_offsets {
  uint32_t mixer;
  uint32_t use_rope;
  size_t norm1, norm2;
  size_t canon_a, canon_c;
  size_t q_norm, k_norm, o_norm;
  size_t wq, wk, wv, wo;
  /* DeltaNet */
  size_t walpha, balpha, wbeta, bbeta;
  /* KDA / GDN-2 low-rank channel-wise gates */
  size_t wa1, wa2, ba_chan;
  size_t we1, we2, be_chan;
  size_t ww1, ww2, bw_chan;
  /* MLA */
  size_t wq_c, wq_r, w_dkv, norm_kv, w_uk, w_uv, w_kr;
  /* G1 output gate */
  size_t wg1, wg2, bg;
  /* mHC (2 sublayers: mixer & MLP) */
  size_t mhc_hres1, mhc_hpre1, mhc_hpost1;
  size_t mhc_hres2, mhc_hpre2, mhc_hpost2;
  /* MLP */
  size_t wgate, wup, wdown;
} layer_offsets;

typedef struct layer_act {
  float *mhc_x1, *mhc_x2, *mix_out, *mlp_out;
  float *x_in;     /* [N*d] */
  float *n1_raw;   /* [N*d] pre-Canon norm1 output (aliased to n1 if canon==0) */
  float *n1;       /* [N*d] */
  /* GQA / SWA */
  float *q_raw, *k_raw, *q, *k, *v, *p;
  double *lse;     /* [B*H*T] when online_softmax==1 */
  /* DeltaNet / KDA / GDN-2 */
  float *q_inv, *k_inv;
  float *a_mid, *a_pre, *alpha;
  float *b_pre, *beta;
  float *e_mid, *e_pre, *erase;
  float *w_mid, *w_pre, *write;
  float *cd_y;     /* [N*d] pre-o_norm recurrent output for KDA/GDN2 */
  double *dn_sprev; /* [B*H*nc*dh*dh] */
  double *dn_W;     /* [B*H*T*dh] */
  /* MLA */
  float *q_c_raw, *q_c, *q_r, *c_raw, *c_hat, *k_c, *v_c, *k_r, *mla_p;
  /* G1 output gate */
  float *ctx_raw, *g_mid, *g_val;
  /* Shared post-mixer & MLP */
  float *ctx;      /* [N*d] */
  float *x2;       /* [N*d] */
  float *n2_raw;   /* [N*d] pre-Canon norm2 output (aliased to n2 if canon<2) */
  float *n2;       /* [N*d] */
  float *gate, *up, *hidden;
} layer_act;

struct bcir_ext_model {
  bcir_ext_spec sp;
  bcir_prec_policy pol;
  bcir_prec_telemetry tel;
  bcir_ext_mem_stats mem;

  size_t gate_rank;
  size_t mhc_c;

  void *pool_mem;
  bcir_tensor_pool *pool;

  size_t n_params;
  float *weights;
  float *grad_tile;
  size_t grad_tile_bytes;
  bcir_opt_state *opt;

  size_t n_tensors;
  bcir_ext_tensor_info *tensors;
  layer_offsets loff[BCIR_EXT_MAX_LAYERS];
  size_t off_embed, off_final_norm, off_lm_head;

  layer_act acts[BCIR_EXT_MAX_LAYERS];
  float *mhc_xfinal;
  float *x_final, *norm_final, *logits;

  float *rope_hd, *rope_dr;

  /* Backward scratch buffers (allocated once in create; zero hot malloc). */
  float *d_mhc_cur, *d_mhc_next;
  float *d_cur, *d_next, *d_n, *d_n_tmp, *d_ctx, *d_gate, *d_up, *d_hid, *d_logits, *rms_inv;
  float *d_q, *d_k, *d_v, *d_a, *d_b, *d_e, *d_w, *d_mid, *d_gpre;
  float *d_qc, *d_qr, *d_chat, *d_craw, *d_kc, *d_vc, *d_kr;

  /* Pre-allocated per-worker / reduction scratch (zero hot malloc). */
  double *rms_dg64;       /* [max(d, dc)] */
  double *attn_max_log;   /* [B*H] */
  uint64_t *attn_promos;  /* [B*H] */
  double *attn_sink_sum;  /* [B*H] */
  double *attn_scores;    /* [B*H*max(T,W)] */
  double *attn_dk_loc;    /* [B*max(H,KH)*T*dh] */
  double *attn_dv_loc;    /* [B*max(H,KH)*T*dh] */
  double *attn_dkr_loc;   /* [B*T*dr] */
  double *dn_work_S;      /* [B*H*dh*dh] */
  double *dn_work_dS;     /* [B*H*dh*dh] */
  double *dn_work_g;      /* [B*H*C*dh] */
  double *dn_work_gam;    /* [B*H*C*dh] */
  double *dn_work_A;      /* [B*H*C*C] */
  double *dn_work_qS;     /* [B*H*dh] */
  double *dn_work_Shist;  /* [B*H*(C+1)*dh*dh] for KDA/GDN-2 exact chunk replay */
  double *dn_work_dP;     /* [B*H*dh*dh] */
  double *dn_work_dW;     /* [B*H*C*dh] */
  double *dn_work_dA;     /* [B*H*C*C] */
  double *dn_work_dg;     /* [B*H*C] */

  void *gemm_scratch;
  size_t gemm_scratch_bytes;
};

static int add_tensor(bcir_ext_model *m, bcir_init_tensor_desc **idescs_p, size_t *cap,
                      const char *name, int layer, int region, size_t rows, size_t cols,
                      int init_role, float const_val, size_t *out_off) {
  size_t off = m->n_params;
  size_t idx = m->n_tensors;
  if (idx >= *cap) {
    size_t old_cap = *cap;
    size_t ncap = (old_cap == 0u) ? 64u : (old_cap * 2u);
    bcir_host_allocator a = bcir_host_allocator_default();
    void *tptr = m->tensors;
    void *iptr = *idescs_p;
    if (!bcir_host_realloc_array(&a, &tptr, old_cap, ncap, sizeof(*m->tensors), 1)) return 0;
    m->tensors = (bcir_ext_tensor_info *)tptr;
    if (!bcir_host_realloc_array(&a, &iptr, old_cap, ncap, sizeof(**idescs_p), 1)) return 0;
    *idescs_p = (bcir_init_tensor_desc *)iptr;
    *cap = ncap;
  }
  bcir_ext_tensor_info *ti = &m->tensors[idx];
  ti->offset = off; ti->rows = rows; ti->cols = cols; ti->layer = layer; ti->region = region;
  snprintf(ti->name, sizeof(ti->name), "%s", name);
  (*idescs_p)[idx].offset = off; (*idescs_p)[idx].rows = rows; (*idescs_p)[idx].cols = cols;
  (*idescs_p)[idx].role = init_role; (*idescs_p)[idx].const_val = const_val;
  m->n_tensors++;
  m->n_params += rows * cols;
  if (region == 1) m->mem.attn_params += rows * cols;
  else if (region == 2) m->mem.mlp_params += rows * cols;
  else m->mem.nonlin_params += rows * cols;
  if (out_off) *out_off = off;
  return 1;
}

/* ==================== DeepSeek mHC Sinkhorn-Knopp Birkhoff Projection ==================== */
#define MHC_SK_ITERS 20

static void mhc_sinkhorn_fwd(size_t C, const float *raw, double H[BCIR_EXT_MHC_MAX_C * BCIR_EXT_MHC_MAX_C]) {
  memset(H, 0, sizeof(double) * BCIR_EXT_MHC_MAX_C * BCIR_EXT_MHC_MAX_C);
  if (C == 0u || C > BCIR_EXT_MHC_MAX_C) return;
  double mx = -1e30;
  for (size_t i = 0; i < C * C; ++i) {
    double v = (double)raw[i];
    if (v > mx) mx = v;
  }
  for (size_t i = 0; i < C * C; ++i) H[i] = exp((double)raw[i] - mx);
  for (int it = 0; it < MHC_SK_ITERS; ++it) {
    for (size_t r = 0; r < C; ++r) {
      double s = 0.0;
      for (size_t c = 0; c < C; ++c) s += H[r * C + c];
      double inv = 1.0 / (s > 1e-30 ? s : 1e-30);
      for (size_t c = 0; c < C; ++c) H[r * C + c] *= inv;
    }
    for (size_t c = 0; c < C; ++c) {
      double s = 0.0;
      for (size_t r = 0; r < C; ++r) s += H[r * C + c];
      double inv = 1.0 / (s > 1e-30 ? s : 1e-30);
      for (size_t r = 0; r < C; ++r) H[r * C + c] *= inv;
    }
  }
}

static void mhc_sinkhorn_bwd(size_t C, const float *raw,
                             const double dH[BCIR_EXT_MHC_MAX_C * BCIR_EXT_MHC_MAX_C],
                             double draw_out[BCIR_EXT_MHC_MAX_C * BCIR_EXT_MHC_MAX_C]) {
  double M[MHC_SK_ITERS + 1][16];
  double R[MHC_SK_ITERS][16];
  double rsum[MHC_SK_ITERS][4];
  double csum[MHC_SK_ITERS][4];
  memset(M, 0, sizeof(M));
  memset(draw_out, 0, sizeof(double) * BCIR_EXT_MHC_MAX_C * BCIR_EXT_MHC_MAX_C);
  if (C == 0u || C > BCIR_EXT_MHC_MAX_C) return;
  double mx = -1e30;
  for (size_t i = 0; i < C * C; ++i) {
    double v = (double)raw[i];
    if (v > mx) mx = v;
  }
  for (size_t i = 0; i < C * C; ++i) M[0][i] = exp((double)raw[i] - mx);
  for (int it = 0; it < MHC_SK_ITERS; ++it) {
    for (size_t r = 0; r < C; ++r) {
      double s = 0.0;
      for (size_t c = 0; c < C; ++c) s += M[it][r * C + c];
      rsum[it][r] = (s > 1e-30) ? s : 1e-30;
      for (size_t c = 0; c < C; ++c) R[it][r * C + c] = M[it][r * C + c] / rsum[it][r];
    }
    for (size_t c = 0; c < C; ++c) {
      double s = 0.0;
      for (size_t r = 0; r < C; ++r) s += R[it][r * C + c];
      csum[it][c] = (s > 1e-30) ? s : 1e-30;
      for (size_t r = 0; r < C; ++r) M[it + 1][r * C + c] = R[it][r * C + c] / csum[it][c];
    }
  }
  double G[16], dR[16];
  for (size_t i = 0; i < C * C; ++i) G[i] = dH[i];
  for (int it = MHC_SK_ITERS - 1; it >= 0; --it) {
    for (size_t c = 0; c < C; ++c) {
      double dot = 0.0;
      for (size_t r = 0; r < C; ++r) dot += G[r * C + c] * M[it + 1][r * C + c];
      double inv = 1.0 / csum[it][c];
      for (size_t r = 0; r < C; ++r) dR[r * C + c] = (G[r * C + c] - dot) * inv;
    }
    for (size_t r = 0; r < C; ++r) {
      double dot = 0.0;
      for (size_t c = 0; c < C; ++c) dot += dR[r * C + c] * R[it][r * C + c];
      double inv = 1.0 / rsum[it][r];
      for (size_t c = 0; c < C; ++c) G[r * C + c] = (dR[r * C + c] - dot) * inv;
    }
  }
  for (size_t i = 0; i < C * C; ++i) draw_out[i] = G[i] * M[0][i];
}

int bcir_mhc_verify_birkhoff(uint32_t C, double *out_max_marginal_err, double *out_max_grad_err) {
  if (C < 2u || C > BCIR_EXT_MHC_MAX_C) return 1;
  float raw[16];
  double H[16], dH[16], draw[16];
  for (size_t i = 0; i < (size_t)C * C; ++i) {
    size_t r = i / C, c = i % C;
    raw[i] = (r == c ? 1.7f : -0.4f) + 0.13f * (float)r - 0.09f * (float)c;
    dH[i] = 0.3 * (double)((int)(i % 5u) - 2);
  }
  mhc_sinkhorn_fwd(C, raw, H);
  double max_marg = 0.0;
  for (size_t r = 0; r < C; ++r) {
    double s = 0.0; for (size_t c = 0; c < C; ++c) s += H[r * C + c];
    double e = fabs(s - 1.0); if (e > max_marg) max_marg = e;
  }
  for (size_t c = 0; c < C; ++c) {
    double s = 0.0; for (size_t r = 0; r < C; ++r) s += H[r * C + c];
    double e = fabs(s - 1.0); if (e > max_marg) max_marg = e;
  }
  mhc_sinkhorn_bwd(C, raw, dH, draw);
  double max_gerr = 0.0;
  const double eps = 1e-4;
  for (size_t i = 0; i < (size_t)C * C; ++i) {
    float orig = raw[i];
    double Hp[16], Hm[16];
    raw[i] = (float)((double)orig + eps); mhc_sinkhorn_fwd(C, raw, Hp);
    raw[i] = (float)((double)orig - eps); mhc_sinkhorn_fwd(C, raw, Hm);
    raw[i] = orig;
    double lp = 0.0, lm = 0.0;
    for (size_t k = 0; k < (size_t)C * C; ++k) { lp += dH[k] * Hp[k]; lm += dH[k] * Hm[k]; }
    double fd = (lp - lm) / (2.0 * eps);
    double e = fabs(fd - draw[i]);
    if (e > max_gerr) max_gerr = e;
  }
  if (out_max_marginal_err) *out_max_marginal_err = max_marg;
  if (out_max_grad_err) *out_max_grad_err = max_gerr;
  return (max_marg < 1e-5 && max_gerr < 1e-4) ? 0 : 1;
}

/* ==================== Canon Causal Conv1d (k=4, Residual) ==================== */
typedef struct canon_ctx {
  size_t B, T, d, col_chunk;
  const float *x, *w, *dy;
  float *y, *dx, *dw;
} canon_ctx;

static void canon_fwd_worker(void *vctx, size_t task) {
  const canon_ctx *c = (const canon_ctx *)vctx;
  size_t c0 = task * c->col_chunk, c1 = (c0 + c->col_chunk <= c->d) ? (c0 + c->col_chunk) : c->d;
  for (size_t b = 0; b < c->B; ++b) {
    for (size_t t = 0; t < c->T; ++t) {
      const float *xt = c->x + (b * c->T + t) * c->d;
      float *yt = c->y + (b * c->T + t) * c->d;
      size_t kmax = (t + 1u < BCIR_EXT_CANON_K) ? (t + 1u) : BCIR_EXT_CANON_K;
      for (size_t j = c0; j < c1; ++j) {
        double acc = (double)xt[j];
        for (size_t k = 0; k < kmax; ++k) {
          acc += (double)c->w[k * c->d + j] * (double)c->x[(b * c->T + (t - k)) * c->d + j];
        }
        yt[j] = (float)acc;
      }
    }
  }
}

static void canon_bwd_worker(void *vctx, size_t task) {
  const canon_ctx *c = (const canon_ctx *)vctx;
  size_t c0 = task * c->col_chunk, c1 = (c0 + c->col_chunk <= c->d) ? (c0 + c->col_chunk) : c->d;
  for (size_t j = c0; j < c1; ++j) {
    double dw_loc[BCIR_EXT_CANON_K] = {0.0, 0.0, 0.0, 0.0};
    for (size_t b = 0; b < c->B; ++b) {
      for (size_t t = 0; t < c->T; ++t) {
        double dyt = (double)c->dy[(b * c->T + t) * c->d + j];
        size_t kmax_w = (t + 1u < BCIR_EXT_CANON_K) ? (t + 1u) : BCIR_EXT_CANON_K;
        for (size_t k = 0; k < kmax_w; ++k) {
          dw_loc[k] += dyt * (double)c->x[(b * c->T + (t - k)) * c->d + j];
        }
        double dx_acc = dyt;
        size_t kmax_x = (c->T - t < BCIR_EXT_CANON_K) ? (c->T - t) : BCIR_EXT_CANON_K;
        for (size_t k = 0; k < kmax_x; ++k) {
          dx_acc += (double)c->w[k * c->d + j] * (double)c->dy[(b * c->T + (t + k)) * c->d + j];
        }
        c->dx[(b * c->T + t) * c->d + j] = (float)dx_acc;
      }
    }
    for (size_t k = 0; k < BCIR_EXT_CANON_K; ++k) {
      c->dw[k * c->d + j] = (float)((double)c->dw[k * c->d + j] + dw_loc[k]);
    }
  }
}

/* ==================== Lifecycle: Create / Destroy ==================== */
bcir_ext_model *bcir_ext_create(const bcir_ext_spec *spec, const bcir_prec_policy *pol) {
  if (!spec || !spec->vocab || !spec->width || !spec->heads || !spec->kvheads ||
      !spec->layers || spec->layers > BCIR_EXT_MAX_LAYERS || !spec->ff ||
      !spec->batch || !spec->time || (spec->width % spec->heads) != 0u ||
      (spec->heads % spec->kvheads) != 0u ||
      spec->gate_mode > BCIR_GATE_G1_LOWRANK ||
      spec->canon_mode > BCIR_CANON_AC ||
      spec->qk_norm > 1u || spec->online_softmax > 1u ||
      (spec->mhc_streams > 1u && spec->mhc_streams != 2u && spec->mhc_streams != 4u)) {
    return NULL;
  }
  for (uint32_t l = 0; l < spec->layers; ++l) {
    if (spec->layer_mixer[l] >= BCIR_MIXER_KIND_COUNT) return NULL;
  }
  double t0 = now_ms();
  bcir_ext_model *m = (bcir_ext_model *)ext_zalloc(1u, sizeof(bcir_ext_model));
  if (!m) return NULL;
  m->sp = *spec;
  if (m->sp.swa_window == 0u) m->sp.swa_window = 32u;
  if (m->sp.delta_chunk == 0u) m->sp.delta_chunk = 16u;
  if (m->sp.mla_dc == 0u) m->sp.mla_dc = m->sp.width / 2u;
  if (m->sp.mla_dr == 0u) m->sp.mla_dr = (m->sp.width / m->sp.heads) / 2u;
  if (m->sp.mla_dr < 2u) m->sp.mla_dr = 2u;
  if (m->sp.mla_dr & 1u) m->sp.mla_dr++;
  if (m->sp.threads == 0u) m->sp.threads = 1u;
  if (pol) m->pol = *pol; else bcir_prec_policy_default(&m->pol);
  bcir_prec_telemetry_reset(&m->tel);

  size_t d = m->sp.width, H = m->sp.heads, KH = m->sp.kvheads, dh = d / H;
  size_t kvd = KH * dh, ff = m->sp.ff, V = m->sp.vocab, B = m->sp.batch, T = m->sp.time;
  size_t dc = m->sp.mla_dc, dr = m->sp.mla_dr;
  size_t rg = (m->sp.gate_rank > 0u) ? (size_t)m->sp.gate_rank : max_sz(16u, dh);
  if (rg > d) rg = d;
  m->gate_rank = rg;
  m->mhc_c = (m->sp.mhc_streams > 1u) ? (size_t)m->sp.mhc_streams : 1u;

  if (m->sp.threads > 1u) {
    size_t psz = bcir_tensor_pool_size(m->sp.threads);
    if (psz > 0u) {
      m->pool_mem = bcir_aligned_calloc64(psz, 1u);
      m->pool = bcir_tensor_pool_init(m->pool_mem, psz, m->sp.threads);
    }
  }

  size_t tcap = 4u + (size_t)m->sp.layers * 32u;
  bcir_init_tensor_desc *idescs = (bcir_init_tensor_desc *)ext_zalloc(tcap, sizeof(*idescs));
  m->tensors = (bcir_ext_tensor_info *)ext_zalloc(tcap, sizeof(*m->tensors));
  if (!idescs || !m->tensors) { ext_dealloc(idescs); bcir_ext_destroy(m); return NULL; }

  add_tensor(m, &idescs, &tcap, "embed", -1, 0, V, d, BCIR_ROLE_EMBED, 0.0f, &m->off_embed);
  char nm[48];
  for (uint32_t l = 0; l < m->sp.layers; ++l) {
    layer_offsets *lo = &m->loff[l];
    lo->mixer = m->sp.layer_mixer[l];
    lo->use_rope = m->sp.layer_use_rope[l];
    snprintf(nm, sizeof(nm), "L%u.norm1", l);
    add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_NORM, 1.0f, &lo->norm1);
    if (m->sp.canon_mode >= BCIR_CANON_A) {
      snprintf(nm, sizeof(nm), "L%u.canon_a", l);
      add_tensor(m, &idescs, &tcap, nm, (int)l, 0, BCIR_EXT_CANON_K, d, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->canon_a);
    }
    if (lo->mixer == BCIR_MIXER_GQA || lo->mixer == BCIR_MIXER_SWA) {
      snprintf(nm, sizeof(nm), "L%u.wq", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wq);
      snprintf(nm, sizeof(nm), "L%u.wk", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, kvd, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wk);
      snprintf(nm, sizeof(nm), "L%u.wv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, kvd, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wv);
      if (m->sp.qk_norm) {
        snprintf(nm, sizeof(nm), "L%u.q_norm", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, dh, BCIR_ROLE_NORM, 1.0f, &lo->q_norm);
        snprintf(nm, sizeof(nm), "L%u.k_norm", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, dh, BCIR_ROLE_NORM, 1.0f, &lo->k_norm);
      }
      snprintf(nm, sizeof(nm), "L%u.wo", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_OUT, 0.0f, &lo->wo);
    } else if (lo->mixer == BCIR_MIXER_DELTANET) {
      snprintf(nm, sizeof(nm), "L%u.wq", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wq);
      snprintf(nm, sizeof(nm), "L%u.wk", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wk);
      snprintf(nm, sizeof(nm), "L%u.wv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wv);
      snprintf(nm, sizeof(nm), "L%u.walpha", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, H, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->walpha);
      snprintf(nm, sizeof(nm), "L%u.balpha", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, H, BCIR_ROLE_CONST_NOWD, 2.5f, &lo->balpha);
      snprintf(nm, sizeof(nm), "L%u.wbeta", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, H, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wbeta);
      snprintf(nm, sizeof(nm), "L%u.bbeta", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, H, BCIR_ROLE_CONST_NOWD, -2.0f, &lo->bbeta);
      snprintf(nm, sizeof(nm), "L%u.wo", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_OUT, 0.0f, &lo->wo);
    } else if (lo->mixer == BCIR_MIXER_KDA) {
      snprintf(nm, sizeof(nm), "L%u.wq", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wq);
      snprintf(nm, sizeof(nm), "L%u.wk", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wk);
      snprintf(nm, sizeof(nm), "L%u.wv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wv);
      snprintf(nm, sizeof(nm), "L%u.wa1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, rg, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wa1);
      snprintf(nm, sizeof(nm), "L%u.wa2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, rg, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wa2);
      snprintf(nm, sizeof(nm), "L%u.ba_chan", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_CONST_NOWD, 2.5f, &lo->ba_chan);
      snprintf(nm, sizeof(nm), "L%u.wbeta", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, H, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wbeta);
      snprintf(nm, sizeof(nm), "L%u.bbeta", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, H, BCIR_ROLE_CONST_NOWD, -2.0f, &lo->bbeta);
      snprintf(nm, sizeof(nm), "L%u.o_norm", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, dh, BCIR_ROLE_NORM, 1.0f, &lo->o_norm);
      snprintf(nm, sizeof(nm), "L%u.wo", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_OUT, 0.0f, &lo->wo);
    } else if (lo->mixer == BCIR_MIXER_GDN2) {
      snprintf(nm, sizeof(nm), "L%u.wq", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wq);
      snprintf(nm, sizeof(nm), "L%u.wk", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wk);
      snprintf(nm, sizeof(nm), "L%u.wv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wv);
      snprintf(nm, sizeof(nm), "L%u.wa1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, rg, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wa1);
      snprintf(nm, sizeof(nm), "L%u.wa2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, rg, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wa2);
      snprintf(nm, sizeof(nm), "L%u.ba_chan", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_CONST_NOWD, 2.5f, &lo->ba_chan);
      snprintf(nm, sizeof(nm), "L%u.we1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, rg, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->we1);
      snprintf(nm, sizeof(nm), "L%u.we2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, rg, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->we2);
      snprintf(nm, sizeof(nm), "L%u.be_chan", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_CONST_NOWD, -2.0f, &lo->be_chan);
      snprintf(nm, sizeof(nm), "L%u.ww1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, rg, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->ww1);
      snprintf(nm, sizeof(nm), "L%u.ww2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, rg, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->ww2);
      snprintf(nm, sizeof(nm), "L%u.bw_chan", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_CONST_NOWD, -2.0f, &lo->bw_chan);
      snprintf(nm, sizeof(nm), "L%u.o_norm", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, dh, BCIR_ROLE_NORM, 1.0f, &lo->o_norm);
      snprintf(nm, sizeof(nm), "L%u.wo", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_OUT, 0.0f, &lo->wo);
    } else if (lo->mixer == BCIR_MIXER_MLA) {
      snprintf(nm, sizeof(nm), "L%u.wq_c", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wq_c);
      if (m->sp.qk_norm) {
        snprintf(nm, sizeof(nm), "L%u.q_norm", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, dh, BCIR_ROLE_NORM, 1.0f, &lo->q_norm);
      }
      if (lo->use_rope) {
        snprintf(nm, sizeof(nm), "L%u.wq_r", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, H * dr, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wq_r);
      }
      snprintf(nm, sizeof(nm), "L%u.w_dkv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, dc, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->w_dkv);
      snprintf(nm, sizeof(nm), "L%u.norm_kv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, dc, BCIR_ROLE_NORM, 1.0f, &lo->norm_kv);
      snprintf(nm, sizeof(nm), "L%u.w_uk", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, dc, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->w_uk);
      snprintf(nm, sizeof(nm), "L%u.w_uv", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, dc, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->w_uv);
      if (lo->use_rope) {
        snprintf(nm, sizeof(nm), "L%u.w_kr", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, dr, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->w_kr);
      }
      snprintf(nm, sizeof(nm), "L%u.wo", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, d, BCIR_ROLE_LINEAR_OUT, 0.0f, &lo->wo);
    }
    if (m->sp.gate_mode == BCIR_GATE_G1_HEADWISE) {
      snprintf(nm, sizeof(nm), "L%u.wg1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, H, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wg1);
      snprintf(nm, sizeof(nm), "L%u.bg", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, H, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->bg);
    } else if (m->sp.gate_mode == BCIR_GATE_G1_LOWRANK) {
      snprintf(nm, sizeof(nm), "L%u.wg1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, rg, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wg1);
      snprintf(nm, sizeof(nm), "L%u.wg2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 1, d, rg, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wg2);
      snprintf(nm, sizeof(nm), "L%u.bg", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->bg);
    }
    if (m->mhc_c > 1u) {
      float pre_init = (m->mhc_c == 2u) ? 0.0f : -1.09861229f; /* sigmoid = 1/C */
      snprintf(nm, sizeof(nm), "L%u.mhc_hres1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, m->mhc_c, m->mhc_c, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->mhc_hres1);
      snprintf(nm, sizeof(nm), "L%u.mhc_hpre1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, m->mhc_c, BCIR_ROLE_CONST_NOWD, pre_init, &lo->mhc_hpre1);
      snprintf(nm, sizeof(nm), "L%u.mhc_hpost1", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, m->mhc_c, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->mhc_hpost1);
      snprintf(nm, sizeof(nm), "L%u.mhc_hres2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, m->mhc_c, m->mhc_c, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->mhc_hres2);
      snprintf(nm, sizeof(nm), "L%u.mhc_hpre2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, m->mhc_c, BCIR_ROLE_CONST_NOWD, pre_init, &lo->mhc_hpre2);
      snprintf(nm, sizeof(nm), "L%u.mhc_hpost2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, m->mhc_c, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->mhc_hpost2);
    }
    snprintf(nm, sizeof(nm), "L%u.norm2", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 0, 1u, d, BCIR_ROLE_NORM, 1.0f, &lo->norm2);
    if (m->sp.canon_mode >= BCIR_CANON_AC) {
      snprintf(nm, sizeof(nm), "L%u.canon_c", l);
      add_tensor(m, &idescs, &tcap, nm, (int)l, 0, BCIR_EXT_CANON_K, d, BCIR_ROLE_ZERO_NOWD, 0.0f, &lo->canon_c);
    }
    snprintf(nm, sizeof(nm), "L%u.wgate", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 2, ff, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wgate);
    snprintf(nm, sizeof(nm), "L%u.wup", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 2, ff, d, BCIR_ROLE_LINEAR_IN, 0.0f, &lo->wup);
    snprintf(nm, sizeof(nm), "L%u.wdown", l); add_tensor(m, &idescs, &tcap, nm, (int)l, 2, d, ff, BCIR_ROLE_LINEAR_OUT, 0.0f, &lo->wdown);
  }
  add_tensor(m, &idescs, &tcap, "final_norm", -1, 0, 1u, d, BCIR_ROLE_NORM, 1.0f, &m->off_final_norm);
  if (!m->sp.tied) add_tensor(m, &idescs, &tcap, "lm_head", -1, 0, V, d, BCIR_ROLE_EMBED, 0.0f, &m->off_lm_head);
  else m->off_lm_head = m->off_embed;

  m->weights = (float *)bcir_aligned_calloc64(m->n_params, sizeof(float));
  bcir_opt_tensor_shape *oshapes = (bcir_opt_tensor_shape *)ext_zalloc(m->n_tensors, sizeof(*oshapes));
  if (!m->weights || !oshapes) { ext_dealloc(idescs); ext_dealloc(oshapes); bcir_ext_destroy(m); return NULL; }
  if (!m->sp.skip_init) {
    bcir_init_weights_c11(m->weights, m->n_params, idescs, m->n_tensors,
                          m->sp.layers, m->sp.width, m->sp.mup_d_base,
                          m->sp.seed, m->sp.init_policy, oshapes);
    /* Initialize mHC H_res diagonally strong so Sinkhorn-Knopp starts near identity. */
    if (m->mhc_c > 1u) {
      for (uint32_t l = 0; l < m->sp.layers; ++l) {
        const layer_offsets *lo = &m->loff[l];
        for (size_t i = 0; i < m->mhc_c; ++i) {
          m->weights[lo->mhc_hres1 + i * m->mhc_c + i] = 3.0f;
          m->weights[lo->mhc_hres2 + i * m->mhc_c + i] = 3.0f;
        }
      }
    }
  } else {
    double mup_lr_scale = (m->sp.mup_d_base > 0u && m->sp.width > 0u) ? ((double)m->sp.mup_d_base / (double)m->sp.width) : 1.0;
    for (size_t s = 0; s < m->n_tensors; ++s) {
      int role = idescs[s].role;
      oshapes[s].offset = idescs[s].offset;
      oshapes[s].rows = idescs[s].rows;
      oshapes[s].cols = idescs[s].cols;
      oshapes[s].apply_wd = (role == BCIR_ROLE_NORM || role == BCIR_ROLE_ZERO_NOWD || role == BCIR_ROLE_CONST_NOWD) ? 0 : 1;
      oshapes[s].lr_mult = (m->sp.init_policy == BCIR_INIT_MUP_V1 && (role == BCIR_ROLE_LINEAR_IN || role == BCIR_ROLE_LINEAR_OUT)) ? mup_lr_scale : 1.0;
      oshapes[s].is_muon_2d = ((role == BCIR_ROLE_LINEAR_IN || role == BCIR_ROLE_LINEAR_OUT) && idescs[s].rows >= 2u && idescs[s].cols >= 2u) ? 1 : 0;
    }
  }
  m->opt = bcir_opt_create(m->pol.opt_kind, m->n_params, oshapes, m->n_tensors);
  ext_dealloc(idescs); ext_dealloc(oshapes);
  if (!m->opt) { bcir_ext_destroy(m); return NULL; }

  if (m->opt->grad_fp32 == NULL) {
    size_t max_t_elems = 0u;
    for (size_t s = 0; s < m->n_tensors; ++s) {
      size_t elems = m->tensors[s].rows * m->tensors[s].cols;
      if (elems > max_t_elems) max_t_elems = elems;
    }
    m->grad_tile_bytes = max_t_elems * sizeof(float);
    m->grad_tile = (float *)bcir_aligned_calloc64(max_t_elems > 0u ? max_t_elems : 1u, sizeof(float));
  }

  /* Allocate activations & track memory honestly. */
  size_t N = B * T, act_b = 0u, prob_b = 0u, kv_per_tok = 0u, rec_b = 0u;
  int has_dn = 0, has_cdelta = 0, has_gdn2 = 0, has_mla = 0, has_mla_rope = 0, has_soft = 0;
  for (uint32_t l = 0; l < m->sp.layers; ++l) {
    layer_act *a = &m->acts[l];
    uint32_t mx = m->loff[l].mixer;
    if (m->mhc_c > 1u) {
      a->mhc_x1 = (float *)bcir_aligned_calloc64(N * m->mhc_c * d, 4u);
      a->mhc_x2 = (float *)bcir_aligned_calloc64(N * m->mhc_c * d, 4u);
      a->mix_out = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->mlp_out = (float *)bcir_aligned_calloc64(N * d, 4u);
      act_b += (2u * N * m->mhc_c * d + 2u * N * d) * 4u;
    }
    a->x_in = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    a->n1 = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    if (m->sp.canon_mode >= BCIR_CANON_A) {
      a->n1_raw = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    } else {
      a->n1_raw = a->n1;
    }
    a->ctx = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    if (m->sp.gate_mode > BCIR_GATE_NONE) {
      a->ctx_raw = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
      if (m->sp.gate_mode == BCIR_GATE_G1_HEADWISE) {
        a->g_val = (float *)bcir_aligned_calloc64(N * H, 4u); act_b += N * H * 4u;
      } else {
        a->g_mid = (float *)bcir_aligned_calloc64(N * rg, 4u);
        a->g_val = (float *)bcir_aligned_calloc64(N * d, 4u);
        act_b += (N * rg + N * d) * 4u;
      }
    } else {
      a->ctx_raw = a->ctx;
    }
    a->x2 = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    a->n2 = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    if (m->sp.canon_mode >= BCIR_CANON_AC) {
      a->n2_raw = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
    } else {
      a->n2_raw = a->n2;
    }
    a->gate = (float *)bcir_aligned_calloc64(N * ff, 4u); act_b += N * ff * 4u;
    a->up = (float *)bcir_aligned_calloc64(N * ff, 4u); act_b += N * ff * 4u;
    a->hidden = (float *)bcir_aligned_calloc64(N * ff, 4u); act_b += N * ff * 4u;

    if (mx == BCIR_MIXER_GQA || mx == BCIR_MIXER_SWA) {
      has_soft = 1;
      a->q = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
      a->k = (float *)bcir_aligned_calloc64(N * kvd, 4u); act_b += N * kvd * 4u;
      a->v = (float *)bcir_aligned_calloc64(N * kvd, 4u); act_b += N * kvd * 4u;
      if (m->sp.qk_norm) {
        a->q_raw = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u;
        a->k_raw = (float *)bcir_aligned_calloc64(N * kvd, 4u); act_b += N * kvd * 4u;
      }
      if (mx == BCIR_MIXER_GQA) {
        if (m->sp.online_softmax) {
          a->lse = (double *)bcir_aligned_calloc64(B * H * T, sizeof(double));
          prob_b += B * H * T * sizeof(double);
        } else {
          a->p = (float *)bcir_aligned_calloc64(B * H * T * T, 4u);
          prob_b += B * H * T * T * 4u;
        }
        kv_per_tok += 2u * kvd * 4u;
      } else {
        size_t W = m->sp.swa_window;
        a->p = (float *)bcir_aligned_calloc64(B * H * T * W, 4u);
        prob_b += B * H * T * W * 4u;
        kv_per_tok += 2u * kvd * 4u;
      }
    } else if (mx == BCIR_MIXER_DELTANET) {
      has_dn = 1;
      size_t C = m->sp.delta_chunk, nc = (T + C - 1u) / C;
      a->q_raw = (float *)bcir_aligned_calloc64(N * d, 4u); a->k_raw = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->q = (float *)bcir_aligned_calloc64(N * d, 4u); a->k = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->v = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->q_inv = (float *)bcir_aligned_calloc64(N * H, 4u); a->k_inv = (float *)bcir_aligned_calloc64(N * H, 4u);
      a->a_pre = (float *)bcir_aligned_calloc64(N * H, 4u); a->b_pre = (float *)bcir_aligned_calloc64(N * H, 4u);
      a->alpha = (float *)bcir_aligned_calloc64(N * H, 4u); a->beta = (float *)bcir_aligned_calloc64(N * H, 4u);
      a->dn_sprev = (double *)bcir_aligned_calloc64(B * H * nc * dh * dh, sizeof(double));
      a->dn_W = (double *)bcir_aligned_calloc64(B * H * T * dh, sizeof(double));
      act_b += (5u * N * d + 6u * N * H) * 4u;
      prob_b += (B * H * nc * dh * dh + B * H * T * dh) * sizeof(double);
      rec_b += H * dh * dh * sizeof(double);
    } else if (mx == BCIR_MIXER_KDA || mx == BCIR_MIXER_GDN2) {
      has_cdelta = 1;
      if (mx == BCIR_MIXER_GDN2) has_gdn2 = 1;
      size_t C = m->sp.delta_chunk, nc = (T + C - 1u) / C;
      a->q_raw = (float *)bcir_aligned_calloc64(N * d, 4u); a->k_raw = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->q = (float *)bcir_aligned_calloc64(N * d, 4u); a->k = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->v = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->q_inv = (float *)bcir_aligned_calloc64(N * H, 4u); a->k_inv = (float *)bcir_aligned_calloc64(N * H, 4u);
      a->a_mid = (float *)bcir_aligned_calloc64(N * rg, 4u);
      a->a_pre = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->alpha = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->cd_y = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->dn_sprev = (double *)bcir_aligned_calloc64(B * H * nc * dh * dh, sizeof(double));
      a->dn_W = (double *)bcir_aligned_calloc64(B * H * T * dh, sizeof(double));
      act_b += (8u * N * d + N * rg + 2u * N * H) * 4u;
      if (mx == BCIR_MIXER_KDA) {
        a->b_pre = (float *)bcir_aligned_calloc64(N * H, 4u);
        a->beta = (float *)bcir_aligned_calloc64(N * H, 4u);
        act_b += 2u * N * H * 4u;
      } else {
        a->e_mid = (float *)bcir_aligned_calloc64(N * rg, 4u);
        a->e_pre = (float *)bcir_aligned_calloc64(N * d, 4u);
        a->erase = (float *)bcir_aligned_calloc64(N * d, 4u);
        a->w_mid = (float *)bcir_aligned_calloc64(N * rg, 4u);
        a->w_pre = (float *)bcir_aligned_calloc64(N * d, 4u);
        a->write = (float *)bcir_aligned_calloc64(N * d, 4u);
        act_b += (2u * N * rg + 4u * N * d) * 4u;
      }
      prob_b += (B * H * nc * dh * dh + B * H * T * dh) * sizeof(double);
      rec_b += H * dh * dh * sizeof(double);
    } else if (mx == BCIR_MIXER_MLA) {
      has_mla = 1; has_soft = 1;
      a->q_c = (float *)bcir_aligned_calloc64(N * d, 4u);
      if (m->sp.qk_norm) { a->q_c_raw = (float *)bcir_aligned_calloc64(N * d, 4u); act_b += N * d * 4u; }
      if (m->loff[l].use_rope) {
        has_mla_rope = 1;
        a->q_r = (float *)bcir_aligned_calloc64(N * H * dr, 4u);
        a->k_r = (float *)bcir_aligned_calloc64(N * dr, 4u);
        act_b += (N * H * dr + N * dr) * 4u;
      }
      a->c_raw = (float *)bcir_aligned_calloc64(N * dc, 4u);
      a->c_hat = (float *)bcir_aligned_calloc64(N * dc, 4u);
      a->k_c = (float *)bcir_aligned_calloc64(N * d, 4u);
      a->v_c = (float *)bcir_aligned_calloc64(N * d, 4u);
      act_b += (3u * N * d + 2u * N * dc) * 4u;
      if (m->sp.online_softmax) {
        a->lse = (double *)bcir_aligned_calloc64(B * H * T, sizeof(double));
        prob_b += B * H * T * sizeof(double);
      } else {
        a->mla_p = (float *)bcir_aligned_calloc64(B * H * T * T, 4u);
        prob_b += B * H * T * T * 4u;
      }
      kv_per_tok += (dc + (m->loff[l].use_rope ? dr : 0u)) * 4u;
    }
  }
  if (m->mhc_c > 1u) {
    m->mhc_xfinal = (float *)bcir_aligned_calloc64(N * m->mhc_c * d, 4u);
    act_b += N * m->mhc_c * d * 4u;
  }
  m->x_final = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->norm_final = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->logits = (float *)bcir_aligned_calloc64(N * V, 4u);
  act_b += (2u * N * d + N * V) * 4u;

  m->rope_hd = (float *)bcir_aligned_calloc64(T * dh, 4u);
  bcir_tensor_rope_table_pool(m->pool, T, dh, m->sp.rope_base, m->rope_hd);
  m->rope_dr = (float *)bcir_aligned_calloc64(T * dr, 4u);
  bcir_tensor_rope_table_pool(m->pool, T, dr, m->sp.rope_base, m->rope_dr);

  size_t scr_b = m->grad_tile_bytes + m->opt->scratch_bytes;
  if (m->mhc_c > 1u) {
    m->d_mhc_cur = (float *)bcir_aligned_calloc64(N * m->mhc_c * d, 4u);
    m->d_mhc_next = (float *)bcir_aligned_calloc64(N * m->mhc_c * d, 4u);
    scr_b += 2u * N * m->mhc_c * d * 4u;
  }
  m->d_cur = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_next = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_n = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_n_tmp = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_ctx = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_gate = (float *)bcir_aligned_calloc64(N * ff, 4u);
  m->d_up = (float *)bcir_aligned_calloc64(N * ff, 4u);
  m->d_hid = (float *)bcir_aligned_calloc64(N * ff, 4u);
  m->d_logits = (float *)bcir_aligned_calloc64(N * V, 4u);
  m->rms_inv = (float *)bcir_aligned_calloc64(N * H, 4u);
  m->d_q = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_k = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->d_v = (float *)bcir_aligned_calloc64(N * d, 4u);
  m->rms_dg64 = (double *)bcir_aligned_calloc64(max_sz(d, dc), sizeof(double));
  scr_b += (8u * N * d + 3u * N * ff + N * V + N * H) * 4u + max_sz(d, dc) * sizeof(double);

  if (m->sp.gate_mode > BCIR_GATE_NONE || has_cdelta) {
    m->d_mid = (float *)bcir_aligned_calloc64(N * rg, 4u);
    m->d_gpre = (float *)bcir_aligned_calloc64(N * d, 4u);
    scr_b += (N * rg + N * d) * 4u;
  }
  if (has_soft) {
    size_t max_tw = max_sz(T, m->sp.swa_window);
    m->attn_max_log = (double *)bcir_aligned_calloc64(B * H, sizeof(double));
    m->attn_promos = (uint64_t *)bcir_aligned_calloc64(B * H, sizeof(uint64_t));
    m->attn_sink_sum = (double *)bcir_aligned_calloc64(B * H, sizeof(double));
    m->attn_scores = (double *)bcir_aligned_calloc64(B * H * max_tw, sizeof(double));
    m->attn_dk_loc = (double *)bcir_aligned_calloc64(B * H * T * dh, sizeof(double));
    m->attn_dv_loc = (double *)bcir_aligned_calloc64(B * H * T * dh, sizeof(double));
    scr_b += (B * H * (2u + max_tw) + 2u * B * H * T * dh) * sizeof(double) + B * H * sizeof(uint64_t);
  }
  if (has_dn || has_cdelta) {
    size_t C = m->sp.delta_chunk;
    m->d_a = (float *)bcir_aligned_calloc64(N * d, 4u);
    m->d_b = (float *)bcir_aligned_calloc64(N * H, 4u);
    m->dn_work_S = (double *)bcir_aligned_calloc64(B * H * dh * dh, sizeof(double));
    m->dn_work_dS = (double *)bcir_aligned_calloc64(B * H * dh * dh, sizeof(double));
    m->dn_work_g = (double *)bcir_aligned_calloc64(B * H * C * dh, sizeof(double));
    m->dn_work_gam = (double *)bcir_aligned_calloc64(B * H * C * dh, sizeof(double));
    m->dn_work_A = (double *)bcir_aligned_calloc64(B * H * C * C, sizeof(double));
    m->dn_work_qS = (double *)bcir_aligned_calloc64(B * H * dh, sizeof(double));
    m->dn_work_dW = (double *)bcir_aligned_calloc64(B * H * C * dh, sizeof(double));
    m->dn_work_dA = (double *)bcir_aligned_calloc64(B * H * C * C, sizeof(double));
    m->dn_work_dg = (double *)bcir_aligned_calloc64(B * H * C, sizeof(double));
    scr_b += (N * d + N * H) * 4u +
             B * H * (2u * dh * dh + 3u * C * dh + 2u * C * C + dh + C) * sizeof(double);
    if (has_cdelta) {
      m->dn_work_Shist = (double *)bcir_aligned_calloc64(B * H * (C + 1u) * dh * dh, sizeof(double));
      m->dn_work_dP = (double *)bcir_aligned_calloc64(B * H * dh * dh, sizeof(double));
      scr_b += B * H * ((C + 2u) * dh * dh) * sizeof(double);
    }
    if (has_gdn2) {
      m->d_e = (float *)bcir_aligned_calloc64(N * d, 4u);
      m->d_w = (float *)bcir_aligned_calloc64(N * d, 4u);
      scr_b += 2u * N * d * 4u;
    }
  }
  if (has_mla) {
    m->d_qc = (float *)bcir_aligned_calloc64(N * d, 4u);
    m->d_chat = (float *)bcir_aligned_calloc64(N * dc, 4u);
    m->d_craw = (float *)bcir_aligned_calloc64(N * dc, 4u);
    m->d_kc = (float *)bcir_aligned_calloc64(N * d, 4u);
    m->d_vc = (float *)bcir_aligned_calloc64(N * d, 4u);
    scr_b += (3u * N * d + 2u * N * dc) * 4u;
    if (has_mla_rope) {
      m->d_qr = (float *)bcir_aligned_calloc64(N * H * dr, 4u);
      m->d_kr = (float *)bcir_aligned_calloc64(N * dr, 4u);
      m->attn_dkr_loc = (double *)bcir_aligned_calloc64(B * T * dr, sizeof(double));
      scr_b += (N * H * dr + N * dr) * 4u + B * T * dr * sizeof(double);
    }
  }

  size_t max_dim = max_sz(max_sz(d, ff), max_sz(V, max_sz(dc, rg)));
  size_t gscr = 0u;
  int gkinds[4] = { m->pol.attn_fwd_gemm, m->pol.attn_bwd_gemm, m->pol.mlp_fwd_gemm, m->pol.mlp_bwd_gemm };
  for (int i = 0; i < 4; ++i) {
    gscr = max_sz(gscr, bcir_prec_mm_scratch_bytes(gkinds[i], N, max_dim, max_dim));
    gscr = max_sz(gscr, bcir_prec_mm_scratch_bytes(gkinds[i], max_dim, max_dim, N));
  }
  if (gscr > 0u) {
    m->gemm_scratch = bcir_aligned_calloc64(gscr, 1u);
    m->gemm_scratch_bytes = gscr;
    scr_b += gscr;
  }

  m->mem.n_params = m->n_params;
  m->mem.weight_bytes = m->n_params * 4u;
  m->mem.opt_only_bytes = m->opt->opt_only_bytes;
  m->mem.persistent_bytes = m->opt->persistent_bytes;
  m->mem.activation_bytes = act_b;
  m->mem.mixer_prob_bytes = prob_b;
  m->mem.scratch_bytes = scr_b;
  m->mem.total_bytes = m->mem.persistent_bytes + act_b + prob_b + scr_b;
  m->mem.infer_kv_bytes_per_tok = kv_per_tok;
  m->mem.infer_rec_state_bytes = rec_b;
  m->mem.init_ms = now_ms() - t0;
  return m;
}

void bcir_ext_destroy(bcir_ext_model *m) {
  if (!m) return;
  if (m->pool) bcir_tensor_pool_stop(m->pool);
  bcir_aligned_free64(m->pool_mem);
  bcir_opt_destroy(m->opt);
  bcir_aligned_free64(m->weights); bcir_aligned_free64(m->grad_tile); ext_dealloc(m->tensors);
  for (uint32_t l = 0; l < m->sp.layers; ++l) {
    layer_act *a = &m->acts[l];
    bcir_aligned_free64(a->mhc_x1); bcir_aligned_free64(a->mhc_x2);
    bcir_aligned_free64(a->mix_out); bcir_aligned_free64(a->mlp_out);
    bcir_aligned_free64(a->x_in);
    if (a->n1_raw != a->n1) bcir_aligned_free64(a->n1_raw);
    bcir_aligned_free64(a->n1);
    bcir_aligned_free64(a->q_raw); bcir_aligned_free64(a->k_raw);
    bcir_aligned_free64(a->q); bcir_aligned_free64(a->k); bcir_aligned_free64(a->v); bcir_aligned_free64(a->p);
    bcir_aligned_free64(a->lse);
    bcir_aligned_free64(a->q_inv); bcir_aligned_free64(a->k_inv);
    bcir_aligned_free64(a->a_mid); bcir_aligned_free64(a->a_pre); bcir_aligned_free64(a->alpha);
    bcir_aligned_free64(a->b_pre); bcir_aligned_free64(a->beta);
    bcir_aligned_free64(a->e_mid); bcir_aligned_free64(a->e_pre); bcir_aligned_free64(a->erase);
    bcir_aligned_free64(a->w_mid); bcir_aligned_free64(a->w_pre); bcir_aligned_free64(a->write);
    bcir_aligned_free64(a->cd_y);
    bcir_aligned_free64(a->dn_sprev); bcir_aligned_free64(a->dn_W);
    bcir_aligned_free64(a->q_c_raw); bcir_aligned_free64(a->q_c); bcir_aligned_free64(a->q_r);
    bcir_aligned_free64(a->c_raw); bcir_aligned_free64(a->c_hat);
    bcir_aligned_free64(a->k_c); bcir_aligned_free64(a->v_c); bcir_aligned_free64(a->k_r); bcir_aligned_free64(a->mla_p);
    if (a->ctx_raw != a->ctx) bcir_aligned_free64(a->ctx_raw);
    bcir_aligned_free64(a->g_mid); bcir_aligned_free64(a->g_val);
    bcir_aligned_free64(a->ctx); bcir_aligned_free64(a->x2);
    if (a->n2_raw != a->n2) bcir_aligned_free64(a->n2_raw);
    bcir_aligned_free64(a->n2);
    bcir_aligned_free64(a->gate); bcir_aligned_free64(a->up); bcir_aligned_free64(a->hidden);
  }
  bcir_aligned_free64(m->mhc_xfinal);
  bcir_aligned_free64(m->x_final); bcir_aligned_free64(m->norm_final); bcir_aligned_free64(m->logits);
  bcir_aligned_free64(m->rope_hd); bcir_aligned_free64(m->rope_dr);
  bcir_aligned_free64(m->d_mhc_cur); bcir_aligned_free64(m->d_mhc_next);
  bcir_aligned_free64(m->d_cur); bcir_aligned_free64(m->d_next);
  bcir_aligned_free64(m->d_n); bcir_aligned_free64(m->d_n_tmp); bcir_aligned_free64(m->d_ctx);
  bcir_aligned_free64(m->d_gate); bcir_aligned_free64(m->d_up); bcir_aligned_free64(m->d_hid);
  bcir_aligned_free64(m->d_logits); bcir_aligned_free64(m->rms_inv);
  bcir_aligned_free64(m->d_q); bcir_aligned_free64(m->d_k); bcir_aligned_free64(m->d_v);
  bcir_aligned_free64(m->d_a); bcir_aligned_free64(m->d_b);
  bcir_aligned_free64(m->d_e); bcir_aligned_free64(m->d_w);
  bcir_aligned_free64(m->d_mid); bcir_aligned_free64(m->d_gpre);
  bcir_aligned_free64(m->d_qc); bcir_aligned_free64(m->d_qr);
  bcir_aligned_free64(m->d_chat); bcir_aligned_free64(m->d_craw);
  bcir_aligned_free64(m->d_kc); bcir_aligned_free64(m->d_vc); bcir_aligned_free64(m->d_kr);
  bcir_aligned_free64(m->rms_dg64);
  bcir_aligned_free64(m->attn_max_log); bcir_aligned_free64(m->attn_promos);
  bcir_aligned_free64(m->attn_sink_sum); bcir_aligned_free64(m->attn_scores);
  bcir_aligned_free64(m->attn_dk_loc); bcir_aligned_free64(m->attn_dv_loc);
  bcir_aligned_free64(m->attn_dkr_loc);
  bcir_aligned_free64(m->dn_work_S); bcir_aligned_free64(m->dn_work_dS);
  bcir_aligned_free64(m->dn_work_g); bcir_aligned_free64(m->dn_work_gam);
  bcir_aligned_free64(m->dn_work_A); bcir_aligned_free64(m->dn_work_qS);
  bcir_aligned_free64(m->dn_work_Shist); bcir_aligned_free64(m->dn_work_dP);
  bcir_aligned_free64(m->dn_work_dW); bcir_aligned_free64(m->dn_work_dA);
  bcir_aligned_free64(m->dn_work_dg);
  bcir_aligned_free64(m->gemm_scratch);
  ext_dealloc(m);
}

void bcir_ext_get_mem_stats(const bcir_ext_model *m, bcir_ext_mem_stats *out) { if (m && out) *out = m->mem; }
void bcir_ext_get_telemetry(const bcir_ext_model *m, bcir_prec_telemetry *out) { if (m && out) *out = m->tel; }
void bcir_ext_reset_telemetry(bcir_ext_model *m) { if (m) bcir_prec_telemetry_reset(&m->tel); }
void bcir_ext_reset_opt(bcir_ext_model *m) { if (m && m->opt) bcir_opt_reset(m->opt); }
size_t bcir_ext_opt_state_bytes(const bcir_ext_model *m) {
  return (m && m->opt) ? bcir_opt_state_bytes(m->opt) : 0u;
}
int bcir_ext_save_opt_state(const bcir_ext_model *m, void *dst, size_t dst_bytes) {
  return (m && m->opt) ? bcir_opt_save_state(m->opt, dst, dst_bytes) : 1;
}
int bcir_ext_load_opt_state(bcir_ext_model *m, const void *src, size_t src_bytes) {
  return (m && m->opt) ? bcir_opt_load_state(m->opt, src, src_bytes) : 1;
}
float *bcir_ext_weights_ptr(bcir_ext_model *m) { return m ? m->weights : NULL; }
float *bcir_ext_grads_fp32_ptr(bcir_ext_model *m) { return m ? m->opt->grad_fp32 : NULL; }
size_t bcir_ext_param_count(const bcir_ext_model *m) { return m ? m->n_params : 0u; }
size_t bcir_ext_tensor_count(const bcir_ext_model *m) { return m ? m->n_tensors : 0u; }
int bcir_ext_tensor_info_at(const bcir_ext_model *m, size_t idx, bcir_ext_tensor_info *out) {
  if (!m || !out || idx >= m->n_tensors) return 1;
  *out = m->tensors[idx];
  return 0;
}

/* ==================== Linear Dispatch (Region Policy + Pair Fusion) ==================== */
static void ext_linear_fwd(bcir_ext_model *m, int region, size_t rows, size_t in, size_t out,
                           const float *x, const float *w, float *y) {
  int gk = (region == 1) ? m->pol.attn_fwd_gemm : (region == 2 ? m->pol.mlp_fwd_gemm : BCIR_GEMM_FP32_KBLOCK);
  if (region == 1) m->tel.gemm_calls_attn++; else if (region == 2) m->tel.gemm_calls_mlp++;
  bcir_prec_mm_pool(m->pool, gk, m->pol.use_vdp_portable, 0, 1, rows, out, in,
                    1.0f, x, w, 0.0f, y, m->gemm_scratch, m->gemm_scratch_bytes);
}

static void ext_linear_bwd(bcir_ext_model *m, int region, size_t rows, size_t in, size_t out,
                           const float *x, const float *w, const float *dy,
                           float *dx, float *dw, float dx_beta, float dw_beta) {
  int gk = (region == 1) ? m->pol.attn_bwd_gemm : (region == 2 ? m->pol.mlp_bwd_gemm : BCIR_GEMM_FP32_KBLOCK);
  if (region == 1) m->tel.gemm_calls_attn += 2u; else if (region == 2) m->tel.gemm_calls_mlp += 2u;
  if (gk == BCIR_GEMM_FP32_KBLOCK) {
    bcir_tensor_mm_call cx = { 0, 0, rows, in, out, 1.0f, dy, w, dx_beta, dx };
    bcir_tensor_mm_call cw = { 1, 0, out, in, rows, 1.0f, dy, x, dw_beta, dw };
    bcir_tensor_mm_pair_pool(m->pool, 2, &cx, &cw);
  } else {
    bcir_prec_mm_pool(m->pool, gk, m->pol.use_vdp_portable, 0, 0, rows, in, out,
                      1.0f, dy, w, dx_beta, dx, m->gemm_scratch, m->gemm_scratch_bytes);
    bcir_prec_mm_pool(m->pool, gk, m->pol.use_vdp_portable, 1, 0, out, in, rows,
                      1.0f, dy, x, dw_beta, dw, m->gemm_scratch, m->gemm_scratch_bytes);
  }
}

/* ==================== RMSNorm (Rule P3 FP64 Guard, Zero Hot Malloc) ==================== */
static void ext_rms_fwd(bcir_ext_model *m, size_t rows, size_t width, const float *x, const float *g, float *y) {
  if (m->pol.fp64_guard_mode == BCIR_FP64_GUARD_OFF) {
    bcir_tensor_rms_pool(m->pool, rows, width, (float)m->sp.rms_eps, x, g, y);
    return;
  }
  double eps = m->sp.rms_eps;
  int force64 = (m->pol.fp64_guard_mode == BCIR_FP64_GUARD_ALWAYS);
  double min_r = m->tel.min_rms_observed;
  uint64_t promos = 0u;
  for (size_t r = 0; r < rows; ++r) {
    const float *xr = x + r * width;
    float *yr = y + r * width;
    double ss = 0.0;
    for (size_t c = 0; c < width; ++c) { double v = (double)xr[c]; ss += v * v; }
    double rms = sqrt(ss / (double)width + eps);
    if (rms < min_r) min_r = rms;
    if (force64 || rms < m->pol.rmsnorm_cond_thresh) {
      promos++;
      double inv = 1.0 / rms;
      for (size_t c = 0; c < width; ++c) yr[c] = (float)((double)xr[c] * inv * (double)g[c]);
    } else {
      float inv = 1.0f / (float)rms;
      for (size_t c = 0; c < width; ++c) yr[c] = (xr[c] * inv) * g[c];
    }
  }
  m->tel.min_rms_observed = min_r;
  m->tel.fp64_norm_promotions += promos;
}

static void ext_rms_bwd(bcir_ext_model *m, size_t rows, size_t width, const float *x, const float *g,
                        const float *dy, float *dx, float *dg, int dx_add) {
  if (m->pol.fp64_guard_mode == BCIR_FP64_GUARD_OFF) {
    unsigned how = dx_add ? BCIR_TENSOR_RMSB_ADD : 0u;
    bcir_tensor_rms_backward_into_pool(m->pool, rows, width, (float)m->sp.rms_eps, x, g, dy, dx, dg, m->rms_inv, how);
    return;
  }
  double eps = m->sp.rms_eps;
  double *dg64 = m->rms_dg64;
  memset(dg64, 0, width * sizeof(double));
  for (size_t r = 0; r < rows; ++r) {
    const float *xr = x + r * width, *dyr = dy + r * width;
    float *dxr = dx + r * width;
    double ss = 0.0;
    for (size_t c = 0; c < width; ++c) { double v = (double)xr[c]; ss += v * v; }
    double inv = 1.0 / sqrt(ss / (double)width + eps);
    double dot = 0.0;
    for (size_t c = 0; c < width; ++c) {
      double gdy = (double)g[c] * (double)dyr[c];
      dot += gdy * (double)xr[c];
      dg64[c] += (double)dyr[c] * ((double)xr[c] * inv);
    }
    double coeff = (dot * inv * inv) / (double)width;
    for (size_t c = 0; c < width; ++c) {
      double val = inv * ((double)g[c] * (double)dyr[c] - (double)xr[c] * coeff);
      dxr[c] = dx_add ? (float)((double)dxr[c] + val) : (float)val;
    }
  }
  for (size_t c = 0; c < width; ++c) dg[c] = (float)((double)dg[c] + dg64[c]);
}

/* ==================== SWA & Materialized/Online GQA Attention ==================== */
typedef struct swa_fwd_ctx {
  size_t B, T, H, KH, dh, W, score_stride;
  const float *q, *k, *v;
  float *p, *y;
  int fp64_mode;
  double logit_thresh;
  double *scores_buf, *max_logits, *sink_sum;
  uint64_t *promos;
} swa_fwd_ctx;

static void swa_fwd_worker(void *vctx, size_t task) {
  swa_fwd_ctx *c = (swa_fwd_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H, kvh = h / (c->H / c->KH);
  size_t T = c->T, dh = c->dh, W = c->W;
  double scale = 1.0 / sqrt((double)dh);
  double max_obs = 0.0, sink = 0.0; uint64_t pr = 0u;
  double *scores = c->scores_buf + task * c->score_stride;
  for (size_t i = 0; i < T; ++i) {
    size_t j0 = (i + 1u > W) ? (i + 1u - W) : 0u;
    size_t len = i + 1u - j0;
    const float *qi = c->q + ((b * T + i) * c->H + h) * dh;
    double mx = -1e30;
    for (size_t u = 0; u < len; ++u) {
      size_t j = j0 + u;
      const float *kj = c->k + ((b * T + j) * c->KH + kvh) * dh;
      double s = 0.0;
      for (size_t d = 0; d < dh; ++d) s += (double)qi[d] * (double)kj[d];
      s *= scale;
      scores[u] = s;
      if (s > mx) mx = s;
      if (fabs(s) > max_obs) max_obs = fabs(s);
    }
    if (c->fp64_mode == BCIR_FP64_GUARD_ALWAYS || (c->fp64_mode == BCIR_FP64_GUARD_CONDITIONAL && fabs(mx) > c->logit_thresh)) pr++;
    double sum = 0.0;
    for (size_t u = 0; u < len; ++u) { scores[u] = exp(scores[u] - mx); sum += scores[u]; }
    double inv = 1.0 / sum;
    float *pi = c->p + ((b * c->H + h) * T + i) * W;
    for (size_t u = 0; u < len; ++u) pi[u] = (float)(scores[u] * inv);
    for (size_t u = len; u < W; ++u) pi[u] = 0.0f;
    if (j0 == 0u) sink += (double)pi[0];
    float *yi = c->y + ((b * T + i) * c->H + h) * dh;
    for (size_t d = 0; d < dh; ++d) {
      double acc = 0.0;
      for (size_t u = 0; u < len; ++u) {
        size_t j = j0 + u;
        const float *vj = c->v + ((b * T + j) * c->KH + kvh) * dh;
        acc += (double)pi[u] * (double)vj[d];
      }
      yi[d] = (float)acc;
    }
  }
  c->max_logits[task] = max_obs;
  c->promos[task] = pr;
  c->sink_sum[task] = sink / (double)T;
}

typedef struct swa_bwd_ctx {
  size_t B, T, H, KH, dh, W, score_stride;
  const float *q, *k, *v, *p, *dy;
  float *dq, *dk, *dv;
  double *scores_buf, *dk_loc, *dv_loc;
} swa_bwd_ctx;

static void swa_bwd_dq_worker(void *vctx, size_t task) {
  swa_bwd_ctx *c = (swa_bwd_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H, kvh = h / (c->H / c->KH);
  size_t T = c->T, dh = c->dh, W = c->W;
  double scale = 1.0 / sqrt((double)dh);
  double *dp = c->scores_buf + task * c->score_stride;
  for (size_t i = 0; i < T; ++i) {
    size_t j0 = (i + 1u > W) ? (i + 1u - W) : 0u;
    size_t len = i + 1u - j0;
    const float *dyi = c->dy + ((b * T + i) * c->H + h) * dh;
    const float *pi = c->p + ((b * c->H + h) * T + i) * W;
    double dot = 0.0;
    for (size_t u = 0; u < len; ++u) {
      size_t j = j0 + u;
      const float *vj = c->v + ((b * T + j) * c->KH + kvh) * dh;
      double s = 0.0;
      for (size_t d = 0; d < dh; ++d) s += (double)dyi[d] * (double)vj[d];
      dp[u] = s; dot += (double)pi[u] * s;
    }
    float *dqi = c->dq + ((b * T + i) * c->H + h) * dh;
    for (size_t d = 0; d < dh; ++d) {
      double acc = 0.0;
      for (size_t u = 0; u < len; ++u) {
        size_t j = j0 + u;
        const float *kj = c->k + ((b * T + j) * c->KH + kvh) * dh;
        double ds = (double)pi[u] * (dp[u] - dot) * scale;
        acc += ds * (double)kj[d];
      }
      dqi[d] = (float)acc;
    }
  }
}

static void swa_bwd_dkv_worker(void *vctx, size_t task) {
  swa_bwd_ctx *c = (swa_bwd_ctx *)vctx;
  size_t b = task / c->KH, kvh = task % c->KH, grp = c->H / c->KH;
  size_t T = c->T, dh = c->dh, W = c->W;
  double scale = 1.0 / sqrt((double)dh);
  double *dk_loc = c->dk_loc + task * T * dh;
  double *dv_loc = c->dv_loc + task * T * dh;
  double *dp = c->scores_buf + task * c->score_stride;
  memset(dk_loc, 0, T * dh * sizeof(double));
  memset(dv_loc, 0, T * dh * sizeof(double));
  for (size_t g = 0; g < grp; ++g) {
    size_t h = kvh * grp + g;
    for (size_t i = 0; i < T; ++i) {
      size_t j0 = (i + 1u > W) ? (i + 1u - W) : 0u;
      size_t len = i + 1u - j0;
      const float *qi = c->q + ((b * T + i) * c->H + h) * dh;
      const float *dyi = c->dy + ((b * T + i) * c->H + h) * dh;
      const float *pi = c->p + ((b * c->H + h) * T + i) * W;
      double dot = 0.0;
      for (size_t u = 0; u < len; ++u) {
        size_t j = j0 + u;
        const float *vj = c->v + ((b * T + j) * c->KH + kvh) * dh;
        double s = 0.0;
        for (size_t d = 0; d < dh; ++d) s += (double)dyi[d] * (double)vj[d];
        dp[u] = s; dot += (double)pi[u] * s;
      }
      for (size_t u = 0; u < len; ++u) {
        size_t j = j0 + u;
        double pv = (double)pi[u];
        double ds = pv * (dp[u] - dot) * scale;
        double *dkj = dk_loc + j * dh, *dvj = dv_loc + j * dh;
        for (size_t d = 0; d < dh; ++d) {
          dkj[d] += ds * (double)qi[d];
          dvj[d] += pv * (double)dyi[d];
        }
      }
    }
  }
  for (size_t j = 0; j < T; ++j) {
    float *dkj = c->dk + ((b * T + j) * c->KH + kvh) * dh;
    float *dvj = c->dv + ((b * T + j) * c->KH + kvh) * dh;
    for (size_t d = 0; d < dh; ++d) { dkj[d] = (float)dk_loc[j * dh + d]; dvj[d] = (float)dv_loc[j * dh + d]; }
  }
}

/* ==================== TRAIN-OPT-5 Online-Softmax O(T) Tiled GQA ==================== */
typedef struct gqa_online_ctx {
  size_t B, T, H, KH, dh;
  const float *q, *k, *v, *y_in, *dy;
  float *y_out, *dq, *dk, *dv;
  double *lse;
  double *max_logits, *sink_sum;
  uint64_t *promos;
  int fp64_mode;
  double logit_thresh;
  double *dk_loc, *dv_loc;
} gqa_online_ctx;

#define ONLINE_TILE_K 32u

static void gqa_online_fwd_worker(void *vctx, size_t task) {
  gqa_online_ctx *c = (gqa_online_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H, kvh = h / (c->H / c->KH);
  size_t T = c->T, dh = c->dh;
  double scale = 1.0 / sqrt((double)dh);
  double max_obs = 0.0, sink = 0.0; uint64_t pr = 0u;
  double acc[256]; /* dh <= 256; fallback handled cleanly */
  double *acc_p = (dh <= 256u) ? acc : (c->dk_loc + task * T * dh);
  double s_tile[ONLINE_TILE_K];
  for (size_t i = 0; i < T; ++i) {
    const float *qi = c->q + ((b * T + i) * c->H + h) * dh;
    double m_run = -1e30, l_run = 0.0, s0 = 0.0;
    for (size_t d = 0; d < dh; ++d) acc_p[d] = 0.0;
    for (size_t j0 = 0; j0 <= i; j0 += ONLINE_TILE_K) {
      size_t j1 = (j0 + ONLINE_TILE_K <= i + 1u) ? (j0 + ONLINE_TILE_K) : (i + 1u);
      double m_tile = -1e30;
      for (size_t j = j0; j < j1; ++j) {
        const float *kj = c->k + ((b * T + j) * c->KH + kvh) * dh;
        double s = 0.0;
        for (size_t d = 0; d < dh; ++d) s += (double)qi[d] * (double)kj[d];
        s *= scale;
        s_tile[j - j0] = s;
        if (j == 0u) s0 = s;
        if (s > m_tile) m_tile = s;
        if (fabs(s) > max_obs) max_obs = fabs(s);
      }
      double m_new = (m_run > m_tile) ? m_run : m_tile;
      double alpha = (l_run > 0.0) ? exp(m_run - m_new) : 0.0;
      l_run *= alpha;
      for (size_t d = 0; d < dh; ++d) acc_p[d] *= alpha;
      for (size_t j = j0; j < j1; ++j) {
        double p = exp(s_tile[j - j0] - m_new);
        l_run += p;
        const float *vj = c->v + ((b * T + j) * c->KH + kvh) * dh;
        for (size_t d = 0; d < dh; ++d) acc_p[d] += p * (double)vj[d];
      }
      m_run = m_new;
    }
    if (c->fp64_mode == BCIR_FP64_GUARD_ALWAYS || (c->fp64_mode == BCIR_FP64_GUARD_CONDITIONAL && fabs(m_run) > c->logit_thresh)) pr++;
    double lse_i = m_run + log(l_run);
    c->lse[(b * c->H + h) * T + i] = lse_i;
    sink += exp(s0 - lse_i);
    double inv_l = 1.0 / l_run;
    float *yi = c->y_out + ((b * T + i) * c->H + h) * dh;
    for (size_t d = 0; d < dh; ++d) yi[d] = (float)(acc_p[d] * inv_l);
  }
  c->max_logits[task] = max_obs;
  c->promos[task] = pr;
  c->sink_sum[task] = sink / (double)T;
}

static void gqa_online_bwd_dq_worker(void *vctx, size_t task) {
  gqa_online_ctx *c = (gqa_online_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H, kvh = h / (c->H / c->KH);
  size_t T = c->T, dh = c->dh;
  double scale = 1.0 / sqrt((double)dh);
  double dq_acc[256];
  double *dq_p = (dh <= 256u) ? dq_acc : (c->dk_loc + task * T * dh);
  for (size_t i = 0; i < T; ++i) {
    const float *qi = c->q + ((b * T + i) * c->H + h) * dh;
    const float *yi = c->y_in + ((b * T + i) * c->H + h) * dh;
    const float *dyi = c->dy + ((b * T + i) * c->H + h) * dh;
    double lse_i = c->lse[(b * c->H + h) * T + i];
    double dot = 0.0;
    for (size_t d = 0; d < dh; ++d) { dot += (double)dyi[d] * (double)yi[d]; dq_p[d] = 0.0; }
    for (size_t j = 0; j <= i; ++j) {
      const float *kj = c->k + ((b * T + j) * c->KH + kvh) * dh;
      const float *vj = c->v + ((b * T + j) * c->KH + kvh) * dh;
      double s = 0.0, dp = 0.0;
      for (size_t d = 0; d < dh; ++d) {
        s += (double)qi[d] * (double)kj[d];
        dp += (double)dyi[d] * (double)vj[d];
      }
      double p = exp(s * scale - lse_i);
      double ds = p * (dp - dot) * scale;
      for (size_t d = 0; d < dh; ++d) dq_p[d] += ds * (double)kj[d];
    }
    float *dqi = c->dq + ((b * T + i) * c->H + h) * dh;
    for (size_t d = 0; d < dh; ++d) dqi[d] = (float)dq_p[d];
  }
}

static void gqa_online_bwd_dkv_worker(void *vctx, size_t task) {
  gqa_online_ctx *c = (gqa_online_ctx *)vctx;
  size_t b = task / c->KH, kvh = task % c->KH, grp = c->H / c->KH;
  size_t T = c->T, dh = c->dh;
  double scale = 1.0 / sqrt((double)dh);
  double *dk_loc = c->dk_loc + task * T * dh;
  double *dv_loc = c->dv_loc + task * T * dh;
  memset(dk_loc, 0, T * dh * sizeof(double));
  memset(dv_loc, 0, T * dh * sizeof(double));
  for (size_t g = 0; g < grp; ++g) {
    size_t h = kvh * grp + g;
    for (size_t i = 0; i < T; ++i) {
      const float *qi = c->q + ((b * T + i) * c->H + h) * dh;
      const float *yi = c->y_in + ((b * T + i) * c->H + h) * dh;
      const float *dyi = c->dy + ((b * T + i) * c->H + h) * dh;
      double lse_i = c->lse[(b * c->H + h) * T + i];
      double dot = 0.0;
      for (size_t d = 0; d < dh; ++d) dot += (double)dyi[d] * (double)yi[d];
      for (size_t j = 0; j <= i; ++j) {
        const float *kj = c->k + ((b * T + j) * c->KH + kvh) * dh;
        const float *vj = c->v + ((b * T + j) * c->KH + kvh) * dh;
        double s = 0.0, dp = 0.0;
        for (size_t d = 0; d < dh; ++d) {
          s += (double)qi[d] * (double)kj[d];
          dp += (double)dyi[d] * (double)vj[d];
        }
        double p = exp(s * scale - lse_i);
        double ds = p * (dp - dot) * scale;
        double *dkj = dk_loc + j * dh, *dvj = dv_loc + j * dh;
        for (size_t d = 0; d < dh; ++d) {
          dkj[d] += ds * (double)qi[d];
          dvj[d] += p * (double)dyi[d];
        }
      }
    }
  }
  for (size_t j = 0; j < T; ++j) {
    float *dkj = c->dk + ((b * T + j) * c->KH + kvh) * dh;
    float *dvj = c->dv + ((b * T + j) * c->KH + kvh) * dh;
    for (size_t d = 0; d < dh; ++d) { dkj[d] = (float)dk_loc[j * dh + d]; dvj[d] = (float)dv_loc[j * dh + d]; }
  }
}

static void gqa_fwd_dispatch(bcir_ext_model *m, size_t B, size_t T, const float *q, const float *k,
                             const float *v, float *p, double *lse, float *y, int *bad) {
  size_t nt = B * m->sp.heads, dh = m->sp.width / m->sp.heads;
  if (m->sp.online_softmax) {
    gqa_online_ctx oc = { B, T, m->sp.heads, m->sp.kvheads, dh, q, k, v, NULL, NULL,
                          y, NULL, NULL, NULL, lse, m->attn_max_log, m->attn_sink_sum, m->attn_promos,
                          m->pol.fp64_guard_mode, m->pol.softmax_logit_thresh, m->attn_dk_loc, m->attn_dv_loc };
    bcir_tensor_pool_run(m->pool, gqa_online_fwd_worker, &oc, nt);
    double ssum = 0.0;
    for (size_t i = 0; i < nt; ++i) {
      if (m->attn_max_log[i] > m->tel.max_softmax_logit) m->tel.max_softmax_logit = m->attn_max_log[i];
      m->tel.fp64_softmax_promotions += m->attn_promos[i];
      ssum += m->attn_sink_sum[i];
    }
    m->tel.first_tok_attn_mean = ssum / (double)nt;
    return;
  }
  if (m->pol.fp64_guard_mode == BCIR_FP64_GUARD_OFF) {
    bcir_tensor_attention_contract_variant_scan_ws_pool(
        m->pool, 0, BCIR_TENSOR_ATTENTION_FUSED, B, T, m->sp.heads, m->sp.kvheads,
        dh, q, k, v, p, y, bad, NULL, 0u);
    double ssum = 0.0;
    for (size_t bh = 0; bh < nt; ++bh) {
      double s = 0.0;
      for (size_t i = 0; i < T; ++i) s += (double)p[(bh * T + i) * T];
      ssum += s / (double)T;
    }
    m->tel.first_tok_attn_mean = ssum / (double)nt;
    return;
  }
  size_t stride = max_sz(m->sp.time, m->sp.swa_window);
  swa_fwd_ctx sc = { B, T, m->sp.heads, m->sp.kvheads, dh, T, stride,
                     q, k, v, p, y, m->pol.fp64_guard_mode, m->pol.softmax_logit_thresh,
                     m->attn_scores, m->attn_max_log, m->attn_sink_sum, m->attn_promos };
  bcir_tensor_pool_run(m->pool, swa_fwd_worker, &sc, nt);
  double ssum = 0.0;
  for (size_t i = 0; i < nt; ++i) {
    if (m->attn_max_log[i] > m->tel.max_softmax_logit) m->tel.max_softmax_logit = m->attn_max_log[i];
    m->tel.fp64_softmax_promotions += m->attn_promos[i];
    ssum += m->attn_sink_sum[i];
  }
  m->tel.first_tok_attn_mean = ssum / (double)nt;
}

static void gqa_bwd_dispatch(bcir_ext_model *m, size_t B, size_t T, const float *q, const float *k,
                             const float *v, float *p, double *lse, const float *y_in,
                             const float *dy, float *dq, float *dk, float *dv) {
  size_t dh = m->sp.width / m->sp.heads;
  if (m->sp.online_softmax) {
    gqa_online_ctx oc = { B, T, m->sp.heads, m->sp.kvheads, dh, q, k, v, y_in, dy,
                          NULL, dq, dk, dv, lse, NULL, NULL, NULL, 0, 0.0,
                          m->attn_dk_loc, m->attn_dv_loc };
    bcir_tensor_pool_run(m->pool, gqa_online_bwd_dq_worker, &oc, B * m->sp.heads);
    bcir_tensor_pool_run(m->pool, gqa_online_bwd_dkv_worker, &oc, B * m->sp.kvheads);
    return;
  }
  if (m->pol.fp64_guard_mode == BCIR_FP64_GUARD_OFF) {
    bcir_tensor_attention_backward_contract_variant_ws_pool(
        m->pool, 0, BCIR_TENSOR_ATTENTION_FUSED, B, T, m->sp.heads, m->sp.kvheads,
        dh, q, k, v, p, dy, dq, dk, dv, NULL, 0u);
    return;
  }
  size_t stride = max_sz(m->sp.time, m->sp.swa_window);
  swa_bwd_ctx sc = { B, T, m->sp.heads, m->sp.kvheads, dh, T, stride,
                     q, k, v, p, dy, dq, dk, dv, m->attn_scores, m->attn_dk_loc, m->attn_dv_loc };
  bcir_tensor_pool_run(m->pool, swa_bwd_dq_worker, &sc, B * m->sp.heads);
  bcir_tensor_pool_run(m->pool, swa_bwd_dkv_worker, &sc, B * m->sp.kvheads);
}

/* ==================== Arch-1 Gated DeltaNet (Scalar Alpha/Beta WY FP64) ==================== */
static void dn_l2norm_fwd(size_t N, size_t H, size_t dh, const float *x, float *y, float *inv_out) {
  for (size_t i = 0; i < N * H; ++i) {
    const float *xi = x + i * dh; float *yi = y + i * dh;
    double ss = 1e-6;
    for (size_t d = 0; d < dh; ++d) ss += (double)xi[d] * (double)xi[d];
    double inv = 1.0 / sqrt(ss);
    inv_out[i] = (float)inv;
    for (size_t d = 0; d < dh; ++d) yi[d] = (float)((double)xi[d] * inv);
  }
}

static void dn_l2norm_bwd(size_t N, size_t H, size_t dh, const float *y, const float *inv_in,
                          const float *dy, float *dx) {
  for (size_t i = 0; i < N * H; ++i) {
    const float *yi = y + i * dh, *dyi = dy + i * dh; float *dxi = dx + i * dh;
    double inv = (double)inv_in[i], dot = 0.0;
    for (size_t d = 0; d < dh; ++d) dot += (double)dyi[d] * (double)yi[d];
    for (size_t d = 0; d < dh; ++d) dxi[d] = (float)(inv * ((double)dyi[d] - dot * (double)yi[d]));
  }
}

typedef struct dn_fwd_ctx {
  size_t B, T, H, dh, C, nc;
  const float *q, *k, *v, *alpha, *beta;
  double *sprev_all, *W_all;
  float *y;
  double *work_S, *work_g, *work_gam, *work_A, *work_qS;
  double *max_off;
  uint64_t *wy_cnt;
} dn_fwd_ctx;

static void dn_wy_fwd_worker(void *vctx, size_t task) {
  dn_fwd_ctx *c = (dn_fwd_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, C = c->C, nc = c->nc;
  double *S = c->work_S + task * dh * dh;
  double *g = c->work_g + task * C * dh;
  double *gam = c->work_gam + task * C * dh;
  double *A = c->work_A + task * C * C;
  double *qS = c->work_qS + task * dh;
  memset(S, 0, dh * dh * sizeof(double));
  double max_a = 0.0; uint64_t solves = 0u;
  for (size_t ci = 0; ci < nc; ++ci) {
    size_t t0 = ci * C, Lc = (t0 + C <= T) ? C : (T - t0);
    double *Sp = c->sprev_all + (task * nc + ci) * dh * dh;
    memcpy(Sp, S, dh * dh * sizeof(double));
    double gacc = 0.0;
    for (size_t i = 0; i < Lc; ++i) {
      double a = (double)c->alpha[(b * T + t0 + i) * H + h];
      gacc += log(a > 1e-12 ? a : 1e-12);
      g[i] = gacc; gam[i] = exp(gacc);
    }
    for (size_t i = 0; i < Lc; ++i) {
      const float *ki = c->k + ((b * T + t0 + i) * H + h) * dh;
      double bi = (double)c->beta[(b * T + t0 + i) * H + h];
      for (size_t j = 0; j < i; ++j) {
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        double kk = 0.0;
        for (size_t d = 0; d < dh; ++d) kk += (double)ki[d] * (double)kj[d];
        double aij = bi * exp(g[i] - g[j]) * kk;
        A[i * C + j] = aij;
        if (fabs(aij) > max_a) max_a = fabs(aij);
      }
      double *Wi = c->W_all + (task * T + t0 + i) * dh;
      const float *vi = c->v + ((b * T + t0 + i) * H + h) * dh;
      for (size_t d = 0; d < dh; ++d) {
        double kS = 0.0;
        for (size_t a = 0; a < dh; ++a) kS += (double)ki[a] * Sp[a * dh + d];
        double val = bi * (double)vi[d] - bi * gam[i] * kS;
        for (size_t j = 0; j < i; ++j) val -= A[i * C + j] * c->W_all[(task * T + t0 + j) * dh + d];
        Wi[d] = val;
      }
    }
    solves++;
    for (size_t i = 0; i < Lc; ++i) {
      const float *qi = c->q + ((b * T + t0 + i) * H + h) * dh;
      for (size_t d = 0; d < dh; ++d) {
        double acc = 0.0;
        for (size_t a = 0; a < dh; ++a) acc += (double)qi[a] * Sp[a * dh + d];
        qS[d] = gam[i] * acc;
      }
      for (size_t j = 0; j <= i; ++j) {
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        double qk = 0.0;
        for (size_t a = 0; a < dh; ++a) qk += (double)qi[a] * (double)kj[a];
        double fac = exp(g[i] - g[j]) * qk;
        const double *Wj = c->W_all + (task * T + t0 + j) * dh;
        for (size_t d = 0; d < dh; ++d) qS[d] += fac * Wj[d];
      }
      float *yi = c->y + ((b * T + t0 + i) * H + h) * dh;
      for (size_t d = 0; d < dh; ++d) yi[d] = (float)qS[d];
    }
    double glast = g[Lc - 1u], gam_last = gam[Lc - 1u];
    for (size_t a = 0; a < dh * dh; ++a) S[a] = gam_last * Sp[a];
    for (size_t j = 0; j < Lc; ++j) {
      double fac = exp(glast - g[j]);
      const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
      const double *Wj = c->W_all + (task * T + t0 + j) * dh;
      for (size_t a = 0; a < dh; ++a) {
        double fk = fac * (double)kj[a];
        for (size_t d = 0; d < dh; ++d) S[a * dh + d] += fk * Wj[d];
      }
    }
  }
  c->max_off[task] = max_a;
  c->wy_cnt[task] = solves;
}

typedef struct dn_bwd_ctx {
  size_t B, T, H, dh, C, nc;
  const float *q, *k, *v, *alpha, *beta, *dy;
  const double *sprev_all, *W_all;
  float *dq, *dk, *dv, *da_pre, *db_pre;
  double *work_dS, *work_g, *work_gam, *work_A, *work_dW, *work_dA, *work_dg;
} dn_bwd_ctx;

static void dn_wy_bwd_worker(void *vctx, size_t task) {
  dn_bwd_ctx *c = (dn_bwd_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, C = c->C, nc = c->nc;
  double *dS = c->work_dS + task * dh * dh;
  double *g = c->work_g + task * C * dh;
  double *gam = c->work_gam + task * C * dh;
  double *A = c->work_A + task * C * C;
  double *dW = c->work_dW + task * C * dh;
  double *dA = c->work_dA + task * C * C;
  double *dg = c->work_dg + task * C;
  memset(dS, 0, dh * dh * sizeof(double));

  for (size_t ci = nc; ci-- > 0u; ) {
    size_t t0 = ci * C, Lc = (t0 + C <= T) ? C : (T - t0);
    const double *Sp = c->sprev_all + (task * nc + ci) * dh * dh;
    double gacc = 0.0;
    for (size_t i = 0; i < Lc; ++i) {
      double a = (double)c->alpha[(b * T + t0 + i) * H + h];
      gacc += log(a > 1e-12 ? a : 1e-12);
      g[i] = gacc; gam[i] = exp(gacc);
    }
    for (size_t i = 0; i < Lc; ++i) {
      const float *ki = c->k + ((b * T + t0 + i) * H + h) * dh;
      double bi = (double)c->beta[(b * T + t0 + i) * H + h];
      for (size_t j = 0; j < i; ++j) {
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        double kk = 0.0;
        for (size_t d = 0; d < dh; ++d) kk += (double)ki[d] * (double)kj[d];
        A[i * C + j] = bi * exp(g[i] - g[j]) * kk;
      }
    }
    memset(dW, 0, Lc * dh * sizeof(double));
    memset(dA, 0, C * C * sizeof(double));
    memset(dg, 0, Lc * sizeof(double));
    double glast = g[Lc - 1u], gam_last = gam[Lc - 1u];

    /* 1. Backprop through S_end -> Sp, g, k, W */
    double dg_last = 0.0;
    for (size_t a = 0; a < dh * dh; ++a) dg_last += dS[a] * gam_last * Sp[a];
    dg[Lc - 1u] += dg_last;
    for (size_t j = 0; j < Lc; ++j) {
      double fac = exp(glast - g[j]);
      const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
      const double *Wj = c->W_all + (task * T + t0 + j) * dh;
      float *dkj = c->dk + ((b * T + t0 + j) * H + h) * dh;
      double dot_kW = 0.0;
      for (size_t a = 0; a < dh; ++a) {
        double dka = 0.0;
        for (size_t d = 0; d < dh; ++d) {
          double v = dS[a * dh + d];
          dka += v * Wj[d];
          dW[j * dh + d] += fac * (double)kj[a] * v;
        }
        dkj[a] = (float)(fac * dka);
        dot_kW += (double)kj[a] * dka;
      }
      double df = fac * dot_kW;
      dg[Lc - 1u] += df; dg[j] -= df;
    }
    for (size_t a = 0; a < dh * dh; ++a) dS[a] *= gam_last;

    /* 2. Backprop through outputs y_i -> q_i, Sp, g, k, W */
    for (size_t i = 0; i < Lc; ++i) {
      const float *qi = c->q + ((b * T + t0 + i) * H + h) * dh;
      const float *dyi = c->dy + ((b * T + t0 + i) * H + h) * dh;
      float *dqi = c->dq + ((b * T + t0 + i) * H + h) * dh;
      double dq_acc[256];
      for (size_t a = 0; a < dh; ++a) dq_acc[a] = 0.0;
      double dgi_sp = 0.0;
      for (size_t a = 0; a < dh; ++a) {
        double s_dy = 0.0;
        for (size_t d = 0; d < dh; ++d) {
          s_dy += Sp[a * dh + d] * (double)dyi[d];
          dS[a * dh + d] += gam[i] * (double)qi[a] * (double)dyi[d];
        }
        dq_acc[a] += gam[i] * s_dy;
        dgi_sp += (double)qi[a] * s_dy;
      }
      dg[i] += gam[i] * dgi_sp;
      for (size_t j = 0; j <= i; ++j) {
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        const double *Wj = c->W_all + (task * T + t0 + j) * dh;
        double qk = 0.0, dyW = 0.0;
        for (size_t a = 0; a < dh; ++a) qk += (double)qi[a] * (double)kj[a];
        for (size_t d = 0; d < dh; ++d) dyW += (double)dyi[d] * Wj[d];
        double eij = exp(g[i] - g[j]);
        double fac = eij * qk;
        for (size_t d = 0; d < dh; ++d) dW[j * dh + d] += fac * (double)dyi[d];
        double dqk = eij * dyW;
        double df = fac * dyW;
        dg[i] += df; dg[j] -= df;
        float *dkj = c->dk + ((b * T + t0 + j) * H + h) * dh;
        for (size_t a = 0; a < dh; ++a) {
          dq_acc[a] += dqk * (double)kj[a];
          dkj[a] = (float)((double)dkj[a] + dqk * (double)qi[a]);
        }
      }
      for (size_t a = 0; a < dh; ++a) dqi[a] = (float)dq_acc[a];
    }

    /* 3. Reverse triangular solve through W_i = rhs_i - sum_{j<i} A[i,j] W_j */
    for (size_t i = Lc; i-- > 0u; ) {
      for (size_t j = 0; j < i; ++j) {
        double aij = A[i * C + j], da = 0.0;
        const double *Wj = c->W_all + (task * T + t0 + j) * dh;
        for (size_t d = 0; d < dh; ++d) {
          da -= dW[i * dh + d] * Wj[d];
          dW[j * dh + d] -= aij * dW[i * dh + d];
        }
        dA[i * C + j] += da;
      }
    }

    /* 4. Backprop through rhs_i and A[i,j] */
    for (size_t i = 0; i < Lc; ++i) {
      const float *ki = c->k + ((b * T + t0 + i) * H + h) * dh;
      const float *vi = c->v + ((b * T + t0 + i) * H + h) * dh;
      float *dki = c->dk + ((b * T + t0 + i) * H + h) * dh;
      float *dvi = c->dv + ((b * T + t0 + i) * H + h) * dh;
      double bi = (double)c->beta[(b * T + t0 + i) * H + h];
      double dbi = 0.0, dgi_rhs = 0.0;
      for (size_t d = 0; d < dh; ++d) {
        double dwi = dW[i * dh + d];
        double kS = 0.0;
        for (size_t a = 0; a < dh; ++a) kS += (double)ki[a] * Sp[a * dh + d];
        dvi[d] = (float)(bi * dwi);
        dbi += dwi * ((double)vi[d] - gam[i] * kS);
        dgi_rhs -= bi * gam[i] * dwi * kS;
        double coef = -bi * gam[i] * dwi;
        for (size_t a = 0; a < dh; ++a) {
          dki[a] = (float)((double)dki[a] + coef * Sp[a * dh + d]);
          dS[a * dh + d] += coef * (double)ki[a];
        }
      }
      dg[i] += dgi_rhs;
      for (size_t j = 0; j < i; ++j) {
        double da = dA[i * C + j];
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        float *dkj = c->dk + ((b * T + t0 + j) * H + h) * dh;
        double kk = 0.0;
        for (size_t a = 0; a < dh; ++a) kk += (double)ki[a] * (double)kj[a];
        double eij = exp(g[i] - g[j]);
        dbi += da * eij * kk;
        double df = da * bi * eij * kk;
        dg[i] += df; dg[j] -= df;
        double ckk = da * bi * eij;
        for (size_t a = 0; a < dh; ++a) {
          dki[a] = (float)((double)dki[a] + ckk * (double)kj[a]);
          dkj[a] = (float)((double)dkj[a] + ckk * (double)ki[a]);
        }
      }
      c->db_pre[(b * T + t0 + i) * H + h] = (float)(dbi * bi * (1.0 - bi));
    }

    /* 5. Backprop through cumulative log-decay g[i] -> alpha -> a_pre */
    double suffix = 0.0;
    for (size_t i = Lc; i-- > 0u; ) {
      suffix += dg[i];
      double ai = (double)c->alpha[(b * T + t0 + i) * H + h];
      c->da_pre[(b * T + t0 + i) * H + h] = (float)(suffix * (1.0 - ai));
    }
  }
}

/* ==================== Unified Channel-Wise WY: KDA (2510.26692) & GDN-2 (2605.22791) ==================== */
typedef struct cdelta_fwd_ctx {
  size_t B, T, H, dh, C, nc;
  int is_gdn2; /* 0 = KDA (channel alpha, scalar beta), 1 = GDN-2 (channel alpha, erase b, write w) */
  const float *q, *k, *v, *alpha, *beta, *erase, *write;
  double *sprev_all, *W_all;
  float *y;
  double *work_S, *work_g, *work_gam, *work_A, *work_qS;
  double *max_off;
  uint64_t *wy_cnt;
} cdelta_fwd_ctx;

static void cdelta_wy_fwd_worker(void *vctx, size_t task) {
  cdelta_fwd_ctx *c = (cdelta_fwd_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, C = c->C, nc = c->nc;
  double *S = c->work_S + task * dh * dh;
  double *g = c->work_g + task * C * dh;
  double *gam = c->work_gam + task * C * dh;
  double *A = c->work_A + task * C * C;
  double *qS = c->work_qS + task * dh;
  memset(S, 0, dh * dh * sizeof(double));
  double max_a = 0.0; uint64_t solves = 0u;

  for (size_t ci = 0; ci < nc; ++ci) {
    size_t t0 = ci * C, Lc = (t0 + C <= T) ? C : (T - t0);
    double *Sp = c->sprev_all + (task * nc + ci) * dh * dh;
    memcpy(Sp, S, dh * dh * sizeof(double));
    /* Cumulative channel-wise log-decay g[i, a] */
    for (size_t a = 0; a < dh; ++a) {
      double acc = 0.0;
      for (size_t i = 0; i < Lc; ++i) {
        double al = (double)c->alpha[((b * T + t0 + i) * H + h) * dh + a];
        acc += log(al > 1e-12 ? al : 1e-12);
        g[i * dh + a] = acc;
        gam[i * dh + a] = exp(acc);
      }
    }
    /* Chunkwise WY solve: A[i,j] = sum_a (b_i[a] * k_i[a] * exp(g[i,a]-g[j,a]) * k_j[a]) */
    for (size_t i = 0; i < Lc; ++i) {
      size_t ti = b * T + t0 + i;
      const float *ki = c->k + (ti * H + h) * dh;
      const float *vi = c->v + (ti * H + h) * dh;
      const float *bi_v = c->is_gdn2 ? (c->erase + (ti * H + h) * dh) : NULL;
      const float *wi_v = c->is_gdn2 ? (c->write + (ti * H + h) * dh) : NULL;
      double bi_s = c->is_gdn2 ? 0.0 : (double)c->beta[ti * H + h];
      const double *gi = g + i * dh, *gami = gam + i * dh;
      for (size_t j = 0; j < i; ++j) {
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        const double *gj = g + j * dh;
        double aij = 0.0;
        for (size_t a = 0; a < dh; ++a) {
          double ba = c->is_gdn2 ? (double)bi_v[a] : bi_s;
          aij += ba * (double)ki[a] * exp(gi[a] - gj[a]) * (double)kj[a];
        }
        A[i * C + j] = aij;
        if (fabs(aij) > max_a) max_a = fabs(aij);
      }
      double *Wi = c->W_all + (task * T + t0 + i) * dh;
      for (size_t d = 0; d < dh; ++d) {
        double kS = 0.0;
        for (size_t a = 0; a < dh; ++a) {
          double ba = c->is_gdn2 ? (double)bi_v[a] : bi_s;
          kS += ba * (double)ki[a] * gami[a] * Sp[a * dh + d];
        }
        double wd = c->is_gdn2 ? (double)wi_v[d] : bi_s;
        double val = wd * (double)vi[d] - kS;
        for (size_t j = 0; j < i; ++j) val -= A[i * C + j] * c->W_all[(task * T + t0 + j) * dh + d];
        Wi[d] = val;
      }
    }
    solves++;
    /* Outputs o_i */
    for (size_t i = 0; i < Lc; ++i) {
      const float *qi = c->q + ((b * T + t0 + i) * H + h) * dh;
      const double *gi = g + i * dh, *gami = gam + i * dh;
      for (size_t d = 0; d < dh; ++d) {
        double acc = 0.0;
        for (size_t a = 0; a < dh; ++a) acc += (double)qi[a] * gami[a] * Sp[a * dh + d];
        qS[d] = acc;
      }
      for (size_t j = 0; j <= i; ++j) {
        const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
        const double *gj = g + j * dh;
        double qk = 0.0;
        for (size_t a = 0; a < dh; ++a) qk += (double)qi[a] * exp(gi[a] - gj[a]) * (double)kj[a];
        const double *Wj = c->W_all + (task * T + t0 + j) * dh;
        for (size_t d = 0; d < dh; ++d) qS[d] += qk * Wj[d];
      }
      float *yi = c->y + ((b * T + t0 + i) * H + h) * dh;
      for (size_t d = 0; d < dh; ++d) yi[d] = (float)qS[d];
    }
    /* End-of-chunk state S_end */
    const double *glast = g + (Lc - 1u) * dh, *gam_last = gam + (Lc - 1u) * dh;
    for (size_t a = 0; a < dh; ++a) {
      double gl = gam_last[a];
      for (size_t d = 0; d < dh; ++d) S[a * dh + d] = gl * Sp[a * dh + d];
    }
    for (size_t j = 0; j < Lc; ++j) {
      const float *kj = c->k + ((b * T + t0 + j) * H + h) * dh;
      const double *gj = g + j * dh;
      const double *Wj = c->W_all + (task * T + t0 + j) * dh;
      for (size_t a = 0; a < dh; ++a) {
        double fk = exp(glast[a] - gj[a]) * (double)kj[a];
        for (size_t d = 0; d < dh; ++d) S[a * dh + d] += fk * Wj[d];
      }
    }
  }
  c->max_off[task] = max_a;
  c->wy_cnt[task] = solves;
}

typedef struct cdelta_bwd_ctx {
  size_t B, T, H, dh, C, nc;
  int is_gdn2;
  const float *q, *k, *v, *alpha, *beta, *erase, *write, *dy;
  const double *sprev_all, *W_all;
  float *dq, *dk, *dv, *da_pre, *db_pre, *de_pre, *dw_pre;
  double *work_dS, *work_Shist, *work_dP, *work_qS;
} cdelta_bwd_ctx;

/* Exact FP64 reverse-mode AD through channel-wise recurrence replayed from
 * saved chunk-start S_prev and WY W_all (0 approximation, O(C*dh^2), 0 malloc). */
static void cdelta_wy_bwd_worker(void *vctx, size_t task) {
  cdelta_bwd_ctx *c = (cdelta_bwd_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, C = c->C, nc = c->nc;
  double *dS = c->work_dS + task * dh * dh;
  double *Shist = c->work_Shist + task * (C + 1u) * dh * dh;
  double *dP = c->work_dP + task * dh * dh;
  double *dW_t = c->work_qS + task * dh;
  memset(dS, 0, dh * dh * sizeof(double));

  for (size_t ci = nc; ci-- > 0u; ) {
    size_t t0 = ci * C, Lc = (t0 + C <= T) ? C : (T - t0);
    const double *Sp = c->sprev_all + (task * nc + ci) * dh * dh;
    memcpy(Shist, Sp, dh * dh * sizeof(double));
    /* Forward replay of S_0 .. S_{Lc-1} using saved WY W_i: S_i = Diag(alpha_i)*S_{i-1} + k_i W_i^T */
    for (size_t i = 0; i < Lc; ++i) {
      size_t ti = b * T + t0 + i;
      const float *ki = c->k + (ti * H + h) * dh;
      const float *ali = c->alpha + (ti * H + h) * dh;
      const double *Wi = c->W_all + (task * T + t0 + i) * dh;
      const double *Sprev = Shist + i * dh * dh;
      double *Scur = Shist + (i + 1u) * dh * dh;
      for (size_t a = 0; a < dh; ++a) {
        double ala = (double)ali[a], ka = (double)ki[a];
        for (size_t d = 0; d < dh; ++d) Scur[a * dh + d] = ala * Sprev[a * dh + d] + ka * Wi[d];
      }
    }
    /* Reverse sweep i = Lc-1 .. 0 */
    for (size_t i = Lc; i-- > 0u; ) {
      size_t ti = b * T + t0 + i;
      const float *qi = c->q + (ti * H + h) * dh;
      const float *ki = c->k + (ti * H + h) * dh;
      const float *vi = c->v + (ti * H + h) * dh;
      const float *ali = c->alpha + (ti * H + h) * dh;
      const float *bi_v = c->is_gdn2 ? (c->erase + (ti * H + h) * dh) : NULL;
      const float *wi_v = c->is_gdn2 ? (c->write + (ti * H + h) * dh) : NULL;
      double bi_s = c->is_gdn2 ? 0.0 : (double)c->beta[ti * H + h];
      const float *dyi = c->dy + (ti * H + h) * dh;
      const double *Wi = c->W_all + (task * T + t0 + i) * dh;
      const double *Sprev = Shist + i * dh * dh;
      const double *Scur = Shist + (i + 1u) * dh * dh;
      float *dqi = c->dq + (ti * H + h) * dh;
      float *dki = c->dk + (ti * H + h) * dh;
      float *dvi = c->dv + (ti * H + h) * dh;
      float *dai = c->da_pre + (ti * H + h) * dh;
      float *dei = c->is_gdn2 ? (c->de_pre + (ti * H + h) * dh) : NULL;
      float *dwi = c->is_gdn2 ? (c->dw_pre + (ti * H + h) * dh) : NULL;

      /* o_i = Scur^T q_i */
      for (size_t a = 0; a < dh; ++a) {
        double acc = 0.0, qa = (double)qi[a];
        for (size_t d = 0; d < dh; ++d) {
          double dyd = (double)dyi[d];
          acc += Scur[a * dh + d] * dyd;
          dS[a * dh + d] += qa * dyd;
        }
        dqi[a] = (float)acc;
      }

      /* Scur = P + k_i W_i^T, where P_{a,d} = alpha_a * Sprev_{a,d} */
      for (size_t d = 0; d < dh; ++d) dW_t[d] = 0.0;
      double dk1[256];
      for (size_t a = 0; a < dh; ++a) {
        double ka = (double)ki[a], acc = 0.0;
        for (size_t d = 0; d < dh; ++d) {
          double ds = dS[a * dh + d];
          dP[a * dh + d] = ds;
          acc += ds * Wi[d];
          dW_t[d] += ka * ds;
        }
        dk1[a] = acc;
      }

      /* W_i = (w_i * v_i) - P^T (b_i * k_i) */
      double dbeta_sum = 0.0;
      for (size_t d = 0; d < dh; ++d) {
        double wd = c->is_gdn2 ? (double)wi_v[d] : bi_s;
        dvi[d] = (float)(dW_t[d] * wd);
        double dw_val = dW_t[d] * (double)vi[d];
        if (c->is_gdn2) dwi[d] = (float)(dw_val * wd * (1.0 - wd));
        else dbeta_sum += dw_val;
      }
      for (size_t a = 0; a < dh; ++a) {
        double ba = c->is_gdn2 ? (double)bi_v[a] : bi_s;
        double ala = (double)ali[a], ka = (double)ki[a], ea = ba * ka;
        double de_a = 0.0, da_a = 0.0;
        for (size_t d = 0; d < dh; ++d) {
          double p_ad = ala * Sprev[a * dh + d];
          de_a -= p_ad * dW_t[d];
          double dp_ad = dP[a * dh + d] - ea * dW_t[d];
          da_a += dp_ad * Sprev[a * dh + d];
          dS[a * dh + d] = ala * dp_ad;
        }
        double db_a = de_a * ka;
        dki[a] = (float)(dk1[a] + de_a * ba);
        dai[a] = (float)(da_a * ala * (1.0 - ala));
        if (c->is_gdn2) dei[a] = (float)(db_a * ba * (1.0 - ba));
        else dbeta_sum += db_a;
      }
      if (!c->is_gdn2) c->db_pre[ti * H + h] = (float)(dbeta_sum * bi_s * (1.0 - bi_s));
    }
  }
}

/* ==================== Arch-4 MLA (Materialized & Online Softmax, RoPE or NoPE) ==================== */
typedef struct mla_att_ctx {
  size_t B, T, H, dh, dr, use_rope, score_stride;
  const float *qc, *qr, *kc, *vc, *kr, *y_in, *dy;
  float *p, *y, *dqc, *dqr, *dkc, *dvc, *dkr;
  double *lse;
  int fp64_mode;
  double logit_thresh;
  double *scores_buf, *max_logits, *sink_sum, *dkr_loc;
  uint64_t *promos;
} mla_att_ctx;

static void mla_fwd_worker(void *vctx, size_t task) {
  mla_att_ctx *c = (mla_att_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, dr = c->use_rope ? c->dr : 0u;
  double scale = 1.0 / sqrt((double)(dh + dr));
  double max_obs = 0.0, sink = 0.0; uint64_t pr = 0u;
  double *scores = c->scores_buf + task * c->score_stride;
  for (size_t i = 0; i < T; ++i) {
    const float *qci = c->qc + ((b * T + i) * H + h) * dh;
    const float *qri = dr ? (c->qr + ((b * T + i) * H + h) * dr) : NULL;
    double mx = -1e30;
    for (size_t j = 0; j <= i; ++j) {
      const float *kcj = c->kc + ((b * T + j) * H + h) * dh;
      double s = 0.0;
      for (size_t d = 0; d < dh; ++d) s += (double)qci[d] * (double)kcj[d];
      if (dr) {
        const float *krj = c->kr + (b * T + j) * dr;
        for (size_t d = 0; d < dr; ++d) s += (double)qri[d] * (double)krj[d];
      }
      s *= scale; scores[j] = s;
      if (s > mx) mx = s;
      if (fabs(s) > max_obs) max_obs = fabs(s);
    }
    if (c->fp64_mode == BCIR_FP64_GUARD_ALWAYS || (c->fp64_mode == BCIR_FP64_GUARD_CONDITIONAL && fabs(mx) > c->logit_thresh)) pr++;
    double sum = 0.0;
    for (size_t j = 0; j <= i; ++j) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
    double inv = 1.0 / sum;
    float *pi = c->p + ((b * H + h) * T + i) * T;
    for (size_t j = 0; j <= i; ++j) pi[j] = (float)(scores[j] * inv);
    for (size_t j = i + 1u; j < T; ++j) pi[j] = 0.0f;
    sink += (double)pi[0];
    float *yi = c->y + ((b * T + i) * H + h) * dh;
    for (size_t d = 0; d < dh; ++d) {
      double acc = 0.0;
      for (size_t j = 0; j <= i; ++j) acc += (double)pi[j] * (double)c->vc[((b * T + j) * H + h) * dh + d];
      yi[d] = (float)acc;
    }
  }
  c->max_logits[task] = max_obs;
  c->promos[task] = pr;
  c->sink_sum[task] = sink / (double)T;
}

static void mla_bwd_bh_worker(void *vctx, size_t task) {
  mla_att_ctx *c = (mla_att_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, dr = c->use_rope ? c->dr : 0u;
  double scale = 1.0 / sqrt((double)(dh + dr));
  double *dp = c->scores_buf + task * c->score_stride;
  for (size_t j = 0; j < T; ++j) {
    memset(c->dkc + ((b * T + j) * H + h) * dh, 0, dh * 4u);
    memset(c->dvc + ((b * T + j) * H + h) * dh, 0, dh * 4u);
  }
  for (size_t i = 0; i < T; ++i) {
    const float *qci = c->qc + ((b * T + i) * H + h) * dh;
    const float *dyi = c->dy + ((b * T + i) * H + h) * dh;
    const float *pi = c->p + ((b * H + h) * T + i) * T;
    double dot = 0.0;
    for (size_t j = 0; j <= i; ++j) {
      const float *vcj = c->vc + ((b * T + j) * H + h) * dh;
      double s = 0.0;
      for (size_t d = 0; d < dh; ++d) s += (double)dyi[d] * (double)vcj[d];
      dp[j] = s; dot += (double)pi[j] * s;
    }
    float *dqci = c->dqc + ((b * T + i) * H + h) * dh;
    float *dqri = dr ? (c->dqr + ((b * T + i) * H + h) * dr) : NULL;
    memset(dqci, 0, dh * 4u);
    if (dr) memset(dqri, 0, dr * 4u);
    for (size_t j = 0; j <= i; ++j) {
      double pv = (double)pi[j], ds = pv * (dp[j] - dot) * scale;
      const float *kcj = c->kc + ((b * T + j) * H + h) * dh;
      float *dkcj = c->dkc + ((b * T + j) * H + h) * dh;
      float *dvcj = c->dvc + ((b * T + j) * H + h) * dh;
      for (size_t d = 0; d < dh; ++d) {
        dqci[d] = (float)((double)dqci[d] + ds * (double)kcj[d]);
        dkcj[d] = (float)((double)dkcj[d] + ds * (double)qci[d]);
        dvcj[d] = (float)((double)dvcj[d] + pv * (double)dyi[d]);
      }
      if (dr) {
        const float *krj = c->kr + (b * T + j) * dr;
        for (size_t d = 0; d < dr; ++d) dqri[d] = (float)((double)dqri[d] + ds * (double)krj[d]);
      }
    }
  }
}

static void mla_bwd_dkr_worker(void *vctx, size_t task) {
  mla_att_ctx *c = (mla_att_ctx *)vctx;
  size_t b = task, T = c->T, H = c->H, dh = c->dh, dr = c->dr;
  double scale = 1.0 / sqrt((double)(dh + dr));
  double *acc = c->dkr_loc + b * T * dr;
  memset(acc, 0, T * dr * sizeof(double));
  for (size_t h = 0; h < H; ++h) {
    for (size_t i = 0; i < T; ++i) {
      const float *qri = c->qr + ((b * T + i) * H + h) * dr;
      const float *dyi = c->dy + ((b * T + i) * H + h) * dh;
      const float *pi = c->p + ((b * H + h) * T + i) * T;
      double dot = 0.0;
      for (size_t j = 0; j <= i; ++j) {
        const float *vcj = c->vc + ((b * T + j) * H + h) * dh;
        double s = 0.0;
        for (size_t d = 0; d < dh; ++d) s += (double)dyi[d] * (double)vcj[d];
        dot += (double)pi[j] * s;
      }
      for (size_t j = 0; j <= i; ++j) {
        const float *vcj = c->vc + ((b * T + j) * H + h) * dh;
        double s = 0.0;
        for (size_t d = 0; d < dh; ++d) s += (double)dyi[d] * (double)vcj[d];
        double ds = (double)pi[j] * (s - dot) * scale;
        for (size_t d = 0; d < dr; ++d) acc[j * dr + d] += ds * (double)qri[d];
      }
    }
  }
  for (size_t j = 0; j < T; ++j)
    for (size_t d = 0; d < dr; ++d) c->dkr[(b * T + j) * dr + d] = (float)acc[j * dr + d];
}

/* TRAIN-OPT-5 Online-Softmax O(T) MLA Forward & Backward */
static void mla_online_fwd_worker(void *vctx, size_t task) {
  mla_att_ctx *c = (mla_att_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, dr = c->use_rope ? c->dr : 0u;
  double scale = 1.0 / sqrt((double)(dh + dr));
  double max_obs = 0.0, sink = 0.0; uint64_t pr = 0u;
  double acc[256], s_tile[ONLINE_TILE_K];
  for (size_t i = 0; i < T; ++i) {
    const float *qci = c->qc + ((b * T + i) * H + h) * dh;
    const float *qri = dr ? (c->qr + ((b * T + i) * H + h) * dr) : NULL;
    double m_run = -1e30, l_run = 0.0, s0 = 0.0;
    for (size_t d = 0; d < dh; ++d) acc[d] = 0.0;
    for (size_t j0 = 0; j0 <= i; j0 += ONLINE_TILE_K) {
      size_t j1 = (j0 + ONLINE_TILE_K <= i + 1u) ? (j0 + ONLINE_TILE_K) : (i + 1u);
      double m_tile = -1e30;
      for (size_t j = j0; j < j1; ++j) {
        const float *kcj = c->kc + ((b * T + j) * H + h) * dh;
        double s = 0.0;
        for (size_t d = 0; d < dh; ++d) s += (double)qci[d] * (double)kcj[d];
        if (dr) {
          const float *krj = c->kr + (b * T + j) * dr;
          for (size_t d = 0; d < dr; ++d) s += (double)qri[d] * (double)krj[d];
        }
        s *= scale; s_tile[j - j0] = s;
        if (j == 0u) s0 = s;
        if (s > m_tile) m_tile = s;
        if (fabs(s) > max_obs) max_obs = fabs(s);
      }
      double m_new = (m_run > m_tile) ? m_run : m_tile;
      double alpha = (l_run > 0.0) ? exp(m_run - m_new) : 0.0;
      l_run *= alpha;
      for (size_t d = 0; d < dh; ++d) acc[d] *= alpha;
      for (size_t j = j0; j < j1; ++j) {
        double p = exp(s_tile[j - j0] - m_new);
        l_run += p;
        const float *vcj = c->vc + ((b * T + j) * H + h) * dh;
        for (size_t d = 0; d < dh; ++d) acc[d] += p * (double)vcj[d];
      }
      m_run = m_new;
    }
    if (c->fp64_mode == BCIR_FP64_GUARD_ALWAYS || (c->fp64_mode == BCIR_FP64_GUARD_CONDITIONAL && fabs(m_run) > c->logit_thresh)) pr++;
    double lse_i = m_run + log(l_run);
    c->lse[(b * H + h) * T + i] = lse_i;
    sink += exp(s0 - lse_i);
    double inv_l = 1.0 / l_run;
    float *yi = c->y + ((b * T + i) * H + h) * dh;
    for (size_t d = 0; d < dh; ++d) yi[d] = (float)(acc[d] * inv_l);
  }
  c->max_logits[task] = max_obs;
  c->promos[task] = pr;
  c->sink_sum[task] = sink / (double)T;
}

static void mla_online_bwd_bh_worker(void *vctx, size_t task) {
  mla_att_ctx *c = (mla_att_ctx *)vctx;
  size_t b = task / c->H, h = task % c->H;
  size_t T = c->T, H = c->H, dh = c->dh, dr = c->use_rope ? c->dr : 0u;
  double scale = 1.0 / sqrt((double)(dh + dr));
  for (size_t j = 0; j < T; ++j) {
    memset(c->dkc + ((b * T + j) * H + h) * dh, 0, dh * 4u);
    memset(c->dvc + ((b * T + j) * H + h) * dh, 0, dh * 4u);
  }
  for (size_t i = 0; i < T; ++i) {
    const float *qci = c->qc + ((b * T + i) * H + h) * dh;
    const float *qri = dr ? (c->qr + ((b * T + i) * H + h) * dr) : NULL;
    const float *yi = c->y_in + ((b * T + i) * H + h) * dh;
    const float *dyi = c->dy + ((b * T + i) * H + h) * dh;
    double lse_i = c->lse[(b * H + h) * T + i];
    double dot = 0.0;
    for (size_t d = 0; d < dh; ++d) dot += (double)dyi[d] * (double)yi[d];
    float *dqci = c->dqc + ((b * T + i) * H + h) * dh;
    float *dqri = dr ? (c->dqr + ((b * T + i) * H + h) * dr) : NULL;
    memset(dqci, 0, dh * 4u);
    if (dr) memset(dqri, 0, dr * 4u);
    for (size_t j = 0; j <= i; ++j) {
      const float *kcj = c->kc + ((b * T + j) * H + h) * dh;
      const float *vcj = c->vc + ((b * T + j) * H + h) * dh;
      double s = 0.0, dp = 0.0;
      for (size_t d = 0; d < dh; ++d) {
        s += (double)qci[d] * (double)kcj[d];
        dp += (double)dyi[d] * (double)vcj[d];
      }
      if (dr) {
        const float *krj = c->kr + (b * T + j) * dr;
        for (size_t d = 0; d < dr; ++d) s += (double)qri[d] * (double)krj[d];
      }
      double p = exp(s * scale - lse_i);
      double ds = p * (dp - dot) * scale;
      float *dkcj = c->dkc + ((b * T + j) * H + h) * dh;
      float *dvcj = c->dvc + ((b * T + j) * H + h) * dh;
      for (size_t d = 0; d < dh; ++d) {
        dqci[d] = (float)((double)dqci[d] + ds * (double)kcj[d]);
        dkcj[d] = (float)((double)dkcj[d] + ds * (double)qci[d]);
        dvcj[d] = (float)((double)dvcj[d] + p * (double)dyi[d]);
      }
      if (dr) {
        const float *krj = c->kr + (b * T + j) * dr;
        for (size_t d = 0; d < dr; ++d) dqri[d] = (float)((double)dqri[d] + ds * (double)krj[d]);
      }
    }
  }
}

static void mla_online_bwd_dkr_worker(void *vctx, size_t task) {
  mla_att_ctx *c = (mla_att_ctx *)vctx;
  size_t b = task, T = c->T, H = c->H, dh = c->dh, dr = c->dr;
  double scale = 1.0 / sqrt((double)(dh + dr));
  double *acc = c->dkr_loc + b * T * dr;
  memset(acc, 0, T * dr * sizeof(double));
  for (size_t h = 0; h < H; ++h) {
    for (size_t i = 0; i < T; ++i) {
      const float *qci = c->qc + ((b * T + i) * H + h) * dh;
      const float *qri = c->qr + ((b * T + i) * H + h) * dr;
      const float *yi = c->y_in + ((b * T + i) * H + h) * dh;
      const float *dyi = c->dy + ((b * T + i) * H + h) * dh;
      double lse_i = c->lse[(b * H + h) * T + i];
      double dot = 0.0;
      for (size_t d = 0; d < dh; ++d) dot += (double)dyi[d] * (double)yi[d];
      for (size_t j = 0; j <= i; ++j) {
        const float *kcj = c->kc + ((b * T + j) * H + h) * dh;
        const float *vcj = c->vc + ((b * T + j) * H + h) * dh;
        const float *krj = c->kr + (b * T + j) * dr;
        double s = 0.0, dp = 0.0;
        for (size_t d = 0; d < dh; ++d) {
          s += (double)qci[d] * (double)kcj[d];
          dp += (double)dyi[d] * (double)vcj[d];
        }
        for (size_t d = 0; d < dr; ++d) s += (double)qri[d] * (double)krj[d];
        double p = exp(s * scale - lse_i);
        double ds = p * (dp - dot) * scale;
        for (size_t d = 0; d < dr; ++d) acc[j * dr + d] += ds * (double)qri[d];
      }
    }
  }
  for (size_t j = 0; j < T; ++j)
    for (size_t d = 0; d < dr; ++d) c->dkr[(b * T + j) * dr + d] = (float)acc[j * dr + d];
}

/* ==================== Forward Pass ==================== */
static int ext_forward_internal(bcir_ext_model *m, const uint32_t *tokens, uint32_t active_time) {
  size_t B = m->sp.batch, T = active_time, N = B * T, d = m->sp.width, H = m->sp.heads;
  size_t KH = m->sp.kvheads, dh = d / H, kvd = KH * dh, ff = m->sp.ff, V = m->sp.vocab;
  size_t dc = m->sp.mla_dc, dr = m->sp.mla_dr, rg = m->gate_rank, mC = m->mhc_c;
  const float *W = m->weights;

  for (size_t i = 0; i < N; ++i) {
    if (tokens[i] >= V) return 1;
    const float *erow = W + m->off_embed + (size_t)tokens[i] * d;
    if (mC > 1u) {
      for (size_t s = 0; s < mC; ++s) memcpy(m->acts[0].mhc_x1 + (i * mC + s) * d, erow, d * 4u);
    } else {
      memcpy(m->acts[0].x_in + i * d, erow, d * 4u);
    }
  }

  int bad = 0;
  size_t score_stride = max_sz(m->sp.time, m->sp.swa_window);
  for (uint32_t l = 0; l < m->sp.layers; ++l) {
    const layer_offsets *lo = &m->loff[l];
    layer_act *a = &m->acts[l];
    float *x_out = (l + 1u < m->sp.layers) ? m->acts[l + 1u].x_in : m->x_final;
    float *mhc_next = (mC > 1u) ? ((l + 1u < m->sp.layers) ? m->acts[l + 1u].mhc_x1 : m->mhc_xfinal) : NULL;

    /* mHC pre-mixer stream aggregation */
    double hres1[16], hpre1[4], hpost1[4], hres2[16], hpre2[4], hpost2[4];
    if (mC > 1u) {
      mhc_sinkhorn_fwd(mC, W + lo->mhc_hres1, hres1);
      for (size_t s = 0; s < mC; ++s) {
        hpre1[s] = sigmoid_d((double)W[lo->mhc_hpre1 + s]);
        hpost1[s] = 2.0 * sigmoid_d((double)W[lo->mhc_hpost1 + s]);
      }
      for (size_t i = 0; i < N; ++i) {
        float *xi = a->x_in + i * d;
        for (size_t c = 0; c < d; ++c) {
          double acc = 0.0;
          for (size_t s = 0; s < mC; ++s) acc += hpre1[s] * (double)a->mhc_x1[(i * mC + s) * d + c];
          xi[c] = (float)acc;
        }
      }
    }

    ext_rms_fwd(m, N, d, a->x_in, W + lo->norm1, a->n1_raw);
    if (m->sp.canon_mode >= BCIR_CANON_A) {
      size_t cc = 32u, nt = (d + cc - 1u) / cc;
      canon_ctx cx = { B, T, d, cc, a->n1_raw, W + lo->canon_a, NULL, a->n1, NULL, NULL };
      bcir_tensor_pool_run(m->pool, canon_fwd_worker, &cx, nt);
    }

    if (lo->mixer == BCIR_MIXER_GQA || lo->mixer == BCIR_MIXER_SWA) {
      float *qt = m->sp.qk_norm ? a->q_raw : a->q;
      float *kt = m->sp.qk_norm ? a->k_raw : a->k;
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wq, qt);
      ext_linear_fwd(m, 1, N, d, kvd, a->n1, W + lo->wk, kt);
      ext_linear_fwd(m, 1, N, d, kvd, a->n1, W + lo->wv, a->v);
      if (m->sp.qk_norm) {
        ext_rms_fwd(m, N * H, dh, a->q_raw, W + lo->q_norm, a->q);
        ext_rms_fwd(m, N * KH, dh, a->k_raw, W + lo->k_norm, a->k);
      }
      if (lo->use_rope) {
        bcir_tensor_rope_lent_pool(m->pool, B, T, H, dh, m->rope_hd, a->q, 0);
        bcir_tensor_rope_lent_pool(m->pool, B, T, KH, dh, m->rope_hd, a->k, 0);
      }
      if (lo->mixer == BCIR_MIXER_GQA) {
        gqa_fwd_dispatch(m, B, T, a->q, a->k, a->v, a->p, a->lse, a->ctx_raw, &bad);
      } else {
        size_t nt = B * H;
        swa_fwd_ctx sc = { B, T, H, KH, dh, m->sp.swa_window, score_stride,
                           a->q, a->k, a->v, a->p, a->ctx_raw,
                           m->pol.fp64_guard_mode, m->pol.softmax_logit_thresh,
                           m->attn_scores, m->attn_max_log, m->attn_sink_sum, m->attn_promos };
        bcir_tensor_pool_run(m->pool, swa_fwd_worker, &sc, nt);
        double ssum = 0.0;
        for (size_t i = 0; i < nt; ++i) {
          if (m->attn_max_log[i] > m->tel.max_softmax_logit) m->tel.max_softmax_logit = m->attn_max_log[i];
          m->tel.fp64_softmax_promotions += m->attn_promos[i];
          ssum += m->attn_sink_sum[i];
        }
        m->tel.first_tok_attn_mean = ssum / (double)nt;
      }
    } else if (lo->mixer == BCIR_MIXER_DELTANET) {
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wq, a->q_raw);
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wk, a->k_raw);
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wv, a->v);
      dn_l2norm_fwd(N, H, dh, a->q_raw, a->q, a->q_inv);
      dn_l2norm_fwd(N, H, dh, a->k_raw, a->k, a->k_inv);
      ext_linear_fwd(m, 1, N, d, H, a->n1, W + lo->walpha, a->a_pre);
      ext_linear_fwd(m, 1, N, d, H, a->n1, W + lo->wbeta, a->b_pre);
      const float *ba = W + lo->balpha, *bb = W + lo->bbeta;
      for (size_t i = 0; i < N; ++i) {
        for (size_t h = 0; h < H; ++h) {
          a->alpha[i * H + h] = sigmoid_f(a->a_pre[i * H + h] + ba[h]);
          a->beta[i * H + h] = sigmoid_f(a->b_pre[i * H + h] + bb[h]);
        }
      }
      size_t C = m->sp.delta_chunk, nc = (T + C - 1u) / C, nt = B * H;
      dn_fwd_ctx dc_ctx = { B, T, H, dh, C, nc, a->q, a->k, a->v, a->alpha, a->beta,
                            a->dn_sprev, a->dn_W, a->ctx_raw,
                            m->dn_work_S, m->dn_work_g, m->dn_work_gam, m->dn_work_A, m->dn_work_qS,
                            m->attn_max_log ? m->attn_max_log : NULL, m->attn_promos ? m->attn_promos : NULL };
      double mo_stack[128]; uint64_t wc_stack[128];
      if (!dc_ctx.max_off) { dc_ctx.max_off = mo_stack; dc_ctx.wy_cnt = wc_stack; }
      bcir_tensor_pool_run(m->pool, dn_wy_fwd_worker, &dc_ctx, nt);
      for (size_t i = 0; i < nt; ++i) {
        if (dc_ctx.max_off[i] > m->tel.max_wy_offdiag) m->tel.max_wy_offdiag = dc_ctx.max_off[i];
        m->tel.fp64_wy_solves += dc_ctx.wy_cnt[i];
      }
    } else if (lo->mixer == BCIR_MIXER_KDA || lo->mixer == BCIR_MIXER_GDN2) {
      int is_g2 = (lo->mixer == BCIR_MIXER_GDN2);
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wq, a->q_raw);
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wk, a->k_raw);
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wv, a->v);
      dn_l2norm_fwd(N, H, dh, a->q_raw, a->q, a->q_inv);
      dn_l2norm_fwd(N, H, dh, a->k_raw, a->k, a->k_inv);
      /* Low-rank channel-wise decay alpha */
      ext_linear_fwd(m, 1, N, d, rg, a->n1, W + lo->wa1, a->a_mid);
      ext_linear_fwd(m, 1, N, rg, d, a->a_mid, W + lo->wa2, a->a_pre);
      const float *bac = W + lo->ba_chan;
      for (size_t i = 0; i < N; ++i)
        for (size_t c = 0; c < d; ++c) a->alpha[i * d + c] = sigmoid_f(a->a_pre[i * d + c] + bac[c]);
      if (!is_g2) {
        ext_linear_fwd(m, 1, N, d, H, a->n1, W + lo->wbeta, a->b_pre);
        const float *bb = W + lo->bbeta;
        for (size_t i = 0; i < N; ++i)
          for (size_t h = 0; h < H; ++h) a->beta[i * H + h] = sigmoid_f(a->b_pre[i * H + h] + bb[h]);
      } else {
        ext_linear_fwd(m, 1, N, d, rg, a->n1, W + lo->we1, a->e_mid);
        ext_linear_fwd(m, 1, N, rg, d, a->e_mid, W + lo->we2, a->e_pre);
        const float *bec = W + lo->be_chan;
        for (size_t i = 0; i < N; ++i)
          for (size_t c = 0; c < d; ++c) a->erase[i * d + c] = sigmoid_f(a->e_pre[i * d + c] + bec[c]);
        ext_linear_fwd(m, 1, N, d, rg, a->n1, W + lo->ww1, a->w_mid);
        ext_linear_fwd(m, 1, N, rg, d, a->w_mid, W + lo->ww2, a->w_pre);
        const float *bwc = W + lo->bw_chan;
        for (size_t i = 0; i < N; ++i)
          for (size_t c = 0; c < d; ++c) a->write[i * d + c] = sigmoid_f(a->w_pre[i * d + c] + bwc[c]);
      }
      size_t C = m->sp.delta_chunk, nc = (T + C - 1u) / C, nt = B * H;
      double mo_stack[128]; uint64_t wc_stack[128];
      double *mo = m->attn_max_log ? m->attn_max_log : mo_stack;
      uint64_t *wc = m->attn_promos ? m->attn_promos : wc_stack;
      cdelta_fwd_ctx cc = { B, T, H, dh, C, nc, is_g2, a->q, a->k, a->v, a->alpha,
                            is_g2 ? NULL : a->beta, is_g2 ? a->erase : NULL, is_g2 ? a->write : NULL,
                            a->dn_sprev, a->dn_W, a->cd_y,
                            m->dn_work_S, m->dn_work_g, m->dn_work_gam, m->dn_work_A, m->dn_work_qS,
                            mo, wc };
      bcir_tensor_pool_run(m->pool, cdelta_wy_fwd_worker, &cc, nt);
      for (size_t i = 0; i < nt; ++i) {
        if (mo[i] > m->tel.max_wy_offdiag) m->tel.max_wy_offdiag = mo[i];
        m->tel.fp64_wy_solves += wc[i];
      }
      ext_rms_fwd(m, N * H, dh, a->cd_y, W + lo->o_norm, a->ctx_raw);
    } else if (lo->mixer == BCIR_MIXER_MLA) {
      float *qct = m->sp.qk_norm ? a->q_c_raw : a->q_c;
      ext_linear_fwd(m, 1, N, d, d, a->n1, W + lo->wq_c, qct);
      if (m->sp.qk_norm) ext_rms_fwd(m, N * H, dh, a->q_c_raw, W + lo->q_norm, a->q_c);
      ext_linear_fwd(m, 1, N, d, dc, a->n1, W + lo->w_dkv, a->c_raw);
      ext_rms_fwd(m, N, dc, a->c_raw, W + lo->norm_kv, a->c_hat);
      ext_linear_fwd(m, 1, N, dc, d, a->c_hat, W + lo->w_uk, a->k_c);
      ext_linear_fwd(m, 1, N, dc, d, a->c_hat, W + lo->w_uv, a->v_c);
      if (lo->use_rope) {
        ext_linear_fwd(m, 1, N, d, H * dr, a->n1, W + lo->wq_r, a->q_r);
        ext_linear_fwd(m, 1, N, d, dr, a->n1, W + lo->w_kr, a->k_r);
        bcir_tensor_rope_lent_pool(m->pool, B, T, H, dr, m->rope_dr, a->q_r, 0);
        bcir_tensor_rope_lent_pool(m->pool, B, T, 1u, dr, m->rope_dr, a->k_r, 0);
      }
      size_t nt = B * H;
      mla_att_ctx mc = { B, T, H, dh, dr, lo->use_rope, score_stride,
                         a->q_c, a->q_r, a->k_c, a->v_c, a->k_r, NULL, NULL,
                         a->mla_p, a->ctx_raw, NULL, NULL, NULL, NULL, NULL,
                         a->lse, m->pol.fp64_guard_mode, m->pol.softmax_logit_thresh,
                         m->attn_scores, m->attn_max_log, m->attn_sink_sum, m->attn_dkr_loc, m->attn_promos };
      bcir_tensor_pool_run(m->pool, m->sp.online_softmax ? mla_online_fwd_worker : mla_fwd_worker, &mc, nt);
      double ssum = 0.0;
      for (size_t i = 0; i < nt; ++i) {
        if (m->attn_max_log[i] > m->tel.max_softmax_logit) m->tel.max_softmax_logit = m->attn_max_log[i];
        m->tel.fp64_softmax_promotions += m->attn_promos[i];
        ssum += m->attn_sink_sum[i];
      }
      m->tel.first_tok_attn_mean = ssum / (double)nt;
    }

    /* Optional G1 SDPA Sigmoid Output Gate (arXiv:2505.06708) */
    if (m->sp.gate_mode == BCIR_GATE_G1_HEADWISE) {
      ext_linear_fwd(m, 1, N, d, H, a->n1, W + lo->wg1, a->g_val);
      const float *bg = W + lo->bg;
      for (size_t i = 0; i < N; ++i) {
        for (size_t h = 0; h < H; ++h) {
          float gv = sigmoid_f(a->g_val[i * H + h] + bg[h]);
          a->g_val[i * H + h] = gv;
          const float *src = a->ctx_raw + (i * H + h) * dh;
          float *dst = a->ctx + (i * H + h) * dh;
          for (size_t k = 0; k < dh; ++k) dst[k] = src[k] * gv;
        }
      }
    } else if (m->sp.gate_mode == BCIR_GATE_G1_LOWRANK) {
      ext_linear_fwd(m, 1, N, d, rg, a->n1, W + lo->wg1, a->g_mid);
      ext_linear_fwd(m, 1, N, rg, d, a->g_mid, W + lo->wg2, a->g_val);
      const float *bg = W + lo->bg;
      for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < d; ++c) {
          float gv = sigmoid_f(a->g_val[i * d + c] + bg[c]);
          a->g_val[i * d + c] = gv;
          a->ctx[i * d + c] = a->ctx_raw[i * d + c] * gv;
        }
      }
    }

    /* Output projection + Residual #1 */
    float *mo_dst = (mC > 1u) ? a->mix_out : a->x2;
    ext_linear_fwd(m, 1, N, d, d, a->ctx, W + lo->wo, mo_dst);
    for (size_t i = 0; i < N * d; ++i) {
      double av = fabs((double)mo_dst[i]);
      if (!isfinite(av)) bad = 1;
      else if (av > m->tel.max_abs_act) m->tel.max_abs_act = av;
    }
    if (mC > 1u) {
      mhc_sinkhorn_fwd(mC, W + lo->mhc_hres2, hres2);
      for (size_t s = 0; s < mC; ++s) {
        hpre2[s] = sigmoid_d((double)W[lo->mhc_hpre2 + s]);
        hpost2[s] = 2.0 * sigmoid_d((double)W[lo->mhc_hpost2 + s]);
      }
      for (size_t i = 0; i < N; ++i) {
        for (size_t r = 0; r < mC; ++r) {
          float *dst = a->mhc_x2 + (i * mC + r) * d;
          for (size_t c = 0; c < d; ++c) {
            double acc = hpost1[r] * (double)a->mix_out[i * d + c];
            for (size_t s = 0; s < mC; ++s) acc += hres1[r * mC + s] * (double)a->mhc_x1[(i * mC + s) * d + c];
            dst[c] = (float)acc;
          }
        }
        float *x2i = a->x2 + i * d;
        for (size_t c = 0; c < d; ++c) {
          double acc = 0.0;
          for (size_t s = 0; s < mC; ++s) acc += hpre2[s] * (double)a->mhc_x2[(i * mC + s) * d + c];
          x2i[c] = (float)acc;
        }
      }
    } else {
      bcir_tensor_add_pool(m->pool, N * d, a->x_in, a->x2);
    }

    /* Pre-MLP RMSNorm + optional Canon-C + SwiGLU MLP */
    ext_rms_fwd(m, N, d, a->x2, W + lo->norm2, a->n2_raw);
    if (m->sp.canon_mode >= BCIR_CANON_AC) {
      size_t cc = 32u, nt = (d + cc - 1u) / cc;
      canon_ctx cx = { B, T, d, cc, a->n2_raw, W + lo->canon_c, NULL, a->n2, NULL, NULL };
      bcir_tensor_pool_run(m->pool, canon_fwd_worker, &cx, nt);
    }
    ext_linear_fwd(m, 2, N, d, ff, a->n2, W + lo->wgate, a->gate);
    ext_linear_fwd(m, 2, N, d, ff, a->n2, W + lo->wup, a->up);
    bcir_tensor_silu_mul_pool(m->pool, N * ff, a->gate, a->up, a->hidden);
    float *ml_dst = (mC > 1u) ? a->mlp_out : x_out;
    ext_linear_fwd(m, 2, N, ff, d, a->hidden, W + lo->wdown, ml_dst);

    /* Residual #2 */
    if (mC > 1u) {
      for (size_t i = 0; i < N; ++i) {
        for (size_t r = 0; r < mC; ++r) {
          float *dst = mhc_next + (i * mC + r) * d;
          for (size_t c = 0; c < d; ++c) {
            double acc = hpost2[r] * (double)a->mlp_out[i * d + c];
            for (size_t s = 0; s < mC; ++s) acc += hres2[r * mC + s] * (double)a->mhc_x2[(i * mC + s) * d + c];
            dst[c] = (float)acc;
          }
        }
      }
    } else {
      bcir_tensor_add_pool(m->pool, N * d, a->x2, x_out);
    }
  }

  if (mC > 1u) {
    double inv_c = 1.0 / (double)mC;
    for (size_t i = 0; i < N; ++i) {
      float *xf = m->x_final + i * d;
      for (size_t c = 0; c < d; ++c) {
        double acc = 0.0;
        for (size_t s = 0; s < mC; ++s) acc += (double)m->mhc_xfinal[(i * mC + s) * d + c];
        xf[c] = (float)(acc * inv_c);
      }
    }
  }
  ext_rms_fwd(m, N, d, m->x_final, W + m->off_final_norm, m->norm_final);
  ext_linear_fwd(m, 0, N, d, V, m->norm_final, W + m->off_lm_head, m->logits);
  for (size_t i = 0; i < N * V; ++i) {
    if (!isfinite(m->logits[i])) { bad = 1; break; }
  }
  if (bad) { m->tel.nonfinite_events++; return 3; }
  return 0;
}

int bcir_ext_forward(bcir_ext_model *m, const uint32_t *tokens, uint32_t active_time, float *out_logits) {
  if (!m || !tokens || active_time == 0u || active_time > m->sp.time) return 1;
  int rc = ext_forward_internal(m, tokens, active_time);
  if (rc != 0) return rc;
  if (out_logits) memcpy(out_logits, m->logits, (size_t)m->sp.batch * active_time * m->sp.vocab * 4u);
  return 0;
}

int bcir_ext_evaluate(bcir_ext_model *m, const uint32_t *tokens, const uint32_t *targets,
                      size_t n_batches, uint32_t active_time, bcir_ext_eval_result *out) {
  if (!m || !tokens || !targets || !out || active_time == 0u || active_time > m->sp.time) return 1;
  double t0 = now_ms(), loss_sum = 0.0;
  uint64_t tok_cnt = 0u, hits = 0u, nf = 0u;
  size_t N = (size_t)m->sp.batch * active_time, V = m->sp.vocab;
  for (size_t bi = 0; bi < n_batches; ++bi) {
    const uint32_t *tk = tokens + bi * N, *tg = targets + bi * N;
    int frc = ext_forward_internal(m, tk, active_time);
    if (frc != 0) { if (frc == 1) return 1; nf++; continue; }
    for (size_t i = 0; i < N; ++i) {
      uint32_t y = tg[i];
      if (y == BCIR_EXT_IGNORE) continue;
      if (y >= V) return 1;
      const float *row = m->logits + i * V;
      double mx = -1e30; uint32_t best = 0u; int row_ok = 1;
      for (size_t v = 0; v < V; ++v) {
        double lv = (double)row[v];
        if (!isfinite(lv)) { nf++; row_ok = 0; break; }
        if (lv > mx) { mx = lv; best = (uint32_t)v; }
      }
      if (!row_ok) continue;
      double se = 0.0;
      for (size_t v = 0; v < V; ++v) se += exp((double)row[v] - mx);
      double ce = (mx + log(se)) - (double)row[y];
      if (!isfinite(ce)) { nf++; continue; }
      loss_sum += ce;
      if (best == y) hits++;
      tok_cnt++;
    }
  }
  out->tokens = tok_cnt; out->nonfinite = nf;
  if (nf > 0u || tok_cnt == 0u) { out->ce = NAN; out->accuracy = 0.0; }
  else { out->ce = loss_sum / (double)tok_cnt; out->accuracy = (double)hits / (double)tok_cnt; }
  out->eval_ms = now_ms() - t0;
  return (nf > 0u || tok_cnt == 0u) ? 3 : 0;
}

/* ==================== Deterministic Embedding Scatter & Streaming Grad Commit ==================== */
typedef struct emb_scat_ctx {
  size_t N, d, col_chunk;
  const uint32_t *tokens;
  const float *dx;
  float *demb;
} emb_scat_ctx;

static void emb_scat_worker(void *vctx, size_t task) {
  emb_scat_ctx *c = (emb_scat_ctx *)vctx;
  size_t c0 = task * c->col_chunk, c1 = (c0 + c->col_chunk <= c->d) ? (c0 + c->col_chunk) : c->d;
  for (size_t i = 0; i < c->N; ++i) {
    float *dst = c->demb + (size_t)c->tokens[i] * c->d;
    const float *src = c->dx + i * c->d;
    for (size_t j = c0; j < c1; ++j) dst[j] += src[j];
  }
}

static float *ext_grad_target(bcir_ext_model *m, size_t off, size_t count) {
  if (m->opt->grad_fp32) return m->opt->grad_fp32 + off;
  memset(m->grad_tile, 0, count * sizeof(float));
  return m->grad_tile;
}

static void ext_grad_commit(bcir_ext_model *m, size_t off, size_t count, int accumulate) {
  if (!m->opt->grad_fp32) bcir_opt_commit_grad_span_fp32(m->opt, off, count, m->grad_tile, accumulate);
}

static void ext_linear_bwd_grad(bcir_ext_model *m, int region, size_t rows, size_t in, size_t out,
                                const float *x, const float *w, const float *dy,
                                float *dx, size_t w_off, float dx_beta, int accumulate) {
  size_t cnt = out * in;
  float *dw = ext_grad_target(m, w_off, cnt);
  ext_linear_bwd(m, region, rows, in, out, x, w, dy, dx, dw, dx_beta, m->opt->grad_fp32 ? 1.0f : 0.0f);
  ext_grad_commit(m, w_off, cnt, accumulate);
}

static void ext_rms_bwd_grad(bcir_ext_model *m, size_t rows, size_t width, const float *x,
                             const float *g, const float *dy, float *dx, size_t g_off,
                             int dx_add, int accumulate) {
  float *dg = ext_grad_target(m, g_off, width);
  ext_rms_bwd(m, rows, width, x, g, dy, dx, dg, dx_add);
  ext_grad_commit(m, g_off, width, accumulate);
}

static void ext_bias_bwd_grad(bcir_ext_model *m, size_t N, size_t dim, const float *dpre,
                              size_t b_off, int accumulate) {
  float *gb = ext_grad_target(m, b_off, dim);
  for (size_t i = 0; i < N; ++i)
    for (size_t c = 0; c < dim; ++c) gb[c] += dpre[i * dim + c];
  ext_grad_commit(m, b_off, dim, accumulate);
}

/* ==================== Loss + Full Analytical Backward Pass ==================== */
int bcir_ext_loss_backward(bcir_ext_model *m, const uint32_t *tokens, const uint32_t *targets,
                           uint32_t active_time, int accumulate,
                           double *out_loss, uint64_t *out_sup_tokens) {
  if (!m || !tokens || !targets || active_time == 0u || active_time > m->sp.time) return 1;
  size_t B = m->sp.batch, T = active_time, N = B * T, d = m->sp.width, H = m->sp.heads;
  size_t KH = m->sp.kvheads, dh = d / H, kvd = KH * dh, ff = m->sp.ff, V = m->sp.vocab;
  size_t dc = m->sp.mla_dc, dr = m->sp.mla_dr, rg = m->gate_rank, mC = m->mhc_c;

  uint64_t sup = 0u;
  for (size_t i = 0; i < N; ++i) {
    if (targets[i] != BCIR_EXT_IGNORE) {
      if (targets[i] >= V) return 1;
      sup++;
    }
  }
  if (out_sup_tokens) *out_sup_tokens = sup;
  if (sup == 0u) return 1;

  int frc = ext_forward_internal(m, tokens, active_time);
  if (frc != 0) return frc;

  if (!accumulate) bcir_opt_zero_grad(m->opt);

  double loss_sum = 0.0, inv_sup = 1.0 / (double)sup;
  for (size_t i = 0; i < N; ++i) {
    uint32_t y = targets[i];
    float *dl = m->d_logits + i * V;
    if (y == BCIR_EXT_IGNORE) { memset(dl, 0, V * 4u); continue; }
    const float *row = m->logits + i * V;
    double mx = -1e30;
    for (size_t v = 0; v < V; ++v) {
      double lv = (double)row[v];
      if (!isfinite(lv)) { m->tel.nonfinite_events++; return 3; }
      if (lv > mx) mx = lv;
    }
    double se = 0.0;
    for (size_t v = 0; v < V; ++v) se += exp((double)row[v] - mx);
    double lse = mx + log(se);
    loss_sum += lse - (double)row[y];
    double inv_se = 1.0 / se;
    for (size_t v = 0; v < V; ++v) {
      double p = exp((double)row[v] - mx) * inv_se;
      dl[v] = (float)((p - (v == y ? 1.0 : 0.0)) * inv_sup);
    }
  }
  if (out_loss) *out_loss = loss_sum * inv_sup;

  const float *W = m->weights;
  size_t score_stride = max_sz(m->sp.time, m->sp.swa_window);
  ext_linear_bwd_grad(m, 0, N, d, V, m->norm_final, W + m->off_lm_head, m->d_logits,
                      m->d_n, m->off_lm_head, 0.0f, accumulate);
  ext_rms_bwd_grad(m, N, d, m->x_final, W + m->off_final_norm, m->d_n, m->d_cur,
                   m->off_final_norm, 0, accumulate);

  if (mC > 1u) {
    float inv_c = (float)(1.0 / (double)mC);
    for (size_t i = 0; i < N; ++i)
      for (size_t s = 0; s < mC; ++s)
        for (size_t c = 0; c < d; ++c) m->d_mhc_cur[(i * mC + s) * d + c] = m->d_cur[i * d + c] * inv_c;
  }

  for (uint32_t l = m->sp.layers; l-- > 0u; ) {
    const layer_offsets *lo = &m->loff[l];
    layer_act *a = &m->acts[l];

    /* --- Residual #2 backward (standard or mHC) --- */
    double hres1[16], hpre1[4], hpost1[4], hres2[16], hpre2[4], hpost2[4];
    if (mC > 1u) {
      mhc_sinkhorn_fwd(mC, W + lo->mhc_hres2, hres2);
      for (size_t s = 0; s < mC; ++s) {
        hpre2[s] = sigmoid_d((double)W[lo->mhc_hpre2 + s]);
        hpost2[s] = 2.0 * sigmoid_d((double)W[lo->mhc_hpost2 + s]);
      }
      double dhres2[16] = {0.0}, dhpost2[4] = {0.0};
      for (size_t i = 0; i < N; ++i) {
        float *dml = m->d_cur + i * d;
        for (size_t c = 0; c < d; ++c) {
          double dmlp_c = 0.0;
          for (size_t r = 0; r < mC; ++r) {
            double dy = (double)m->d_mhc_cur[(i * mC + r) * d + c];
            dmlp_c += hpost2[r] * dy;
            dhpost2[r] += dy * (double)a->mlp_out[i * d + c];
          }
          dml[c] = (float)dmlp_c;
        }
        for (size_t s = 0; s < mC; ++s) {
          float *dx2s = m->d_mhc_next + (i * mC + s) * d;
          for (size_t c = 0; c < d; ++c) {
            double acc = 0.0, xs = (double)a->mhc_x2[(i * mC + s) * d + c];
            for (size_t r = 0; r < mC; ++r) {
              double dy = (double)m->d_mhc_cur[(i * mC + r) * d + c];
              acc += hres2[r * mC + s] * dy;
              dhres2[r * mC + s] += dy * xs;
            }
            dx2s[c] = (float)acc;
          }
        }
      }
      double draw2[16];
      mhc_sinkhorn_bwd(mC, W + lo->mhc_hres2, dhres2, draw2);
      float *ghr2 = ext_grad_target(m, lo->mhc_hres2, mC * mC);
      for (size_t k = 0; k < mC * mC; ++k) ghr2[k] += (float)draw2[k];
      ext_grad_commit(m, lo->mhc_hres2, mC * mC, accumulate);
      float *ghpo2 = ext_grad_target(m, lo->mhc_hpost2, mC);
      for (size_t r = 0; r < mC; ++r) {
        double sg = 0.5 * hpost2[r];
        ghpo2[r] += (float)(dhpost2[r] * 2.0 * sg * (1.0 - sg));
      }
      ext_grad_commit(m, lo->mhc_hpost2, mC, accumulate);
    } else {
      memcpy(m->d_next, m->d_cur, N * d * 4u);
    }

    /* --- MLP backward --- */
    ext_linear_bwd_grad(m, 2, N, ff, d, a->hidden, W + lo->wdown, m->d_cur, m->d_hid, lo->wdown, 0.0f, accumulate);
    bcir_tensor_swiglu_backward_pool(m->pool, N * ff, a->gate, a->up, m->d_hid, m->d_gate, m->d_up);
    ext_linear_bwd_grad(m, 2, N, d, ff, a->n2, W + lo->wgate, m->d_gate, m->d_n, lo->wgate, 0.0f, accumulate);
    ext_linear_bwd_grad(m, 2, N, d, ff, a->n2, W + lo->wup, m->d_up, m->d_n, lo->wup, 1.0f, accumulate);
    const float *dn2_for_rms = m->d_n;
    if (m->sp.canon_mode >= BCIR_CANON_AC) {
      size_t cc = 32u, nt = (d + cc - 1u) / cc;
      float *gc = ext_grad_target(m, lo->canon_c, BCIR_EXT_CANON_K * d);
      canon_ctx cx = { B, T, d, cc, a->n2_raw, W + lo->canon_c, m->d_n, NULL, m->d_n_tmp, gc };
      bcir_tensor_pool_run(m->pool, canon_bwd_worker, &cx, nt);
      ext_grad_commit(m, lo->canon_c, BCIR_EXT_CANON_K * d, accumulate);
      dn2_for_rms = m->d_n_tmp;
    }

    if (mC > 1u) {
      ext_rms_bwd_grad(m, N, d, a->x2, W + lo->norm2, dn2_for_rms, m->d_next, lo->norm2, 0, accumulate);
      /* Add d_x2 back into d_mhc_next via hpre2 */
      double dhpre2[4] = {0.0};
      for (size_t i = 0; i < N; ++i) {
        const float *dx2i = m->d_next + i * d;
        for (size_t s = 0; s < mC; ++s) {
          float *dst = m->d_mhc_next + (i * mC + s) * d;
          const float *src = a->mhc_x2 + (i * mC + s) * d;
          for (size_t c = 0; c < d; ++c) {
            double dx = (double)dx2i[c];
            dst[c] = (float)((double)dst[c] + hpre2[s] * dx);
            dhpre2[s] += dx * (double)src[c];
          }
        }
      }
      float *ghpr2 = ext_grad_target(m, lo->mhc_hpre2, mC);
      for (size_t s = 0; s < mC; ++s) ghpr2[s] += (float)(dhpre2[s] * hpre2[s] * (1.0 - hpre2[s]));
      ext_grad_commit(m, lo->mhc_hpre2, mC, accumulate);

      /* Residual #1 backward through mHC */
      mhc_sinkhorn_fwd(mC, W + lo->mhc_hres1, hres1);
      for (size_t s = 0; s < mC; ++s) {
        hpre1[s] = sigmoid_d((double)W[lo->mhc_hpre1 + s]);
        hpost1[s] = 2.0 * sigmoid_d((double)W[lo->mhc_hpost1 + s]);
      }
      double dhres1[16] = {0.0}, dhpost1[4] = {0.0};
      for (size_t i = 0; i < N; ++i) {
        float *dmix = m->d_next + i * d;
        for (size_t c = 0; c < d; ++c) {
          double dmc = 0.0;
          for (size_t r = 0; r < mC; ++r) {
            double dy = (double)m->d_mhc_next[(i * mC + r) * d + c];
            dmc += hpost1[r] * dy;
            dhpost1[r] += dy * (double)a->mix_out[i * d + c];
          }
          dmix[c] = (float)dmc;
        }
        for (size_t s = 0; s < mC; ++s) {
          float *dx1s = m->d_mhc_cur + (i * mC + s) * d;
          for (size_t c = 0; c < d; ++c) {
            double acc = 0.0, xs = (double)a->mhc_x1[(i * mC + s) * d + c];
            for (size_t r = 0; r < mC; ++r) {
              double dy = (double)m->d_mhc_next[(i * mC + r) * d + c];
              acc += hres1[r * mC + s] * dy;
              dhres1[r * mC + s] += dy * xs;
            }
            dx1s[c] = (float)acc;
          }
        }
      }
      double draw1[16];
      mhc_sinkhorn_bwd(mC, W + lo->mhc_hres1, dhres1, draw1);
      float *ghr1 = ext_grad_target(m, lo->mhc_hres1, mC * mC);
      for (size_t k = 0; k < mC * mC; ++k) ghr1[k] += (float)draw1[k];
      ext_grad_commit(m, lo->mhc_hres1, mC * mC, accumulate);
      float *ghpo1 = ext_grad_target(m, lo->mhc_hpost1, mC);
      for (size_t r = 0; r < mC; ++r) {
        double sg = 0.5 * hpost1[r];
        ghpo1[r] += (float)(dhpost1[r] * 2.0 * sg * (1.0 - sg));
      }
      ext_grad_commit(m, lo->mhc_hpost1, mC, accumulate);
    } else {
      ext_rms_bwd_grad(m, N, d, a->x2, W + lo->norm2, dn2_for_rms, m->d_next, lo->norm2, 1, accumulate);
      memcpy(m->d_cur, m->d_next, N * d * 4u);
    }

    /* --- Mixer output projection + optional G1 gate backward --- */
    ext_linear_bwd_grad(m, 1, N, d, d, a->ctx, W + lo->wo, m->d_next, m->d_ctx, lo->wo, 0.0f, accumulate);
    if (m->sp.gate_mode == BCIR_GATE_G1_HEADWISE) {
      for (size_t i = 0; i < N; ++i) {
        for (size_t h = 0; h < H; ++h) {
          double gv = (double)a->g_val[i * H + h], dg = 0.0;
          const float *cr = a->ctx_raw + (i * H + h) * dh;
          float *dc_p = m->d_ctx + (i * H + h) * dh;
          for (size_t k = 0; k < dh; ++k) {
            double dck = (double)dc_p[k];
            dg += dck * (double)cr[k];
            dc_p[k] = (float)(dck * gv);
          }
          m->d_gpre[i * H + h] = (float)(dg * gv * (1.0 - gv));
        }
      }
    } else if (m->sp.gate_mode == BCIR_GATE_G1_LOWRANK) {
      for (size_t i = 0; i < N * d; ++i) {
        double gv = (double)a->g_val[i], dck = (double)m->d_ctx[i];
        m->d_gpre[i] = (float)(dck * (double)a->ctx_raw[i] * gv * (1.0 - gv));
        m->d_ctx[i] = (float)(dck * gv);
      }
    }

    /* --- Mixer core backward --- */
    if (lo->mixer == BCIR_MIXER_GQA || lo->mixer == BCIR_MIXER_SWA) {
      if (lo->mixer == BCIR_MIXER_GQA) {
        gqa_bwd_dispatch(m, B, T, a->q, a->k, a->v, a->p, a->lse, a->ctx_raw, m->d_ctx, m->d_q, m->d_k, m->d_v);
      } else {
        swa_bwd_ctx sc = { B, T, H, KH, dh, m->sp.swa_window, score_stride,
                           a->q, a->k, a->v, a->p, m->d_ctx, m->d_q, m->d_k, m->d_v,
                           m->attn_scores, m->attn_dk_loc, m->attn_dv_loc };
        bcir_tensor_pool_run(m->pool, swa_bwd_dq_worker, &sc, B * H);
        bcir_tensor_pool_run(m->pool, swa_bwd_dkv_worker, &sc, B * KH);
      }
      if (lo->use_rope) {
        bcir_tensor_rope_lent_pool(m->pool, B, T, H, dh, m->rope_hd, m->d_q, 1);
        bcir_tensor_rope_lent_pool(m->pool, B, T, KH, dh, m->rope_hd, m->d_k, 1);
      }
      if (m->sp.qk_norm) {
        ext_rms_bwd_grad(m, N * H, dh, a->q_raw, W + lo->q_norm, m->d_q, m->d_q, lo->q_norm, 0, accumulate);
        ext_rms_bwd_grad(m, N * KH, dh, a->k_raw, W + lo->k_norm, m->d_k, m->d_k, lo->k_norm, 0, accumulate);
      }
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wq, m->d_q, m->d_n, lo->wq, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, kvd, a->n1, W + lo->wk, m->d_k, m->d_n, lo->wk, 1.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, kvd, a->n1, W + lo->wv, m->d_v, m->d_n, lo->wv, 1.0f, accumulate);
    } else if (lo->mixer == BCIR_MIXER_DELTANET) {
      size_t C = m->sp.delta_chunk, nc = (T + C - 1u) / C;
      dn_bwd_ctx db = { B, T, H, dh, C, nc, a->q, a->k, a->v, a->alpha, a->beta, m->d_ctx,
                        a->dn_sprev, a->dn_W, m->d_q, m->d_k, m->d_v, m->d_a, m->d_b,
                        m->dn_work_dS, m->dn_work_g, m->dn_work_gam, m->dn_work_A,
                        m->dn_work_dW, m->dn_work_dA, m->dn_work_dg };
      bcir_tensor_pool_run(m->pool, dn_wy_bwd_worker, &db, B * H);
      dn_l2norm_bwd(N, H, dh, a->q, a->q_inv, m->d_q, m->d_q);
      dn_l2norm_bwd(N, H, dh, a->k, a->k_inv, m->d_k, m->d_k);
      ext_bias_bwd_grad(m, N, H, m->d_a, lo->balpha, accumulate);
      ext_bias_bwd_grad(m, N, H, m->d_b, lo->bbeta, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wq, m->d_q, m->d_n, lo->wq, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wk, m->d_k, m->d_n, lo->wk, 1.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wv, m->d_v, m->d_n, lo->wv, 1.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, H, a->n1, W + lo->walpha, m->d_a, m->d_n, lo->walpha, 1.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, H, a->n1, W + lo->wbeta, m->d_b, m->d_n, lo->wbeta, 1.0f, accumulate);
    } else if (lo->mixer == BCIR_MIXER_KDA || lo->mixer == BCIR_MIXER_GDN2) {
      int is_g2 = (lo->mixer == BCIR_MIXER_GDN2);
      ext_rms_bwd_grad(m, N * H, dh, a->cd_y, W + lo->o_norm, m->d_ctx, m->d_ctx, lo->o_norm, 0, accumulate);
      size_t C = m->sp.delta_chunk, nc = (T + C - 1u) / C;
      cdelta_bwd_ctx cb = { B, T, H, dh, C, nc, is_g2, a->q, a->k, a->v, a->alpha,
                            is_g2 ? NULL : a->beta, is_g2 ? a->erase : NULL, is_g2 ? a->write : NULL,
                            m->d_ctx, a->dn_sprev, a->dn_W,
                            m->d_q, m->d_k, m->d_v, m->d_a, is_g2 ? NULL : m->d_b,
                            is_g2 ? m->d_e : NULL, is_g2 ? m->d_w : NULL,
                            m->dn_work_dS, m->dn_work_Shist, m->dn_work_dP, m->dn_work_qS };
      bcir_tensor_pool_run(m->pool, cdelta_wy_bwd_worker, &cb, B * H);
      dn_l2norm_bwd(N, H, dh, a->q, a->q_inv, m->d_q, m->d_q);
      dn_l2norm_bwd(N, H, dh, a->k, a->k_inv, m->d_k, m->d_k);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wq, m->d_q, m->d_n, lo->wq, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wk, m->d_k, m->d_n, lo->wk, 1.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wv, m->d_v, m->d_n, lo->wv, 1.0f, accumulate);
      /* Alpha low-rank backward */
      ext_bias_bwd_grad(m, N, d, m->d_a, lo->ba_chan, accumulate);
      ext_linear_bwd_grad(m, 1, N, rg, d, a->a_mid, W + lo->wa2, m->d_a, m->d_mid, lo->wa2, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, rg, a->n1, W + lo->wa1, m->d_mid, m->d_n, lo->wa1, 1.0f, accumulate);
      if (!is_g2) {
        ext_bias_bwd_grad(m, N, H, m->d_b, lo->bbeta, accumulate);
        ext_linear_bwd_grad(m, 1, N, d, H, a->n1, W + lo->wbeta, m->d_b, m->d_n, lo->wbeta, 1.0f, accumulate);
      } else {
        ext_bias_bwd_grad(m, N, d, m->d_e, lo->be_chan, accumulate);
        ext_linear_bwd_grad(m, 1, N, rg, d, a->e_mid, W + lo->we2, m->d_e, m->d_mid, lo->we2, 0.0f, accumulate);
        ext_linear_bwd_grad(m, 1, N, d, rg, a->n1, W + lo->we1, m->d_mid, m->d_n, lo->we1, 1.0f, accumulate);
        ext_bias_bwd_grad(m, N, d, m->d_w, lo->bw_chan, accumulate);
        ext_linear_bwd_grad(m, 1, N, rg, d, a->w_mid, W + lo->ww2, m->d_w, m->d_mid, lo->ww2, 0.0f, accumulate);
        ext_linear_bwd_grad(m, 1, N, d, rg, a->n1, W + lo->ww1, m->d_mid, m->d_n, lo->ww1, 1.0f, accumulate);
      }
    } else if (lo->mixer == BCIR_MIXER_MLA) {
      mla_att_ctx mc = { B, T, H, dh, dr, lo->use_rope, score_stride,
                         a->q_c, a->q_r, a->k_c, a->v_c, a->k_r, a->ctx_raw, m->d_ctx,
                         a->mla_p, NULL, m->d_qc, m->d_qr, m->d_kc, m->d_vc, m->d_kr,
                         a->lse, 0, 0.0, m->attn_scores, NULL, NULL, m->attn_dkr_loc, NULL };
      bcir_tensor_pool_run(m->pool, m->sp.online_softmax ? mla_online_bwd_bh_worker : mla_bwd_bh_worker, &mc, B * H);
      if (lo->use_rope) {
        bcir_tensor_pool_run(m->pool, m->sp.online_softmax ? mla_online_bwd_dkr_worker : mla_bwd_dkr_worker, &mc, B);
        bcir_tensor_rope_lent_pool(m->pool, B, T, H, dr, m->rope_dr, m->d_qr, 1);
        bcir_tensor_rope_lent_pool(m->pool, B, T, 1u, dr, m->rope_dr, m->d_kr, 1);
      }
      if (m->sp.qk_norm) {
        ext_rms_bwd_grad(m, N * H, dh, a->q_c_raw, W + lo->q_norm, m->d_qc, m->d_qc, lo->q_norm, 0, accumulate);
      }
      ext_linear_bwd_grad(m, 1, N, dc, d, a->c_hat, W + lo->w_uk, m->d_kc, m->d_chat, lo->w_uk, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, dc, d, a->c_hat, W + lo->w_uv, m->d_vc, m->d_chat, lo->w_uv, 1.0f, accumulate);
      ext_rms_bwd_grad(m, N, dc, a->c_raw, W + lo->norm_kv, m->d_chat, m->d_craw, lo->norm_kv, 0, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, d, a->n1, W + lo->wq_c, m->d_qc, m->d_n, lo->wq_c, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, dc, a->n1, W + lo->w_dkv, m->d_craw, m->d_n, lo->w_dkv, 1.0f, accumulate);
      if (lo->use_rope) {
        ext_linear_bwd_grad(m, 1, N, d, H * dr, a->n1, W + lo->wq_r, m->d_qr, m->d_n, lo->wq_r, 1.0f, accumulate);
        ext_linear_bwd_grad(m, 1, N, d, dr, a->n1, W + lo->w_kr, m->d_kr, m->d_n, lo->w_kr, 1.0f, accumulate);
      }
    }

    /* Accumulate G1 output gate backward into m->d_n (dx_beta = 1.0f) */
    if (m->sp.gate_mode == BCIR_GATE_G1_HEADWISE) {
      ext_bias_bwd_grad(m, N, H, m->d_gpre, lo->bg, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, H, a->n1, W + lo->wg1, m->d_gpre, m->d_n, lo->wg1, 1.0f, accumulate);
    } else if (m->sp.gate_mode == BCIR_GATE_G1_LOWRANK) {
      ext_bias_bwd_grad(m, N, d, m->d_gpre, lo->bg, accumulate);
      ext_linear_bwd_grad(m, 1, N, rg, d, a->g_mid, W + lo->wg2, m->d_gpre, m->d_mid, lo->wg2, 0.0f, accumulate);
      ext_linear_bwd_grad(m, 1, N, d, rg, a->n1, W + lo->wg1, m->d_mid, m->d_n, lo->wg1, 1.0f, accumulate);
    }

    /* Optional Canon-A backward + norm1 backward */
    const float *dn1_for_rms = m->d_n;
    if (m->sp.canon_mode >= BCIR_CANON_A) {
      size_t cc = 32u, nt = (d + cc - 1u) / cc;
      float *gc = ext_grad_target(m, lo->canon_a, BCIR_EXT_CANON_K * d);
      canon_ctx cx = { B, T, d, cc, a->n1_raw, W + lo->canon_a, m->d_n, NULL, m->d_n_tmp, gc };
      bcir_tensor_pool_run(m->pool, canon_bwd_worker, &cx, nt);
      ext_grad_commit(m, lo->canon_a, BCIR_EXT_CANON_K * d, accumulate);
      dn1_for_rms = m->d_n_tmp;
    }

    if (mC > 1u) {
      ext_rms_bwd_grad(m, N, d, a->x_in, W + lo->norm1, dn1_for_rms, m->d_cur, lo->norm1, 0, accumulate);
      double dhpre1[4] = {0.0};
      for (size_t i = 0; i < N; ++i) {
        const float *dx1i = m->d_cur + i * d;
        for (size_t s = 0; s < mC; ++s) {
          float *dst = m->d_mhc_cur + (i * mC + s) * d;
          const float *src = a->mhc_x1 + (i * mC + s) * d;
          for (size_t c = 0; c < d; ++c) {
            double dx = (double)dx1i[c];
            dst[c] = (float)((double)dst[c] + hpre1[s] * dx);
            dhpre1[s] += dx * (double)src[c];
          }
        }
      }
      float *ghpr1 = ext_grad_target(m, lo->mhc_hpre1, mC);
      for (size_t s = 0; s < mC; ++s) ghpr1[s] += (float)(dhpre1[s] * hpre1[s] * (1.0 - hpre1[s]));
      ext_grad_commit(m, lo->mhc_hpre1, mC, accumulate);
    } else {
      ext_rms_bwd_grad(m, N, d, a->x_in, W + lo->norm1, dn1_for_rms, m->d_cur, lo->norm1, 1, accumulate);
    }
  }

  /* Collapse mHC input streams back to embedding gradient */
  if (mC > 1u) {
    for (size_t i = 0; i < N; ++i) {
      float *dst = m->d_cur + i * d;
      for (size_t c = 0; c < d; ++c) {
        double acc = 0.0;
        for (size_t s = 0; s < mC; ++s) acc += (double)m->d_mhc_cur[(i * mC + s) * d + c];
        dst[c] = (float)acc;
      }
    }
  }

  /* Deterministic embedding scatter */
  {
    size_t cc = 32u, nt = (d + cc - 1u) / cc;
    float *gemb = ext_grad_target(m, m->off_embed, V * d);
    emb_scat_ctx ec = { N, d, cc, tokens, m->d_cur, gemb };
    bcir_tensor_pool_run(m->pool, emb_scat_worker, &ec, nt);
    ext_grad_commit(m, m->off_embed, V * d, accumulate || m->sp.tied);
  }
  return 0;
}

int bcir_ext_train_loop(bcir_ext_model *m, const uint32_t *tokens, const uint32_t *targets,
                        const uint32_t *active_times_or_null, const double *lrs,
                        size_t steps, const bcir_adamw_params *opt_cfg,
                        bcir_ext_step_metrics *events) {
  if (!m || !tokens || !targets || !opt_cfg) return 1;
  size_t stride = (size_t)m->sp.batch * m->sp.time;
  for (size_t s = 0; s < steps; ++s) {
    double t0 = now_ms();
    uint32_t act_T = active_times_or_null ? active_times_or_null[s] : m->sp.time;
    const uint32_t *tk = tokens + s * stride, *tg = targets + s * stride;
    double loss = 0.0, gnorm = 0.0; uint64_t sup = 0u; int clipped = 0;
    int rc = bcir_ext_loss_backward(m, tk, tg, act_T, 0, &loss, &sup);
    if (rc == 0) {
      bcir_adamw_params cur = *opt_cfg;
      if (lrs) cur.lr = lrs[s];
      rc = bcir_opt_step(m->pool, m->opt, m->weights, NULL, &cur, 1.0, &gnorm, &clipped, &m->tel);
    }
    if (events) {
      events[s].loss = loss; events[s].grad_norm = gnorm; events[s].step_ms = now_ms() - t0;
      events[s].clipped = clipped; events[s].status = rc; events[s].step = m->opt->step;
      events[s].supervised_tokens = sup;
      events[s].fp64_promotions = m->tel.fp64_norm_promotions + m->tel.fp64_softmax_promotions + m->tel.fp64_wy_solves;
      events[s].nonfinite_events = m->tel.nonfinite_events;
    }
    if (rc != 0) return rc;
  }
  return 0;
}

/* ==================== Gate 0 Mathematical Identity Verifiers ==================== */
int bcir_deltanet_verify_wy_vs_recurrent(uint32_t B, uint32_t T, uint32_t H, uint32_t dh,
                                         uint32_t C, double *out_max_abs_diff) {
  if (B == 0u || T == 0u || H == 0u || dh == 0u || C == 0u) return 1;
  size_t N = (size_t)B * T, nc = (T + C - 1u) / C, nt = (size_t)B * H;
  float *q = (float *)ext_zalloc(N * H * dh, 4u), *k = (float *)ext_zalloc(N * H * dh, 4u);
  float *v = (float *)ext_zalloc(N * H * dh, 4u), *al = (float *)ext_zalloc(N * H, 4u), *be = (float *)ext_zalloc(N * H, 4u);
  float *y_wy = (float *)ext_zalloc(N * H * dh, 4u);
  double *sprev = (double *)ext_zalloc(nt * nc * dh * dh, sizeof(double));
  double *Wall = (double *)ext_zalloc(nt * T * dh, sizeof(double));
  double *wS = (double *)ext_zalloc(nt * dh * dh, sizeof(double));
  double *wg = (double *)ext_zalloc(nt * C * dh, sizeof(double));
  double *wgam = (double *)ext_zalloc(nt * C * dh, sizeof(double));
  double *wA = (double *)ext_zalloc(nt * C * C, sizeof(double));
  double *wqS = (double *)ext_zalloc(nt * dh, sizeof(double));
  double *mo = (double *)ext_zalloc(nt, sizeof(double));
  uint64_t *wc = (uint64_t *)ext_zalloc(nt, sizeof(uint64_t));
  float *qi = (float *)ext_zalloc(N * H, 4u), *ki = (float *)ext_zalloc(N * H, 4u);
  double *S = (double *)ext_zalloc((size_t)dh * dh, sizeof(double)), *kS = (double *)ext_zalloc(dh, sizeof(double));
  int ok = (q && k && v && al && be && y_wy && sprev && Wall && wS && wg && wgam && wA && wqS && mo && wc && qi && ki && S && kS);
  double max_diff = 0.0;
  if (ok) {
    uint64_t r = 123456789ULL;
    for (size_t i = 0; i < N * H * dh; ++i) {
      r = r * 6364136223846793005ULL + 1ULL; q[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f);
      r = r * 6364136223846793005ULL + 1ULL; k[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f);
      r = r * 6364136223846793005ULL + 1ULL; v[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f);
    }
    dn_l2norm_fwd(N, H, dh, q, q, qi); dn_l2norm_fwd(N, H, dh, k, k, ki);
    for (size_t i = 0; i < N * H; ++i) {
      r = r * 6364136223846793005ULL + 1ULL; al[i] = 0.7f + 0.28f * ((float)(r >> 40) / (float)(1 << 24));
      r = r * 6364136223846793005ULL + 1ULL; be[i] = 0.05f + 0.85f * ((float)(r >> 40) / (float)(1 << 24));
    }
    dn_fwd_ctx ctx = { B, T, H, dh, C, nc, q, k, v, al, be, sprev, Wall, y_wy, wS, wg, wgam, wA, wqS, mo, wc };
    for (size_t t = 0; t < nt; ++t) dn_wy_fwd_worker(&ctx, t);

    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H; ++h) {
        memset(S, 0, (size_t)dh * dh * sizeof(double));
        for (size_t t = 0; t < T; ++t) {
          const float *qt = q + ((b * T + t) * H + h) * dh;
          const float *kt = k + ((b * T + t) * H + h) * dh;
          const float *vt = v + ((b * T + t) * H + h) * dh;
          double a = (double)al[(b * T + t) * H + h], bt = (double)be[(b * T + t) * H + h];
          for (size_t d = 0; d < dh; ++d) {
            double acc = 0.0; for (size_t j = 0; j < dh; ++j) acc += (double)kt[j] * S[j * dh + d];
            kS[d] = acc;
          }
          for (size_t j = 0; j < dh; ++j)
            for (size_t d = 0; d < dh; ++d)
              S[j * dh + d] = a * (S[j * dh + d] - bt * (double)kt[j] * kS[d]) + bt * (double)kt[j] * (double)vt[d];
          const float *yt = y_wy + ((b * T + t) * H + h) * dh;
          for (size_t d = 0; d < dh; ++d) {
            double yr = 0.0; for (size_t j = 0; j < dh; ++j) yr += (double)qt[j] * S[j * dh + d];
            double diff = fabs((double)yt[d] - (double)(float)yr);
            if (diff > max_diff) max_diff = diff;
          }
        }
      }
    }
  }
  ext_dealloc(q); ext_dealloc(k); ext_dealloc(v); ext_dealloc(al); ext_dealloc(be); ext_dealloc(y_wy); ext_dealloc(sprev); ext_dealloc(Wall);
  ext_dealloc(wS); ext_dealloc(wg); ext_dealloc(wgam); ext_dealloc(wA); ext_dealloc(wqS); ext_dealloc(qi); ext_dealloc(ki); ext_dealloc(mo); ext_dealloc(wc); ext_dealloc(S); ext_dealloc(kS);
  if (out_max_abs_diff) *out_max_abs_diff = max_diff;
  return (ok && max_diff < 1e-5) ? 0 : 1;
}

static int cdelta_verify_common(uint32_t B, uint32_t T, uint32_t H, uint32_t dh, uint32_t C,
                                int is_gdn2, double *out_max_abs_diff) {
  if (B == 0u || T == 0u || H == 0u || dh == 0u || C == 0u) return 1;
  size_t N = (size_t)B * T, nc = (T + C - 1u) / C, nt = (size_t)B * H, nd = N * H * dh;
  float *q = (float *)ext_zalloc(nd, 4u), *k = (float *)ext_zalloc(nd, 4u), *v = (float *)ext_zalloc(nd, 4u);
  float *al = (float *)ext_zalloc(nd, 4u), *be = (float *)ext_zalloc(N * H, 4u);
  float *er = (float *)ext_zalloc(nd, 4u), *wr = (float *)ext_zalloc(nd, 4u);
  float *y_wy = (float *)ext_zalloc(nd, 4u);
  double *sprev = (double *)ext_zalloc(nt * nc * dh * dh, sizeof(double));
  double *Wall = (double *)ext_zalloc(nt * T * dh, sizeof(double));
  double *wS = (double *)ext_zalloc(nt * dh * dh, sizeof(double));
  double *wg = (double *)ext_zalloc(nt * C * dh, sizeof(double));
  double *wgam = (double *)ext_zalloc(nt * C * dh, sizeof(double));
  double *wA = (double *)ext_zalloc(nt * C * C, sizeof(double));
  double *wqS = (double *)ext_zalloc(nt * dh, sizeof(double));
  double *mo = (double *)ext_zalloc(nt, sizeof(double));
  uint64_t *wc = (uint64_t *)ext_zalloc(nt, sizeof(uint64_t));
  float *qi = (float *)ext_zalloc(N * H, 4u), *ki = (float *)ext_zalloc(N * H, 4u);
  double *S = (double *)ext_zalloc((size_t)dh * dh, sizeof(double)), *kS = (double *)ext_zalloc(dh, sizeof(double));
  int ok = (q && k && v && al && be && er && wr && y_wy && sprev && Wall && wS && wg && wgam && wA && wqS && mo && wc && qi && ki && S && kS);
  double max_diff = 0.0;
  if (ok) {
    uint64_t r = is_gdn2 ? 0xC0FFEE1234ULL : 0xBEEF56789ULL;
    for (size_t i = 0; i < nd; ++i) {
      r = r * 6364136223846793005ULL + 1ULL; q[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f);
      r = r * 6364136223846793005ULL + 1ULL; k[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f);
      r = r * 6364136223846793005ULL + 1ULL; v[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f);
      r = r * 6364136223846793005ULL + 1ULL; al[i] = 0.65f + 0.33f * ((float)(r >> 40) / (float)(1 << 24));
      r = r * 6364136223846793005ULL + 1ULL; er[i] = 0.05f + 0.85f * ((float)(r >> 40) / (float)(1 << 24));
      r = r * 6364136223846793005ULL + 1ULL; wr[i] = 0.05f + 0.85f * ((float)(r >> 40) / (float)(1 << 24));
    }
    dn_l2norm_fwd(N, H, dh, q, q, qi); dn_l2norm_fwd(N, H, dh, k, k, ki);
    for (size_t i = 0; i < N * H; ++i) {
      r = r * 6364136223846793005ULL + 1ULL; be[i] = 0.05f + 0.85f * ((float)(r >> 40) / (float)(1 << 24));
    }
    cdelta_fwd_ctx ctx = { B, T, H, dh, C, nc, is_gdn2, q, k, v, al, is_gdn2 ? NULL : be,
                           is_gdn2 ? er : NULL, is_gdn2 ? wr : NULL, sprev, Wall, y_wy,
                           wS, wg, wgam, wA, wqS, mo, wc };
    for (size_t t = 0; t < nt; ++t) cdelta_wy_fwd_worker(&ctx, t);

    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H; ++h) {
        memset(S, 0, (size_t)dh * dh * sizeof(double));
        for (size_t t = 0; t < T; ++t) {
          size_t ti = b * T + t;
          const float *qt = q + (ti * H + h) * dh, *kt = k + (ti * H + h) * dh, *vt = v + (ti * H + h) * dh;
          const float *alt = al + (ti * H + h) * dh;
          const float *ert = is_gdn2 ? (er + (ti * H + h) * dh) : NULL;
          const float *wrt = is_gdn2 ? (wr + (ti * H + h) * dh) : NULL;
          double bt_s = is_gdn2 ? 0.0 : (double)be[ti * H + h];
          /* P = Diag(alpha) S_{t-1}, kS[d] = sum_a (b_a * k_a * P[a,d]) */
          for (size_t d = 0; d < dh; ++d) {
            double acc = 0.0;
            for (size_t a = 0; a < dh; ++a) {
              double ba = is_gdn2 ? (double)ert[a] : bt_s;
              acc += ba * (double)kt[a] * (double)alt[a] * S[a * dh + d];
            }
            kS[d] = acc;
          }
          for (size_t a = 0; a < dh; ++a) {
            double ala = (double)alt[a], ka = (double)kt[a];
            for (size_t d = 0; d < dh; ++d) {
              double wd = is_gdn2 ? (double)wrt[d] : bt_s;
              S[a * dh + d] = ala * S[a * dh + d] - ka * kS[d] + ka * wd * (double)vt[d];
            }
          }
          const float *yt = y_wy + (ti * H + h) * dh;
          for (size_t d = 0; d < dh; ++d) {
            double yr = 0.0;
            for (size_t a = 0; a < dh; ++a) yr += (double)qt[a] * S[a * dh + d];
            double diff = fabs((double)yt[d] - (double)(float)yr);
            if (diff > max_diff) max_diff = diff;
          }
        }
      }
    }
  }
  ext_dealloc(q); ext_dealloc(k); ext_dealloc(v); ext_dealloc(al); ext_dealloc(be); ext_dealloc(er); ext_dealloc(wr); ext_dealloc(y_wy);
  ext_dealloc(sprev); ext_dealloc(Wall); ext_dealloc(wS); ext_dealloc(wg); ext_dealloc(wgam); ext_dealloc(wA); ext_dealloc(wqS);
  ext_dealloc(qi); ext_dealloc(ki); ext_dealloc(mo); ext_dealloc(wc); ext_dealloc(S); ext_dealloc(kS);
  if (out_max_abs_diff) *out_max_abs_diff = max_diff;
  return (ok && max_diff < 1e-5) ? 0 : 1;
}

int bcir_kda_verify_wy_vs_recurrent(uint32_t B, uint32_t T, uint32_t H, uint32_t dh,
                                    uint32_t C, double *out_max_abs_diff) {
  return cdelta_verify_common(B, T, H, dh, C, 0, out_max_abs_diff);
}

int bcir_gdn2_verify_wy_vs_recurrent(uint32_t B, uint32_t T, uint32_t H, uint32_t dh,
                                     uint32_t C, double *out_max_abs_diff) {
  return cdelta_verify_common(B, T, H, dh, C, 1, out_max_abs_diff);
}

int bcir_mla_verify_absorption(uint32_t T, uint32_t H, uint32_t dh, uint32_t dc, uint32_t dr,
                               double *out_max_abs_diff) {
  if (T == 0u || H == 0u || dh == 0u || dc == 0u) return 1;
  size_t d = (size_t)H * dh;
  float *qc = (float *)ext_zalloc(T * d, 4u), *qr = (float *)ext_zalloc(dr ? (size_t)T * H * dr : 1u, 4u);
  float *chat = (float *)ext_zalloc(T * dc, 4u), *kr = (float *)ext_zalloc(dr ? (size_t)T * dr : 1u, 4u);
  float *wuk = (float *)ext_zalloc(d * dc, 4u), *wuv = (float *)ext_zalloc(d * dc, 4u);
  float *kc = (float *)ext_zalloc(T * d, 4u), *vc = (float *)ext_zalloc(T * d, 4u);
  double *s_u = (double *)ext_zalloc(T, sizeof(double)), *s_a = (double *)ext_zalloc(T, sizeof(double));
  double *q_abs = (double *)ext_zalloc(dc, sizeof(double)), *c_ctx = (double *)ext_zalloc(dc, sizeof(double));
  int ok = (qc && qr && chat && kr && wuk && wuv && kc && vc && s_u && s_a && q_abs && c_ctx);
  double max_diff = 0.0;
  if (ok) {
    uint64_t r = 987654321ULL;
    for (size_t i = 0; i < T * d; ++i) { r = r * 6364136223846793005ULL + 1; qc[i] = 0.2f * ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f); }
    for (size_t i = 0; i < (size_t)T * H * dr; ++i) { r = r * 6364136223846793005ULL + 1; qr[i] = 0.2f * ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f); }
    for (size_t i = 0; i < (size_t)T * dc; ++i) { r = r * 6364136223846793005ULL + 1; chat[i] = 0.2f * ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f); }
    for (size_t i = 0; i < (size_t)T * dr; ++i) { r = r * 6364136223846793005ULL + 1; kr[i] = 0.2f * ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f); }
    for (size_t i = 0; i < d * dc; ++i) { r = r * 6364136223846793005ULL + 1; wuk[i] = 0.2f * ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f); }
    for (size_t i = 0; i < d * dc; ++i) { r = r * 6364136223846793005ULL + 1; wuv[i] = 0.2f * ((float)(int32_t)(r >> 40) / (float)(1 << 23) - 1.0f); }

    bcir_tensor_mm(NULL, 0, 1, T, d, dc, 1.0f, chat, wuk, 0.0f, kc);
    bcir_tensor_mm(NULL, 0, 1, T, d, dc, 1.0f, chat, wuv, 0.0f, vc);

    double scale = 1.0 / sqrt((double)(dh + dr));
    for (size_t h = 0; h < H; ++h) {
      const float *wuk_h = wuk + h * dh * dc, *wuv_h = wuv + h * dh * dc;
      for (size_t i = 0; i < T; ++i) {
        const float *qci = qc + (i * H + h) * dh;
        const float *qri = dr ? (qr + (i * H + h) * dr) : NULL;
        for (size_t c = 0; c < dc; ++c) {
          double acc = 0.0; for (size_t a = 0; a < dh; ++a) acc += (double)wuk_h[a * dc + c] * (double)qci[a];
          q_abs[c] = acc;
        }
        double mu = -1e30, ma = -1e30;
        for (size_t j = 0; j <= i; ++j) {
          const float *kcj = kc + (j * H + h) * dh, *chj = chat + j * dc;
          double su = 0.0, sa = 0.0, sr = 0.0;
          for (size_t a = 0; a < dh; ++a) su += (double)qci[a] * (double)kcj[a];
          for (size_t c = 0; c < dc; ++c) sa += q_abs[c] * (double)chj[c];
          if (dr) {
            const float *krj = kr + j * dr;
            for (size_t a = 0; a < dr; ++a) sr += (double)qri[a] * (double)krj[a];
          }
          s_u[j] = (su + sr) * scale; s_a[j] = (sa + sr) * scale;
          if (s_u[j] > mu) mu = s_u[j];
          if (s_a[j] > ma) ma = s_a[j];
        }
        double zu = 0.0, za = 0.0;
        for (size_t j = 0; j <= i; ++j) { s_u[j] = exp(s_u[j] - mu); zu += s_u[j]; s_a[j] = exp(s_a[j] - ma); za += s_a[j]; }
        memset(c_ctx, 0, dc * sizeof(double));
        for (size_t j = 0; j <= i; ++j) {
          double pa = s_a[j] / za; const float *chj = chat + j * dc;
          for (size_t c = 0; c < dc; ++c) c_ctx[c] += pa * (double)chj[c];
        }
        for (size_t a = 0; a < dh; ++a) {
          double yu = 0.0, ya = 0.0;
          for (size_t j = 0; j <= i; ++j) yu += (s_u[j] / zu) * (double)vc[(j * H + h) * dh + a];
          for (size_t c = 0; c < dc; ++c) ya += (double)wuv_h[a * dc + c] * c_ctx[c];
          double diff = fabs(yu - ya);
          if (diff > max_diff) max_diff = diff;
        }
      }
    }
  }
  ext_dealloc(qc); ext_dealloc(qr); ext_dealloc(chat); ext_dealloc(kr); ext_dealloc(wuk); ext_dealloc(wuv); ext_dealloc(kc); ext_dealloc(vc);
  ext_dealloc(s_u); ext_dealloc(s_a); ext_dealloc(q_abs); ext_dealloc(c_ctx);
  if (out_max_abs_diff) *out_max_abs_diff = max_diff;
  return (ok && max_diff < 1e-6) ? 0 : 1;
}
