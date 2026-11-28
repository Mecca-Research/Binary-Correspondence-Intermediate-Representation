/*===- bcir_precision.c - Executable BCIR Precision Policy (P1-P6) Engine ---===
 * Pure ISO C11 implementation (zero POSIX-only APIs, zero hot-path malloc in
 * bcir_opt_step, deterministic across 1T..NT, compiled with -ffp-contract=off).
 *===------------------------------------------------------------------------===*/
#include "bcir_precision.h"
#include "bcir_host_alloc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512BF16__) && defined(__AVX512F__)
#include <immintrin.h>
#define BCIR_HAS_AVX512_BF16 1
#else
#define BCIR_HAS_AVX512_BF16 0
#endif

static void *prec_alloc(size_t bytes) {
  bcir_host_allocator a = bcir_host_allocator_default();
  return bcir_host_allocate(&a, bytes);
}

static void *prec_zalloc(size_t count, size_t elem) {
  size_t bytes = 0u;
  if (!bcir_size_mul(count, elem, &bytes)) return NULL;
  bcir_host_allocator a = bcir_host_allocator_default();
  return bcir_host_allocate_zeroed(&a, bytes);
}

static void prec_dealloc(void *ptr) {
  bcir_host_allocator a = bcir_host_allocator_default();
  bcir_host_deallocate(&a, ptr);
}

void *bcir_aligned_calloc64(size_t count, size_t elem_size) {
  if (elem_size != 0u && count > SIZE_MAX / elem_size) return NULL;
  size_t bytes = count * elem_size;
  if (bytes == 0u) bytes = 64u;
  if (bytes > SIZE_MAX - 64u - sizeof(void *)) return NULL;
  size_t total = bytes + 64u + sizeof(void *);
  void *raw = prec_alloc(total);
  if (!raw) return NULL;
  uintptr_t start = (uintptr_t)raw + sizeof(void *);
  uintptr_t aligned = (start + 63u) & ~(uintptr_t)63u;
  ((void **)aligned)[-1] = raw;
  memset((void *)aligned, 0, bytes);
  return (void *)aligned;
}

void bcir_aligned_free64(void *ptr) {
  if (!ptr) return;
  prec_dealloc(((void **)ptr)[-1]);
}

static const bcir_num_contract_desc g_gemm_descs[BCIR_GEMM_KIND_COUNT] = {
  { BCIR_GEMM_FP32_KBLOCK, "fp32_kblock", "IEEE binary32", "8-bit bias 127",
    "RNE (fused kblock=448)", "FP32", "ascending KBLOCK=448", "IEEE subnormal", "IEEE +/-Inf", 1e-6, 1, 32.0 },
  { BCIR_GEMM_BF16_NDT, "bf16_ndt_amx", "BF16 (1s8e7m)", "8-bit bias 127",
    "RNE (VCVTNEPS2BF16)", "FP32 (even/odd TDPBF16PS)", "32-depth blocks", "FTZ/DAZ", "IEEE +/-Inf", 1e-2, 32, 16.0 },
  { BCIR_GEMM_BF16_VDP, "bf16_vdp_avx512", "BF16 (1s8e7m)", "8-bit bias 127",
    "RNE (VCVTNEPS2BF16)", "FP32 (exact pair VDPBF16PS fma1->fma0)", "ascending 2-term pairs", "FTZ/DAZ", "IEEE +/-Inf", 1e-2, 2, 16.0 },
  { BCIR_GEMM_FP8_E4M3_BLK32, "fp8_e4m3_blk32", "FP8 E4M3FN (1s4e3m)", "4-bit bias 7 + FP32 blk32 scale",
    "RNE", "FP32 (kblock)", "ascending KBLOCK=448", "FP8 subnormal", "saturate +/-448", 6.25e-2, 32, 9.0 },
  { BCIR_GEMM_FP8_E5M2_BLK32, "fp8_e5m2_blk32", "FP8 E5M2 (1s5e2m)", "5-bit bias 15 + FP32 blk32 scale",
    "RNE", "FP32 (kblock)", "ascending KBLOCK=448", "FP8 subnormal", "saturate +/-57344", 1.25e-1, 32, 9.0 },
  { BCIR_GEMM_FP4_MX_BLK32, "mxfp4_e2m1_blk32", "MXFP4 E2M1 (1s2e1m)", "2-bit + shared E8M0 blk32",
    "RNE", "FP32 (kblock)", "ascending KBLOCK=448", "E2M1 {0, 0.5}", "saturate +/-6.0", 2.5e-1, 32, 4.25 },
  { BCIR_GEMM_Q8_BLK32, "bcirq8_g32_pow2", "INT8 [-127,127]", "shared pow2 2^e blk32",
    "RNE (|w-w_hat|<=2^(e-1))", "FP32 (kblock)", "ascending KBLOCK=448", "zero", "saturate +/-127", 8.0e-3, 32, 8.5 }
};

static const char *g_opt_names[BCIR_OPT_KIND_COUNT] = {
  "adamw_fp32", "adamw_q8_moments", "adamw_q8_moments_bf16_grad",
  "adamw_adafactor_m2", "adamw_adafactor_bf16_grad",
  "muon_fp32", "muon_q8_bf16_grad"
};

const bcir_num_contract_desc *bcir_prec_gemm_desc(int gemm_kind) {
  return (gemm_kind >= 0 && gemm_kind < BCIR_GEMM_KIND_COUNT) ? &g_gemm_descs[gemm_kind] : NULL;
}
const char *bcir_prec_opt_name(int opt_kind) {
  return (opt_kind >= 0 && opt_kind < BCIR_OPT_KIND_COUNT) ? g_opt_names[opt_kind] : "unknown";
}
void bcir_prec_policy_default(bcir_prec_policy *pol) {
  if (!pol) return;
  memset(pol, 0, sizeof(*pol));
  pol->fp64_guard_mode = BCIR_FP64_GUARD_CONDITIONAL;
  pol->rmsnorm_cond_thresh = 1e-4;
  pol->softmax_logit_thresh = 25.0;
}
void bcir_prec_telemetry_reset(bcir_prec_telemetry *t) {
  if (!t) return;
  memset(t, 0, sizeof(*t));
  t->min_rms_observed = 1e30;
}

static inline uint32_t f32_to_u32(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static inline float u32_to_f32(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static inline float bf16_to_f32(uint16_t b) { return u32_to_f32((uint32_t)b << 16); }
static inline uint16_t f32_to_bf16(float x) { return (uint16_t)bcir_tensor_bf16_bits(x); }
static inline float flush_subnormal_f32(float x) {
  uint32_t u = f32_to_u32(x);
  return (((u >> 23) & 0xFFu) == 0u) ? u32_to_f32(u & 0x80000000u) : x;
}

float bcir_quant_fp8_e4m3_scalar(float x) {
  if (!isfinite(x)) return x;
  float sgn = (x < 0.0f) ? -1.0f : 1.0f;
  float ax = fabsf(x);
  if (ax >= 448.0f) return sgn * 448.0f;
  if (ax == 0.0f) return 0.0f;
  if (ax < 0.015625f) return sgn * (rintf(ax * 512.0f) * (1.0f / 512.0f));
  int e;
  float m = frexpf(ax, &e);
  float val = ldexpf(rintf(m * 16.0f), e - 4);
  if (val > 448.0f) val = 448.0f;
  return sgn * val;
}

float bcir_quant_fp8_e5m2_scalar(float x) {
  if (!isfinite(x)) return x;
  float sgn = (x < 0.0f) ? -1.0f : 1.0f;
  float ax = fabsf(x);
  if (ax >= 57344.0f) return sgn * 57344.0f;
  if (ax == 0.0f) return 0.0f;
  if (ax < (1.0f / 16384.0f)) return sgn * (rintf(ax * 65536.0f) * (1.0f / 65536.0f));
  int e;
  float m = frexpf(ax, &e);
  float val = ldexpf(rintf(m * 8.0f), e - 3);
  if (val > 57344.0f) val = 57344.0f;
  return sgn * val;
}

float bcir_quant_fp4_e2m1_scalar(float x) {
  if (!isfinite(x)) return x;
  float sgn = (x < 0.0f) ? -1.0f : 1.0f;
  float ax = fabsf(x);
  float q = (ax <= 0.25f) ? 0.0f : (ax < 0.75f) ? 0.5f : (ax <= 1.25f) ? 1.0f :
            (ax < 1.75f) ? 1.5f : (ax <= 2.50f) ? 2.0f : (ax < 3.50f) ? 3.0f :
            (ax <= 5.00f) ? 4.0f : 6.0f;
  return sgn * q;
}

void bcir_quant_q8_g32_encode(const float *src, int8_t *q_out, int8_t *exp_out, size_t n) {
  size_t nb = (n + 31u) / 32u;
  for (size_t b = 0; b < nb; ++b) {
    size_t s = b * 32u, e = (s + 32u <= n) ? (s + 32u) : n;
    float amax = 0.0f;
    for (size_t i = s; i < e; ++i) { float a = fabsf(src[i]); if (a > amax) amax = a; }
    if (amax == 0.0f || !isfinite(amax)) {
      exp_out[b] = 0;
      for (size_t i = s; i < e; ++i) q_out[i] = 0;
      continue;
    }
    int exp_i = (int)ceil(log2((double)amax / 127.0));
    if (exp_i < -126) exp_i = -126;
    if (exp_i > 127) exp_i = 127;
    exp_out[b] = (int8_t)exp_i;
    float inv = ldexpf(1.0f, -exp_i);
    for (size_t i = s; i < e; ++i) {
      long qi = lrintf(src[i] * inv);
      if (qi > 127) qi = 127;
      if (qi < -127) qi = -127;
      q_out[i] = (int8_t)qi;
    }
  }
}

void bcir_quant_q8_g32_decode(const int8_t *q_in, const int8_t *exp_in, float *dst, size_t n) {
  size_t nb = (n + 31u) / 32u;
  for (size_t b = 0; b < nb; ++b) {
    size_t s = b * 32u, e = (s + 32u <= n) ? (s + 32u) : n;
    float scale = ldexpf(1.0f, (int)exp_in[b]);
    for (size_t i = s; i < e; ++i) dst[i] = (float)q_in[i] * scale;
  }
}

typedef struct qdq_task_ctx {
  int gemm_kind;
  const float *src;
  float *dst;
  size_t n, blocks_per_task, total_blocks;
} qdq_task_ctx;

static void qdq_worker(void *vctx, size_t task) {
  const qdq_task_ctx *c = (const qdq_task_ctx *)vctx;
  size_t b0 = task * c->blocks_per_task;
  size_t b1 = b0 + c->blocks_per_task;
  if (b1 > c->total_blocks) b1 = c->total_blocks;
  for (size_t b = b0; b < b1; ++b) {
    size_t s = b * 32u, e = (s + 32u <= c->n) ? (s + 32u) : c->n;
    float amax = 0.0f;
    for (size_t i = s; i < e; ++i) { float a = fabsf(c->src[i]); if (a > amax) amax = a; }
    if (amax == 0.0f || !isfinite(amax)) {
      for (size_t i = s; i < e; ++i) c->dst[i] = c->src[i];
      continue;
    }
    if (c->gemm_kind == BCIR_GEMM_FP8_E4M3_BLK32) {
      float scale = amax / 448.0f, inv = 448.0f / amax;
      for (size_t i = s; i < e; ++i) c->dst[i] = bcir_quant_fp8_e4m3_scalar(c->src[i] * inv) * scale;
    } else if (c->gemm_kind == BCIR_GEMM_FP8_E5M2_BLK32) {
      float scale = amax / 57344.0f, inv = 57344.0f / amax;
      for (size_t i = s; i < e; ++i) c->dst[i] = bcir_quant_fp8_e5m2_scalar(c->src[i] * inv) * scale;
    } else if (c->gemm_kind == BCIR_GEMM_FP4_MX_BLK32) {
      int exp_i = (int)ceil(log2((double)amax / 6.0));
      if (exp_i < -126) exp_i = -126;
      if (exp_i > 127) exp_i = 127;
      float scale = ldexpf(1.0f, exp_i), inv = ldexpf(1.0f, -exp_i);
      for (size_t i = s; i < e; ++i) c->dst[i] = bcir_quant_fp4_e2m1_scalar(c->src[i] * inv) * scale;
    } else if (c->gemm_kind == BCIR_GEMM_Q8_BLK32) {
      int exp_i = (int)ceil(log2((double)amax / 127.0));
      if (exp_i < -126) exp_i = -126;
      if (exp_i > 127) exp_i = 127;
      float scale = ldexpf(1.0f, exp_i), inv = ldexpf(1.0f, -exp_i);
      for (size_t i = s; i < e; ++i) {
        long qi = lrintf(c->src[i] * inv);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        c->dst[i] = (float)qi * scale;
      }
    }
  }
}

void bcir_quant_dequant_blk32_pool(bcir_tensor_pool *pool, int gemm_kind,
                                   const float *src, float *dst, size_t n) {
  if (n == 0u) return;
  size_t total_blocks = (n + 31u) / 32u, bpt = 128u;
  qdq_task_ctx ctx = { gemm_kind, src, dst, n, bpt, total_blocks };
  bcir_tensor_pool_run(pool, qdq_worker, &ctx, (total_blocks + bpt - 1u) / bpt);
}

int bcir_mm_bf16_vdp_variants(void) {
  int v = 1;
#if BCIR_HAS_AVX512_BF16
  v |= 2;
#endif
  return v;
}

size_t bcir_mm_bf16_vdp_scratch_bytes(size_t m, size_t n, size_t k) {
  size_t k_pairs = (k + 1u) / 2u, n_pad = ((n + 15u) / 16u) * 16u;
  size_t a_bytes = (m * k_pairs * 2u * 2u + 63u) & ~(size_t)63u;
  size_t b_bytes = (n_pad * k_pairs * 2u * 2u + 63u) & ~(size_t)63u;
  return a_bytes + b_bytes + 128u;
}

/* Exact VDPBF16PS semantics (0 bit drift between AVX-512 BF16 and portable C11):
 * DAZ/FTZ, upper pair element (a1*b1) fused first, then lower (a0*b0). */
static inline float vdpbf16_pair_portable(float acc, uint16_t a0_b, uint16_t a1_b,
                                          uint16_t b0_b, uint16_t b1_b) {
  float a0 = flush_subnormal_f32(bf16_to_f32(a0_b));
  float a1 = flush_subnormal_f32(bf16_to_f32(a1_b));
  float b0 = flush_subnormal_f32(bf16_to_f32(b0_b));
  float b1 = flush_subnormal_f32(bf16_to_f32(b1_b));
  float ac = flush_subnormal_f32(acc);
  return flush_subnormal_f32(fmaf(a0, b0, flush_subnormal_f32(fmaf(a1, b1, ac))));
}

typedef struct vdp_pack_a_ctx { int ta; size_t m, k, k_pairs; float alpha; const float *a; uint16_t *pa; } vdp_pack_a_ctx;
static void vdp_pack_a_worker(void *vctx, size_t task) {
  const vdp_pack_a_ctx *c = (const vdp_pack_a_ctx *)vctx;
  size_t i0 = task * 16u, i1 = (i0 + 16u <= c->m) ? (i0 + 16u) : c->m;
  for (size_t i = i0; i < i1; ++i) {
    uint16_t *dst = c->pa + i * (2u * c->k_pairs);
    if (!c->ta) {
      const float *src = c->a + i * c->k;
      if (c->alpha == 1.0f) {
        size_t l = 0;
#if BCIR_HAS_AVX512_BF16
        for (; l + 32u <= c->k; l += 32u) {
          __m512 v0 = _mm512_loadu_ps(src + l), v1 = _mm512_loadu_ps(src + l + 16u);
          _mm512_storeu_si512((void *)(dst + l), (__m512i)_mm512_cvtne2ps_pbh(v1, v0));
        }
#endif
        for (; l < c->k; ++l) dst[l] = f32_to_bf16(src[l]);
      } else {
        for (size_t l = 0; l < c->k; ++l) dst[l] = f32_to_bf16(c->alpha * src[l]);
      }
    } else {
      for (size_t l = 0; l < c->k; ++l) {
        float v = c->a[l * c->m + i];
        dst[l] = f32_to_bf16((c->alpha == 1.0f) ? v : (c->alpha * v));
      }
    }
    if (c->k < 2u * c->k_pairs) dst[c->k] = 0u;
  }
}

typedef struct vdp_pack_b_ctx { int tb; size_t n, k, k_pairs, n_blocks; const float *b; uint16_t *pb; } vdp_pack_b_ctx;
static void vdp_pack_b_worker(void *vctx, size_t task) {
  const vdp_pack_b_ctx *c = (const vdp_pack_b_ctx *)vctx;
  size_t nb = task, j0 = nb * 16u;
  uint16_t *dst = c->pb + nb * c->k_pairs * 32u;
  size_t cols = (j0 + 16u <= c->n) ? 16u : (c->n - j0);
  for (size_t p = 0; p < c->k_pairs; ++p) {
    size_t l0 = 2u * p, l1 = l0 + 1u;
    uint16_t *dp = dst + p * 32u;
    if (!c->tb) {
      const float *r0 = c->b + l0 * c->n + j0;
      const float *r1 = (l1 < c->k) ? (c->b + l1 * c->n + j0) : NULL;
      for (size_t jj = 0; jj < cols; ++jj) {
        dp[2u * jj] = f32_to_bf16(r0[jj]);
        dp[2u * jj + 1u] = r1 ? f32_to_bf16(r1[jj]) : 0u;
      }
    } else {
      for (size_t jj = 0; jj < cols; ++jj) {
        const float *col = c->b + (j0 + jj) * c->k;
        dp[2u * jj] = f32_to_bf16(col[l0]);
        dp[2u * jj + 1u] = (l1 < c->k) ? f32_to_bf16(col[l1]) : 0u;
      }
    }
    for (size_t jj = cols; jj < 16u; ++jj) { dp[2u * jj] = 0u; dp[2u * jj + 1u] = 0u; }
  }
}

typedef struct vdp_gemm_ctx {
  int variant; size_t m, n, k_pairs, n_blocks; float beta;
  const uint16_t *pa, *pb; float *c; size_t m_tile;
} vdp_gemm_ctx;

static void vdp_gemm_worker(void *vctx, size_t task) {
  const vdp_gemm_ctx *g = (const vdp_gemm_ctx *)vctx;
  size_t n_tiles = (g->n_blocks + 1u) / 2u;
  size_t mt = task / n_tiles, nt = task % n_tiles;
  size_t i0 = mt * g->m_tile, i1 = (i0 + g->m_tile <= g->m) ? (i0 + g->m_tile) : g->m;
  size_t nb0 = nt * 2u, nb1 = nb0 + 1u;
  int has_b1 = (nb1 < g->n_blocks);
  size_t j0 = nb0 * 16u;
  size_t cols0 = (j0 + 16u <= g->n) ? 16u : (g->n - j0);
  size_t cols1 = has_b1 ? (((j0 + 32u) <= g->n) ? 16u : (g->n - (j0 + 16u))) : 0u;
  const uint16_t *b0 = g->pb + nb0 * g->k_pairs * 32u;
  const uint16_t *b1 = has_b1 ? (g->pb + nb1 * g->k_pairs * 32u) : NULL;

#if BCIR_HAS_AVX512_BF16
  if (g->variant == 2) {
    __mmask16 mask0 = (cols0 == 16u) ? 0xFFFFu : (uint16_t)((1u << cols0) - 1u);
    __mmask16 mask1 = (cols1 == 16u) ? 0xFFFFu : (uint16_t)((1u << cols1) - 1u);
    size_t i = i0;
    for (; i + 6u <= i1; i += 6u) {
      __m512 c00, c10, c20, c30, c40, c50, c01, c11, c21, c31, c41, c51;
      if (g->beta == 0.0f) {
        c00 = c10 = c20 = c30 = c40 = c50 = c01 = c11 = c21 = c31 = c41 = c51 = _mm512_setzero_ps();
      } else {
        __m512 vb = _mm512_set1_ps(g->beta);
        c00 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask0, g->c + (i + 0u) * g->n + j0));
        c10 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask0, g->c + (i + 1u) * g->n + j0));
        c20 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask0, g->c + (i + 2u) * g->n + j0));
        c30 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask0, g->c + (i + 3u) * g->n + j0));
        c40 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask0, g->c + (i + 4u) * g->n + j0));
        c50 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask0, g->c + (i + 5u) * g->n + j0));
        if (has_b1) {
          c01 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask1, g->c + (i + 0u) * g->n + j0 + 16u));
          c11 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask1, g->c + (i + 1u) * g->n + j0 + 16u));
          c21 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask1, g->c + (i + 2u) * g->n + j0 + 16u));
          c31 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask1, g->c + (i + 3u) * g->n + j0 + 16u));
          c41 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask1, g->c + (i + 4u) * g->n + j0 + 16u));
          c51 = _mm512_mul_ps(vb, _mm512_maskz_loadu_ps(mask1, g->c + (i + 5u) * g->n + j0 + 16u));
        } else { c01 = c11 = c21 = c31 = c41 = c51 = _mm512_setzero_ps(); }
      }
      const uint32_t *a0 = (const uint32_t *)(g->pa + (i + 0u) * 2u * g->k_pairs);
      const uint32_t *a1 = (const uint32_t *)(g->pa + (i + 1u) * 2u * g->k_pairs);
      const uint32_t *a2 = (const uint32_t *)(g->pa + (i + 2u) * 2u * g->k_pairs);
      const uint32_t *a3 = (const uint32_t *)(g->pa + (i + 3u) * 2u * g->k_pairs);
      const uint32_t *a4 = (const uint32_t *)(g->pa + (i + 4u) * 2u * g->k_pairs);
      const uint32_t *a5 = (const uint32_t *)(g->pa + (i + 5u) * 2u * g->k_pairs);
      if (has_b1) {
        for (size_t p = 0; p < g->k_pairs; ++p) {
          __m512bh bv0 = (__m512bh)_mm512_loadu_si512((const void *)(b0 + p * 32u));
          __m512bh bv1 = (__m512bh)_mm512_loadu_si512((const void *)(b1 + p * 32u));
          __m512bh av0 = (__m512bh)_mm512_set1_epi32((int)a0[p]);
          __m512bh av1 = (__m512bh)_mm512_set1_epi32((int)a1[p]);
          __m512bh av2 = (__m512bh)_mm512_set1_epi32((int)a2[p]);
          __m512bh av3 = (__m512bh)_mm512_set1_epi32((int)a3[p]);
          __m512bh av4 = (__m512bh)_mm512_set1_epi32((int)a4[p]);
          __m512bh av5 = (__m512bh)_mm512_set1_epi32((int)a5[p]);
          c00 = _mm512_dpbf16_ps(c00, av0, bv0); c01 = _mm512_dpbf16_ps(c01, av0, bv1);
          c10 = _mm512_dpbf16_ps(c10, av1, bv0); c11 = _mm512_dpbf16_ps(c11, av1, bv1);
          c20 = _mm512_dpbf16_ps(c20, av2, bv0); c21 = _mm512_dpbf16_ps(c21, av2, bv1);
          c30 = _mm512_dpbf16_ps(c30, av3, bv0); c31 = _mm512_dpbf16_ps(c31, av3, bv1);
          c40 = _mm512_dpbf16_ps(c40, av4, bv0); c41 = _mm512_dpbf16_ps(c41, av4, bv1);
          c50 = _mm512_dpbf16_ps(c50, av5, bv0); c51 = _mm512_dpbf16_ps(c51, av5, bv1);
        }
      } else {
        for (size_t p = 0; p < g->k_pairs; ++p) {
          __m512bh bv0 = (__m512bh)_mm512_loadu_si512((const void *)(b0 + p * 32u));
          c00 = _mm512_dpbf16_ps(c00, (__m512bh)_mm512_set1_epi32((int)a0[p]), bv0);
          c10 = _mm512_dpbf16_ps(c10, (__m512bh)_mm512_set1_epi32((int)a1[p]), bv0);
          c20 = _mm512_dpbf16_ps(c20, (__m512bh)_mm512_set1_epi32((int)a2[p]), bv0);
          c30 = _mm512_dpbf16_ps(c30, (__m512bh)_mm512_set1_epi32((int)a3[p]), bv0);
          c40 = _mm512_dpbf16_ps(c40, (__m512bh)_mm512_set1_epi32((int)a4[p]), bv0);
          c50 = _mm512_dpbf16_ps(c50, (__m512bh)_mm512_set1_epi32((int)a5[p]), bv0);
        }
      }
      _mm512_mask_storeu_ps(g->c + (i + 0u) * g->n + j0, mask0, c00);
      _mm512_mask_storeu_ps(g->c + (i + 1u) * g->n + j0, mask0, c10);
      _mm512_mask_storeu_ps(g->c + (i + 2u) * g->n + j0, mask0, c20);
      _mm512_mask_storeu_ps(g->c + (i + 3u) * g->n + j0, mask0, c30);
      _mm512_mask_storeu_ps(g->c + (i + 4u) * g->n + j0, mask0, c40);
      _mm512_mask_storeu_ps(g->c + (i + 5u) * g->n + j0, mask0, c50);
      if (has_b1) {
        _mm512_mask_storeu_ps(g->c + (i + 0u) * g->n + j0 + 16u, mask1, c01);
        _mm512_mask_storeu_ps(g->c + (i + 1u) * g->n + j0 + 16u, mask1, c11);
        _mm512_mask_storeu_ps(g->c + (i + 2u) * g->n + j0 + 16u, mask1, c21);
        _mm512_mask_storeu_ps(g->c + (i + 3u) * g->n + j0 + 16u, mask1, c31);
        _mm512_mask_storeu_ps(g->c + (i + 4u) * g->n + j0 + 16u, mask1, c41);
        _mm512_mask_storeu_ps(g->c + (i + 5u) * g->n + j0 + 16u, mask1, c51);
      }
    }
    for (; i < i1; ++i) {
      __m512 c0 = (g->beta == 0.0f) ? _mm512_setzero_ps()
                  : _mm512_mul_ps(_mm512_set1_ps(g->beta), _mm512_maskz_loadu_ps(mask0, g->c + i * g->n + j0));
      __m512 c1 = (g->beta == 0.0f || !has_b1) ? _mm512_setzero_ps()
                  : _mm512_mul_ps(_mm512_set1_ps(g->beta), _mm512_maskz_loadu_ps(mask1, g->c + i * g->n + j0 + 16u));
      const uint32_t *ap = (const uint32_t *)(g->pa + i * 2u * g->k_pairs);
      for (size_t p = 0; p < g->k_pairs; ++p) {
        __m512bh av = (__m512bh)_mm512_set1_epi32((int)ap[p]);
        c0 = _mm512_dpbf16_ps(c0, av, (__m512bh)_mm512_loadu_si512((const void *)(b0 + p * 32u)));
        if (has_b1) c1 = _mm512_dpbf16_ps(c1, av, (__m512bh)_mm512_loadu_si512((const void *)(b1 + p * 32u)));
      }
      _mm512_mask_storeu_ps(g->c + i * g->n + j0, mask0, c0);
      if (has_b1) _mm512_mask_storeu_ps(g->c + i * g->n + j0 + 16u, mask1, c1);
    }
    return;
  }
#endif
  for (size_t i = i0; i < i1; ++i) {
    const uint16_t *ap = g->pa + i * 2u * g->k_pairs;
    for (size_t jj = 0; jj < cols0; ++jj) {
      float acc = (g->beta == 0.0f) ? 0.0f : (g->beta * g->c[i * g->n + j0 + jj]);
      for (size_t p = 0; p < g->k_pairs; ++p) {
        const uint16_t *bp = b0 + p * 32u + 2u * jj;
        acc = vdpbf16_pair_portable(acc, ap[2u * p], ap[2u * p + 1u], bp[0], bp[1]);
      }
      g->c[i * g->n + j0 + jj] = acc;
    }
    if (has_b1) {
      for (size_t jj = 0; jj < cols1; ++jj) {
        float acc = (g->beta == 0.0f) ? 0.0f : (g->beta * g->c[i * g->n + j0 + 16u + jj]);
        for (size_t p = 0; p < g->k_pairs; ++p) {
          const uint16_t *bp = b1 + p * 32u + 2u * jj;
          acc = vdpbf16_pair_portable(acc, ap[2u * p], ap[2u * p + 1u], bp[0], bp[1]);
        }
        g->c[i * g->n + j0 + 16u + jj] = acc;
      }
    }
  }
}

int bcir_mm_bf16_vdp_variant(bcir_tensor_pool *pool, int variant,
                             int ta, int tb, size_t m, size_t n, size_t k,
                             float alpha, const float *a, const float *b,
                             float beta, float *c, void *scratch, size_t scratch_bytes) {
  int avail = bcir_mm_bf16_vdp_variants();
  if (variant == 0) variant = (avail & 2) ? 2 : 1;
  if (!(avail & variant)) return 0;
  if (m == 0u || n == 0u) return variant;
  if (k == 0u) {
    for (size_t i = 0; i < m * n; ++i) c[i] = (beta == 0.0f) ? 0.0f : (beta * c[i]);
    return variant;
  }
  size_t need = bcir_mm_bf16_vdp_scratch_bytes(m, n, k);
  void *owned = NULL;
  if (!scratch || scratch_bytes < need) {
    owned = bcir_aligned_calloc64(need, 1u);
    if (!owned) return 0;
    scratch = owned;
  }
  size_t k_pairs = (k + 1u) / 2u, n_blocks = (n + 15u) / 16u;
  size_t a_bytes = (m * k_pairs * 2u * 2u + 63u) & ~(size_t)63u;
  uint8_t *base = (uint8_t *)(((uintptr_t)scratch + 63u) & ~(uintptr_t)63u);
  uint16_t *pa = (uint16_t *)base, *pb = (uint16_t *)(base + a_bytes);

  vdp_pack_a_ctx pactx = { ta, m, k, k_pairs, alpha, a, pa };
  bcir_tensor_pool_run(pool, vdp_pack_a_worker, &pactx, (m + 15u) / 16u);
  vdp_pack_b_ctx pbctx = { tb, n, k, k_pairs, n_blocks, b, pb };
  bcir_tensor_pool_run(pool, vdp_pack_b_worker, &pbctx, n_blocks);
  size_t m_tile = 24u, m_tiles = (m + m_tile - 1u) / m_tile, n_tiles = (n_blocks + 1u) / 2u;
  vdp_gemm_ctx gctx = { variant, m, n, k_pairs, n_blocks, beta, pa, pb, c, m_tile };
  bcir_tensor_pool_run(pool, vdp_gemm_worker, &gctx, m_tiles * n_tiles);
  bcir_aligned_free64(owned);
  return variant;
}

size_t bcir_prec_mm_scratch_bytes(int gemm_kind, size_t m, size_t n, size_t k) {
  if (gemm_kind == BCIR_GEMM_FP32_KBLOCK) return 0u;
  if (gemm_kind == BCIR_GEMM_BF16_NDT) return bcir_tensor_mm_bf16_workspace(m, n, k) + 128u;
  if (gemm_kind == BCIR_GEMM_BF16_VDP) return bcir_mm_bf16_vdp_scratch_bytes(m, n, k);
  return (m * k + k * n) * sizeof(float) + 256u;
}

int bcir_prec_mm_pool(bcir_tensor_pool *pool, int gemm_kind, int force_portable,
                      int ta, int tb, size_t m, size_t n, size_t k,
                      float alpha, const float *a, const float *b,
                      float beta, float *c, void *scratch, size_t scratch_bytes) {
  if (gemm_kind == BCIR_GEMM_FP32_KBLOCK) {
    bcir_tensor_mm_kblock_pool(pool, ta, tb, m, n, k, alpha, a, b, beta, c);
    return 1;
  }
  if (gemm_kind == BCIR_GEMM_BF16_NDT) {
    size_t need = bcir_tensor_mm_bf16_workspace(m, n, k);
    void *owned = NULL, *ws = scratch;
    size_t wsb = scratch_bytes;
    if (!ws || wsb < need + 64u) {
      owned = bcir_aligned_calloc64(need + 64u, 1u);
      if (!owned) return 0;
      ws = owned;
      wsb = need + 64u;
    }
    uintptr_t aligned = ((uintptr_t)ws + 63u) & ~(uintptr_t)63u;
    int ok = bcir_tensor_mm_bf16_pool(pool, ta, tb, m, n, k, alpha, a, b, beta, c,
                                      (void *)aligned, wsb - (size_t)(aligned - (uintptr_t)ws));
    bcir_aligned_free64(owned);
    return ok;
  }
  if (gemm_kind == BCIR_GEMM_BF16_VDP) {
    return bcir_mm_bf16_vdp_variant(pool, force_portable ? 1 : 0, ta, tb, m, n, k,
                                    alpha, a, b, beta, c, scratch, scratch_bytes) != 0;
  }
  size_t na = m * k, nb = k * n, need = (na + nb) * sizeof(float) + 128u;
  void *owned = NULL, *ws = scratch;
  if (!ws || scratch_bytes < need) {
    owned = bcir_aligned_calloc64(need, 1u);
    if (!owned) return 0;
    ws = owned;
  }
  float *qa = (float *)(((uintptr_t)ws + 63u) & ~(uintptr_t)63u), *qb = qa + na;
  bcir_quant_dequant_blk32_pool(pool, gemm_kind, a, qa, na);
  bcir_quant_dequant_blk32_pool(pool, gemm_kind, b, qb, nb);
  bcir_tensor_mm_kblock_pool(pool, ta, tb, m, n, k, alpha, qa, qb, beta, c);
  bcir_aligned_free64(owned);
  return 1;
}

/* ==================== Deterministic 5-Step Quintic Newton-Schulz ==================== */
/* Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) from Jordan et al. (2024)
 * and proven convergent in arXiv:2609.39595 (Sep 2026). Operates on X of shape
 * (p x q) with p <= q in-place using deterministic bcir_tensor_mm_kblock_pool
 * so 1T..NT produce bit-identical orthogonalized updates. */
static void newton_schulz_5step(bcir_tensor_pool *pool, size_t p, size_t q,
                                float *x_buf, float *y_buf,
                                float *a_buf, float *b_buf, float *a2_buf) {
  const float ca = 3.4445f, cb = -4.7750f, cc = 2.0315f;
  size_t pq = p * q, pp = p * p;
  double ss = 0.0;
  for (size_t i = 0; i < pq; ++i) {
    double v = (double)x_buf[i];
    ss += v * v;
  }
  double norm = sqrt(ss);
  if (!(norm > 0.0) || !isfinite(norm)) {
    memset(x_buf, 0, pq * sizeof(float));
    return;
  }
  float inv = (float)(1.0 / (norm + 1e-7));
  for (size_t i = 0; i < pq; ++i) x_buf[i] *= inv;

  float *cur = x_buf, *nxt = y_buf;
  for (int it = 0; it < 5; ++it) {
    bcir_tensor_mm_kblock_pool(pool, 0, 1, p, p, q, 1.0f, cur, cur, 0.0f, a_buf);
    bcir_tensor_mm_kblock_pool(pool, 0, 0, p, p, p, 1.0f, a_buf, a_buf, 0.0f, a2_buf);
    for (size_t i = 0; i < pp; ++i) b_buf[i] = cb * a_buf[i] + cc * a2_buf[i];
    for (size_t i = 0; i < pq; ++i) nxt[i] = ca * cur[i];
    bcir_tensor_mm_kblock_pool(pool, 0, 0, p, q, p, 1.0f, b_buf, cur, 1.0f, nxt);
    float *tmp = cur;
    cur = nxt;
    nxt = tmp;
  }
  if (cur != x_buf) memcpy(x_buf, cur, pq * sizeof(float));
}

int bcir_newton_schulz_verify(bcir_tensor_pool *pool, size_t rows, size_t cols,
                              double *out_ortho_err) {
  if (rows == 0u || cols == 0u) return 1;
  size_t p = (rows <= cols) ? rows : cols;
  size_t q = (rows <= cols) ? cols : rows;
  float *x = (float *)bcir_aligned_calloc64(p * q, sizeof(float));
  float *y = (float *)bcir_aligned_calloc64(p * q, sizeof(float));
  float *a = (float *)bcir_aligned_calloc64(p * p, sizeof(float));
  float *b = (float *)bcir_aligned_calloc64(p * p, sizeof(float));
  float *a2 = (float *)bcir_aligned_calloc64(p * p, sizeof(float));
  if (!x || !y || !a || !b || !a2) {
    bcir_aligned_free64(x); bcir_aligned_free64(y);
    bcir_aligned_free64(a); bcir_aligned_free64(b); bcir_aligned_free64(a2);
    return 2;
  }
  uint64_t r = 0x123456789ABCDEF0ULL;
  for (size_t i = 0; i < p * q; ++i) {
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    x[i] = ((float)(int32_t)(r >> 40) / (float)(1 << 23)) - 1.0f;
  }
  /* Add small diagonal boost so matrix is well-conditioned */
  for (size_t i = 0; i < p; ++i) x[i * q + i] += 2.0f;
  newton_schulz_5step(pool, p, q, x, y, a, b, a2);
  /* Check ||X X^T - I||_F / sqrt(p) */
  bcir_tensor_mm_kblock_pool(pool, 0, 1, p, p, q, 1.0f, x, x, 0.0f, a);
  double err2 = 0.0;
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < p; ++j) {
      double diff = (double)a[i * p + j] - ((i == j) ? 1.0 : 0.0);
      err2 += diff * diff;
    }
  }
  double rms_err = sqrt(err2 / (double)p);
  bcir_aligned_free64(x); bcir_aligned_free64(y);
  bcir_aligned_free64(a); bcir_aligned_free64(b); bcir_aligned_free64(a2);
  if (out_ortho_err) *out_ortho_err = rms_err;
  /* 5-step quintic maps singular values into [0.70, 1.25], so ||XX^T - I||_F/sqrt(p) < 0.35 */
  return (isfinite(rms_err) && rms_err < 0.35) ? 0 : 1;
}

/* ==================== Optimizer Lifecycle & Step ==================== */
static int tensor_uses_muon(const bcir_opt_tensor_shape *sh) {
  return sh && sh->is_muon_2d && sh->rows >= 2u && sh->cols >= 2u;
}

bcir_opt_state *bcir_opt_create(int opt_kind, size_t n_params,
                                const bcir_opt_tensor_shape *shapes, size_t n_shapes) {
  if (opt_kind < 0 || opt_kind >= BCIR_OPT_KIND_COUNT || n_params == 0u) return NULL;
  if ((opt_kind == BCIR_OPT_ADAFACTOR_M2 || opt_kind == BCIR_OPT_ADAFACTOR_BF16_GRAD ||
       opt_kind == BCIR_OPT_MUON_FP32 || opt_kind == BCIR_OPT_Q8_MUON_BF16_GRAD) &&
      (!shapes || n_shapes == 0u)) {
    return NULL;
  }
  bcir_opt_state *opt = (bcir_opt_state *)prec_zalloc(1u, sizeof(*opt));
  if (!opt) return NULL;
  opt->opt_kind = opt_kind;
  opt->n_params = n_params;
  opt->n_blocks = (n_params + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
  opt->beta1_power = opt->beta2_power = 1.0;

  if (shapes && n_shapes > 0u) {
    opt->n_shapes = n_shapes;
    opt->shapes = (bcir_opt_tensor_shape *)prec_zalloc(n_shapes, sizeof(*shapes));
    if (!opt->shapes) { bcir_opt_destroy(opt); return NULL; }
    memcpy(opt->shapes, shapes, n_shapes * sizeof(*shapes));
  }

  opt->adagc_beta = 0.98;
  opt->adagc_lambda = 1.05;
  opt->adagc_abs_floor = 1e-4;
  opt->adagc_warmup_steps = 1u;

  size_t ob = 0u, sb = 0u;
  if (opt->n_shapes > 0u) {
    opt->adagc_ema = (double *)prec_zalloc(opt->n_shapes, sizeof(double));
    opt->tensor_grad_norm = (double *)prec_zalloc(opt->n_shapes, sizeof(double));
    opt->tensor_adagc_scale = (double *)prec_zalloc(opt->n_shapes, sizeof(double));
    opt->tensor_grad_mom_cos = (double *)prec_zalloc(opt->n_shapes, sizeof(double));
    opt->tensor_clip_hits = (uint64_t *)prec_zalloc(opt->n_shapes, sizeof(uint64_t));
    opt->kahan_mask = (uint8_t *)prec_zalloc(opt->n_shapes, sizeof(uint8_t));
    opt->kahan_comp = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    if (!opt->adagc_ema || !opt->tensor_grad_norm || !opt->tensor_adagc_scale ||
        !opt->tensor_grad_mom_cos || !opt->tensor_clip_hits ||
        !opt->kahan_mask || !opt->kahan_comp) {
      bcir_opt_destroy(opt);
      return NULL;
    }
    for (size_t s = 0; s < opt->n_shapes; ++s) opt->tensor_adagc_scale[s] = 1.0;
    sb += opt->n_shapes * (4u * sizeof(double) + sizeof(uint64_t) + sizeof(uint8_t) + sizeof(float *));
  }
  int use_bf16_grad = (opt_kind == BCIR_OPT_Q8_MOMENTS_BF16_GRAD ||
                       opt_kind == BCIR_OPT_ADAFACTOR_BF16_GRAD ||
                       opt_kind == BCIR_OPT_Q8_MUON_BF16_GRAD);
  if (use_bf16_grad) {
    opt->grad_bf16 = (uint16_t *)bcir_aligned_calloc64(n_params, 2u);
    if (!opt->grad_bf16) { bcir_opt_destroy(opt); return NULL; }
    ob += n_params * 2u;
  } else {
    opt->grad_fp32 = (float *)bcir_aligned_calloc64(n_params, 4u);
    if (!opt->grad_fp32) { bcir_opt_destroy(opt); return NULL; }
    ob += n_params * 4u;
  }

  /* Pre-allocate deterministic norm reduction scratch (zero hot-path malloc). */
  size_t norm_tasks = (n_params + 65535u) / 65536u;
  opt->norm_tasks_cap = norm_tasks;
  opt->norm_partials = (double *)prec_zalloc(norm_tasks, sizeof(double));
  opt->norm_bads = (int *)prec_zalloc(norm_tasks, sizeof(int));
  if (!opt->norm_partials || !opt->norm_bads) { bcir_opt_destroy(opt); return NULL; }
  sb += norm_tasks * (sizeof(double) + sizeof(int));

  if (opt_kind == BCIR_OPT_FP32) {
    opt->m1_fp32 = (float *)bcir_aligned_calloc64(n_params, 4u);
    opt->m2_fp32 = (float *)bcir_aligned_calloc64(n_params, 4u);
    if (!opt->m1_fp32 || !opt->m2_fp32) { bcir_opt_destroy(opt); return NULL; }
    ob += 2u * n_params * 4u;
  } else if (opt_kind == BCIR_OPT_Q8_MOMENTS || opt_kind == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
    opt->m1_q8 = (int8_t *)bcir_aligned_calloc64(n_params, 1u);
    opt->m1_scale = (float *)bcir_aligned_calloc64(opt->n_blocks, 4u);
    opt->m2_q8 = (uint8_t *)bcir_aligned_calloc64(n_params, 1u);
    opt->m2_log_min = (float *)bcir_aligned_calloc64(opt->n_blocks, 4u);
    opt->m2_log_step = (float *)bcir_aligned_calloc64(opt->n_blocks, 4u);
    size_t qt = (opt->n_blocks + 255u) / 256u;
    opt->q8_tasks_cap = qt;
    opt->q8_clamps_buf = (uint64_t *)prec_zalloc(qt, sizeof(uint64_t));
    if (!opt->m1_q8 || !opt->m1_scale || !opt->m2_q8 || !opt->m2_log_min ||
        !opt->m2_log_step || !opt->q8_clamps_buf) {
      bcir_opt_destroy(opt);
      return NULL;
    }
    ob += 2u * n_params + opt->n_blocks * 12u;
    sb += qt * sizeof(uint64_t);
  } else if (opt_kind == BCIR_OPT_ADAFACTOR_M2 || opt_kind == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
    opt->m1_fp32 = (float *)bcir_aligned_calloc64(n_params, 4u);
    opt->af_row = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    opt->af_col = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    if (!opt->m1_fp32 || !opt->af_row || !opt->af_col) { bcir_opt_destroy(opt); return NULL; }
    ob += n_params * 4u;
    size_t total_cols = 0u;
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      size_t r = opt->shapes[s].rows, c = opt->shapes[s].cols;
      if (c <= 1u || r <= 1u) {
        opt->af_row[s] = (float *)bcir_aligned_calloc64(r * c, 4u);
        if (!opt->af_row[s]) { bcir_opt_destroy(opt); return NULL; }
        ob += r * c * 4u;
      } else {
        opt->af_row[s] = (float *)bcir_aligned_calloc64(r, 4u);
        opt->af_col[s] = (float *)bcir_aligned_calloc64(c, 4u);
        if (!opt->af_row[s] || !opt->af_col[s]) { bcir_opt_destroy(opt); return NULL; }
        ob += (r + c) * 4u;
        total_cols += c;
      }
    }
    /* One disjoint column accumulator per 2D shape so workers never allocate or collide. */
    if (total_cols > 0u) {
      opt->af_col_cap = total_cols;
      opt->af_col_scratch = (double *)bcir_aligned_calloc64(total_cols, sizeof(double));
      if (!opt->af_col_scratch) { bcir_opt_destroy(opt); return NULL; }
      sb += total_cols * sizeof(double);
    }
  } else if (opt_kind == BCIR_OPT_MUON_FP32 || opt_kind == BCIR_OPT_Q8_MUON_BF16_GRAD) {
    int q8_m = (opt_kind == BCIR_OPT_Q8_MUON_BF16_GRAD);
    opt->adam_m1_fp32 = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    opt->adam_m2_fp32 = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    if (q8_m) {
      opt->muon_m1_q8 = (int8_t **)prec_zalloc(opt->n_shapes, sizeof(int8_t *));
      opt->muon_m1_scale = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    } else {
      opt->muon_m1_fp32 = (float **)prec_zalloc(opt->n_shapes, sizeof(float *));
    }
    if (!opt->adam_m1_fp32 || !opt->adam_m2_fp32 ||
        (q8_m ? (!opt->muon_m1_q8 || !opt->muon_m1_scale) : !opt->muon_m1_fp32)) {
      bcir_opt_destroy(opt);
      return NULL;
    }
    size_t max_pq = 0u, max_pp = 0u;
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      const bcir_opt_tensor_shape *sh = &opt->shapes[s];
      size_t len = sh->rows * sh->cols;
      if (tensor_uses_muon(sh)) {
        size_t p = (sh->rows <= sh->cols) ? sh->rows : sh->cols;
        if (len > max_pq) max_pq = len;
        if (p * p > max_pp) max_pp = p * p;
        if (q8_m) {
          size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
          opt->muon_m1_q8[s] = (int8_t *)bcir_aligned_calloc64(len, 1u);
          opt->muon_m1_scale[s] = (float *)bcir_aligned_calloc64(nb, 4u);
          if (!opt->muon_m1_q8[s] || !opt->muon_m1_scale[s]) { bcir_opt_destroy(opt); return NULL; }
          ob += len + nb * 4u;
        } else {
          opt->muon_m1_fp32[s] = (float *)bcir_aligned_calloc64(len, 4u);
          if (!opt->muon_m1_fp32[s]) { bcir_opt_destroy(opt); return NULL; }
          ob += len * 4u;
        }
      } else {
        opt->adam_m1_fp32[s] = (float *)bcir_aligned_calloc64(len, 4u);
        opt->adam_m2_fp32[s] = (float *)bcir_aligned_calloc64(len, 4u);
        if (!opt->adam_m1_fp32[s] || !opt->adam_m2_fp32[s]) { bcir_opt_destroy(opt); return NULL; }
        ob += 2u * len * 4u;
      }
    }
    if (max_pq > 0u) {
      opt->muon_ns_x = (float *)bcir_aligned_calloc64(max_pq, 4u);
      opt->muon_ns_y = (float *)bcir_aligned_calloc64(max_pq, 4u);
      opt->muon_ns_a = (float *)bcir_aligned_calloc64(max_pp, 4u);
      opt->muon_ns_b = (float *)bcir_aligned_calloc64(max_pp, 4u);
      opt->muon_ns_a2 = (float *)bcir_aligned_calloc64(max_pp, 4u);
      if (!opt->muon_ns_x || !opt->muon_ns_y || !opt->muon_ns_a ||
          !opt->muon_ns_b || !opt->muon_ns_a2) {
        bcir_opt_destroy(opt);
        return NULL;
      }
      opt->muon_ns_scratch_bytes = (2u * max_pq + 3u * max_pp) * 4u;
      sb += opt->muon_ns_scratch_bytes;
    }
  }
  opt->opt_only_bytes = ob;
  opt->persistent_bytes = n_params * 4u + ob;
  opt->scratch_bytes = sb;
  return opt;
}

void bcir_opt_destroy(bcir_opt_state *opt) {
  if (!opt) return;
  bcir_aligned_free64(opt->grad_fp32); bcir_aligned_free64(opt->grad_bf16);
  bcir_aligned_free64(opt->m1_fp32); bcir_aligned_free64(opt->m2_fp32);
  bcir_aligned_free64(opt->m1_q8); bcir_aligned_free64(opt->m1_scale);
  bcir_aligned_free64(opt->m2_q8); bcir_aligned_free64(opt->m2_log_min);
  bcir_aligned_free64(opt->m2_log_step);
  if (opt->af_row) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->af_row[s]); prec_dealloc(opt->af_row); }
  if (opt->af_col) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->af_col[s]); prec_dealloc(opt->af_col); }
  if (opt->muon_m1_fp32) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->muon_m1_fp32[s]); prec_dealloc(opt->muon_m1_fp32); }
  if (opt->muon_m1_q8) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->muon_m1_q8[s]); prec_dealloc(opt->muon_m1_q8); }
  if (opt->muon_m1_scale) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->muon_m1_scale[s]); prec_dealloc(opt->muon_m1_scale); }
  if (opt->adam_m1_fp32) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->adam_m1_fp32[s]); prec_dealloc(opt->adam_m1_fp32); }
  if (opt->adam_m2_fp32) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->adam_m2_fp32[s]); prec_dealloc(opt->adam_m2_fp32); }
  if (opt->kahan_comp) { for (size_t s = 0; s < opt->n_shapes; ++s) bcir_aligned_free64(opt->kahan_comp[s]); prec_dealloc(opt->kahan_comp); }
  prec_dealloc(opt->kahan_mask);
  prec_dealloc(opt->adagc_ema);
  prec_dealloc(opt->tensor_grad_norm);
  prec_dealloc(opt->tensor_adagc_scale);
  prec_dealloc(opt->tensor_grad_mom_cos);
  prec_dealloc(opt->tensor_clip_hits);
  bcir_aligned_free64(opt->muon_ns_x); bcir_aligned_free64(opt->muon_ns_y);
  bcir_aligned_free64(opt->muon_ns_a); bcir_aligned_free64(opt->muon_ns_b);
  bcir_aligned_free64(opt->muon_ns_a2);
  prec_dealloc(opt->norm_partials); prec_dealloc(opt->norm_bads); prec_dealloc(opt->q8_clamps_buf);
  bcir_aligned_free64(opt->af_col_scratch);
  prec_dealloc(opt->shapes);
  prec_dealloc(opt);
}

void bcir_opt_reset(bcir_opt_state *opt) {
  if (!opt) return;
  opt->step = 0u; opt->beta1_power = opt->beta2_power = 1.0;
  if (opt->grad_fp32) memset(opt->grad_fp32, 0, opt->n_params * 4u);
  if (opt->grad_bf16) memset(opt->grad_bf16, 0, opt->n_params * 2u);
  if (opt->m1_fp32) memset(opt->m1_fp32, 0, opt->n_params * 4u);
  if (opt->m2_fp32) memset(opt->m2_fp32, 0, opt->n_params * 4u);
  if (opt->m1_q8) memset(opt->m1_q8, 0, opt->n_params);
  if (opt->m1_scale) memset(opt->m1_scale, 0, opt->n_blocks * 4u);
  if (opt->m2_q8) memset(opt->m2_q8, 0, opt->n_params);
  if (opt->m2_log_min) memset(opt->m2_log_min, 0, opt->n_blocks * 4u);
  if (opt->m2_log_step) memset(opt->m2_log_step, 0, opt->n_blocks * 4u);
  if (opt->af_row) {
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      size_t r = opt->shapes[s].rows, c = opt->shapes[s].cols;
      if (opt->af_row[s]) memset(opt->af_row[s], 0, ((c <= 1u || r <= 1u) ? r * c : r) * 4u);
      if (opt->af_col && opt->af_col[s]) memset(opt->af_col[s], 0, c * 4u);
    }
  }
  for (size_t s = 0; s < opt->n_shapes; ++s) {
    size_t len = opt->shapes[s].rows * opt->shapes[s].cols;
    size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
    if (opt->muon_m1_fp32 && opt->muon_m1_fp32[s]) memset(opt->muon_m1_fp32[s], 0, len * 4u);
    if (opt->muon_m1_q8 && opt->muon_m1_q8[s]) memset(opt->muon_m1_q8[s], 0, len);
    if (opt->muon_m1_scale && opt->muon_m1_scale[s]) memset(opt->muon_m1_scale[s], 0, nb * 4u);
    if (opt->adam_m1_fp32 && opt->adam_m1_fp32[s]) memset(opt->adam_m1_fp32[s], 0, len * 4u);
    if (opt->adam_m2_fp32 && opt->adam_m2_fp32[s]) memset(opt->adam_m2_fp32[s], 0, len * 4u);
    if (opt->kahan_comp && opt->kahan_comp[s]) memset(opt->kahan_comp[s], 0, len * 4u);
    if (opt->adagc_ema) opt->adagc_ema[s] = 0.0;
    if (opt->tensor_grad_norm) opt->tensor_grad_norm[s] = 0.0;
    if (opt->tensor_adagc_scale) opt->tensor_adagc_scale[s] = 1.0;
    if (opt->tensor_grad_mom_cos) opt->tensor_grad_mom_cos[s] = 0.0;
    if (opt->tensor_clip_hits) opt->tensor_clip_hits[s] = 0u;
  }
}

void bcir_adagc_config_default(bcir_adagc_config *cfg) {
  if (!cfg) return;
  memset(cfg, 0, sizeof(*cfg));
  cfg->enable = 0;
  cfg->beta = 0.98;
  cfg->lambda_mult = 1.05;
  cfg->abs_floor = 1e-4;
  cfg->warmup_steps = 1u;
}

int bcir_opt_configure_adagc(bcir_opt_state *opt, const bcir_adagc_config *cfg) {
  if (!opt || !cfg) return 1;
  if (!cfg->enable) {
    opt->adagc_enable = 0;
    if (opt->tensor_adagc_scale) {
      for (size_t s = 0; s < opt->n_shapes; ++s) opt->tensor_adagc_scale[s] = 1.0;
    }
    return 0;
  }
  if (!opt->shapes || opt->n_shapes == 0u) return 2;
  if (!isfinite(cfg->beta) || cfg->beta <= 0.0 || cfg->beta >= 1.0 ||
      !isfinite(cfg->lambda_mult) || cfg->lambda_mult <= 0.0 ||
      !isfinite(cfg->abs_floor) || cfg->abs_floor < 0.0) {
    return 3;
  }
  opt->adagc_enable = 1;
  opt->adagc_beta = cfg->beta;
  opt->adagc_lambda = cfg->lambda_mult;
  opt->adagc_abs_floor = cfg->abs_floor;
  opt->adagc_warmup_steps = cfg->warmup_steps > 0u ? cfg->warmup_steps : 1u;
  return 0;
}

int bcir_opt_enable_kahan_tensor(bcir_opt_state *opt, size_t shape_idx, int enable) {
  if (!opt || !opt->shapes || shape_idx >= opt->n_shapes || !opt->kahan_mask || !opt->kahan_comp) {
    return 1;
  }
  size_t len = opt->shapes[shape_idx].rows * opt->shapes[shape_idx].cols;
  size_t bytes = len * sizeof(float);
  if (enable) {
    if (!opt->kahan_mask[shape_idx]) {
      float *buf = (float *)bcir_aligned_calloc64(len > 0u ? len : 1u, sizeof(float));
      if (!buf) return 2;
      opt->kahan_comp[shape_idx] = buf;
      opt->kahan_mask[shape_idx] = 1u;
      opt->kahan_active_tensors++;
      opt->kahan_bytes += bytes;
      opt->opt_only_bytes += bytes;
      opt->persistent_bytes += bytes;
    }
  } else if (opt->kahan_mask[shape_idx]) {
    bcir_aligned_free64(opt->kahan_comp[shape_idx]);
    opt->kahan_comp[shape_idx] = NULL;
    opt->kahan_mask[shape_idx] = 0u;
    opt->kahan_active_tensors--;
    opt->kahan_bytes -= bytes;
    opt->opt_only_bytes -= bytes;
    opt->persistent_bytes -= bytes;
  }
  return 0;
}

int bcir_opt_set_tensor_lr_mult(bcir_opt_state *opt, size_t shape_idx, double lr_mult) {
  if (!opt || !opt->shapes || shape_idx >= opt->n_shapes ||
      !isfinite(lr_mult) || lr_mult <= 0.0) {
    return 1;
  }
  opt->shapes[shape_idx].lr_mult = lr_mult;
  return 0;
}

int bcir_opt_enable_tensor_telemetry(bcir_opt_state *opt, int enable) {
  if (!opt) return 1;
  if (enable && (!opt->shapes || opt->n_shapes == 0u)) return 2;
  opt->tensor_tel_enabled = enable ? 1 : 0;
  return 0;
}

int bcir_opt_get_tensor_telemetry(const bcir_opt_state *opt, size_t shape_idx,
                                  bcir_tensor_opt_telemetry *out) {
  if (!opt || !out || !opt->shapes || shape_idx >= opt->n_shapes) return 1;
  out->last_grad_norm = opt->tensor_grad_norm ? opt->tensor_grad_norm[shape_idx] : 0.0;
  out->ema_grad_norm = opt->adagc_ema ? opt->adagc_ema[shape_idx] : 0.0;
  out->last_adagc_scale = opt->tensor_adagc_scale ? opt->tensor_adagc_scale[shape_idx] : 1.0;
  out->grad_mom_cos = opt->tensor_grad_mom_cos ? opt->tensor_grad_mom_cos[shape_idx] : 0.0;
  out->lr_mult = (opt->shapes[shape_idx].lr_mult > 0.0) ? opt->shapes[shape_idx].lr_mult : 1.0;
  out->adagc_clip_hits = opt->tensor_clip_hits ? opt->tensor_clip_hits[shape_idx] : 0u;
  out->kahan_enabled = (opt->kahan_mask && opt->kahan_mask[shape_idx]) ? 1 : 0;
  return 0;
}

void bcir_opt_zero_grad(bcir_opt_state *opt) {
  if (!opt) return;
  if (opt->grad_fp32) memset(opt->grad_fp32, 0, opt->n_params * 4u);
  if (opt->grad_bf16) memset(opt->grad_bf16, 0, opt->n_params * 2u);
}

#define BCIR_OPT_STATE_MAGIC      0x424349524F505431ULL /* "BCIROPT1" */
#define BCIR_OPT_STATE_VERSION    1ULL
#define BCIR_OPT_STATE_VERSION_V2 2ULL
#define BCIR_OPT_STATE_HDR_U64    10u
#define BCIR_OPT_STATE_HDR_B      (BCIR_OPT_STATE_HDR_U64 * sizeof(uint64_t))
#define BCIR_OPT_V2_EXT_U64       5u
#define BCIR_OPT_V2_EXT_B         (BCIR_OPT_V2_EXT_U64 * sizeof(uint64_t))

static int opt_uses_v2_state(const bcir_opt_state *opt) {
  return opt && (opt->adagc_enable != 0 || opt->kahan_active_tensors > 0u);
}

static uint64_t opt_fnv1a_u64(uint64_t h, uint64_t v) {
  for (unsigned b = 0u; b < 8u; ++b) {
    h ^= (v >> (8u * b)) & 0xFFu;
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t opt_shapes_hash(const bcir_opt_state *opt) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t s = 0; s < opt->n_shapes; ++s) {
    const bcir_opt_tensor_shape *sh = &opt->shapes[s];
    h = opt_fnv1a_u64(h, (uint64_t)sh->offset);
    h = opt_fnv1a_u64(h, (uint64_t)sh->rows);
    h = opt_fnv1a_u64(h, (uint64_t)sh->cols);
    h = opt_fnv1a_u64(h, (uint64_t)(tensor_uses_muon(sh) ? 1u : 0u));
  }
  return h;
}

static int opt_f32_buf_finite(const unsigned char *p, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    float v = 0.0f;
    memcpy(&v, p + i * 4u, 4u);
    if (!isfinite(v)) return 0;
  }
  return 1;
}

static int opt_f64_buf_nonneg_finite(const unsigned char *p, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    double v = 0.0;
    memcpy(&v, p + i * 8u, 8u);
    if (!isfinite(v) || v < 0.0) return 0;
  }
  return 1;
}

static size_t opt_v1_payload_bytes(const bcir_opt_state *opt) {
  size_t total = 0u;
  int k = opt->opt_kind;
  if (k == BCIR_OPT_FP32) {
    total = 2u * opt->n_params * 4u;
  } else if (k == BCIR_OPT_Q8_MOMENTS || k == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
    total = 2u * opt->n_params + 3u * opt->n_blocks * 4u;
  } else if (k == BCIR_OPT_ADAFACTOR_M2 || k == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
    total = opt->n_params * 4u;
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      size_t r = opt->shapes[s].rows, c = opt->shapes[s].cols;
      total += (c <= 1u || r <= 1u) ? (r * c * 4u) : ((r + c) * 4u);
    }
  } else if (k == BCIR_OPT_MUON_FP32 || k == BCIR_OPT_Q8_MUON_BF16_GRAD) {
    int q8_m = (k == BCIR_OPT_Q8_MUON_BF16_GRAD);
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      const bcir_opt_tensor_shape *sh = &opt->shapes[s];
      size_t len = sh->rows * sh->cols;
      if (tensor_uses_muon(sh)) {
        if (q8_m) {
          size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
          total += len + nb * 4u;
        } else {
          total += len * 4u;
        }
      } else {
        total += 2u * len * 4u;
      }
    }
  } else {
    return 0u;
  }
  return total;
}

size_t bcir_opt_state_bytes(const bcir_opt_state *opt) {
  if (!opt) return 0u;
  size_t v1_pay = opt_v1_payload_bytes(opt);
  if (v1_pay == 0u) return 0u;
  size_t total = BCIR_OPT_STATE_HDR_B + v1_pay;
  if (opt_uses_v2_state(opt)) {
    total += BCIR_OPT_V2_EXT_B;
    if (opt->adagc_enable) {
      total += opt->n_shapes * (sizeof(double) + sizeof(uint64_t));
    }
    if (opt->kahan_active_tensors > 0u) {
      total += opt->n_shapes + opt->kahan_bytes;
    }
  }
  return total;
}

int bcir_opt_save_state(const bcir_opt_state *opt, void *dst, size_t dst_bytes) {
  if (!opt || !dst) return 1;
  size_t need = bcir_opt_state_bytes(opt);
  if (need < BCIR_OPT_STATE_HDR_B || dst_bytes != need) return 2;
  unsigned char *p = (unsigned char *)dst;
  uint64_t b1_bits = 0u, b2_bits = 0u;
  memcpy(&b1_bits, &opt->beta1_power, sizeof(uint64_t));
  memcpy(&b2_bits, &opt->beta2_power, sizeof(uint64_t));
  int v2 = opt_uses_v2_state(opt);
  uint64_t hdr[BCIR_OPT_STATE_HDR_U64] = {
    BCIR_OPT_STATE_MAGIC,
    v2 ? BCIR_OPT_STATE_VERSION_V2 : BCIR_OPT_STATE_VERSION,
    (uint64_t)opt->opt_kind,
    (uint64_t)opt->n_params,
    (uint64_t)opt->n_blocks,
    (uint64_t)opt->n_shapes,
    opt_shapes_hash(opt),
    opt->step,
    b1_bits,
    b2_bits,
  };
  memcpy(p, hdr, BCIR_OPT_STATE_HDR_B);
  p += BCIR_OPT_STATE_HDR_B;
  int k = opt->opt_kind;
  if (k == BCIR_OPT_FP32) {
    size_t nb = opt->n_params * 4u;
    memcpy(p, opt->m1_fp32, nb); p += nb;
    memcpy(p, opt->m2_fp32, nb); p += nb;
  } else if (k == BCIR_OPT_Q8_MOMENTS || k == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
    size_t np = opt->n_params, blk = opt->n_blocks * 4u;
    memcpy(p, opt->m1_q8, np); p += np;
    memcpy(p, opt->m1_scale, blk); p += blk;
    memcpy(p, opt->m2_q8, np); p += np;
    memcpy(p, opt->m2_log_min, blk); p += blk;
    memcpy(p, opt->m2_log_step, blk); p += blk;
  } else if (k == BCIR_OPT_ADAFACTOR_M2 || k == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
    size_t nb = opt->n_params * 4u;
    memcpy(p, opt->m1_fp32, nb); p += nb;
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      size_t r = opt->shapes[s].rows, c = opt->shapes[s].cols;
      if (c <= 1u || r <= 1u) {
        size_t bytes = r * c * 4u;
        memcpy(p, opt->af_row[s], bytes); p += bytes;
      } else {
        memcpy(p, opt->af_row[s], r * 4u); p += r * 4u;
        memcpy(p, opt->af_col[s], c * 4u); p += c * 4u;
      }
    }
  } else if (k == BCIR_OPT_MUON_FP32 || k == BCIR_OPT_Q8_MUON_BF16_GRAD) {
    int q8_m = (k == BCIR_OPT_Q8_MUON_BF16_GRAD);
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      const bcir_opt_tensor_shape *sh = &opt->shapes[s];
      size_t len = sh->rows * sh->cols;
      if (tensor_uses_muon(sh)) {
        if (q8_m) {
          size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
          memcpy(p, opt->muon_m1_q8[s], len); p += len;
          memcpy(p, opt->muon_m1_scale[s], nb * 4u); p += nb * 4u;
        } else {
          memcpy(p, opt->muon_m1_fp32[s], len * 4u); p += len * 4u;
        }
      } else {
        memcpy(p, opt->adam_m1_fp32[s], len * 4u); p += len * 4u;
        memcpy(p, opt->adam_m2_fp32[s], len * 4u); p += len * 4u;
      }
    }
  } else {
    return 3;
  }
  if (v2) {
    uint64_t ext_flags = (opt->adagc_enable ? 1ULL : 0ULL) |
                         (opt->kahan_active_tensors > 0u ? 2ULL : 0ULL);
    uint64_t beta_b = 0u, lam_b = 0u, flr_b = 0u;
    memcpy(&beta_b, &opt->adagc_beta, sizeof(uint64_t));
    memcpy(&lam_b, &opt->adagc_lambda, sizeof(uint64_t));
    memcpy(&flr_b, &opt->adagc_abs_floor, sizeof(uint64_t));
    uint64_t ehdr[BCIR_OPT_V2_EXT_U64] = {
      ext_flags, opt->adagc_warmup_steps, beta_b, lam_b, flr_b
    };
    memcpy(p, ehdr, BCIR_OPT_V2_EXT_B); p += BCIR_OPT_V2_EXT_B;
    if (ext_flags & 1ULL) {
      size_t eb = opt->n_shapes * sizeof(double);
      size_t hb = opt->n_shapes * sizeof(uint64_t);
      memcpy(p, opt->adagc_ema, eb); p += eb;
      memcpy(p, opt->tensor_clip_hits, hb); p += hb;
    }
    if (ext_flags & 2ULL) {
      memcpy(p, opt->kahan_mask, opt->n_shapes); p += opt->n_shapes;
      for (size_t s = 0; s < opt->n_shapes; ++s) {
        if (opt->kahan_mask[s]) {
          size_t kb = opt->shapes[s].rows * opt->shapes[s].cols * sizeof(float);
          memcpy(p, opt->kahan_comp[s], kb); p += kb;
        }
      }
    }
  }
  return ((size_t)(p - (unsigned char *)dst) == need) ? 0 : 4;
}

int bcir_opt_load_state(bcir_opt_state *opt, const void *src, size_t src_bytes) {
  if (!opt || !src) return 1;
  size_t need = bcir_opt_state_bytes(opt);
  if (need < BCIR_OPT_STATE_HDR_B || src_bytes != need) return 2;
  const unsigned char *p = (const unsigned char *)src;
  uint64_t hdr[BCIR_OPT_STATE_HDR_U64];
  memcpy(hdr, p, BCIR_OPT_STATE_HDR_B);
  int v2 = opt_uses_v2_state(opt);
  uint64_t exp_ver = v2 ? BCIR_OPT_STATE_VERSION_V2 : BCIR_OPT_STATE_VERSION;
  if (hdr[0] != BCIR_OPT_STATE_MAGIC || hdr[1] != exp_ver ||
      hdr[2] != (uint64_t)opt->opt_kind || hdr[3] != (uint64_t)opt->n_params ||
      hdr[4] != (uint64_t)opt->n_blocks || hdr[5] != (uint64_t)opt->n_shapes ||
      hdr[6] != opt_shapes_hash(opt)) {
    return 3;
  }
  double b1p = 0.0, b2p = 0.0;
  memcpy(&b1p, &hdr[8], sizeof(double));
  memcpy(&b2p, &hdr[9], sizeof(double));
  if (!isfinite(b1p) || !isfinite(b2p) || b1p < 0.0 || b1p > 1.0 || b2p < 0.0 || b2p > 1.0) {
    return 4;
  }
  /* Pass 1: validate finiteness of all float/double payload buffers before mutating opt. */
  const unsigned char *q = p + BCIR_OPT_STATE_HDR_B;
  size_t v1_pay = opt_v1_payload_bytes(opt);
  int k = opt->opt_kind;
  if (k == BCIR_OPT_FP32) {
    if (!opt_f32_buf_finite(q, 2u * opt->n_params)) return 5;
    q += v1_pay;
  } else if (k == BCIR_OPT_Q8_MOMENTS || k == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
    size_t np = opt->n_params, blk = opt->n_blocks * 4u;
    q += np;
    if (!opt_f32_buf_finite(q, opt->n_blocks)) return 5;
    q += blk + np;
    if (!opt_f32_buf_finite(q, 2u * opt->n_blocks)) return 5;
    q += 2u * blk;
  } else if (k == BCIR_OPT_ADAFACTOR_M2 || k == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
    if (!opt_f32_buf_finite(q, v1_pay / 4u)) return 5;
    q += v1_pay;
  } else if (k == BCIR_OPT_MUON_FP32 || k == BCIR_OPT_Q8_MUON_BF16_GRAD) {
    int q8_m = (k == BCIR_OPT_Q8_MUON_BF16_GRAD);
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      const bcir_opt_tensor_shape *sh = &opt->shapes[s];
      size_t len = sh->rows * sh->cols;
      if (tensor_uses_muon(sh)) {
        if (q8_m) {
          size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
          q += len;
          if (!opt_f32_buf_finite(q, nb)) return 5;
          q += nb * 4u;
        } else {
          if (!opt_f32_buf_finite(q, len)) return 5;
          q += len * 4u;
        }
      } else {
        if (!opt_f32_buf_finite(q, 2u * len)) return 5;
        q += 2u * len * 4u;
      }
    }
  } else {
    return 3;
  }
  uint64_t ehdr[BCIR_OPT_V2_EXT_U64] = {0u};
  double v2_beta = 0.0, v2_lam = 0.0, v2_flr = 0.0;
  if (v2) {
    uint64_t exp_flags = (opt->adagc_enable ? 1ULL : 0ULL) |
                         (opt->kahan_active_tensors > 0u ? 2ULL : 0ULL);
    memcpy(ehdr, q, BCIR_OPT_V2_EXT_B); q += BCIR_OPT_V2_EXT_B;
    if (ehdr[0] != exp_flags) return 3;
    memcpy(&v2_beta, &ehdr[2], sizeof(double));
    memcpy(&v2_lam, &ehdr[3], sizeof(double));
    memcpy(&v2_flr, &ehdr[4], sizeof(double));
    if (exp_flags & 1ULL) {
      if (!isfinite(v2_beta) || v2_beta <= 0.0 || v2_beta >= 1.0 ||
          !isfinite(v2_lam) || v2_lam <= 0.0 || !isfinite(v2_flr) || v2_flr < 0.0) {
        return 4;
      }
      if (!opt_f64_buf_nonneg_finite(q, opt->n_shapes)) return 5;
      q += opt->n_shapes * (sizeof(double) + sizeof(uint64_t));
    }
    if (exp_flags & 2ULL) {
      if (memcmp(q, opt->kahan_mask, opt->n_shapes) != 0) return 3;
      q += opt->n_shapes;
      for (size_t s = 0; s < opt->n_shapes; ++s) {
        if (opt->kahan_mask[s]) {
          size_t len = opt->shapes[s].rows * opt->shapes[s].cols;
          if (!opt_f32_buf_finite(q, len)) return 5;
          q += len * sizeof(float);
        }
      }
    }
  }
  if ((size_t)(q - (const unsigned char *)src) != need) return 6;

  /* Pass 2: commit validated state atomically. */
  p += BCIR_OPT_STATE_HDR_B;
  if (k == BCIR_OPT_FP32) {
    size_t nb = opt->n_params * 4u;
    memcpy(opt->m1_fp32, p, nb); p += nb;
    memcpy(opt->m2_fp32, p, nb); p += nb;
  } else if (k == BCIR_OPT_Q8_MOMENTS || k == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
    size_t np = opt->n_params, blk = opt->n_blocks * 4u;
    memcpy(opt->m1_q8, p, np); p += np;
    memcpy(opt->m1_scale, p, blk); p += blk;
    memcpy(opt->m2_q8, p, np); p += np;
    memcpy(opt->m2_log_min, p, blk); p += blk;
    memcpy(opt->m2_log_step, p, blk); p += blk;
  } else if (k == BCIR_OPT_ADAFACTOR_M2 || k == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
    size_t nb = opt->n_params * 4u;
    memcpy(opt->m1_fp32, p, nb); p += nb;
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      size_t r = opt->shapes[s].rows, c = opt->shapes[s].cols;
      if (c <= 1u || r <= 1u) {
        size_t bytes = r * c * 4u;
        memcpy(opt->af_row[s], p, bytes); p += bytes;
      } else {
        memcpy(opt->af_row[s], p, r * 4u); p += r * 4u;
        memcpy(opt->af_col[s], p, c * 4u); p += c * 4u;
      }
    }
  } else if (k == BCIR_OPT_MUON_FP32 || k == BCIR_OPT_Q8_MUON_BF16_GRAD) {
    int q8_m = (k == BCIR_OPT_Q8_MUON_BF16_GRAD);
    for (size_t s = 0; s < opt->n_shapes; ++s) {
      const bcir_opt_tensor_shape *sh = &opt->shapes[s];
      size_t len = sh->rows * sh->cols;
      if (tensor_uses_muon(sh)) {
        if (q8_m) {
          size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
          memcpy(opt->muon_m1_q8[s], p, len); p += len;
          memcpy(opt->muon_m1_scale[s], p, nb * 4u); p += nb * 4u;
        } else {
          memcpy(opt->muon_m1_fp32[s], p, len * 4u); p += len * 4u;
        }
      } else {
        memcpy(opt->adam_m1_fp32[s], p, len * 4u); p += len * 4u;
        memcpy(opt->adam_m2_fp32[s], p, len * 4u); p += len * 4u;
      }
    }
  }
  if (v2) {
    p += BCIR_OPT_V2_EXT_B;
    if (ehdr[0] & 1ULL) {
      size_t eb = opt->n_shapes * sizeof(double);
      size_t hb = opt->n_shapes * sizeof(uint64_t);
      opt->adagc_warmup_steps = ehdr[1] > 0u ? ehdr[1] : 1u;
      opt->adagc_beta = v2_beta;
      opt->adagc_lambda = v2_lam;
      opt->adagc_abs_floor = v2_flr;
      memcpy(opt->adagc_ema, p, eb); p += eb;
      memcpy(opt->tensor_clip_hits, p, hb); p += hb;
    }
    if (ehdr[0] & 2ULL) {
      p += opt->n_shapes;
      for (size_t s = 0; s < opt->n_shapes; ++s) {
        if (opt->kahan_mask[s]) {
          size_t kb = opt->shapes[s].rows * opt->shapes[s].cols * sizeof(float);
          memcpy(opt->kahan_comp[s], p, kb); p += kb;
        }
      }
    }
  }
  if ((size_t)(p - (const unsigned char *)src) != need) return 6;
  opt->step = hdr[7];
  opt->beta1_power = b1p;
  opt->beta2_power = b2p;
  bcir_opt_zero_grad(opt);
  return 0;
}

void bcir_opt_commit_grad_fp32(bcir_opt_state *opt, const float *grad_work, int accumulate) {
  if (!opt || !grad_work) return;
  bcir_opt_commit_grad_span_fp32(opt, 0u, opt->n_params, grad_work, accumulate);
}

void bcir_opt_commit_grad_span_fp32(bcir_opt_state *opt, size_t offset, size_t count,
                                    const float *grad_buf, int accumulate) {
  if (!opt || !grad_buf || count == 0u || offset + count > opt->n_params) return;
  if (opt->grad_fp32) {
    float *dst = opt->grad_fp32 + offset;
    if (!accumulate) memcpy(dst, grad_buf, count * 4u);
    else for (size_t i = 0; i < count; ++i) dst[i] += grad_buf[i];
  } else if (opt->grad_bf16) {
    uint16_t *dst = opt->grad_bf16 + offset;
    if (!accumulate) for (size_t i = 0; i < count; ++i) dst[i] = f32_to_bf16(grad_buf[i]);
    else for (size_t i = 0; i < count; ++i) dst[i] = f32_to_bf16(bf16_to_f32(dst[i]) + grad_buf[i]);
  }
}

static inline float opt_read_grad(const bcir_opt_state *opt, const float *gw, size_t i) {
  return gw ? gw[i] : (opt->grad_fp32 ? opt->grad_fp32[i] : bf16_to_f32(opt->grad_bf16[i]));
}

static inline void opt_kahan_add_delta(float *w_ptr, float *comp_ptr, double delta) {
  double w = (double)(*w_ptr);
  double y = delta - (double)(*comp_ptr);
  float t = (float)(w + y);
  *comp_ptr = (float)(((double)t - w) - y);
  *w_ptr = t;
}

static void fill_tensor_meta(const bcir_opt_state *opt, size_t idx, size_t *cursor,
                             double *lr_mult, int *apply_wd, double *t_scale,
                             float **kcomp, size_t *k_local) {
  *lr_mult = 1.0; *apply_wd = 1; *t_scale = 1.0; *kcomp = NULL; *k_local = 0u;
  if (!opt->shapes || opt->n_shapes == 0u) return;
  while (*cursor + 1u < opt->n_shapes &&
         idx >= opt->shapes[*cursor].offset + opt->shapes[*cursor].rows * opt->shapes[*cursor].cols) (*cursor)++;
  const bcir_opt_tensor_shape *sh = &opt->shapes[*cursor];
  if (idx >= sh->offset && idx < sh->offset + sh->rows * sh->cols) {
    *lr_mult = (sh->lr_mult > 0.0) ? sh->lr_mult : 1.0;
    *apply_wd = sh->apply_wd;
    if (opt->adagc_enable && opt->tensor_adagc_scale) *t_scale = opt->tensor_adagc_scale[*cursor];
    if (opt->kahan_comp && opt->kahan_comp[*cursor]) {
      *kcomp = opt->kahan_comp[*cursor];
      *k_local = idx - sh->offset;
    }
  }
}

typedef struct opt_norm_ctx {
  const bcir_opt_state *opt; const float *weights, *gw; double grad_scale;
  size_t chunk; double *partials; int *bad_flags;
} opt_norm_ctx;

static void opt_norm_worker(void *vctx, size_t task) {
  opt_norm_ctx *c = (opt_norm_ctx *)vctx;
  size_t s = task * c->chunk, e = (s + c->chunk <= c->opt->n_params) ? (s + c->chunk) : c->opt->n_params;
  double ss = 0.0; int bad = 0;
  if (c->gw) {
    for (size_t i = s; i < e; ++i) {
      float w = c->weights[i], g = c->gw[i];
      if (!isfinite(w) || !isfinite(g)) { bad = 1; break; }
      double gs = (double)g * c->grad_scale; ss += gs * gs;
    }
  } else if (c->opt->grad_fp32) {
    const float *gp = c->opt->grad_fp32;
    for (size_t i = s; i < e; ++i) {
      float w = c->weights[i], g = gp[i];
      if (!isfinite(w) || !isfinite(g)) { bad = 1; break; }
      double gs = (double)g * c->grad_scale; ss += gs * gs;
    }
  } else {
    const uint16_t *gb = c->opt->grad_bf16;
    for (size_t i = s; i < e; ++i) {
      float w = c->weights[i], g = bf16_to_f32(gb[i]);
      if (!isfinite(w) || !isfinite(g)) { bad = 1; break; }
      double gs = (double)g * c->grad_scale; ss += gs * gs;
    }
  }
  c->partials[task] = ss; c->bad_flags[task] = bad;
}

typedef struct opt_tensor_stats_ctx {
  bcir_opt_state *opt;
  const float *gw;
  double grad_scale;
} opt_tensor_stats_ctx;

/* One task per shape -> strictly ascending reduction order, 0 bit drift 1T..NT. */
static void opt_tensor_stats_worker(void *vctx, size_t task) {
  opt_tensor_stats_ctx *c = (opt_tensor_stats_ctx *)vctx;
  bcir_opt_state *opt = c->opt;
  size_t s = task;
  const bcir_opt_tensor_shape *sh = &opt->shapes[s];
  size_t off = sh->offset, len = sh->rows * sh->cols;
  double g_ss = 0.0, m_ss = 0.0, dot_gm = 0.0;
  int want_cos = opt->tensor_tel_enabled && (opt->step > 0u);
  int k = opt->opt_kind;
  int is_muon_t = (k == BCIR_OPT_MUON_FP32 || k == BCIR_OPT_Q8_MUON_BF16_GRAD) && tensor_uses_muon(sh);

  for (size_t j = 0; j < len; ++j) {
    size_t idx = off + j;
    double g = (double)opt_read_grad(opt, c->gw, idx) * c->grad_scale;
    g_ss += g * g;
    if (want_cos) {
      double mprev = 0.0;
      if (k == BCIR_OPT_FP32 || k == BCIR_OPT_ADAFACTOR_M2 || k == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
        mprev = (double)opt->m1_fp32[idx];
      } else if (k == BCIR_OPT_Q8_MOMENTS || k == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
        size_t b = idx / BCIR_OPT_Q8_BLOCK;
        mprev = (double)opt->m1_q8[idx] * (double)opt->m1_scale[b];
      } else if (k == BCIR_OPT_MUON_FP32) {
        mprev = is_muon_t ? (double)opt->muon_m1_fp32[s][j] : (double)opt->adam_m1_fp32[s][j];
      } else if (k == BCIR_OPT_Q8_MUON_BF16_GRAD) {
        if (is_muon_t) {
          size_t b = j / BCIR_OPT_Q8_BLOCK;
          mprev = (double)opt->muon_m1_q8[s][j] * (double)opt->muon_m1_scale[s][b];
        } else {
          mprev = (double)opt->adam_m1_fp32[s][j];
        }
      }
      m_ss += mprev * mprev;
      dot_gm += g * mprev;
    }
  }
  double gn = sqrt(g_ss);
  opt->tensor_grad_norm[s] = gn;
  if (want_cos && g_ss > 1e-30 && m_ss > 1e-30) {
    double cosv = dot_gm / sqrt(g_ss * m_ss);
    if (cosv > 1.0) cosv = 1.0;
    else if (cosv < -1.0) cosv = -1.0;
    opt->tensor_grad_mom_cos[s] = cosv;
  } else {
    opt->tensor_grad_mom_cos[s] = 0.0;
  }

  if (opt->adagc_enable) {
    double beta = opt->adagc_beta, obeta = 1.0 - beta;
    uint64_t warmup = opt->adagc_warmup_steps > 0u ? opt->adagc_warmup_steps : 1u;
    double ema = opt->adagc_ema[s];
    double scale_s = 1.0;
    if (opt->step < warmup || ema <= 0.0) {
      ema = (ema <= 0.0) ? gn : (beta * ema + obeta * gn);
    } else {
      double thr = opt->adagc_lambda * ema;
      if (thr < opt->adagc_abs_floor) thr = opt->adagc_abs_floor;
      double g_clip = gn;
      if (gn > thr && gn > 0.0) {
        scale_s = thr / gn;
        g_clip = thr;
        opt->tensor_clip_hits[s]++;
      }
      ema = beta * ema + obeta * g_clip;
    }
    opt->adagc_ema[s] = ema;
    opt->tensor_adagc_scale[s] = scale_s;
  } else {
    opt->tensor_adagc_scale[s] = 1.0;
  }
}

typedef struct opt_fp32_ctx {
  bcir_opt_state *opt; float *weights; const float *gw;
  double lr, b1, b2, bc1, bc2, eps, wd, clip_scale; size_t chunk;
} opt_fp32_ctx;

static void opt_fp32_worker(void *vctx, size_t task) {
  opt_fp32_ctx *c = (opt_fp32_ctx *)vctx;
  size_t s = task * c->chunk, e = (s + c->chunk <= c->opt->n_params) ? (s + c->chunk) : c->opt->n_params;
  size_t cur = 0u;
  if (c->opt->shapes && c->opt->n_shapes > 0u) {
    size_t lo = 0u, hi = c->opt->n_shapes - 1u;
    while (lo < hi) {
      size_t mid = (lo + hi) / 2u;
      if (c->opt->shapes[mid].offset + c->opt->shapes[mid].rows * c->opt->shapes[mid].cols <= s) lo = mid + 1u;
      else hi = mid;
    }
    cur = lo;
  }
  float *m1p = c->opt->m1_fp32, *m2p = c->opt->m2_fp32, *wp = c->weights;
  const float *gp = c->gw ? c->gw : c->opt->grad_fp32;
  double b1 = c->b1, b2 = c->b2, ob1 = 1.0 - b1, ob2 = 1.0 - b2;
  double inv_bc1 = 1.0 / c->bc1, inv_bc2 = 1.0 / c->bc2, eps = c->eps, wd = c->wd, cs = c->clip_scale;
  size_t i = s;
  while (i < e) {
    double lrm = 1.0, t_scale = 1.0; int awd = 1; size_t seg_end = e, sh_off = 0u;
    float *kcomp = NULL;
    if (c->opt->shapes && cur < c->opt->n_shapes) {
      const bcir_opt_tensor_shape *sh = &c->opt->shapes[cur];
      size_t sh_end = sh->offset + sh->rows * sh->cols;
      if (i >= sh->offset && i < sh_end) {
        lrm = (sh->lr_mult > 0.0) ? sh->lr_mult : 1.0; awd = sh->apply_wd;
        sh_off = sh->offset;
        if (c->opt->adagc_enable && c->opt->tensor_adagc_scale) t_scale = c->opt->tensor_adagc_scale[cur];
        if (c->opt->kahan_comp) kcomp = c->opt->kahan_comp[cur];
        if (sh_end < seg_end) seg_end = sh_end;
      }
      if (seg_end >= sh_end) cur++;
    }
    double lr_eff = c->lr * lrm, step_c = lr_eff * inv_bc1;
    double wd_fac = (awd && wd > 0.0) ? (1.0 - lr_eff * wd) : 1.0;
    double g_mul = cs * t_scale;
    if (!kcomp) {
      for (; i < seg_end; ++i) {
        double g = (double)gp[i] * g_mul;
        double m1 = b1 * (double)m1p[i] + ob1 * g;
        double m2 = b2 * (double)m2p[i] + ob2 * g * g;
        m1p[i] = (float)m1; m2p[i] = (float)m2;
        wp[i] = (float)((double)wp[i] * wd_fac - step_c * m1 / (sqrt(m2 * inv_bc2) + eps));
      }
    } else {
      for (; i < seg_end; ++i) {
        double g = (double)gp[i] * g_mul;
        double m1 = b1 * (double)m1p[i] + ob1 * g;
        double m2 = b2 * (double)m2p[i] + ob2 * g * g;
        m1p[i] = (float)m1; m2p[i] = (float)m2;
        double delta = (awd && wd > 0.0 ? -lr_eff * wd * (double)wp[i] : 0.0) -
                       step_c * m1 / (sqrt(m2 * inv_bc2) + eps);
        opt_kahan_add_delta(&wp[i], &kcomp[i - sh_off], delta);
      }
    }
  }
}

typedef struct opt_q8_ctx {
  bcir_opt_state *opt; float *weights; const float *gw;
  double lr, b1, b2, bc1, bc2, eps, wd, clip_scale; size_t blocks_per_task; uint64_t *floor_clamps;
} opt_q8_ctx;

static void opt_q8_worker(void *vctx, size_t task) {
  opt_q8_ctx *c = (opt_q8_ctx *)vctx;
  size_t b0 = task * c->blocks_per_task;
  size_t b1_end = (b0 + c->blocks_per_task <= c->opt->n_blocks) ? (b0 + c->blocks_per_task) : c->opt->n_blocks;
  size_t cur = 0u;
  if (c->opt->shapes && c->opt->n_shapes > 0u) {
    size_t s0 = b0 * BCIR_OPT_Q8_BLOCK, lo = 0u, hi = c->opt->n_shapes - 1u;
    while (lo < hi) {
      size_t mid = (lo + hi) / 2u;
      if (c->opt->shapes[mid].offset + c->opt->shapes[mid].rows * c->opt->shapes[mid].cols <= s0) lo = mid + 1u;
      else hi = mid;
    }
    cur = lo;
  }
  double b1 = c->b1, b2 = c->b2, ob1 = 1.0 - b1, ob2 = 1.0 - b2;
  double inv_bc1 = 1.0 / c->bc1, inv_bc2 = 1.0 / c->bc2, eps = c->eps, wd = c->wd, cs = c->clip_scale;
  double tmp_m1[BCIR_OPT_Q8_BLOCK], tmp_m2[BCIR_OPT_Q8_BLOCK];
  uint64_t clamps = 0u;
  for (size_t b = b0; b < b1_end; ++b) {
    size_t s = b * BCIR_OPT_Q8_BLOCK, e = (s + BCIR_OPT_Q8_BLOCK <= c->opt->n_params) ? (s + BCIR_OPT_Q8_BLOCK) : c->opt->n_params;
    size_t len = e - s;
    double s1 = (double)c->opt->m1_scale[b], lmin = (double)c->opt->m2_log_min[b], lstep = (double)c->opt->m2_log_step[b];
    double max_abs_m1 = 0.0, max_log_m2 = -1e30, min_log_m2 = 1e30;
    int any_pos_m2 = 0;
    for (size_t j = 0; j < len; ++j) {
      size_t i = s + j, k_local = 0u; double lrm, t_scale; int awd; float *kcomp = NULL;
      fill_tensor_meta(c->opt, i, &cur, &lrm, &awd, &t_scale, &kcomp, &k_local);
      double lr_eff = c->lr * lrm;
      double g = (double)opt_read_grad(c->opt, c->gw, i) * (cs * t_scale);
      double prev_m1 = (double)c->opt->m1_q8[i] * s1;
      uint8_t c2 = c->opt->m2_q8[i];
      double prev_m2 = (c2 == 0u) ? 0.0 : exp2(lmin + (double)(c2 - 1u) * lstep);
      double m1 = b1 * prev_m1 + ob1 * g, m2 = b2 * prev_m2 + ob2 * g * g;
      tmp_m1[j] = m1; tmp_m2[j] = m2;
      if (fabs(m1) > max_abs_m1) max_abs_m1 = fabs(m1);
      if (m2 > 1e-34) {
        double lm = log2(m2);
        if (lm > max_log_m2) max_log_m2 = lm;
        if (lm < min_log_m2) min_log_m2 = lm;
        any_pos_m2 = 1;
      }
      if (!kcomp) {
        double w = (double)c->weights[i];
        if (awd && wd > 0.0) w -= lr_eff * wd * w;
        w -= lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
        c->weights[i] = (float)w;
      } else {
        double delta = (awd && wd > 0.0 ? -lr_eff * wd * (double)c->weights[i] : 0.0) -
                       lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
        opt_kahan_add_delta(&c->weights[i], &kcomp[k_local], delta);
      }
    }
    if (max_abs_m1 > 0.0) {
      double sc = max_abs_m1 / 127.0, inv = 127.0 / max_abs_m1;
      c->opt->m1_scale[b] = (float)sc;
      for (size_t j = 0; j < len; ++j) {
        long q = lrint(tmp_m1[j] * inv);
        c->opt->m1_q8[s + j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
      }
    } else {
      c->opt->m1_scale[b] = 0.0f;
      for (size_t j = 0; j < len; ++j) c->opt->m1_q8[s + j] = 0;
    }
    if (!any_pos_m2) {
      c->opt->m2_log_min[b] = c->opt->m2_log_step[b] = 0.0f;
      for (size_t j = 0; j < len; ++j) c->opt->m2_q8[s + j] = 0u;
    } else {
      if (min_log_m2 < max_log_m2 - 48.0) { min_log_m2 = max_log_m2 - 48.0; clamps++; }
      double span = max_log_m2 - min_log_m2, step = (span > 1e-6) ? (span / 254.0) : 1.0;
      c->opt->m2_log_min[b] = (float)min_log_m2; c->opt->m2_log_step[b] = (float)step;
      for (size_t j = 0; j < len; ++j) {
        if (tmp_m2[j] <= 1e-34) c->opt->m2_q8[s + j] = 0u;
        else {
          double lm = log2(tmp_m2[j]);
          if (lm <= min_log_m2) c->opt->m2_q8[s + j] = 1u;
          else {
            long q = 1L + lrint((lm - min_log_m2) / step);
            c->opt->m2_q8[s + j] = (uint8_t)(q < 1L ? 1L : (q > 255L ? 255L : q));
          }
        }
      }
    }
  }
  c->floor_clamps[task] = clamps;
}

typedef struct opt_af_ctx {
  bcir_opt_state *opt; float *weights; const float *gw;
  double lr, b1, b2, bc1, bc2, eps, wd, clip_scale;
} opt_af_ctx;

static void opt_af_worker(void *vctx, size_t task) {
  opt_af_ctx *c = (opt_af_ctx *)vctx;
  size_t s = task;
  const bcir_opt_tensor_shape *sh = &c->opt->shapes[s];
  size_t off = sh->offset, R = sh->rows, C = sh->cols;
  double lr_eff = c->lr * ((sh->lr_mult > 0.0) ? sh->lr_mult : 1.0);
  double b1 = c->b1, b2 = c->b2, ob1 = 1.0 - b1, ob2 = 1.0 - b2;
  double inv_bc1 = 1.0 / c->bc1, inv_bc2 = 1.0 / c->bc2, eps = c->eps, wd = c->wd;
  double t_scale = (c->opt->adagc_enable && c->opt->tensor_adagc_scale) ? c->opt->tensor_adagc_scale[s] : 1.0;
  double cs = c->clip_scale * t_scale;
  float *kcomp = c->opt->kahan_comp ? c->opt->kahan_comp[s] : NULL;
  if (R <= 1u || C <= 1u) {
    size_t len = R * C; float *v = c->opt->af_row[s];
    for (size_t k = 0; k < len; ++k) {
      size_t i = off + k; double g = (double)opt_read_grad(c->opt, c->gw, i) * cs;
      double m1 = b1 * (double)c->opt->m1_fp32[i] + ob1 * g, m2 = b2 * (double)v[k] + ob2 * g * g;
      c->opt->m1_fp32[i] = (float)m1; v[k] = (float)m2;
      if (!kcomp) {
        double w = (double)c->weights[i];
        if (sh->apply_wd && wd > 0.0) w -= lr_eff * wd * w;
        w -= lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
        c->weights[i] = (float)w;
      } else {
        double delta = (sh->apply_wd && wd > 0.0 ? -lr_eff * wd * (double)c->weights[i] : 0.0) -
                       lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
        opt_kahan_add_delta(&c->weights[i], &kcomp[k], delta);
      }
    }
  } else {
    float *vr = c->opt->af_row[s], *vc = c->opt->af_col[s];
    double invC = 1.0 / (double)C, invR = 1.0 / (double)R;
    /* Locate disjoint pre-allocated column scratch for shape s (zero malloc). */
    size_t col_off = 0u;
    for (size_t q = 0; q < s; ++q) {
      if (c->opt->shapes[q].rows > 1u && c->opt->shapes[q].cols > 1u) col_off += c->opt->shapes[q].cols;
    }
    double *cacc = c->opt->af_col_scratch + col_off;
    memset(cacc, 0, C * sizeof(double));
    for (size_t r = 0; r < R; ++r) {
      double rmean = 0.0;
      for (size_t col = 0; col < C; ++col) {
        double g = (double)opt_read_grad(c->opt, c->gw, off + r * C + col) * cs;
        double g2 = g * g; rmean += g2; cacc[col] += g2;
      }
      vr[r] = (float)(b2 * (double)vr[r] + ob2 * (rmean * invC));
    }
    for (size_t col = 0; col < C; ++col) vc[col] = (float)(b2 * (double)vc[col] + ob2 * (cacc[col] * invR));
    double mean_vr = 0.0;
    for (size_t r = 0; r < R; ++r) mean_vr += (double)vr[r];
    mean_vr = fmax(mean_vr * invR, 1e-30);
    for (size_t r = 0; r < R; ++r) {
      double rfac = (double)vr[r] / mean_vr;
      for (size_t col = 0; col < C; ++col) {
        size_t k = r * C + col, i = off + k;
        double g = (double)opt_read_grad(c->opt, c->gw, i) * cs;
        double m1 = b1 * (double)c->opt->m1_fp32[i] + ob1 * g, m2 = rfac * (double)vc[col];
        c->opt->m1_fp32[i] = (float)m1;
        if (!kcomp) {
          double w = (double)c->weights[i];
          if (sh->apply_wd && wd > 0.0) w -= lr_eff * wd * w;
          w -= lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
          c->weights[i] = (float)w;
        } else {
          double delta = (sh->apply_wd && wd > 0.0 ? -lr_eff * wd * (double)c->weights[i] : 0.0) -
                         lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
          opt_kahan_add_delta(&c->weights[i], &kcomp[k], delta);
        }
      }
    }
  }
}

/* Muon-FP32 (12.0 B/p) and Muon-8L/AdamW-32 + BF16Grad (~7.05 B/p, arXiv:2509.23106v3).
 * Non-Muon tensors run AdamW on the pool; 2D Muon tensors run 5-step Newton-Schulz
 * polar orthogonalization using bcir_tensor_mm_kblock_pool (0 bit drift 1T..NT). */
typedef struct opt_muon_adam_ctx {
  bcir_opt_state *opt; float *weights; const float *gw;
  double lr, b1, b2, bc1, bc2, eps, wd, clip_scale;
} opt_muon_adam_ctx;

static void opt_muon_adam_worker(void *vctx, size_t task) {
  opt_muon_adam_ctx *c = (opt_muon_adam_ctx *)vctx;
  size_t s = task;
  const bcir_opt_tensor_shape *sh = &c->opt->shapes[s];
  if (tensor_uses_muon(sh)) return;
  size_t off = sh->offset, len = sh->rows * sh->cols;
  double lr_eff = c->lr * ((sh->lr_mult > 0.0) ? sh->lr_mult : 1.0);
  double b1 = c->b1, b2 = c->b2, ob1 = 1.0 - b1, ob2 = 1.0 - b2;
  double inv_bc1 = 1.0 / c->bc1, inv_bc2 = 1.0 / c->bc2, eps = c->eps, wd = c->wd;
  double t_scale = (c->opt->adagc_enable && c->opt->tensor_adagc_scale) ? c->opt->tensor_adagc_scale[s] : 1.0;
  double cs = c->clip_scale * t_scale;
  float *m1p = c->opt->adam_m1_fp32[s], *m2p = c->opt->adam_m2_fp32[s];
  float *kcomp = c->opt->kahan_comp ? c->opt->kahan_comp[s] : NULL;
  for (size_t k = 0; k < len; ++k) {
    size_t i = off + k;
    double g = (double)opt_read_grad(c->opt, c->gw, i) * cs;
    double m1 = b1 * (double)m1p[k] + ob1 * g;
    double m2 = b2 * (double)m2p[k] + ob2 * g * g;
    m1p[k] = (float)m1; m2p[k] = (float)m2;
    if (!kcomp) {
      double w = (double)c->weights[i];
      if (sh->apply_wd && wd > 0.0) w -= lr_eff * wd * w;
      w -= lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
      c->weights[i] = (float)w;
    } else {
      double delta = (sh->apply_wd && wd > 0.0 ? -lr_eff * wd * (double)c->weights[i] : 0.0) -
                     lr_eff * (m1 * inv_bc1) / (sqrt(m2 * inv_bc2) + eps);
      opt_kahan_add_delta(&c->weights[i], &kcomp[k], delta);
    }
  }
}

static void opt_muon_2d_update(bcir_tensor_pool *pool, bcir_opt_state *opt, float *weights,
                               const float *gw, const bcir_adamw_params *cfg,
                               double clip_scale, bcir_prec_telemetry *tel) {
  int q8_m = (opt->opt_kind == BCIR_OPT_Q8_MUON_BF16_GRAD);
  double mu = cfg->beta1, omu = 1.0 - mu;
  double tmp_m1[BCIR_OPT_Q8_BLOCK];
  for (size_t s = 0; s < opt->n_shapes; ++s) {
    const bcir_opt_tensor_shape *sh = &opt->shapes[s];
    if (!tensor_uses_muon(sh)) continue;
    size_t off = sh->offset, R = sh->rows, C = sh->cols, len = R * C;
    int trans = (R > C);
    size_t p = trans ? C : R;
    size_t q = trans ? R : C;
    float *xbuf = opt->muon_ns_x;
    double t_scale = (opt->adagc_enable && opt->tensor_adagc_scale) ? opt->tensor_adagc_scale[s] : 1.0;
    double cs = clip_scale * t_scale;
    float *kcomp = opt->kahan_comp ? opt->kahan_comp[s] : NULL;

    if (!q8_m) {
      float *m1p = opt->muon_m1_fp32[s];
      for (size_t r = 0; r < R; ++r) {
        for (size_t col = 0; col < C; ++col) {
          size_t k = r * C + col;
          double g = (double)opt_read_grad(opt, gw, off + k) * cs;
          double m1 = mu * (double)m1p[k] + omu * g;
          m1p[k] = (float)m1;
          float nest = (float)(mu * m1 + omu * g);
          if (trans) xbuf[col * R + r] = nest;
          else xbuf[k] = nest;
        }
      }
    } else {
      int8_t *mq = opt->muon_m1_q8[s];
      float *ms = opt->muon_m1_scale[s];
      size_t nb = (len + BCIR_OPT_Q8_BLOCK - 1u) / BCIR_OPT_Q8_BLOCK;
      for (size_t b = 0; b < nb; ++b) {
        size_t st = b * BCIR_OPT_Q8_BLOCK;
        size_t en = (st + BCIR_OPT_Q8_BLOCK <= len) ? (st + BCIR_OPT_Q8_BLOCK) : len;
        size_t blen = en - st;
        double sc_old = (double)ms[b];
        double max_abs = 0.0;
        for (size_t j = 0; j < blen; ++j) {
          size_t k = st + j;
          double g = (double)opt_read_grad(opt, gw, off + k) * cs;
          double m_prev = (double)mq[k] * sc_old;
          double m1 = mu * m_prev + omu * g;
          tmp_m1[j] = m1;
          if (fabs(m1) > max_abs) max_abs = fabs(m1);
          float nest = (float)(mu * m1 + omu * g);
          if (trans) {
            size_t r = k / C, col = k % C;
            xbuf[col * R + r] = nest;
          } else {
            xbuf[k] = nest;
          }
        }
        if (max_abs > 0.0) {
          double sc = max_abs / 127.0, inv = 127.0 / max_abs;
          ms[b] = (float)sc;
          for (size_t j = 0; j < blen; ++j) {
            long qv = lrint(tmp_m1[j] * inv);
            mq[st + j] = (int8_t)(qv > 127 ? 127 : (qv < -127 ? -127 : qv));
          }
        } else {
          ms[b] = 0.0f;
          for (size_t j = 0; j < blen; ++j) mq[st + j] = 0;
        }
      }
    }

    newton_schulz_5step(pool, p, q, opt->muon_ns_x, opt->muon_ns_y,
                        opt->muon_ns_a, opt->muon_ns_b, opt->muon_ns_a2);
    if (tel) tel->muon_ns_calls++;

    /* Moonlight RMS-matched scaling: 0.2 * sqrt(max(R, C)) matches AdamW update RMS. */
    double lr_eff = cfg->lr * ((sh->lr_mult > 0.0) ? sh->lr_mult : 1.0);
    double rms_scale = 0.2 * sqrt((double)q);
    double step_scale = lr_eff * rms_scale;
    double wd_fac = (sh->apply_wd && cfg->weight_decay > 0.0) ? (1.0 - lr_eff * cfg->weight_decay) : 1.0;
    if (!kcomp) {
      for (size_t r = 0; r < R; ++r) {
        for (size_t col = 0; col < C; ++col) {
          size_t k = r * C + col;
          float upd = trans ? xbuf[col * R + r] : xbuf[k];
          weights[off + k] = (float)((double)weights[off + k] * wd_fac - step_scale * (double)upd);
        }
      }
    } else {
      for (size_t r = 0; r < R; ++r) {
        for (size_t col = 0; col < C; ++col) {
          size_t k = r * C + col;
          float upd = trans ? xbuf[col * R + r] : xbuf[k];
          double delta = (sh->apply_wd && cfg->weight_decay > 0.0 ? -lr_eff * cfg->weight_decay * (double)weights[off + k] : 0.0) -
                         step_scale * (double)upd;
          opt_kahan_add_delta(&weights[off + k], &kcomp[k], delta);
        }
      }
    }
  }
}

int bcir_opt_step(bcir_tensor_pool *pool, bcir_opt_state *opt, float *weights,
                  const float *grad_work_or_null, const bcir_adamw_params *cfg,
                  double grad_scale, double *out_grad_norm, int *out_clipped,
                  bcir_prec_telemetry *tel) {
  if (!opt || !weights || !cfg || !isfinite(cfg->lr) || cfg->lr < 0.0 ||
      !isfinite(grad_scale) || !isfinite(cfg->epsilon) || cfg->epsilon <= 0.0) {
    return 1;
  }
  size_t chunk = 65536u, n_tasks = (opt->n_params + chunk - 1u) / chunk;
  opt_norm_ctx nctx = { opt, weights, grad_work_or_null, grad_scale, chunk,
                        opt->norm_partials, opt->norm_bads };
  bcir_tensor_pool_run(pool, opt_norm_worker, &nctx, n_tasks);
  double sum_sq = 0.0; int any_bad = 0;
  for (size_t t = 0; t < n_tasks; ++t) {
    if (opt->norm_bads[t]) any_bad = 1;
    sum_sq += opt->norm_partials[t];
  }
  if (any_bad || !isfinite(sum_sq)) {
    if (tel) tel->nonfinite_events++;
    return 3;
  }

  double gnorm = sqrt(sum_sq), clip_scale = grad_scale; int clipped = 0;
  if ((opt->adagc_enable || opt->tensor_tel_enabled) && opt->shapes && opt->n_shapes > 0u) {
    opt_tensor_stats_ctx sctx = { opt, grad_work_or_null, grad_scale };
    bcir_tensor_pool_run(pool, opt_tensor_stats_worker, &sctx, opt->n_shapes);
    if (opt->adagc_enable) {
      double post_sq = 0.0;
      for (size_t s = 0; s < opt->n_shapes; ++s) {
        double gn_s = opt->tensor_grad_norm[s] * opt->tensor_adagc_scale[s];
        post_sq += gn_s * gn_s;
        if (opt->tensor_adagc_scale[s] < 1.0) clipped = 1;
      }
      double post_gnorm = sqrt(post_sq);
      if (cfg->grad_clip > 0.0 && post_gnorm > cfg->grad_clip) {
        clip_scale = grad_scale * (cfg->grad_clip / post_gnorm);
        clipped = 1;
      }
    } else if (cfg->grad_clip > 0.0 && gnorm > cfg->grad_clip) {
      clip_scale = grad_scale * (cfg->grad_clip / gnorm);
      clipped = 1;
    }
  } else if (cfg->grad_clip > 0.0 && gnorm > cfg->grad_clip) {
    clip_scale = grad_scale * (cfg->grad_clip / gnorm);
    clipped = 1;
  }
  if (out_grad_norm) *out_grad_norm = gnorm;
  if (out_clipped) *out_clipped = clipped;
  if (tel) {
    tel->total_steps++;
    if (clipped) tel->grad_clip_hits++;
    if (gnorm > tel->max_grad_norm) tel->max_grad_norm = gnorm;
  }

  uint64_t next_step = opt->step + 1u;
  double b1p = opt->beta1_power * cfg->beta1, b2p = opt->beta2_power * cfg->beta2;
  double bc1 = 1.0 - b1p, bc2 = 1.0 - b2p;

  if (opt->opt_kind == BCIR_OPT_FP32) {
    opt_fp32_ctx fctx = { opt, weights, grad_work_or_null, cfg->lr, cfg->beta1, cfg->beta2,
                          bc1, bc2, cfg->epsilon, cfg->weight_decay, clip_scale, chunk };
    bcir_tensor_pool_run(pool, opt_fp32_worker, &fctx, n_tasks);
  } else if (opt->opt_kind == BCIR_OPT_Q8_MOMENTS || opt->opt_kind == BCIR_OPT_Q8_MOMENTS_BF16_GRAD) {
    size_t bpt = 256u, qt = (opt->n_blocks + bpt - 1u) / bpt;
    memset(opt->q8_clamps_buf, 0, qt * sizeof(uint64_t));
    opt_q8_ctx qctx = { opt, weights, grad_work_or_null, cfg->lr, cfg->beta1, cfg->beta2,
                        bc1, bc2, cfg->epsilon, cfg->weight_decay, clip_scale, bpt, opt->q8_clamps_buf };
    bcir_tensor_pool_run(pool, opt_q8_worker, &qctx, qt);
    if (tel) for (size_t t = 0; t < qt; ++t) tel->q8_m2_floor_clamps += opt->q8_clamps_buf[t];
  } else if (opt->opt_kind == BCIR_OPT_ADAFACTOR_M2 || opt->opt_kind == BCIR_OPT_ADAFACTOR_BF16_GRAD) {
    opt_af_ctx actx = { opt, weights, grad_work_or_null, cfg->lr, cfg->beta1, cfg->beta2,
                        bc1, bc2, cfg->epsilon, cfg->weight_decay, clip_scale };
    bcir_tensor_pool_run(pool, opt_af_worker, &actx, opt->n_shapes);
  } else if (opt->opt_kind == BCIR_OPT_MUON_FP32 || opt->opt_kind == BCIR_OPT_Q8_MUON_BF16_GRAD) {
    opt_muon_adam_ctx mactx = { opt, weights, grad_work_or_null, cfg->lr, cfg->beta1, cfg->beta2,
                                bc1, bc2, cfg->epsilon, cfg->weight_decay, clip_scale };
    bcir_tensor_pool_run(pool, opt_muon_adam_worker, &mactx, opt->n_shapes);
    opt_muon_2d_update(pool, opt, weights, grad_work_or_null, cfg, clip_scale, tel);
  }
  opt->step = next_step; opt->beta1_power = b1p; opt->beta2_power = b2p;
  return 0;
}

/* ==================== Deterministic C11 Ziggurat + mup_v1 Init ==================== */
static inline uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static inline double u64_to_u01(uint64_t x) { return ((double)(x >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

static void build_zig_tables(uint32_t kn[128], double wn[128], double fn[128]) {
  const double m1 = 2147483648.0, vn = 9.91256303526217e-3;
  double dn = 3.442619855899, tn = dn, q = vn / exp(-0.5 * dn * dn);
  kn[0] = (uint32_t)((dn / q) * m1); kn[1] = 0u;
  wn[0] = q / m1; wn[127] = dn / m1; fn[0] = 1.0; fn[127] = exp(-0.5 * dn * dn);
  for (int i = 126; i >= 1; --i) {
    dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
    kn[i + 1] = (uint32_t)((dn / tn) * m1); tn = dn;
    fn[i] = exp(-0.5 * dn * dn); wn[i] = dn / m1;
  }
}

static void fill_normal_c11(float *dst, size_t n, double std, uint64_t *rng) {
  uint32_t kn[128]; double wn[128], fn[128];
  build_zig_tables(kn, wn, fn);
  const double r = 3.442619855899;
  for (size_t i = 0; i < n; ++i) {
    for (;;) {
      uint64_t u = splitmix64_next(rng);
      int32_t hz = (int32_t)(u >> 32);
      uint32_t iz = (uint32_t)(u & 127u);
      uint32_t ahz = (hz < 0) ? (uint32_t)(-(int64_t)hz) : (uint32_t)hz;
      if (ahz < kn[iz]) { dst[i] = (float)((double)hz * wn[iz] * std); break; }
      if (iz == 0u) {
        double x, y;
        do {
          x = -log(u64_to_u01(splitmix64_next(rng))) / r;
          y = -log(u64_to_u01(splitmix64_next(rng)));
        } while (y + y < x * x);
        dst[i] = (float)(((hz > 0) ? (r + x) : -(r + x)) * std);
        break;
      }
      double x = (double)hz * wn[iz];
      if (fn[iz] + u64_to_u01(splitmix64_next(rng)) * (fn[iz - 1] - fn[iz]) < exp(-0.5 * x * x)) {
        dst[i] = (float)(x * std);
        break;
      }
    }
  }
}

void bcir_init_weights_c11(float *weights, size_t n_params,
                           const bcir_init_tensor_desc *descs, size_t n_descs,
                           uint32_t n_layers, uint32_t d_model, uint32_t d_base,
                           uint64_t seed, int init_policy,
                           bcir_opt_tensor_shape *out_opt_shapes) {
  memset(weights, 0, n_params * 4u);
  uint64_t rng = seed ^ 0x517CC1B727220A95ULL;
  double resid_scale = 1.0 / sqrt(2.0 * (double)(n_layers > 0u ? n_layers : 1u));
  double mup_lr_scale = (d_base > 0u && d_model > 0u) ? ((double)d_base / (double)d_model) : 1.0;
  for (size_t s = 0; s < n_descs; ++s) {
    const bcir_init_tensor_desc *d = &descs[s];
    size_t len = d->rows * d->cols;
    float *ptr = weights + d->offset;
    double lrm = 1.0; int awd = 1; int is_m2d = 0;
    if (d->role == BCIR_ROLE_NORM) {
      for (size_t i = 0; i < len; ++i) ptr[i] = 1.0f;
      awd = 0;
    } else if (d->role == BCIR_ROLE_ZERO_NOWD) {
      memset(ptr, 0, len * 4u);
      awd = 0;
    } else if (d->role == BCIR_ROLE_CONST_NOWD) {
      for (size_t i = 0; i < len; ++i) ptr[i] = d->const_val;
      awd = 0;
    } else if (d->role == BCIR_ROLE_EMBED) {
      fill_normal_c11(ptr, len, 0.02, &rng);
      awd = 1;
    } else {
      double std = 0.02;
      if (init_policy == BCIR_INIT_RESIDUAL_SCALED || init_policy == BCIR_INIT_MUP_V1) {
        std = 1.0 / sqrt((double)(d->cols > 0u ? d->cols : 1u));
        if (d->role == BCIR_ROLE_LINEAR_OUT) std *= resid_scale;
      }
      fill_normal_c11(ptr, len, std, &rng);
      if (init_policy == BCIR_INIT_MUP_V1) lrm = mup_lr_scale;
      awd = 1;
      is_m2d = (d->rows >= 2u && d->cols >= 2u) ? 1 : 0;
    }
    if (out_opt_shapes) {
      out_opt_shapes[s].offset = d->offset;
      out_opt_shapes[s].rows = d->rows;
      out_opt_shapes[s].cols = d->cols;
      out_opt_shapes[s].lr_mult = lrm;
      out_opt_shapes[s].apply_wd = awd;
      out_opt_shapes[s].is_muon_2d = is_m2d;
    }
  }
}
