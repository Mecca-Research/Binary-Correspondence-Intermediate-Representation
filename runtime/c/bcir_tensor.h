/* Native float32 tensor kernels. Row-major spans are caller-owned, non-overlapping
 * except the documented in-place vector/RoPE operations. Dimensions and capacities
 * are admitted by bcir_decoder_plan before these hot kernels are called. No heap,
 * global state, fast-math, framework dependency or device discovery; the one host
 * query is the CPU's instruction-set support, which only chooses among bit-identical
 * variants of the same kernels. Providers complete all writes before returning and
 * never throw across the C ABI. */
#ifndef BCIR_TENSOR_H
#define BCIR_TENSOR_H
#include <stddef.h>
#ifndef BCIR_DT_API
#if defined(_WIN32) && defined(BCIR_DT_BUILD_SHARED)
#define BCIR_DT_API __declspec(dllexport)
#else
#define BCIR_DT_API
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef void (*bcir_tensor_mm_fn)(void *ctx, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
typedef void (*bcir_tensor_silu_fn)(void *ctx, size_t n, const float *x, float *y);
typedef struct bcir_tensor_provider {
  void *ctx;
  bcir_tensor_mm_fn mm;       /* NULL selects portable C; same transpose/alpha/beta contract */
  bcir_tensor_silu_fn silu;   /* NULL selects stable libm C; computes x*sigmoid(x), same VJP */
} bcir_tensor_provider;
BCIR_DT_API void bcir_tensor_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c);
/* The portable GEMM's instruction-set variants: bit 1 portable C, bit 2 AVX2, bit 4 AVX-512F
 * (x86-64 Linux with GCC or Clang). Every variant computes the same rounded operations in the
 * same order, so all of them are bit-identical; bcir_tensor_mm runs the widest one the host
 * supports. bcir_tensor_mm_variant runs one variant and returns it, or returns 0 and writes
 * nothing when the host cannot run it. */
BCIR_DT_API int bcir_tensor_mm_variants(void);
BCIR_DT_API int bcir_tensor_mm_variant(int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
/* Optional LP64 CBLAS adapter: ctx points to a caller-owned function pointer.
 * No BLAS headers or link dependency. The adapter uses portable C on shapes beyond
 * the LP64 provider's INT_MAX dimensions. */
typedef void (*bcir_cblas_sgemm_fn)(int, int, int, int, int, int, float,
    const float *, int, const float *, int, float, float *, int);
BCIR_DT_API void bcir_tensor_cblas_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c);
/* y = x W^T, W is out x in; dx overwritten or accumulated, dW accumulated. */
BCIR_DT_API void bcir_tensor_linear(const bcir_tensor_provider *p, size_t rows, size_t in,
    size_t out, const float *x, const float *w, float *y);
BCIR_DT_API void bcir_tensor_linear_backward(const bcir_tensor_provider *p, size_t rows,
    size_t in, size_t out, const float *x, const float *w, const float *dy,
    float *dx, float *dw, float dx_beta);
BCIR_DT_API void bcir_tensor_silu(void *ctx, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_swiglu_backward(size_t n, const float *gate, const float *up,
    const float *dy, float *dg, float *du);
BCIR_DT_API void bcir_tensor_rms(size_t rows, size_t width, float eps, const float *x,
    const float *g, float *y);
BCIR_DT_API void bcir_tensor_rms_backward(size_t rows, size_t width, float eps, const float *x,
    const float *g, const float *dy, float *dx, float *dg);
/* Half-split Llama RoPE, [batch,time,head,channel]; inverse applies its transpose. */
BCIR_DT_API void bcir_tensor_rope(size_t batch, size_t time, size_t heads, size_t dim,
    double base, float *x, int inverse);
/* p: [batch,query-head,time,time]; causal future entries are exactly zero.
 * q/k/v: [batch,time,head,channel]. Backward sums all GQA groups into dk/dv. */
BCIR_DT_API void bcir_tensor_attention(size_t batch, size_t time, size_t heads, size_t kvheads,
    size_t dim, const float *q, const float *k, const float *v, float *p, float *y);
BCIR_DT_API void bcir_tensor_attention_backward(size_t batch, size_t time, size_t heads,
    size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    const float *p, const float *dy, float *dq, float *dk, float *dv);
/* The same dq, dk and dv, bit for bit, through the GEMM kernel; p is rewritten in place with
 * the score adjoint (p_ij (dp_ij - dot_i) / sqrt(dim)), so it no longer holds probabilities. */
BCIR_DT_API void bcir_tensor_attention_backward_inplace(size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    float *p, const float *dy, float *dq, float *dk, float *dv);
#ifdef __cplusplus
}
#endif
#endif
