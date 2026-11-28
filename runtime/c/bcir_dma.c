/*===- bcir_dma.c - DMA programs from strided views ------------------------------------------------------===
 *
 * See bcir_dma.h. The checks run in the oracle's order, so a view wrong in several ways is refused for the same
 * reasons on both rails.
 *===----------------------------------------------------------------------===*/
#include "bcir_dma.h"

/* a * b + c, or 0 with *bad set when it leaves 64 bits */
static uint64_t mul_add(uint64_t a, uint64_t b, uint64_t c, int *bad) {
  if (b && a > (UINT64_MAX - c) / b) {
    *bad = 1;
    return 0;
  }
  return a * b + c;
}

/* the view's last byte, one past: offset + sum((d - 1) * s) * elem + elem */
static uint64_t view_end(const bcir_dma_view *v, int *bad) {
  uint64_t elems = 0;
  for (uint32_t k = 0; k < v->ndim; k++) elems = mul_add(v->shape[k] - 1u, v->strides[k], elems, bad);
  uint64_t bytes = mul_add(elems, v->elem_bytes, v->elem_bytes, bad);
  if (bytes > UINT64_MAX - v->offset) *bad = 1;
  return *bad ? 0 : v->offset + bytes;
}

unsigned bcir_dma_admit(const bcir_dma_view *v, const bcir_dma_bank *banks, uint32_t n_banks) {
  if (!v || !banks || v->bank >= n_banks) return BCIR_DMA_BAD_BANK;
  const bcir_dma_bank *b = &banks[v->bank];
  if (v->ndim == 0u) return BCIR_DMA_BAD_EMPTY;
  if (v->nstrides != v->ndim || v->ndim > BCIR_DMA_MAX_DIMS) return BCIR_DMA_BAD_RANK;
  for (uint32_t k = 0; k < v->ndim; k++)
    if (v->shape[k] < 1u || v->strides[k] < 1u) return BCIR_DMA_BAD_EXTENT;
  if (v->elem_bytes < 1u) return BCIR_DMA_BAD_ELEM;
  unsigned bits = 0;
  if (b->align == 0u || v->offset % b->align) bits |= BCIR_DMA_BAD_ALIGN;
  int bad = 0;
  uint64_t end = view_end(v, &bad);
  if (bad) return bits | BCIR_DMA_BAD_OVERFLOW;
  if (end > b->capacity) bits |= BCIR_DMA_BAD_CAPACITY;
  if (b->native_tile > 1u) {
    if (v->ndim >= 2u && v->shape[v->ndim - 2u] % b->native_tile) bits |= BCIR_DMA_BAD_TILE_OUTER;
    if (v->shape[v->ndim - 1u] % b->native_tile) bits |= BCIR_DMA_BAD_TILE_INNER;
  }
  return bits;
}

/* A view's runs in walk order: an odometer over the outer dimensions (all of them when the innermost stride is
 * not 1), each position one run of `run` bytes. */
typedef struct runs {
  const bcir_dma_view *v;
  uint32_t outer;                  /* the dimensions the odometer walks */
  uint64_t index[BCIR_DMA_MAX_DIMS];
  uint64_t run;                    /* bytes a run holds */
  uint64_t at;                     /* the current run's first byte */
  int done;
} runs;

static void runs_start(runs *r, const bcir_dma_view *v) {
  int unit = v->strides[v->ndim - 1u] == 1u;
  r->v = v;
  r->outer = unit ? v->ndim - 1u : v->ndim;
  r->run = (unit ? v->shape[v->ndim - 1u] : 1u) * v->elem_bytes; /* admitted: no wrap (the end fits) */
  for (uint32_t k = 0; k < BCIR_DMA_MAX_DIMS; k++) r->index[k] = 0;
  r->at = v->offset;
  r->done = 0;
}

/* the next run: the odometer's last dimension first */
static void runs_next(runs *r) {
  const bcir_dma_view *v = r->v;
  for (uint32_t k = r->outer; k-- > 0;) {
    if (++r->index[k] < v->shape[k]) {
      uint64_t elems = 0;
      for (uint32_t j = 0; j < r->outer; j++) elems += r->index[j] * v->strides[j];
      r->at = v->offset + elems * v->elem_bytes;
      return;
    }
    r->index[k] = 0;
  }
  r->done = 1;
}

int bcir_dma_descriptors(const bcir_dma_view *src, const bcir_dma_view *dst, const bcir_dma_bank *banks,
                         uint32_t n_banks, bcir_dma_desc *out, size_t capacity, size_t *count, unsigned *refusal) {
  if (!src || !dst || !banks || !count || (capacity && !out)) return BCIR_DMA_ERR_ARG;
  *count = 0;
  if (refusal) *refusal = 0;
  unsigned bits = bcir_dma_admit(src, banks, n_banks);
  if (bits) {
    if (refusal) *refusal = bits;
    return bits & BCIR_DMA_BAD_OVERFLOW ? BCIR_DMA_ERR_OVERFLOW : BCIR_DMA_ERR_SRC;
  }
  bits = bcir_dma_admit(dst, banks, n_banks);
  if (bits) {
    if (refusal) *refusal = bits;
    return bits & BCIR_DMA_BAD_OVERFLOW ? BCIR_DMA_ERR_OVERFLOW : BCIR_DMA_ERR_DST;
  }
  if (src->ndim != dst->ndim) return BCIR_DMA_ERR_SHAPE;
  for (uint32_t k = 0; k < src->ndim; k++)
    if (src->shape[k] != dst->shape[k]) return BCIR_DMA_ERR_SHAPE;
  if (src->elem_bytes != dst->elem_bytes) return BCIR_DMA_ERR_WIDTH;
  if (src->bank == dst->bank) {
    int bad = 0;
    uint64_t a1 = view_end(src, &bad), b1 = view_end(dst, &bad);
    if (src->offset < b1 && dst->offset < a1) return BCIR_DMA_ERR_OVERLAP;
  }
  runs s, d;
  runs_start(&s, src);
  runs_start(&d, dst);
  uint64_t si = 0, dj = 0; /* bytes consumed within the current runs */
  size_t n = 0;
  bcir_dma_desc last = {0, 0, 0}; /* the ring's last descriptor, kept past its capacity so *count stays exact */
  while (!s.done && !d.done) {
    uint64_t take = s.run - si < d.run - dj ? s.run - si : d.run - dj;
    uint64_t so = s.at + si, dov = d.at + dj;
    if (n && last.src + last.nbytes == so && last.dst + last.nbytes == dov) {
      last.nbytes += take;
    } else {
      last.src = so;
      last.dst = dov;
      last.nbytes = take;
      n++;
    }
    if (n <= capacity) out[n - 1] = last;
    si += take;
    dj += take;
    if (si == s.run) { runs_next(&s); si = 0; }
    if (dj == d.run) { runs_next(&d); dj = 0; }
  }
  *count = n;
  /* equal shapes and widths drain both sides together; a remainder would be a partial program */
  if (!s.done || !d.done || si || dj) return BCIR_DMA_ERR_ARG;
  return n <= capacity ? BCIR_DMA_OK : BCIR_DMA_ERR_SPACE;
}
