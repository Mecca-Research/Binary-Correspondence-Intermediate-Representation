/*===- bcir_dma.h - DMA programs from strided views: admission and the descriptor ring ---------------------===
 *
 * N12 of the native port audit (docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md), the run-time half a driver
 * needs: the C twin of bcir/kbcir/device_manifest.py's `check_strided_view` (D-R4, the strided view the only
 * allocation currency at the driver seam) and of bcir/kbcir/dma.py's `dma_descriptors` (A2: a view pair compiled
 * into a scatter-gather ring). The planning half -- pricing, the transfer module, its events -- stays the
 * oracle's and the law rail's.
 *
 * A view is a bank, a byte offset, an element width, and per dimension an extent and a stride in elements
 * (innermost last). Admission, against the bank's capacity, alignment and native tile, refuses what the oracle
 * refuses and says why as bits, one per message the oracle would write: the early refusals (an empty shape, a
 * stride vector of the wrong length, an extent or stride below 1, an element width below 1, an unknown bank)
 * stand alone; the alignment, the capacity and each of the two innermost extents against the tile accumulate.
 *
 * The ring: each view's contiguous runs in walk order (a unit innermost stride coalesces its last dimension;
 * anything else is per element), paired by a two-pointer byte walk -- each descriptor the smaller remainder of
 * the two current runs -- and a descriptor merged into the one before while both sides stay contiguous. Refused
 * first: either view's admission, unequal shapes or element widths, and overlapping spans on one bank.
 *
 * Freestanding: <stddef.h> and <stdint.h>, no heap, no libc, no recursion (an odometer walks the dimensions).
 * Every product is checked: a view whose arithmetic leaves 64 bits is refused (BCIR_DMA_ERR_OVERFLOW) where the
 * oracle's integers would not wrap.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DMA_H
#define BCIR_DMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_DMA_MAX_DIMS 8u

/* admission refusals: the early ones alone, the rest accumulating */
#define BCIR_DMA_BAD_BANK 0x001u       /* no such bank */
#define BCIR_DMA_BAD_EMPTY 0x002u      /* no dimension */
#define BCIR_DMA_BAD_RANK 0x004u       /* strides not one per dimension */
#define BCIR_DMA_BAD_EXTENT 0x008u     /* an extent or a stride below 1 */
#define BCIR_DMA_BAD_ELEM 0x010u       /* an element width below 1 */
#define BCIR_DMA_BAD_ALIGN 0x020u      /* the offset off the bank's alignment */
#define BCIR_DMA_BAD_CAPACITY 0x040u   /* the last byte past the bank */
#define BCIR_DMA_BAD_TILE_OUTER 0x080u /* the second-innermost extent off the native tile */
#define BCIR_DMA_BAD_TILE_INNER 0x100u /* the innermost extent off the native tile */
#define BCIR_DMA_BAD_OVERFLOW 0x200u   /* arithmetic past 64 bits (C only) */

enum {
  BCIR_DMA_OK = 0,
  BCIR_DMA_ERR_ARG = 1,      /* a NULL, a rank past BCIR_DMA_MAX_DIMS */
  BCIR_DMA_ERR_SRC = 2,      /* the source view refused (its bits in *refusal) */
  BCIR_DMA_ERR_DST = 3,      /* the destination view refused */
  BCIR_DMA_ERR_SHAPE = 4,    /* the shapes differ */
  BCIR_DMA_ERR_WIDTH = 5,    /* the element widths differ */
  BCIR_DMA_ERR_OVERLAP = 6,  /* one bank, overlapping spans */
  BCIR_DMA_ERR_SPACE = 7,    /* more descriptors than the ring holds (*count says how many it needs) */
  BCIR_DMA_ERR_OVERFLOW = 8  /* arithmetic past 64 bits */
};

typedef struct bcir_dma_bank {
  uint64_t capacity;    /* bytes */
  uint32_t native_tile; /* 1: untiled */
  uint32_t align;       /* bytes, >= 1 */
} bcir_dma_bank;

typedef struct bcir_dma_view {
  uint32_t bank;
  uint32_t ndim;
  uint64_t offset;      /* bytes */
  uint64_t elem_bytes;
  uint64_t shape[BCIR_DMA_MAX_DIMS];
  uint64_t strides[BCIR_DMA_MAX_DIMS]; /* elements */
  uint32_t nstrides;    /* must equal ndim */
} bcir_dma_view;

typedef struct bcir_dma_desc {
  uint64_t src;    /* byte offset in the source bank */
  uint64_t dst;    /* byte offset in the destination bank */
  uint64_t nbytes;
} bcir_dma_desc;

/* The view's admission against `banks`: 0, or the refusal bits. */
unsigned bcir_dma_admit(const bcir_dma_view *view, const bcir_dma_bank *banks, uint32_t n_banks);

/* The view pair's ring into out[0 .. capacity): its length in *count. On BCIR_DMA_ERR_SRC/DST the view's bits
 * are in *refusal (may be NULL); on BCIR_DMA_ERR_SPACE *count is the length the ring needs. */
int bcir_dma_descriptors(const bcir_dma_view *src, const bcir_dma_view *dst, const bcir_dma_bank *banks,
                         uint32_t n_banks, bcir_dma_desc *out, size_t capacity, size_t *count, unsigned *refusal);

#ifdef __cplusplus
}
#endif

#endif
