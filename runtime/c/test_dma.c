/*===- test_dma.c - DMA programs from strided views: contracts, and a driver for the oracle's comparison ----===
 *
 *   test_dma          the contracts the oracle cannot state: a ring past its capacity says the length it needs and
 *                     writes no further, arithmetic past 64 bits is refused, every NULL, a rank past the limit, a
 *                     dense copy one descriptor, a scatter one per element. Prints `test_dma: OK`.
 *   test_dma FILE     one request per line, as bcir/tests/test_c_dma.py writes them:
 *                       `bank CAP TILE ALIGN`   appends a bank (no output); `reset` clears them
 *                       `admit V`               -> the refusal bits, 0 when admitted
 *                       `dma V V`               -> `ok N S:D:B ...`, or `ERR CODE BITS`
 *                     V is `BANK OFF ELEM NDIM S... NSTR T...`, decimal.
 *===----------------------------------------------------------------------===*/
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_dma.h"

static int failures;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; }          \
  } while (0)

static bcir_dma_view view2(uint32_t bank, uint64_t off, uint64_t d0, uint64_t d1, uint64_t s0, uint64_t s1) {
  bcir_dma_view v;
  memset(&v, 0, sizeof v);
  v.bank = bank;
  v.offset = off;
  v.elem_bytes = 4;
  v.ndim = v.nstrides = 2;
  v.shape[0] = d0;
  v.shape[1] = d1;
  v.strides[0] = s0;
  v.strides[1] = s1;
  return v;
}

static void contracts(void) {
  const bcir_dma_bank banks[2] = {{1u << 20, 1, 64}, {1u << 20, 16, 64}};
  bcir_dma_desc ring[64];
  size_t n;
  unsigned bits;
  bcir_dma_view a = view2(0, 0, 16, 32, 32, 1), b = view2(1, 0, 16, 32, 32, 1);
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_OK && n == 1);
  CHECK(ring[0].src == 0 && ring[0].dst == 0 && ring[0].nbytes == 16u * 32u * 4u);
  /* gapped source rows: one descriptor a row */
  a = view2(0, 0, 16, 32, 64, 1);
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_OK && n == 16);
  CHECK(ring[1].src == 64u * 4u && ring[1].dst == 32u * 4u && ring[1].nbytes == 32u * 4u);
  /* a ring too short: the length it needs, nothing written past it */
  memset(ring, 0xEE, sizeof ring);
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 5, &n, &bits) == BCIR_DMA_ERR_SPACE && n == 16);
  CHECK(ring[5].nbytes == 0xEEEEEEEEEEEEEEEEu && ring[4].nbytes == 32u * 4u);
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, NULL, 0, &n, &bits) == BCIR_DMA_ERR_SPACE && n == 16);
  /* a non-unit innermost stride: per element */
  a = view2(0, 0, 16, 16, 16, 2);
  b = view2(1, 4096, 16, 16, 16, 1);
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_ERR_SPACE && n == 256);
  /* refusals, in the oracle's order */
  a = view2(0, 4, 16, 32, 32, 1); /* off the alignment */
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_ERR_SRC && bits == BCIR_DMA_BAD_ALIGN);
  a = view2(1, 0, 15, 15, 15, 1); /* a 15 x 15 view on a 16-tiled bank */
  CHECK(bcir_dma_admit(&a, banks, 2) == (BCIR_DMA_BAD_TILE_OUTER | BCIR_DMA_BAD_TILE_INNER));
  a = view2(0, 0, 16, 32, 32, 1);
  b = view2(0, 1024, 16, 32, 32, 1); /* one bank, overlapping */
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_ERR_OVERLAP);
  b = view2(1, 0, 16, 16, 16, 1);
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_ERR_SHAPE);
  b = view2(1, 0, 16, 32, 32, 1);
  b.elem_bytes = 2;
  CHECK(bcir_dma_descriptors(&a, &b, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_ERR_WIDTH);
  /* arithmetic past 64 bits */
  const bcir_dma_bank huge[1] = {{UINT64_MAX, 1, 1}};
  a = view2(0, 0, 1u << 31, 1u << 31, UINT64_MAX / 4, 1);
  CHECK(bcir_dma_admit(&a, huge, 1) & BCIR_DMA_BAD_OVERFLOW);
  b = a;
  CHECK(bcir_dma_descriptors(&a, &b, huge, 1, ring, 64, &n, &bits) == BCIR_DMA_ERR_OVERFLOW);
  a = view2(0, UINT64_MAX - 8u, 4, 4, 4, 1); /* the offset itself carries the end past 64 bits */
  CHECK(bcir_dma_admit(&a, huge, 1) & BCIR_DMA_BAD_OVERFLOW);
  /* a rank past the limit, a stride vector of another length, no bank, every NULL */
  a = view2(0, 0, 16, 32, 32, 1);
  a.ndim = a.nstrides = BCIR_DMA_MAX_DIMS + 1u;
  CHECK(bcir_dma_admit(&a, banks, 2) == BCIR_DMA_BAD_RANK);
  a = view2(0, 0, 16, 32, 32, 1);
  a.nstrides = 1;
  CHECK(bcir_dma_admit(&a, banks, 2) == BCIR_DMA_BAD_RANK);
  a.bank = 2;
  CHECK(bcir_dma_admit(&a, banks, 2) == BCIR_DMA_BAD_BANK);
  CHECK(bcir_dma_admit(NULL, banks, 2) == BCIR_DMA_BAD_BANK);
  a = view2(0, 0, 16, 32, 32, 1);
  CHECK(bcir_dma_descriptors(NULL, &a, banks, 2, ring, 64, &n, &bits) == BCIR_DMA_ERR_ARG);
  CHECK(bcir_dma_descriptors(&a, &a, banks, 2, ring, 64, NULL, &bits) == BCIR_DMA_ERR_ARG);
  CHECK(bcir_dma_descriptors(&a, &a, banks, 2, NULL, 4, &n, &bits) == BCIR_DMA_ERR_ARG);
}

static bcir_dma_bank banks[16];
static uint32_t n_banks;

/* one view from the text at *p; 0 when malformed */
static int read_view(char **p, bcir_dma_view *v) {
  char *end;
  memset(v, 0, sizeof *v);
  uint64_t f[4];
  for (int k = 0; k < 4; k++) {
    f[k] = strtoull(*p, &end, 10);
    if (end == *p) return 0;
    *p = end;
  }
  v->bank = (uint32_t)f[0];
  v->offset = f[1];
  v->elem_bytes = f[2];
  if (f[3] > BCIR_DMA_MAX_DIMS) return 0;
  v->ndim = (uint32_t)f[3];
  for (uint32_t k = 0; k < v->ndim; k++) {
    v->shape[k] = strtoull(*p, &end, 10);
    if (end == *p) return 0;
    *p = end;
  }
  uint64_t ns = strtoull(*p, &end, 10);
  if (end == *p || ns > BCIR_DMA_MAX_DIMS) return 0;
  *p = end;
  v->nstrides = (uint32_t)ns;
  for (uint32_t k = 0; k < v->nstrides; k++) {
    v->strides[k] = strtoull(*p, &end, 10);
    if (end == *p) return 0;
    *p = end;
  }
  return 1;
}

static int drive(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("TOOL cannot read the file\n"); return 2; }
  static char line[4096];
  static bcir_dma_desc ring[4096];
  int rc = 0;
  while (fgets(line, sizeof line, f)) {
    char *p = line;
    bcir_dma_view a, b;
    if (!strncmp(line, "bank ", 5) && n_banks < 16) {
      unsigned long long cap, tile, align;
      if (sscanf(line + 5, "%llu %llu %llu", &cap, &tile, &align) != 3) { rc = 2; break; }
      banks[n_banks].capacity = cap;
      banks[n_banks].native_tile = (uint32_t)tile;
      banks[n_banks].align = (uint32_t)align;
      n_banks++;
    } else if (!strcmp(line, "reset\n")) {
      n_banks = 0;
    } else if (!strncmp(line, "admit ", 6) && (p = line + 6, read_view(&p, &a))) {
      printf("%u\n", bcir_dma_admit(&a, banks, n_banks));
    } else if (!strncmp(line, "dma ", 4) && (p = line + 4, read_view(&p, &a)) && read_view(&p, &b)) {
      size_t n = 0;
      unsigned bits = 0;
      int r = bcir_dma_descriptors(&a, &b, banks, n_banks, ring, sizeof ring / sizeof ring[0], &n, &bits);
      if (r) {
        printf("ERR %d %u\n", r, bits);
        continue;
      }
      printf("ok %llu", (unsigned long long)n);
      for (size_t k = 0; k < n; k++)
        printf(" %" PRIu64 ":%" PRIu64 ":%" PRIu64, ring[k].src, ring[k].dst, ring[k].nbytes);
      printf("\n");
    } else {
      rc = 2;
      break;
    }
  }
  if (rc) printf("TOOL malformed line\n");
  fclose(f);
  return rc;
}

int main(int argc, char **argv) {
  if (argc == 2) return drive(argv[1]);
  contracts();
  if (failures) { printf("test_dma: %d FAILED\n", failures); return 1; }
  printf("test_dma: OK\n");
  return 0;
}
