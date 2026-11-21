/*===- fuzz_devman.c - libFuzzer target: a hostile device-manifest envelope, and what binds from it -----------===
 *
 * The input is the envelope. A tampered one -- its digest well formed but not re-derived, which a fuzzer reaches
 * often -- is tried again with the first run of 64 lowercase hex digits replaced by the digest its contents
 * re-derive, which reaches the consistency laws and the binding. Whatever loads must be what the laws admit: a
 * named device, banks of positive capacity, tile and power-of-two alignment, a square zero-diagonal symmetric
 * matrix; it must bind, agree with a probe of itself, and admit a one-element view at each bank's start. A refused
 * envelope must leave the caller's manifest untouched. ASan/UBSan catch any read past the envelope or the
 * workspace.
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_devman.h"

static uint8_t work[(1u << 22) + 64u];

static int is_hex(uint8_t c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }

static int load(const uint8_t *data, size_t size, bcir_devman *m) {
  size_t need = bcir_devman_workspace_size(size);
  if (!need || need > sizeof work) return -1;
  bcir_devman keep;
  memset(&keep, 0x5A, sizeof keep);
  *m = keep;
  uint32_t where;
  unsigned bits;
  bcir_jer_diag diag;
  int rc = bcir_devman_load(data, size, NULL, 0, 0, 0, work, need, m, &where, &bits, &diag);
  if (rc != BCIR_DEVMAN_OK && rc != BCIR_DEVMAN_ERR_TAMPERED && memcmp(m, &keep, sizeof keep) != 0) abort();
  if (rc == BCIR_DEVMAN_ERR_INVALID && bits == 0) abort();
  return rc;
}

static void admitted(const bcir_devman *m) {
  if (m->device_len == 0 || m->n_banks == 0 || m->n_rows != m->n_banks) abort();
  for (uint32_t i = 0; i < m->n_banks; i++) {
    const bcir_devman_bank *b = &m->banks[i];
    if (b->capacity < 1 || b->native_tile < 1 || b->align < 1 || (b->align & (b->align - 1)) ||
        b->domain >= BCIR_DEVMAN_DOMAINS || b->name_len == 0)
      abort();
    if (m->row_off[i + 1u] - m->row_off[i] != m->n_banks) abort();
    for (uint32_t j = 0; j < m->n_banks; j++) {
      int64_t d = m->distance[(size_t)i * m->n_banks + j], e = m->distance[(size_t)j * m->n_banks + i];
      if ((i == j && d != 0) || (i != j && d < 1) || d != e) abort();
    }
  }
  static bcir_dma_bank table[256];
  static bcir_devman_probe self[256];
  if (bcir_devman_dma_banks(m, table, 256) != BCIR_DEVMAN_OK) return; /* a tile or alignment past 32 bits */
  for (uint32_t i = 0; i < m->n_banks; i++) {
    self[i].name = m->banks[i].name;
    self[i].name_len = m->banks[i].name_len;
    self[i].has_capacity = self[i].has_tile = 1;
    self[i].capacity = (uint64_t)m->banks[i].capacity;
    self[i].native_tile = (uint64_t)m->banks[i].native_tile;
    bcir_dma_view v;
    memset(&v, 0, sizeof v);
    v.bank = i;
    v.elem_bytes = 1;
    v.ndim = v.nstrides = 1;
    v.shape[0] = 1;
    v.strides[0] = 1;
    if (table[i].native_tile == 1u && bcir_dma_admit(&v, table, m->n_banks) != 0) abort();
  }
  size_t messages = 1;
  if (bcir_devman_probe_agree(m, self, m->n_banks, &messages) != 0 || messages != 0) abort();
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  bcir_devman m;
  int rc = load(data, size, &m);
  if (rc == BCIR_DEVMAN_OK) admitted(&m);
  if (rc != BCIR_DEVMAN_ERR_TAMPERED || size > (1u << 16)) return 0;
  /* re-seal: the first run of 64 lowercase hex digits becomes the digest the contents re-derive */
  static uint8_t copy[1u << 16];
  memcpy(copy, data, size);
  static const char nib[] = "0123456789abcdef";
  for (size_t k = 0; k + 64u <= size; k++) {
    size_t run = 0;
    while (run < 64u && is_hex(copy[k + run])) run++;
    if (run < 64u) continue;
    for (unsigned j = 0; j < 32u; j++) {
      copy[k + 2u * j] = (uint8_t)nib[m.digest[j] >> 4];
      copy[k + 2u * j + 1u] = (uint8_t)nib[m.digest[j] & 15u];
    }
    if (load(copy, size, &m) == BCIR_DEVMAN_OK) admitted(&m);
    break;
  }
  return 0;
}
