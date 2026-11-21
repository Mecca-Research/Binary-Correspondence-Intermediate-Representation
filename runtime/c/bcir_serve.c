/*===- bcir_serve.c - the serving loop's control plane: the paged KV page table and the token schema -------===
 *
 * See bcir_serve.h. The checks run in the oracle's order (PagedKV.__init__, step_row, page_of, evict;
 * check_token_dfa), so an input wrong in several ways is refused for the same first reason on both rails.
 *===----------------------------------------------------------------------===*/
#include "bcir_serve.h"

/* a * b, or 0 with *bad set when it leaves 64 bits */
static uint64_t mul64(uint64_t a, uint64_t b, int *bad) {
  if (b && a > UINT64_MAX / b) {
    *bad = 1;
    return 0;
  }
  return a * b;
}

static void page_view(const bcir_kv_table *t, uint64_t offset, bcir_dma_view *v) {
  uint8_t *z = (uint8_t *)v;
  for (size_t k = 0; k < sizeof *v; k++) z[k] = 0;
  v->bank = t->bank;
  v->offset = offset;
  v->elem_bytes = 4;
  v->ndim = v->nstrides = 2;
  v->shape[0] = t->page_size;
  v->shape[1] = t->row_elems;
  v->strides[0] = t->row_elems;
  v->strides[1] = 1;
}

int bcir_kv_init(bcir_kv_table *t, uint32_t n_layers, uint32_t kv_dim, uint32_t capacity, uint32_t page_size,
                 const bcir_dma_bank *banks, uint32_t n_banks, uint32_t bank, uint32_t base_rid,
                 bcir_kv_page *pages, size_t max_pages, uint32_t *bad_page, unsigned *bad_bits) {
  if (bad_page) *bad_page = 0;
  if (bad_bits) *bad_bits = 0;
  if (!t || !banks || (max_pages && !pages)) return BCIR_KV_ERR_ARG;
  if (capacity < 1u || capacity > BCIR_KV_MAX_CONTEXT) return BCIR_KV_ERR_CAPACITY;
  if (page_size < 1u || page_size > capacity) return BCIR_KV_ERR_PAGE_SIZE;
  if (base_rid < BCIR_KV_MIN_RID) return BCIR_KV_ERR_RID;
  if (bank >= n_banks) return BCIR_KV_ERR_BANK;
  uint32_t n_pages = capacity / page_size + (capacity % page_size != 0u);
  if (n_pages > BCIR_KV_MAX_PAGES) return BCIR_KV_ERR_PAGES;
  if (base_rid > UINT32_MAX - (n_pages - 1u)) return BCIR_KV_ERR_RID_RANGE;
  if (max_pages < n_pages) {
    if (bad_page) *bad_page = n_pages;
    return BCIR_KV_ERR_SPACE;
  }
  /* one position: a K row and a V row per layer per kv head */
  uint64_t elems = (uint64_t)n_layers * kv_dim;
  bcir_kv_table nt;
  nt.capacity = capacity;
  nt.page_size = page_size;
  nt.n_pages = n_pages;
  nt.pos = 0;
  nt.bank = bank;
  nt.pages = pages;
  int bad = elems > UINT64_MAX / 2u;
  nt.row_elems = bad ? 0 : 2u * (elems ? elems : 1u);
  nt.page_bytes = mul64(mul64(page_size, nt.row_elems, &bad), 4u, &bad);
  for (uint32_t p = 0; p < n_pages; p++) {
    uint64_t offset = mul64(p, nt.page_bytes, &bad);
    unsigned bits = BCIR_DMA_BAD_OVERFLOW;
    if (!bad) {
      bcir_dma_view v;
      page_view(&nt, offset, &v);
      bits = bcir_dma_admit(&v, banks, n_banks);
    }
    if (bits) { /* D-R4, live: no view, no memory */
      if (bad_page) *bad_page = p;
      if (bad_bits) *bad_bits = bits;
      return BCIR_KV_ERR_VIEW;
    }
  }
  /* committed only after every page has passed */
  for (uint32_t p = 0; p < n_pages; p++) {
    pages[p].rid = base_rid + p;
    pages[p].map_gen = 0;
    pages[p].data_gen = 0;
    pages[p].live = 1;
  }
  *t = nt;
  return BCIR_KV_OK;
}

int bcir_kv_view(const bcir_kv_table *t, uint32_t p, bcir_dma_view *view) {
  if (!t || !t->pages || !view) return BCIR_KV_ERR_ARG;
  if (p >= t->n_pages) return BCIR_KV_ERR_NO_PAGE;
  page_view(t, (uint64_t)p * t->page_bytes, view); /* admitted at init: no wrap */
  return BCIR_KV_OK;
}

int bcir_kv_page_of(const bcir_kv_table *t, uint64_t pos, uint32_t *page) {
  if (!t || !t->pages || !page) return BCIR_KV_ERR_ARG;
  if (pos >= t->capacity) return BCIR_KV_ERR_POS;
  *page = (uint32_t)(pos / t->page_size);
  return BCIR_KV_OK;
}

int bcir_kv_append(bcir_kv_table *t) {
  if (!t || !t->pages) return BCIR_KV_ERR_ARG;
  if (t->pos >= t->capacity) return BCIR_KV_ERR_FULL;
  bcir_kv_page *pg = &t->pages[t->pos / t->page_size];
  if (!pg->live) return BCIR_KV_ERR_FREED;
  pg->data_gen++;
  t->pos++;
  return BCIR_KV_OK;
}

int bcir_kv_evict(bcir_kv_table *t, uint32_t p, bcir_dma_view *freed) {
  if (!t || !t->pages) return BCIR_KV_ERR_ARG;
  if (p >= t->n_pages) return BCIR_KV_ERR_NO_PAGE;
  if (t->pos < t->capacity) return BCIR_KV_ERR_LIVE;
  bcir_kv_page *pg = &t->pages[p];
  if (!pg->live) return BCIR_KV_ERR_EVICTED;
  pg->live = 0;
  pg->map_gen++;
  if (freed) page_view(t, (uint64_t)p * t->page_bytes, freed);
  return BCIR_KV_OK;
}

unsigned bcir_dfa_check(const bcir_token_dfa *d, uint32_t vocab_size) {
  if (!d) return BCIR_DFA_BAD_LAYOUT;
  if (vocab_size < 1u) return BCIR_DFA_BAD_VOCAB;
  uint32_t n = d->n_states;
  if (n < 1u) return BCIR_DFA_BAD_EMPTY;
  if (n > BCIR_DFA_MAX_STATES) return BCIR_DFA_BAD_STATES;
  /* the layout first: offsets from 0, never falling, and arrays behind every entry */
  if (!d->offsets || d->offsets[0] != 0u) return BCIR_DFA_BAD_LAYOUT;
  for (uint32_t s = 0; s < n; s++)
    if (d->offsets[s + 1u] < d->offsets[s]) return BCIR_DFA_BAD_LAYOUT;
  if (d->offsets[n] && (!d->tokens || !d->next)) return BCIR_DFA_BAD_LAYOUT;
  unsigned bits = 0;
  if (d->start >= n) bits |= BCIR_DFA_BAD_START;
  uint64_t transitions = 0;
  for (uint32_t s = 0; s < n; s++) {
    uint32_t lo = d->offsets[s], hi = d->offsets[s + 1u];
    /* the oracle counts a state's edges -- its distinct tokens -- before it reads the state */
    for (uint32_t e = lo; e < hi; e++)
      if (e == lo || d->tokens[e] != d->tokens[e - 1u]) transitions++;
    if (transitions > BCIR_DFA_MAX_TRANSITIONS) {
      bits |= BCIR_DFA_BAD_TRANSITIONS;
      break;
    }
    for (uint32_t e = lo; e < hi; e++) {
      uint32_t tok = d->tokens[e];
      if (e > lo && tok < d->tokens[e - 1u]) {
        bits |= BCIR_DFA_BAD_ORDER;
        continue;
      }
      if (e > lo && tok == d->tokens[e - 1u]) { /* the edge map keeps the first entry */
        if (tok < vocab_size) bits |= BCIR_DFA_BAD_DUPLICATE;
        continue;
      }
      if (tok >= vocab_size) bits |= BCIR_DFA_BAD_TOKEN;
      if (d->next[e] >= n) bits |= BCIR_DFA_BAD_TARGET;
    }
  }
  return bits;
}

int bcir_dfa_step(const bcir_token_dfa *d, uint32_t state, const float *logits, uint32_t vocab, uint32_t *token,
                  uint32_t *next) {
  if (!d || !d->offsets || !logits || !token || !next || state >= d->n_states) return BCIR_DFA_ERR_ARG;
  uint32_t lo = d->offsets[state], hi = d->offsets[state + 1u];
  if (lo == hi) return BCIR_DFA_ERR_DEADLOCK; /* refuse rather than emit off the schema */
  uint32_t best = lo;
  for (uint32_t e = lo; e < hi; e++) {
    if (d->tokens[e] >= vocab) return BCIR_DFA_ERR_ARG;
    /* ascending, a strict >: ties keep the lowest id, and a NaN displaces nothing */
    if (logits[d->tokens[e]] > logits[d->tokens[best]]) best = e;
  }
  *token = d->tokens[best];
  *next = d->next[best];
  return BCIR_DFA_OK;
}
