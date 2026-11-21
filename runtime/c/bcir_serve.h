/*===- bcir_serve.h - the serving loop's control plane: the paged KV page table and the token schema -------===
 *
 * N10 of the native port audit (docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md), the part of serving a native
 * server runs beside the decoder's numerics, which stay bcir_decode.c's and bcir_llama.c's:
 *
 *   - The page table of bcir/frontends/models/paged_kv.py's PagedKV. Construction is allocation: each page is a
 *     strided view, (page_size, row_elems) f32 elements at page * page_bytes, admitted against its bank by
 *     bcir_dma_admit (D-R4) before any token flows, the first page refused naming the page and the admission's
 *     bits. A row written past the capacity is refused (gem.kv_cache's law, pos <= capacity); each row written
 *     bumps the data_gen of the page that owns it (R11's currency); eviction, refused while the session is live
 *     (attention is full-context, so every page is still read), bumps the page's map_gen and frees its view.
 *     bcir_kv_append is the accounting of one row, called after the decoder has written it.
 *   - The output schema of bcir/frontends/models/serve.py's TokenDFA: a token-level DFA checked as
 *     check_token_dfa checks it, refusals as bits, one per kind of message the oracle writes; and one
 *     constrained step, _masked_argmax and the state's edge -- the admitted token whose logit is greatest, ties
 *     to the lowest id, a NaN never displacing the token before it -- with a state that admits no token refused
 *     as a deadlock rather than an id emitted off the schema.
 *
 * The DFA's one spelling: per state its entries (token, successor) in a run of `tokens` and `next`, the runs
 * delimited by `offsets` (n_states + 1, from 0, never falling), each run's tokens ascending. A token repeated in
 * a run is the oracle's duplicate (its edge map keeps the first entry, as this reader does); a run that falls is
 * refused (BCIR_DFA_BAD_ORDER), and so are offsets that do not delimit runs (BCIR_DFA_BAD_LAYOUT), neither of
 * which the oracle's tuples can spell.
 *
 * Freestanding: <stddef.h> and <stdint.h> (and bcir_dma.h), no heap, no libc, no recursion. The caller owns every
 * array; a page table's page records are the caller's, n_pages of them.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_SERVE_H
#define BCIR_SERVE_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_dma.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_KV_MAX_CONTEXT (1u << 20) /* decode._MAX_REFERENCE_CONTEXT */
#define BCIR_KV_MAX_PAGES (1u << 16)   /* paged_kv._MAX_PAGES */
#define BCIR_KV_MIN_RID 5u             /* RIDs below 5 are the registry's own */

enum {
  BCIR_KV_OK = 0,
  BCIR_KV_ERR_ARG = 1,       /* a NULL (C only) */
  BCIR_KV_ERR_CAPACITY = 2,  /* capacity outside [1, BCIR_KV_MAX_CONTEXT] */
  BCIR_KV_ERR_PAGE_SIZE = 3, /* page_size outside [1, capacity] */
  BCIR_KV_ERR_RID = 4,       /* base_rid below BCIR_KV_MIN_RID */
  BCIR_KV_ERR_BANK = 5,      /* no such bank */
  BCIR_KV_ERR_PAGES = 6,     /* more pages than BCIR_KV_MAX_PAGES */
  BCIR_KV_ERR_RID_RANGE = 7, /* the page RIDs run past 2**32 - 1 */
  BCIR_KV_ERR_SPACE = 8,     /* fewer page records than pages (C only; *bad_page says how many it needs) */
  BCIR_KV_ERR_VIEW = 9,      /* a page's view refused: the page in *bad_page, bcir_dma_admit's bits in *bad_bits */
  BCIR_KV_ERR_FULL = 10,     /* a row past the capacity: the paging lie */
  BCIR_KV_ERR_FREED = 11,    /* a row into an evicted page: the use-after-free shape */
  BCIR_KV_ERR_POS = 12,      /* a position outside [0, capacity) */
  BCIR_KV_ERR_NO_PAGE = 13,  /* no such page */
  BCIR_KV_ERR_LIVE = 14,     /* eviction before the session completes: a numerics lie */
  BCIR_KV_ERR_EVICTED = 15   /* a page evicted twice: the double-free shape */
};

typedef struct bcir_kv_page {
  uint32_t rid;
  uint32_t map_gen;  /* bumped by the page's eviction */
  uint32_t data_gen; /* bumped by each row written into the page */
  uint32_t live;     /* 1 while the page holds its view, 0 once evicted */
} bcir_kv_page;

typedef struct bcir_kv_table {
  uint32_t capacity, page_size, n_pages, pos; /* pos: the rows written */
  uint32_t bank;
  uint64_t row_elems;  /* f32 elements a position holds: 2 * max(1, n_layers * kv_dim) */
  uint64_t page_bytes; /* page_size * row_elems * 4 */
  bcir_kv_page *pages;
} bcir_kv_table;

/* Allocate the page table: every check in the oracle's order, then each page's view admitted in turn; *t and
 * pages[0 .. n_pages) are written only when every page is admitted. *bad_page and *bad_bits (either may be NULL)
 * say which page was refused and why; on BCIR_KV_ERR_SPACE *bad_page is the page count the table needs. */
int bcir_kv_init(bcir_kv_table *t, uint32_t n_layers, uint32_t kv_dim, uint32_t capacity, uint32_t page_size,
                 const bcir_dma_bank *banks, uint32_t n_banks, uint32_t bank, uint32_t base_rid,
                 bcir_kv_page *pages, size_t max_pages, uint32_t *bad_page, unsigned *bad_bits);

/* The view page `p` holds (whether or not it is still live). */
int bcir_kv_view(const bcir_kv_table *t, uint32_t p, bcir_dma_view *view);

/* The page that owns position `pos`. */
int bcir_kv_page_of(const bcir_kv_table *t, uint64_t pos, uint32_t *page);

/* One row written: the capacity law, then the owning page's view, then its data_gen and the position. */
int bcir_kv_append(bcir_kv_table *t);

/* Evict page `p` once the session is complete: its map_gen bumps and its view, returned in *freed (may be
 * NULL), is free. */
int bcir_kv_evict(bcir_kv_table *t, uint32_t p, bcir_dma_view *freed);

#define BCIR_DFA_MAX_STATES (1u << 16)      /* serve._MAX_DFA_STATES */
#define BCIR_DFA_MAX_TRANSITIONS (1u << 20) /* serve._MAX_DFA_TRANSITIONS */

/* check refusals: the first three alone, the rest accumulating */
#define BCIR_DFA_BAD_VOCAB 0x001u       /* vocab_size below 1 */
#define BCIR_DFA_BAD_EMPTY 0x002u       /* no state */
#define BCIR_DFA_BAD_STATES 0x004u      /* more states than BCIR_DFA_MAX_STATES */
#define BCIR_DFA_BAD_START 0x008u       /* the start state is not a state */
#define BCIR_DFA_BAD_TRANSITIONS 0x010u /* more transitions than BCIR_DFA_MAX_TRANSITIONS: the walk stops there */
#define BCIR_DFA_BAD_TOKEN 0x020u       /* a token outside the vocab */
#define BCIR_DFA_BAD_DUPLICATE 0x040u   /* a token a state admits twice */
#define BCIR_DFA_BAD_TARGET 0x080u      /* an edge to a state that does not exist */
#define BCIR_DFA_BAD_ORDER 0x100u       /* a run of tokens that falls (C only) */
#define BCIR_DFA_BAD_LAYOUT 0x200u      /* a NULL, or offsets that do not delimit runs (C only, alone) */

enum {
  BCIR_DFA_OK = 0,
  BCIR_DFA_ERR_ARG = 1,     /* a NULL, a state that is not one, a token outside the logits */
  BCIR_DFA_ERR_DEADLOCK = 2 /* the state admits no token */
};

typedef struct bcir_token_dfa {
  uint32_t n_states, start;
  const uint32_t *offsets; /* n_states + 1 */
  const uint32_t *tokens;  /* offsets[n_states] */
  const uint32_t *next;    /* the successor of each entry */
} bcir_token_dfa;

/* check_token_dfa's verdict: 0, or the refusal bits. */
unsigned bcir_dfa_check(const bcir_token_dfa *dfa, uint32_t vocab_size);

/* One constrained step from `state` over logits[0 .. vocab): the admitted token in *token, its successor in
 * *next. `dfa` is one bcir_dfa_check admitted for this vocab. */
int bcir_dfa_step(const bcir_token_dfa *dfa, uint32_t state, const float *logits, uint32_t vocab, uint32_t *token,
                  uint32_t *next);

#ifdef __cplusplus
}
#endif

#endif
