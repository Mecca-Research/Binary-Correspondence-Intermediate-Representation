/*===- test_serve.c - the serving control plane: contracts, and a driver for the oracle's comparison ------===
 *
 *   test_serve          the contracts the oracle cannot state: a page table with too few page records says how
 *                       many it needs and writes nothing, page arithmetic past 64 bits is refused, the DFA's
 *                       layout and order laws, its transition bound, every NULL. Prints `test_serve: OK`.
 *   test_serve FILE     one request per line, as bcir/tests/test_c_serve.py writes them:
 *                         `bank CAP TILE ALIGN`, `reset`                 the banks (no output)
 *                         `kv NL KVDIM CAP PSIZE BANK RID`             -> `ok NPAGES ROWELEMS PAGEBYTES` | `ERR C P B`
 *                         `append`                                     -> `ok POS` | `ERR C`
 *                         `pageof POS`, `evict P`                      -> `ok P`, `ok OFFSET` | `ERR C`
 *                         `state`                                      -> `pos N RID:MAP:DATA:LIVE ...`
 *                         `dfa VOCAB START N K T:S ... K T:S ...`      -> `check BITS`
 *                         `step STATE L0 L1 ...`                       -> `ok TOKEN NEXT` | `ERR C`
 *                       decimal, the logits as C99 strtof reads them (hexadecimal, nan, inf).
 *===----------------------------------------------------------------------===*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_serve.h"

static int failures;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; }          \
  } while (0)

static uint32_t big_tokens[BCIR_DFA_MAX_TRANSITIONS + 1u], big_next[BCIR_DFA_MAX_TRANSITIONS + 1u];

static void contracts(void) {
  const bcir_dma_bank banks[2] = {{1u << 20, 16, 64}, {UINT64_MAX, 1, 1}};
  bcir_kv_page pages[8], none[1];
  bcir_kv_table t;
  uint32_t bad = 7;
  unsigned bits = 7;
  /* 2 layers x kv_dim 8: 32 elements a row, 16-row pages of 2 KiB */
  CHECK(bcir_kv_init(&t, 2, 8, 32, 16, banks, 2, 0, 7000, pages, 8, &bad, &bits) == BCIR_KV_OK);
  CHECK(t.n_pages == 2 && t.row_elems == 32 && t.page_bytes == 2048 && pages[1].rid == 7001 && bad == 0);
  /* too few page records: the count it needs, nothing written */
  memset(none, 0xEE, sizeof none);
  CHECK(bcir_kv_init(&t, 2, 8, 32, 8, banks, 2, 0, 7000, none, 1, &bad, &bits) == BCIR_KV_ERR_SPACE && bad == 4);
  CHECK(none[0].rid == 0xEEEEEEEEu && t.n_pages == 2);
  /* a page that fragments the tile: refused before anything is written */
  CHECK(bcir_kv_init(&t, 2, 8, 30, 15, banks, 2, 0, 7000, pages, 8, &bad, &bits) == BCIR_KV_ERR_VIEW && bad == 0 &&
        bits == BCIR_DMA_BAD_TILE_OUTER);
  CHECK(t.n_pages == 2 && t.page_size == 16);
  /* page arithmetic past 64 bits: refused as overflow, on the page that leaves 64 bits */
  CHECK(bcir_kv_init(&t, 0xFFFFFFFFu, 0xFFFFFFFFu, 4, 1, banks, 2, 1, 7000, pages, 8, &bad, &bits) ==
            BCIR_KV_ERR_VIEW &&
        bad == 0 && bits == BCIR_DMA_BAD_OVERFLOW);
  /* 2**62-byte pages: the fourth one's end is 2**64, past what a view can say */
  CHECK(bcir_kv_init(&t, 1u << 31, 1u << 28, 4, 1, banks, 2, 1, 7000, pages, 8, &bad, &bits) == BCIR_KV_ERR_VIEW &&
        bad == 3 && bits == BCIR_DMA_BAD_OVERFLOW);
  /* the last RID: 2**32 - 1 is a page's, one more is not */
  CHECK(bcir_kv_init(&t, 2, 8, 32, 16, banks, 2, 0, UINT32_MAX - 1u, pages, 8, &bad, &bits) == BCIR_KV_OK);
  CHECK(bcir_kv_init(&t, 2, 8, 48, 16, banks, 2, 0, UINT32_MAX - 1u, pages, 8, &bad, &bits) == BCIR_KV_ERR_RID_RANGE);
  CHECK(bcir_kv_init(&t, 2, 8, 32, 16, banks, 2, 0, 7000, pages, 8, &bad, &bits) == BCIR_KV_OK);
  /* every NULL */
  CHECK(bcir_kv_init(NULL, 2, 8, 32, 16, banks, 2, 0, 7000, pages, 8, &bad, &bits) == BCIR_KV_ERR_ARG);
  CHECK(bcir_kv_init(&t, 2, 8, 32, 16, NULL, 2, 0, 7000, pages, 8, &bad, &bits) == BCIR_KV_ERR_ARG);
  CHECK(bcir_kv_init(&t, 2, 8, 32, 16, banks, 2, 0, 7000, NULL, 8, NULL, NULL) == BCIR_KV_ERR_ARG);
  CHECK(bcir_kv_append(NULL) == BCIR_KV_ERR_ARG && bcir_kv_evict(NULL, 0, NULL) == BCIR_KV_ERR_ARG);
  uint32_t page;
  CHECK(bcir_kv_page_of(&t, 0, NULL) == BCIR_KV_ERR_ARG && bcir_kv_page_of(NULL, 0, &page) == BCIR_KV_ERR_ARG);
  bcir_dma_view v;
  CHECK(bcir_kv_view(&t, 0, NULL) == BCIR_KV_ERR_ARG && bcir_kv_view(&t, 2, &v) == BCIR_KV_ERR_NO_PAGE);
  CHECK(bcir_kv_view(&t, 1, &v) == BCIR_KV_OK && v.offset == 2048 && v.shape[0] == 16 && v.strides[0] == 32);

  /* the DFA's layout and order */
  const uint32_t off[3] = {0, 2, 3}, tok[3] = {4, 9, 1}, nxt[3] = {1, 0, 0};
  bcir_token_dfa d = {2, 0, off, tok, nxt};
  CHECK(bcir_dfa_check(&d, 16) == 0);
  CHECK(bcir_dfa_check(NULL, 16) == BCIR_DFA_BAD_LAYOUT);
  const uint32_t falling[3] = {9, 4, 1};
  d.tokens = falling;
  CHECK(bcir_dfa_check(&d, 16) == BCIR_DFA_BAD_ORDER);
  d.tokens = tok;
  const uint32_t from1[3] = {1, 2, 3}, down[3] = {0, 3, 2};
  d.offsets = from1;
  CHECK(bcir_dfa_check(&d, 16) == BCIR_DFA_BAD_LAYOUT);
  d.offsets = down;
  CHECK(bcir_dfa_check(&d, 16) == BCIR_DFA_BAD_LAYOUT);
  d.offsets = off;
  d.next = NULL;
  CHECK(bcir_dfa_check(&d, 16) == BCIR_DFA_BAD_LAYOUT);
  d.next = nxt;
  /* the transition bound: 2**20 transitions are admitted, one more is refused */
  static uint32_t big_off[2];
  for (uint32_t k = 0; k <= BCIR_DFA_MAX_TRANSITIONS; k++) big_tokens[k] = k;
  big_off[1] = BCIR_DFA_MAX_TRANSITIONS;
  bcir_token_dfa b = {1, 0, big_off, big_tokens, big_next};
  CHECK(bcir_dfa_check(&b, BCIR_DFA_MAX_TRANSITIONS) == 0);
  big_off[1] = BCIR_DFA_MAX_TRANSITIONS + 1u;
  CHECK(bcir_dfa_check(&b, BCIR_DFA_MAX_TRANSITIONS + 1u) == BCIR_DFA_BAD_TRANSITIONS);
  big_tokens[BCIR_DFA_MAX_TRANSITIONS] = BCIR_DFA_MAX_TRANSITIONS - 1u; /* a repeat is one transition */
  CHECK(bcir_dfa_check(&b, BCIR_DFA_MAX_TRANSITIONS + 1u) == BCIR_DFA_BAD_DUPLICATE);
  /* a step's ties, NaN and refusals */
  float logits[16] = {0};
  uint32_t token, next;
  CHECK(bcir_dfa_step(&d, 0, logits, 16, &token, &next) == BCIR_DFA_OK && token == 4 && next == 1);
  logits[9] = 1.0f;
  CHECK(bcir_dfa_step(&d, 0, logits, 16, &token, &next) == BCIR_DFA_OK && token == 9 && next == 0);
  logits[4] = NAN;
  CHECK(bcir_dfa_step(&d, 0, logits, 16, &token, &next) == BCIR_DFA_OK && token == 4); /* NaN first: kept */
  const uint32_t empty_off[3] = {0, 0, 1};
  bcir_token_dfa e = {2, 0, empty_off, tok, nxt};
  CHECK(bcir_dfa_step(&e, 0, logits, 16, &token, &next) == BCIR_DFA_ERR_DEADLOCK);
  CHECK(bcir_dfa_step(&e, 2, logits, 16, &token, &next) == BCIR_DFA_ERR_ARG);
  CHECK(bcir_dfa_step(&d, 0, logits, 5, &token, &next) == BCIR_DFA_ERR_ARG); /* token 9 past the logits */
  CHECK(bcir_dfa_step(&d, 0, NULL, 16, &token, &next) == BCIR_DFA_ERR_ARG);
  CHECK(bcir_dfa_step(NULL, 0, logits, 16, &token, &next) == BCIR_DFA_ERR_ARG);
}

#define MAX_STATES 256u
#define MAX_ENTRIES 4096u
#define MAX_VOCAB 4096u

static bcir_dma_bank banks[16];
static uint32_t n_banks;
static bcir_kv_page pages[BCIR_KV_MAX_PAGES];
static bcir_kv_table table;
static int have_table;
static uint32_t dfa_off[MAX_STATES + 1u], dfa_tok[MAX_ENTRIES], dfa_next[MAX_ENTRIES];
static bcir_token_dfa dfa;
static float logits[MAX_VOCAB];

static int read_u64(char **p, unsigned long long *v) {
  char *end;
  *v = strtoull(*p, &end, 10);
  if (end == *p) return 0;
  *p = end;
  return 1;
}

static int drive(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("TOOL cannot read the file\n"); return 2; }
  static char line[1 << 16];
  int rc = 0;
  while (fgets(line, sizeof line, f)) {
    char *p = line;
    unsigned long long a[6];
    if (!strncmp(line, "bank ", 5) && n_banks < 16) {
      p += 5;
      if (!read_u64(&p, &a[0]) || !read_u64(&p, &a[1]) || !read_u64(&p, &a[2])) { rc = 2; break; }
      banks[n_banks].capacity = a[0];
      banks[n_banks].native_tile = (uint32_t)a[1];
      banks[n_banks].align = (uint32_t)a[2];
      n_banks++;
    } else if (!strcmp(line, "reset\n")) {
      n_banks = 0;
      have_table = 0;
    } else if (!strncmp(line, "kv ", 3)) {
      p += 3;
      int k = 0;
      while (k < 6 && read_u64(&p, &a[k])) k++;
      if (k != 6) { rc = 2; break; }
      uint32_t bad = 0;
      unsigned bits = 0;
      int r = bcir_kv_init(&table, (uint32_t)a[0], (uint32_t)a[1], (uint32_t)a[2], (uint32_t)a[3], banks, n_banks,
                           (uint32_t)a[4], (uint32_t)a[5], pages, BCIR_KV_MAX_PAGES, &bad, &bits);
      if (r) {
        printf("ERR %d %u %u\n", r, bad, bits);
        have_table = 0;
      } else {
        printf("ok %u %llu %llu\n", table.n_pages, (unsigned long long)table.row_elems,
               (unsigned long long)table.page_bytes);
        have_table = 1;
      }
    } else if (!strcmp(line, "append\n") && have_table) {
      int r = bcir_kv_append(&table);
      if (r) printf("ERR %d\n", r);
      else printf("ok %u\n", table.pos);
    } else if (!strncmp(line, "pageof ", 7) && have_table) {
      p += 7;
      uint32_t page = 0;
      if (!read_u64(&p, &a[0])) { rc = 2; break; }
      int r = bcir_kv_page_of(&table, a[0], &page);
      if (r) printf("ERR %d\n", r);
      else printf("ok %u\n", page);
    } else if (!strncmp(line, "evict ", 6) && have_table) {
      p += 6;
      bcir_dma_view v;
      if (!read_u64(&p, &a[0])) { rc = 2; break; }
      int r = bcir_kv_evict(&table, a[0] > UINT32_MAX ? UINT32_MAX : (uint32_t)a[0], &v);
      if (r) printf("ERR %d\n", r);
      else printf("ok %llu\n", (unsigned long long)v.offset);
    } else if (!strcmp(line, "state\n") && have_table) {
      printf("pos %u", table.pos);
      for (uint32_t k = 0; k < table.n_pages; k++)
        printf(" %u:%u:%u:%u", pages[k].rid, pages[k].map_gen, pages[k].data_gen, pages[k].live);
      printf("\n");
    } else if (!strncmp(line, "dfa ", 4)) {
      p += 4;
      if (!read_u64(&p, &a[0]) || !read_u64(&p, &a[1]) || !read_u64(&p, &a[2]) || a[2] > MAX_STATES) {
        rc = 2;
        break;
      }
      uint32_t e = 0;
      dfa_off[0] = 0;
      for (uint32_t s = 0; s < a[2] && !rc; s++) {
        unsigned long long k, tk, nx;
        if (!read_u64(&p, &k) || e + k > MAX_ENTRIES) { rc = 2; break; }
        for (unsigned long long j = 0; j < k; j++) {
          if (!read_u64(&p, &tk) || *p != ':') { rc = 2; break; }
          p++;
          if (!read_u64(&p, &nx)) { rc = 2; break; }
          dfa_tok[e] = (uint32_t)tk;
          dfa_next[e] = (uint32_t)nx;
          e++;
        }
        dfa_off[s + 1u] = e;
      }
      if (rc) break;
      dfa.n_states = (uint32_t)a[2];
      dfa.start = (uint32_t)a[1];
      dfa.offsets = dfa_off;
      dfa.tokens = dfa_tok;
      dfa.next = dfa_next;
      printf("check %u\n", bcir_dfa_check(&dfa, (uint32_t)a[0]));
    } else if (!strncmp(line, "step ", 5)) {
      p += 5;
      uint32_t vocab = 0, token = 0, next = 0;
      if (!read_u64(&p, &a[0])) { rc = 2; break; }
      for (;;) {
        char *end;
        float x = strtof(p, &end);
        if (end == p) break;
        if (vocab == MAX_VOCAB) { rc = 2; break; }
        logits[vocab++] = x;
        p = end;
      }
      if (rc) break;
      int r = bcir_dfa_step(&dfa, (uint32_t)a[0], logits, vocab, &token, &next);
      if (r) printf("ERR %d\n", r);
      else printf("ok %u %u\n", token, next);
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
  if (failures) { printf("test_serve: %d FAILED\n", failures); return 1; }
  printf("test_serve: OK\n");
  return 0;
}
