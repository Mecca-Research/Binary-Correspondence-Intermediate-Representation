/*===- test_devman.c - device manifests bound at instantiation: contracts, and a driver for the oracle's comparison ===
 *
 *   test_devman          the contracts the oracle cannot state: a short workspace or bank table, a tile past 32 bits,
 *                        the canonical digest of a known envelope, every NULL. Prints `test_devman: OK`.
 *   test_devman FILE     one request per line, as bcir/tests/test_c_devman.py writes them:
 *                          `load PATH TARGET CALGEN`     -> `ok DIGEST N NAME:DOMAIN:CAP:TILE:ALIGN ... rows R`
 *                                                           | `ERR CODE WHERE BITS`
 *                          `probe K NAME CAP TILE ...`   -> `probe BITS MESSAGES` (against the last manifest loaded)
 *                          `admit V`                     -> the bound table's admission bits, or `ERR CODE`
 *                        TARGET and NAME in hexadecimal, `-` for none; CAP, TILE and CALGEN decimal or `-`; V as
 *                        test_dma reads it (`BANK OFF ELEM NDIM S... NSTR T...`).
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_devman.h"

static int failures;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; }          \
  } while (0)

static unsigned char work[1u << 22];

static const char known[] =
    "{\"kind\": \"bcir.device_manifest\", \"schema\": 1, \"device\": \"d\", \"target\": \"t\", \"cal_gen\": 0,\n"
    " \"banks\": [[\"a\", 0, 4096, 1, 64]], \"distance\": [[0]],\n"
    " \"digest\": \"%s\"}";

/* the oracle's digest of `known`: sha256(b'{"banks": [["a", 0, 4096, 1, 64]], "cal_gen": 0, "device": "d", '
 * b'"distance": [[0]], "target": "t"}') */
static const char known_digest[] = "f569ed00289f997535344dca0daa03a989547d2e44171d39b804a625c8a5ae6a";

static int load_text(const char *text, bcir_devman *m, unsigned *bits) {
  uint32_t where;
  bcir_jer_diag d;
  return bcir_devman_load((const uint8_t *)text, strlen(text), NULL, 0, 0, 0, work, sizeof work, m, &where, bits, &d);
}

static void contracts(void) {
  char doc[1024];
  bcir_devman m;
  unsigned bits;
  snprintf(doc, sizeof doc, known, known_digest);
  CHECK(load_text(doc, &m, &bits) == BCIR_DEVMAN_OK && m.n_banks == 1 && m.banks[0].capacity == 4096);
  /* the digest re-derives octet for octet: one octet off is tampering */
  char bad[1024];
  memcpy(bad, doc, sizeof bad);
  char *at = strstr(bad, known_digest);
  at[0] = at[0] == '0' ? '1' : '0';
  CHECK(load_text(bad, &m, &bits) == BCIR_DEVMAN_ERR_TAMPERED);
  /* a short workspace: refused without the reader's stack, and after the scan without room for what it counted;
   * the reader's own refusal comes first (a malformed document in the same short workspace) */
  uint32_t where;
  bcir_jer_diag d;
  size_t need = bcir_devman_workspace_size(strlen(doc));
  CHECK(need > 0 && need <= sizeof work);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, work, 16, &m, &where, &bits, &d) ==
        BCIR_DEVMAN_ERR_SPACE);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, work, 2048, &m, &where, &bits, &d) ==
        BCIR_DEVMAN_ERR_SPACE);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc) - 1u, NULL, 0, 0, 0, work, 2048, &m, &where, &bits, &d) ==
        BCIR_DEVMAN_ERR_READ);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, work, need, &m, &where, &bits, &d) ==
        BCIR_DEVMAN_OK);
  /* the smallest workspace that loads, found by bisection, is never written past */
  size_t lo = 0, hi = need;
  while (lo + 1u < hi) {
    size_t mid = lo + (hi - lo) / 2u;
    if (bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, work, mid, &m, &where, &bits, &d) ==
        BCIR_DEVMAN_OK)
      hi = mid;
    else
      lo = mid;
  }
  memset(work + hi, 0xA5, 8192);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, work, hi, &m, &where, &bits, &d) ==
        BCIR_DEVMAN_OK);
  int untouched = 1;
  for (size_t k = 0; k < 8192u; k++)
    if (work[hi + k] != 0xA5) untouched = 0;
  CHECK(untouched && hi < need);
  CHECK(bcir_devman_workspace_size((size_t)BCIR_DEVMAN_MAX_BYTES + 1u) == 0);
  /* the binding: a short table, and a tile past what a bank table holds */
  bcir_dma_bank table[2];
  CHECK(bcir_devman_dma_banks(&m, table, 0) == BCIR_DEVMAN_ERR_SPACE);
  CHECK(bcir_devman_dma_banks(&m, table, 2) == BCIR_DEVMAN_OK && table[0].capacity == 4096 && table[0].align == 64);
  bcir_devman_bank wide = m.banks[0];
  wide.native_tile = (int64_t)1 << 32;
  bcir_devman w = m;
  w.banks = &wide;
  CHECK(bcir_devman_dma_banks(&w, table, 2) == BCIR_DEVMAN_ERR_WIDTH);
  /* probes */
  bcir_devman_probe p[2] = {{(const uint8_t *)"a", 1, 1, 0, 4096, 2}, {(const uint8_t *)"z", 1, 0, 0, 0, 0}};
  size_t count = 9;
  CHECK(bcir_devman_probe_agree(&m, p, 2, &count) == (BCIR_DEVMAN_PROBE_UNKNOWN) && count == 1);
  p[0].has_tile = 1;
  CHECK(bcir_devman_probe_agree(&m, p, 1, &count) == BCIR_DEVMAN_PROBE_TILE && count == 1);
  CHECK(bcir_devman_bank_index(&m, (const uint8_t *)"a", 1) == 0 && bcir_devman_bank_index(&m, NULL, 0) == -1);
  /* every NULL */
  CHECK(bcir_devman_load(NULL, 4, NULL, 0, 0, 0, work, sizeof work, &m, NULL, NULL, NULL) == BCIR_DEVMAN_ERR_ARG);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 1, 0, 0, work, sizeof work, &m, NULL, NULL, NULL) ==
        BCIR_DEVMAN_ERR_ARG);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, NULL, 0, &m, NULL, NULL, NULL) ==
        BCIR_DEVMAN_ERR_ARG);
  CHECK(bcir_devman_load((const uint8_t *)doc, strlen(doc), NULL, 0, 0, 0, work, sizeof work, NULL, NULL, NULL,
                         NULL) == BCIR_DEVMAN_ERR_ARG);
  CHECK(bcir_devman_dma_banks(NULL, table, 2) == BCIR_DEVMAN_ERR_ARG);
  CHECK(bcir_devman_probe_agree(NULL, p, 2, &count) == 0 && count == 0);
}

static int unhex(const char *s, unsigned char *out, size_t cap, size_t *n) {
  size_t len = strlen(s);
  *n = 0;
  if (!strcmp(s, "-")) return 1;
  if (len % 2u || len / 2u > cap) return 0;
  for (size_t k = 0; k < len; k += 2) {
    unsigned v;
    if (sscanf(s + k, "%2x", &v) != 1) return 0;
    out[(*n)++] = (unsigned char)v;
  }
  return 1;
}

static bcir_devman man;
static int have_man;
static unsigned char doc_buf[(16u << 20) + 16u];
static unsigned char names[64][256];

static int read_view(char **p, bcir_dma_view *v) {
  char *end;
  memset(v, 0, sizeof *v);
  unsigned long long f[4];
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
  unsigned long long ns = strtoull(*p, &end, 10);
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
  static char line[1 << 16];
  int rc = 0;
  while (fgets(line, sizeof line, f)) {
    char a[4][4096];
    line[strcspn(line, "\n")] = 0;
    if (!strncmp(line, "load ", 5) && sscanf(line + 5, "%4095s %4095s %4095s", a[0], a[1], a[2]) == 3) {
      FILE *g = fopen(a[0], "rb");
      if (!g) { rc = 2; break; }
      size_t len = fread(doc_buf, 1, sizeof doc_buf, g);
      fclose(g);
      unsigned char target[2048];
      size_t target_len;
      if (!unhex(a[1], target, sizeof target, &target_len)) { rc = 2; break; }
      int has_cal = strcmp(a[2], "-") != 0;
      unsigned long long cal = has_cal ? strtoull(a[2], NULL, 10) : 0;
      uint32_t where = 0;
      unsigned bits = 0;
      bcir_jer_diag d;
      /* the workspace the header promises is enough for any envelope of this length */
      size_t need = bcir_devman_workspace_size(len);
      static unsigned char *big;
      static size_t big_len;
      unsigned char *ws = work;
      size_t ws_len = sizeof work;
      if (need > sizeof work) {
        if (need > big_len) {
          free(big);
          big = (unsigned char *)malloc(need);
          big_len = big ? need : 0;
        }
        if (!big) { rc = 2; break; }
        ws = big;
        ws_len = big_len;
      }
      int r = bcir_devman_load(doc_buf, len, target, target_len, has_cal, cal, ws, ws_len, &man, &where, &bits, &d);
      have_man = r == BCIR_DEVMAN_OK;
      if (r) {
        printf("ERR %d %u %u\n", r, where, bits);
        continue;
      }
      printf("ok ");
      for (int k = 0; k < 32; k++) printf("%02x", man.digest[k]);
      printf(" %u", man.n_banks);
      for (uint32_t i = 0; i < man.n_banks; i++) {
        printf(" ");
        for (size_t k = 0; k < man.banks[i].name_len; k++) printf("%02x", man.banks[i].name[k]);
        printf(":%u:%lld:%lld:%lld", man.banks[i].domain, (long long)man.banks[i].capacity,
               (long long)man.banks[i].native_tile, (long long)man.banks[i].align);
      }
      printf(" rows %u\n", man.n_rows);
    } else if (!strncmp(line, "probe ", 6) && have_man) {
      char *p = line + 6, *end;
      unsigned long k = strtoul(p, &end, 10);
      bcir_devman_probe obs[64];
      if (end == p || k > 64) { rc = 2; break; }
      p = end;
      for (unsigned long j = 0; j < k && !rc; j++) {
        int used;
        if (sscanf(p, " %4095s %4095s %4095s%n", a[0], a[1], a[2], &used) != 3) { rc = 2; break; }
        p += used;
        size_t n;
        if (!unhex(a[0], names[j], sizeof names[j], &n)) { rc = 2; break; }
        obs[j].name = names[j];
        obs[j].name_len = n;
        obs[j].has_capacity = strcmp(a[1], "-") != 0;
        obs[j].capacity = obs[j].has_capacity ? strtoull(a[1], NULL, 10) : 0;
        obs[j].has_tile = strcmp(a[2], "-") != 0;
        obs[j].native_tile = obs[j].has_tile ? strtoull(a[2], NULL, 10) : 0;
      }
      if (rc) break;
      size_t messages = 0;
      unsigned bits = bcir_devman_probe_agree(&man, obs, k, &messages);
      printf("probe %u %zu\n", bits, messages);
    } else if (!strncmp(line, "admit ", 6) && have_man) {
      char *p = line + 6;
      bcir_dma_view v;
      static bcir_dma_bank table[256];
      if (!read_view(&p, &v)) { rc = 2; break; }
      int r = bcir_devman_dma_banks(&man, table, 256);
      if (r) printf("ERR %d\n", r);
      else printf("%u\n", bcir_dma_admit(&v, table, man.n_banks));
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
  if (failures) { printf("test_devman: %d FAILED\n", failures); return 1; }
  printf("test_devman: OK\n");
  return 0;
}
