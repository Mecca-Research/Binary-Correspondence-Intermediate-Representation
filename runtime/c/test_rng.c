/*===- test_rng.c - the counter-based RNG's known answers and contracts, and its text driver ---===
 *
 *   test_rng          the published Philox4x32-10 known answers, then the streams' contracts: a fill equals its
 *                     elements one by one from any start, `below` stays under its bound and a bound of 1 or 0
 *                     takes what it says, a permutation is one, the floats stay in [0, 1). Prints `test_rng: OK`.
 *   test_rng FILE     one request per line, printed as bcir/tests/test_c_rng.py prints the oracle's answer:
 *                     `philox C0 C1 C2 C3 K0 K1` -> four words; `fill K0 K1 STREAM FIRST N` -> N words;
 *                     `below K0 K1 STREAM INDEX BOUND` -> `VALUE NEXT`; `perm K0 K1 STREAM INDEX N` -> `P,... NEXT`;
 *                     `f32 BITS` and `f64 HI LO` -> the value as `%a`. Numbers are decimal; words print as hex.
 *===----------------------------------------------------------------------===*/
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_rng.h"

static int failures;
#define CHECK(cond)                                                                    \
  do {                                                                                 \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
  } while (0)

static void test_known_answers(void) {
  static const uint32_t kat[3][10] = {
      {0, 0, 0, 0, 0, 0, 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u},
      {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0x408f276du, 0x41c83b0eu,
       0xa20bc7c6u, 0x6d5451fdu},
      {0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xa4093822u, 0x299f31d0u, 0xd16cfe09u, 0x94fdccebu,
       0x5001e420u, 0x24126ea1u},
  };
  for (int k = 0; k < 3; k++) {
    uint32_t out[4];
    bcir_philox4x32_10(kat[k], kat[k] + 4, out);
    CHECK(out[0] == kat[k][6] && out[1] == kat[k][7] && out[2] == kat[k][8] && out[3] == kat[k][9]);
  }
}

static void test_streams(void) {
  const uint32_t key[2] = {0x01234567u, 0x89abcdefu};
  uint32_t buf[64];
  for (uint64_t first = 0; first < 9; first++) {   /* every lane offset a fill may start at */
    uint64_t next = bcir_rng_fill_u32(key, 77, first, buf, 37);
    CHECK(next == first + 37);
    for (size_t k = 0; k < 37; k++) CHECK(buf[k] == bcir_rng_u32(key, 77, first + k));
  }
  /* the last elements of the counter space, and two streams that differ only in the high word */
  CHECK(bcir_rng_fill_u32(key, 0, UINT64_MAX - 3, buf, 4) == 0);   /* wraps to 0: past the last element */
  CHECK(buf[3] == bcir_rng_u32(key, 0, UINT64_MAX));
  CHECK(bcir_rng_u32(key, 1, 5) != bcir_rng_u32(key, (uint64_t)1 << 32 | 1, 5));
  uint64_t at = 0;
  CHECK(bcir_rng_below(key, 3, &at, 0) == 0 && at == 0);
  CHECK(bcir_rng_below(key, 3, &at, 1) == 0 && at == 1);
  for (uint32_t bound = 1; bound < 2000; bound += 7) {
    uint64_t index = bound;
    uint32_t v = bcir_rng_below(key, 9, &index, bound);
    CHECK(v < bound && index > bound);
  }
  uint64_t big = 0;
  CHECK(bcir_rng_below(key, 9, &big, UINT32_MAX) < UINT32_MAX);
  uint32_t perm[300];
  unsigned char seen[300];
  uint64_t pidx = 11;
  bcir_rng_permutation(key, 4, &pidx, perm, 300);
  memset(seen, 0, sizeof seen);
  for (int k = 0; k < 300; k++) { CHECK(perm[k] < 300 && !seen[perm[k]]); if (perm[k] < 300) seen[perm[k]] = 1; }
  CHECK(pidx >= 11 + 299);
  uint64_t none = 5;
  bcir_rng_permutation(key, 4, &none, perm, 1);
  CHECK(perm[0] == 0 && none == 5);
  bcir_rng_permutation(key, 4, &none, perm, 0);
  CHECK(none == 5);
  CHECK(bcir_rng_unit_f32(0) == 0.0f && bcir_rng_unit_f32(UINT32_MAX) < 1.0f);
  CHECK(bcir_rng_unit_f32(UINT32_MAX) == 1.0f - 0x1p-24f);
  CHECK(bcir_rng_unit_f64(0, 0) == 0.0 && bcir_rng_unit_f64(UINT32_MAX, UINT32_MAX) < 1.0);
  CHECK(bcir_rng_unit_f64(UINT32_MAX, UINT32_MAX) == 1.0 - 0x1p-53);
}

static int words(char *line, uint64_t *v, int n) {
  char *p = line;
  for (int k = 0; k < n; k++) {
    char *end;
    while (*p == ' ') p++;
    if (*p < '0' || *p > '9') return 0;
    v[k] = strtoull(p, &end, 10);
    p = end;
  }
  while (*p == ' ' || *p == '\n' || *p == '\r') p++;
  return *p == 0;
}

static int drive(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("TOOL cannot read the file\n"); return 2; }
  char line[512];
  uint64_t v[8];
  int rc = 0;
  while (fgets(line, sizeof line, f)) {
    char *rest = strchr(line, ' ');
    if (!rest) { printf("TOOL malformed line\n"); rc = 2; break; }
    *rest++ = 0;
    if (!strcmp(line, "philox") && words(rest, v, 6)) {
      uint32_t c[4] = {(uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[3]}, k[2] = {(uint32_t)v[4], (uint32_t)v[5]}, o[4];
      bcir_philox4x32_10(c, k, o);
      printf("%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n", o[0], o[1], o[2], o[3]);
    } else if (!strcmp(line, "fill") && words(rest, v, 5) && v[4] <= 4096) {
      uint32_t k[2] = {(uint32_t)v[0], (uint32_t)v[1]}, out[4096];
      bcir_rng_fill_u32(k, v[2], v[3], out, (size_t)v[4]);
      for (uint64_t i = 0; i < v[4]; i++) printf("%s%08" PRIx32, i ? " " : "", out[i]);
      printf("\n");
    } else if (!strcmp(line, "below") && words(rest, v, 5)) {
      uint32_t k[2] = {(uint32_t)v[0], (uint32_t)v[1]};
      uint64_t index = v[3];
      uint32_t r = bcir_rng_below(k, v[2], &index, (uint32_t)v[4]);
      printf("%" PRIu32 " %" PRIu64 "\n", r, index);
    } else if (!strcmp(line, "perm") && words(rest, v, 5) && v[4] <= 4096) {
      uint32_t k[2] = {(uint32_t)v[0], (uint32_t)v[1]}, out[4096];
      uint64_t index = v[3];
      bcir_rng_permutation(k, v[2], &index, out, (uint32_t)v[4]);
      for (uint64_t i = 0; i < v[4]; i++) printf("%s%" PRIu32, i ? "," : "", out[i]);
      printf("%s%" PRIu64 "\n", v[4] ? " " : "- ", index);
    } else if (!strcmp(line, "f32") && words(rest, v, 1)) {
      printf("%a\n", (double)bcir_rng_unit_f32((uint32_t)v[0]));
    } else if (!strcmp(line, "f64") && words(rest, v, 2)) {
      printf("%a\n", bcir_rng_unit_f64((uint32_t)v[0], (uint32_t)v[1]));
    } else {
      printf("TOOL malformed line\n");
      rc = 2;
      break;
    }
  }
  fclose(f);
  return rc;
}

int main(int argc, char **argv) {
  if (argc == 2) return drive(argv[1]);
  test_known_answers();
  test_streams();
  if (failures) { printf("test_rng: %d FAILED\n", failures); return 1; }
  printf("test_rng: OK\n");
  return 0;
}
