/*===- test_digest_hw.c - the hardware digests against the standards and the portable code, and a driver ---===
 *
 *   test_digest_hw          on every path the probe reports, and on the portable code: FIPS 180-4's SHA-256
 *                           vectors and zlib's CRC-32 check value; then each path against the portable code over
 *                           every length to 1,100 bytes from every alignment, updates split at every place, and
 *                           running states. Prints the paths it ran, then `test_digest_hw: OK`.
 *   test_digest_hw FILE     one request per line, as bcir/tests/test_c_digest_hw.py writes them:
 *                           `probe` and `compiled` -> the mask; `sha256 MASK HEX` -> the digest in hex;
 *                           `crc32 MASK STATE HEX` -> bcir_crc32_step_hw's state in hex; `witness` -> the four
 *                           kernel counters, or `off` in a build without BCIR_DIGEST_HW_WITNESS.
 *   test_digest_hw --bench  each path's throughput over 1 MiB, in MB/s: wall-clock, indicative only.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bcir_crc32.h"
#include "bcir_digest_hw.h"
#include "bcir_sha256.h"

static int failures;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; }          \
  } while (0)

static const char *path_name(unsigned bit) {
  switch (bit) {
  case BCIR_DIGEST_SHA256_SHANI: return "sha256:sha-ni";
  case BCIR_DIGEST_SHA256_ARMV8: return "sha256:armv8";
  case BCIR_DIGEST_CRC32_CLMUL: return "crc32:pclmulqdq";
  case BCIR_DIGEST_CRC32_ARMV8: return "crc32:armv8";
  default: return "?";
  }
}

static void hex(const uint8_t *d, size_t n, char *out) {
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < n; i++) { out[2 * i] = digits[d[i] >> 4]; out[2 * i + 1] = digits[d[i] & 15u]; }
  out[2 * n] = 0;
}

/* a deterministic byte pattern: xorshift32 */
static void fill(uint8_t *p, size_t n, uint32_t seed) {
  uint32_t x = seed ? seed : 1u;
  for (size_t i = 0; i < n; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; p[i] = (uint8_t)(x >> 11); }
}

static void test_vectors(unsigned mask) {
  static const struct { const char *msg; const char *want; } fips[] = {
    {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
    {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
     "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrst"
     "nopqrstu",
     "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
  };
  char got[65];
  uint8_t out[32];
  for (size_t k = 0; k < sizeof fips / sizeof fips[0]; k++) {
    bcir_sha256_digest_hw((const uint8_t *)fips[k].msg, strlen(fips[k].msg), out, mask);
    hex(out, 32, got);
    CHECK(strcmp(got, fips[k].want) == 0);
  }
  /* FIPS 180-4's million 'a', in uneven pieces */
  static uint8_t a[1100];
  memset(a, 'a', sizeof a);
  bcir_sha256 s;
  bcir_sha256_init(&s);
  for (size_t done = 0, step = 1; done < 1000000u; step = step % 997u + 61u) { /* steps of 1 to 1,057 bytes */
    size_t take = 1000000u - done < step ? 1000000u - done : step;
    bcir_sha256_update_hw(&s, a, take, mask);
    done += take;
  }
  bcir_sha256_final(&s, out);
  hex(out, 32, got);
  CHECK(strcmp(got, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") == 0);
  CHECK(bcir_crc32_hw((const uint8_t *)"123456789", 9, mask) == 0xCBF43926u);
  CHECK(bcir_crc32_hw(NULL, 0, mask) == 0u);
}

static void test_against_portable(unsigned mask) {
  static uint8_t buf[1100 + 16 + 4096];
  fill(buf, sizeof buf, 0x9e3779b9u ^ mask);
  uint8_t want[32], got[32];
  for (size_t off = 0; off < 16; off++) {
    for (size_t n = 0; n <= 1100; n += off ? 7 : 1) {
      const uint8_t *p = buf + off;
      uint32_t state = (uint32_t)(n * 2654435761u) ^ (uint32_t)off;
      CHECK(bcir_crc32_step_hw(state, p, n, mask) == bcir_crc32_step(state, p, n));
      bcir_sha256_digest(p, n, want);
      bcir_sha256_digest_hw(p, n, got, mask);
      CHECK(memcmp(want, got, 32) == 0);
    }
  }
  /* an update split at every place, around partial blocks, each half on the path */
  for (size_t n = 0; n <= 300; n += 13) {
    for (size_t cut = 0; cut <= n; cut++) {
      bcir_sha256 s;
      bcir_sha256_init(&s);
      bcir_sha256_update_hw(&s, buf, cut, mask);
      bcir_sha256_update_hw(&s, buf + cut, n - cut, mask);
      bcir_sha256_final(&s, got);
      bcir_sha256_digest(buf, n, want);
      CHECK(memcmp(want, got, 32) == 0);
      uint32_t c = bcir_crc32_step_hw(bcir_crc32_step_hw(0xFFFFFFFFu, buf, cut, mask), buf + cut, n - cut, mask);
      CHECK(c == bcir_crc32_step(0xFFFFFFFFu, buf, n));
    }
  }
  /* a long buffer: many four-lane folds and SHA blocks */
  CHECK(bcir_crc32_step_hw(7u, buf, sizeof buf, mask) == bcir_crc32_step(7u, buf, sizeof buf));
  bcir_sha256_digest(buf, sizeof buf, want);
  bcir_sha256_digest_hw(buf, sizeof buf, got, mask);
  CHECK(memcmp(want, got, 32) == 0);
}

static int self_test(void) {
  unsigned probe = bcir_digest_probe(), compiled = bcir_digest_compiled();
  CHECK((probe & ~compiled) == 0u);
  printf("paths:");
  for (unsigned bit = 1u; bit <= 8u; bit <<= 1)
    if (probe & bit) printf(" %s", path_name(bit));
  printf("%s\n", probe ? "" : " none (the portable code only)");
  test_vectors(0u);
  test_against_portable(0u);
  for (unsigned bit = 1u; bit <= 8u; bit <<= 1)
    if (probe & bit) { test_vectors(bit); test_against_portable(bit); }
  test_vectors(probe);
  test_against_portable(probe);
  /* a bit this build does not carry is ignored, not run */
  test_vectors(~compiled & 0xFu);
  if (failures) { printf("test_digest_hw: %d FAILED\n", failures); return 1; }
  printf("test_digest_hw: OK\n");
  return 0;
}

static int bench(void) {
  enum { SIZE = 1 << 20, ROUNDS = 64 };
  uint8_t *buf = malloc(SIZE);
  if (!buf) { printf("TOOL out of memory\n"); return 2; }
  fill(buf, SIZE, 5u);
  unsigned probe = bcir_digest_probe();
  for (int which = 0; which < 2; which++) {
    for (unsigned bit = 0u; bit <= 8u; bit = bit ? bit << 1 : 1u) {
      unsigned kind = which ? (BCIR_DIGEST_CRC32_CLMUL | BCIR_DIGEST_CRC32_ARMV8)
                            : (BCIR_DIGEST_SHA256_SHANI | BCIR_DIGEST_SHA256_ARMV8);
      if (bit && (!(probe & bit) || !(kind & bit))) continue;
      uint8_t out[32];
      uint32_t sink = 0;
      clock_t t0 = clock();
      for (int r = 0; r < ROUNDS; r++) {
        if (which) sink ^= bcir_crc32_hw(buf, SIZE, bit);
        else { bcir_sha256_digest_hw(buf, SIZE, out, bit); sink ^= out[r & 31]; }
      }
      double s = (double)(clock() - t0) / CLOCKS_PER_SEC;
      printf("%s %s %.0f MB/s (sink %u)\n", which ? "crc32" : "sha256", bit ? path_name(bit) : "portable",
             s > 0 ? (double)SIZE * ROUNDS / s / 1e6 : 0.0, (unsigned)sink & 1u);
    }
  }
  free(buf);
  return 0;
}

static int unhex(const char *s, uint8_t *out, size_t cap, size_t *n) {
  size_t len = strspn(s, "0123456789abcdef");
  if (len % 2 || len / 2 > cap) return 0;
  for (size_t i = 0; i < len / 2; i++) {
    unsigned hi = (unsigned)(s[2 * i] <= '9' ? s[2 * i] - '0' : s[2 * i] - 'a' + 10);
    unsigned lo = (unsigned)(s[2 * i + 1] <= '9' ? s[2 * i + 1] - '0' : s[2 * i + 1] - 'a' + 10);
    out[i] = (uint8_t)(hi << 4 | lo);
  }
  *n = len / 2;
  s += len;
  return *s == 0 || *s == '\n' || *s == '\r';
}

static int drive(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("TOOL cannot read the file\n"); return 2; }
  static char line[(1 << 17) + 64];
  static uint8_t data[1 << 16];
  int rc = 0;
  while (fgets(line, sizeof line, f)) {
    unsigned mask, state;
    int at = 0;
    size_t n = 0;
    char out[65];
    if (!strcmp(line, "probe\n")) printf("%u\n", bcir_digest_probe());
    else if (!strcmp(line, "compiled\n")) printf("%u\n", bcir_digest_compiled());
    else if (!strcmp(line, "witness\n")) {
#ifdef BCIR_DIGEST_HW_WITNESS
      printf("%lu %lu %lu %lu\n", bcir_digest_hw_witness[0], bcir_digest_hw_witness[1], bcir_digest_hw_witness[2],
             bcir_digest_hw_witness[3]);
#else
      printf("off\n");
#endif
    } else if (sscanf(line, "sha256 %u %n", &mask, &at) == 1 && at && unhex(line + at, data, sizeof data, &n)) {
      uint8_t d[32];
      bcir_sha256_digest_hw(data, n, d, mask);
      hex(d, 32, out);
      printf("%s\n", out);
    } else if (sscanf(line, "crc32 %u %u %n", &mask, &state, &at) == 2 && at &&
               unhex(line + at, data, sizeof data, &n)) {
      printf("%08x\n", (unsigned)bcir_crc32_step_hw((uint32_t)state, data, n, mask));
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
  if (argc == 2 && !strcmp(argv[1], "--bench")) return bench();
  if (argc == 2) return drive(argv[1]);
  return self_test();
}
