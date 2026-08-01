/*===- test_jer_simd.cpp - line-protocol driver for the J5 SIMD rail -------===
 *
 * Hosted test driver. The protocol mirrors the other twins so the Python differential can
 * drive it without a second marshalling convention to get wrong.
 *
 * Input, one command per line:
 *
 *   tiers                        report which tiers this build has and this CPU allows
 *   utf8 <tier> <hex>            validate at a pinned tier; `-` spells the empty document
 *   bench <tier> <rounds> <iterations> <hex>   per-round median nanoseconds, and the CPU
 *                                              each round ran on (-1 where unavailable)
 *
 * Output:
 *
 *   tiers <available> <name> <compiled-scalar,sse2,avx2,neon>
 *   ok <status> <offset>         status and byte offset, exactly as the scalar rail gives
 *   sample <tier> <round> <ns> <cpu>
 *===----------------------------------------------------------------------===*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "bcir_jer_simd.h"

/* `sched_getcpu` is glibc/bionic, so the CPU a round ran on is Linux-only. Included outside
 * the anonymous namespace because it is a system header. */
#if defined(__linux__)
#include <sched.h>
#endif

namespace {

constexpr size_t kMaxBytes = 1u << 22;
constexpr size_t kMaxLine = kMaxBytes * 2 + 64;
constexpr size_t kMaxRounds = 256;
constexpr size_t kMaxIterations = 4096;

int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* `-` spells the empty document, which the scalar rail accepts and which a hex string
 * cannot express. */
long unhex(const char *text, unsigned char *out, size_t cap) {
  size_t n = 0;
  if (text[0] == '-' && (text[1] == '\0' || text[1] == '\n' || text[1] == '\r')) return 0;
  while (*text != '\0' && *text != '\n' && *text != '\r') {
    int hi = hex_nibble(static_cast<unsigned char>(text[0]));
    int lo = text[1] ? hex_nibble(static_cast<unsigned char>(text[1])) : -1;
    if (hi < 0 || lo < 0 || n >= cap) return -1;
    out[n++] = static_cast<unsigned char>((hi << 4) | lo);
    text += 2;
  }
  return static_cast<long>(n);
}

uint64_t now_ns() {
  /* ISO C11 timespec_get, matching bcir_microbench.c and bcir_asn1_bench.c: the same clock
   * the rest of the repository's measurement uses, so numbers are comparable across it. */
  struct timespec t = {0, 0};
  if (timespec_get(&t, TIME_UTC) != TIME_UTC || t.tv_sec < 0 || t.tv_nsec < 0) return 0;
  return static_cast<uint64_t>(t.tv_sec) * 1000000000u + static_cast<uint64_t>(t.tv_nsec);
}

/* Which CPU the calling thread is on right now, or -1 where the host cannot say.
 *
 * Reported per round because the first DEDICATED aarch64 host available for J5's advantage
 * clause is a phone, and a phone is big.LITTLE: a Snapdragon 8 Gen 3 pairs one Cortex-X4
 * with four A720s and three A520s, and the same code on the largest and the smallest core
 * differs by more than the SIMD advantage being measured. Without this the scheduler's
 * choice is an INVISIBLE variable; with it, "every round on cpu7" is evidence a reader can
 * check, and a run that migrated says so instead of quietly averaging two machines.
 *
 * -1 means UNKNOWN and must never read as "did not migrate": an absent measurement passing
 * for a clean one is the failure this whole line of reporting exists to stop. */
int current_cpu() {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

int cmp_u64(const void *a, const void *b) {
  uint64_t x = *static_cast<const uint64_t *>(a), y = *static_cast<const uint64_t *>(b);
  return x < y ? -1 : (x > y ? 1 : 0);
}

bcir_jer_simd_tier parse_tier(const char *name) {
  if (std::strcmp(name, "sse2") == 0) return BCIR_JER_SIMD_SSE2;
  if (std::strcmp(name, "avx2") == 0) return BCIR_JER_SIMD_AVX2;
  if (std::strcmp(name, "neon") == 0) return BCIR_JER_SIMD_NEON;
  if (std::strcmp(name, "auto") == 0) return bcir_jer_simd_tier_available();
  return BCIR_JER_SIMD_SCALAR;
}

}  // namespace

int main() {
  static char line[kMaxLine];
  static char hex[kMaxLine];
  static unsigned char data[kMaxBytes];
  static uint64_t batch[kMaxIterations];
  static uint64_t rounds_ns[kMaxRounds];
  static int rounds_cpu[kMaxRounds];
  uint64_t sink = 0;

  while (std::fgets(line, static_cast<int>(sizeof(line)), stdin) != nullptr) {
    char op[32];
    if (std::sscanf(line, "%31s", op) != 1) continue;

    if (std::strcmp(op, "tiers") == 0) {
      bcir_jer_simd_tier best = bcir_jer_simd_tier_available();
      std::printf("tiers %d %s %d,%d,%d,%d\n", static_cast<int>(best),
                  bcir_jer_simd_tier_name(best),
                  bcir_jer_simd_tier_compiled(BCIR_JER_SIMD_SCALAR),
                  bcir_jer_simd_tier_compiled(BCIR_JER_SIMD_SSE2),
                  bcir_jer_simd_tier_compiled(BCIR_JER_SIMD_AVX2),
                  bcir_jer_simd_tier_compiled(BCIR_JER_SIMD_NEON));
      continue;
    }

    if (std::strcmp(op, "utf8") == 0) {
      char tier_name[16];
      long len;
      bcir_jer_diag diag;
      bcir_jer_status status;
      if (std::sscanf(line, "%31s %15s %s", op, tier_name, hex) != 3) {
        std::printf("ok -1 0\n");
        continue;
      }
      len = unhex(hex, data, sizeof(data));
      if (len < 0) {
        std::printf("ok -1 0\n");
        continue;
      }
      std::memset(&diag, 0, sizeof(diag));
      status = bcir_jer_validate_utf8_at(parse_tier(tier_name), data,
                                         static_cast<size_t>(len), &diag);
      std::printf("ok %d %lu\n", static_cast<int>(status),
                  static_cast<unsigned long>(diag.offset));
      continue;
    }

    if (std::strcmp(op, "bench") == 0) {
      char tier_name[16];
      long rounds = 0, iterations = 0, len;
      bcir_jer_simd_tier tier;
      if (std::sscanf(line, "%31s %15s %ld %ld %s", op, tier_name, &rounds, &iterations,
                      hex) != 5 ||
          rounds < 1 || rounds > static_cast<long>(kMaxRounds) || iterations < 1 ||
          iterations > static_cast<long>(kMaxIterations)) {
        std::printf("sample bad 0 0\n");
        continue;
      }
      len = unhex(hex, data, sizeof(data));
      if (len < 0) {
        std::printf("sample bad 0 0\n");
        continue;
      }
      tier = parse_tier(tier_name);
      /* Warmup, discarded: the first pass over a cold buffer measures the memory system. */
      for (long i = 0; i < iterations; i++) {
        bcir_jer_diag diag;
        sink += static_cast<uint64_t>(
            bcir_jer_validate_utf8_at(tier, data, static_cast<size_t>(len), &diag));
      }
      for (long r = 0; r < rounds; r++) {
        for (long i = 0; i < iterations; i++) {
          bcir_jer_diag diag;
          uint64_t t0 = now_ns();
          sink += static_cast<uint64_t>(
              bcir_jer_validate_utf8_at(tier, data, static_cast<size_t>(len), &diag));
          batch[i] = now_ns() - t0;
        }
        std::qsort(batch, static_cast<size_t>(iterations), sizeof(batch[0]), cmp_u64);
        /* The per-round MEDIAN, not the mean: one preempted iteration must not move the
         * round, and `timespec_get` granularity quantizes a short iteration to zero. */
        rounds_ns[r] = batch[iterations / 2];
        rounds_cpu[r] = current_cpu();
      }
      for (long r = 0; r < rounds; r++)
        std::printf("sample %s %ld %llu %d\n", tier_name, r,
                    static_cast<unsigned long long>(rounds_ns[r]), rounds_cpu[r]);
      continue;
    }
  }
  /* Consume the sink so no validation above is dead code. */
  if (sink == 0xFFFFFFFFFFFFFFFFull) std::printf("sink %llu\n",
                                                 static_cast<unsigned long long>(sink));
  return 0;
}
