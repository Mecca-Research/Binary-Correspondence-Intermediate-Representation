/*===- test_jer_index.cpp - drive both bounding passes over one document ---===
 *
 * Hosted, and a TEST driver only. It runs `bcir_jer_scan` and `bcir_jer_index_scan` over the
 * same octets under the same limits and prints both answers, so the Python differential
 * compares them rather than trusting either.
 *
 * The work ceiling is a parameter because that is where the two rails could most easily
 * disagree: the index charges a whitespace RUN in one call, and 4.3's budget failing
 * mid-run is exactly the case a bulk charge could get subtly wrong -- same verdict, wrong
 * octet. Sweeping the ceiling walks the failure point across every position in the document.
 *
 * Input, one command per line:
 *
 *   both <work> <hex>     `work` 0 keeps the default ceiling; `-` spells the empty document
 *
 * Output:
 *
 *   both <status> <offset> <needed> <nodes> | <status> <offset> <needed> <nodes>
 *          ^ bcir_jer_scan                     ^ bcir_jer_index_scan
 *===----------------------------------------------------------------------===*/
#include <cstdio>
#include <cstring>

#include "bcir_jer.h"
#include "bcir_jer_index.h"

namespace {

constexpr size_t kMaxBytes = 1u << 22;
constexpr size_t kMaxLine = kMaxBytes * 2 + 64;
constexpr size_t kMaxDepth = 64;

int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* `-` spells the EMPTY document, which both rails accept and which a hex string cannot
 * otherwise express. */
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

}  // namespace

int main() {
  static char line[kMaxLine];
  static char hex[kMaxLine];
  static unsigned char data[kMaxBytes];
  static bcir_jer_level scalar_stack[kMaxDepth];
  static bcir_jer_level index_stack[kMaxDepth];

  while (std::fgets(line, static_cast<int>(sizeof(line)), stdin) != nullptr) {
    char op[32];
    unsigned long work = 0;
    long len;
    if (std::sscanf(line, "%31s", op) != 1) continue;
    if (std::strcmp(op, "both") != 0) continue;
    if (std::sscanf(line, "%31s %lu %s", op, &work, hex) != 3) {
      std::printf("both -1 0 0 0 | -1 0 0 0\n");
      continue;
    }
    len = unhex(hex, data, sizeof(data));
    if (len < 0) {
      std::printf("both -1 0 0 0 | -1 0 0 0\n");
      continue;
    }

    bcir_jer_limits limits;
    bcir_jer_default_limits(&limits);
    /* Only ever TIGHTENED -- 4.3 says a caller may narrow a limit and never silently widen
     * one, and a driver that widened it would be measuring a different contract. */
    if (work != 0 && static_cast<uint64_t>(work) < limits.work) limits.work = work;

    bcir_jer_diag scalar_diag;
    bcir_jer_diag index_diag;
    uint64_t scalar_nodes = 0;
    uint64_t index_nodes = 0;
    bcir_jer_status scalar_status = bcir_jer_scan(
        data, static_cast<size_t>(len), &limits, scalar_stack, kMaxDepth, &scalar_nodes,
        &scalar_diag);
    bcir_jer_status index_status = bcir_jer_index_scan(
        data, static_cast<size_t>(len), &limits, index_stack, kMaxDepth, &index_nodes,
        &index_diag);

    std::printf("both %d %llu %llu %llu | %d %llu %llu %llu\n",
                static_cast<int>(scalar_status),
                static_cast<unsigned long long>(scalar_diag.offset),
                static_cast<unsigned long long>(scalar_diag.needed),
                static_cast<unsigned long long>(scalar_nodes),
                static_cast<int>(index_status),
                static_cast<unsigned long long>(index_diag.offset),
                static_cast<unsigned long long>(index_diag.needed),
                static_cast<unsigned long long>(index_nodes));
  }
  return 0;
}
