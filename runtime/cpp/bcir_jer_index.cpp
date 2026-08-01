/*===- bcir_jer_index.cpp - the dispatch loop, rebuilt on the public cursor -===
 *
 * See bcir_jer_index.h for why this exists and what it deliberately is not.
 *
 * Every semantic decision below is `bcir_jer_scan`'s, reached through the exported cursor:
 * `bcir_jer_scan_spend` charges 4.3's budget, `bcir_jer_scan_string_token` and
 * `bcir_jer_scan_number_token` consume a token under 4.3's limits, and
 * `bcir_jer_scan_literal_token` recognises `true`/`false`/`null`. What this file owns is the
 * ORDER those are called in and the container bookkeeping -- the part a vector pass can help
 * with, and the only part a differential has to cover.
 *
 * The clause references are the C file's, deliberately: this loop must be readable beside
 * `bcir_jer_scan` and seen to make the same decisions.
 *===----------------------------------------------------------------------===*/
#include "bcir_jer_index.h"

#if defined(__x86_64__) || defined(_M_X64)
#define BCIR_INDEX_X86 1
#include <immintrin.h>
#elif defined(__aarch64__)
#define BCIR_INDEX_ARM 1
#include <arm_neon.h>
#endif

namespace {

/* ECMA-404 clause 4, exactly: SPACE, HORIZONTAL TABULATION, LINE FEED, CARRIAGE RETURN.
 * Nothing else -- and in particular not FORM FEED or VERTICAL TABULATION, which some JSON
 * readers admit and which would let two rails disagree about a document's shape.
 *
 * ONE LIST, NAMED ONCE. The scalar predicate and every vector pass below are written against
 * these four constants rather than each spelling the set out, so there is no second
 * definition of "whitespace" that could drift from the first. That is the same discipline
 * `bcir_jer_simd.cpp` follows for "is this octet ASCII", and for the same reason: the risk 8's
 * table names is not a slow vector pass, it is a vector pass that quietly means something
 * else. */
constexpr uint8_t kSpace = 0x20;
constexpr uint8_t kTab = 0x09;
constexpr uint8_t kLineFeed = 0x0A;
constexpr uint8_t kReturn = 0x0D;

inline bool is_space(uint8_t c) {
  return c == kSpace || c == kTab || c == kLineFeed || c == kReturn;
}

inline bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }

bcir_jer_status fail(bcir_jer_diag *diag, bcir_jer_status status, size_t offset,
                     uint64_t needed) {
  if (diag != nullptr) {
    diag->status = status;
    diag->offset = offset;
    diag->needed = needed;
    diag->sink_code = 0;
  }
  return status;
}

void clear(bcir_jer_diag *diag) {
  if (diag != nullptr) {
    diag->status = BCIR_JER_OK;
    diag->offset = BCIR_JER_NO_OFFSET;
    diag->needed = 0;
    diag->sink_code = 0;
  }
}

/* THE VECTOR PASS, and the only one in this file.
 *
 * The whole accelerable question is "how far does the run of whitespace at `pos` extend",
 * and the answer is positional: every octet in it costs exactly one work unit at its own
 * offset. 7.4 settles what that means -- entering a run of `n` octets with `w` spent against
 * ceiling `L`, the budget fails, when it fails, at offset `L - w` with `needs = L + 1`.
 * Closed form, no re-walk. That is what licenses a block-at-a-time scan: skipping ahead
 * cannot lose the failure position, because the position is arithmetic rather than a
 * consequence of having walked there.
 *
 * Each helper returns a LOWER BOUND -- the last block boundary still entirely whitespace --
 * and `whitespace_run` finishes it scalar. Two reasons, both load-bearing. The bound must be
 * made exact or the dispatch could stop mid-run, make no progress and spin. And keeping the
 * exactness in one scalar line means a tier can only ever be slow or conservative, never
 * wrong about where the run ends. */

#if defined(BCIR_INDEX_X86)
/* 16 octets. SSE2 is x86-64 baseline, so this needs no feature check; it is still tier-gated
 * so a caller can pin scalar.
 *
 * Four equality compares OR'd together, rather than a range test or a shuffle-based lookup.
 * A shuffle table is the faster idiom and is what a JSON scanner usually reaches for -- it is
 * not used here because it encodes the whitespace set as a permuted nibble table that no
 * longer reads like `is_space`, and this file's whole claim is that a reviewer can see the
 * two agree. The compares are already far below the cost of the scalar loop they replace. */
size_t whitespace_run_sse2(const uint8_t *data, size_t len) {
  const __m128i space = _mm_set1_epi8(static_cast<char>(kSpace));
  const __m128i tab = _mm_set1_epi8(static_cast<char>(kTab));
  const __m128i feed = _mm_set1_epi8(static_cast<char>(kLineFeed));
  const __m128i ret = _mm_set1_epi8(static_cast<char>(kReturn));
  size_t at = 0;
  for (; at + 16 <= len; at += 16) {
    __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + at));
    __m128i ws = _mm_or_si128(
        _mm_or_si128(_mm_cmpeq_epi8(block, space), _mm_cmpeq_epi8(block, tab)),
        _mm_or_si128(_mm_cmpeq_epi8(block, feed), _mm_cmpeq_epi8(block, ret)));
    /* Every lane matched means every octet is whitespace, so the mask is all ones. Anything
     * else means the run ends inside this block, and the scalar tail finds where. */
    if (_mm_movemask_epi8(ws) != 0xFFFF) return at;
  }
  return at;
}

__attribute__((target("avx2"))) size_t whitespace_run_avx2(const uint8_t *data, size_t len) {
  const __m256i space = _mm256_set1_epi8(static_cast<char>(kSpace));
  const __m256i tab = _mm256_set1_epi8(static_cast<char>(kTab));
  const __m256i feed = _mm256_set1_epi8(static_cast<char>(kLineFeed));
  const __m256i ret = _mm256_set1_epi8(static_cast<char>(kReturn));
  size_t at = 0;
  for (; at + 32 <= len; at += 32) {
    __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + at));
    __m256i ws = _mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(block, space), _mm256_cmpeq_epi8(block, tab)),
        _mm256_or_si256(_mm256_cmpeq_epi8(block, feed), _mm256_cmpeq_epi8(block, ret)));
    if (_mm256_movemask_epi8(ws) != -1) return at;
  }
  /* Finish with the narrower width rather than dropping straight to scalar: a 31-octet tail
   * is the ordinary case at the end of an indented run. */
  return at + whitespace_run_sse2(data + at, len - at);
}
#endif

#if defined(BCIR_INDEX_ARM)
size_t whitespace_run_neon(const uint8_t *data, size_t len) {
  const uint8x16_t space = vdupq_n_u8(kSpace);
  const uint8x16_t tab = vdupq_n_u8(kTab);
  const uint8x16_t feed = vdupq_n_u8(kLineFeed);
  const uint8x16_t ret = vdupq_n_u8(kReturn);
  size_t at = 0;
  for (; at + 16 <= len; at += 16) {
    uint8x16_t block = vld1q_u8(data + at);
    uint8x16_t ws = vorrq_u8(vorrq_u8(vceqq_u8(block, space), vceqq_u8(block, tab)),
                             vorrq_u8(vceqq_u8(block, feed), vceqq_u8(block, ret)));
    /* No movemask on NEON. `vceqq_u8` sets a matching lane to 0xFF, so an all-whitespace
     * block has MINIMUM 0xFF -- one horizontal reduce settles all sixteen. */
    if (vminvq_u8(ws) != 0xFF) return at;
  }
  return at;
}
#endif

/* Tier dispatch, then the scalar finish that makes the bound exact. */
size_t whitespace_run(bcir_jer_simd_tier tier, const uint8_t *data, size_t len, size_t pos) {
  const uint8_t *from = data + pos;
  size_t span = len - pos;
  size_t run;
  switch (tier) {
#if defined(BCIR_INDEX_X86)
    case BCIR_JER_SIMD_AVX2: run = whitespace_run_avx2(from, span); break;
    case BCIR_JER_SIMD_SSE2: run = whitespace_run_sse2(from, span); break;
#endif
#if defined(BCIR_INDEX_ARM)
    case BCIR_JER_SIMD_NEON: run = whitespace_run_neon(from, span); break;
#endif
    default: run = 0; break;
  }
  while (run < span && is_space(from[run])) run++;
  return run;
}

}  // namespace

namespace {

bcir_jer_status index_scan(bcir_jer_simd_tier tier, const uint8_t *data, size_t len,
                           const bcir_jer_limits *limits, bcir_jer_level *stack,
                           size_t stack_entries, uint64_t *nodes, bcir_jer_diag *diag) {
  bcir_jer_scan_cursor cursor;
  size_t pos = 0;
  uint32_t depth = 0;
  uint64_t counted = 0;
  bcir_jer_status st;

  clear(diag);
  if (nodes != nullptr) *nodes = 0;
  if (limits == nullptr) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (data == nullptr && len != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (stack == nullptr && limits->depth != 0)
    return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (stack_entries < limits->depth)
    return fail(diag, BCIR_JER_OVERFLOW, BCIR_JER_NO_OFFSET,
                static_cast<uint64_t>(limits->depth));
  if (static_cast<uint64_t>(len) > limits->input_bytes)
    return fail(diag, BCIR_JER_INPUT_TOO_LARGE, 0, static_cast<uint64_t>(len));

  bcir_jer_scan_begin(&cursor, limits, diag);

  while (pos < len) {
    uint8_t byte = data[pos];

    if (is_space(byte)) {
      /* The one place this loop differs in SHAPE from the scalar rail: a run of whitespace is
       * charged in ONE call rather than one octet at a time. That is the step a vector pass
       * replaces, so it has to be exact rather than merely cheaper.
       *
       * 7.4 is what makes it exact. The charge is uniform and positional -- octet k of the
       * run is the (work + k + 1)th unit -- so with ceiling `L` the first octet to exceed is
       * k = L - work, and the scalar rail would report exactly that offset with
       * `needs = L + 1`. Both are closed form, so the bulk path reproduces them by
       * arithmetic instead of by re-walking the run.
       *
       * Getting this wrong is quiet: reporting the run's START would still refuse the same
       * documents and still hand a caller the wrong octet, which is precisely the failure
       * 4.2's offset contract exists to prevent. */
      size_t run = whitespace_run(tier, data, len, pos);
      uint64_t ceiling = limits->work;
      if (static_cast<uint64_t>(run) > ceiling - cursor.work) {
        size_t failing = static_cast<size_t>(ceiling - cursor.work);
        return bcir_jer_scan_spend(&cursor, (ceiling - cursor.work) + 1, pos + failing);
      }
      st = bcir_jer_scan_spend(&cursor, static_cast<uint64_t>(run), pos);
      if (st != BCIR_JER_OK) return st;
      pos += run;
      continue;
    }

    st = bcir_jer_scan_spend(&cursor, 1, pos);
    if (st != BCIR_JER_OK) return st;

    if (byte == '{' || byte == '[') {
      depth++;
      if (depth > limits->depth) return fail(diag, BCIR_JER_DEPTH_EXCEEDED, pos, depth);
      stack[depth - 1].count = 0;
      stack[depth - 1].is_object = static_cast<uint8_t>(byte == '{');
      stack[depth - 1].state = 0;
      counted++;
      if (counted > limits->nodes) return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      pos++;
      continue;
    }
    if (byte == '}' || byte == ']') {
      if (depth == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      if (stack[depth - 1].is_object != static_cast<uint8_t>(byte == '}'))
        return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      depth--;
      pos++;
      continue;
    }
    if (byte == ',') {
      uint64_t cap;
      if (depth == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      stack[depth - 1].count++;
      cap = stack[depth - 1].is_object ? limits->members : limits->elements;
      if (stack[depth - 1].count + 1 > cap)
        return fail(diag,
                    stack[depth - 1].is_object ? BCIR_JER_MEMBERS_EXCEEDED
                                               : BCIR_JER_ELEMENTS_EXCEEDED,
                    pos, stack[depth - 1].count + 1);
      pos++;
      continue;
    }
    if (byte == ':') {
      pos++;
      continue;
    }
    if (byte == '"') {
      size_t end = pos;
      st = bcir_jer_scan_string_token(data, len, pos, &cursor, &end);
      if (st != BCIR_JER_OK) return st;
      pos = end;
      counted++;
      if (counted > limits->nodes) return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      continue;
    }
    if (byte == '-' || is_digit(byte)) {
      size_t end = pos;
      st = bcir_jer_scan_number_token(data, len, pos, &cursor, &end);
      if (st != BCIR_JER_OK) return st;
      pos = end;
      counted++;
      if (counted > limits->nodes) return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      continue;
    }
    {
      /* `true`, `false`, `null` -- and nothing else. Refusing here rather than downstream is
       * what keeps the non-JSON `NaN` and `Infinity` literals out of the bounded path. */
      size_t taken = bcir_jer_scan_literal_token(data, len, pos);
      if (taken == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      st = bcir_jer_scan_spend(&cursor, static_cast<uint64_t>(taken), pos);
      if (st != BCIR_JER_OK) return st;
      pos += taken;
      counted++;
      if (counted > limits->nodes) return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
    }
  }
  if (depth != 0) return fail(diag, BCIR_JER_MALFORMED, len, 0);
  if (nodes != nullptr) *nodes = counted;
  return BCIR_JER_OK;
}

}  // namespace

extern "C" bcir_jer_status bcir_jer_index_scan(const uint8_t *data, size_t len,
                                               const bcir_jer_limits *limits,
                                               bcir_jer_level *stack, size_t stack_entries,
                                               uint64_t *nodes, bcir_jer_diag *diag) {
  return index_scan(bcir_jer_simd_tier_available(), data, len, limits, stack, stack_entries,
                    nodes, diag);
}

extern "C" bcir_jer_status bcir_jer_index_scan_at(bcir_jer_simd_tier tier, const uint8_t *data,
                                                  size_t len, const bcir_jer_limits *limits,
                                                  bcir_jer_level *stack, size_t stack_entries,
                                                  uint64_t *nodes, bcir_jer_diag *diag) {
  /* A tier this build did not compile, or this CPU does not advertise, degrades to scalar
   * rather than faulting -- J5's "no unsupported-CPU fault" clause, and the reason a caller
   * who pins a width still gets an answer instead of a question about the machine. */
  if (!bcir_jer_simd_tier_compiled(tier) || tier > bcir_jer_simd_tier_available())
    tier = BCIR_JER_SIMD_SCALAR;
  return index_scan(tier, data, len, limits, stack, stack_entries, nodes, diag);
}
