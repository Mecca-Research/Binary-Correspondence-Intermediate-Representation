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

namespace {

/* ECMA-404 clause 4, exactly: SPACE, HORIZONTAL TABULATION, LINE FEED, CARRIAGE RETURN.
 * Nothing else -- and in particular not FORM FEED or VERTICAL TABULATION, which some JSON
 * readers admit and which would let two rails disagree about a document's shape. */
inline bool is_space(uint8_t c) {
  return c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D;
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

/* Where a vector pass will go.
 *
 * The whole accelerable question is "how far does the run of whitespace at `pos` extend",
 * and the answer is positional: every octet in it costs exactly one work unit at its own
 * offset. 7.4 settles what that means -- entering a run of `n` octets with `w` spent against
 * ceiling `L`, the budget fails, when it fails, at offset `L - w` with `needs = L + 1`.
 * Closed form, no re-walk. So a bulk charge is exact and a block-at-a-time scan is sound.
 *
 * It is scalar here on purpose. The equivalence harness has to prove the SEAM is sufficient
 * before the optimization arrives, or a later failure cannot be attributed to one or the
 * other. */
inline size_t whitespace_run(const uint8_t *data, size_t len, size_t pos) {
  size_t run = 0;
  while (pos + run < len && is_space(data[pos + run])) run++;
  return run;
}

}  // namespace

extern "C" bcir_jer_status bcir_jer_index_scan(const uint8_t *data, size_t len,
                                               const bcir_jer_limits *limits,
                                               bcir_jer_level *stack, size_t stack_entries,
                                               uint64_t *nodes, bcir_jer_diag *diag) {
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
      size_t run = whitespace_run(data, len, pos);
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
