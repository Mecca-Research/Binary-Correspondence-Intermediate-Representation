/*===- fuzz_jer.c - libFuzzer entry for the bounded X.697 JER reader --------===
 *
 * The totality contract: for ANY input bytes, every bcir_jer_* entry point returns a status
 * and never reads outside the buffer nor writes outside the caller's output. There is no
 * "valid input" notion here -- these are the functions a C peer runs BEFORE any type is
 * consulted, on octets an attacker chose, so the property under test is that no escape,
 * nesting, number or UTF-8 sequence can walk a cursor out of bounds.
 *
 * WHAT THIS TARGET REACHES THAT THE DIFFERENTIAL DOES NOT. bcir/tests/test_c_jer.py compares
 * the two rails on documents Python can also read. Three things have no Python counterpart
 * and live only here:
 *
 *   - the caller-owned STACK, driven at a depth limit the input itself chooses, so a
 *     document deeper than the array is refused rather than written past;
 *   - the caller-owned SCRATCH, driven with an ample buffer, a deliberately undersized one
 *     and a NULL measuring call -- the measure-then-write path is where an off-by-one hides;
 *   - the SINK, refusing at an input-chosen event index, which unwinds the parse from the
 *     middle of a walk.
 *
 * The limits are derived FROM the input rather than fixed, because a limit that is never
 * reached is a branch that is never fuzzed: J1's ceilings are most of this reader's refusal
 * surface, and hammering one profile would leave the other nine untested.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_jer.h"

#define MAX_DEPTH 64
#define SCRATCH 4096

/* A sink that refuses at a chosen event, and otherwise reads every field it is handed --
 * touching the text is what makes ASan prove the pointer and length are a real region. */
typedef struct fuzz_sink {
  long refuse_at;
  long seen;
  uint64_t checksum;
} fuzz_sink;

static int sink(void *ctx, bcir_jer_event event, size_t offset, const uint8_t *text,
                size_t len) {
  fuzz_sink *state = (fuzz_sink *)ctx;
  size_t at;
  state->checksum += (uint64_t)event + (uint64_t)offset;
  for (at = 0; at < len; at++) state->checksum += text[at];
  if (state->refuse_at >= 0 && state->seen++ == state->refuse_at) return -3;
  return 0;
}

static void drive(const uint8_t *data, size_t len, const bcir_jer_limits *limits,
                  long refuse_at, size_t scratch_cap, size_t stack_entries) {
  static bcir_jer_level stack[MAX_DEPTH];
  static uint8_t scratch[SCRATCH];
  bcir_jer_diag diag;
  uint64_t nodes = 0;
  fuzz_sink state;

  (void)bcir_jer_scan(data, len, limits, stack, stack_entries, &nodes, &diag);
  (void)bcir_jer_validate_utf8(data, len, &diag);

  state.refuse_at = refuse_at;
  state.seen = 0;
  state.checksum = 0;
  (void)bcir_jer_parse(data, len, limits, stack, stack_entries, scratch, scratch_cap,
                       sink, &state, &diag);
  /* And with no sink at all: a pure validity check that builds nothing. */
  (void)bcir_jer_parse(data, len, limits, stack, stack_entries, scratch, scratch_cap,
                       0, 0, &diag);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t buffer[1 << 16];
  bcir_jer_limits limits;
  bcir_jer_diag diag;
  bcir_jer_frame frame;
  size_t len = size > sizeof(buffer) ? sizeof(buffer) : size;
  size_t at;
  size_t written = 0;

  for (at = 0; at < len; at++) buffer[at] = data[at];

  bcir_jer_default_limits(&limits);
  drive(buffer, len, &limits, -1, SCRATCH, MAX_DEPTH);

  if (len > 0) {
    /* A profile the input chose, so every ceiling is reachable. Each is clamped to the
     * default, which is also the "tightened, never expanded" rule the API enforces. */
    bcir_jer_limits chosen;
    bcir_jer_limits accepted;
    bcir_jer_default_limits(&chosen);
    chosen.depth = (uint32_t)(data[0] % (MAX_DEPTH + 1));
    chosen.nodes = data[len / 2] % 64u;
    chosen.members = data[len - 1] % 32u;
    chosen.elements = data[len - 1] % 32u;
    chosen.string_bytes = data[0] % 48u;
    chosen.number_bytes = data[len / 2] % 24u;
    chosen.integer_digits = data[0] % 20u;
    chosen.exponent_magnitude = data[len - 1] % 400u;
    chosen.work = (uint64_t)data[len / 2] * 8u + 1u;
    chosen.input_bytes = len;
    if (bcir_jer_limits_tightened(&limits, &chosen, &accepted) == BCIR_JER_OK) {
      /* A stack SHORTER than the depth the limits allow must be refused, not overrun. */
      drive(buffer, len, &accepted, -1, SCRATCH, MAX_DEPTH);
      drive(buffer, len, &accepted, (long)(data[0] % 16u), 7, MAX_DEPTH);
      drive(buffer, len, &accepted, -1, 0, MAX_DEPTH);
      drive(buffer, len, &accepted, -1, SCRATCH, accepted.depth == 0 ? 0
                                                                    : accepted.depth - 1);
    }
    /* The strict profile, whose small ceilings reach refusal paths the default rarely
     * does -- and whose input_bytes limit the corpus regularly exceeds. */
    bcir_jer_strict_limits(&limits);
    drive(buffer, len, &limits, -1, SCRATCH, MAX_DEPTH);
  }

  /* The escape decoder, directly: the densest code in the reader, and the one holding the
   * surrogate pairing. An ample buffer, an undersized one, and the NULL measuring call. */
  {
    static uint8_t wide[SCRATCH];
    static uint8_t narrow[5];
    (void)bcir_jer_unescape(buffer, len, wide, sizeof(wide), &written, &diag);
    (void)bcir_jer_unescape(buffer, len, narrow, sizeof(narrow), &written, &diag);
    (void)bcir_jer_unescape(buffer, len, 0, 0, &written, &diag);
  }

  /* The UTF-8 scalar decoder at every offset, including the at-end and past-end edges. */
  for (at = 0; at <= len + 1; at++) {
    uint32_t code = 0;
    size_t width = 0;
    (void)bcir_jer_utf8_next(buffer, len, at, &code, &width);
  }

  /* And the frame, whose declared length is attacker-controlled by construction. */
  (void)bcir_jer_unframe(buffer, len, &frame, &diag);
  return 0;
}
