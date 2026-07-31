/*===- fuzz_emit.c - libFuzzer target for the plan-driven encoder ----------===
 *
 * The thirteenth libFuzzer target, and the second whose DESCRIPTOR is fuzzed as well as its
 * input -- fuzz_oer.c was the first, for the same reason. A plan whose declared member
 * counts, tags and element flags disagree with the value stream is exactly the shape that
 * walks a cursor out of bounds, and it is reachable whenever a descriptor and a value come
 * from different places. For anything that compiles a schema once and encodes many values,
 * that is always.
 *
 * The input is split: the first octet chooses the encoding rules and how much of the
 * remainder is plan text, and the rest is the value stream. Splitting rather than fuzzing
 * one of the two in isolation is what lets the fuzzer find disagreements BETWEEN them, which
 * is where the interesting bugs live.
 *
 * Deliberately small caps for nodes, members, scratch and output: a target that can always
 * allocate its way out never exercises the SHORT paths, and those paths do arithmetic on
 * attacker-influenced sizes.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_emit.h"

#define FUZZ_NODES 64
#define FUZZ_MEMBERS 64
#define FUZZ_CONSTRAINTS 64
#define FUZZ_SCRATCH 256
#define FUZZ_OUT 4096

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static bcir_emit_node nodes[FUZZ_NODES];
  static bcir_emit_member members[FUZZ_MEMBERS];
  static bcir_emit_constraint constraints[FUZZ_CONSTRAINTS];
  static uint32_t scratch[FUZZ_SCRATCH];
  static uint8_t out[FUZZ_OUT];
  bcir_emit_plan plan;
  bcir_emit_diag diag;
  bcir_emit_rules rules;
  size_t plan_len, written = 0;
  uint8_t control;

  if (size < 2) return 0;
  control = data[0];
  rules = (bcir_emit_rules)(control & 0x03u);
  data++;
  size--;

  /* Split the rest between descriptor and value. The split is attacker-chosen, so a plan
   * that describes more than the stream carries -- and one that describes less -- are both
   * reachable rather than merely imaginable. */
  plan_len = ((size_t)(control >> 2) * size) / 64u;
  if (plan_len > size) plan_len = size;

  if (bcir_emit_parse_plan((const char *)data, plan_len, nodes, FUZZ_NODES, members,
                           FUZZ_MEMBERS, constraints, FUZZ_CONSTRAINTS, &plan,
                           &diag) != BCIR_EMIT_OK)
    return 0;

  /* The return value is deliberately ignored: every status is a legitimate answer here.
   * What must hold is that none of them wrote outside a buffer, which the sanitizers
   * decide -- not this function. */
  (void)bcir_emit(&plan, rules, data + plan_len, size - plan_len, out, sizeof(out), &written,
                  scratch, FUZZ_SCRATCH, 32, &diag);

  /* A success that claims more octets than the buffer holds would mean the OUT_SHORT path
   * had already overrun; assert it by reading the reported count back. */
  if (written > sizeof(out)) __builtin_trap();
  return 0;
}
