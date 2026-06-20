/* Designated initializers for a file-scope dispatch / jump table (#designated): an enum-indexed table
 * with [OP] = value designators, including a gap that zero-fills (§6.7.10). Both rails parse the
 * designated initializer; the table is referenced by name (defined in the source), so the emitted C is
 * Clang-behaviour-equivalent. This is the driver opcode-dispatch / jump-table pattern. */
enum { OP_NOP, OP_INC, OP_DEC, OP_DBL, OP_NEG };
static const unsigned WEIGHT[5] = { [OP_NOP]=0u, [OP_INC]=1u, [OP_DBL]=4u, [OP_NEG]=8u };  /* OP_DEC -> 0 */

unsigned weigh(unsigned op) { return WEIGHT[op % 5u]; }
