# Control Flow: Branches, Switches, Calls, Returns, and Comparisons

## Key takeaways

- Every basic block needs one terminator, and every edge must be reflected in PHI incoming block labels.
- Conditional branches split on an `i1`; do not embed nested instruction expressions inside branch conditions.
- `switch` encodes multiway integer dispatch and still requires well-formed target blocks and defaults.
- `indirectbr` is specialized and target-sensitive; prefer direct branches unless modeling low-level dispatch tables.
- A `call` is *not* a terminator; `invoke` and `callbr` are. ABI attributes (`sret`, `byval`, `zeroext`, `signext`) must match on the declaration and every call site.
- Signedness lives in the comparison predicate, not in the integer type, and float comparisons must choose ordered or unordered before they choose a relation.
- `select` evaluates both arms: it cannot make a side effect conditional and it does not filter poison.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Unconditional branches and block stitching | [`01-unconditional-br.md`](01-unconditional-br.md) |
| Conditional branches and boolean-producing instructions | [`02-conditional-br.md`](02-conditional-br.md) |
| Switch syntax, default edges, and lowering expectations | [`03-switch.md`](03-switch.md) |
| Indirect branch address tables and verifier constraints | [`04-indirectbr.md`](04-indirectbr.md) |
| Calls, calling conventions, ABI attributes, tail-call markers, and returns | [`05-call-and-ret.md`](05-call-and-ret.md) |
| `icmp`/`fcmp` predicates, vector masks, and branchless `select` | [`06-comparisons-and-select.md`](06-comparisons-and-select.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
