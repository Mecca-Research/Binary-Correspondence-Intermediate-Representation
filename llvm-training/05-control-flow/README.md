# Control Flow: Branches, Switches, and Indirect Branches

## Key takeaways

- Every basic block needs one terminator, and every edge must be reflected in PHI incoming block labels.
- Conditional branches split on an `i1`; do not embed nested instruction expressions inside branch conditions.
- `switch` encodes multiway integer dispatch and still requires well-formed target blocks and defaults.
- `indirectbr` is specialized and target-sensitive; prefer direct branches unless modeling low-level dispatch tables.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Unconditional branches and block stitching | [`01-unconditional-br.md`](01-unconditional-br.md) |
| Conditional branches and boolean-producing instructions | [`02-conditional-br.md`](02-conditional-br.md) |
| Switch syntax, default edges, and lowering expectations | [`03-switch.md`](03-switch.md) |
| Indirect branch address tables and verifier constraints | [`04-indirectbr.md`](04-indirectbr.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
