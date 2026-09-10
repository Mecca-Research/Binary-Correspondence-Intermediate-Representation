# Foundations: LLVM IR and SSA

## Key takeaways

- LLVM IR is a typed, SSA-based compiler IR that sits between frontends, optimizers, and backends.
- Named temporaries are single-assignment values; changing a value means producing a new SSA name.
- `phi` nodes merge values from predecessor blocks and are the first control-flow shape to verify when rewriting IR.
- IR is not assembly: preserve semantic facts such as types, dominance, and undefined/poison behavior instead of modeling bytes only.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| What LLVM IR is and how modules/functions/basic blocks fit together | [`01-what-is-llvm-ir.md`](01-what-is-llvm-ir.md) |
| SSA naming, dominance, and PHI-node merge rules | [`02-ssa.md`](02-ssa.md) |
| How IR differs from assembly and from other intermediate representations | [`03-ir-vs-asm-vs-other-irs.md`](03-ir-vs-asm-vs-other-irs.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
