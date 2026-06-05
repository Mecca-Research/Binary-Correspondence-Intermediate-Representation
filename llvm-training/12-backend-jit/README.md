# Backend and JIT: Codegen, TableGen, ORC, MC, and Relocations

## Key takeaways

- The backend consumes optimized LLVM IR through target lowering, instruction selection, scheduling, register allocation, and MC emission.
- TableGen source describes target facts; generated `*Gen*.inc` files normally live in the build tree, not the source tree.
- ORC/LLJIT layers separate symbol definition, lookup, materialization, and object linking responsibilities.
- Relocation and MC issues are usually symbol/layout problems, so preserve target triple, data layout, and object ownership evidence.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Code generation pipeline and target lowering checkpoints | [`01-codegen-pipeline.md`](01-codegen-pipeline.md) |
| TableGen files, generated includes, and target descriptions | [`02-tablegen.md`](02-tablegen.md) |
| ORC/LLJIT concepts and missing-symbol diagnostics | [`03-orc-jit.md`](03-orc-jit.md) |
| MC emission, symbols, and relocations | [`04-mc-and-relocations.md`](04-mc-and-relocations.md) |
| ORC layers, object ownership, and diagnostic flow | [`05-orc-layers.md`](05-orc-layers.md) |
| Custom BCIR intrinsic lowering, stackmaps/patchpoints, runtime fallback, and JIT policy | [`06-custom-bcir-intrinsics.md`](06-custom-bcir-intrinsics.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
