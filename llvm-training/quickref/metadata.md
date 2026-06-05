# Quickref: Metadata

## Rules of thumb

- Metadata annotates IR; it is not a substitute for required types, control flow, memory semantics, or ABI fields.
- Optimizers may preserve, rewrite, merge, or drop metadata depending on the transform and metadata kind.
- Debug locations should describe the transformed instruction honestly; stale locations are worse than absent locations.
- Profile, loop, TBAA, alias-scope, and BCIR diagnostic metadata must be small, intentional, and reviewable.

## Common metadata families

| Family | Use | Caution |
| --- | --- | --- |
| `!dbg` / DI nodes | Source locations and debug structure. | Drop or update when code motion/rewrite invalidates the location. |
| `!prof` | Branch weights and profile guidance. | Stale weights can mislead optimization. |
| `!llvm.loop` | Loop vectorization/unroll/interleave hints. | Hints do not override legality. |
| TBAA / alias scopes | Alias-analysis facts. | Incorrect facts can permit invalid memory reordering. |
| Module flags | Module-wide ABI/debug/profile contracts. | Linking incompatible flags can fail or change semantics. |
| BCIR diagnostics | Lowering provenance and review evidence. | Keep core correctness independent of metadata. |

## Preservation checklist

- When cloning or splitting instructions, decide whether each metadata attachment still applies.
- When merging control flow, recompute branch weights or remove them.
- When changing memory type, layout, or address space, revisit TBAA and alias-scope metadata.
- Run a verifier pass and inspect optimized output if a downstream pass depends on metadata surviving.

## Deep links

- [`../06-metadata/README.md`](../06-metadata/README.md)
- [`../06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md)
- [`../06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md)
- [`../06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md)
- [`../bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md)
