# Advanced IR: Intrinsics, Attributes, Poison, and Fast Math

## Key takeaways

- Intrinsics are strongly typed contracts; many require exact operand types, immediate arguments, or target feature availability.
- Attributes and calling-convention details affect optimization and ABI lowering, so copy them only when the contract remains true.
- `undef`, `poison`, and `freeze` are distinct; use `freeze` before control-flow decisions that may observe poison.
- Fast-math flags trade IEEE guarantees for optimization freedom and should be attached only when the source semantics allow them.
- Operand bundles are call-site semantic payloads; preserve them when cloning or rewriting calls.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Common LLVM intrinsics and declaration patterns | [`01-common-intrinsics.md`](01-common-intrinsics.md) |
| Target-specific intrinsics and feature constraints | [`02-target-specific-intrinsics.md`](02-target-specific-intrinsics.md) |
| Special types such as token and metadata-sensitive constructs | [`03-special-types-and-tokens.md`](03-special-types-and-tokens.md) |
| Function, parameter, return, and call-site attributes | [`04-attributes.md`](04-attributes.md) |
| Poison, undef, freeze, and UB-safe speculation | [`05-poison-undef-freeze.md`](05-poison-undef-freeze.md) |
| Fast-math flags and floating-point optimization contracts | [`06-fast-math-flags.md`](06-fast-math-flags.md) |
| Operand bundles on `call` and `invoke` | [`07-operand-bundles.md`](07-operand-bundles.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.

## Adversarial fixtures

Use the [adversarial exercise track](../exercises/adversarial/) to stress poison
control flow, operand-bundle preservation, target intrinsic constraints, ABI
attributes, `memory(...)` claims, and varargs assumptions. The track requires a
semantic verdict in addition to LLVM parser/verifier results.
