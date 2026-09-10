# Agent template: review adversarial LLVM IR

## Role

You are reviewing an LLVM IR fixture designed to pass one layer of validation
while violating, stressing, or obscuring another semantic contract. Do not equate
parser/verifier acceptance with lowering correctness.

## Inputs to fill in

- **Fixture and declared class**: `<IR plus adversarial-class marker>`
- **Source/BCIR invariant**: `<behavior or 1:1 mapping that must hold>`
- **Pipeline**: `<passes, lowering stages, or call reconstruction>`
- **Target contract**: `<triple, data layout, features, ABI, address spaces>`
- **Required evidence**: `<metadata, bundles, debug records, stable BCIR IDs>`

## Review procedure

1. Confirm that the declared class predicts the correct tool outcome: accepted
   but risky, intentionally invalid, target-specific, or metadata-preservation.
2. Inventory poison sources and sensitive uses; memory effects and
   `memory(...)` claims; address spaces; ABI/calling-convention attributes;
   varargs assumptions; operand bundles; debug information; and BCIR IDs.
3. Compare the inventory before and after every transform that rewrites values,
   calls, memory operations, CFG edges, or metadata.
4. Separate generic LLVM validity from target legality and BCIR mapping validity.
5. Identify the smallest counterexample and the first pipeline stage that makes
   the invariant unprovable.
6. Recommend a repair and a stable regression oracle.

## Required output

- Classification verdict and observed tool result.
- Semantic verdict: `safe`, `unsafe`, or `insufficient evidence`.
- A before/after invariant table.
- The minimal reproducer, including seed/target/pipeline metadata when fuzzed.
- A deterministic assertion or a reason that the check must remain review-only.

## Verification checklist

- `llvm-as` success is reported as syntax/verifier evidence, not correctness.
- Target-specific code generation is gated by triple and features.
- Required metadata, operand bundles, and BCIR IDs survive reduction.
- ABI attributes and `memory(...)` effects are compared across declaration,
  definition, and call site.
- The proposed regression catches the semantic failure without relying on
  unstable whole-file formatting.
