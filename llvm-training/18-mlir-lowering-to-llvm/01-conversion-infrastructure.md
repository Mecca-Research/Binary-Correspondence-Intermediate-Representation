# 01 — MLIR conversion infrastructure

This lesson names the moving parts used by the rest of the chapter. MLIR
conversion is a legality-driven rewrite framework: you describe which operations
are allowed to remain, provide patterns for the rest, and choose how strict the
conversion must be.

## `ConversionTarget`

A `ConversionTarget` is the policy object for a lowering boundary. It answers:

- which dialects are legal after this pass;
- which operations are illegal and must be rewritten;
- which operations are dynamically legal only when their operands, result types,
  or attributes already satisfy a predicate.

For BCIR, a target might declare the LLVM, arith, memref, func, vector, and
builtin dialects legal while marking `bcir.graph`, `bcir.vertex`,
`bcir.register_prelock`, and `bcir.gaadmsf.*` illegal.

## Dynamic legality

Dynamic legality is the escape hatch for staged lowering. An operation can remain
only if it already satisfies a lowering invariant. Common BCIR examples:

- `func.func` is legal only after all argument/result types have converted to
  LLVM-compatible or ABI-compatible types.
- A bridge op such as `bcir.claim_anchor` is legal only until metadata has been
  attached to all replacement operations.
- A memory operation is legal only if a layout attribute has been translated to a
  concrete stride/offset representation.

Dynamic legality is safer than blanket legality because it lets the conversion
framework report exactly which unconverted operation still violates the contract.

## Partial conversion

Partial conversion applies patterns to illegal operations but permits explicitly
legal and dynamically legal operations to remain. Use it while bringing up BCIR
lowering in layers:

```text
bcir.graph + bcir.vertex
  -> affine/scf/vector staging ops + bcir.claim_anchor
  -> LLVM dialect + metadata
```

Partial conversion is also useful when the pass intentionally lowers only one
slice, such as register prelocks, and leaves graph traversal for a later pass.

## Full conversion

Full conversion requires every operation in the conversion region to be legal
under the target after rewrites finish. Use full conversion at hard boundaries:

- before translating LLVM dialect to LLVM IR;
- before handing the module to an LLVM-only pass pipeline;
- in tests that prove no BCIR operations remain after the final lowering pass.

A common mistake is to move to full conversion before the `TypeConverter` and
materializations are complete. That makes diagnostics noisy because many
unrelated bridge values fail at once.

## `RewritePatternSet`

`RewritePatternSet` owns the patterns that can be applied by conversion. Keep
pattern sets grouped by lowering layer:

- BCIR graph-to-affine/vector patterns;
- BCIR custom-type-to-LLVM patterns;
- BCIR register/HAM/GAADMSF-to-runtime patterns;
- generic arith, func, memref, vector, and scf lowering patterns.

Layering pattern sets makes it clear which pass owns which semantic decision and
reduces accidental rewrites across BCIR boundaries.

## Conversion pattern families

- `ConversionPattern` is the generic base for rewriting operations while using a
  `TypeConverter` and converted operands.
- `OpConversionPattern<OpT>` is the typed helper for a specific operation class;
  prefer it for real BCIR operations because it gives typed accessors and clearer
  diagnostics.
- Plain rewrite patterns can still be useful before conversion, but conversion
  patterns are the right tool when operands or results may change type.

## BCIR review questions

Before approving a conversion pass, ask:

1. Which BCIR operations are illegal at this boundary?
2. Which operations are dynamically legal, and what invariant do they prove?
3. Is this pass partial because another pass owns the remaining BCIR ops, or is
   it full because LLVM IR translation is next?
4. Does every pattern preserve claim IDs, graph IDs, HAM hints, and diagnostics
   before erasing the source op?
