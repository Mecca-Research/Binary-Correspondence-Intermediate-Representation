# 03 — `RewritePattern`, `ConversionPattern`, and `ConversionTarget`

Conversion quality depends on matching the rewrite mechanism to the legality
contract. This lesson connects the pattern APIs with BCIR-specific failure modes.

## Pattern roles

| Pattern kind | Best use |
| --- | --- |
| `RewritePattern` | Pre-conversion cleanup, canonicalization, or rewrites that do not need converted operands. |
| `ConversionPattern` | Generic conversion when the operation class is dynamic or shared across dialects. |
| `OpConversionPattern<OpT>` | Typed operation lowering with converted operands and BCIR-specific diagnostics. |

`OpConversionPattern` should be the default for operations such as
`bcir.graph`, `bcir.register_prelock`, `bcir.ham_hint`, and
`bcir.gaadmsf.transfer`.

## A BCIR operation rewrite checklist

For each operation pattern, document:

1. Source op and target replacement operations.
2. Result type mapping supplied by the `TypeConverter`.
3. Attribute mapping: copied, translated, materialized into data, or retired.
4. Metadata mapping for claim IDs, graph IDs, and diagnostics.
5. Whether the replacement is legal under the same `ConversionTarget`.

If item 5 is false, the pass must either run another conversion pattern in the
same driver or explicitly choose partial conversion.

## Conversion target structure

A final BCIR-to-LLVM-dialect target often looks conceptually like this:

```c++
ConversionTarget target(ctx);
target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                       func::FuncDialect, BuiltinDialect>();
target.addIllegalDialect<bcir::BCIRDialect>();
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return converter.isSignatureLegal(op.getFunctionType());
});
```

The exact legal dialect set depends on the phase. An affine-staging pass may
make affine and vector legal while keeping LLVM dialect out of scope; a final
translation pass should require every non-LLVM staging op to lower away.

## Pitfall: marking illegal ops legal too early

Marking `bcir.graph` legal because a pattern is not ready hides the missing
lowering from the conversion driver. Instead:

- leave the op illegal;
- use partial conversion if the pass is intentionally incomplete;
- add a dynamic legality predicate only when the operation has a well-defined
  post-pass role.

## Pitfall: replacement ops that are not legal

A pattern that rewrites `bcir.gaadmsf.transfer` to a custom
`bcir.runtime_call_placeholder` has only moved the problem unless the placeholder
is legal for this phase and illegal for the final phase. Bridge operations are
acceptable, but their lifetime must be explicit.

## Pitfall: unregistered syntax in examples

Training examples may use generic or unregistered dialect syntax to demonstrate
shape without requiring a compiled BCIR dialect. Such examples should be checked
with `mlir-opt --allow-unregistered-dialect` and documented as sketches, not as
verifier-complete dialect tests.
