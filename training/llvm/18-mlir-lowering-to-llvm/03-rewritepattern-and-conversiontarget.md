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

## The same pattern, written as data

Everything above is C++: a `RewritePattern` is compiled, linked into the driver, and
from the outside it is opaque. The `pdl` dialect states the same match-and-replace as
*IR* — a value the compiler can read, store, ship and inspect like any other module.
[`examples/pdl-pattern-as-data.mlir`](examples/pdl-pattern-as-data.mlir) is the
declarative form of one of the smallest useful rewrites, `x + 0` becomes `x`:

```mlir
pdl.pattern @add_zero : benefit(1) {
  %type = pdl.type : i32
  %zero_attr = pdl.attribute = 0 : i32
  %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%type : !pdl.type)
  %zero = pdl.result 0 of %zero_op
  %lhs = pdl.operand : %type
  %add = pdl.operation "arith.addi"(%lhs, %zero : !pdl.value, !pdl.value) -> (%type : !pdl.type)

  pdl.rewrite %add {
    pdl.replace %add with (%lhs : !pdl.value)
  }
}
```

There is no control flow in it. `pdl.operation`, `pdl.operand`, `pdl.result`,
`pdl.attribute` and `pdl.type` describe the *shape* being looked for; `pdl.rewrite`
names the operation the match is rooted at and holds what to do once it succeeds. This
is the pattern-level counterpart of the IRDL projection in
[`../22-bcir-approach/04-dialect-as-data.md`](../22-bcir-approach/04-dialect-as-data.md),
which makes the same move for a dialect's *definition*.

## What that pattern compiles into

The declarative form is not what runs. `--convert-pdl-to-pdl-interp` lowers it into the
`pdl_interp` dialect: a control-flow graph in which every block asks one question. It is
worth reading once, because it is the clearest picture available of what a pattern costs
and of how much a pattern driver does on your behalf.

```mlir
pdl_interp.func @matcher(%arg0: !pdl.operation) {
  %0 = pdl_interp.get_operand 1 of %arg0
  %1 = pdl_interp.get_defining_op of %0 : !pdl.value
  pdl_interp.is_not_null %1 : !pdl.operation -> ^bb2, ^bb1
^bb1:  // 18 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ...
  pdl_interp.finalize
^bb2:  // pred: ^bb0
  pdl_interp.check_operation_name of %arg0 is "arith.addi" -> ^bb3, ^bb1
^bb3:  // pred: ^bb2
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb4, ^bb1
```

Three things in that output are absent from the source pattern, and each is a fact about
matching rather than about this example:

- **Checks nobody wrote.** `pdl_interp.check_operand_count`,
  `pdl_interp.check_result_count`, `pdl_interp.is_not_null` and `pdl_interp.are_equal`
  are all generated. Because `%lhs` and the `arith.addi` result were declared with the
  same `%type`, the compiler emits `pdl_interp.get_value_type` on each and an equality
  check between them. In a C++ pattern every one of these is a line the author can
  forget, and forgetting one is a crash rather than a failed match.
- **The order is the compiler's.** The first question asked is not "is this an
  `arith.addi`" — it is whether operand 1 has a defining operation at all. Predicate
  ordering is a compilation decision here, which is precisely what a declarative pattern
  buys and a hand-written `matchAndRewrite` cannot give you.
- **Failure has one exit.** Every failing edge branches to the same block, whose whole
  body is `pdl_interp.finalize`; the block comment records how many predecessors reach
  it. The matcher is a shared decision tree, not a list of patterns tried in turn — which
  is the reason to compile patterns together at all.

Matching and rewriting also come apart. The matcher never rewrites; it ends in
`pdl_interp.record_match @rewriters::@add_zero(...) : benefit(1)`, and the rewrite itself
is a second function in a nested module whose body is `pdl_interp.replace`. The
`benefit(1)` written in the source survives into the recorded match, because benefit is
what the driver sorts by when two patterns match the same operation — the same
`PatternBenefit` that orders the C++ patterns above.

## Pitfall: expecting a stock pass to apply the pattern

`--convert-pdl-to-pdl-interp` *compiles* a pattern; nothing in a release `mlir-opt` runs
one. Applying PDL needs a driver — a `PDLPatternModule` handed to the greedy rewriter or
to a conversion driver — and that is C++ a project links for itself. So the honest
statement for a corpus like this one is that the compile step is checked here and the
apply step is not; an example claiming to show `x + 0` folding away through PDL would be
demonstrating a pass that does not exist.

That is the same shape as the limitation in
[`../20-clang-frontend/07-emitc-the-other-direction.md`](../20-clang-frontend/07-emitc-the-other-direction.md):
the dialect ships, the last hop to the final artifact needs a tool the release does not
carry, and saying so costs less than a fixture that quietly tests something else.

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

## Checks

[`examples/pdl-pattern-as-data.mlir`](examples/pdl-pattern-as-data.mlir) is registered in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json) as a conversion
fixture: [`../tools/verify-mlir-examples.sh`](../tools/verify-mlir-examples.sh) runs
`--convert-pdl-to-pdl-interp` over it and requires the result to contain
`pdl_interp.func @matcher`, `pdl_interp.check_operation_name`,
`pdl_interp.check_operand_count` and `pdl_interp.finalize` — and to contain no
`pdl.pattern`, `pdl.rewrite` or `pdl.replace`. Requiring both halves is the point: a
conversion that quietly left the declarative form in place, or one that emitted an empty
module, satisfies a requirement list on its own and fails the pair.
