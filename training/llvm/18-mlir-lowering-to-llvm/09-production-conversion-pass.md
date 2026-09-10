# A Production Conversion Pass

The reference implementation for this chapter is
[`../../mlir/lib/passes/BCIRConvertToLLVM.cpp`](../../../mlir/lib/passes/BCIRConvertToLLVM.cpp),
a conversion pass this repository builds and tests in CI. Every API named below
is checked against it by
[`../tools/verify-mlir-rail-references.py`](../tools/verify-mlir-rail-references.py).

## The five pieces, and what each one decides

| Piece | Decides |
| --- | --- |
| `TypeConverter` | what each source type *becomes* |
| `ConversionTarget` | what is legal *after* the pass |
| `RewritePatternSet` | how each op is rewritten |
| `applyPartialConversion` / `applyFullConversion` | whether leftovers are allowed |
| materialization hooks | how to bridge a converted and an unconverted value |

Getting the *legality* wrong is the most common failure, and it does not look
like a legality problem: the pass reports "failed to legalize operation", names
an op you did not write a pattern for, and the actual cause is a target
declaration that made something illegal without a pattern to fix it.

## Skeleton

```cpp
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

struct ComputeOpLowering : public OpConversionPattern<ComputeOp> {
  using OpConversionPattern<ComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // adaptor.getX() gives the ALREADY-CONVERTED operands.
    // op.getX() gives the originals -- using them here is the classic bug.
    Value lowered = LLVM::AddOp::create(rewriter, op.getLoc(),
                                        adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

void ConvertToLLVMPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  LLVMTypeConverter converter(ctx);

  LLVMConversionTarget target(*ctx);
  target.addLegalOp<ModuleOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<ComputeOpLowering, BarrierOpLowering>(converter, ctx);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
```

### `adaptor` versus `op`

This is the rule that separates a conversion pattern from an ordinary rewrite
pattern. During dialect conversion, operands may already have been replaced by
values of their *converted* types. `adaptor.getLhs()` is the converted value;
`op.getLhs()` is the original, which may be about to be erased.

Reading through `op` in a conversion pattern produces IR that verifies at the
moment it is built and fails later, when the original value disappears. The
symptom appears far from the cause.

## `TypeConverter`

```cpp
converter.addConversion([](BCIRHandleType type) -> std::optional<Type> {
  return LLVM::LLVMPointerType::get(type.getContext());
});
```

- Conversions are tried **most recently added first**, so a later registration
  shadows an earlier one for the same type.
- Returning `std::nullopt` means "I do not handle this type"; returning a
  failure means the conversion is impossible.
- A converter that does not handle a type leaves it unconverted — which makes
  every op using it illegal, with an error that names the op rather than the
  type.

### Materializations are the bridge, not an optimization

```cpp
converter.addSourceMaterialization(...);   // converted value -> original type
converter.addTargetMaterialization(...);   // original value  -> converted type
```

They exist for the boundary where converted and unconverted regions meet — a
partial conversion, a block argument the framework must re-type, a pattern that
ran before its neighbour. Without them, partial conversion fails at exactly the
boundaries you intended to leave alone. See
[`02-typeconverter-and-materialization.md`](02-typeconverter-and-materialization.md).

## Partial versus full conversion

| | `applyPartialConversion` | `applyFullConversion` |
| --- | --- | --- |
| Leftover illegal ops | allowed if legal by the target | failure |
| Use when | lowering one dialect among several; staged pipelines | the pass must fully eliminate a dialect |
| Honest failure mode | reports what remains | reports the first thing it could not legalize |

This repository's `bcir-aot` deliberately uses the partial form and **may leave
residual BCIR/GEM operations**. That is a design decision, not an incomplete
implementation: a pass that pretends to have fully lowered something it did not
produces an artifact whose remaining work is invisible. A partial pass that
reports its residue is auditable.

## The greedy rewriter is a different tool

```cpp
if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
  return signalPassFailure();
```

Use the greedy driver for **canonicalization-shaped** work: local rewrites,
folding, cleanups, no type changes, no legality target. Use dialect conversion
when types change or when "what is legal afterwards" is the point. Mixing them
— running a greedy pass and calling it a lowering — produces a conversion with
no legality contract, which is exactly the thing that later fails silently.

## API migration: what changed between MLIR 22 and 23

This repository moved its rail to LLVM/MLIR 23, and the migration was almost
entirely two mechanical renames. They are worth knowing because they affect
every conversion pattern ever written:

| Old (≤22) | New (22 and 23) |
| --- | --- |
| `builder.create<OpTy>(loc, args...)` | `OpTy::create(builder, loc, args...)` |
| `rewriter.create<OpTy>(loc, args...)` | `OpTy::create(rewriter, loc, args...)` |
| `applyPatternsAndFoldGreedily(...)` | `applyPatternsGreedily(...)` |

Both new spellings exist in MLIR 22, and the old ones are gone or deprecated in
23 — so the migration can be done *before* the upgrade, on a tree that still
builds against the older release. That is the general shape of a safe MLIR
version move: adopt the new spelling while both work, then change the pin.

Three further habits that make MLIR upgrades survivable:

- **Pin one coherent major.** Libraries, `mlir-tblgen`, and the tool must all
  come from the same release; the C++ ABI is not stable across majors.
- **Build the tool with the compiler that built the libraries.** In this
  repository, a `bcir-opt` built with the system compiler against
  differently-built MLIR archives crashed at startup.
- **Expect ODS to move too.** Op definitions are not insulated from release
  changes — for example, a symbol op that once carried an SSA result may stop
  being allowed to, forcing a real design change rather than a rename.

## Debugging a conversion

```bash
bcir-opt --bcir-convert-to-llvm input.mlir              # the pass
bcir-opt --debug-only=dialect-conversion input.mlir     # per-op legality trace
bcir-opt --mlir-print-ir-after-all input.mlir           # IR between passes
bcir-opt --mlir-print-op-generic input.mlir             # exact op structure
bcir-opt --verify-diagnostics negative.mlir             # pin the diagnostic
mlir-translate --mlir-to-llvmir lowered.mlir            # LLVM dialect -> LLVM IR
```

`--debug-only=dialect-conversion` (on a debug build) prints why each operation
was considered legal or illegal and which pattern was tried. It is the fastest
route from "failed to legalize" to the missing pattern or the over-broad target
rule.

## Pitfalls

- **Reading `op` instead of `adaptor`.** The defining error of this API.
- **Marking a dialect illegal with no pattern for one of its ops.** Legality
  and patterns must be declared together.
- **Forgetting to register a dialect the pass *emits*.** Failure happens at op
  creation, not at pass startup.
- **Skipping materializations in a partial conversion.** Fails exactly at the
  boundary you meant to preserve.
- **`applyFullConversion` where the design is staged.** Produces a hard failure
  for a situation the pipeline intended.
- **Using the greedy driver as a lowering.** No legality contract.
- **Mixing MLIR majors.** Link succeeds, behaviour does not.
- **Landing a lowering with no negative fixture.** MLIR's `expected-error`
  makes the refusal itself testable; a conversion with no test for what it
  *refuses* has an untested half.

## BCIR notes

- Conversion is where a BCIR claim becomes a realization. The legality question
  was already answered by the verifier passes; the conversion pass must not
  re-open it, and must not consult a cost or a measurement to decide whether to
  fire.
- Residual BCIR/GEM ops after `bcir-aot` are a *reported* state. Treat the
  residue as the pass's output, not as a defect to hide behind a full
  conversion.
- Metadata that a lowering consumes must be dropped explicitly by that lowering,
  the same discipline as
  [`../17-new-pass-manager/06-building-an-out-of-tree-pass.md`](../17-new-pass-manager/06-building-an-out-of-tree-pass.md).
  Metadata left behind after the contract it describes is gone will be believed
  by the next pass.

## See also

- [`08-production-dialect-and-build-integration.md`](08-production-dialect-and-build-integration.md) — building and registering the dialect
- [`03-rewritepattern-and-conversiontarget.md`](03-rewritepattern-and-conversiontarget.md) — the concepts in isolation
- [`examples/production-dialect/README.md`](examples/production-dialect/README.md) — the standalone skeleton
- [`../../mlir/README.md`](../../../mlir/README.md) — the production rail
