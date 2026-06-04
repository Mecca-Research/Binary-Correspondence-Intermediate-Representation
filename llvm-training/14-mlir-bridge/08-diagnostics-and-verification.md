# 14.8 — Diagnostics and Verification

A lowering bridge is only trustworthy if it fails early and explains why. BCIR
has graph topology, attributes, HAM hints, register/resource bindings, and ABI
calls; each of those can be malformed independently. Verifiers should reject
invalid source IR before conversion, while conversion diagnostics should explain
why a valid source operation cannot be represented at the requested target level.

## Verifier layers

| Layer | Examples of checks | Failure style |
|---|---|---|
| Parser / importer | Required attributes exist, custom type syntax is well formed | Hard error with source location |
| Operation verifier | Edge endpoint spaces match edge type, HAM confidence is in range | Hard error on the operation |
| Symbol/schema verifier | Attribute names resolve, runtime ABI symbol exists, graph space is known | Error referencing both use and definition |
| Canonicalization verifier | Normalized edge kind and resource IDs are internally consistent | Error before destructive rewrite |
| Conversion legality | No illegal BCIR ops remain for the target stage | Pass failure with op location |
| LLVM dialect verifier | LLVM types, calls, branches, and attrs are legal | MLIR verifier failure |
| LLVM IR verifier | Textual `.ll` assembles and passes `opt -passes=verify` | Tool failure with line/IR context |

## Diagnostic principles

- Prefer **specific operation errors** over generic pass failure messages.
- Report the source fact that cannot be lowered: graph space, edge kind,
  attribute key, register class, target feature, or ABI version.
- Distinguish optional hints from required semantics. Dropping an unsupported
  optional HAM hint can be a remark; dropping a required resource binding is an
  error.
- Preserve locations through rewrites so final LLVM-dialect errors can point
  back to the BCIR operation that caused them.
- Include the selected ABI or descriptor version in diagnostics when schema
  drift is possible.

## Suggested verifier checks by concept

| Concept | Verifier questions |
|---|---|
| Vertex identity | Does the ID width match the declared graph space? Are constants in range? Are dynamic IDs typed as fixed-width values rather than accidental `index` values? |
| Edge topology | Do source/destination vertex spaces match the edge type? Is direction canonical? Does the edge kind exist in the schema? |
| Attributes | Does the attribute key exist for this edge or vertex kind? Does the result type match the schema? Is mutable runtime state represented as operands/calls rather than immutable MLIR attributes? |
| HAM hints | Is the policy known? Is confidence in range? Is distance non-negative? Is the hint optional unless a runtime scheduling contract says otherwise? |
| Register/resource binding | Is the register/resource class known for the target? Does the value type fit the class? Is `required = true` backed by a lowering path? |
| Runtime ABI calls | Does the callee declaration match the ABI table? Are pointer, integer, and descriptor widths target-correct? Are memory effects and calling convention documented? |

## What survives lowering: runtime ABI calls

| Stage | Representation | What survives | What may be lost |
|---|---|---|---|
| BCIR dialect | symbolic runtime op or high-level operation selected for runtime ownership | Operation intent, schema names, verifier context | Nothing if ABI selection has not run |
| Canonical BCIR | resolved callee symbol, numeric keys, normalized operands | Stable ABI identity and argument intent | Friendly operation aliases |
| Mid-level MLIR | `func.call`/call-like op with converted types | Call boundary and typed arguments | Domain-specific op type |
| LLVM dialect | `llvm.func` declaration and `llvm.call` | ABI spelling, concrete LLVM types, calling convention attrs | BCIR semantic names except symbols/metadata |
| LLVM IR | `declare` and `call` | Linkable runtime boundary | MLIR symbol-table/verifier context |

## Example diagnostics

Good diagnostic shape:

```text
error: cannot lower required BCIR register binding to x86_64 LLVM dialect
  op: bcir.bind_register required=true reg_class="vec_pred"
  reason: target feature table has no vec_pred class
  note: optional bindings may be dropped, but required bindings need a backend or ABI lowering
```

Poor diagnostic shape:

```text
error: conversion failed
```

The first message tells a user whether to change the source, target, or lowering
configuration. The second only tells them that the bridge is a black box.

## Verification workflow

For documentation examples that include final `.ll`, run the repository example
verifier:

```sh
llvm-training/tools/verify-examples.sh
```

For markdown-only exercises, the exercise verifier checks that reference answers
exist and that all checked LLVM IR solutions assemble and verify:

```sh
llvm-training/tools/verify-exercises.sh
```

MLIR snippets in this chapter are illustrative unless a future harness adds
`mlir-opt` to the training verifier. When adding executable MLIR tests, document
the exact pass pipeline and make absence of MLIR tools a clean skip rather than a
silent pass.

## See also

- [`07-pass-pipeline.md`](07-pass-pipeline.md) — pipeline stages where diagnostics should fire.
- [`examples/bcir-final.ll`](examples/bcir-final.ll) — final LLVM IR artifact checked by the example verifier.
- [`../exercises/038-review-mlir-diagnostic-plan.prompt.md`](../exercises/038-review-mlir-diagnostic-plan.prompt.md) — diagnostic review exercise.
