# 07 — Custom types, attributes, and metadata

BCIR lowering succeeds when each custom type and attribute has an explicit fate.
This lesson separates semantic facts from diagnostics and optimization hints.

## Classify before lowering

| Fact | Question | Usual fate |
| --- | --- | --- |
| Semantic | Would removing it change behavior? | Lower to operands, memory, calls, or ABI data. |
| Diagnostic | Is it needed to explain or audit behavior? | Preserve as metadata, debug info, or side-table IDs. |
| Optimization hint | May the compiler ignore it? | Preserve as metadata/intrinsic hint or drop after use. |
| Planning-only | Is it needed only to pick a lowering path? | Drop after the pass that consumes it. |

## Custom attributes

For every custom attribute on a BCIR op, a conversion pattern should do one of
these explicitly:

- copy it to the replacement op if the target dialect accepts it;
- translate it to LLVM metadata;
- encode it in a descriptor field or global table;
- pass it to a runtime wrapper;
- delete it with a comment or test proving it was planning-only.

## Claim and diagnostic metadata

Claim IDs should be stable across lowering. Good anchors include:

- instruction metadata on the replacement memory operation or call;
- named metadata listing graph/claim relationships;
- descriptor fields carrying IDs to runtime wrappers;
- debug locations for source spans.

Do not attach all metadata only to temporary bridge operations. If the bridge op
is erased, the diagnostic link is erased with it.

## LLVM metadata versus executable semantics

LLVM optimizers may drop, merge, or ignore many metadata attachments. Therefore:

- metadata is appropriate for diagnostics and non-binding hints;
- operands/calls/memory are required for behavior;
- wrappers are useful when a fact is both executable and diagnosable.

This rule is especially important for HAM hints and register prelocks. A prelock
that affects correctness must be represented as a table lookup, pointer, or ABI
call, even if metadata also names the original claim.

## Round-trip expectations

LLVM dialect examples should translate to valid LLVM IR. Textual LLVM IR
examples should assemble and pass `opt -passes=verify`. If an example uses
unregistered BCIR dialect syntax, document that it is a syntax-level sketch and
verify only with `mlir-opt --allow-unregistered-dialect`.
