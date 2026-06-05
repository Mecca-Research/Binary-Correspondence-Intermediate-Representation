# MLIR Lowering to LLVM for BCIR

This chapter is the dedicated lowering companion to
[`../14-mlir-bridge/`](../14-mlir-bridge/). The bridge chapter explains why a
frontend or BCIR-like domain model may enter MLIR; this chapter explains how to
make the lowering boundary explicit, auditable, and friendly to LLVM IR tools.

## Key takeaways

- Treat lowering as a contract, not a syntax rewrite: every surviving BCIR fact
  must have a chosen LLVM representation before the source operation disappears.
- Use MLIR conversion infrastructure deliberately: `ConversionTarget` declares
  legality, `RewritePatternSet` owns rewrites, and `TypeConverter` owns type and
  boundary materialization.
- Prefer partial conversion while developing a new BCIR lowering. Switch to full
  conversion only after all illegal BCIR operations, custom types, and temporary
  bridge operations are covered.
- Attach claim IDs, graph IDs, HAM hints, and diagnostic metadata before graph
  identity is flattened into structs, tables, calls, or LLVM metadata.
- The LLVM dialect is close to LLVM IR, but not identical. Verify that emitted
  LLVM dialect can translate to valid textual LLVM IR and round-trip through the
  normal LLVM verifier.

## Lesson map

1. [`01-conversion-infrastructure.md`](01-conversion-infrastructure.md) —
   legality, conversion modes, and rewrite containers.
2. [`02-typeconverter-and-materialization.md`](02-typeconverter-and-materialization.md) —
   type conversion and source, target, and argument materialization.
3. [`03-rewritepattern-and-conversiontarget.md`](03-rewritepattern-and-conversiontarget.md) —
   conversion pattern anatomy and legality pitfalls.
4. [`04-bcir-dialect-to-llvm.md`](04-bcir-dialect-to-llvm.md) — BCIR graph,
   register, HAM, GAADMSF, and diagnostic lowering choices.
5. [`05-affine-vector-llvm-lowering-pipeline.md`](05-affine-vector-llvm-lowering-pipeline.md) —
   staged affine/vector/LLVM pipelines.
6. [`06-transform-dialect-for-bcir.md`](06-transform-dialect-for-bcir.md) —
   scripting BCIR lowering strategies with transform dialect handles.
7. [`07-custom-types-attributes-and-metadata.md`](07-custom-types-attributes-and-metadata.md) —
   preserving custom type, attribute, claim, and metadata intent.

## Lowering pipeline overview

A robust BCIR lowering usually follows this staged shape:

```text
BCIR dialect module
  ├─ canonicalize BCIR graph/resource operations
  ├─ attach claim IDs, graph IDs, diagnostic metadata, and HAM policy
  ├─ lower structured graph traversal to affine/scf/vector where profitable
  ├─ lower BCIR custom types through a TypeConverter
  ├─ rewrite register binding to explicit pointer/table lookups
  ├─ rewrite GAADMSF operations to calls, intrinsics, or vector/LLVM ops
  ├─ lower surviving memref/vector/func/arith/scf/affine ops to LLVM dialect
  └─ translate LLVM dialect to LLVM IR, then run llvm-as/opt verification
```

Use [`../17-new-pass-manager/`](../17-new-pass-manager/) after translation when
LLVM IR passes must preserve BCIR metadata or honor custom runtime boundaries.
Use [`../reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md)
when selecting built-in intrinsic declarations for prefetch, vector, memory, or
metadata-related examples.

## What should survive to LLVM IR

| BCIR fact | LLVM-level representation | Why it survives |
| --- | --- | --- |
| Claim ID / proof provenance | `!bcir.claim`, `!annotation`, debug locations, or explicit runtime argument | Needed for diagnostics and post-lowering audits. |
| Vertex and edge identity needed by runtime | Packed structs, index arrays, adjacency tables, or stable metadata nodes | Runtime and debug tools need a way to map lowered code back to graph facts. |
| Register/resource binding | Explicit pointer, resource-table lookup, ABI field, or wrapper-call argument | LLVM register allocation cannot infer BCIR's logical binding contract. |
| HAM hint policy | Custom LLVM metadata, `llvm.prefetch`, or a runtime/intrinsic wrapper | Hints guide placement or prefetch without becoming mandatory semantics. |
| GAADMSF runtime boundary | Direct call, intrinsic wrapper, or vectorized LLVM operation sequence | Data movement must become executable instructions or ABI calls. |
| Diagnostic severity/source span | Debug info, named metadata, or side-table pointer | Required to explain lowered failures after BCIR ops disappear. |

## What should lower away

| BCIR/MLIR construct | Lowered form | Why it should disappear |
| --- | --- | --- |
| `bcir.graph` container op | Functions, loops, structs, globals, and metadata | LLVM IR has no graph container operation. |
| Symbolic vertex/edge attributes used only for planning | Constants, table entries, or removed attributes | Planning-only attributes should not constrain optimization. |
| Custom BCIR value types | LLVM dialect integer, pointer, vector, struct, or memref descriptor types | LLVM IR needs concrete data layout and ABI types. |
| Structured affine/vector staging ops | LLVM dialect loops, vector intrinsics, or scalarized operations | Staging dialects are intermediate optimization forms. |
| Unresolved register prelocks | Loads from resource tables or call arguments | Late codegen needs explicit operands, not abstract prelock claims. |
| Transform dialect scripts | No runtime artifact | Transform dialect controls compilation; it is not program semantics. |

## Examples

- [`examples/bcir-graph-to-affine.mlir`](examples/bcir-graph-to-affine.mlir) —
  graph traversal shaped for affine lowering.
- [`examples/bcir-graph-to-vector.mlir`](examples/bcir-graph-to-vector.mlir) —
  graph edge weights staged as vector operations.
- [`examples/bcir-graph-to-llvm-dialect.mlir`](examples/bcir-graph-to-llvm-dialect.mlir) —
  illustrative LLVM-dialect boundary.
- [`examples/bcir-register-prelock-ham-hints.mlir`](examples/bcir-register-prelock-ham-hints.mlir) —
  source-level prelock and HAM hint shape.
- [`examples/bcir-register-prelock-ham-hints-lowered.ll`](examples/bcir-register-prelock-ham-hints-lowered.ll) —
  standalone LLVM IR with explicit resource lookup, prefetch, and metadata.
- [`examples/bcir-conversion-pass-skeleton.cpp.md`](examples/bcir-conversion-pass-skeleton.cpp.md) —
  implementation skeleton for a BCIR conversion pass.

## Cross-links

- [`../14-mlir-bridge/`](../14-mlir-bridge/) for MLIR basics and the prior
  bridge-level walkthrough.
- [`../bcir-mapping/`](../bcir-mapping/) for direct BCIR-to-LLVM IR lowering
  rules and examples.
- [`../17-new-pass-manager/`](../17-new-pass-manager/) for LLVM IR pass ordering,
  custom analyses, and metadata-preservation policy.
- [`../reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md)
  for intrinsic declaration and usage patterns.

## Pitfalls checklist

- Do not mark BCIR operations legal merely to quiet conversion failures; legal
  means the operation may remain after the selected conversion boundary.
- Do not drop custom attributes during type conversion. Copy, translate, or
  intentionally retire every attribute with a diagnostic note.
- If examples use unregistered dialect syntax, document verifier expectations
  and run MLIR syntax checks with `--allow-unregistered-dialect`.
- Do not lower BCIR graph identity before claim IDs and diagnostic metadata have
  been attached to replacement operations or side tables.
- Do not emit LLVM dialect that cannot translate to valid LLVM IR; always run the
  MLIR translator and the LLVM verifier when tools are available.

## Checks

From the repository root, run:

```sh
./llvm-training/tools/verify-mlir-examples.sh
./llvm-training/tools/verify-examples.sh
```
