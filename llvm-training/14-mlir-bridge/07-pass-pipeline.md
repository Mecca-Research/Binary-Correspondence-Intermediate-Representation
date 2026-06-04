# 14.7 — Pass Pipeline for BCIR to LLVM IR

A BCIR MLIR bridge should lower in stages. Each stage should have a small legal
dialect set, a clear verifier story, and examples that show the same operation
before and after conversion. Avoid a single monolithic pass that jumps directly
from rich BCIR operations to textual LLVM IR; it is harder to diagnose, harder to
optimize, and easier to let semantic facts disappear unnoticed.

## Suggested pipeline

```text
bcir-parse/import
  -> bcir-verify-schema
  -> bcir-canonicalize
  -> bcir-normalize-topology
  -> bcir-materialize-descriptors or bcir-select-runtime-abi
  -> convert-bcir-to-arith-memref-func-cf
  -> canonicalize/cse/symbol-dce
  -> convert-scf-to-cf
  -> convert-memref-to-llvm
  -> convert-func-to-llvm
  -> reconcile-unrealized-casts
  -> llvm-dialect-verify
  -> translate-to-llvmir
  -> llvm-as/opt verification
```

The exact pass names in a production tree may differ, but the ordering principle
is stable: verify and canonicalize while the source concepts are visible, then
lower through increasingly concrete dialects.

## Stage responsibilities

| Stage | Primary job | Typical inputs | Typical outputs |
|---|---|---|---|
| Import / parse | Create valid BCIR dialect IR | BCIR source, JSON, existing IR, frontend AST | `builtin.module` with `bcir.*` ops |
| Schema verification | Reject impossible graph/resource facts | vertices, edges, attributes, ABI symbols | same IR plus diagnostics |
| Canonicalization | Fold aliases and normalize local forms | redundant lookups, duplicate hints | canonical BCIR ops |
| Topology/resource normalization | Resolve edge direction, register/resource symbols | symbolic spaces and names | numeric keys, descriptor refs, canonical edges |
| Descriptor or ABI selection | Choose direct memory vs runtime calls | graph and attribute ops | memref/descriptor ops or ABI-call placeholders |
| Dialect conversion | Remove illegal BCIR ops | `bcir`, `arith`, `memref`, `func`, `cf` | standard/control-flow/LLVM-compatible ops |
| LLVM conversion | Convert types, calls, branches, memory | memref/func/cf/arith | `llvm` dialect |
| Translation | Emit LLVM IR bitcode/text | LLVM dialect module | `.bc` or `.ll` |

## Pipeline branch: descriptor-backed lowering

Descriptor-backed lowering is best when the compiler owns the graph layout:

```text
bcir.graph + bcir.edge + bcir.attribute
  -> memref descriptors for edge targets and attribute tables
  -> arith index calculations and memref.load/store
  -> LLVM descriptor structs, GEPs, loads, and stores
```

This branch exposes more to optimization, but schema changes require the bridge
and runtime to agree exactly on struct layout, alignment, and index width.

## Pipeline branch: runtime-backed lowering

Runtime-backed lowering is best when the runtime owns schema evolution or
scheduling policy:

```text
bcir.graph + bcir.edge + bcir.attribute + bcir.ham_hint
  -> resolved ABI symbols and numeric keys
  -> llvm.call @bcir_lookup_child / @bcir_get_edge_attr_f32 / @bcir_schedule_hint
  -> textual LLVM IR declarations and calls
```

This branch keeps the compiler simpler and more robust across storage changes,
but it hides memory behavior behind calls unless attributes and memory-effect
models are precise.

## What survives lowering: register/resource binding

| Stage | Representation | What survives | What may be lost |
|---|---|---|---|
| BCIR dialect | `bcir.bind_register` or resource-binding attrs | Logical resource, class, preference, required/optional bit | Nothing if verifier checks target/resource names |
| Canonical BCIR | resolved resource IDs and normalized classes | Stable logical IDs and required-vs-optional distinction | Friendly names and aliases |
| Mid-level MLIR | explicit ABI operands, target attrs, or unchanged value plus remark | Required ABI facts if modeled explicitly | Optional preferences unsupported by target |
| LLVM dialect | target intrinsic, inline-asm constraint, call-convention choice, or no-op for optional hint | Enforceable target constraint | Generic BCIR resource vocabulary |
| LLVM IR | inline asm, target intrinsic, ABI signature, or ordinary SSA value | Only constraints expressible in LLVM/backend | Optional preference unless carried in target metadata |

## Pipeline invariants

- No pass after `convert-bcir-*` should need to understand custom BCIR types.
- No required resource or register binding may be dropped without a conversion
  failure diagnostic.
- Every runtime call should be declared once and reused consistently.
- Every descriptor layout should be versioned or tied to a module/schema symbol.
- Every pass that erases a BCIR op should either preserve its source location or
  attach a diagnostic breadcrumb to the replacement.

## Example artifacts

The example directory contains a staged paper trail:

1. [`examples/bcir-dialect-source-sketch.mlir`](examples/bcir-dialect-source-sketch.mlir) — rich source-level dialect sketch.
2. [`examples/bcir-canonicalized.mlir`](examples/bcir-canonicalized.mlir) — canonical BCIR after normalization.
3. [`examples/bcir-lowered-llvm-dialect.mlir`](examples/bcir-lowered-llvm-dialect.mlir) — LLVM dialect shape before translation.
4. [`examples/bcir-final.ll`](examples/bcir-final.ll) — feasible textual LLVM IR equivalent.

## See also

- [`06-conversion-patterns.md`](06-conversion-patterns.md) — pattern contracts used by conversion stages.
- [`09-bcir-mlir-end-to-end.md`](09-bcir-mlir-end-to-end.md) — walk-through of the staged examples.
- [`../bcir-mapping/06-claim-lowering-pipeline.md`](../bcir-mapping/06-claim-lowering-pipeline.md) — direct BCIR-to-LLVM claim lowering.
