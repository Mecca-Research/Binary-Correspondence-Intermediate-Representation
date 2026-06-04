# 14.7 — Pass Pipeline and Diagnostics

A BCIR MLIR pipeline should be staged so each pass has a small, checkable
contract. Agents should not treat `mlir-opt ... | mlir-translate` as an opaque
black box; they should inspect the IR after each boundary where information is
lost.

## Suggested staged pipeline

```text
bcir-parse
  -> bcir-canonicalize
  -> bcir-verify-schema
  -> bcir-lower-graph-to-descriptors
  -> convert-scf-to-cf
  -> convert-func-to-llvm
  -> finalize-memref-to-llvm
  -> reconcile-unrealized-casts
  -> mlir-translate --mlir-to-llvmir
```

The exact pass names depend on the implementation, but the ordering constraints
matter:

- Canonicalize and verify while BCIR operations are still visible.
- Lower graph topology before erasing schema attributes that describe it.
- Convert structured control flow only after loop/branch invariants are checked.
- Reconcile casts at the end; leftover `builtin.unrealized_conversion_cast`
  usually means a missing type materialization rule.

## Inspection checkpoints

| Checkpoint | What to inspect |
|---|---|
| After `bcir-canonicalize` | Normalized edge directions, deduplicated attributes, stable symbol names. |
| After descriptor lowering | No semantic BCIR ops remain except explicitly legal markers. |
| After `convert-scf-to-cf` | Blocks/branches match structured loop and if-region intent. |
| After LLVM conversion | Types are `!llvm.*`, integer widths are concrete, calls are declared. |
| After LLVM IR translation | `.ll` verifies with `llvm-as` and `opt -passes=verify`. |

## Debugging commands

Use these as templates when an implementation provides the named passes:

```bash
mlir-opt input.mlir \
  -bcir-canonicalize \
  -bcir-lower-graph-to-descriptors \
  -mlir-print-ir-after-all \
  -mlir-disable-threading

mlir-opt input.mlir \
  -pass-pipeline='builtin.module(func.func(bcir-lower-graph-to-descriptors))'
```

Print-after-all is noisy but valuable when a custom conversion pattern drops an
attribute or creates an illegal cast. For concise review, keep small fixtures in
`14-mlir-bridge/examples/` that isolate one graph fragment.

## Failure triage

- **Parser failure**: check operation spelling, dialect registration, and custom
  assembly forms.
- **Verifier failure before conversion**: the BCIR dialect invariant is wrong or
  the fixture is intentionally invalid.
- **Conversion legality failure**: add or narrow a rewrite pattern.
- **Unrealized cast remains**: type converter is incomplete.
- **LLVM translation failure**: an LLVM dialect op has an unsupported type,
  missing symbol, or invalid region/control-flow shape.
- **LLVM verifier failure**: inspect the final `.ll`; the issue is now in LLVM
  IR semantics, not just MLIR syntax.

The final check should always include the canonical LLVM example verifier when a
lowering emits `.ll` files into this training corpus.
