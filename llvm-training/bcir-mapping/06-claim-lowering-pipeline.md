# Claim Lowering Pipeline

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/kernel/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


BCIR claim lowering is the repeatable path from semantic operation records to
verifier-valid LLVM IR. The important discipline is to lower one concern at a
time: claim fields, resource binding, operation dispatch, scheduling metadata,
and diagnostics should not be hidden inside one opaque helper.

## BCIR-level meaning

- A claim identifies an operation, its read/write resources, hazard domain,
  lane information, immediates, and optional cost or provenance records.
- The lowering pipeline binds symbolic resource IDs before it emits the actual
  operation body or runtime wrapper.
- Schedules, batches, and phases choose order; they should not change claim
  semantics while lowering one claim.
- Diagnostics need stable claim IDs after field packing and ABI conversion.

## Recommended lowering stages

1. **Normalize** symbolic resources, lane masks, op names, and immediate values
   into a canonical claim record.
2. **Pack** fixed-width claim fields using the schema in
   `runtime/llvm/bcir_claim_schema.ll`.
3. **Bind** reads and writes by lowering resource IDs to registry-table GEPs and
   base-pointer loads.
4. **Dispatch** each operation either to plain LLVM instructions or to an
   explicit runtime-call wrapper.
5. **Stabilize** poison-capable speculative predicates with `freeze` before they
   control branches, selects, vector masks, or metadata-preserving if-conversion;
   see the [BCIR `freeze` safe-speculation rule][bcir-freeze-rule].
6. **Attach** advisory metadata, prefetches, debug/provenance tags, and verifier
   assertions only after the core dataflow is explicit.

## Example source and lowered IR

- Source-like input: [`examples/claim-resource-lookup.bcir.txt`](examples/claim-resource-lookup.bcir.txt)
- Checked output: [`examples/claim-resource-lookup.ll`](examples/claim-resource-lookup.ll)
- Runtime-wrapper source: [`examples/bcir-operation.prompt.md`](examples/bcir-operation.prompt.md)
- Checked wrapper output: [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll)
- Safe-speculation example:
  [`../13-advanced-ir/examples/bcir-freeze-safe-speculation.ll`](../13-advanced-ir/examples/bcir-freeze-safe-speculation.ll)

## Verifier commands

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
```

## Verifier risks

- Nested expressions such as `load (getelementptr ...)` are invalid; every step
  must be an instruction with its own result.
- Claim struct indexes must match the canonical claim schema or the verifier may
  accept IR that the runtime interprets incorrectly.
- Operation wrappers must agree with declarations on return type, pointer type,
  attributes, and argument order.
- Runtime-call declarations should be declarations in example modules unless the
  example owns the implementation.
- If the pipeline speculates a poison-capable predicate into `br`, `switch`,
  `select`, or a vector mask without `freeze`, the IR may verify but still have
  undefined behavior for some BCIR inputs.

## Optimization risks

- Inlining can blur the boundary between claim decoding and operation execution;
  keep wrappers named and small for debugging.
- DCE can remove advisory-only calls if they are modeled as side-effect-free.
- CSE can merge equivalent address calculations and make diagnostics harder to
  associate unless metadata or naming conventions are preserved.

[bcir-freeze-rule]: ../13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze
