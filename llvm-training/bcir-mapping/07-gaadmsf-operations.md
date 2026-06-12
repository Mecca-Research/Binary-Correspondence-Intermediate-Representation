# GAADMSF Operation Lowering

GAADMSF-style operations are graph-aware data movement and scheduling forms:
loads, stores, copies, scatters, reductions, and metadata-only scheduling hints
that operate over graph fragments. LLVM IR should receive explicit memory
addresses, loop/control structure, and calls; it should not invent a hidden graph
operation primitive.

## BCIR-level meaning

- Graph vertices and edges name the logical topology being traversed.
- Attributes carry payloads such as weights, offsets, flags, or costs.
- Data movement operations read and write resources selected by graph positions.
- Scheduling hints may describe locality or traversal order without changing
  the value semantics of the operation.

## Likely LLVM IR representation

- Lower stable graph fragments to struct arrays and field GEPs.
- Lower dynamic graph payload lookup to byte-address calculations followed by a
  typed load or store.
- Lower bulk or target-specific operations to runtime wrappers when the operation
  cannot be expressed clearly as scalar/vector LLVM IR.
- Keep advisory scheduling data in metadata or explicit runtime-side profile
  records.

## Example source and lowered IR

- Source-like graph fragment: [`examples/graph-fragment.bcir.txt`](examples/graph-fragment.bcir.txt)
- Checked struct-array output: [`examples/graph-fragment-struct-gep.ll`](examples/graph-fragment-struct-gep.ll)
- Mixed-stride graph source: [`examples/mixed-stride-graph.bcir.txt`](examples/mixed-stride-graph.bcir.txt)
- Checked byte-offset output: [`examples/mixed-stride-byte-offset.ll`](examples/mixed-stride-byte-offset.ll)

## Verifier commands

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll -o /dev/null
```

## Verifier risks

- Struct field indexes must match the named struct layout in the module.
- Variable GEP indexes into arrays must be integer values of legal type.
- PHI nodes in graph walks must list exactly one incoming value per predecessor.
- `inbounds` is only correct when the lowered graph index is proven in range.

## Optimization risks

- Alias ambiguity between graph arrays and payload arrays can block vectorization.
- Reassociation of byte-offset arithmetic can hide the original row/column
  meaning from diagnostics.
- Incorrect alignment on lowered payload loads can become a target-specific crash
  or silent performance bug.

## Hardware-aware continuation

For programming pulses, flow execution, calibration state, and the choice between metadata, intrinsics, and MIR, continue with [`../19-hardware-aware/README.md`](../19-hardware-aware/README.md).
