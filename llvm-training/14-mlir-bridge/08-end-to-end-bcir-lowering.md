# 14.8 — End-to-End BCIR Lowering

This walkthrough ties the MLIR bridge together with a small BCIR graph fragment.
The goal is not to define a production dialect; it is to show the evidence an
agent should preserve when reviewing a BCIR-to-LLVM lowering.

## 1. Source graph fact

A graph edge connects two vertex IDs and carries an integer weight. At the BCIR
dialect level, the graph relationship is explicit:

```mlir
%edge = "bcir.edge.lookup"(%graph, %src, %dst) : (!bcir.graph, i64, i64) -> !bcir.edge
%weight = "bcir.attribute.read"(%edge) {name = "weight"} : (!bcir.edge) -> i32
```

The verifier can still ask domain questions: does the graph schema define a
`weight` attribute, can the edge kind carry it, and is the lookup total or
fallible?

## 2. Descriptor lowering

After type conversion, the graph is a pointer to a runtime descriptor and the
edge is a pointer to a three-field record:

```mlir
%edge_desc = func.call @bcir_edge_lookup(%graph_ptr, %src, %dst)
  : (!llvm.ptr, i64, i64) -> !llvm.ptr
%weight = func.call @bcir_edge_weight(%edge_desc) : (!llvm.ptr) -> i32
```

This is the point where ABI drift is easiest to introduce. The descriptor schema
must match every runtime module that defines or consumes `@bcir_edge_weight`.

## 3. LLVM dialect shape

The call-based form may lower to direct loads if the descriptor layout is known:

```mlir
%field = llvm.getelementptr %edge_desc[0, 2]
  : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64, i32)>
%weight = llvm.load %field : !llvm.ptr -> i32
```

The integer field index is now part of the ABI. If the BCIR schema changes, the
lowering, runtime, and tests must change together.

## 4. Final LLVM IR

[`examples/bcir-final.ll`](examples/bcir-final.ll) is the final standalone LLVM
IR fixture for this path. It stores edge records as `%bcir.edge = type { i64,
i64, i32 }` and implements a checked weight load from an array of descriptors.
The file is intentionally simple so `llvm-as` and `opt -passes=verify` can check
it without a full MLIR toolchain.

## Review questions

1. Which BCIR facts are still visible at each pipeline stage?
2. Where did graph topology become pointer arithmetic?
3. Which operation made the descriptor layout an ABI contract?
4. Are missing edges represented by a branch, a sentinel, or undefined behavior?
5. Which final LLVM attributes or metadata would help later passes reason about
   aliasing, alignment, and branch likelihood?

If an agent cannot answer those questions from the checked examples, the MLIR
bridge has lost too much information for reliable BCIR review.
