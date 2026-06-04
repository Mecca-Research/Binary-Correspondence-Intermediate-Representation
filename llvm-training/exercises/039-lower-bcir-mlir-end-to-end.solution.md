# Solution 039: BCIR MLIR end-to-end lowering

Canonical BCIR should replace friendly schema names with stable keys while the
custom graph concepts are still visible. The root remains vertex identity in the
`claim` space with fixed `i64` ID `0`. The `contains` edge becomes edge key `3`,
and the `weight` attribute becomes attribute key `7` with result type `f32`.

In a runtime-backed LLVM-dialect lowering, the graph itself is carried as an
opaque `!llvm.ptr`, the root identity is an `i64`, and schema keys are `i32`
constants. The child lookup has the shape:

```mlir
%child = llvm.call @bcir_lookup_child(%graph, %root_id, %edge_key)
  : (!llvm.ptr, i64, i32) -> !llvm.ptr
```

The attribute lookup has the shape:

```mlir
%weight = llvm.call @bcir_get_edge_attr_f32(%graph, %root_id, %attr_key)
  : (!llvm.ptr, i64, i32) -> f32
```

In textual LLVM IR, the custom vertex type is erased. Its surviving carriers are
the root `i64` ID, the child `ptr` returned by `@bcir_lookup_child`, the graph
`ptr`, and the numeric schema keys used by ABI calls. The final calls look like:

```llvm
%child = call ptr @bcir_lookup_child(ptr %graph, i64 0, i32 3)
%weight = call float @bcir_get_edge_attr_f32(ptr %graph, i64 0, i32 7)
```
