# Solution 033: MLIR graph op to LLVM dialect lowering

A reasonable lowering first converts `!bcir.graph` to a descriptor pointer or an
LLVM dialect struct value containing base pointers and strides. The `index`
vertex operand is converted to the configured LLVM index integer type. The pass
then projects the vertex-field table for `rank`, computes the byte or element
offset for `%vertex`, and emits LLVM-dialect `llvm.getelementptr` plus
`llvm.load` operations with the correct result type and alignment.

BCIR metadata such as field name, graph schema version, or placement hints should
not be used as a substitute for executable address computation. If downstream
tools need the field identity, the lowering may attach non-semantic metadata to
the load or record a named module-level catalog, while preserving the executable
contract in ordinary LLVM-dialect pointer arithmetic and memory operations.

When applying the same pattern to the vertex-graph walkthrough in
[`../14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md),
vertex IDs and edge rows remain executable table fields or SSA values. Optional
register-binding preferences should not be modeled as portable physical-register
requirements; they can lower to binding slots, runtime calls, target hooks, or be
dropped if advisory. Metadata hints may be attached to related loads/calls or
recorded in named metadata, but the lowered program must still be correct if an
optimizer ignores them.
