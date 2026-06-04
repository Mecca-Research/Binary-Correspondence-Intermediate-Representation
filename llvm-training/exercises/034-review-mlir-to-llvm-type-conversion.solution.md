# Solution 034: MLIR-to-LLVM type conversion review

Converting `index` to `i32` is only safe when the target data layout and ABI say
that MLIR index values fit in 32 bits. Many 64-bit targets lower `index` to
`i64`, so this choice needs explicit target evidence.

Lowering `memref<?xf32>` to a bare `ptr` erases rank, offset, size, and stride
information. A descriptor is usually needed unless the ABI separately supplies
all shape information. Lowering `!bcir.graph` to a bare `ptr` may be acceptable
at a runtime ABI boundary, but internal lowered code often needs a graph
descriptor containing vertex bases, edge bases, field offsets, counts, and
strides. `vector<4xf32>` to `<4 x float>` is structurally natural, but the pass
must still respect target vector legality and alignment.
