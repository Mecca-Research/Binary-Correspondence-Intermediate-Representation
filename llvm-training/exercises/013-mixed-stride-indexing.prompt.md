# Exercise 013: Lower mixed stride indexing to byte offsets

## BCIR concept being modeled

Model the BCIR pattern where a layout profile combines row strides, element
strides, and an additional bias to compute an address. The lowering should make
all mixed units explicit before loading from memory.

Write a standalone LLVM IR module that defines:

```llvm
define i32 @bcir.exercise.load_mixed_stride(ptr %base, i64 %row, i64 %col, i64 %row_stride_bytes, i64 %element_stride_bytes, i64 %bias_bytes)
```

The function should compute:

```text
offset = row * row_stride_bytes + col * element_stride_bytes + bias_bytes
```

Then it should compute `base + offset` as an `i8` pointer GEP, load an `i32`, and
return it.

## Required LLVM constructs

- `mul` and `add` instructions for explicit byte-offset arithmetic.
- `getelementptr i8` for byte-addressed pointer movement.
- A final typed scalar `load i32` from the computed address.
- No `ptrtoint` or `inttoptr`; keep the computation in GEP form.

## Expected verification command

```sh
llvm-as -disable-output llvm-training/exercises/013-mixed-stride-indexing.solution.ll
```

## Expected observation

The module assembles successfully. The learner should observe that a BCIR mixed
stride profile can lower to simple SSA arithmetic followed by a byte-wise GEP.

## Optional runtime reference

Compare this with the byte-offset load/store wrappers in `runtime/llvm/bcir_ops.ll`
and the strided prefetch helper in `runtime/llvm/bcir_prefetch_profiles.ll`.
