# EVAL

Use these self-check questions after reading the MLIR bridge path:

1. Which BCIR facts should remain as MLIR operations or attributes before descriptor lowering?
2. When should `!bcir.vertex` lower to an integer ID versus an LLVM pointer?
3. Why should materialization operations be searchable and temporary?
4. Which legality stage should reject a surviving `bcir.attribute.read`?
5. What does a leftover `builtin.unrealized_conversion_cast` usually indicate?
6. Where does `bcir-final.ll` encode the edge weight as an LLVM struct field?
7. Which ABI contract breaks if `%bcir.edge = type { i64, i64, i32 }` changes?
8. Which command verifies the final LLVM IR without requiring an MLIR toolchain?
