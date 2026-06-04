# Exercise 034: Review MLIR-to-LLVM type conversion

## Task family

This is an **MLIR bridge review** exercise. Audit a lowering plan for type
conversion hazards before converting to LLVM IR.

## Candidate conversion table

| Source type | Candidate LLVM representation |
| --- | --- |
| `index` | `i32` |
| `memref<?xf32>` | `ptr` |
| `!bcir.graph` | `ptr` |
| `vector<4xf32>` | `<4 x float>` |

## Required review points

Explain which conversions need target data-layout evidence, which erase shape or
ownership information, and what auxiliary descriptor fields may be needed.

## Verification command

Markdown review exercise; no MLIR tool is required. The checked-in reference
answer is:

```sh
cat llvm-training/exercises/034-review-mlir-to-llvm-type-conversion.solution.md
```
