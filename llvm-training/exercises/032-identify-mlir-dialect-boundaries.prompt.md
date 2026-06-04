# Exercise 032: Identify MLIR dialect boundaries

## Task family

This is an **MLIR bridge review** exercise. Identify where a mixed-dialect MLIR
module crosses from domain-specific BCIR operations into standard, memref, arith,
func, and LLVM-dialect operations.

## Input sketch

```mlir
func.func @kernel(%g: !bcir.graph, %i: index) -> i32 {
  %v = bcir.vertex_attr %g[%i] : i32
  %c0 = arith.constant 0 : i32
  %p = arith.cmpi sgt, %v, %c0 : i32
  cf.cond_br %p, ^hot, ^cold
^hot:
  %r = bcir.runtime.call @touch(%g) : (!bcir.graph) -> i32
  return %r : i32
^cold:
  return %c0 : i32
}
```

## Required review points

Explain which operations are BCIR-specific, which are generic MLIR operations,
which are control-flow operations, and what must be true before lowering to the
LLVM dialect.

## Verification command

Markdown review exercise; no MLIR tool is required. The checked-in reference
answer is:

```sh
cat llvm-training/exercises/032-identify-mlir-dialect-boundaries.solution.md
```
