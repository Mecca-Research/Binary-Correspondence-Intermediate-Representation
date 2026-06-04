# Exercise 011: Lower logical register IDs to bound storage pointers

## BCIR concept being modeled

Model the BCIR register binding pattern where a claim stores logical read and
write register IDs, and a lowering pass resolves those IDs through a binding
table before performing memory operations. Use a compact claim schema shaped like
`{ i64 control, [4 x i32] rd_rids, [4 x i32] wr_rids }`.

Write a standalone LLVM IR module that defines:

```llvm
define i32 @bcir.exercise.copy_bound_register(ptr %claim, ptr %register_table)
```

The function should load `rd_rids[0]` and `wr_rids[0]`, use both IDs as indices
into `register_table` (an array of pointers), load an `i32` from the read-bound
pointer, store it to the write-bound pointer, and return the copied value.

## Required LLVM constructs

- A named `%bcir.claim.binding` struct containing two nested `[4 x i32]` arrays.
- Two-level `getelementptr inbounds` operations to access array elements inside
  the struct.
- Pointer-table lookup using `getelementptr ptr, ptr %register_table, ...`.
- `zext` from `i32` register IDs to `i64` pointer-table indices.
- A scalar `load` and `store` through the bound pointers.

## Expected verification command

```sh
llvm-as -disable-output llvm-training/exercises/011-register-binding-pattern.solution.ll
```

## Expected observation

The module assembles successfully. The learner should observe that logical BCIR
register references are not LLVM registers themselves; they lower to table
lookups that produce concrete memory pointers.

## Optional runtime reference

Compare this with the read/write register accessor pattern in
`runtime/llvm/bcir_claim_accessors.ll`.
