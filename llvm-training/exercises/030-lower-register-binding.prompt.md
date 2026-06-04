# Exercise 030: Lower BCIR register binding

## Task family

This is a **BCIR lowering** exercise. Resolve logical BCIR register identifiers
through a runtime binding table before loading or storing payload data.

## Required LLVM constructs

Write a standalone module containing:

- `define i64 @lower_register_binding(ptr %binding_table, i32 %logical_id)`.
- Zero-extension from a logical 32-bit register ID to an index.
- A table lookup that loads a payload pointer.
- A final load from the resolved payload pointer.

## Expected observation

The module should assemble and show that logical BCIR registers are not LLVM SSA
registers; they are data-driven bindings resolved through memory.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/030-lower-register-binding.solution.ll
```
