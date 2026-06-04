# Exercise 035: Diagnose a missing symbol relocation

## Task family

This is a **backend/JIT review** exercise. Diagnose why object emission or JIT
linking fails when lowered IR references a runtime symbol that is not available.

## Scenario

A module contains:

```llvm
declare void @bcir_runtime_touch(ptr)

define void @kernel(ptr %ctx) {
entry:
  call void @bcir_runtime_touch(ptr %ctx)
  ret void
}
```

The JIT reports an unresolved external symbol for `bcir_runtime_touch`.

## Required review points

Explain how to distinguish IR verification success from link-time symbol
resolution, where the missing definition can be supplied, and what naming or
visibility mistakes to check.

## Verification command

Markdown review exercise; no LLVM assembler command is required. The checked-in
reference answer is:

```sh
cat llvm-training/exercises/035-diagnose-missing-symbol-relocation.solution.md
```
