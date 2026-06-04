# Exercise 036: Identify an ORC JIT layer failure

## Task family

This is a **backend/JIT review** exercise. Classify an ORC JIT failure by the
layer most likely responsible.

## Scenario

A kernel parses and verifies, but `lookup("kernel")` fails after object emission.
Logs show that the IR transform layer ran, object generation ran, and symbol
lookup in the selected `JITDylib` returned no definition.

## Required review points

Identify likely failures in the compile layer, object/link layer, and symbol
resolution layer. Recommend the first logs or probes to add.

## Verification command

Markdown review exercise; no LLVM assembler command is required. The checked-in
reference answer is:

```sh
cat llvm-training/exercises/036-identify-orc-layer-failure.solution.md
```
