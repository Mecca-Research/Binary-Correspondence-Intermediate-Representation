# Exercise 037: Review TableGen-to-MCInst lowering

## Task family

This is a **backend review** exercise. Trace how an instruction definition should
flow from TableGen records through instruction selection or lowering into an
`MCInst` suitable for encoding.

## Scenario

A target adds a pseudo instruction `BCIR_PREFETCHrr` with operands `(base,
offset)` and expects the assembler printer to emit a real prefetch instruction,
but generated code still contains an unexpanded pseudo.

## Required review points

Explain the roles of TableGen instruction records, pseudo expansion, register
classes, operand types, `MCInst` opcodes, and encoder/asm-printer coverage.

## Verification command

Markdown review exercise; no LLVM assembler command is required. The checked-in
reference answer is:

```sh
cat llvm-training/exercises/037-tablegen-to-mcinst-review.solution.md
```
