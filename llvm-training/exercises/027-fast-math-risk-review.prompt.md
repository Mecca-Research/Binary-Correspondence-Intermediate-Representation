# Exercise 027: Review fast-math flag risk

## Task family

This is a **review** exercise. Judge whether fast-math flags are safe for a
numeric lowering that crosses a BCIR or MLIR boundary.

## Candidate IR to review

```llvm
define float @score(float %a, float %b, float %c) {
entry:
  %sum = fadd fast float %a, %b
  %scaled = fmul reassoc nnan ninf float %sum, %c
  ret float %scaled
}
```

## Required review points

Explain:

- Which behavior `fast`, `reassoc`, `nnan`, and `ninf` allow optimizers to assume.
- Why these flags are risky when NaN, infinity, signed zero, or exact evaluation
  order matters.
- How to document a safe lowering when source semantics really do allow the
  flags.

## Expected observation

A good answer recommends either dropping flags or preserving only those backed by
source-language, BCIR, or MLIR operation semantics.

## Verification command

Markdown review exercise; no LLVM assembler command is required. The checked-in
reference answer is:

```sh
cat llvm-training/exercises/027-fast-math-risk-review.solution.md
```
