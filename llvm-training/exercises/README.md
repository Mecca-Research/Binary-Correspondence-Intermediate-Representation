# Exercises

This directory contains beginner and advanced LLVM IR exercises. The first
exercise family focuses on standalone IR writing; later families cover repair,
review, and optimization-pass reasoning. Follow the conventions in
[`../EXAMPLES.md`](../EXAMPLES.md). Each exercise should include:

- a prompt describing the task;
- the expected command to check the learner's answer or the checked-in solution;
- the expected observation, such as successful assembly or a specific diagnostic;
- an optional standalone `*.solution.ll` file containing one expected answer.

Repair exercises keep broken starting points as `*.invalid.ll.txt` so they remain
visibly intentional failures and do not enter any known-good `.ll` verification
loop. Optimization-pass reasoning exercises may include `*.input.ll` and
`*.after-<pass>.ll` teaching snapshots; these snapshots document expected shape,
not a byte-for-byte contract for every LLVM release.

Solutions must assemble with LLVM >= 15, where opaque pointers are the default.
Use `ptr` for pointer-typed values instead of typed pointers such as `i32*`.

If your LLVM tools are installed with a version suffix, replace `llvm-as` in the
commands with the matching binary, for example `llvm-as-15` or `llvm-as-18`.

## Exercise list

### Standalone IR writing

1. [`001-add.prompt.md`](001-add.prompt.md) — write `@add(i32, i32)`.
2. [`002-if-else-phi.prompt.md`](002-if-else-phi.prompt.md) — write an if/else with `phi`.
3. [`003-loop-counter.prompt.md`](003-loop-counter.prompt.md) — write a loop counter.
4. [`004-global-load-store.prompt.md`](004-global-load-store.prompt.md) — load and store a global.
5. [`005-struct-gep.prompt.md`](005-struct-gep.prompt.md) — index a struct field with `getelementptr`.
6. [`006-array-of-structs-gep.prompt.md`](006-array-of-structs-gep.prompt.md) — index a field inside an array of structs.
7. [`007-vector-reduction.prompt.md`](007-vector-reduction.prompt.md) — call a vector reduction intrinsic.
8. [`008-cmpxchg-loop.prompt.md`](008-cmpxchg-loop.prompt.md) — write an atomic compare-exchange retry loop.
9. [`009-intrinsic-metadata.prompt.md`](009-intrinsic-metadata.prompt.md) — attach metadata to an intrinsic call.
10. [`010-vertex-edge-attribute-lowering.prompt.md`](010-vertex-edge-attribute-lowering.prompt.md) — lower graph-style vertex and edge attributes.
11. [`011-register-binding-pattern.prompt.md`](011-register-binding-pattern.prompt.md) — resolve logical register IDs through a binding table.
12. [`012-custom-intrinsic-wrapper.prompt.md`](012-custom-intrinsic-wrapper.prompt.md) — wrap an LLVM intrinsic behind a BCIR-style operation.
13. [`013-mixed-stride-indexing.prompt.md`](013-mixed-stride-indexing.prompt.md) — lower mixed row and element strides to byte-offset addressing.
14. [`014-ham-hint-metadata.prompt.md`](014-ham-hint-metadata.prompt.md) — preserve HAM hints as LLVM metadata.

### Review and repair

15. [`015-mlir-to-llvm-lowering-review.prompt.md`](015-mlir-to-llvm-lowering-review.prompt.md) — review an MLIR-to-LLVM lowering for BCIR patterns.
16. [`016-fix-phi-predecessor.prompt.md`](016-fix-phi-predecessor.prompt.md) — repair a `phi` incoming-block list.
17. [`017-fix-duplicate-symbol.prompt.md`](017-fix-duplicate-symbol.prompt.md) — repair duplicate global definitions.
18. [`018-fix-immarg-intrinsic.prompt.md`](018-fix-immarg-intrinsic.prompt.md) — repair an intrinsic call with an `immarg` operand.
19. [`019-fix-atomic-ordering.prompt.md`](019-fix-atomic-ordering.prompt.md) — repair an invalid atomic store ordering.

### Optimization pass reasoning

20. [`020-predict-mem2reg.prompt.md`](020-predict-mem2reg.prompt.md) — predict promotion from stack slots to SSA values.
21. [`021-predict-simplifycfg.prompt.md`](021-predict-simplifycfg.prompt.md) — predict diamond CFG folding to `select`.
22. [`022-predict-loop-vectorizer.prompt.md`](022-predict-loop-vectorizer.prompt.md) — reason about loop-vectorization legality and expected vector IR shape.

## Verification

Run a solution through the assembler and discard the bitcode output:

```sh
llvm-as -disable-output llvm-training/exercises/001-add.solution.ll
```

Each prompt gives the exact command for its matching solution, broken input, or
optimization-pass input. Use these expectations by family:

- Standalone IR-writing solutions and fixed repair solutions should assemble with
  `llvm-as -disable-output`.
- Broken repair inputs are intentionally named `*.invalid.ll.txt`; they should be
  rejected by `llvm-as` and should not be renamed to plain `.ll`.
- Optimization reasoning inputs should assemble before running `opt`; pass output
  can differ across LLVM versions, so prompts describe structural observations
  such as new `phi`, `select`, vector-body, or remainder-loop patterns.
- Language-agnostic review prompts come before any future pass-implementation
  exercises. If C++ pass skeleton tasks are added later, keep them in a separate
  non-verified family and document build requirements locally.
