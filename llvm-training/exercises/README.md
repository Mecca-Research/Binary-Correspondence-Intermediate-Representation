# Exercises

This directory contains beginner and advanced LLVM IR exercises. The first
exercise family focuses on standalone IR writing; later families cover repair,
review, and optimization-pass reasoning. Follow the conventions in
[`../EXAMPLES.md`](../EXAMPLES.md). Each exercise should include:

- a prompt describing the task;
- the expected command to check the learner's answer or the checked-in solution;
- the expected observation, such as successful assembly, a specific diagnostic,
  or a review checklist;
- a standalone `*.solution.ll` file for executable LLVM IR answers, or a
  `*.solution.md` file for markdown-only review answers.

Repair exercises keep broken starting points as `*.invalid.ll.txt` so they remain
visibly intentional failures and do not enter any known-good `.ll` verification
loop. Optimization-pass reasoning exercises may include `*.input.ll` and
`*.after-<pass>.ll` teaching snapshots; these snapshots document expected shape,
not a byte-for-byte contract for every LLVM release.

Solutions must assemble with LLVM >= 15, where opaque pointers are the default.
Use `ptr` for pointer-typed values instead of typed pointers such as `i32*`.

Solutions that are executable LLVM IR must assemble with LLVM >= 15, where opaque
pointers are the default. Use `ptr` for pointer-typed values instead of typed
pointers such as `i32*`. Markdown solutions are review references and should be
read with `cat`, not assembled.

If your LLVM tools are installed with a version suffix, replace `llvm-as` and
`opt` in the commands with the matching binaries, for example `llvm-as-15`,
`llvm-as-18`, `opt-15`, or `opt-18`.

## Exercise families

- **Standalone IR writing**: write small, complete LLVM IR modules and verify
  them with `llvm-as`.
- **Review and repair**: inspect broken or risky IR, explain the issue, and, when
  applicable, compare against a fixed `*.solution.ll`.
- **Optimization pass reasoning**: assemble input snapshots and run a named pass,
  then compare the structural output against the prompt's expected observation.
- **Beginner/intermediate metadata and semantic review**: practice preserving
  debug/profile metadata, auditing attributes, and recognizing poison or
  floating-point-contract hazards.
- **BCIR lowering**: lower Binary Correspondence Intermediate Representation
  concepts to ordinary LLVM IR constructs such as GEPs, explicit byte offsets,
  runtime calls, prefetch intrinsics, and metadata catalogs.
- **MLIR bridge review**: reason about dialect boundaries, operation lowering,
  and type conversion before or during conversion to the LLVM dialect.
- **Backend/JIT review**: diagnose symbol resolution, ORC layer failures, and the
  path from target descriptions to `MCInst` emission.

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

### Beginner/intermediate metadata and semantic review

23. [`023-debug-metadata-preservation.prompt.md`](023-debug-metadata-preservation.prompt.md) — preserve useful `!dbg` locations after simplifying IR.
24. [`024-profile-metadata-branch-weights.prompt.md`](024-profile-metadata-branch-weights.prompt.md) — attach `!prof` branch-weight metadata.
25. [`025-attribute-contract-review.prompt.md`](025-attribute-contract-review.prompt.md) — review whether LLVM attributes are proven contracts.
26. [`026-poison-freeze-repair.prompt.md`](026-poison-freeze-repair.prompt.md) — repair poison-prone control flow with `freeze`.
27. [`027-fast-math-risk-review.prompt.md`](027-fast-math-risk-review.prompt.md) — review fast-math flags and floating-point semantic risk.

### BCIR lowering

28. [`028-lower-vertex-edge-fragment.prompt.md`](028-lower-vertex-edge-fragment.prompt.md) — lower a BCIR vertex/edge fragment to GEPs, loads, and stores.
29. [`029-lower-ham-prefetch-hint.prompt.md`](029-lower-ham-prefetch-hint.prompt.md) — lower a HAM hint to `llvm.prefetch` plus metadata.
30. [`030-lower-register-binding.prompt.md`](030-lower-register-binding.prompt.md) — lower logical register IDs through a binding table.
31. [`031-lower-runtime-call-boundary.prompt.md`](031-lower-runtime-call-boundary.prompt.md) — lower a runtime operation to an explicit ABI call boundary.

### MLIR bridge review

32. [`032-identify-mlir-dialect-boundaries.prompt.md`](032-identify-mlir-dialect-boundaries.prompt.md) — identify BCIR, generic, control-flow, and LLVM-lowering boundaries.
33. [`033-lower-mlir-graph-op-to-llvm-dialect.prompt.md`](033-lower-mlir-graph-op-to-llvm-dialect.prompt.md) — outline lowering of a graph op to LLVM-dialect loads.
34. [`034-review-mlir-to-llvm-type-conversion.prompt.md`](034-review-mlir-to-llvm-type-conversion.prompt.md) — review index, memref, graph, and vector type conversion hazards.

### Backend/JIT review

35. [`035-diagnose-missing-symbol-relocation.prompt.md`](035-diagnose-missing-symbol-relocation.prompt.md) — diagnose an unresolved runtime symbol relocation.
36. [`036-identify-orc-layer-failure.prompt.md`](036-identify-orc-layer-failure.prompt.md) — classify an ORC JIT failure by layer.
37. [`037-tablegen-to-mcinst-review.prompt.md`](037-tablegen-to-mcinst-review.prompt.md) — trace a pseudo instruction from TableGen to `MCInst` emission.

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
