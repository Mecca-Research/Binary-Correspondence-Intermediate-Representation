# Exercises

This directory contains beginner, intermediate, and advanced LLVM IR exercises.
The first exercise family focuses on standalone IR writing; later families cover
repair, review, optimization-pass reasoning, BCIR lowering, MLIR bridge reviews,
backend/JIT diagnosis, and advanced BCIR verification/debugging. Follow the conventions in
[`../EXAMPLES.md`](../EXAMPLES.md). Each exercise should include:

- a prompt describing the task;
- the expected command to check the learner's answer or the checked-in solution;
- the expected observation, such as successful assembly, a specific diagnostic,
  or a review checklist;
- a standalone `*.solution.ll` file for executable LLVM IR answers, or a
  `*.solution.md` file for markdown-only review answers; and
- a declarative grading manifest under [`../autograder/manifests/`](../autograder/manifests/)
  that validates against
  [`exercise.schema.json`](../autograder/schema/exercise.schema.json).

## Graded-set manifest policy

A new numbered exercise **must have a manifest before it enters the graded
set**. The prose prompt remains the learner-facing teaching explanation; the
manifest references that prompt and encodes only the machine-readable grading
contract, including answer kind, tools and minimum versions, tool-absence
behavior, checks and their explicit points, determinism, timeout, tags, license,
and difficulty. Do not move teaching prose into JSON.

Every check has an explicit `points` value, and the check-point sum must equal
the manifest's `score`. Tool-backed checks must declare their tools in both
`required_tools` and `minimum_tool_versions`. Choose the `tool_absence_policy`
explicitly: `hard_failure`, `unscored_skip`, or `reduced_confidence_score`.

Validate the complete manifest set from the repository root:

```sh
python3 llvm-training/tools/verify-exercise-manifests.py
```

The validator checks the schema contract, all referenced paths, unique exercise
and check IDs, exact point totals, tool declarations, and one-to-one coverage of
numbered `*.prompt.md` files. The checked-in manifests are organized
incrementally by the family tags documented below: standalone LLVM IR, repair,
optimizer prediction, BCIR lowering, MLIR review, backend/JIT review, and
adversarial analysis.

Repair exercises keep broken starting points as `*.invalid.ll.txt` so they remain
visibly intentional failures and do not enter any known-good `.ll` verification
loop. Optimization-pass reasoning exercises may include `*.input.ll` and
`*.after-<pass>.ll` teaching snapshots; these snapshots document expected shape,
not a byte-for-byte contract for every LLVM release.

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
- **Advanced BCIR verification and debugging**: design custom verifier passes,
  encode graph schemas as LLVM metadata, and repair GAADMSF-style lowering
  failures.
- **Adversarial IR and fuzzing**: classify verifier-valid semantic hazards,
  expected-invalid inputs, target-specific cases, and metadata-preservation
  seeds, then build reproducible BCIR-aware fuzz oracles around them.

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
33. [`033-lower-mlir-graph-op-to-llvm-dialect.prompt.md`](033-lower-mlir-graph-op-to-llvm-dialect.prompt.md) — outline lowering of a graph op to LLVM-dialect loads; pair with [`../14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md).
34. [`034-review-mlir-to-llvm-type-conversion.prompt.md`](034-review-mlir-to-llvm-type-conversion.prompt.md) — review index, memref, graph, and vector type conversion hazards.

### Backend/JIT review

35. [`035-diagnose-missing-symbol-relocation.prompt.md`](035-diagnose-missing-symbol-relocation.prompt.md) — diagnose an unresolved runtime symbol relocation.
36. [`036-identify-orc-layer-failure.prompt.md`](036-identify-orc-layer-failure.prompt.md) — classify an ORC JIT failure by layer.
37. [`037-tablegen-to-mcinst-review.prompt.md`](037-tablegen-to-mcinst-review.prompt.md) — trace a pseudo instruction from TableGen to `MCInst` emission.

### Advanced BCIR verification and debugging

38. [`038-custom-pass-bcir-invariants.prompt.md`](038-custom-pass-bcir-invariants.prompt.md) — design a verifier-style pass for BCIR lowering invariants.
39. [`039-graph-description-to-llvm-metadata.prompt.md`](039-graph-description-to-llvm-metadata.prompt.md) — encode a graph description as LLVM metadata attached to scalar IR.
40. [`040-debug-gaadmsf-lowering.prompt.md`](040-debug-gaadmsf-lowering.prompt.md) — debug a GAADMSF lowering with an invalid `phi` predecessor.

### Binary-analysis evidence review

41. [`041-interpret-static-binary-evidence.prompt.md`](041-interpret-static-binary-evidence.prompt.md) — interpret manifest-backed static evidence without promoting feature similarity to semantic equivalence.
42. [`042-review-evidence-provenance.prompt.md`](042-review-evidence-provenance.prompt.md) — separate deterministic static provenance from optional host-sensitive performance measurements.

## Agent-training templates

Reusable prompt templates live under [`templates/`](templates/). They are not
numbered exercises and are intended to seed new agent-training tasks or reviews:

- [`templates/lower-bcir-graph-fragment-1to1-registers.prompt.md`](templates/lower-bcir-graph-fragment-1to1-registers.prompt.md) — lower a BCIR graph fragment while preserving one-to-one logical-register correspondence.
- [`templates/add-metadata-preserve-verifier-validity.prompt.md`](templates/add-metadata-preserve-verifier-validity.prompt.md) — add BCIR metadata without breaking LLVM verifier validity.
- [`templates/review-mixed-stride-lowering.prompt.md`](templates/review-mixed-stride-lowering.prompt.md) — review mixed byte, element, row, and graph-edge stride lowering.
- [`templates/diagnose-optimizer-bcir-mapping-drift.prompt.md`](templates/diagnose-optimizer-bcir-mapping-drift.prompt.md) — diagnose optimizer-induced BCIR mapping drift between IR snapshots.
- [`templates/review-adversarial-ir.prompt.md`](templates/review-adversarial-ir.prompt.md) — review IR that can pass one validation layer while violating another semantic contract.
- [`templates/fuzz-bcir-lowering.prompt.md`](templates/fuzz-bcir-lowering.prompt.md) — design a reproducible BCIR lowering fuzzer with structural and semantic oracles.
- [`templates/preserve-metadata-through-pass.prompt.md`](templates/preserve-metadata-through-pass.prompt.md) — define and test metadata transfer policy for instruction-rewriting passes.

## Adversarial IR and fuzzing track

The adversarial track lives under [`adversarial/`](adversarial/). Its
[README](adversarial/README.md) defines four explicit fixture classes:
assemble-valid but semantically risky, intentionally invalid, target-specific,
and metadata-preservation. It also supplies a threat model for poison, metadata,
address spaces, operand bundles, stale debug info, BCIR 1:1 mapping, target
intrinsics, ABI attributes, `memory(...)`, and varargs.

Do not infer expected behavior from `.ll` versus `.ll.txt` alone. Read the
fixture's `; adversarial-class:` marker and run:

```sh
llvm-training/tools/verify-adversarial-fixtures.sh
```

The track includes focused fixtures for poison-sensitive branches, metadata
loss, address-space collapse, operand-bundle loss, and BCIR mapping drift. Use
the three adversarial prompt templates above to turn a seed into a review task,
a reproducible fuzz campaign, or a pass-specific metadata regression.

## Verification

Run a solution through the assembler and discard the bitcode output:

```sh
llvm-as -disable-output llvm-training/exercises/001-add.solution.ll
```

Run all checked-in executable LLVM IR solutions through the repository verifier:

```sh
llvm-training/tools/verify-exercises.sh
```

Each prompt gives the exact command for its matching solution, broken input,
markdown answer, or optimization-pass input. Use these expectations by family:

- Standalone IR-writing solutions, fixed repair solutions, metadata exercises,
  BCIR lowering solutions, and advanced graph/debugging solutions should
  assemble with `llvm-as -disable-output`.
- The verifier also runs `opt -passes=verify` on every checked-in
  `*.solution.ll` file.
- Broken repair inputs are intentionally named `*.invalid.ll.txt`; they may be
  rejected by `llvm-as`, or they may assemble while still being semantically
  unsafe. Semantic-only fixtures should carry the
  `; llvm-training-invalid-kind: semantic-only` marker. Adversarial fixtures also use the
  classification contract documented in `adversarial/README.md`; some are plain
  `.ll` because successful assembly is part of the lesson.
- Optimization reasoning inputs should assemble before running `opt`; pass output
  can differ across LLVM versions, so prompts describe structural observations
  such as new `phi`, `select`, vector-body, or remainder-loop patterns.
- Markdown-only review answers use `*.solution.md`; they are validated by the
  verifier as present, non-empty reference answers rather than assembled.
- MLIR bridge exercises in this directory are currently markdown review tasks.
  If future exercises add checked-in `*.mlir` solutions, document the required
  `mlir-opt` pipeline in the prompt and extend the verifier with an optional
  `mlir-opt` check that skips cleanly when MLIR tools are not installed.
- Language-agnostic review prompts come before any future pass-implementation
  exercises. If C++ pass skeleton tasks are added later, keep them in a separate
  non-verified family and document build requirements locally.

## Submission and attempt-directory convention

Submit attempts beneath one configured root, with one directory per permanent
three-digit exercise ID:

```text
attempts/
  001/answer.ll
  032/answer.md
  033/answer.mlir
```

The answer extension is determined by the manifest/registry answer kind:
`.ll` for LLVM IR and pass-output answers, `.mlir` for MLIR answers, and `.md`
for review or diagnostic answers. Do not include build scripts, object files,
shared libraries, executables, or symlinks that escape the attempt root. The
grader treats absent, empty, oversized, malformed, and verifier-rejected answers
as submission outcomes rather than trusting generated code.

The declarative manifest is the scoring authority. Its stable ID must match the
prompt filename and attempt directory; check IDs are also stable once reports or
datasets consume them. Rubric points must sum exactly to the declared score,
tool-backed checks must state minimum versions and absence policy, and optional
tool skips must remain explicit and unearned. Run both integration gates after
adding or changing an exercise:

```bash
python3 llvm-training/tools/verify-exercise-manifests.py
python3 llvm-training/tools/grade-exercises.py --self-test --format json
```
