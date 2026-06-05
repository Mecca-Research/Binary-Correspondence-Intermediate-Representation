# llvm-training — Agent Context Repo for LLVM IR

A curated, agent-readable reference for LLVM IR. Designed to be cloned
and read by an LLM coding agent (Claude, Codex, etc.) **before** taking on
LLVM-related tasks, so the agent walks into the work with verified
syntax, semantics, and a catalog of common failure modes.

This is **not** a fine-tuning corpus. It's a context pack: dense, indexed,
example-first. Standalone files under `*/examples/*.ll` are guaranteed to
assemble against a modern `llvm-as` (LLVM >= 15, opaque pointers). Embedded
chapter snippets may be illustrative fragments unless they are explicitly made
testable.

## How an agent should consume this repo

1. **Always start with [`INDEX.md`](INDEX.md).** It's the topic -> file lookup. Don't grep
   the tree blind.
   For BCIR-specific lowering idioms, jump from there to the dedicated
   [`BCIR pattern index`](indexes/bcir-patterns.md).
2. **Follow the path in [`CURRICULUM.md`](CURRICULUM.md)** if learning end-to-end. Skip
   to a leaf chapter if doing a targeted task.
3. **Use [`quickref/`](quickref/) for one-page cheat sheets** on opaque
   pointers, BCIR lowering, vectorization, metadata, the new pass manager,
   advanced IR contracts, and MLIR bridge reviews.
4. **Treat `08-pitfalls/` as a checklist** before writing or reviewing
   LLVM IR. Every pitfall is tied to a real bug that shipped — most of
   them caught in the sibling BCIR project.
5. **`10-grammar/llvm-ir.tm`** is the formal grammar. Use it as a
   formal syntax aid; verify against the target LLVM version's `llvm-as`
   and LangRef.

## Layout

```
llvm-training/
├── README.md             you are here
├── START_HERE.md         fastest orientation path for agents and humans
├── CURRICULUM.md         reading order (30-min / 2-hr / deep paths)
├── RECIPES.md            task-oriented lookup paths for common LLVM work
├── INDEX.md              topic / symbol -> file map
├── quickref/             one-page cheat sheets for common agent tasks
├── SEMVER.md             compatibility and versioning policy for this pack
├── EVAL.md               evaluation checklist for agent-usefulness
├── NOTICE.md             attribution
├── 00-foundations/       what IR is, SSA, IR vs asm/other IRs
├── 01-syntax/            modules, functions, basic blocks, instr format
├── 02-types/             primitive, composite, opaque, pointer
├── 03-constants/         integer, float, string, global vs local
├── 04-memory/            alloca, load/store, globals, address spaces
├── 05-control-flow/      br, conditional br, switch, indirectbr
├── 06-metadata/          metadata syntax, debug info, profiling, loop hints
├── 07-optimization/      opt pass model, analyses, transforms, deep BCIR risks, PGO/LTO/BOLT
├── 08-pitfalls/          real-world bugs (mostly from BCIR review)
├── 09-vectorization/     Loop/SLP vectorizers, diagnostics, vector IR patterns
├── 10-grammar/           Textmapper grammar (formal syntax)
├── 11-concurrency/       atomics, volatile, C++/Rust memory-model mapping
├── 12-backend-jit/       backend pipeline, TableGen, ORC/LLJIT, MC, relocations
├── 13-advanced-ir/       intrinsics, attributes, UB/poison, ABI details
├── 14-mlir-bridge/       MLIR concepts and LLVM dialect lowering paths
├── 15-binary-analysis/   post-codegen analysis, side channels, traces/counters
├── 16-exception-handling/ exception-handling IR and funclets
├── 17-new-pass-manager/  modern PassBuilder plugins, callbacks, BCIR pipelines
├── exercises/            runnable prompts, expected observations, solutions
├── indexes/              generated or focused lookup indexes
├── tools/                example verification and smoke-test scripts
├── **/examples/*.ll      standalone examples that must assemble
└── reference/            instruction quickref, intrinsics, glossary
```

Numbered directories follow the reading order. Future chapters may add a
dedicated instruction encyclopedia or additional toolchain material.

## New examples and advanced examples summary

The expanded corpus now has both beginner examples and advanced artifacts:

- **Beginner runnable IR**: compact `*.ll` modules in foundations, syntax, types,
  memory, control flow, metadata, concurrency, vectorization, and advanced-IR
  chapters. These are the first files to read when learning LLVM IR syntax.
- **Before/after optimization examples**: paired files such as
  `*-before.ll`, `*-after-mem2reg.ll`, `*-after-simplifycfg.ll`, and
  `*-after-o2.ll` explain how `opt` rewrites IR and what is stable versus
  LLVM-version-dependent.
- **BCIR lowering examples**: checked LLVM IR under `bcir-mapping/examples/`
  demonstrates graph fragments, claim/resource lookup, HAM hints, runtime-call
  wrappers, mixed-stride addressing, and diagnostic metadata preservation.
- **MLIR bridge examples**: `14-mlir-bridge/examples/*.mlir` illustrates dialect
  and LLVM-dialect shapes; these are MLIR artifacts, not standalone `.ll` files.
- **Backend/JIT diagnostics examples**: TableGen and LLJIT outline artifacts in
  `12-backend-jit/examples/` are review aids for target descriptions, ORC layer
  ownership, MC emission, relocations, and missing-symbol failures.
- **Binary-analysis evidence artifacts**: CSV trace/counter/BCSA samples in
  `15-binary-analysis/examples/` document evidence schemas and must be reviewed
  with chapter prose rather than sent to `llvm-as`.
- **Modern pass-manager examples**: `17-new-pass-manager/examples/` includes
  pass-plugin and adaptive-pipeline C++ sketches plus GAADMSF before/after IR
  for modern `opt -passes=...` walkthroughs.
- **Repair, prediction, and advanced review exercises**: `exercises/016`-`027`
  include invalid fixtures, pass-output prediction tasks, metadata preservation
  checks, and UB/poison/fast-math review prompts; `exercises/028`-`040` cover
  BCIR lowering, MLIR bridge review, backend/JIT diagnostics, custom-pass
  invariants, graph metadata, and GAADMSF debugging.


## Quick reference paths

Use these one-page sheets when you already know the lesson family and need a
fast pre-edit checklist:

| Task | Quickref | Deep context |
| --- | --- | --- |
| Opaque pointer migration | [`quickref/opaque-pointers.md`](quickref/opaque-pointers.md) | [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) |
| BCIR lowering | [`quickref/bcir-lowering.md`](quickref/bcir-lowering.md) | [`bcir-mapping/README.md`](bcir-mapping/README.md) |
| Vectorization | [`quickref/vectorization.md`](quickref/vectorization.md) | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Metadata preservation | [`quickref/metadata.md`](quickref/metadata.md) | [`06-metadata/README.md`](06-metadata/README.md), [`bcir-mapping/10-metadata-and-diagnostics.md`](bcir-mapping/10-metadata-and-diagnostics.md) |
| New pass manager pipelines | [`quickref/new-pass-manager.md`](quickref/new-pass-manager.md) | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md), [`07-optimization/08-deep-optimization-lessons.md`](07-optimization/08-deep-optimization-lessons.md) |
| Advanced intrinsics/attributes/poison/fast math | [`quickref/advanced-ir.md`](quickref/advanced-ir.md) | [`13-advanced-ir/README.md`](13-advanced-ir/README.md), [`reference/intrinsics-quickref.md`](reference/intrinsics-quickref.md) |
| Operand bundles, GC/coroutine/convergence tokens, and matrix intrinsics | [`quickref/advanced-ir.md`](quickref/advanced-ir.md) | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md), [`13-advanced-ir/07-operand-bundles.md`](13-advanced-ir/07-operand-bundles.md), [`reference/intrinsics-quickref.md`](reference/intrinsics-quickref.md) |
| MLIR-to-LLVM bridge review | [`quickref/mlir-bridge.md`](quickref/mlir-bridge.md) | [`14-mlir-bridge/README.md`](14-mlir-bridge/README.md) |

Use [`EXAMPLES.md`](EXAMPLES.md) for naming and verification rules before adding
new artifacts to any of these families.
For repository-wide contributor guidance, including BCIR mapping and
metadata-preservation expectations, see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Example, exercise, and snippet conventions

Use the conventions in [`EXAMPLES.md`](EXAMPLES.md) consistently so readers
and CI know what is runnable versus illustrative. In short:

- **Standalone examples** live in `*/examples/*.ll` (for example,
  `00-foundations/examples/simple-add.ll`) and must assemble with LLVM >= 15
  opaque pointers.
- **Intentionally invalid examples** should use `.ll.txt` or include `invalid`
  in the filename so broad verification commands can skip them.
- **Pass-output examples** should use clear before/after names such as
  `foo-before.ll`, `foo-after-mem2reg.ll`, or `foo-after-o2.ll`.
- **Chapter examples** should have a local `examples/README.md` or a short
  section listing the commands for that chapter's examples.
- **Exercises** should document the prompt, expected command, expected
  observation, and optional solution file.
- **Fenced `llvm` snippets** in chapter prose may be fragments: single
  instructions, declarations, partial functions, or before/after excerpts. Do
  not assume a fenced snippet is independently runnable unless the chapter says
  so or links to a standalone example.

If an embedded snippet is intended to be part of the assembly guarantee, move it
into `examples/*.ll` or add a dedicated extraction/test path before documenting
it as runnable.

## Verifying and smoke-testing standalone examples

Use the checked-in tool scripts from the repository root:

```bash
./llvm-training/tools/verify-examples.sh
./llvm-training/tools/smoke-llc.sh
./llvm-training/tools/smoke-lli.sh
./llvm-training/tools/verify-exercises.sh
./llvm-training/tools/verify-invalid-fixtures.sh
./llvm-training/tools/verify-opt-diff.sh
./llvm-training/tools/verify-opaque-pointers.sh
./llvm-training/tools/verify-manifest.sh
./llvm-training/tools/verify-csv-schema.sh
./llvm-training/tools/verify-mlir-examples.sh
./llvm-training/tools/verify-bcir-mapping.sh
```

The same checks are also available as CMake custom targets after configuring the
repository. These targets are suitable for minimal CI or local images because
they skip cleanly when their optional LLVM tools are unavailable:

```bash
cmake --build build --target llvm-training-verify-examples
cmake --build build --target llvm-training-smoke-llc
cmake --build build --target llvm-training-smoke-lli
cmake --build build --target llvm-training-verify-exercises
cmake --build build --target llvm-training-verify-invalid-fixtures
cmake --build build --target llvm-training-verify-opt-diff
cmake --build build --target llvm-training-verify-opaque-pointers
cmake --build build --target llvm-training-verify-manifest
cmake --build build --target llvm-training-verify-csv-schema
cmake --build build --target llvm-training-verify-mlir-examples
cmake --build build --target llvm-training-verify-bcir-mapping
```

`verify-examples.sh` checks every known-good standalone `*/examples/*.ll` file
with both `llvm-as` and `opt -passes=verify`, skipping `.ll.txt` files and any
`.ll` file with `invalid` in its name. The intentionally invalid tripwire
fixture `llvm-training/examples/broken-example.ll.txt` proves the skip rule is
working and should never be renamed to a known-good `.ll` example. Anything else
in those known-good files that doesn't assemble and verify shouldn't ship.

`verify-exercises.sh` applies the same assembler-and-verifier contract to every
checked-in `llvm-training/exercises/*.solution.ll` reference answer. Use
`verify-invalid-fixtures.sh` for intentionally broken `.invalid.ll.txt` repair
fixtures, `verify-opt-diff.sh` for golden optimizer-output pairs,
`verify-mlir-examples.sh` for MLIR syntax coverage, `verify-bcir-mapping.sh` for
BCIR source-like fragments and lowered companions, and `verify-csv-schema.sh` for
binary-analysis evidence tables.

The smoke scripts are intentionally narrower: `smoke-llc.sh` emits assembly for
a curated portable subset, while `smoke-lli.sh` runs only modules with a safe
`main` or explicitly documented runnable entrypoint. Most examples are
assembly-only because they are library-style snippets, optimization
before/after artifacts, target-lowering examples, or intrinsic/metadata
demonstrations rather than complete programs. See
`llvm-training/examples/README.md` for the current standalone example manifest
and per-file commands.

## How big is this repo, and how big should it get?

| Stage | Files | Size | Purpose |
|---|---|---|---|
| Seed (current) | ~40 | ~150 KB | Foundations + syntax + pitfalls + grammar |
| Curated (target) | ~200 | ~50-200 MB | Add instr encyclopedia, metadata, MLIR overview, toolchain, exercises |
| Training corpus (stage 2) | 100k+ | 10-100 GB | Paired (source, IR), (IR, opt-IR), (IR, asm) examples for fine-tuning. Out of scope here. |

The curated target plateaus around 100-200 MB because quality and
indexability matter more than raw volume for an agent-context repo.

Recent advanced paths: agents doing non-foundational LLVM work should start
with [`RECIPES.md`](RECIPES.md) for task-based routes, then jump directly to
[`15-binary-analysis/README.md`](15-binary-analysis/README.md) for binary
analysis/security/performance workflows or
[`07-optimization/08-deep-optimization-lessons.md`](07-optimization/08-deep-optimization-lessons.md) for
BCIR-specific optimizer legality risks, or
[`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) for
modern profile-guided, link-time, and post-link optimization context.

## How BCIR Uses This

Use this repo as a BCIR LLVM IR task index, with BCIR-specific lowering notes in
[`bcir-mapping/README.md`](bcir-mapping/README.md).

| BCIR task | Read first |
|---|---|
| Writing or reviewing runtime `.ll` files | Syntax ([modules](01-syntax/01-modules-functions-blocks.md), [instructions](01-syntax/02-instruction-format.md)), [types](02-types/02-composite-types.md), [memory](04-memory/02-load-store.md), [control flow](05-control-flow/02-conditional-br.md), [pitfalls](08-pitfalls/README.md) |
| Debugging verifier errors | [`08-pitfalls/README.md`](08-pitfalls/README.md) |
| Changing BCIR runtime ABI structs | [type schema drift](08-pitfalls/05-type-schema-drift.md), [BCIR runtime ABI mapping](bcir-mapping/05-runtime-abi.md) |
| Adding intrinsics or attributes | [common intrinsics](13-advanced-ir/01-common-intrinsics.md), [attributes](13-advanced-ir/04-attributes.md), [target-specific intrinsics](13-advanced-ir/02-target-specific-intrinsics.md), [immarg pitfall](08-pitfalls/06-immarg-violation.md) |
| Reviewing undefined-value or poison hazards | [poison, undef, and freeze](13-advanced-ir/05-poison-undef-freeze.md), [attributes](13-advanced-ir/04-attributes.md) |
| Deciding whether relaxed floating-point math is safe | [fast-math flags](13-advanced-ir/06-fast-math-flags.md), [vectorization](09-vectorization/README.md) |
| Adding atomic/concurrent behavior | [atomic orderings](11-concurrency/01-atomic-orderings.md), [atomic instructions](11-concurrency/02-atomic-instructions.md), [volatile vs atomic](11-concurrency/03-volatile-vs-atomic.md), [C++/Rust mapping](11-concurrency/04-memory-model-mapping.md) |
| Optimizing generated IR | [pass model](07-optimization/01-pass-model.md), [analysis passes](07-optimization/02-common-analysis-passes.md), [transform passes](07-optimization/03-common-transform-passes.md), [deep BCIR optimizer lessons](07-optimization/08-deep-optimization-lessons.md), [modern pass infrastructure](17-new-pass-manager/README.md), [debugging passes](07-optimization/05-debugging-passes.md), [PGO/LTO/BOLT](07-optimization/06-pgo-lto-bolt.md), [vectorization](09-vectorization/README.md) |
| Planning MLIR lowering | [MLIR overview](14-mlir-bridge/01-what-is-mlir.md), [lowering to LLVM dialect](14-mlir-bridge/03-lowering-to-llvm-dialect.md), [BCIR dialect sketch](14-mlir-bridge/04-bcir-as-custom-dialect.md) |
| Backend/JIT experiments | [codegen pipeline](12-backend-jit/01-codegen-pipeline.md), [ORC JIT](12-backend-jit/03-orc-jit.md), [MC and relocations](12-backend-jit/04-mc-and-relocations.md) |
| Security/performance binary analysis | [microarchitecture side channels](15-binary-analysis/01-microarchitecture-side-channels.md), [dynamic traces/counters](15-binary-analysis/02-dynamic-traces-and-counters.md), [interpretable BCSA features](15-binary-analysis/03-interpretable-bcsa-features.md) |

## Relationship to the BCIR project

This repo lives inside the BCIR project tree on purpose. BCIR is a
practical case study for almost every pitfall documented here —
`08-pitfalls/` cross-references commits (`1f62e86`, `5754354`) where
real instances were fixed.

## License & attribution

Apache-2.0 (matches the BCIR repo). See [`NOTICE.md`](NOTICE.md) for source attribution.
