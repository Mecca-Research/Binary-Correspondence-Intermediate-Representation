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
2. **Follow the path in [`CURRICULUM.md`](CURRICULUM.md)** if learning end-to-end. Skip
   to a leaf chapter if doing a targeted task.
3. **Treat `08-pitfalls/` as a checklist** before writing or reviewing
   LLVM IR. Every pitfall is tied to a real bug that shipped — most of
   them caught in the sibling BCIR project.
4. **`10-grammar/llvm-ir.tm`** is the formal grammar. Use it as a
   formal syntax aid; verify against the target LLVM version's `llvm-as`
   and LangRef.

## Layout

```
llvm-training/
├── README.md             you are here
├── CURRICULUM.md         reading order (30-min / 2-hr / deep paths)
├── INDEX.md              topic / symbol -> file map
├── NOTICE.md             attribution
├── 00-foundations/       what IR is, SSA, IR vs asm/other IRs
├── 01-syntax/            modules, functions, basic blocks, instr format
├── 02-types/             primitive, composite, opaque, pointer
├── 03-constants/         integer, float, string, global vs local
├── 04-memory/            alloca, load/store, globals, address spaces
├── 05-control-flow/      br, conditional br, switch, indirectbr
├── 06-metadata/          metadata syntax, debug info, profiling, loop hints
├── 07-optimization/      opt pass model, analyses, transforms, opt levels
├── 08-pitfalls/          real-world bugs (mostly from BCIR review)
├── 09-vectorization/     Loop/SLP vectorizers, diagnostics, vector IR patterns
├── 10-grammar/           Textmapper grammar (formal syntax)
├── 11-concurrency/       atomic orderings, atomic instructions, volatile
├── 12-backend-jit/       backend pipeline, TableGen, ORC/LLJIT
├── tools/                example verification and smoke-test scripts
├── **/examples/*.ll      standalone examples that must assemble
└── reference/            instruction quickref, intrinsics, glossary
```

Numbered directories follow the reading order. Future chapters may add a
dedicated instruction encyclopedia or additional toolchain material.

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
```

`verify-examples.sh` checks every known-good standalone `*/examples/*.ll` file
with both `llvm-as` and `opt -passes=verify`, skipping `.ll.txt` files and any
`.ll` file with `invalid` in its name. Anything else in those known-good files
that doesn't assemble and verify shouldn't ship.

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

## How BCIR Uses This

Use this repo as a BCIR LLVM IR task index, with BCIR-specific lowering notes in
[`bcir-mapping/README.md`](bcir-mapping/README.md).

| BCIR task | Read first |
|---|---|
| Writing or reviewing runtime `.ll` files | Syntax ([modules](01-syntax/01-modules-functions-blocks.md), [instructions](01-syntax/02-instruction-format.md)), [types](02-types/02-composite-types.md), [memory](04-memory/02-load-store.md), [control flow](05-control-flow/02-conditional-br.md), [pitfalls](08-pitfalls/README.md) |
| Debugging verifier errors | [`08-pitfalls/README.md`](08-pitfalls/README.md) |
| Changing BCIR runtime ABI structs | [type schema drift](08-pitfalls/05-type-schema-drift.md), [BCIR runtime ABI mapping](bcir-mapping/05-runtime-abi.md) |
| Adding intrinsics | [common intrinsics](13-advanced-ir/01-common-intrinsics.md), [target-specific intrinsics](13-advanced-ir/02-target-specific-intrinsics.md), [immarg pitfall](08-pitfalls/06-immarg-violation.md) |
| Adding atomic/concurrent behavior | [atomic orderings](11-concurrency/01-atomic-orderings.md), [atomic instructions](11-concurrency/02-atomic-instructions.md), [volatile vs atomic](11-concurrency/03-volatile-vs-atomic.md) |
| Optimizing generated IR | [pass model](07-optimization/01-pass-model.md), [transform passes](07-optimization/03-common-transform-passes.md), [vectorization](09-vectorization/README.md) |
| Planning MLIR lowering | [MLIR bridge overview](14-mlir-bridge/README.md), [type conversion](14-mlir-bridge/05-type-conversion-and-materialization.md), [conversion patterns](14-mlir-bridge/06-conversion-patterns.md), [end-to-end lowering](14-mlir-bridge/08-end-to-end-bcir-lowering.md) |
| Backend/JIT experiments | [codegen pipeline](12-backend-jit/01-codegen-pipeline.md), [ORC JIT](12-backend-jit/03-orc-jit.md) |

## Relationship to the BCIR project

This repo lives inside the BCIR project tree on purpose. BCIR is a
practical case study for almost every pitfall documented here —
`08-pitfalls/` cross-references commits (`1f62e86`, `5754354`) where
real instances were fixed.

## License & attribution

Apache-2.0 (matches the BCIR repo). See [`NOTICE.md`](NOTICE.md) for source attribution.
