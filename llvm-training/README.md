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
├── 08-pitfalls/          real-world bugs (mostly from BCIR review)
├── 10-grammar/           Textmapper grammar (formal syntax)
├── **/examples/*.ll      standalone examples that must assemble
└── reference/            instruction quickref, intrinsics, glossary
```

Numbered directories follow the reading order. Gaps (no `07-`, `09-`)
are reserved for future chapters: instruction encyclopedia and toolchain.

## Example and snippet conventions

Use these labels consistently in chapters so readers and CI know what is
runnable versus illustrative:

- **Standalone examples** live in `*/examples/*.ll` (for example,
  `00-foundations/examples/simple-add.ll`) and must assemble with `llvm-as`.
- **Fenced `llvm` snippets** in chapter prose may be fragments: single
  instructions, declarations, partial functions, or before/after excerpts. Do
  not assume a fenced snippet is independently runnable unless the chapter says
  so or links to a standalone example.
- **Intentionally invalid snippets** should be labeled **invalid** or
  **verifier failure example** near the fence or section heading, and should
  explain the expected parser/verifier failure.
- If an embedded snippet is intended to be part of the assembly guarantee, move
  it into `examples/*.ll` or add a dedicated extraction/test path before
  documenting it as runnable.

## Verifying standalone examples

```bash
find llvm-training -path '*/examples/*.ll' -print0 | sort -z | while IFS= read -r -d '' f; do
  llvm-as "$f" -o /dev/null || exit 1
done
```

CI runs this check only against standalone `*/examples/*.ll` files when
`llvm-as` is available. Anything in those files that doesn't assemble
shouldn't ship. See `llvm-training/examples/README.md` for the current
standalone example manifest and per-file commands.

## How big is this repo, and how big should it get?

| Stage | Files | Size | Purpose |
|---|---|---|---|
| Seed (current) | ~40 | ~150 KB | Foundations + syntax + pitfalls + grammar |
| Curated (target) | ~200 | ~50-200 MB | Add instr encyclopedia, metadata, MLIR overview, toolchain, exercises |
| Training corpus (stage 2) | 100k+ | 10-100 GB | Paired (source, IR), (IR, opt-IR), (IR, asm) examples for fine-tuning. Out of scope here. |

The curated target plateaus around 100-200 MB because quality and
indexability matter more than raw volume for an agent-context repo.

## Relationship to the BCIR project

This repo lives inside the BCIR project tree on purpose. BCIR is a
practical case study for almost every pitfall documented here —
`08-pitfalls/` cross-references commits (`1f62e86`, `5754354`) where
real instances were fixed.

## License & attribution

Apache-2.0 (matches the BCIR repo). See [`NOTICE.md`](NOTICE.md) for source attribution.
