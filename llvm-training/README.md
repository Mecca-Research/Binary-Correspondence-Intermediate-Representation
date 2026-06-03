# llvm-training — Agent Context Repo for LLVM IR

A curated, agent-readable reference for LLVM IR. Designed to be cloned
and read by an LLM coding agent (Claude, Codex, etc.) **before** taking on
LLVM-related tasks, so the agent walks into the work with verified
syntax, semantics, and a catalog of common failure modes.

This is **not** a fine-tuning corpus. It's a context pack: dense, indexed,
example-first, with every `.ll` snippet guaranteed to assemble against a
modern `llvm-as` (LLVM >= 15, opaque pointers).

## How an agent should consume this repo

1. **Always start with [`INDEX.md`](INDEX.md).** It's the topic -> file lookup. Don't grep
   the tree blind.
2. **Follow the path in [`CURRICULUM.md`](CURRICULUM.md)** if learning end-to-end. Skip
   to a leaf chapter if doing a targeted task.
3. **Treat `08-pitfalls/` as a checklist** before writing or reviewing
   LLVM IR. Every pitfall is tied to a real bug that shipped — most of
   them caught in the sibling BCIR project.
4. **`10-grammar/llvm-ir.tm`** is the formal grammar. Use it for
   syntax-edge questions. It is the source of truth when prose disagrees.

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
├── 08-pitfalls/          real-world bugs (mostly from BCIR review)
├── 10-grammar/           Textmapper grammar (formal syntax)
└── reference/            instruction quickref, intrinsics, glossary
```

Numbered directories follow the reading order. Gaps (no `06-`, `07-`,
`09-`) are reserved for future chapters: instruction encyclopedia,
metadata/debug-info, toolchain.

## Verifying examples

```bash
for f in llvm-training/**/examples/*.ll; do
  llvm-as "$f" -o /dev/null || echo "FAIL: $f"
done
```

CI should run this. Anything that doesn't assemble shouldn't ship.

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
