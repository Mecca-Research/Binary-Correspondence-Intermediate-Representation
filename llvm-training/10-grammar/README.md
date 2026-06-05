# Formal Grammar

This directory holds a full local snapshot of the **Textmapper grammar** for
LLVM IR (`llvm-ir.tm`) from `llir/grammar`. Use it as a local syntax reference
so agents can answer grammar-shape questions without leaving the repository. It
is still not the final authority for every LLVM version: the LLVM LangRef is the
canonical documentation for LLVM IR, and the target installed `llvm-as` is the
practical parser authority for what your toolchain accepts.

## Key takeaways

- Use the grammar as a syntax aid, then validate real IR with the target LLVM version's assembler and verifier.
- LLVM textual IR separates top-level entities, instruction forms, metadata attachments, and attributes; do not treat snippets as free-form text.
- Opaque-pointer-era IR still carries explicit source and access types on instructions even though pointer values print as `ptr`.
- Grammar checks catch parse shape, while semantic verifier checks catch dominance, type, PHI, and ordering rules.

## Version expectations

LLVM IR syntax and accepted constructs vary by LLVM release. The local
`llvm-ir.tm` grammar is a full upstream snapshot rather than a hand-trimmed
excerpt, but it may still lag newer LLVM releases. The verification stamp near
the top of `llvm-ir.tm` is an anchor for the toolchain/version last used during
local review; it is not a guarantee that this grammar is canonical for all LLVM
versions. Verify grammar-sensitive examples against both the target LLVM
version's `llvm-as` and the corresponding LangRef:
https://llvm.org/docs/LangRef.html

## What is Textmapper?

Textmapper (https://textmapper.org/) is a parser generator. The
`.tm` file is the input grammar from which a Go-based parser (the
`github.com/llir/ll` project) is generated.

## How to use this grammar as a reference

1. **Find the production for the construct you're researching.**
   For example, "what's the exact shape of an `atomicrmw`?"
   Open `llvm-ir.tm`, search for `AtomicRMWInst`:
   ```
   AtomicRMWInst -> AtomicRMWInst
     : 'atomicrmw' Volatileopt Op=AtomicOp Dst=TypeValue ',' X=TypeValue
       SyncScopeopt Ordering=AtomicOrdering Align=(',' Align)?
       Metadata=(',' MetadataAttachment)+?
     ;
   ```
2. **Follow the references.** Each capitalized name on the right-hand
   side is another production. E.g., `AtomicOp` lists the valid
   second-keywords (`add`, `sub`, `xchg`, `or`, ...).
3. **Look for `opt` suffixes.** `Volatileopt` means the `Volatile`
   token is optional.
4. **Look for the `%interface` declarations.** They group related
   productions (e.g., `Instruction`, `Terminator`, `Type`,
   `Constant`).

## Top-level navigation

| Section | Find by searching |
|---|---|
| Module structure | `Module -> Module` |
| Top-level entities | `TopLevelEntity` |
| Functions and their headers | `FuncHeader`, `FuncDef`, `FuncDecl` |
| Types | `Type -> Type`, `FirstClassType` |
| Constants | `Constant -> Constant`, `ConstantExpr` |
| Instructions | `ValueInstruction`, `Terminator` |
| Metadata | `MDTuple`, `Metadata`, `SpecializedMDNode` |
| Attribute lists | `FuncAttribute`, `ParamAttribute`, `ReturnAttribute` |
| Linkage / visibility / etc | `Linkage`, `Visibility`, `Preemption`, `DLLStorageClass` |
| Atomic orderings | `AtomicOrdering` |
| Comparison predicates | `IPred`, `FPred` |
| Calling conventions | `CallingConv` |
| Debug info (DI*) | `DICompileUnit`, `DIFile`, `DISubprogram`, ... |

## Grammar conventions

- `'keyword'` — literal token
- `Foo` (capitalized) — non-terminal
- `Foo?` or `Fooopt` — optional
- `Foo*` — zero or more
- `Foo+` — one or more
- `(A separator B)+` — list of `A` separated by `B`, at least one
- `Foo=Bar` — captures the matched `Bar` under name `Foo` in the AST
- `%interface Foo;` — declares `Foo` as a category; any production
  marked `-> Foo` belongs to it

## When prose vs grammar disagree

Use `llvm-ir.tm` as a syntax-focused local reference when prose in this
repo is ambiguous, but do not treat it as canonical for all LLVM
versions. The LangRef remains the canonical documentation, the target
installed `llvm-as` is the practical parser authority, and this local
grammar may lag upstream LLVM.

## Source attribution

`llvm-ir.tm` declares its own package as
`package = "github.com/llir/ll"`. It is a full local snapshot of the
Textmapper grammar from `llir/grammar` at commit
`5a3820b516f7903e27ad16ebe4add1ec634f1c05`, consumed by the `llir/ll` Go
parser project. The upstream grammar is offered under 0BSD and Unlicense terms;
see [`../NOTICE.md`](../NOTICE.md) for repository attribution. Keep the local
snapshot so agents do not have to fetch upstream while reading the training
corpus.

## Updating

Grammar changes when LLVM adds new constructs (new attributes, intrinsics,
metadata nodes, or instruction spellings). When the LLVM version moves forward,
refresh this full snapshot from upstream `llir/grammar`, record the source commit
in the header of `llvm-ir.tm`, and update the attribution note above if the
license or source location changes.

## See also

- [`../INDEX.md`](../INDEX.md) (top level) — topic → prose-file map
- Each chapter file's "See also" section references back here when
  there's an ambiguous syntactic question
