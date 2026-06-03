# Formal Grammar

This directory holds the **Textmapper grammar** for LLVM IR
(`llvm-ir.tm`). Use it as a local syntax reference, not as the final
authority for every LLVM version. The LLVM LangRef is the canonical
documentation for LLVM IR, and the target installed `llvm-as` is the
practical parser authority for what your toolchain accepts.

## Version expectations

LLVM IR syntax and accepted constructs vary by LLVM release. The local
`llvm-ir.tm` grammar is a useful reference for syntax questions, but it
may lag upstream LLVM. Verify grammar-sensitive examples against both
the target LLVM version's `llvm-as` and the corresponding LangRef:
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
`package = "github.com/llir/ll"`. It's the Textmapper grammar from
the `llir/llvm` Go project, used here as a reference. Consult that
project for the upstream definition of record.

## Updating

Grammar changes when LLVM adds new constructs (new attributes,
intrinsics, metadata nodes). When the LLVM version moves forward, the
grammar usually needs a small update — track upstream
`github.com/llir/ll`.

## See also

- [`../INDEX.md`](../INDEX.md) (top level) — topic → prose-file map
- Each chapter file's "See also" section references back here when
  there's an ambiguous syntactic question
