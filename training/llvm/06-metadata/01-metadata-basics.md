# Metadata Basics

## TL;DR

Metadata is LLVM IR side information. It can help optimizers, debuggers,
and analysis tools, but it is not part of the language semantics of the
program. If a metadata attachment is missing, stale, or dropped by an
optimizer, the program's result must remain the same.

Official reference: [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata).

```llvm
%v = load i32, ptr %p, align 4, !dbg !12, !tbaa !20
br i1 %cond, label %hot, label %cold, !prof !30
!30 = !{!"branch_weights", i32 90, i32 10}
```

## Metadata identifiers and literals

LLVM metadata has a few common spellings:

| Syntax | Meaning |
|---|---|
| `!0`, `!42` | Numbered metadata references. The definition appears at module scope. |
| `!{ ... }` | A metadata tuple/node literal. |
| `distinct !{ ... }` | A node that must not be uniqued with an identical node. |
| `!"text"` | A metadata string. The leading `!` is required. |
| `!name` | Named metadata list or named metadata reference, depending on context. |

```llvm
!0 = !{!"tag", i32 7}
!1 = !{!0, ptr @global_value}
!my.named.list = !{!0, !1}
```

The most common error is writing a bare string inside a metadata tuple:

```llvm
; invalid: "branch_weights" is not a metadata string
!0 = !{"branch_weights", i32 90, i32 10}

; valid
!1 = !{!"branch_weights", i32 90, i32 10}
```

## `distinct` versus uniqued metadata

Ordinary metadata nodes may be uniqued: if two nodes have identical
contents, LLVM is allowed to treat them as the same node. `distinct`
requests object identity.

Use `distinct` when identity matters:

```llvm
!0 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, unit: !2)
!3 = distinct !{!3, !4} ; self-referential loop metadata
```

Loop metadata is often self-referential (`!3` contains `!3`) so LLVM can
identify the loop ID. Debug-info roots such as `DICompileUnit` and
`DISubprogram` are also commonly `distinct`.

## Named metadata

Named metadata lives at module scope and maps a name to a list of
metadata nodes:

```llvm
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}
```

Important named metadata includes:

- `!llvm.dbg.cu` — compile units that anchor source-level debug info.
- `!llvm.module.flags` — module-wide flags such as DWARF version and
  debug-info version.
- Target- or pass-specific named lists used by LLVM components.

## Instruction attachments

Instruction attachments are appended after the instruction's ordinary
operands and flags:

```llvm
%p = load ptr, ptr %slot, align 8, !nonnull !0
%x = load i32, ptr %p, align 4, !range !1, !dbg !2
br i1 %ok, label %then, label %else, !prof !3
br i1 %again, label %loop, label %exit, !llvm.loop !4
```

The spelling is always:

```text
, !<kind> !<metadata-id-or-node>
```

Some attachments, such as `!dbg`, point to specialized debug-info nodes.
Others, such as `!prof`, `!range`, or `!llvm.loop`, point to ordinary
metadata tuples with formats documented by the LangRef.

## Common metadata tags

| Tag | Usually attached to | Purpose |
|---|---|---|
| `!dbg` | functions and instructions | Maps IR back to source code for debuggers and diagnostics. |
| `!tbaa` | loads/stores | Type-Based Alias Analysis facts for memory disambiguation. |
| `!prof` | branches, switches, calls, functions | Profiling data such as branch weights and entry counts. |
| `!range` | loads, calls, invokes | Value range fact, commonly `[lo, hi)` integer intervals. |
| `!nonnull` | loads | Asserts a loaded pointer is not null. |
| `!llvm.loop` | loop latch branch | Loop transformation hints such as vectorization or unroll preferences. |

## Semantics rule: metadata must not change the language meaning

Metadata can describe facts or preferences, but it cannot make an
otherwise different program semantically valid. Examples:

- `!prof` can tell the optimizer that one branch is hotter than another;
  it cannot make the cold branch unreachable.
- `!llvm.loop` can request vectorization or unrolling; it cannot permit a
  transformation that would violate memory or poison semantics.
- `!dbg` can map an instruction to `foo.c:10:5`; it cannot make that
  instruction behave as if it came from a different source statement.
- `!nonnull` and `!range` encode facts that optimizers may use. If the
  facts are false, later optimization can become misleading or invalid.

## Pitfalls

- **Malformed tuples.** Metadata strings must be written as `!"..."`.
- **Wrong attachment kind.** A syntactically valid tuple may still be the
  wrong shape for a particular metadata kind.
- **Assuming metadata is permanent.** Optimizers may legally drop metadata
  they cannot preserve.
- **False facts.** Optimization metadata is not a comment; incorrect
  `!range`, `!nonnull`, `!tbaa`, or alias metadata can mislead
  transformations.
- **Confusing named metadata with numbered nodes.** `!llvm.dbg.cu` is a
  named list; `!0` is a concrete metadata node reference.

## See also

- [`02-debug-info.md`](02-debug-info.md) — debug-info node graph and source locations
- [`03-profile-and-optimization-metadata.md`](03-profile-and-optimization-metadata.md) — `!prof`, `!llvm.loop`, and optimization hints
- [`../01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) — shorter syntax introduction
- [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata)
