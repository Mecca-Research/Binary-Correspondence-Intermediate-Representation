# Comments and Metadata

## TL;DR

- **Comments** start with `;` and run to end of line. There are
  **no block comments**. (The LangRef-adjacent ANTLR/Textmapper
  grammars treat `/* ... */` as non-standard.)
- **Metadata** is the structured "everything else": debug info,
  alias scopes, profile counts, optimizer hints. Metadata never
  affects the *semantics* of the program, only what tools can know
  about it.

```llvm
; This is a single-line comment.
%x = add i32 1, 2     ; trailing comment

%y = load i32, ptr %p, !dbg !42, !tbaa !7
                                  ^^^^^^   ^^^^^
                                  metadata attachments
```

## Comments

The lexer drops everything from `;` to the newline. Comments may
appear:

- On their own line
- After an instruction (trailing)
- Inside metadata definitions

```llvm
; Module header comments
source_filename = "example.ll"   ; trailing OK

; Comment between top-level entities is fine
@g = global i32 0
```

There's no other comment syntax. Some LLVM projects have used `//` in
markdown documentation, but the assembler rejects it.

## Metadata: the basics

Metadata is structured side-information attached to:

- **The module** — via named lists like `!llvm.module.flags`,
  `!llvm.dbg.cu`.
- **Functions** — e.g., `!dbg` attached to a `define`.
- **Instructions** — `!dbg`, `!tbaa`, `!alias.scope`, `!llvm.loop`,
  etc.
- **Globals** — debug info, prof data.

Metadata never participates in code generation directly. It's read by
optimizers, debuggers, and analysis tools.

### Three kinds of metadata syntax

1. **Metadata strings** — `!"text"`. **The `!` prefix is mandatory;**
   `"text"` alone is a regular string constant.

2. **Metadata nodes** — `!{ ... }`. A tuple of metadata operands.
   Operands can be metadata strings, integers, types, value
   references, or other metadata nodes (referenced by `!N`).

3. **Metadata IDs** — `!N` (where N is a number) or `!Name`. A
   reference to a top-level metadata definition.

```llvm
; A metadata tuple
!0 = !{!"hello", i32 7, ptr @g}

; A metadata string is a tuple of one element, often:
!1 = !{!"branch_weights", i32 30, i32 70}

; Metadata references other metadata
!2 = !{!0, !1}

; Named metadata: a list of metadata-node IDs
!my.list = !{!0, !1, !2}

; LLVM-defined named lists
!llvm.module.flags = !{!10}
!10 = !{i32 2, !"Dwarf Version", i32 4}
```

### Specialized metadata nodes (debug info)

Debug info uses *specialized* metadata constructors that look like
function-call syntax with named fields:

```llvm
!7 = distinct !DICompileUnit(
       language: DW_LANG_C99,
       file: !1,
       producer: "clang version 18",
       isOptimized: true,
       emissionKind: FullDebug
     )
!1 = !DIFile(filename: "main.c", directory: "/src")
```

Common specialized nodes:

- `!DICompileUnit` — root of a debug-info graph
- `!DIFile` — a source file
- `!DISubprogram` — a function's debug info
- `!DILocalVariable`, `!DIGlobalVariable` — variable info
- `!DILocation` — source line/column attached to instructions via
  `!dbg`
- `!DIBasicType`, `!DIDerivedType`, `!DICompositeType` — type info
- `!DIExpression` — DWARF expression for variable location

See `10-grammar/llvm-ir.tm` (`%interface SpecializedMDNode`) for the
full list.

The `distinct` keyword means "do not deduplicate this node with
identical-content nodes". Without it, LLVM may unify identical
metadata.

## Attaching metadata to instructions

The syntax is `, !<name> !<id>` appended to an instruction:

```llvm
%v = load i32, ptr %p, align 4, !dbg !100, !tbaa !20

call void @check(i32 %v) !dbg !101    ; allowed on call too

define i32 @foo() !dbg !50 {           ; on the function header
  ; ...
}
```

You can attach multiple metadata kinds in any order.

### Common attachments

| Kind | Purpose |
|---|---|
| `!dbg` | Source location for debug info |
| `!tbaa` | Type-based alias analysis class |
| `!alias.scope`, `!noalias` | Noalias scoping |
| `!nontemporal` | Streaming load/store hint |
| `!nonnull` | Loaded pointer is not null |
| `!range` | Loaded integer is in `[lo, hi)` |
| `!llvm.loop` | Attached to loop latch's terminator; carries `llvm.loop.*` properties |
| `!prof` | Branch probability or function entry count |
| `!annotation` | Free-form annotations |

## Module flags

`!llvm.module.flags` is a special list. Each entry is a tuple
`{ behavior, "name", value }`:

```llvm
!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"wchar_size", i32 4}              ; Error if mismatched
!1 = !{i32 7, !"PIC Level", i32 2}               ; Require value (highest wins)
!2 = !{i32 8, !"PIE Level", i32 2}               ; Min-of-values
```

`behavior` is an integer encoding what to do on a mismatch when
modules are linked. Codes are: `1` (error), `2` (warning), `3`
(require), `4` (override), `5` (append), `6` (append-unique),
`7` (max), `8` (min).

## Combining comments and metadata

A practical example, distilled:

```llvm
source_filename = "factorial.c"

; Recursive factorial. Debug info attached for source mapping.
define i32 @factorial(i32 %n) !dbg !5 {
entry:
  %cmp = icmp sle i32 %n, 1, !dbg !6
  br i1 %cmp, label %base, label %recurse, !dbg !7

base:
  ret i32 1, !dbg !8

recurse:
  %nm1     = sub i32 %n, 1, !dbg !9
  %sub     = call i32 @factorial(i32 %nm1), !dbg !10
  %result  = mul i32 %n, %sub, !dbg !11
  ret i32 %result, !dbg !12
}

!llvm.dbg.cu       = !{!0}
!llvm.module.flags = !{!100}
!100 = !{i32 2, !"Dwarf Version", i32 4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true)
!1 = !DIFile(filename: "factorial.c", directory: "/src")
!5 = distinct !DISubprogram(name: "factorial", scope: !1, file: !1, line: 1, type: !20, unit: !0)
!6 = !DILocation(line: 2, column: 10, scope: !5)
!7 = !DILocation(line: 2, column: 3,  scope: !5)
!8 = !DILocation(line: 3, column: 5,  scope: !5)
!9 = !DILocation(line: 5, column: 20, scope: !5)
!10 = !DILocation(line: 5, column: 10, scope: !5)
!11 = !DILocation(line: 5, column: 5,  scope: !5)
!12 = !DILocation(line: 6, column: 3,  scope: !5)
!20 = !DISubroutineType(types: !21)
!21 = !{!22, !22}
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
```

## Pitfalls

- **Bare `"string"` instead of `!"string"` in a metadata tuple.** The
  bare form is a regular string constant; metadata requires the `!`
  prefix. `llvm-as` error: *"expected metadata operand"*. This was a
  real bug in BCIR's `bcir_master_reference_v2.ll` — see
  `08-pitfalls/01-nested-instruction-expressions.md` and the fix in
  commit `1f62e86`.

- **No block comments.** `/* ... */` does not parse.

- **Forgetting `distinct` on a debug-info node** when uniquing would
  break the debugger's ability to distinguish two source entities.
  Default is to dedupe.

- **Referencing a metadata ID that doesn't exist.** All `!N` must
  resolve.

- **Attaching `!dbg` without a `!llvm.dbg.cu` in the module.** Tools
  expect at least one compile unit to anchor the graph.

## See also

- `01-modules-functions-blocks.md` — where named metadata lists live
- `reference/glossary.md` — metadata terminology
- `10-grammar/llvm-ir.tm` — every specialized DI node's field list
