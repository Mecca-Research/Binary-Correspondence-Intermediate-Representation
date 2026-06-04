# Pitfall 07 — Debug Metadata Bloat

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=verify <bcir-output-with-debug-metadata>.ll -o /dev/null` plus debug-info size checks | Uniquify common debug metadata and avoid emitting duplicate `DILocation` nodes for equivalent locations. | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md); [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md); [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) |

## The symptom

```text
large .ll file full of thousands of nearly identical !DILocation nodes
```

Or downstream:

```text
object/debug sections are unexpectedly huge
```

This is usually not a verifier error. The IR is valid, but assembly,
optimization, object emission, and debugger startup become much slower than the
program merits.

## Minimal reproducer

```llvm
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1,
                             producer: "toy", isOptimized: false,
                             runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "bloat.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1,
                            type: !5, scopeLine: 1,
                            spFlags: DISPFlagDefinition, unit: !0,
                            retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}

; These all describe the same source position but are separate nodes.
!10 = !DILocation(line: 2, column: 3, scope: !4)
!11 = !DILocation(line: 2, column: 3, scope: !4)
!12 = !DILocation(line: 2, column: 3, scope: !4)

define i32 @f(i32 %x) !dbg !4 {
entry:
  %a = add i32 %x, 1, !dbg !10
  %b = add i32 %a, 2, !dbg !11
  %c = add i32 %b, 3, !dbg !12
  ret i32 %c, !dbg !12
}
```

A generator that emits a fresh `!DILocation` for every instruction in a large
module can create millions of metadata nodes even when only a few source
locations are distinct.

## Why it happens

Debug metadata is part of the IR metadata graph. `!dbg` attachments are cheap
when they reuse canonical metadata nodes, but expensive when a generator creates
new equivalent nodes for every instruction. This commonly happens when a printer
has no interning table and treats debug-location emission like instruction
emission.

`distinct` nodes are especially important: they are intentionally unique and
must not be used for every location unless uniqueness is semantically required.
Ordinary `DILocation` nodes can usually be uniqued by LLVM when constructed
through the C++ APIs, but text generators can still bloat output by spelling many
separate numbered nodes.

## Fix pattern

Intern debug metadata by semantic key:

```text
(file, subprogram-or-lexical-scope, line, column, inlinedAt) -> !DILocation
```

Then reuse the same metadata ID for repeated positions:

```llvm
!10 = !DILocation(line: 2, column: 3, scope: !4)

define i32 @f(i32 %x) !dbg !4 {
entry:
  %a = add i32 %x, 1, !dbg !10
  %b = add i32 %a, 2, !dbg !10
  %c = add i32 %b, 3, !dbg !10
  ret i32 %c, !dbg !10
}
```

Also consider lowering the debug-info emission mode for generated helper code,
or dropping locations from mechanically generated instructions that do not map to
useful source positions.

## BCIR-relevant note

BCIR-style translators often create many small IR helper operations for one
binary instruction or one recovered semantic step. Attach debug locations to the
source-level or binary-level event you want users to see, not to every temporary
helper value. If the same machine address or recovered source span explains many
IR instructions, reuse one location node.

## See also

- [`../06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) — metadata graph basics
- [`../06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) — `DICompileUnit`, `DISubprogram`, and `DILocation`
- [`../01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) — metadata attachments in instruction syntax
