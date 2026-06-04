# Debug Info Metadata

## TL;DR

LLVM source-level debug info is a metadata graph anchored by
`!llvm.dbg.cu`. Instructions and functions use `!dbg` attachments to
point into that graph. The most important practical path is:

```text
instruction --!dbg--> DILocation --scope--> DISubprogram --file--> DIFile
```

Official reference: [Source Level Debugging with LLVM](https://llvm.org/docs/SourceLevelDebugging.html).

See the standalone example: [`examples/debug-location.ll`](examples/debug-location.ll).

## Minimal debug-info graph

A small function with locations usually includes these nodes:

| Node | Role |
|---|---|
| `DIFile` | Source file name and directory. |
| `DICompileUnit` | Compilation-unit root; listed in `!llvm.dbg.cu`. |
| `DISubprogram` | Source-level function/subprogram; usually attached to `define ... !dbg !N`. |
| `DILocation` | Source line/column/scope; attached to instructions as `!dbg !N`. |
| `DILocalVariable` | A source variable, often used by `llvm.dbg.value` or debug records. |

```llvm
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1,
                             producer: "clang", isOptimized: false,
                             runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
```

## `DIFile`

`DIFile` names the source file:

```llvm
!1 = !DIFile(filename: "main.c", directory: "/src/project")
```

To read the path, combine `directory` and `filename`. If `filename` is
absolute, tools may treat it as already complete. Generated code should
prefer stable, reproducible paths when possible.

## `DICompileUnit`

`DICompileUnit` is the root for debug info from one source compilation:

```llvm
!0 = distinct !DICompileUnit(language: DW_LANG_C11,
                             file: !1,
                             producer: "my-frontend",
                             isOptimized: false,
                             runtimeVersion: 0,
                             emissionKind: FullDebug)
!llvm.dbg.cu = !{!0}
```

If IR contains `!dbg` locations but no compile unit, debuggers and
post-processing tools do not have a complete graph to interpret.

## `DISubprogram`

`DISubprogram` describes a source function and acts as the scope for its
locations and variables:

```llvm
define i32 @add_one(i32 %x) !dbg !5 {
  ; ...
}

!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one",
                            scope: !1, file: !1, line: 3,
                            type: !6, scopeLine: 3,
                            flags: DIFlagPrototyped,
                            spFlags: DISPFlagDefinition,
                            unit: !0, retainedNodes: !9)
```

The `scope` and `file` usually point to the containing `DIFile` or a
lexical scope; `unit` points back to the `DICompileUnit`.

## `DILocation`

`DILocation` maps an IR instruction to a source line and column within a
scope:

```llvm
%y = add i32 %x, 1, !dbg !13
!13 = !DILocation(line: 4, column: 12, scope: !5)
```

A debugger resolves this as:

1. Read the instruction's `!dbg !13` attachment.
2. Open `!13 = !DILocation(line: 4, column: 12, scope: !5)`.
3. Follow `scope: !5` to the `DISubprogram` or lexical scope.
4. Follow the scope's `file: !1` to `DIFile(filename: ..., directory: ...)`.
5. Report `directory/filename:line:column`.

For the example above, if `!1` is `DIFile(filename: "debug-location.c",
directory: "/workspace/examples")`, the location is:

```text
/workspace/examples/debug-location.c:4:12
```

## `DILocalVariable`

A local variable node describes source variable identity:

```llvm
call void @llvm.dbg.value(metadata i32 %x, metadata !10,
                          metadata !DIExpression()), !dbg !12

!10 = !DILocalVariable(name: "x", arg: 1, scope: !5,
                       file: !1, line: 3, type: !8)
```

`DILocalVariable` says what source variable is being described. The debug
value record or intrinsic says which IR value currently represents it.
Modern LLVM also has debug records in addition to the older intrinsic
spelling; source-location reasoning still flows through the metadata
nodes above.

## Pitfalls

- **Malformed debug metadata.** Missing `!llvm.dbg.cu`, missing module
  flags, broken references, or wrong specialized-node fields can produce
  assembler, verifier, or downstream debug failures.
- **Stale source locations.** After inlining, outlining, code motion, or
  generated-code rewrites, an instruction may no longer correspond to
  the original source line. Preserve locations carefully, or drop them
  when they become misleading.
- **Scope/file mismatches.** A `DILocation` can point to a scope whose
  `file` is not the line's actual source file. That yields confusing
  breakpoints and diagnostics.
- **Assuming every instruction has `!dbg`.** Optimized IR often has
  instructions with no useful location.
- **Assuming metadata survives optimization.** Passes may drop debug
  locations they cannot preserve accurately.

## See also

- [`examples/debug-location.ll`](examples/debug-location.ll)
- [`01-metadata-basics.md`](01-metadata-basics.md)
- [Source Level Debugging with LLVM](https://llvm.org/docs/SourceLevelDebugging.html)
- [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata)
