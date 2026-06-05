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

## Common `DIExpression` operators

A `DIExpression` is a small DWARF-like expression that explains how to
interpret the value named by `llvm.dbg.value`, `llvm.dbg.declare`, or a
debug record. Empty `!DIExpression()` means the IR value is the source
value directly. Non-empty expressions are common after SROA, register
allocation preparation, address calculation, and debug-info salvage.

| Operator | Typical shape | Meaning | Common use |
|---|---|---|---|
| `DW_OP_plus_uconst` | `!DIExpression(DW_OP_plus_uconst, 8)` | Add an unsigned byte offset to the current address/value. | Describe a field, stack slot offset, or salvaged address computation. |
| `DW_OP_deref` | `!DIExpression(DW_OP_deref)` | Treat the current value as an address and load the value found there. | Turn a pointer-valued location into the source value it points at. |
| `DW_OP_stack_value` | `!DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)` | The expression result is the variable value itself, not an address to dereference. | Salvage optimized computations such as `x + 4` when no materialized instruction remains. |
| `DW_OP_LLVM_fragment` | `!DIExpression(DW_OP_LLVM_fragment, 32, 16)` | This debug value covers only a bit slice: offset 32, size 16 in this example. | Describe one piece of an aggregate, split scalar, or partially available variable. |
| `DW_OP_LLVM_arg` | `!DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)` | Refer to explicit operands of a multi-value debug record/expression. | Combine several IR values into one source variable description after optimization. |

### Variable fragments

`DW_OP_LLVM_fragment` lets LLVM describe a source variable whose storage is
split across several IR values or whose value is only partially known. The
two operands are a bit offset and a bit size within the source variable. For
example, `!DIExpression(DW_OP_LLVM_fragment, 0, 32)` describes the low
32-bit piece of a larger variable, while `!DIExpression(DW_OP_LLVM_fragment,
32, 32)` describes the next 32-bit piece.

Fragments are especially important after scalar replacement of aggregates,
load splitting, vector lane extraction, or partial dead-code elimination. A
pass may preserve a debug value for the live pieces while dropping fragments
that are no longer recoverable. Debuggers can then show the known pieces of
a variable instead of either pretending that the whole variable is available
or losing all variable visibility.

See the fragment example: [`examples/debug-variable-fragments.ll`](examples/debug-variable-fragments.ll).

## Preserving DI through optimization

Optimization passes should treat debug info as semantic information, not as
decoration to copy mechanically. For every transformed instruction and
variable location, decide whether the old debug information is still true,
needs an updated expression, can be salvaged, or must be dropped. See also
[`../08-pitfalls/08-stale-debug-locations.md`](../08-pitfalls/08-stale-debug-locations.md)
for a focused discussion of stale line locations.

### Variable-location intrinsics and records

- **`llvm.dbg.declare`** describes a source variable that lives at an
  address, such as an `alloca`. It is a good fit before optimization or
  while the memory object remains the canonical variable location. If a pass
  promotes, splits, or removes that memory object, keeping the old declare can
  make the debugger read stale storage. Convert it to value-based locations,
  fragment it, or drop it.
- **`llvm.dbg.value`** describes the current value of a source variable. It
  is the usual representation after mem2reg, SROA, instruction combining,
  and other value-transforming passes. Update the `DIExpression` when the IR
  value is still recoverable but no longer exactly identical to the source
  value.
- **Debug records** are the newer in-IR representation for the same concepts.
  The preservation rules are the same: addresses must still point at the
  variable, values must still compute the variable, and fragments must not
  overlap incorrectly or outlive the piece they describe.

### Dropped, stale, and salvaged locations

- **Dropped locations are better than wrong locations.** If an instruction is
  newly synthesized, hoisted into a place where the original line is
  misleading, or no longer corresponds to a user-visible operation, omit its
  `!dbg` attachment or set the debug location to an intentional compiler-
  generated scope.
- **Stale locations are accepted by the verifier but harmful to users.** A
  cloned instruction with an old `!dbg` may still assemble, yet breakpoints,
  profiles, diagnostics, and correspondence tools can attribute behavior to
  the wrong source line. Cross-check cloned, sunk, hoisted, outlined, and
  inlined instructions against the stale-location pitfall.
- **Salvaged debug info is useful when the value can still be expressed.** If
  an optimized-away instruction computed `x + 4`, a pass may keep the variable
  visible with a `DIExpression` such as `DW_OP_plus_uconst, 4,
  DW_OP_stack_value`. Salvage only when the expression is still precise; if
  the value is approximate or depends on removed side effects, drop it.

See the optimization example: [`examples/debug-info-optimization.ll`](examples/debug-info-optimization.ll).

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
- [`examples/debug-variable-fragments.ll`](examples/debug-variable-fragments.ll)
- [`examples/debug-info-optimization.ll`](examples/debug-info-optimization.ll)
- [`../08-pitfalls/08-stale-debug-locations.md`](../08-pitfalls/08-stale-debug-locations.md)
- [`01-metadata-basics.md`](01-metadata-basics.md)
- [Source Level Debugging with LLVM](https://llvm.org/docs/SourceLevelDebugging.html)
- [LLVM LangRef — Metadata](https://llvm.org/docs/LangRef.html#metadata)
