# Pitfall 08 — Stale Debug Locations After Rewrites

## The symptom

```text
breakpoint stops on the wrong source line
```

Or, in optimization diagnostics and debugger views:

```text
instruction attributed to a line that no longer computes that value
```

This is usually valid IR, so `llvm-as` and the verifier may accept it. The bug is
semantic: tools now tell users the wrong story.

## Minimal reproducer

```llvm
; Imagine line 10 is "x = a + b" and line 40 is "return x".
define i32 @f(i32 %a, i32 %b) !dbg !4 {
entry:
  %sum = add i32 %a, %b, !dbg !10
  br label %exit, !dbg !10

exit:
  ; A generator or pass rewrote the return value but kept the old add location.
  %normalized = and i32 %sum, 255, !dbg !10   ; ❌ line 10 is stale here
  ret i32 %normalized, !dbg !40
}

!10 = !DILocation(line: 10, column: 7, scope: !4)
!40 = !DILocation(line: 40, column: 3, scope: !4)
```

The `and` may be a real lowered operation, but source line 10 no longer describes
what it does.

## Why it happens

Many transformations clone or move instructions. If the transform blindly copies
`!dbg`, the new instruction inherits a source location from an instruction with a
different meaning. Code motion makes this worse: a location that was true before
hoisting, sinking, outlining, or inlining may be misleading afterward.

LLVM preserves debug info best when a pass deliberately decides whether each new
instruction has the same source meaning, a merged meaning, or no reliable source
location.

## Fix pattern

Use one of three policies for every generated instruction:

1. **Preserve** the old `!dbg` only when the new instruction represents the same
   source operation.
2. **Replace** it with a location for the new source-level construct, wrapper, or
   synthetic helper scope.
3. **Drop** it when keeping a location would mislead users.

For the example above:

```llvm
%normalized = and i32 %sum, 255             ; ✓ no misleading !dbg
ret i32 %normalized, !dbg !40
```

If the instruction is compiler-generated but still useful in a debugger, attach a
stable artificial or generated-code scope rather than a user's unrelated line.

## BCIR-relevant note

BCIR may map IR back to machine addresses instead of source lines. The same rule
applies: do not keep a machine-address location after a rewrite if the new IR no
longer corresponds to that instruction. For recovery pipelines, stale locations
can make provenance reports, binary/source correspondence, and validator output
point at the wrong basic block.

## See also

- [`../06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) — stale source-location pitfalls
- [`../07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) — transforms that clone or move IR
- [`../07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md) — pass responsibility and verification points
