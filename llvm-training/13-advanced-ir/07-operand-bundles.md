# Operand Bundles on Calls and Invokes

Operand bundles attach extra, optimizer-visible operands to a call site without
making those operands normal callee parameters. They are part of the `call`,
`invoke`, or `callbr` instruction that carries them, and they are written after
the callee argument list and call-site function attributes.

This chapter focuses on the `call` and `invoke` forms because those are the
forms most frontends and IR-rewriting passes encounter.

## Call-site syntax

The general syntax is a bracketed, comma-separated list of named bundles:

```llvm
[ "tag"(<typed operand>, ...), "other-tag"() ]
```

Each bundle has:

- a string tag, such as `"deopt"` or `"funclet"`;
- zero or more typed operands; and
- call-site scope, meaning the bundle belongs to that one call-like instruction,
  not to the callee declaration.

For ordinary `call`, put the bundle list after the argument list and any
call-site function attributes:

```llvm
declare void @callee(ptr)

define void @call_with_deopt(i32 %id, ptr %state) {
entry:
  call void @callee(ptr %state) [ "deopt"(i32 %id, ptr %state) ]
  ret void
}
```

For `invoke`, put the bundle list before the `to` and `unwind` labels:

```llvm
declare i32 @may_throw(ptr)
declare i32 @personality(...)

define i32 @invoke_with_deopt(i32 %id, ptr %state) personality ptr @personality {
entry:
  %r = invoke i32 @may_throw(ptr %state) [ "deopt"(i32 %id, ptr %state) ]
          to label %ok unwind label %lpad

ok:
  ret i32 %r

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret i32 -1
}
```

The common built-in tags carry different contracts:

| Bundle | Shape | Meaning |
| --- | --- | --- |
| `"deopt"` | `[ "deopt"(i32 %id, ptr %state) ]` | Deoptimization state associated with this call site. Optimizers must keep enough information for the runtime or deoptimizer to reconstruct the source-level state if execution deoptimizes here. |
| `"funclet"` | `[ "funclet"(token %pad) ]` | Windows EH funclet membership. The token is the active `catchpad` or `cleanuppad`; calls inside that funclet must carry the matching token. |
| `"gc-live"` | `[ "gc-live"(ptr %obj) ]` | GC live-root information, valid on `gc.statepoint` calls rather than arbitrary runtime calls. Treat it as part of the statepoint's liveness contract. |
| `"clang.arc.attachedcall"` | `[ "clang.arc.attachedcall"(ptr @callee) ]` | Clang ARC call pairing used by Objective-C ARC lowering. It records that special retain/claim behavior is attached to the call result. |

Because some tags have verifier restrictions, do not move a spelling from prose
onto an arbitrary call and assume it will assemble. For example, a `gc-live`
bundle belongs on `llvm.experimental.gc.statepoint`, while `funclet` needs a
real EH pad token in the active funclet.

## Why bundles affect optimizer freedom

Operand bundles are not comments. A call with operand bundles can have semantics
that are not visible from the callee type or ordinary argument list:

- A `"deopt"` bundle may mention values that are not passed to the callee but
  must remain available at a potential deoptimization point.
- A `"funclet"` bundle identifies the EH funclet that owns the call. Dropping or
  substituting the token can make Windows EH lowering incorrect.
- A `"gc-live"` bundle keeps GC roots visible to statepoint rewriting and stack
  map generation.
- A `"clang.arc.attachedcall"` bundle changes how ARC runtime calls are paired
  with a call result.
- Unknown or frontend-defined bundle tags must still be treated conservatively
  unless the transform has tag-specific knowledge.

As a result, transformations must not reason about the call from only the callee
operand and ordinary call arguments. The bundle operands can block deletion,
hoisting, sinking, merging, tail-call formation, inlining cleanup, or call
replacement unless the pass proves the transformed call preserves the same
call-site contract.

## Rewriting calls safely

When a pass clones, outlines, replaces, or changes a call-like instruction, use
this checklist:

1. **Copy operand bundles by default.** If the new instruction represents the
   same dynamic call site, preserve the exact bundle tags and operands.
2. **Update SSA operands deliberately.** If normal operands are remapped through
   cloning or inlining, remap bundle operands through the same value map.
3. **Do not invent restricted bundles.** Only create `"funclet"`, `"gc-live"`,
   or ARC bundles in the lowering pipeline that owns the corresponding EH, GC,
   or language-runtime invariant.
4. **Do not drop bundles for convenience.** Removing a bundle is a semantic
   change. It needs the same level of justification as removing an argument,
   attribute, personality, or EH edge.
5. **Preserve bundles across `call`/`invoke` conversions.** If a call is changed
   to an `invoke`, or an `invoke` is simplified to a `call`, carry the bundle
   list unless the transformation also eliminates the semantic reason for the
   bundle.

## Minimal examples

The standalone fixtures are intentionally tiny and assembly-oriented:

- [`examples/operand-bundles-deopt.ll`](examples/operand-bundles-deopt.ll)
  shows `"deopt"` bundles on both `call` and `invoke`.
- [`examples/operand-bundles-funclet.ll`](examples/operand-bundles-funclet.ll)
  shows a `"funclet"` bundle on a call inside a Windows cleanup funclet.

Check them with:

```bash
llvm-as llvm-training/13-advanced-ir/examples/operand-bundles-deopt.ll -o /dev/null
llvm-as llvm-training/13-advanced-ir/examples/operand-bundles-funclet.ll -o /dev/null
opt -passes=verify llvm-training/13-advanced-ir/examples/operand-bundles-deopt.ll -o /dev/null
opt -passes=verify llvm-training/13-advanced-ir/examples/operand-bundles-funclet.ll -o /dev/null
```

## BCIR checklist

For BCIR-generated or BCIR-transformed LLVM IR:

- Treat operand bundles as part of the call-site semantic payload, not as
  metadata-like decoration.
- Preserve bundle operands when wrapping runtime calls, redirecting callees, or
  splitting exceptional edges.
- Keep GC and deoptimization state in explicit operands when the runtime needs
  it after optimization.
- Add regression examples whenever a lowering pass rewrites calls with bundles;
  a missing bundle is often verifier-valid IR but semantically wrong IR.
