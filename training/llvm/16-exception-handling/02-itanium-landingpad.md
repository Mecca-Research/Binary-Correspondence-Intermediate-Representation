# Itanium-Style `invoke`, `landingpad`, and `resume`

Itanium-style EH represents exceptional control flow with an `invoke` terminator
and a `landingpad` instruction in the unwind destination.

## `invoke`

`invoke` is a call and a terminator at the same time:

```llvm
%value = invoke i32 @may_throw(i32 %x)
           to label %normal unwind label %lpad
```

- The **normal successor** receives control when the callee returns normally.
- The **unwind successor** receives control when an exception propagates into the
  caller.
- If the callee returns a value, the `invoke` result is available only along the
  normal edge.
- Operand bundles, when present, are written before `to label`.

Use plain `call` when unwinding is irrelevant to the IR being modeled. Use
`invoke` when cleanup, catch, or rethrow behavior is part of the function CFG.

## `landingpad`

A `landingpad` must be the first non-`phi` instruction in its landing pad block,
and that block is normally an unwind destination of an `invoke`.

```llvm
%lp = landingpad { ptr, i32 }
        cleanup
        catch ptr null
```

The result type is personality-dependent. For common C++ Itanium examples,
`{ ptr, i32 }` represents the exception object pointer plus a selector value.
Clauses describe what this pad can handle:

| Clause | Meaning |
| --- | --- |
| `cleanup` | Run cleanup code even if the exception is not caught here. |
| `catch <ty> <value>` | Match a catch type encoded for the active personality. |
| `filter <array-ty> <value>` | Legacy filter list form used by older EH schemes. |

A landing pad may branch to local handling code, return after catching, or use
`resume` to continue unwinding.

## `resume`

`resume` is the terminator that rethrows the exception package produced by the
landing pad:

```llvm
resume { ptr, i32 } %lp
```

Use it after cleanup-only work when the current function does not consume the
exception. Do not construct an arbitrary value for `resume`; preserve the package
from the `landingpad` unless a frontend/personality-specific lowering has a
well-defined reason to rewrite it.

## Minimal shape

See [`examples/invoke-landingpad.ll`](examples/invoke-landingpad.ll) for a pad
that catches locally and [`examples/cleanup-resume.ll`](examples/cleanup-resume.ll)
for a cleanup that resumes propagation.

## Common pitfalls

- Replacing `invoke` with `call` drops the exceptional CFG edge and can remove
  required cleanups.
- Moving ordinary instructions before `landingpad` breaks the pad placement rule.
- Duplicating an `invoke` without cloning its operand bundles changes call-site
  semantics.
- Splitting a landing pad block requires care: keep the `landingpad` in the pad
  block and move ordinary work to a successor block.
