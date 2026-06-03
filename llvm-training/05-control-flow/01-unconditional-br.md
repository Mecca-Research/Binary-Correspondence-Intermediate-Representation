# Unconditional Branch (`br label %target`)

## TL;DR

```llvm
br label %target
```

Transfers control to a labeled basic block within the same function.
No condition is evaluated. Used to chain blocks, jump back to a loop
header, or merge control flow after `if`-style splits.

## Syntax

```
br label %<block-name>
```

It's a **terminator** — must be the last instruction in its basic
block.

## Examples

### Chain two blocks

```llvm
define void @demo() {
entry:
  br label %work

work:
  ; some instructions
  ret void
}
```

### Loop back

```llvm
define void @forever() {
entry:
  br label %loop

loop:
  ; do something
  br label %loop          ; infinite loop
}
```

### Common after if/else

```llvm
define i32 @abs(i32 %x) {
entry:
  %neg = icmp slt i32 %x, 0
  br i1 %neg, label %negate, label %merge

negate:
  %n = sub i32 0, %x
  br label %merge

merge:
  %r = phi i32 [ %n, %negate ], [ %x, %entry ]
  ret i32 %r
}
```

The unconditional `br label %merge` from `negate` is what makes the
phi node well-defined (two predecessors: `entry` and `negate`).

## When to use vs not

✅ Falling through to the next block
✅ Loop back-edges
✅ Merging control flow after a conditional

❌ Where execution should just continue — basic blocks don't merge
   implicitly; the branch is required, but you might be over-fragmenting

## Pitfalls

- **Forgetting the branch.** If your basic block doesn't end in a
  terminator, the verifier rejects with *"Block does not have a
  terminator"*.

- **Branching to a label in another function.** Labels are
  function-scoped; you can't `br` across functions. Use a `call`
  for that.

- **Putting code after `br`.** Anything after a terminator in the
  same block is unreachable; the verifier rejects it.

## See also

- [`02-conditional-br.md`](02-conditional-br.md) — conditional version
- [`../00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) — `phi` requires predecessor labels
- [`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) — block structure
