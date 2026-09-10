# Conditional Branch (`br i1 %cond, label %t, label %f`)

## TL;DR

```llvm
br i1 %cond, label %if_true, label %if_false
```

Evaluates an `i1` condition and jumps to one of two labels. Used for
`if`/`else`, loop exit conditions, and any 2-way branch.

## Syntax

```
br i1 <cond>, label %<true-target>, label %<false-target>
```

- `<cond>` is any `i1` value (constant, instruction result, etc.).
- Both labels must be valid basic blocks in the same function.
- Terminator — must end its block.

## Producing the condition

The condition is almost always the result of `icmp` or `fcmp`:

```llvm
%cmp = icmp sgt i32 %a, %b               ; signed greater-than
br i1 %cmp, label %a_wins, label %b_wins
```

```llvm
%nz  = icmp ne i32 %x, 0
br i1 %nz, label %nonzero, label %zero
```

But it can also be a Boolean value passed in, a load of an `i1` from
memory, or a constant:

```llvm
br i1 1, label %always, label %never     ; constant — `opt` will fold
br i1 %arg, label %take_path, label %skip
```

## Example: if/else

```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %a_wins, label %b_wins

a_wins:
  ret i32 %a

b_wins:
  ret i32 %b
}
```

## Example: loop with exit condition

```llvm
define i32 @sum(i32 %n) {
entry:
  br label %loop

loop:
  %i        = phi i32 [ 0, %entry ], [ %i_next,   %loop ]
  %sum      = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %sum_next = add i32 %sum, %i
  %i_next   = add i32 %i, 1
  %done     = icmp eq i32 %i_next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum_next
}
```

## Branch weights

You can hint the optimizer about branch probabilities via metadata:

```llvm
br i1 %cmp, label %hot, label %cold, !prof !0
!0 = !{!"branch_weights", i32 99, i32 1}
```

The weights need not sum to 100; their ratio matters.

## Combined patterns

### Short-circuit `&&`

```llvm
%a_ok = icmp ne i32 %a, 0
br i1 %a_ok, label %check_b, label %both_zero

check_b:
  %b_ok = icmp ne i32 %b, 0
  br i1 %b_ok, label %neither_zero, label %both_zero
```

### Short-circuit `||`

```llvm
%a_ok = icmp ne i32 %a, 0
br i1 %a_ok, label %any_nonzero, label %check_b

check_b:
  %b_ok = icmp ne i32 %b, 0
  br i1 %b_ok, label %any_nonzero, label %both_zero
```

### `select` for ternary expression (no branch)

If both arms are cheap and side-effect-free, prefer `select`:

```llvm
%r = select i1 %cmp, i32 %a, i32 %b
```

This avoids a branch entirely.

## Pitfalls

- **Condition not `i1`.** `br i32 %v, ...` doesn't parse. If you have
  a wider value, narrow it: `%c = icmp ne i32 %v, 0; br i1 %c, ...`.

- **Same label for both arms.** `br i1 %c, label %x, label %x` parses
  but is suspicious — same destination either way. Use `br label %x`.

- **Forgetting phi node update after adding a branch.** If you add a
  new edge into a block that has a phi, you must add a new
  `[ value, predecessor ]` entry. The verifier catches this — see
  [`../08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md).

- **Putting code after `br`.** Unreachable; verifier rejects.

- **Branching across functions.** Not allowed; labels are function-
  scoped.

## See also

- [`01-unconditional-br.md`](01-unconditional-br.md)
- [`../00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) — phi and predecessor lists
- [`03-switch.md`](03-switch.md) — for multi-way branching
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — `icmp`/`fcmp` predicates
