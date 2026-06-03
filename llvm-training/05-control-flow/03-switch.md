# Switch Statements (`switch i32 %val, label %default [ ... ]`)

## TL;DR

```llvm
switch i32 %v, label %default [
  i32 0, label %case_zero
  i32 1, label %case_one
  i32 42, label %case_42
]
```

Multi-way branch on an integer value. Each case is a constant integer
paired with a target label. A required `default` label catches the
fallthrough.

## Syntax

```
switch <int-type> <value>, label %<default> [
  <int-type> <const>, label %<target>
  <int-type> <const>, label %<target>
  ...
]
```

- The value being switched is any integer type.
- Each case label uses the **same integer type** as the value.
- Case constants must be **unique**.
- The `default` label is **required** (even if you think it's
  unreachable — use `unreachable` in that block).
- Terminator — last instruction of its block.

## Examples

### Multi-way dispatch

```llvm
define void @state_machine(i32 %state) {
entry:
  switch i32 %state, label %unknown [
    i32 0, label %s0
    i32 1, label %s1
    i32 2, label %s2
  ]

s0:
  ; ...
  ret void
s1:
  ; ...
  ret void
s2:
  ; ...
  ret void
unknown:
  unreachable
}
```

### Enumerated values

```llvm
@.fmt_red   = private unnamed_addr constant [5 x i8] c"red\0A\00"
@.fmt_green = private unnamed_addr constant [7 x i8] c"green\0A\00"
@.fmt_blue  = private unnamed_addr constant [6 x i8] c"blue\0A\00"

declare i32 @puts(ptr)

define void @print_color(i32 %c) {
entry:
  switch i32 %c, label %bad [
    i32 0, label %red
    i32 1, label %green
    i32 2, label %blue
  ]
red:
  call i32 @puts(ptr @.fmt_red)
  ret void
green:
  call i32 @puts(ptr @.fmt_green)
  ret void
blue:
  call i32 @puts(ptr @.fmt_blue)
  ret void
bad:
  unreachable
}
```

### "Fallthrough" via shared labels

LLVM IR switches don't have C-style fallthrough. To get the effect,
point multiple cases at the same label:

```llvm
switch i32 %v, label %default [
  i32 1, label %small
  i32 2, label %small      ; cases 1 and 2 share the small handler
  i32 3, label %small
  i32 100, label %big
]
```

## When to use vs `br` chain

- ✅ Switch — 3+ cases, dense integer values, no fallthrough
- ✅ `br` chain — 1–2 cases, ranges, or non-trivial conditions
- ✅ Indirect branch — runtime-computed targets (`indirectbr`)

The backend chooses between jump-table, binary search, and if-else
chains based on case density.

## Branch weights

```llvm
switch i32 %v, label %default [
  i32 0, label %a
  i32 1, label %b
], !prof !0
!0 = !{!"branch_weights", i32 1, i32 10, i32 89}
                                ; default,   case 0,  case 1
```

The first weight is the default; subsequent weights match case order.

## Pitfalls

- **Duplicate case values.** `i32 0, label %a` and `i32 0, label %b`
  — verifier rejects.

- **Wrong type on a case.** `switch i32 %v, ...` with `i64 0, label
  %a` mismatches; case constants must match the value's type.

- **Missing `default`.** Required. If you've proven all reachable
  cases are listed, point `default` at a block containing
  `unreachable`.

- **Switch as a non-terminator.** Putting code after a `switch` in
  the same block — unreachable, rejected.

- **Massive sparse switches.** Code generation may explode. Consider
  hashing or computed branches for very sparse cases.

## See also

- [`02-conditional-br.md`](02-conditional-br.md) — for 2-way
- [`04-indirectbr.md`](04-indirectbr.md) — for runtime-computed targets
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — `select` as a value-level
  alternative
