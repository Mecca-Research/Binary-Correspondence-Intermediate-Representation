# Static Single Assignment (SSA) Form

## TL;DR

In SSA form, every variable is **defined exactly once**. If a high-level
language reassigns `x = x + 1`, the IR introduces a fresh name (`%x1
= ...`, `%x2 = ...`). When control flow merges values from different
paths, a `phi` node picks the right one based on which predecessor
block we came from.

SSA is what makes LLVM optimizations work. Once you understand `phi`,
you understand 80% of why IR looks the way it does.

## Why SSA?

Without SSA, "what is the value of `x` here?" requires walking back
through control flow. With SSA, it's a one-step lookup: each name
points to exactly one definition.

Concretely, SSA makes these analyses trivial:

- **Constant propagation** — if `%x = add i32 1, 2`, `%x` is just 3.
- **Dead code elimination** — if no instruction uses `%x`, delete it.
- **Common subexpression elimination** — if two `add` instructions
  have identical operands, they produce the same value.
- **Liveness analysis** — a name is live from its definition to its
  last use. No re-definition complication.

## Renaming, by example

Source:
```c
x = 1;
x = x + 2;
y = x * 3;
```

Non-SSA "pseudo-IR":
```
x = 1
x = x + 2
y = x * 3
```

SSA:
```llvm
%x1 = add i32 0, 1         ; (or just use the constant 1)
%x2 = add i32 %x1, 2
%y1 = mul i32 %x2, 3
```

Each name appears as a LHS exactly once.

## The phi instruction

When control flow merges and a value depends on which path was taken,
you need a `phi`. Source:

```c
int x;
if (cond) { x = 1; } else { x = 2; }
y = x + 3;
```

SSA IR:

```llvm
entry:
  br i1 %cond, label %if_true, label %if_false

if_true:
  br label %merge

if_false:
  br label %merge

merge:
  %x = phi i32 [ 1, %if_true ], [ 2, %if_false ]
  %y = add i32 %x, 3
  ret i32 %y
```

A `phi` node:

- **Must be at the start of a basic block** (before any non-phi
  instruction).
- **Has one (value, predecessor-label) pair per incoming edge**.
- **All values must have the same type** as the phi's declared type.

If you forget a predecessor, or list one that doesn't actually branch
to this block, `llvm-as` rejects with *"PHI node entries do not match
predecessors"*. See `08-pitfalls/02-phi-predecessor-mismatch.md`.

## Loops in SSA

A loop counter requires a `phi` because the counter has two
definitions: its initial value and the increment. They merge at the
loop header.

```llvm
define i32 @sum_to(i32 %n) {
entry:
  br label %loop

loop:
  %i      = phi i32 [ 0,  %entry ], [ %i_next,   %loop ]
  %sum    = phi i32 [ 0,  %entry ], [ %sum_next, %loop ]
  %i_next   = add i32 %i, 1
  %sum_next = add i32 %sum, %i
  %done     = icmp eq i32 %i_next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum_next
}
```

Two phis at the loop header: one for the counter, one for the
accumulator. Both list the same two predecessors (`%entry` and
`%loop`).

## What SSA does NOT mean

- ❌ "Variables are immutable in memory."
  SSA only constrains **virtual registers** (the `%foo` names). Memory
  (`alloca`, `load`, `store`) is not in SSA — you can `store` to the
  same address as many times as you like.

- ❌ "You can't read a `%foo` before its definition."
  Forward references work for labels. `br label %later` is fine even
  if `later:` appears farther down. But you can't reference a
  *value-producing* `%foo` before the instruction that defines it.

- ❌ "Loops require lots of phis."
  Often just one or two. Many loop-carried values can be expressed
  without phis (e.g., loop-invariant pointer arithmetic).

## The mem-to-reg pattern

Many frontends initially emit non-SSA-looking IR by putting every
local variable behind `alloca`/`load`/`store`:

```llvm
define i32 @add(i32 %a, i32 %b) {
  %a.addr = alloca i32
  %b.addr = alloca i32
  store i32 %a, ptr %a.addr
  store i32 %b, ptr %b.addr
  %0 = load i32, ptr %a.addr
  %1 = load i32, ptr %b.addr
  %2 = add i32 %0, %1
  ret i32 %2
}
```

The `mem2reg` pass (run by `-O1` and above) promotes the `alloca`s
into SSA registers and inserts phi nodes where needed. This is the
canonical way to lower mutable locals into SSA form.

## Pitfalls

- **phi node not at block start.** All phis must appear before any
  non-phi instruction in a block. Move them up.

- **Mismatched predecessor list.** If block `%merge` is reachable from
  three places, every phi in `%merge` must have three incoming pairs.

- **Trying to "reassign" a `%foo` later.** Make a new name: `%foo1`,
  `%foo2`. The verifier rejects redefinition.

- **Forgetting that constants don't need SSA names.** `add i32 5, 10`
  is fine — no `%five = ...` needed.

## See also

- `01-what-is-llvm-ir.md` — broader context
- `05-control-flow/02-conditional-br.md` — branches that feed phis
- `08-pitfalls/02-phi-predecessor-mismatch.md` — the real-world error
  message and fix
- `examples/ssa-phi.ll` — the if/else example, runnable
