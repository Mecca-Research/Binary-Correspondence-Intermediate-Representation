# Indirect Branching (`indirectbr ptr %addr, [label %1, label %2]`)

## TL;DR

```llvm
indirectbr ptr %addr, [ label %bb1, label %bb2, label %bb3 ]
```

Branches to a basic block whose address is computed at runtime.
`%addr` must be a value of type `ptr` that was produced by the
`blockaddress` constant expression. The bracketed list enumerates
**all possible** targets — anything else is UB.

Used to implement jump tables (interpreters, threaded code),
computed `goto` (GCC extension), and certain coroutine patterns.

## Syntax

```
indirectbr ptr <address>, [ label %<target>, label %<target>, ... ]
```

- Address operand is a `ptr`.
- Target list **must** contain every possible target. Missing targets
  cause undefined behavior, not a verifier error.
- Terminator — last instruction in its block.

## Producing block addresses

The `blockaddress` constant expression returns a `ptr` to a basic
block in a named function:

```llvm
@jumptable = global [3 x ptr] [
  ptr blockaddress(@interpret, %op_add),
  ptr blockaddress(@interpret, %op_sub),
  ptr blockaddress(@interpret, %op_ret)
]
```

`blockaddress(@func, %label)` — the function and the label must
match.

## Example: a tiny interpreter dispatch

```llvm
@dispatch = global [3 x ptr] [
  ptr blockaddress(@run, %op_a),
  ptr blockaddress(@run, %op_b),
  ptr blockaddress(@run, %op_c)
]

define void @run(i32 %op) {
entry:
  %idx_ptr = getelementptr inbounds [3 x ptr], ptr @dispatch, i32 0, i32 %op
  %target  = load ptr, ptr %idx_ptr, align 8
  indirectbr ptr %target, [ label %op_a, label %op_b, label %op_c ]

op_a:
  ; ...
  ret void
op_b:
  ; ...
  ret void
op_c:
  ; ...
  ret void
}
```

## Threaded interpreters

Each "opcode handler" ends in another `indirectbr` to the next
instruction:

```llvm
; (sketch — pseudocode-ish)
op_add:
  ; do the add
  %next_op_p = getelementptr i32, ptr %pc_ptr, i32 1
  %next_op   = load i32, ptr %next_op_p, align 4
  %next_p    = getelementptr inbounds [N x ptr], ptr @dispatch, i32 0, i32 %next_op
  %next_t    = load ptr, ptr %next_p, align 8
  indirectbr ptr %next_t, [ label %op_add, label %op_sub, ... ]
```

This is how "direct-threaded" interpreters work; each handler dispatches
the next opcode without going back through a central loop.

## Why both an address operand and a target list?

The target list lets the verifier and optimizer reason about
control flow without inferring it from the address. It also limits
the legal set of targets — anything outside the list is UB, which
allows the compiler to assume the branch only goes where listed.

## Pitfalls

- **Incomplete target list.** Listing only some of the possible
  targets and then branching to an unlisted one is UB. Optimizer may
  do wild things.

- **`blockaddress` for a deleted block.** If a transformation removes
  a block whose address was taken, the resulting `blockaddress(...)`
  is bad. Generally LLVM keeps such blocks alive automatically, but
  pass-pipeline interactions can surprise you.

- **Taking the address of `entry`.** Some passes have edge cases
  around `blockaddress(@f, %entry)`. Safer to take addresses of
  inner blocks.

- **Forgetting that `indirectbr` is a terminator.** Anything after
  it in the same block is unreachable.

- **Using a non-`ptr` operand.** The address must be a `ptr` (the
  result of a `blockaddress` or computation from one). An integer
  address requires `inttoptr` first.

## See also

- `02-conditional-br.md`, `03-switch.md` — direct alternatives
- `reference/instruction-quickref.md` — `blockaddress` constant
  expression
- `01-syntax/01-modules-functions-blocks.md` — block labels are
  function-scoped
