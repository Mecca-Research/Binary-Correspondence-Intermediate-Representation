# Pitfall 02 — PHI Node Predecessor Mismatch

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/kernel/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| `runtime/llvm/bcir_batch_executor.ll` | `5754354` | `llvm-as runtime/llvm/bcir_batch_executor.ll -o /dev/null` | Make phi incoming labels match the block's actual CFG predecessors. | [`00-foundations/02-ssa.md`](../00-foundations/02-ssa.md); [`05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md); [`05-control-flow/01-unconditional-br.md`](../05-control-flow/01-unconditional-br.md) |

## The error

```
llvm-as: assembly parsed, but does not verify as correct!
PHI node entries do not match predecessors!
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
```

## What's happening

A `phi` instruction must list **one (value, predecessor) pair for
every predecessor that actually flows into the block**. If your CFG
adds a new edge into a block (e.g., a side path), every phi in that
block has to learn about the new predecessor.

Symmetric bug: listing a predecessor that doesn't actually branch to
this block.

## Minimal reproducer

```llvm
define void @demo() {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]   ; expects %body to branch here
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, 10
  br i1 %done, label %exit, label %prefetch

prefetch:
  br label %cont               ; ⟵ this edge flows into %cont, not back to %loop

cont:
  br label %loop               ; ⟵ here's the actual back-edge

exit:
  ret void
}
```

The phi says "I come from `%body`", but the actual back-edge comes
from `%cont`. Verifier rejects.

## Fix

List the **real** predecessors:

```llvm
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %cont ]
```

## The real BCIR instance

`runtime/llvm/bcir_batch_executor.ll` had a loop where the iteration
counter's phi listed `%body` as the back-edge predecessor:

```llvm
%i = phi i64 [ 0, %entry ], [ %next, %body ]
```

But the body block actually branched to either `%cont` directly or
through `%do_prefetch → %cont`, so the *real* predecessor of `%loop`
was `%cont`, not `%body`. Fixed in commit `5754354` ("Fix batch
executor loop PHI predecessor mismatch"):

```llvm
%i = phi i64 [ 0, %entry ], [ %next, %cont ]
```

## How to debug

When you see this error:

1. **Identify the block** containing the phi (the line number points
   at it).
2. **Find every basic block** that ends with a terminator targeting
   this block. Those are the *actual* predecessors.
3. **Compare to the phi's incoming list.** Add missing predecessors;
   remove extras.

For each predecessor `P` you add to the phi, you need to know what
value flows in from P. If the value isn't defined on the path from
P, you've got a real bug (use of an undefined value), not just a
mismatched-list bug.

## Checklist for editing CFG

Whenever you:

- Add a new `br ... label %X` to a block: check every phi in `%X`
  and add the new incoming pair.
- Remove a branch to `%X`: remove the corresponding incoming pair
  from every phi in `%X`.
- Replace one block-name with another (e.g., refactor `%body` →
  `%cont` as a successor): update the phi incoming labels too.

## See also

- [`../00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) — what phi is for
- [`../05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md) — branches that feed phis
- [`../05-control-flow/01-unconditional-br.md`](../05-control-flow/01-unconditional-br.md)
