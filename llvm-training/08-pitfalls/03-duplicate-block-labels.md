# Pitfall 03 — Duplicate Block Labels or SSA Names

## The error

```
llvm-as: foo.ll:42:1: error: multiple definition of local value named 'mmio_check'
```

or

```
llvm-as: foo.ll:55:1: error: multiple definition of local value named '%v'
```

## What's happening

Inside a single function:

- **Every basic block label must be unique.**
- **Every SSA name must be unique.**
- **Labels and SSA names share a namespace.** (You can't have both
  a label `%foo` and an SSA value `%foo`.)

If a function has two blocks named `mmio_check:`, the second one
shadows the first — and `llvm-as` rejects this.

## How it typically happens

**Copy-paste duplication.** Someone needs the same MMIO-check
sequence twice in a function, copy-pastes the block, and forgets to
rename the labels and SSA names.

```llvm
define i1 @verify(ptr %p) {
entry:
  br i1 %some_cond, label %mmio_check, label %mmio_skip   ; line 5

mmio_check:                                ; line 10  ← first definition
  %v = load i32, ptr %p
  ; ...
  br label %mmio_merge

mmio_skip:
  br label %mmio_merge

mmio_merge:
  %r = phi i1 [ ... ]

  ; ↓↓↓ duplicated block, names not renamed ↓↓↓
  br i1 %other_cond, label %mmio_check, label %mmio_skip   ; line 25

mmio_check:                                ; line 28  ← DUPLICATE
  %v = load i32, ptr %p                    ; ←  also duplicate SSA name
  ; ...
  br label %mmio_merge

mmio_skip:                                  ; DUPLICATE
  br label %mmio_merge

mmio_merge:                                 ; DUPLICATE
  ; ...
}
```

`llvm-as` rejects: "multiple definition of local value named ...".

## Fix

Rename the second copy distinctly:

```llvm
mmio_check_2:
  %v2 = load i32, ptr %p
  ...

mmio_skip_2:
  ...

mmio_merge_2:
  ...
```

Or — better — **refactor**. If you really do the same check twice,
extract it into a helper function:

```llvm
define i1 @do_mmio_check(ptr %p) {
  ; check logic once
  ret i1 %ok
}

define i1 @verify(ptr %p) {
  ...
  %ok1 = call i1 @do_mmio_check(ptr %p)
  ...
  %ok2 = call i1 @do_mmio_check(ptr %p)
  ...
}
```

## The real BCIR instance

`runtime/llvm/bcir_claim_verify.ll` (pre-`1f62e86`) had **two copies
of an MMIO-check block trio** (`mmio_check`, `mmio_skip`,
`mmio_merge`) in the same function. The copy-paste also duplicated
SSA names (`%h`, `%rid_sel`, `%rid_idx`, `%rid_idx64`,
`%rid_in_range`, `%res`, `%domain_p`, `%domain`, `%is_mmio`,
`%vol_or_bar`, `%has_vol_or_bar`, `%mmio_ok`, `%mmio_gate`).

Fixed by deleting the duplicate block trio (the second copy was
dead code anyway).

## See also

- [`../00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) — SSA = single definition
- [`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) — labels and identifiers
- [`02-phi-predecessor-mismatch.md`](02-phi-predecessor-mismatch.md) — phi nodes referencing
  predecessor labels
