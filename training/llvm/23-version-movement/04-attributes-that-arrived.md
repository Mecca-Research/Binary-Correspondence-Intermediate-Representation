# Three attributes you will meet before you meet anything else new

Sixteen attributes arrived between LLVM 18 and 23. Three of them are on the first line of
ordinary compiler output, so they are the ones a reader hits first — and one of them is a
respelling of something this corpus already taught, which is worse than a gap.

Compile the same C on both compilers and the shift is one line:

```c
struct Big { long a, b, c, d; };
int clamp(struct Big b) { return (int)(b.a & 0xff); }
```

```llvm
; clang 18 -O2
define dso_local i32 @clamp(ptr nocapture noundef readonly byval(%struct.Big) align 8 %0)

; clang 23 -O2
define dso_local range(i32 0, 256) i32 @clamp(ptr nofree noundef readonly byval(%struct.Big) align 8 captures(none) %0)
```

Same source, same optimisation level, same semantics. Three changes.

## `captures(...)` replaces `nocapture`

`nocapture` is gone from the attribute table LLVM 23 defines. It is **not** gone from what
`llvm-as` will read: feed it `ptr nocapture %p` and it parses, auto-upgrading the attribute
to `captures(none)` on the way in. Old IR keeps working; what changed is the spelling LLVM
*emits* and the one its tables know about.

That distinction matters and is easy to get backwards. Compare it with
[`ptrtoaddr`](02-ptrtoaddr.md), which older assemblers reject outright with `expected
instruction opcode` — a genuinely unknown token has no upgrade path, while a renamed
attribute does. "Removed" can mean either, and only one of them breaks your input.

The replacement says more than the old attribute could:

| Spelling | Means |
| --- | --- |
| `captures(none)` | Exactly what `nocapture` meant |
| `captures(address)` | The address may be compared or hashed, but not dereferenced later |
| `captures(address, provenance)` | The pointer may be reconstructed and used |
| `captures(ret: address)` | The capture happens only through the return value |

The old attribute was one bit: captured or not. The new one names *which part* of the
pointer escapes, because the address and the provenance are separately useful facts — the
same distinction that produced [`ptrtoaddr`](02-ptrtoaddr.md) in the same era. A pointer
whose address escapes but whose provenance does not can still be reasoned about for
aliasing.

**Why this one matters more than the other fifteen:** this corpus's own frontend
normalizer strips `captures(...)` so that checked-in snapshots keep assembling on the
baseline. A construct your tooling deletes is not a construct your documentation
explains, and for a while here it stood in for one.

## `range(...)` on a return value

`range(i32 0, 256)` on the return says the value is in `[0, 256)` — half-open, low
inclusive, high exclusive. `clang 18` had no way to state it; the fact existed and was
thrown away at the function boundary.

The corpus teaches `!range` metadata on `load` and `call` in several places, so the
concept is familiar — but metadata is *droppable by definition*, and an attribute is not.
That is the whole point of moving the fact: a return range now survives passes that are
free to discard metadata.

Read it as a promise you are making, not a description: if the function ever returns a
value outside the range, the result is poison and everything downstream is undefined.

## `dead_on_return` on C++ by-value parameters

```
define linkonce_odr void @take(ptr dead_on_return noundef byval(%struct.S) align 8 %s)
```

`dead_on_return(32)` shows up five times at `-O0` in a C++ file that passes a
non-trivially-destructible object by value. It says the callee may treat the pointed-to
memory as dead when it returns — the caller will not read it again — which is exactly the
C++ rule that a by-value parameter is destroyed by the callee.

You will see it in C++ output long before you see it anywhere else, and only rarely in C.

## The other thirteen

The remaining arrivals are dispositioned in
[`../reference/langref-delta-dispositions.json`](../reference/langref-delta-dispositions.json)
with a reason each. In brief: `flatten`, `noipa`, `nooutline`, `nodivergencesource` and
`hybrid_patchable` are optimiser and target directives a frontend emits only when asked;
`coro_elide_safe` is coroutine-lowering plumbing; `noext` completes the
`zeroext`/`signext` family for targets where neither extension is required; and the
`sanitize_*` additions (`sanitize_type`, `sanitize_realtime`,
`sanitize_realtime_blocking`, `sanitize_numerical_stability`, `sanitize_alloc_token`)
mark functions for sanitizers whose instrumentation the corpus teaches by shape rather
than by the attribute that requests it.

`denormal_fpenv` is the one other rename: it folds the departed `denormal-fp-math` and
`denormal-fp-math-f32` string attributes into one.

## Pitfalls checklist

- Do not write `nocapture` in new IR for LLVM 23. It still assembles — the auto-upgrade
  accepts it — but it is not the spelling the language defines any more, and reading it
  back shows you `captures(none)`. Equally, do not read its absence from newer output as
  "the optimiser lost the fact".
- Do not treat `captures(address)` as equivalent to `captures(none)`. They differ exactly
  where aliasing questions get interesting.
- Do not read `range(i32 0, 256)` as inclusive at both ends. The high bound is exclusive.
- Do not assume an attribute your normalizer strips is an attribute your reader has been
  taught.

## Checks

The `nocapture` → `captures` and `denormal-fp-math` → `denormal_fpenv` renames appear in
the rename table in [`01-reading-the-delta.md`](01-reading-the-delta.md), whose rows
[`../tools/verify-langref-delta.py`](../tools/verify-langref-delta.py) checks against both
surface snapshots. This chapter is cited as the teaching home for `captures`, `range` and
`dead_on_return` in the disposition table, and that gate verifies the citation actually
names each of them.
