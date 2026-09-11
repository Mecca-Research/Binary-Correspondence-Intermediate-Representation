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

Same source, same optimisation level, same semantics. Three changes — and only two of
them are new *attributes*:

| Change | What kind of change |
| --- | --- |
| `nocapture` becomes `captures(none)` | a rename; the old spelling still assembles |
| `range(i32 0, 256)` appears on the return | an attribute that did not exist in LLVM 18 |
| `nofree` appears on the parameter | an attribute LLVM 18 already had, newly *inferred* |

The third is the one to be careful with. `nofree` is in LLVM 18's attribute table; nothing
arrived. What changed is that clang 23's inference is strong enough to prove it here — a
difference in the optimiser, not in the language. A diff of two compilers' output mixes the
two freely, which is why this chapter works from the attribute tables rather than from
eyeballing output: the tables say what the language gained, and only that.

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

The corpus already had an example that produces this one, in
[`../20-clang-frontend/examples/cxx-object-model.cpp`](../20-clang-frontend/examples/cxx-object-model.cpp):

```cpp
struct Guard { Guard(); ~Guard(); int n; };
int guarded() { return Guard().n; }
```

`clang++ 23 -O0` emits the temporary's destructor call as:

```llvm
call void @_ZN5GuardD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %1)
declare void @_ZN5GuardD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4))
```

`clang++ 18` on the same file emits the same call without it.

Read the placement carefully, because the obvious guess is wrong. `dead_on_return` is not
on a by-value *parameter* of some function that takes an object; it is on the `this`
pointer of the **destructor call**. It says: once `~Guard()` returns, the 4 bytes it was
given are dead — nobody will read that storage again. The number in the parentheses is the
size of the region, not a count of anything.

That is a fact the frontend knows and LLVM could not previously be told: a temporary's
storage is finished the moment its destructor completes, so the stack slot is free to be
reused and any store into it afterwards is dead. C++ is where this shows up first, because
C++ is where the frontend routinely emits calls that end an object's lifetime.

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
names each of them — in a code block or an inline code span, not merely somewhere in the
prose, since `range` is an ordinary English word and a loose search would have been
satisfied by a chapter that never showed the spelling.

The `dead_on_return` line above is not transcribed from memory either:
[`../tools/verify-frontend-lowering.py`](../tools/verify-frontend-lowering.py) compiles
`cxx-object-model.cpp` and requires the attribute to appear on `@_ZN5GuardD1Ev`'s pointer
argument with that exact size. The claim is marked as needing clang 23, so on the LLVM 18
baseline it is reported as skipped rather than failed — and the gate counts skipped claims
separately, because a run whose claims all skipped has verified nothing.
