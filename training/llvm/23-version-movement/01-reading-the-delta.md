# Reading a LangRef delta without being fooled by it

LLVM's language moves every release. Instructions arrive, attributes are renamed,
intrinsics graduate out of `llvm.experimental.*`. A corpus that teaches LLVM has to
track that movement, and the obvious way to track it — diff two releases and read the
result — produces a list that is **wrong in three specific ways** before you have
looked at a single entry.

This chapter is about those three ways, because each one produces a confident false
statement about the language, and all three are visible in the real LLVM 18 → 23 delta.

## Where the delta comes from

Not from a changelog. LLVM writes its own surface into three generated files that ship
with the development headers, and those are what LangRef documents:

| File | Declares |
| --- | --- |
| `llvm/IR/Instruction.def` | every instruction opcode |
| `llvm/IR/Attributes.td` | every function, parameter and return attribute |
| `llvm/IR/IntrinsicEnums.inc` | every intrinsic, with its dotted name in a trailing comment |

Reading those beats reading release notes, because they cannot be out of date with
respect to the compiler you actually have. `training/llvm/tools/verify-langref-delta.py`
does exactly that, snapshots each major into `reference/llvm-surface-<major>.json`, and
diffs the snapshots:

<!-- generated: langref-delta -->
| surface | arrived | departed |
| --- | ---: | ---: |
| instructions | 1 | 0 |
| attributes | 16 | 3 |
| intrinsics | 102 | 9 |

Of those 131, **7** are taught by a chapter and **124** are declared out of scope with a stated reason.
<!-- /generated -->

Every one of those items is disposed of in
[`../reference/langref-delta-dispositions.json`](../reference/langref-delta-dispositions.json):
either a chapter teaches it, or there is a written reason it is out of scope. The gate
refuses an item with no disposition, a disposition with no reason, and a citation that
does not mention its own subject.

## Trap 1 — an implementation split is not a language change

Between 18 and 23 the opcode `Br` disappears and `CondBr` and `UncondBr` appear. Read as
a language delta, that says: *LLVM removed the branch instruction and added two new
ones.*

It says nothing of the kind. LLVM split one C++ enumerator into two so the class
hierarchy could distinguish conditional from unconditional branches. **The IR text did
not change.** Both spellings were `br` before and both are `br` now:

```llvm
br label %exit                        ; UncondBr in 23, Br in 18 — same text
br i1 %c, label %then, label %else    ; CondBr   in 23, Br in 18 — same text
```

A reader who "learned" this delta would go looking for `condbr` in LangRef and not find
it. The gate suppresses the split explicitly, by name, in `INTERNAL_OPCODE_SPLITS` —
not by a heuristic, because a heuristic that guesses which renames are cosmetic will
eventually guess wrong about a real one.

## Trap 2 — a file-layout change is not a withdrawal

LLVM 18's `IntrinsicEnums.inc` contains four SVE intrinsics —
`llvm.aarch64.sve.pmov.to.pred.lane` and three siblings. LLVM 23's does not. Diff the
files and they read as *intrinsics withdrawn from the language.*

They were never in the language. Target intrinsics belong to a target; they live in
`IntrinsicsAArch64.h` and its siblings, and those four were emitted into the common file
by an accident of how that header was generated. Nothing about them changed for anyone
writing portable IR.

The gate drops every target-prefixed name from both sides before comparing. That is why
the intrinsic departures below number nine and not thirteen — and it is worth noticing
that the four false entries were *plausible*: they had real names, they really did
vanish from a real file, and a delta tool with no notion of "target-independent" would
have reported them with total confidence.

## Trap 3 — a rename reads as an arrival plus a departure

Three attributes depart between 18 and 23, and none of them is a capability the language
lost:

| Departed | What actually happened |
| --- | --- |
| `nocapture` | Respelled `captures(none)`, part of a finer-grained capture model |
| `denormal-fp-math` | Folded into `denormal_fpenv` |
| `denormal-fp-math-f32` | Folded into `denormal_fpenv` |

Each is one *half* of a rename whose other half is sitting in the arrivals list. Read the
two lists separately and you conclude LLVM dropped denormal control and added something
unrelated.

This one has teeth here specifically, because the corpus's own frontend normalizer
already knows about `captures`: it strips the newer spelling so snapshots keep
assembling on the baseline. **Stripping a spelling is not teaching it.** A gate that
removes a construct so a check keeps passing and a chapter that explains the construct
are different things, and the first can quietly stand in for the second for years.

## The pattern worth more than the list: graduation

Five of the nine departed intrinsics are the same event:

```
llvm.experimental.stepvector           ->  llvm.stepvector
llvm.experimental.vector.reverse       ->  llvm.vector.reverse
llvm.experimental.vector.interleave2   ->  llvm.vector.interleave2
llvm.experimental.vector.deinterleave2 ->  llvm.vector.deinterleave2
llvm.experimental.vector.splice        ->  llvm.vector.splice.left
                                       ->  llvm.vector.splice.right
```

**And graduation does not have to preserve the name.** Four of those five are a prefix
change and nothing else. The fifth is not: `llvm.experimental.vector.splice` did not
become `llvm.vector.splice` — there is no such intrinsic in LLVM 23 — it became *two*
intrinsics, `llvm.vector.splice.left` and `llvm.vector.splice.right`, because the
direction that used to be encoded in a signed immediate is now part of the name.

That row is worth dwelling on, because guessing it is the obvious mistake and the delta
itself does not stop you making it: `llvm.experimental.vector.splice` is in the departed
list, the `llvm.vector.splice.*` names are in the arrived list, and nothing pairs them up
for you. A rename is a hypothesis you form by reading two lists side by side, and this
one is wrong in the way that matters — the name you would guess does not exist.

The same release expanded the interleave family from `2` to `2` through `8` while
graduating it, so an arrival list also mixes *renames* with genuinely new siblings.

`llvm.experimental.` is not decoration. It is a stability marker, and it means *this
name may change or vanish without a deprecation period*. When the design settles, the
intrinsic graduates and the experimental spelling is removed outright.

The practical consequence: **IR or a pass that hard-codes an `llvm.experimental.` name
is pinned to a release**, and will break on upgrade in a way that pinning a stable
intrinsic never does. If you must use one, isolate the name behind a helper so the
upgrade is one edit rather than a search.

### What `llvm.stepvector` actually does

Naming an intrinsic in a rename table is not teaching it, and this chapter says as much a
few paragraphs up. So, for the one graduate a reader cannot route around:

```llvm
%idx = call <vscale x 4 x i64> @llvm.stepvector.nxv4i64()   ; <0, 1, 2, 3, ...>
```

It produces a vector whose lane *i* holds the value *i*. On a fixed-width vector you would
write that as a constant — `<i64 0, i64 1, i64 2, i64 3>` — and never need an intrinsic.

A **scalable** vector has no compile-time lane count: `<vscale x 4 x i64>` is four lanes
times a factor the hardware chooses at run time. You cannot write its elements out, so
there is no constant to write, and `llvm.stepvector` is the only way a lane-index vector
comes into existence at all. Anything built on lane position in SVE or RVV output —
strided addressing, an induction variable, a mask derived from lane number — starts here.

That is why it graduated: not a new capability, but one that scalable vectors made
unavoidable.

Two more departures are the same idea in a different shape — a replacement by a core
construct rather than a renamed intrinsic:

```llvm
; LLVM 18 and earlier
%h = call i16 @llvm.convert.to.fp16(float %f)
%f2 = call float @llvm.convert.from.fp16(i16 %h)

; LLVM 23: the intrinsics are gone; the instructions always could do this
%h2 = fptrunc float %f to half
%f3 = fpext half %h2 to float
```

## What actually arrived in the language

Exactly one instruction: **`ptrtoaddr`**. It is taught in
[`02-ptrtoaddr.md`](02-ptrtoaddr.md), because it is easy to mistake for `ptrtoint` and
the difference matters on any target where a pointer carries more than an address.

The attribute and intrinsic arrivals are dispositioned individually. Most are declared
out of scope with a reason — a teaching corpus is not a LangRef mirror, and saying so in
writing is better than implying coverage it does not have.

## Pitfalls checklist

- Do not read a diff of two `Instruction.def` files as a language delta. Check whether
  the IR spelling changed before believing an opcode arrived or left.
- Do not treat a name's disappearance from a generated file as its removal from the
  language; check which file it moved to first.
- Do not read arrivals and departures as independent lists. Renames appear in both, and
  the two halves only make sense together.
- Do not assume a construct your tooling strips is a construct your documentation
  explains.
- Do not hard-code an `llvm.experimental.` name anywhere you cannot cheaply change it.

## Checks

[`../tools/verify-langref-delta.py`](../tools/verify-langref-delta.py) regenerates the
table above from the two surface snapshots, requires every moved item to carry a
disposition, and verifies that each chapter cited as teaching an item actually names it.
Where the LLVM development headers for a major are installed it also re-reads that
major's surface from the toolchain and refuses a stale snapshot — the CI job that
installs `llvm-18-dev` owns the 18 snapshot, and the LLVM 23 job owns the 23 one.
