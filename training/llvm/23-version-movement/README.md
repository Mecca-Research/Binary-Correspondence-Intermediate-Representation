# 23 — Version movement: reading LLVM's language as it changes

LLVM is not a fixed language. Every release moves the surface a little: an instruction
arrives, an attribute is respelled, an intrinsic graduates out of
`llvm.experimental.*` and its old name stops existing. Material that is correct for one
release is wrong for the next, quietly, in ways no example that still assembles will
reveal.

This chapter is about that movement — how to measure it, how to be fooled by it, and what
actually changed between the release this corpus enforces as its baseline and the release
it tracks.

| Chapter | Covers |
| --- | --- |
| [`01-reading-the-delta.md`](01-reading-the-delta.md) | Where a delta comes from, and the three ways a naive diff of two LLVM releases states something false |
| [`02-ptrtoaddr.md`](02-ptrtoaddr.md) | The one instruction LLVM 23 added, and why it is not a synonym for `ptrtoint` |
| [`03-conversions-and-shifts.md`](03-conversions-and-shifts.md) | The cast and shift opcodes the corpus listed but never demonstrated — including the out-of-range conversion that is `poison` |
| [`04-attributes-that-arrived.md`](04-attributes-that-arrived.md) | `captures(...)`, `range(...)` and `dead_on_return` — the three new attributes on the first line of ordinary output |

## Why this is a chapter and not a changelog

A changelog tells you what its author chose to mention. The three files LLVM generates
for its own build — `Instruction.def`, `Attributes.td`, `IntrinsicEnums.inc` — tell you
what the compiler you have actually implements, and they cannot disagree with it.

So the delta here is *measured*, from a snapshot of each release's surface taken from
those files, checked in as
[`../reference/llvm-surface-18.json`](../reference/llvm-surface-18.json) and
[`../reference/llvm-surface-23.json`](../reference/llvm-surface-23.json). Every item that
moved carries a disposition — a chapter that teaches it, or a written reason it is out of
scope — and the gate refuses an item that has neither.

That is the same discipline the rest of the corpus lives under, applied to time instead of
to content: a gap you have named is a gap; a gap nobody wrote down is a mistake waiting to
be discovered by a reader.

## The three traps, in one line each

- **An implementation split is not a language change.** `Br` became `CondBr` and
  `UncondBr` in the C++ enum; the IR still says `br` and always did.
- **A file-layout change is not a withdrawal.** Four target intrinsics stopped being
  emitted into the common header. They were never part of the target-independent
  language.
- **A rename reads as an arrival plus a departure.** `nocapture` did not vanish; it is
  spelled `captures(none)` now, and the two halves only make sense read together.

## What actually changed

One instruction arrived: `ptrtoaddr`. Sixteen attributes arrived and three departed, all
three of the departures being halves of renames. A hundred and two intrinsics arrived and
nine departed, five of those nine being graduations out of `llvm.experimental.*`.

The exact counts are regenerated into
[`01-reading-the-delta.md`](01-reading-the-delta.md) from the snapshots, so they cannot
drift from the tree.

## Verification boundary

**Checked:** the two surface snapshots against the installed toolchain wherever those
headers exist; every moved item has a disposition; every chapter cited as teaching an
item actually names it; both examples assemble and verify.

**Version-gated:** [`examples/ptrtoaddr-vs-ptrtoint.ll`](examples/ptrtoaddr-vs-ptrtoint.ll)
declares `; REQUIRES: llvm >= 23` and is skipped on older assemblers with the reason
printed. CI's `LLVM training corpus (LLVM 23)` job installs 23 and is what verifies it, so
the skip has an owner rather than being a hole.

**Not checked:** whether a disposition's stated reason is a *good* reason. The gate can
tell that a reason exists and that a cited chapter names its subject; judging whether the
corpus drew the scope line in the right place is a reader's job, and the dispositions are
checked in as data so that judgement can be made without reading the tool.
