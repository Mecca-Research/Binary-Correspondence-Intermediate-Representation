# 24 — MLIR infrastructure: the parts that are not a dialect

Chapters [`14-mlir-bridge`](../14-mlir-bridge) and
[`18-mlir-lowering-to-llvm`](../18-mlir-lowering-to-llvm) teach MLIR by lowering something
through it. That is the right way in, and it leaves a gap: the *framework* those chapters
stand on is never explained. A reader who follows them can write a conversion pattern
without being able to say what a region is, why their pipeline string is rejected, or what
the quoted-op syntax in half the corpus's own `.mlir` files is.

This chapter is that framework.

| Chapter | Covers |
| --- | --- |
| [`01-regions-blocks-and-the-two-syntaxes.md`](01-regions-blocks-and-the-two-syntaxes.md) | Op → region → block nesting, block arguments instead of phi nodes, the custom and generic syntaxes, inherent versus discardable attributes |
| [`02-interfaces.md`](02-interfaces.md) | How a generic pass reaches a dialect it has never heard of; op, type and attribute interfaces; traits, and why `IsolatedFromAbove` is load-bearing |
| [`03-the-pass-manager.md`](03-the-pass-manager.md) | Pipelines as trees, anchoring, the misspelled anchor that silently runs nothing, analyses and the inverted invalidation default, parallel execution |
| [`04-bytecode-and-partial-lowering.md`](04-bytecode-and-partial-lowering.md) | `unrealized_conversion_cast` as a diagnosis, `--reconcile-unrealized-casts`, and MLIR bytecode with its versioning story |

## What this chapter refuses to be

**Not a dialect tour.** MLIR 23 registers 47 dialects. Teaching them one by one produces a
worse reference than the upstream documentation and dates faster. The framework does not
date at the same rate: regions, interfaces and pass anchoring have been stable across
every release this corpus has tracked, while the dialect list moves every time.

**Not a substitute for upstream docs.** Where a reader needs the exhaustive list of an
interface's methods, that is upstream's job. What is here is the part that is hard to get
from a reference: which mechanism answers which question, and what the failure looks like
when you pick the wrong one.

## The through-line

Every chapter here is an instance of one idea: **MLIR is extensible, so nothing in the
framework may name a dialect.** Passes reach ops through interfaces. Pipelines name
anchors by operation name, resolved at run time. Unknown dialects print generically.
Bytecode versions per dialect. Once you see that constraint, the design choices stop being
arbitrary and start being forced.

## Verification boundary

**Checked, by running a real `mlir-opt`:** that the custom and generic syntaxes round-trip
to the same module and that the generic form has the shape chapter 01 describes; that
`--inline` reaches `func.func` purely through its interfaces; that a function-scoped pass
named at module level fails and the nested form succeeds; that a non-existent anchor is
still accepted silently; that a cancelling cast pair reconciles while a lone cast survives;
and that bytecode carries its magic and round-trips exactly.

**Pinned rather than endorsed:** the silent acceptance of a non-existent pass anchor. It is
current MLIR behaviour and a real trap, so the gate asserts it — and says, in its own
failure message, that a future MLIR rejecting it is an improvement that should be met by
rewriting the chapter, not by loosening the check.

**Not checked:** the analysis-invalidation and threading rules in chapter 03. Those are
properties of C++ pass code, not of anything `mlir-opt` can be asked from the command
line, and this corpus does not build a pass plugin to assert them. They are stated with
their mechanism so a reader can verify them against upstream, and named here so the gap is
visible rather than implied.

**Owner of the skip:** every behavioural check needs `mlir-opt`. Where it is absent the
gate says so and names its owner; CI's `LLVM training corpus (LLVM 23)` job installs the
toolchain and passes `--require-tools`, so absence there is a failure rather than a quiet
pass.
