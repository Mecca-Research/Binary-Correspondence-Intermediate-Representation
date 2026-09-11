# Regions, blocks, and the two syntaxes

MLIR has one structural idea and repeats it all the way down: **an operation holds
regions, a region holds blocks, a block holds operations**. A function body is a region.
An `scf.if`'s then-branch is a region. A module is an operation with a region. There is no
special case for "the top level" — `builtin.module` is an op like any other.

Get that, and MLIR's syntax stops looking arbitrary.

## Block arguments replace phi nodes

MLIR has no `phi`. Where LLVM merges values at a join point with a phi node listing
`[value, predecessor]` pairs, MLIR gives the block **arguments** and every branch passes
them:

```mlir
  cf.br ^loop(%zero, %zero : i32, i32)

^loop(%i: i32, %acc: i32):
  %done = arith.cmpi sge, %i, %n : i32
  cf.cond_br %done, ^exit(%acc : i32), ^body(%i, %acc : i32, i32)
```

Same information, opposite direction. A phi looks *backwards* from the join and names its
predecessors; block arguments look *forwards* from each branch and name what it passes.
The forward form has a practical consequence: you cannot write a branch that forgets to
supply a value, because the argument list is part of the branch. The classic LLVM bug of
adding a predecessor and forgetting to update the phi has no MLIR equivalent.

The runnable version is
[`examples/regions-and-blocks.mlir`](examples/regions-and-blocks.mlir).

## Every op has two spellings

This is the single most confusing thing about reading MLIR, and it is rarely stated
plainly: **the same operation has a custom form and a generic form, and both are valid
input.**

```mlir
// custom form -- what a registered dialect chooses to print
%0 = arith.addi %arg0, %arg1 : i32

// generic form -- what every op can always be printed and parsed as
%0 = "arith.addi"(%arg0, %arg1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
```

The custom form is defined by the dialect and exists for readability. The generic form is
universal: operation name in quotes, operands in parentheses, attributes in braces, then
the type signature.

Print any module generically with `mlir-opt --mlir-print-op-generic`. The two parse to the
same module — that is checked, not assumed.

**Why this matters in practice:** a dialect MLIR has not registered *cannot* be printed in
custom form, because the custom form lives in the dialect's own C++. So a tool that meets
an unregistered dialect falls back to generic syntax.

Generic syntax is necessary for that, but on its own it is **not sufficient**. Hand
`mlir-opt` an operation from a dialect it has never heard of and it refuses even in
generic form:

```console
$ mlir-opt unregistered.mlir
error: operation being parsed with an unregistered dialect. If this is intended,
please use -allow-unregistered-dialect with the MLIR tool used
```

There are two ways past that, and they are not equivalent:

- `--allow-unregistered-dialect` tells the tool to carry the operation as opaque text. It
  can round-trip it and nothing more — no verification, no folding, no lowering. This
  corpus's own registry uses it for exactly that, on its Tier 1 unregistered-dialect
  sketches.
- **Registering the dialect** — from an IRDL file with `--irdl-file`, or by linking it in
  — gives the tool the operation's real definition, so it can verify it.

That second one is what makes the IRDL projection in
[`../22-bcir-approach/04-dialect-as-data.md`](../22-bcir-approach/04-dialect-as-data.md)
worth having: stock `mlir-opt` does not merely tolerate BCIR IR, it *checks* it. Generic
syntax is how the operation is spelled; the IRDL file is what makes it meaningful.

## Inherent and discardable attributes

Look again at the generic form:

```mlir
%0 = "arith.addi"(%arg0, %arg1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
```

The `<{ ... }>` is not decoration. Attributes inside it are **inherent**: they belong to the
operation's definition, so the registered op knows their names and meanings and the
verifier can reason about them. Attributes printed *outside* that group are
**discardable**: any pass may attach them, and any pass may drop them without asking.

Inherent does not mean *mandatory*. An ODS definition may declare an attribute
`OptionalAttr`, and plenty do; such an operation is perfectly valid without it. The
distinction is about **who owns the attribute**, not about whether every instance carries
one: an inherent attribute is part of the op's contract, so dropping one changes what the
op means and the verifier gets a say. A discardable attribute is a note anyone may leave
and anyone may erase.

That distinction is the answer to a question the lowering chapters raise repeatedly —
which attributes survive a rewrite? An inherent attribute is part of what the operation
*is*, so a rewrite that drops one has changed the operation's meaning and owes an
explanation. A discardable attribute is advisory, and a pass that drops one is not
misbehaving.

It is also the direct MLIR analogue of the LLVM lesson in
[`../23-version-movement/04-attributes-that-arrived.md`](../23-version-movement/04-attributes-that-arrived.md):
LLVM moved return-range facts from droppable `!range` metadata to a non-droppable
`range(...)` attribute for exactly this reason.

## Pitfalls checklist

- Do not look for `phi` in MLIR. Look at the block's argument list and at what each
  branch passes.
- Do not assume a `.mlir` file you cannot read uses a syntax you do not know. It is
  probably the generic form, which you now can read.
- Do not treat `<{...}>` as a print quirk. Inside is inherent, outside is discardable, and
  the difference decides what survives a pass.
- Do not expect an unregistered dialect to round-trip in custom form. It cannot; that is
  what generic form is for.

## Checks

[`../tools/verify-mlir-infrastructure.py`](../tools/verify-mlir-infrastructure.py) prints
the example generically, asserts the shape this chapter describes (quoted op names, a
`<{...}>` group), parses the generic form back, and requires it to produce byte-identical
output to parsing the custom form — so "two spellings of one module" is a checked claim.
