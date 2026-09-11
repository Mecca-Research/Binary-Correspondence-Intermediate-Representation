# Interfaces: how a pass works on a dialect it has never heard of

MLIR's central engineering problem is that anyone can add a dialect, and the passes were
written before that dialect existed. Inlining, canonicalization, bufferization and the
LLVM lowering all have to work on operations their authors never saw.

**Interfaces are the answer, and they are the only answer.** A pass that special-cases
dialects does not scale past the dialects it names; a pass written against an interface
works on every op that implements it, forever.

## The demonstration

```mlir
func.func private @callee(%a: i32) -> i32 {
  %r = arith.addi %a, %a : i32
  return %r : i32
}

func.func @caller(%x: i32) -> i32 {
  %y = call @callee(%x) : (i32) -> i32
  return %y : i32
}
```

`mlir-opt --inline` produces:

```mlir
func.func @caller(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  return %0 : i32
}
```

**The inliner contains no reference to the `func` dialect.** It works because:

- `func.call` implements `CallOpInterface` — "I am a call, and here is my callee"
- `func.func` implements `CallableOpInterface` — "I am callable, and here is my body"

Those two describe the call graph, and they are what lets the inliner *find* something to
inline without knowing any dialect. They are necessary — and on their own they are not
enough. MLIR also asks the **dialect** whether inlining is legal here, through a
`DialectInlinerInterface`: may this operation be inlined, may this region, and what
materialization is needed when types or control flow have to be adapted at the boundary.

The `func` example above works because upstream registers that interface too. A dialect
that implements only the two op interfaces will watch `--inline` leave every call exactly
where it was — which is a confusing result if you believed the two-interface recipe was
the whole story. It is the *shape* of the mechanism that generalises: the pass asks
questions through interfaces rather than switching on dialect names, and a dialect answers
as many of them as it wants to participate in.

Runnable: [`examples/interfaces-inlining.mlir`](examples/interfaces-inlining.mlir).

## The three kinds

| Kind | Attached to | Answers |
| --- | --- | --- |
| **Op interface** | An operation | "Can I inline this? Does it have side effects? What does it branch to?" |
| **Type interface** | A type | "Is this shaped? What is its element type?" |
| **Attribute interface** | An attribute | "Is this an integer-like attribute, whatever dialect defined it?" |

Op interfaces are the ones you meet first and author most.

Type interfaces are worth one correction, because the obvious guess is wrong: a
`TypeConverter` does **not** discover conversions by asking types about their interfaces.
You register callbacks with `addConversion`, one per type you intend to handle, and the
converter runs them in reverse order of registration —
[`../18-mlir-lowering-to-llvm/02-typeconverter-and-materialization.md`](../18-mlir-lowering-to-llvm/02-typeconverter-and-materialization.md)
does exactly that for each custom type. Type interfaces are for asking a type a *question*
it can answer generically (`ShapedType`: what is your element type, what is your rank), not
for dispatching a conversion to it.

## Interfaces and traits you will meet reading real IR

- `MemoryEffectOpInterface` — what an op reads and writes. Canonicalization and
  dead-code elimination consult it before removing anything; an op with unmodelled effects
  is one nobody may delete.
- `CallOpInterface` / `CallableOpInterface` — the call graph, as above.
- `BranchOpInterface` / `RegionBranchOpInterface` — where control goes, including into and
  out of regions. This is how a pass reasons about `scf.if` without knowing `scf`.
- `InferTypeOpInterface` — an op that computes its own result types, so a builder does not
  have to be told them.
- `Symbol` / `SymbolTable` — named things and the scopes holding them, which is how
  `@callee` resolves at all. These two are *traits*, not interfaces; see below.

## Traits are not interfaces

A **trait** is a statically dispatched mixin: `Pure`, `Commutative`,
`SameOperandsAndResultType`, `IsolatedFromAbove`, `Symbol`, `SymbolTable`. Attaching one
states a fact about the operation, and often buys generic verification for free — but a
trait is not merely a marker. It can carry methods and verification hooks, resolved at
compile time because the concrete op type is known.

An **interface** is dynamically dispatched: a pass holding an unknown `Operation *` asks
whether it implements the interface and, if so, calls through it. The dialect's
implementation answers.

The practical difference is *who can ask*. A trait is available to code that already knows
the concrete op type; an interface is available to code that does not — which is why the
generic passes are written against interfaces. `Symbol` and `SymbolTable` are traits, and
appear in the list above for what they let you look up, not as counter-examples.

`IsolatedFromAbove` is the trait worth knowing by name, because it is load-bearing for the
pass manager: it means the op's regions do not reference values defined outside it, which
is what makes it safe to run passes on two such ops **in parallel**. That is the subject of
[`03-the-pass-manager.md`](03-the-pass-manager.md).

## Pitfalls checklist

- Do not write a pass that switches on dialect names. Find the interface, or add one.
- Do not assume an op with no `MemoryEffectOpInterface` is side-effect free. Unmodelled is
  not the same as none, and the conservative reading is the correct one.
- Do not confuse a trait with an interface: a trait is a fact, an interface is a question
  you can ask.
- Do not attach `IsolatedFromAbove` to an op whose regions do capture outer values. The
  pass manager will run it concurrently on that promise.

## Checks

[`../tools/verify-mlir-infrastructure.py`](../tools/verify-mlir-infrastructure.py) runs
`--inline` over the example and requires both halves of the claim: that the call is gone,
and that the callee's body arrived in the caller. If the inliner ever stopped reaching
`func.func` through its interfaces, this chapter's demonstration would stop working and
the gate would say so.
