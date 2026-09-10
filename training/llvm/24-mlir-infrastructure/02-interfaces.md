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

Any dialect implementing those two gets the same inliner with no change to the inliner.
That is the whole mechanism, and it is why adding a dialect to MLIR does not mean
reimplementing the pass pipeline.

Runnable: [`examples/interfaces-inlining.mlir`](examples/interfaces-inlining.mlir).

## The three kinds

| Kind | Attached to | Answers |
| --- | --- | --- |
| **Op interface** | An operation | "Can I inline this? Does it have side effects? What does it branch to?" |
| **Type interface** | A type | "Is this shaped? What is its element type?" |
| **Attribute interface** | An attribute | "Is this an integer-like attribute, whatever dialect defined it?" |

Op interfaces are the ones you meet first and author most. The other two matter at
conversion boundaries: a `TypeConverter` that must handle types from dialects it does not
know reaches them through type interfaces, which is what makes
[`../18-mlir-lowering-to-llvm/02-typeconverter-and-materialization.md`](../18-mlir-lowering-to-llvm/02-typeconverter-and-materialization.md)
work for a custom type at all.

## Interfaces you will meet reading real IR

- `MemoryEffectOpInterface` — what an op reads and writes. Canonicalization and
  dead-code elimination consult it before removing anything; an op with unmodelled effects
  is one nobody may delete.
- `CallOpInterface` / `CallableOpInterface` — the call graph, as above.
- `BranchOpInterface` / `RegionBranchOpInterface` — where control goes, including into and
  out of regions. This is how a pass reasons about `scf.if` without knowing `scf`.
- `InferTypeOpInterface` — an op that computes its own result types, so a builder does not
  have to be told them.
- `Symbol` / `SymbolTable` — named things and the scopes holding them, which is how
  `@callee` resolves at all.

## Traits are not interfaces

A **trait** is a compile-time marker with no methods: `Pure`, `Commutative`,
`SameOperandsAndResultType`, `IsolatedFromAbove`. It tells the framework a fact and often
buys generic verification for free.

An **interface** has methods and is dynamically dispatched: the pass calls it and gets an
answer computed by the dialect.

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
