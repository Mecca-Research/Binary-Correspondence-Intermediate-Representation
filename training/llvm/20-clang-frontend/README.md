# Clang Frontend Internals: Driver, AST, Sema, and CodeGen

## Key takeaways

- By the time LLVM IR exists, most of the interesting decisions have already
  been made. Type layout, argument classification, name mangling, overload
  resolution, implicit conversions, destructor placement, and bit-field access
  are **frontend** results, not optimizer results.
- The driver (`clang`) and the frontend (`clang -cc1`) are different programs
  with different flags. Nearly every "clang ignored my flag" question is
  actually a driver/frontend boundary question; `clang -###` answers it.
- The AST is the frontend's semantic truth. `-ast-dump` shows exactly what
  Sema built — including the implicit nodes (conversions, materialized
  temporaries, destructor calls) that explain IR you did not write.
- Argument classification is target law. The same C signature produces
  different IR signatures on x86-64 SysV and AArch64 AAPCS, and both are
  correct. A frontend that guesses here produces code that links and crashes.
- A source-language concept that has no IR counterpart — a bit-field, a union,
  a reference, a template — is *erased* by lowering. Debugging it means reading
  what it was erased into.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Driver vs `-cc1`, compilation phases, and where each flag applies | [`01-driver-and-frontend-pipeline.md`](01-driver-and-frontend-pipeline.md) |
| AST node families, `-ast-dump`, and what Sema adds implicitly | [`02-ast-and-sema.md`](02-ast-and-sema.md) |
| Exact C lowering: locals, aggregates, bit-fields, unions, control flow, `volatile` | [`03-c-lowering-rules.md`](03-c-lowering-rules.md) |
| Exact C++ lowering: mangling, vtables, constructors, temporaries, templates, EH | [`04-cxx-lowering-rules.md`](04-cxx-lowering-rules.md) |
| ABI and target lowering: argument classification, `sret`/`byval`, extension rules | [`05-abi-and-target-lowering.md`](05-abi-and-target-lowering.md) |
| How this maps onto BCIR's own C front-end twin | [`06-bcir-cfront-correspondence.md`](06-bcir-cfront-correspondence.md) |
| [`07-emitc-the-other-direction.md`](07-emitc-the-other-direction.md) | MLIR emitting C: why `arith.addi` becomes a `uint32_t` round trip, and what `--mlir-to-cpp` does not give you |

## Expected prerequisites

- [`01-syntax/`](../01-syntax) and [`02-types/`](../02-types) — you need to be
  able to read the IR the frontend emits.
- [`04-memory/`](../04-memory) — `alloca`, `load`/`store`, and `getelementptr`
  are what almost every C construct becomes.
- [`13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) — the
  ABI attributes this chapter watches the frontend attach.

## Examples

All examples in [`examples/`](examples) are **checked**: the sources compile,
and [`../tools/verify-frontend-lowering.py`](../tools/verify-frontend-lowering.py)
asserts the structural claims this chapter makes about their output.

| Artifact | Kind |
| --- | --- |
| `struct-layout.c` | C source: padding, packing, bit-fields, unions, array parameters |
| `abi-boundary.c` | C source: struct passing/returning, narrow integers, varargs |
| `control-flow-lowering.c` | C source: ternary, short-circuit, loops, `switch`, `volatile` |
| `cxx-object-model.cpp` | C++ source: virtuals, overloads, templates, temporaries |
| `*.ll` | normalized `clang -S -emit-llvm` snapshots of the above |

The `.ll` files are **snapshots**, not hand-written examples. They are
regenerated with:

```bash
python3 training/llvm/tools/verify-frontend-lowering.py --update
```

and checked (without regenerating) by:

```bash
python3 training/llvm/tools/verify-frontend-lowering.py
```

Attribute groups, module flags, and the `ident` string are stripped from the
snapshots: they are build configuration rather than lowering, and their
spellings change faster than this corpus's LLVM >= 15 baseline permits.

## Verification boundary

- The claims this chapter makes about **lowering rules** are asserted by the
  gate above, structurally, so a Clang release that changes a rule fails the
  gate instead of quietly making the prose wrong.
- The claims it makes about **Clang's internal class structure** (`Sema`,
  `CodeGenFunction`, `ASTContext`) are review material: they are stable enough
  to navigate by and are not machine-checked here.
- Everything target-specific is stated with its triple. A statement without a
  triple is a statement about the C standard, not about an ABI.

## What this chapter is not

It is not a Clang contributor guide. It does not cover writing a Clang plugin,
adding an attribute, extending the preprocessor, or modifying Sema. It covers
what a person or an agent needs in order to **read** frontend output and
attribute an IR-level surprise to the right stage.
