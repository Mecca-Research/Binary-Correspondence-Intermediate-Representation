# 14.3 — Lowering to the LLVM Dialect

MLIR lowering is usually a sequence of partial conversions. High-level dialects
are rewritten into progressively lower dialects until the remaining IR can be
translated to LLVM IR or handed to LLVM's optimizer and backend.

## LLVM dialect vs textual LLVM IR

The **LLVM dialect** is an MLIR dialect whose operations model LLVM IR concepts:
functions, globals, integer and floating-point operations, memory operations,
branches, calls, and LLVM pointer types.

Textual **LLVM IR** is the `.ll` assembly language parsed by `llvm-as` and
printed by `llvm-dis`.

They are related but not identical:

| LLVM dialect in MLIR | Textual LLVM IR |
|---|---|
| MLIR operations such as `llvm.func`, `llvm.add`, `llvm.load`, `llvm.br`. | LLVM instructions and declarations such as `define`, `add`, `load`, `br`. |
| Uses MLIR regions, blocks, attributes, symbol tables, and dialect types. | Uses LLVM module/function/basic-block grammar. |
| Still participates in MLIR passes and conversion legality checks. | Consumed by LLVM core tools such as `llvm-as`, `opt`, and `llc`. |
| Can coexist temporarily with other MLIR dialects during partial lowering. | A whole module must obey LLVM IR rules. |
| Translation to LLVM IR is a final serialization/bridging step, not a normal textual rename. | Already in LLVM's native IR syntax. |

Do not assume that copying MLIR LLVM-dialect text into a `.ll` file will parse.
The syntax and verifier are different.

## Typical lowering stack

A common path looks like this:

```text
custom/domain dialect
  ↓ canonicalize + legalize domain invariants
structured dialects: scf, affine, linalg, tensor, vector, memref, arith, func
  ↓ bufferize / convert structured control flow / expand complex ops
lower-level dialects: cf, memref, arith, func, vector
  ↓ convert-to-llvm patterns and ABI conversion
llvm dialect
  ↓ translate MLIR LLVM dialect to LLVM IR
LLVM IR (.ll/.bc)
  ↓ opt / llc / ORC JIT
machine code or JIT execution
```

A small project may skip many steps. A graph compiler may keep custom graph ops
longer, lower graph traversal to `scf`/`cf`, lower memory views to `memref`, and
only then convert to the LLVM dialect.

## Conversion and legality

MLIR conversion is usually expressed in terms of:

- **Conversion target**: declares which dialects or operations are legal after a
  pass.
- **Type converter**: maps source types to destination types, such as
  `memref<?xi32>` to a descriptor type or `!bcir.vertex` to `!llvm.ptr` plus ID
  fields.
- **Rewrite patterns**: replace illegal operations with legal operations.
- **Materializations**: bridge temporary type mismatches during partial
  conversion.

A good conversion pass makes its contract explicit: after the pass, no
`bcir.walk` remains; or no `memref` types remain; or only `llvm` and `builtin`
operations remain.

## Lowering structured control flow

Structured dialects often carry more invariants than LLVM IR:

- `scf.for` has a single induction variable and loop-carried values.
- `scf.if` has exactly delimited then/else regions.
- `affine.for` carries affine bounds and maps.
- Graph or tensor dialects may represent whole computations as one operation.

Lowering flattens this structure into blocks and branches. Block arguments turn
into PHI nodes after translation to LLVM IR. Any loop or branch metadata must be
reattached deliberately; it is not preserved just because the source operation
had an attribute.

## Lowering memory and types

The risky part of lowering is often not arithmetic; it is ABI and memory:

- `memref` values may lower to descriptors containing allocated pointer,
  aligned pointer, offset, sizes, and strides.
- `index` lowers to a target-sized integer, so the target data layout matters.
- Custom opaque domain types must lower to concrete LLVM-compatible types.
- Alignment, address spaces, aliasing, volatility, and atomic semantics need
  explicit representation.

If a high-level dialect has a vertex handle, edge handle, or graph view, decide
whether the LLVM boundary sees a pointer, integer ID, pair/struct, descriptor,
or runtime-library object.

## Common lowering pitfalls

1. **Forgetting data layout**: `index` and pointer-sized arithmetic depend on
   target layout.
2. **Dropping attributes**: source hints vanish unless converted to LLVM
   attributes, metadata, constants, runtime calls, or intentionally discarded.
3. **Confusing LLVM dialect syntax with `.ll` syntax**: `llvm.func` is MLIR;
   `define` is LLVM IR.
4. **Lowering structured loops too early**: flattening before canonicalization
   can make dependence, shape, or graph analyses harder.
5. **Incorrect memref descriptor assumptions**: descriptor layout is an ABI
   contract; hand-written LLVM IR must match it exactly.
6. **Losing block-argument/PHI correspondence**: loop-carried values and branch
   joins must become valid LLVM PHI nodes with complete predecessor lists.
7. **Using attributes for runtime facts**: dynamic graph weights, addresses, or
   register choices must become values when they are not compile-time constants.
8. **Missing side-effect modeling**: custom operations that read/write memory or
   runtime state must say so, or passes may reorder them incorrectly.
9. **Erasing alias/address-space facts**: high-level memory spaces and graph
   ownership need explicit LLVM address spaces, metadata, ABI fields, or runtime
   checks.
10. **Assuming hints are semantics**: HAM hints, prefetch levels, and scheduling
    preferences should not change correctness if ignored.

## See also

- [`examples/lowered-llvm-dialect.mlir`](examples/lowered-llvm-dialect.mlir) — illustrative LLVM-dialect text.
- [`../08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md) — PHI mistakes after CFG lowering.
- [`../08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md) — schema drift when multiple low-level files duplicate type layouts.
