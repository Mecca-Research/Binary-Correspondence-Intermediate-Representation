# A minimal complete MLIR dialect (standalone skeleton)

The smallest dialect that still exercises every mechanism a production dialect
needs, written so it can be read in one sitting.

Chapters: [`../../08-production-dialect-and-build-integration.md`](../../08-production-dialect-and-build-integration.md)
and [`../../09-production-conversion-pass.md`](../../09-production-conversion-pass.md).

## Verification boundary — read this first

**Nothing in this directory is built by any training-corpus gate.** Building it
requires an MLIR development install (`mlir-tblgen`, the MLIR CMake package,
and MLIR headers), which the corpus deliberately does not assume; the corpus
gates assume only `llvm-as` and `opt`.

The **production** MLIR dialect in this repository — the one that CI builds,
runs `lit` tests against, and validates on two LLVM majors — is
[`../../../../mlir/`](../../../../mlir). The chapters cite it directly, and
[`../../../tools/verify-mlir-rail-references.py`](../../../tools/verify-mlir-rail-references.py)
checks those citations on every run, so the prose cannot drift away from the
code it describes.

Treat this skeleton as **reviewed reference code**: complete, idiomatic, and
mirroring the production rail's structure, but verified by reading rather than
by building.

`scale.mlir` is the exception — it is a registered Tier-1 syntax sketch in
[`../../../autograder/mlir-examples.json`](../../../autograder/mlir-examples.json)
and is parsed by the MLIR rail with `--allow-unregistered-dialect`.

## Files

| File | Role |
| --- | --- |
| `TrainingOps.td` | ODS: dialect, a parameterized type, an enum attribute, a shared attribute bundle, three ops |
| `TrainingDialect.h` | hand-written header pulling in the generated `.inc` files |
| `TrainingDialect.cpp` | `initialize()`, generated definitions, and the one hand-written verifier |
| `TrainingToLLVM.cpp` | a partial conversion to the LLVM dialect |
| `training-opt.cpp` | the tool: registry, pass registration, `MlirOptMain` |
| `CMakeLists.txt` | TableGen wiring, dialect library, tool |
| `scale.mlir` | input for the pass, and a syntax sketch |

## The dialect

```mlir
%scaled = training.scale %x { factor = 3 : i32 } : i32
%lane   = training.lane_cast %scaled : i32 to !training.lane<width = 4>
          training.emit %lane : !training.lane<width = 4>
```

| Op | Traits | Notes |
| --- | --- | --- |
| `training.scale` | `Pure`, `SameOperandsAndResultType` | plus one hand-written rule: the factor must be positive |
| `training.lane_cast` | `Pure` | produces the dialect's custom type |
| `training.emit` | none | side-effecting on purpose; `Pure` here would let the optimizer delete it |

## What each file is there to demonstrate

- **`TrainingOps.td`** — a shared `defvar` argument bundle so a contract is
  declared once rather than repeated per op; `OptionalAttr`/`DefaultValuedAttr`
  so a new field does not invalidate existing IR; traits doing the verification
  a trait can do, and `hasVerifier` only for the rule that needs code.
- **`TrainingDialect.cpp`** — the three registrations (types, ops, and the
  dialect itself) and a verifier whose diagnostics name the offending value.
- **`TrainingToLLVM.cpp`** — `adaptor` rather than `op` for operands;
  `getDependentDialects` declaring what the pass *emits*; a `TypeConverter` for
  the custom type; a `ConversionTarget` that marks the two unlowered ops legal
  *because* no pattern handles them; and `applyPartialConversion`, so residual
  ops survive and are visible instead of failing the module.
- **`CMakeLists.txt`** — `set(LLVM_TARGET_DEFINITIONS ...)` as positional
  state, one `mlir_tablegen` call per generator, a single
  `add_public_tablegen_target`, and `${CMAKE_CURRENT_BINARY_DIR}` on the include
  path because generated `.inc` files live in the build tree.

## Build (needs an MLIR development install)

```bash
cmake -S llvm-training/18-mlir-lowering-to-llvm/examples/production-dialect \
      -B build/training-dialect \
      -DMLIR_DIR="$(llvm-config --prefix)/lib/cmake/mlir" \
      -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build build/training-dialect -j2
```

## Run

```bash
build/training-dialect/bin/training-opt --training-to-llvm \
  llvm-training/18-mlir-lowering-to-llvm/examples/production-dialect/scale.mlir
```

Expected: `training.scale` becomes `llvm.mul` against an `llvm.mlir.constant`;
`training.lane_cast` and `training.emit` are unchanged, because the pass is a
partial conversion that declares them legal.

Syntax-only check, with no build at all:

```bash
mlir-opt --allow-unregistered-dialect \
  llvm-training/18-mlir-lowering-to-llvm/examples/production-dialect/scale.mlir
```

## Version note

Written against the MLIR 22/23 API: `OpTy::create(builder, loc, ...)` rather
than `builder.create<OpTy>(loc, ...)`, and `applyPatternsGreedily` rather than
`applyPatternsAndFoldGreedily`. Both new spellings exist in MLIR 22 and the old
ones are gone or deprecated in 23, which is what makes the migration safe to do
before moving the version pin — the lesson this repository's own rail learned
when it moved to LLVM 23.
