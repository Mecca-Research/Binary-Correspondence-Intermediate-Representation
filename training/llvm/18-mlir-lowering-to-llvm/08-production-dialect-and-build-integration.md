# Production Dialect Implementation and Build Integration

Chapters [`01`](01-conversion-infrastructure.md)–[`07`](07-custom-types-attributes-and-metadata.md)
describe conversion concepts. This chapter covers the part that a sketch cannot
show: **how ODS becomes a built, registered, testable dialect**, and where each
piece of that machinery lives.

It is anchored on a production dialect that this repository builds in CI — the
BCIR law rail in [`../../mlir/`](../../../mlir). Every file and CMake idiom cited
below is checked by
[`../tools/verify-mlir-rail-references.py`](../tools/verify-mlir-rail-references.py),
so this chapter fails the build rather than going quietly stale.

A minimal standalone skeleton, for reading in one sitting, is in
[`examples/production-dialect/`](examples/production-dialect).

## The build graph, end to end

```
  *.td  ──mlir-tblgen──▶  *.h.inc / *.cpp.inc  ──▶  dialect library  ──▶  tool
   │                            (build tree)          (MLIRBCIR)         (bcir-opt)
   │                                                       │
   └── ODS: dialect, ops, types, attributes, interfaces     └── passes register here
```

Four rules follow from that picture, and most first-time build failures are one
of them:

1. **Generated `.inc` files live in the build tree, never the source tree.**
   They are included by hand-written headers, so the build directory must be on
   the include path:

   ```cmake
   include_directories(${CMAKE_CURRENT_BINARY_DIR})  # TableGen-generated *.inc
   ```

2. **Each generator invocation is a separate `mlir_tablegen` call**, grouped
   into one target with `add_public_tablegen_target`. That target is what other
   targets depend on; depending on the `.td` file directly does not order the
   build correctly.

3. **`set(LLVM_TARGET_DEFINITIONS ...)` is positional state.** It applies to the
   `mlir_tablegen` calls that follow it, until the next `set`. Reordering blocks
   silently generates the wrong thing.

4. **One `.td` can drive several generators.** The dialect and its ops come from
   the same umbrella file, with different flags.

### The real wiring

From [`../../mlir/CMakeLists.txt`](../../../mlir/CMakeLists.txt):

```cmake
# --- enums + attributes ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIRAttrs.td)
mlir_tablegen(BCIREnums.h.inc   -gen-enum-decls)
mlir_tablegen(BCIREnums.cpp.inc -gen-enum-defs)
mlir_tablegen(BCIRAttrs.h.inc   -gen-attrdef-decls)
mlir_tablegen(BCIRAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(BCIRAttrsIncGen)

# --- types ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIRTypes.td)
mlir_tablegen(BCIRTypes.h.inc   -gen-typedef-decls)
mlir_tablegen(BCIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(BCIRTypesIncGen)

# --- op interfaces ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIRInterfaces.td)
mlir_tablegen(BCIRInterfaces.h.inc   -gen-op-interface-decls)
mlir_tablegen(BCIRInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(BCIRInterfacesIncGen)

# --- dialect + ops (umbrella pulls the whole family) ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIROps.td)
mlir_tablegen(BCIRDialect.h.inc   -gen-dialect-decls -dialect=bcir)
mlir_tablegen(BCIRDialect.cpp.inc -gen-dialect-defs  -dialect=bcir)
mlir_tablegen(BCIROps.h.inc       -gen-op-decls)
mlir_tablegen(BCIROps.cpp.inc     -gen-op-defs)
add_public_tablegen_target(BCIROpsIncGen)

add_custom_target(bcir_tablegen DEPENDS
  BCIRAttrsIncGen BCIRTypesIncGen BCIRInterfacesIncGen BCIROpsIncGen)

add_mlir_dialect_library(MLIRBCIR
  lib/BCIRDialect.cpp
  lib/BCIRPasses.cpp                 # registration only
  lib/passes/BCIRVerifyPass.cpp
  lib/passes/BCIRPromotePass.cpp
  lib/passes/BCIRConvertToLLVM.cpp
  ...)
```

The generator flags, in the order you will need them:

| Flag | Produces |
| --- | --- |
| `-gen-dialect-decls` / `-gen-dialect-defs` | the dialect class; needs `-dialect=<name>` |
| `-gen-op-decls` / `-gen-op-defs` | one C++ class per operation |
| `-gen-typedef-decls` / `-gen-typedef-defs` | custom type classes |
| `-gen-attrdef-decls` / `-gen-attrdef-defs` | custom attribute classes |
| `-gen-enum-decls` / `-gen-enum-defs` | enum attributes and their parsers |
| `-gen-op-interface-decls` / `-gen-op-interface-defs` | op interfaces |
| `-gen-pass-decls -name <Group>` | pass base classes and registration from a `Passes.td` |
| `-gen-rewriters` | DRR patterns declared in TableGen |

## Writing ODS that survives review

Real ODS from [`../../mlir/include/BCIR/BCIRCoreOps.td`](../../../mlir/include/BCIR/BCIRCoreOps.td):

```tablegen
defvar BCIR_ClaimAttrs = (ins
  BCIR_LaneAttr:$lane,
  BCIR_StrideClassAttr:$stride_class,
  DefaultValuedAttr<I32Attr, "1">:$stride_k,
  BCIR_DomainAttr:$domain,
  BCIR_HazardModeAttr:$hazard,
  OptionalAttr<BCIR_PrecisionAttr>:$precision,
  DefaultValuedAttr<BoolAttr, "false">:$dynamic,
  OptionalAttr<BCIR_TimingAttr>:$timing,
  OptionalAttr<BCIR_LifetimeAttr>:$lifetime,
  ...);
```

Three practices visible in that fragment, all of which are load-bearing:

- **Shared argument bundles via `defvar`.** A contract carried by several ops is
  declared once. The alternative — repeating fifteen attributes across five op
  definitions — is how the fifth one ends up missing an attribute.
- **`OptionalAttr` and `DefaultValuedAttr` for every new field.** This is what
  makes a new attribute *non-disturbing*: existing IR still parses, still
  verifies, and still hashes the same. Adding a required attribute invalidates
  the entire existing corpus in one commit.
- **Comments that name the law and the mirror.** Each attribute says which
  verifier law reads it and which Python-side field it mirrors. In a two-rail
  system, an ODS field with no named counterpart is a field that will drift.

### Traits are where the cheap verification lives

```tablegen
def Training_ScaleOp : Training_Op<"scale", [Pure, SameOperandsAndResultType]> {
```

Prefer a trait over hand-written `verify()` code whenever one fits:
`Pure`, `SameOperandsAndResultType`, `SameOperandsAndResultShape`,
`Commutative`, `Idempotent`, `Involution`, `Terminator`,
`SingleBlockImplicitTerminator<"...">`, `IsolatedFromAbove`,
`DeclareOpInterfaceMethods<...>`. Traits are checked uniformly, compose, and
inform the optimizer; a hand-written check does none of those things.

Write `let hasVerifier = 1;` only for the invariants no trait expresses — which,
in a domain dialect, is usually the interesting half.

## Registration: three separate things

Registration trips people up because there are three of them and they fail
differently.

```cpp
// 1. The dialect, into a registry (mlir/tools/bcir-opt.cpp)
mlir::DialectRegistry registry;
registry.insert<bcir::BCIRDialect, mlir::LLVM::LLVMDialect,
                mlir::func::FuncDialect, ...>();
return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "BCIR optimizer driver\n", registry));
```

```cpp
// 2. The passes, by name (mlir/lib/BCIRPasses.cpp)
::mlir::registerPass([] { return createVerifyPass(); });
::mlir::registerPass([] { return createPromoteLanesPass(); });
::mlir::registerPass([] { return createConvertToLLVMPass(); });
```

```cpp
// 3. The ops, into the dialect (generated; called from initialize())
addOperations<
#define GET_OP_LIST
#include "BCIROps.cpp.inc"
    >();
```

| Missing registration | Symptom |
| --- | --- |
| Dialect | `error: dialect 'x' not found`, or ops parse only with `--allow-unregistered-dialect` |
| Pass | `unknown pass name` from the pipeline parser |
| Ops in `initialize()` | dialect loads, its ops do not exist |
| A dialect a *pattern* produces | conversion fails at the moment the op is created |

The last row is the one that costs an afternoon: a conversion pass must register
every dialect it can *emit*, not just the ones it can read.

## Testing: `lit` + `FileCheck`

The production rail's pass tests live under
[`../../mlir/test/passes/`](../../../mlir/test/passes) (the current inventory is in
the generated [`../../docs/STATUS.md`](../../../docs/STATUS.md), never in prose).
The shape is always the same:

```mlir
// RUN: bcir-opt --bcir-verify %s | FileCheck %s
// RUN: not bcir-opt --bcir-verify %s.invalid 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-LABEL: func @scaled
// CHECK: training.scale
```

Conventions worth adopting wholesale:

- **`CHECK-LABEL` per function.** It resets FileCheck's matching window, so one
  failure does not cascade into ten.
- **A negative test per verifier rule.** MLIR's `expected-error` directive with
  `mlir-opt -verify-diagnostics` pins the *diagnostic*, not just the failure:

  ```mlir
  // expected-error @+1 {{lane width must be a power of two}}
  %0 = training.scale %arg0 { lane = 3 : i32 } : i32
  ```

  The production rail carries an `expected-error` marker for every law it
  enforces. A law with no negative fixture is a law nobody has watched fail.
- **`--split-input-file`** to keep many small cases in one file.
- **Pin the pipeline in the `RUN` line**, not in a shell wrapper, so the test is
  reproducible from its own text.

## Pitfalls

- **Editing a generated `.inc` file.** It is regenerated on the next build.
- **Forgetting `${CMAKE_CURRENT_BINARY_DIR}` on the include path.** Every
  generated include fails to resolve, and the error names the `.inc` file rather
  than the missing path.
- **Reordering `set(LLVM_TARGET_DEFINITIONS ...)` blocks.** Silently generates
  from the wrong `.td`.
- **A required new ODS attribute.** Invalidates all existing IR; use
  `OptionalAttr`/`DefaultValuedAttr` and land the law vacuously.
- **Hand-written `verify()` where a trait exists.** More code, less information
  for the optimizer.
- **Registering the dialect but not the passes** (or the reverse). Two different
  errors, both reported as "unknown".
- **Mixing MLIR major versions between the libraries and the tool.** The C++ ABI
  is not stable; build both with the same toolchain. In this repository, an
  `bcir-opt` built with a different compiler than the MLIR libraries crashed at
  startup — a documented lesson, not a hypothetical.

## BCIR notes

- The law rail exists so that legality is decided **before** cost. Ops carry
  their contract as attributes and the verifier passes read it; the cost passes
  come after and never override a verdict.
- Cost-recomputation passes in this rail are marked *informs-only* in their own
  source comments. That labelling is the two-truth separation made visible at
  the file level: a pass that recomputes a price may annotate, never reject.
- `bcir-aot` is deliberately a **partial** preparation: it may leave residual
  BCIR/GEM operations rather than pretending a full lowering succeeded. Partial
  conversion that reports what it could not convert is the honest design, and
  [`09-production-conversion-pass.md`](09-production-conversion-pass.md) shows
  the mechanism.

## See also

- [`09-production-conversion-pass.md`](09-production-conversion-pass.md) — the conversion pass in production form
- [`examples/production-dialect/README.md`](examples/production-dialect/README.md) — the minimal standalone skeleton
- [`../14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md) — the conceptual sketch this chapter makes concrete
- [`../../mlir/README.md`](../../../mlir/README.md) — the production rail's own documentation
