# BCIR compiled MLIR dialect + conversion

This section is the **compiled** MLIR path for BCIR — distinct from the pure-IR
`../irdl/` projection.

- **This is compilation.** ODS/TableGen op definitions, generated C++ op
  classes, a registered C++ verifier, custom types/attributes, and
  dialect-conversion patterns that lower BCIR → the LLVM dialect → LLVM IR.
- **Opt-in.** Requires MLIR/LLVM development libraries. Built only with
  `-DBCIR_ENABLE_MLIR=ON`; OFF by default so the default build and CI stay light
  and need no MLIR toolchain.

## Relationship to the IRDL projection

`../irdl/` is the declarative, no-build structural definition (the contract).
This section is the executable implementation of that same dialect plus the
lowering pipeline. Keeping them in sibling directories makes the boundary
explicit: structure/validation lives in `irdl/`; compiled ops, verifier, and
conversion live here.

## Planned layout

```
include/bcir/mlir/     public dialect headers (generated + hand-written)
lib/Dialect/           ODS/TableGen op + type + attribute definitions, verifier
lib/Conversion/        BCIR -> LLVM dialect conversion patterns + TypeConverter
test/                  lit/FileCheck tests for ops, verifier, and lowering
```

## Lowering target

The conversion target is **legal LLVM dialect / LLVM IR only**
(`llvm.load/store/atomicrmw/cmpxchg/fence/call`), consistent with the
non-regression rules in `docs/BCIR_Codex_Blueprint.md`. Atomics are never
rewritten into load/op/store pseudo-atomics.
