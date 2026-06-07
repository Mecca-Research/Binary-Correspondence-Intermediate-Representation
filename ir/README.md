# BCIR — the Intermediate Representation

This directory **is** the Binary Correspondence Intermediate Representation. It
is separate from `../llvm-training/`, which is an LLVM/MLIR training corpus for
agents and is *not* part of the IR (see the top-level `AGENTS.md`).

## Pipeline and section ownership

```
bcir.surface(text) ──► bcir.core(typed graph) ──┬─► irdl  (pure-IR dialect definition, no compile)
                                                 └─► mlir  (compiled dialect + conversion) ─► mlir.llvm ─► llvm ir
                                                                                              llvm  (textual emitter + ABI substrate)
                                                 runtime (GEM execution engine)
```

| Section    | Role | Builds | Default build? |
|------------|------|--------|----------------|
| `surface/` | tokenizer + parser + ROP/MAP verifier | `bcir-surface` (C++) | yes |
| `core/`    | canonical typed graph model + surface→core builder | `bcir-core` (C++) | yes |
| `irdl/`    | declarative dialect projection — **pure IR, no compilation** | `mlir-opt` round-trip test (skips if absent) | tests only, optional |
| `mlir/`    | compiled MLIR dialect + conversion to LLVM dialect | `bcir-mlir` (C++/MLIR) | no — `-DBCIR_ENABLE_MLIR=ON` |
| `llvm/`    | legal LLVM IR emission + ABI substrate | `bcir-llvm` (C++) | yes |
| `runtime/` | GEM execution engine | `gem-runtime` (C++) | yes |

## Why IRDL and MLIR are separate sections

- **`irdl/` is the contract**: a declarative, no-build structural definition of
  the dialect (types/ops/enums/constraints) that `mlir-opt --irdl-file` loads.
  It is the single source of truth for structure and never needs compiling.
- **`mlir/` is the implementation**: ODS/TableGen + generated C++ ops + a
  registered verifier + conversion patterns. It is opt-in because it needs MLIR
  dev libraries.

Keeping them apart prevents the recurring confusion where "dialect" and "LLVM"
each meant several different things. The old C++ `dialect/` (a hand-written
surface parser, not an MLIR dialect) is now `surface/`, freeing the word
"dialect" for the real MLIR dialect under `mlir/`.

See `../docs/BCIR_Repo_Structure.md` for the full rationale and build matrix.
