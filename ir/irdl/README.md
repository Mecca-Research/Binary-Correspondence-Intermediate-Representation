# BCIR IRDL projection (pure IR — no compilation)

This section defines the BCIR dialect **declaratively**, as IR, using MLIR's
[IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/) (IR Definition Language).

- **No compilation.** No TableGen/ODS, no C++ op classes, no lowering. The
  definition is data that `mlir-opt` loads at runtime:

  ```bash
  mlir-opt --irdl-file=bcir.irdl.mlir program-in-bcir-dialect.mlir
  ```

- **Single source of truth for structure.** Types, ops, attributes, and the
  closed enums (`U/UX/T/GGG/A/H`, edge kinds, hazard kinds, contract modes,
  opcodes) are projected here from `../core/include/bcir/bcir_ir.hpp`. The repo
  previously defined lanes/opcodes in four places (C++ model, surface parser,
  the deleted `.ll` seed metadata, training docs); IRDL replaces that drift with
  one declarative definition.

## Scope boundary

| IRDL encodes (here)                          | Stays in the C++ surface verifier        |
|----------------------------------------------|------------------------------------------|
| operand/result typing                        | phase monotonicity across ops            |
| attribute presence + types                   | epoch/phase legality (reset rules)       |
| closed enum membership (lane, hazard, …)     | hazard contracts (RAW/WAR/WAW proofs)    |
| scalar bounds (phase 0..4095, lane 0..63)    | concurrent registry / atomic constraints |

IRDL is structural. Anything cross-op or semantic remains authoritative in
`../surface/` (and, later, in the compiled MLIR verifier in `../mlir/`).

## Layout

```
bcir.irdl.mlir   the dialect definition (scaffold today; full projection next)
examples/        programs in the bcir dialect to load against the definition
test/            negative fixtures the IRDL constraints must reject
```

## Build / test

The round-trip test is registered by CMake **only when `mlir-opt` is found**
(`BCIR_ENABLE_MLIR_TOOL_TESTS=ON`, default). Without `mlir-opt` it skips
cleanly — the IRDL projection is never on the default build's critical path.

## Next build steps (tracked)

1. Project the closed enums (lane/edge/hazard/contract/opcode) as `irdl.is` /
   `irdl.any_of` constraints.
2. Project the base types (`@vertex`, `@edge`, `@graph`, registry/claim) with
   parameters.
3. Project the op set (`load/store/bin/lane/phase/barrier`, `map_*`, graph ops).
4. Add positive `examples/` and negative `test/` fixtures.
5. Add a parity guard so every enum/op in `bcir_ir.hpp` has an IRDL construct.
