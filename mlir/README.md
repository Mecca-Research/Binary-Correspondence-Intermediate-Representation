# `mlir/` — the BCIR dialect family (the IR law)

Source of truth for BCIR syntax/types/attributes/operations, authored in
TableGen/ODS per [`../docs/BCIR_LANGREF.md`](../docs/BCIR_LANGREF.md), plus a pure
**IRDL projection**. The Python package under [`../bcir/`](../bcir/) is the
executable conformance oracle that must agree with these definitions
([`../docs/PARITY.md`](../docs/PARITY.md)).

> **Validated on LLVM 18** (`mlir-opt`/`mlir-tblgen`/`bcir-opt` 18.1.3), gated in
> CI (job `mlir-rail-validate`):
> - **ODS rail** — every TableGen generator (decls **and** defs) passes for the
>   whole `.td` family: `tools/wsl/tblgen_check.sh`.
> - **IRDL rail** — the projection loads into stock `mlir-opt` and the
>   generic-syntax corpus in `test/irdl/` round-trips: `tools/irdl/check_corpus.sh`.
> - **Compiled `bcir-opt` (LangRef M3)** — `lib/BCIRDialect.cpp` + `tools/bcir-opt.cpp`
>   build the real dialect (`tools/wsl/build_mlir.sh`); the *pretty* ODS corpus in
>   `examples/` parses/verifies + FileCheck-round-trips through it
>   (`tools/wsl/check_ods_examples.sh`).

## Dual rail

```
Track A (ODS):   include/BCIR/*.td + passes/*.td + CMakeLists.txt -> bcir-opt
                 full dialect, pretty syntax, R1-R12 verifier, optimizer.
Track B (IRDL):  irdl/bcir.irdl.mlir -> stock mlir-opt --irdl-file=...
                 pure-data structural projection, generic syntax, portability proof.
```

## Dialect family (one `bcir` namespace during bootstrap; distinct layers of law)

| File | Layer | Defines |
|---|---|---|
| `include/BCIR/BCIRDialect.td` | — | dialect + base Op/Type/Attr classes |
| `include/BCIR/BCIRAttrs.td` | — | Lane/StrideClass/Domain/Hazard/Verify/Bounds/Layout/Policy/Semiring/CostClass/MemTier/Access enums; Precision; 12-d CostVector |
| `include/BCIR/BCIRTypes.td` | — | `!bcir.resource`, `!bcir.path`, `!bcir.stream` |
| `include/BCIR/BCIRInterfaces.td` | — | `KBCIRCostOpInterface` |
| `include/BCIR/BCIRCoreOps.td` | BCIR-0..2 | module/registry/resource/phase/claim/index_range/load/store/compute/barrier |
| `include/BCIR/BCIRTargetOps.td` | H | `target.capability` |
| `include/BCIR/BCIRMemOps.td` | CT1 | `mem.tier` / `mem.ham` / `mem.cxl_swap` |
| `include/BCIR/BCIRKBCIROps.td` | BCIR-3 | `kbcir.policy/path/plan/select` |
| `include/BCIR/BCIRGEMOps.td` | BCIR-4 | `gem.stream_pack/lane_segment/prefetch/block` |
| `include/BCIR/BCIRTraceOps.td` | cross | `trace.note` |
| `include/BCIR/BCIRVerifyOps.td` | M1 | `verify.*` (R1–R12 as IR) |
| `include/BCIR/BCIROptOps.td` | M2 | `opt.*` (rewrite/layout/mem laws as IR) |
| `include/BCIR/BCIRLoweringContractOps.td` | M3 | `isa.*` / `packet.*` / `target.lower_contract` |
| `include/BCIR/BCIREventOps.td` | M5 | `event.stream/kind/emit/consume` |
| `include/BCIR/BCIRTransducerOps.td` | M5 | `fsm.machine/state/transition/stack/capture/reduce` |
| `include/BCIR/BCIRParseOps.td` | M5 | `parse.grammar/token/rule/lower_to_fsm` |
| `include/BCIR/BCIRBinaryFormatOps.td` | M5 | `binary.format/field/record/decode` |
| `include/BCIR/BCIROps.td` | — | umbrella (op-gen entry point) |
| `passes/GEMPasses.td` | — | `-bcir-classify-lanes`, `-bcir-select-realization`, `-bcir-batch`, `-bcir-schedule`, `-bcir-lower-to-llvm` |

The canonical pretty IR is `examples/full_vec_add_ct1.mlir`; the IRDL smoke is
`test/irdl/00_smoke_generic.mlir`.
