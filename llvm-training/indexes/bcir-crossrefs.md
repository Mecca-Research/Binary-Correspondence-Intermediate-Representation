# Index: Cross-references to the BCIR project

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


The training corpus is separate from the BCIR implementation. Use this table to
move from a training concept to the current executable oracle (`bcir/`) or MLIR
law (`mlir/`) without relying on the retired `ir/` and `runtime/llvm/` trees.

| Concept | Current BCIR implementation/law | Training path |
|---|---|---|
| Registry-first resources and claims | [`bcir/model/graph.py`](../../bcir/model/graph.py), [`bcir/frontends/rop.py`](../../bcir/frontends/rop.py), [`bcir/frontends/map.py`](../../bcir/frontends/map.py) | [`bcir-mapping/05-runtime-abi.md`](../bcir-mapping/05-runtime-abi.md), [`bcir-mapping/06-claim-lowering-pipeline.md`](../bcir-mapping/06-claim-lowering-pipeline.md) |
| Claim parsing and source-to-IR boundary | [`bcir/etl/parse.py`](../../bcir/etl/parse.py) | [`bcir-mapping/examples/claim-resource-lookup.bcir.txt`](../bcir-mapping/examples/claim-resource-lookup.bcir.txt) |
| Graph, lane, opcode, and stride semantics | [`bcir/model/graph.py`](../../bcir/model/graph.py), [`bcir/model/lanes.py`](../../bcir/model/lanes.py), [`bcir/model/opcodes.py`](../../bcir/model/opcodes.py) | [`bcir-mapping/01-vertex-edge-attribute.md`](../bcir-mapping/01-vertex-edge-attribute.md), [`bcir-mapping/03-mixed-stride-graphs.md`](../bcir-mapping/03-mixed-stride-graphs.md) |
| K_BCIR realization and calibration | [`bcir/kbcir/realize.py`](../../bcir/kbcir/realize.py), [`bcir/kbcir/calibrate.py`](../../bcir/kbcir/calibrate.py), [`bcir/kbcir/cost.py`](../../bcir/kbcir/cost.py) | [`17-new-pass-manager/04-adaptive-bcir-pipelines.md`](../17-new-pass-manager/04-adaptive-bcir-pipelines.md), [`19-hardware-aware/03-calibration-governor.md`](../19-hardware-aware/03-calibration-governor.md) |
| GEM execution, concurrency, and async tokens | [`bcir/gem/execute.py`](../../bcir/gem/execute.py), [`bcir/gem/concurrency.py`](../../bcir/gem/concurrency.py), [`bcir/gem/async_tokens.py`](../../bcir/gem/async_tokens.py) | [`19-hardware-aware/02-programming-pulses-and-flow-execution.md`](../19-hardware-aware/02-programming-pulses-and-flow-execution.md), [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| StreamPack prefetch/provenance | [`bcir/gem/streampack.py`](../../bcir/gem/streampack.py) | [`bcir-mapping/04-ham-hints.md`](../bcir-mapping/04-ham-hints.md), [`bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md) |
| LLVM AOT/JIT lowering | [`bcir/lower/llvm.py`](../../bcir/lower/llvm.py), [`bcir/lower/jit.py`](../../bcir/lower/jit.py) | [`12-backend-jit/README.md`](../12-backend-jit/README.md), [`12-backend-jit/07-advanced-orc-runtime-integration.md`](../12-backend-jit/07-advanced-orc-runtime-integration.md) |
| Target selection and code generation | [`bcir/codegen/targets.py`](../../bcir/codegen/targets.py), [`bcir/codegen/codegen.py`](../../bcir/codegen/codegen.py) | [`19-hardware-aware/04-riscv-and-target-specific-codegen.md`](../19-hardware-aware/04-riscv-and-target-specific-codegen.md), [`19-hardware-aware/05-machineir-and-mir-customization.md`](../19-hardware-aware/05-machineir-and-mir-customization.md) |
| Executable verifier laws | [`bcir/verify/__init__.py`](../../bcir/verify/__init__.py), [`bcir/tests/test_verify.py`](../../bcir/tests/test_verify.py) | [`bcir-mapping/11-normal-forms-and-verification.md`](../bcir-mapping/11-normal-forms-and-verification.md) |
| MLIR operation/type/attribute law | [`mlir/include/BCIR/`](../../mlir/include/BCIR/), [`mlir/lib/BCIRDialect.cpp`](../../mlir/lib/BCIRDialect.cpp) | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md), [`18-mlir-lowering-to-llvm/README.md`](../18-mlir-lowering-to-llvm/README.md) |
| Compiled BCIR conversion and verification passes | [`mlir/lib/BCIRPasses.cpp`](../../mlir/lib/BCIRPasses.cpp), [`mlir/test/passes/`](../../mlir/test/passes/) | [`18-mlir-lowering-to-llvm/04-bcir-dialect-to-llvm.md`](../18-mlir-lowering-to-llvm/04-bcir-dialect-to-llvm.md), [`17-new-pass-manager/02-custom-passes-and-analyses.md`](../17-new-pass-manager/02-custom-passes-and-analyses.md) |
| IRDL projection | [`mlir/irdl/bcir.irdl.mlir`](../../mlir/irdl/bcir.irdl.mlir), [`mlir/test/irdl/`](../../mlir/test/irdl/) | [`14-mlir-bridge/02-dialects-and-operations.md`](../14-mlir-bridge/02-dialects-and-operations.md) |

For parity expectations between the Python oracle and MLIR law, read
[`docs/PARITY.md`](../../docs/PARITY.md).
