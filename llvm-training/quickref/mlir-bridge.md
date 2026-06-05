# Quickref: MLIR Bridge

Use this sheet when a frontend, BCIR dialect sketch, or MLIR LLVM-dialect file
is the source of the final LLVM IR.

## Lowering decision path

| Stage | Read | Output to review |
| --- | --- | --- |
| Identify MLIR structure | [`../14-mlir-bridge/01-what-is-mlir.md`](../14-mlir-bridge/01-what-is-mlir.md) | Modules, operations, regions, blocks, attributes, and types. |
| Find dialect boundaries | [`../14-mlir-bridge/02-dialects-and-operations.md`](../14-mlir-bridge/02-dialects-and-operations.md) | Which facts belong in domain dialects versus generic/control-flow/LLVM dialects. |
| Convert to LLVM dialect | [`../14-mlir-bridge/03-lowering-to-llvm-dialect.md`](../14-mlir-bridge/03-lowering-to-llvm-dialect.md) | Pointer, memref-descriptor, index-width, vector, and call-boundary decisions. |
| Preserve BCIR domain concepts | [`../14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md) | `bcir.vertex`, `bcir.edge`, HAM hints, register binding, and mixed-stride operations. |
| Walk a graph lowering | [`../14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md) | Source MLIR, LLVM-dialect MLIR, and textual LLVM IR for the same graph. |

## BCIR lowering checkpoints

- Keep vertex IDs, edge lists, register bindings, and HAM hints explicit until
  the lowering has committed to concrete structs, GEPs, loads/stores, metadata,
  or runtime ABI calls.
- Use [`../bcir-mapping/README.md`](../bcir-mapping/README.md) once the output is
  textual LLVM IR or an ABI wrapper rather than an MLIR operation.
- Use [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md)
  for the final LLVM instruction syntax and
  [`advanced-ir.md`](advanced-ir.md) when intrinsics, attributes, poison, or
  fast-math contracts appear in the generated IR.

## Verification commands

```bash
./llvm-training/tools/verify-mlir-examples.sh
./llvm-training/tools/verify-examples.sh
./llvm-training/tools/verify-bcir-mapping.sh
```
