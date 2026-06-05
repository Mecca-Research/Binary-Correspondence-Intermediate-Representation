# 05 — Affine, vector, and LLVM lowering pipeline

BCIR graph operations do not have to lower directly to LLVM dialect. When the
shape is regular enough, use affine and vector dialects as optimization staging
areas.

## Pipeline shape

```text
BCIR graph dialect
  -> canonical BCIR graph fragments
  -> affine loops for regular traversal
  -> vector dialect for packed edge/vertex work
  -> SCF/arith/memref lowering
  -> LLVM dialect
  -> LLVM IR
```

The purpose of staging is to expose loop bounds, strides, vector widths, and
memory effects before the graph identity is flattened.

## Graph to affine

Lower to affine when the graph fragment has statically known or symbolically
bounded structure:

- fixed fanout;
- rectangular adjacency windows;
- mixed-stride memory access expressible as affine maps;
- predictable loop-carried claim or diagnostic state.

The example [`examples/bcir-graph-to-affine.mlir`](examples/bcir-graph-to-affine.mlir)
shows the intended shape.

## Graph to vector

Lower to vector when multiple edges or vertex attributes can be processed
uniformly:

- edge weights in `vector<4xf32>`;
- vertex IDs in `vector<4xi64>`;
- masked transfers for ragged graph tails;
- vectorized HAM prefetch planning.

The example [`examples/bcir-graph-to-vector.mlir`](examples/bcir-graph-to-vector.mlir)
shows a vector staging boundary.

## Vector/affine to LLVM dialect

Generic MLIR lowering usually handles arith, scf, affine, memref, and vector
pieces, but BCIR metadata must be carried manually. Attach graph/claim metadata
to the memory operation, call, or descriptor that will still exist after generic
lowering.

## Pass-order guidance

1. Canonicalize BCIR operations while graph identity is still explicit.
2. Attach metadata and descriptor side tables.
3. Lower regular graph fragments to affine/vector.
4. Lower BCIR resource/HAM/GAADMSF operations to calls, intrinsics, or LLVM ops.
5. Run generic dialect conversion.
6. Translate LLVM dialect and verify LLVM IR.

Running generic conversion too early can obscure graph structure and block vector
or affine opportunities.
