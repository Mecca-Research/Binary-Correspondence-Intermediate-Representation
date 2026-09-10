// Strategy sketch: canonicalize a BCIR graph, normalize its edges, establish
// register/HAM policy, and drive staged lowering. The transform operations are
// upstream operations; pass names beginning with `bcir-` are illustrative
// downstream passes that a BCIR tool must register.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    // Establish canonical payload shapes before operation-name matching.
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    %vertices = transform.select "bcir.vertex" in %root
      : (!transform.any_op) -> !transform.op<"bcir.vertex">
    %edges = transform.select "bcir.edge" in %root
      : (!transform.any_op) -> !transform.op<"bcir.edge">
    transform.print %vertices {name = "vertices before BCIR normalization"}
    transform.print %edges {name = "edges before BCIR normalization"}

    // Each registered pass consumes its root handle because it may rewrite the
    // payload body. Re-match nested payload operations after every such pass.
    %normalized = transform.apply_registered_pass
        "bcir-normalize-edge-attributes" to %root
        : (!transform.any_op) -> !transform.any_op
    %normalized_vertices = transform.select "bcir.vertex" in %normalized
      : (!transform.any_op) -> !transform.op<"bcir.vertex">
    transform.print %normalized_vertices {name = "vertices before pre-lock"}

    %prelocked = transform.apply_registered_pass
        "bcir-prelock-register-bindings" to %normalized
        : (!transform.any_op) -> !transform.any_op
    %hint_vertices = transform.select "bcir.vertex" in %prelocked
      : (!transform.any_op) -> !transform.op<"bcir.vertex">
    transform.print %hint_vertices {name = "vertices receiving HAM policy"}

    %hinted = transform.apply_registered_pass
        "bcir-attach-ham-hints" to %prelocked
        : (!transform.any_op) -> !transform.any_op

    // The BCIR pass materializes graph traversal as affine/vector staging IR.
    %staged = transform.apply_registered_pass
        "bcir-graph-to-affine-vector" to %hinted
        : (!transform.any_op) -> !transform.any_op

    // Representative generic lowering order. A downstream driver must register
    // these passes and adjust the sequence for its pinned MLIR toolchain.
    %no_affine = transform.apply_registered_pass "lower-affine" to %staged
        : (!transform.any_op) -> !transform.any_op
    %no_scf = transform.apply_registered_pass "convert-scf-to-cf" to %no_affine
        : (!transform.any_op) -> !transform.any_op
    %vectors_lowered = transform.apply_registered_pass
        "convert-vector-to-llvm" to %no_scf
        : (!transform.any_op) -> !transform.any_op
    %memrefs_lowered = transform.apply_registered_pass
        "finalize-memref-to-llvm" to %vectors_lowered
        : (!transform.any_op) -> !transform.any_op
    %arith_lowered = transform.apply_registered_pass
        "convert-arith-to-llvm" to %memrefs_lowered
        : (!transform.any_op) -> !transform.any_op
    %funcs_lowered = transform.apply_registered_pass
        "convert-func-to-llvm" to %arith_lowered
        : (!transform.any_op) -> !transform.any_op
    %llvm = transform.apply_registered_pass
        "reconcile-unrealized-casts" to %funcs_lowered
        : (!transform.any_op) -> !transform.any_op

    transform.print %llvm {name = "BCIR payload at the LLVM dialect boundary"}
    transform.yield
  }
}
