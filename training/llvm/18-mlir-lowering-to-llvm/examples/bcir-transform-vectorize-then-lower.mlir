// Strategy sketch: select BCIR vertices, invoke a BCIR-owned selective
// vectorization pass, then lower vector/scf/memref/arith/func operations to LLVM
// dialect. Pass names beginning with `bcir-` are illustrative downstream passes.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    %vertices = transform.select "bcir.vertex" in %root
      : (!transform.any_op) -> !transform.op<"bcir.vertex">
    transform.print %vertices {name = "BCIR vectorization candidates"}

    // The BCIR pass checks unit stride, pre-lock conflicts, and HAM/claim
    // preservation. Non-vectorizable vertices follow its documented scalar
    // fallback; unexpected semantic failures must be diagnosed and propagated.
    %vector_root = transform.apply_registered_pass "bcir-vectorize-vertices"
        with options = {"vector-width" = 4 : i64, "tail-policy" = "masked"}
        to %root
        : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %vector_root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    %vector_loops = transform.apply_registered_pass
        "convert-vector-to-scf" to %vector_root
        : (!transform.any_op) -> !transform.any_op
    %cfg = transform.apply_registered_pass "convert-scf-to-cf" to %vector_loops
        : (!transform.any_op) -> !transform.any_op
    %vector_llvm = transform.apply_registered_pass
        "convert-vector-to-llvm" to %cfg
        : (!transform.any_op) -> !transform.any_op
    %memref_llvm = transform.apply_registered_pass
        "finalize-memref-to-llvm" to %vector_llvm
        : (!transform.any_op) -> !transform.any_op
    %arith_llvm = transform.apply_registered_pass
        "convert-arith-to-llvm" to %memref_llvm
        : (!transform.any_op) -> !transform.any_op
    %func_llvm = transform.apply_registered_pass
        "convert-func-to-llvm" to %arith_llvm
        : (!transform.any_op) -> !transform.any_op
    %llvm = transform.apply_registered_pass
        "reconcile-unrealized-casts" to %func_llvm
        : (!transform.any_op) -> !transform.any_op

    // A required downstream verifier should fail if BCIR or staging operations
    // remain, or if claim/prelock/HAM diagnostics were lost during replacement.
    %verified = transform.apply_registered_pass
        "bcir-verify-llvm-boundary" to %llvm
        : (!transform.any_op) -> !transform.any_op
    transform.print %verified {name = "verified LLVM dialect payload"}
    transform.yield
  }
}
