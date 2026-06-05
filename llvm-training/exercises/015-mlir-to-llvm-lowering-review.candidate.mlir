// Minimal MLIR-like candidate for Exercise 015. This is intentionally
// pseudo-MLIR: it gives reviewers a stable source shape without requiring an
// MLIR dialect implementation in the repository.
bcir.claim %c {
  rd = [0, 1], wr = [2], lane = H, stride = mixed,
  ham_hint = { domain = HBM, locality = 3 }
}
%src = bcir.bind.read %c[0]
%dst = bcir.bind.write %c[0]
%v = bcir.load %src[%row, %col] { row_stride_bytes = %rs, element_stride_bytes = 4 }
bcir.store %dst[%row, %col], %v { ham_hint = #bcir.ham_hint<HBM, 3> }
