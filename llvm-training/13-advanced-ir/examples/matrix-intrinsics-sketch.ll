; Verifier-safe sketch of target-independent LLVM matrix intrinsics.
; The vectors below model 2x2 matrices flattened into four f32 lanes.
; This is not a BCIR intrinsic: it is a compact LLVM reference point for
; preserving matrix shape before a backend, runtime, or JIT chooses how to
; lower the tile.

source_filename = "matrix-intrinsics-sketch.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

; A: <m x k>, B: <k x n>, result: <m x n>. The shape operands are immarg.
declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(
  <4 x float>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

; Transpose a flattened matrix. The row and column operands are immarg.
declare <4 x float> @llvm.matrix.transpose.v4f32(
  <4 x float>, i32 immarg, i32 immarg)

; Column-major movement uses a runtime stride plus immediate volatility/shape.
declare <4 x float> @llvm.matrix.column.major.load.v4f32.i64(
  ptr nocapture, i64, i1 immarg, i32 immarg, i32 immarg)
declare void @llvm.matrix.column.major.store.v4f32.i64(
  <4 x float>, ptr nocapture writeonly, i64, i1 immarg, i32 immarg, i32 immarg)

define void @matrix_2x2_multiply_sketch(ptr %a, ptr %b, ptr %c, i64 %stride) {
entry:
  %a_vec = call <4 x float> @llvm.matrix.column.major.load.v4f32.i64(
    ptr %a, i64 %stride, i1 false, i32 2, i32 2)
  %b_vec = call <4 x float> @llvm.matrix.column.major.load.v4f32.i64(
    ptr %b, i64 %stride, i1 false, i32 2, i32 2)

  ; Optional shape-preserving transform before multiply.
  %b_t = call <4 x float> @llvm.matrix.transpose.v4f32(
    <4 x float> %b_vec, i32 2, i32 2)

  ; Pseudo-lowering boundary: a later pass may scalarize, vectorize, select a
  ; target tile instruction, or rewrite an adjacent BCIR hook to a runtime call.
  %product = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(
    <4 x float> %a_vec, <4 x float> %b_t, i32 2, i32 2, i32 2)

  call void @llvm.matrix.column.major.store.v4f32.i64(
    <4 x float> %product, ptr %c, i64 %stride, i1 false, i32 2, i32 2)
  ret void
}
