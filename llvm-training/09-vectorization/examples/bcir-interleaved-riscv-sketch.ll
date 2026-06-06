; BCIR/RISC-V scalable-vector sketch for an interleaved lowering boundary.
;
; This file is target-specific teaching IR. It intentionally uses scalable vector
; types to show the caveat: a BCIR pipeline may describe element streams and
; masks, but it must not assume a compile-time number of RVV lanes.
;
; Try as a verifier/inspection input on LLVM builds with scalable-vector support:
;   opt -S -passes=verify bcir-interleaved-riscv-sketch.ll -o -

source_filename = "bcir-interleaved-riscv-sketch.ll"
target triple = "riscv64-unknown-linux-gnu"

define void @bcir_rvv_interleaved_boundary(ptr %pairs,
                                          ptr %xs,
                                          ptr %ys,
                                          <vscale x 4 x i1> %active,
                                          <vscale x 4 x i32> %x.lanes,
                                          <vscale x 4 x i32> %y.lanes) {
entry:
  ; A real lowering would form x/y lane vectors from an interleaved memory group
  ; using target cost information. This sketch keeps the lane vectors as inputs
  ; so the lesson can focus on the scalable-vector boundary and active mask.
  call void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32> %x.lanes, ptr %xs, i32 4, <vscale x 4 x i1> %active)
  call void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32> %y.lanes, ptr %ys, i32 4, <vscale x 4 x i1> %active)
  ret void
}

declare void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32>, ptr nocapture, i32 immarg, <vscale x 4 x i1>)
