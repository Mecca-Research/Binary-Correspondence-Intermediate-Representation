; Vector Predication: the llvm.vp.* family, and the two things that gate a lane.
;
; A masked operation asks ONE question per lane: is this lane active? A vector-predicated
; operation asks TWO, and they are independent:
;
;   %mask  -- which lanes are active, lane by lane (a per-lane conditional)
;   %evl   -- how many lanes are live at all (the "explicit vector length", a tail bound)
;
; Every llvm.vp.* intrinsic ends with exactly those two operands. Lanes at or beyond %evl
; take no part in the operation and their results are poison; lanes below %evl behave as
; the mask says. That is why VP handles a loop tail without a scalar remainder loop: the
; final iteration simply passes a smaller %evl.
;
; These assemble on the corpus baseline -- the family predates it -- so no REQUIRES
; directive is needed. What is version-dependent is how well a target lowers them.

target triple = "riscv64-unknown-linux-gnu"

; The tail-handling loop VP exists for. llvm.experimental.get.vector.length asks the
; target how many elements it is willing to process given the remaining trip count, the
; vectorization factor, and whether the type is scalable. On RVV this becomes `vsetvli`,
; the instruction that sets the active length -- so %evl is not a software fiction, it is
; the value the hardware is configured with.
define void @vp_add_loop(ptr %dst, ptr %a, ptr %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %remaining = sub i64 %n, %i
  %evl = call i32 @llvm.experimental.get.vector.length.i64(i64 %remaining, i32 4, i1 true)

  %pa = getelementptr i32, ptr %a, i64 %i
  %pb = getelementptr i32, ptr %b, i64 %i
  %pd = getelementptr i32, ptr %dst, i64 %i

  ; No lane-local condition here, so the mask is all-true and %evl alone bounds the work.
  ; Splitting the two concerns is the point: a conditional inside the loop body would
  ; change the mask and leave %evl exactly as it is.
  %all = shufflevector <vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0),
                       <vscale x 4 x i1> poison,
                       <vscale x 4 x i32> zeroinitializer

  %va = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr %pa, <vscale x 4 x i1> %all, i32 %evl)
  %vb = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr %pb, <vscale x 4 x i1> %all, i32 %evl)
  %vs = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %va, <vscale x 4 x i32> %vb,
                                                     <vscale x 4 x i1> %all, i32 %evl)
  call void @llvm.vp.store.nxv4i32.p0(<vscale x 4 x i32> %vs, ptr %pd,
                                      <vscale x 4 x i1> %all, i32 %evl)

  %stepped = zext i32 %evl to i64
  %next = add i64 %i, %stepped
  %done = icmp uge i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Both gates at once: a lane-local condition AND a tail bound. The mask says which lanes
; the branch selected; %evl says how many lanes exist this iteration. Neither substitutes
; for the other -- a mask cannot shorten the vector, and a length cannot express "every
; lane except the third".
define <vscale x 4 x i32> @vp_masked_and_bounded(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b,
                                                 <vscale x 4 x i32> %limit, i32 %evl) {
  %mask = icmp ult <vscale x 4 x i32> %a, %limit
  %r = call <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b,
                                                    <vscale x 4 x i1> %mask, i32 %evl)
  ret <vscale x 4 x i32> %r
}

; For contrast, the masked family this corpus already taught. llvm.masked.load takes a
; mask and a PASSTHRU for the inactive lanes, and has no length operand at all: it cannot
; express a tail, only a per-lane predicate. VP's load needs no passthru because lanes at
; or beyond %evl are poison by definition rather than merged from somewhere.
define <4 x i32> @masked_for_contrast(ptr %p, <4 x i1> %mask, <4 x i32> %passthru) {
  %v = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr %p, i32 4, <4 x i1> %mask, <4 x i32> %passthru)
  ret <4 x i32> %v
}

declare i32 @llvm.experimental.get.vector.length.i64(i64, i32 immarg, i1 immarg)
declare <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr, <vscale x 4 x i1>, i32)
declare void @llvm.vp.store.nxv4i32.p0(<vscale x 4 x i32>, ptr, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.add.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i1>, i32)
declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i32 immarg, <4 x i1>, <4 x i32>)
