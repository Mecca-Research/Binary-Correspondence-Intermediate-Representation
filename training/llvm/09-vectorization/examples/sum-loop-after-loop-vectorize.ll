; Example of the kind of IR produced after Loop Vectorization.
;
; This is a teaching-sized, cleaned-up snapshot rather than a guarantee of
; exact output from every LLVM version or target. Real output may use a
; different vector width, add interleaving, or include runtime checks.
;
; Expected observation:
;   The vector loop uses <4 x i32> values, vector loads/stores, and a reduction
;   intrinsic to combine vector lanes for the scalar sum result.

source_filename = "sum-loop-after-loop-vectorize.ll"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)

define i32 @sum_loop(ptr noalias nocapture readonly %a, i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %vector.check, label %exit

vector.check:
  %min.iters.check = icmp ult i64 %n, 4
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:
  %n.vec = and i64 %n, -4
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.sum = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %vec.sum.next, %vector.body ]
  %p.vec = getelementptr inbounds i32, ptr %a, i64 %index
  %wide.load = load <4 x i32>, ptr %p.vec, align 4
  %vec.sum.next = add <4 x i32> %vec.sum, %wide.load
  %index.next = add nuw i64 %index, 4
  %vector.done = icmp eq i64 %index.next, %n.vec
  br i1 %vector.done, label %middle.block, label %vector.body

middle.block:
  %sum.reduced = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %vec.sum.next)
  %all.done = icmp eq i64 %n.vec, %n
  br i1 %all.done, label %exit, label %scalar.ph

scalar.ph:
  %i.scalar = phi i64 [ 0, %vector.check ], [ %n.vec, %middle.block ]
  %sum.scalar = phi i32 [ 0, %vector.check ], [ %sum.reduced, %middle.block ]
  br label %scalar.loop

scalar.loop:
  %i = phi i64 [ %i.scalar, %scalar.ph ], [ %i.next, %scalar.loop ]
  %sum = phi i32 [ %sum.scalar, %scalar.ph ], [ %sum.next, %scalar.loop ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p, align 4
  %sum.next = add i32 %sum, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %scalar.loop

exit:
  %result = phi i32 [ 0, %entry ], [ %sum.reduced, %middle.block ], [ %sum.next, %scalar.loop ]
  ret i32 %result
}

define void @add_arrays(ptr noalias nocapture readonly %a,
                        ptr noalias nocapture readonly %b,
                        ptr noalias nocapture writeonly %c,
                        i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %vector.check, label %exit

vector.check:
  %min.iters.check = icmp ult i64 %n, 4
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:
  %n.vec = and i64 %n, -4
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %pa.vec = getelementptr inbounds i32, ptr %a, i64 %index
  %pb.vec = getelementptr inbounds i32, ptr %b, i64 %index
  %pc.vec = getelementptr inbounds i32, ptr %c, i64 %index
  %wide.load.a = load <4 x i32>, ptr %pa.vec, align 4
  %wide.load.b = load <4 x i32>, ptr %pb.vec, align 4
  %wide.sum = add <4 x i32> %wide.load.a, %wide.load.b
  store <4 x i32> %wide.sum, ptr %pc.vec, align 4
  %index.next = add nuw i64 %index, 4
  %vector.done = icmp eq i64 %index.next, %n.vec
  br i1 %vector.done, label %middle.block, label %vector.body

middle.block:
  %all.done = icmp eq i64 %n.vec, %n
  br i1 %all.done, label %exit, label %scalar.ph

scalar.ph:
  %i.scalar = phi i64 [ 0, %vector.check ], [ %n.vec, %middle.block ]
  br label %scalar.loop

scalar.loop:
  %i = phi i64 [ %i.scalar, %scalar.ph ], [ %i.next, %scalar.loop ]
  %pa = getelementptr inbounds i32, ptr %a, i64 %i
  %pb = getelementptr inbounds i32, ptr %b, i64 %i
  %pc = getelementptr inbounds i32, ptr %c, i64 %i
  %va = load i32, ptr %pa, align 4
  %vb = load i32, ptr %pb, align 4
  %sum = add i32 %va, %vb
  store i32 %sum, ptr %pc, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %scalar.loop

exit:
  ret void
}
