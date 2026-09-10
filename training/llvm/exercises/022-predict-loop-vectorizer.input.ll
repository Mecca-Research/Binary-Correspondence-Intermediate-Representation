source_filename = "022-predict-loop-vectorizer.input.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @scale_i32(ptr noalias nocapture readonly %src,
                       ptr noalias nocapture writeonly %dst,
                       i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %srcp = getelementptr inbounds i32, ptr %src, i64 %i
  %dstp = getelementptr inbounds i32, ptr %dst, i64 %i
  %v = load i32, ptr %srcp, align 4
  %scaled = shl i32 %v, 1
  store i32 %scaled, ptr %dstp, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
