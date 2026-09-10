; Teaching snapshot for:
;   opt -S -passes=loop-unroll loop-unroll-count2-before.ll -o loop-unroll-count2-after.ll

target triple = "x86_64-unknown-linux-gnu"
source_filename = "loop-unroll-count2-before.ll"

define void @scale_eight(ptr nocapture %a) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next.1, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p, align 4
  %scaled = shl i32 %v, 1
  store i32 %scaled, ptr %p, align 4
  %next = add nuw nsw i64 %i, 1
  %p.1 = getelementptr inbounds i32, ptr %a, i64 %next
  %v.1 = load i32, ptr %p.1, align 4
  %scaled.1 = shl i32 %v.1, 1
  store i32 %scaled.1, ptr %p.1, align 4
  %next.1 = add nuw nsw i64 %i, 2
  %done.1 = icmp eq i64 %next.1, 8
  br i1 %done.1, label %exit, label %loop

exit:
  ret void
}
