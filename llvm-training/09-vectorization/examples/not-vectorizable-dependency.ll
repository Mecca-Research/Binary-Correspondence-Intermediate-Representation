; Loop with a true loop-carried dependency.
;
; Try:
;   opt -S -passes=loop-vectorize not-vectorizable-dependency.ll -o -
;
; Expected observation:
;   Normal loop vectorization is missed because iteration i loads a[i - 1],
;   which may be the value stored by iteration i - 1. LLVM must preserve that
;   sequential dependency.

source_filename = "not-vectorizable-dependency.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @prefix_increment(ptr nocapture %a, i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 1
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 1, %entry ], [ %i.next, %loop ]
  %prev.index = add nsw i64 %i, -1
  %prev.ptr = getelementptr inbounds i32, ptr %a, i64 %prev.index
  %cur.ptr = getelementptr inbounds i32, ptr %a, i64 %i
  %prev = load i32, ptr %prev.ptr, align 4
  %next = add i32 %prev, 1
  store i32 %next, ptr %cur.ptr, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
