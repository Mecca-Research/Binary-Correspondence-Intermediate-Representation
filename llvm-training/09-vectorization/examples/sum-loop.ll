; Scalar loop-vectorization input.
;
; Try:
;   opt -S -passes=loop-vectorize sum-loop.ll -o -
;   opt -S -passes='default<O3>' sum-loop.ll -o -

source_filename = "sum-loop.ll"
target triple = "x86_64-unknown-linux-gnu"

define i32 @sum_loop(ptr noalias nocapture readonly %a, i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p, align 4
  %sum.next = add i32 %sum, %v
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %result = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  ret i32 %result
}

define void @add_arrays(ptr noalias nocapture readonly %a,
                        ptr noalias nocapture readonly %b,
                        ptr noalias nocapture writeonly %c,
                        i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds i32, ptr %a, i64 %i
  %pb = getelementptr inbounds i32, ptr %b, i64 %i
  %pc = getelementptr inbounds i32, ptr %c, i64 %i
  %va = load i32, ptr %pa, align 4
  %vb = load i32, ptr %pb, align 4
  %sum = add i32 %va, %vb
  store i32 %sum, ptr %pc, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
