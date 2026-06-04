; Loop containing an unknown call.
;
; Try:
;   opt -S -passes=loop-vectorize not-vectorizable-call.ll -o -
;
; Expected observation:
;   Normal loop vectorization is missed because @opaque has no vector form and
;   is not marked readnone/readonly. LLVM must assume the call may have side
;   effects or access memory in ways that interact with the loop.

source_filename = "not-vectorizable-call.ll"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @opaque(i32)

define void @call_in_loop(ptr noalias nocapture readonly %a,
                          ptr noalias nocapture writeonly %c,
                          i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %ap = getelementptr inbounds i32, ptr %a, i64 %i
  %cp = getelementptr inbounds i32, ptr %c, i64 %i
  %v = load i32, ptr %ap, align 4
  %called = call i32 @opaque(i32 %v)
  store i32 %called, ptr %cp, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
