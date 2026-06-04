; Run:
;   opt -S -passes=loop-rotate loop-rotate-while-before.ll -o loop-rotate-while-after.ll
;
; A while-shaped memory loop. The original header tests before entering the
; body. Rotation creates a guarded preheader and moves the recurring test to
; the latch so later loop passes see a do-while-shaped loop.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "loop-rotate-while-before.ll"

define i32 @sum_until_count(ptr nocapture readonly %a, i64 %n) {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %body ]
  %keep = icmp ult i64 %i, %n
  br i1 %keep, label %body, label %exit

body:
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p, align 4
  %sum.next = add i32 %sum, %v
  %next = add nuw i64 %i, 1
  br label %header

exit:
  ret i32 %sum
}
