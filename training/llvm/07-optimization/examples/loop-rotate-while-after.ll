; Teaching snapshot for:
;   opt -S -passes=loop-rotate loop-rotate-while-before.ll -o loop-rotate-while-after.ll

target triple = "x86_64-unknown-linux-gnu"
source_filename = "loop-rotate-while-before.ll"

define i32 @sum_until_count(ptr nocapture readonly %a, i64 %n) {
entry:
  %keep1 = icmp ult i64 0, %n
  br i1 %keep1, label %body.preheader, label %exit

body.preheader:
  br label %body

body:
  %sum = phi i32 [ 0, %body.preheader ], [ %sum.next, %body ]
  %i = phi i64 [ 0, %body.preheader ], [ %next, %body ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p, align 4
  %sum.next = add i32 %sum, %v
  %next = add nuw i64 %i, 1
  %keep = icmp ult i64 %next, %n
  br i1 %keep, label %body, label %exit.loopexit

exit.loopexit:
  %sum.lcssa = phi i32 [ %sum.next, %body ]
  br label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %sum.lcssa, %exit.loopexit ]
  ret i32 %result
}
