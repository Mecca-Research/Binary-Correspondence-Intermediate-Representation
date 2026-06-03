; Exercise 002: If/else control flow joined by a phi node.
; Verification: llvm-as -disable-output llvm-training/exercises/002-if-else-phi.solution.ll

source_filename = "002-if-else-phi.solution.ll"

define i32 @select_max(i32 %a, i32 %b) {
entry:
  %is_greater = icmp sgt i32 %a, %b
  br i1 %is_greater, label %then, label %else

then:
  br label %merge

else:
  br label %merge

merge:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}
