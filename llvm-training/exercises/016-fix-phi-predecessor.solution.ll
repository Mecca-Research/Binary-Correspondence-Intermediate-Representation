source_filename = "016-fix-phi-predecessor.solution.ll"

define i32 @clamp_positive(i32 %x) {
entry:
  %ispos = icmp sgt i32 %x, 0
  br i1 %ispos, label %positive, label %merge

positive:
  br label %merge

merge:
  %result = phi i32 [ %x, %positive ], [ 0, %entry ]
  ret i32 %result
}
