; Fixed form: one incoming pair for each actual predecessor.
source_filename = "phi-predecessor-fixed.c"

define i32 @fixed_phi(i1 %c) {
entry:
  br i1 %c, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
  %x = phi i32 [ 1, %left ], [ 2, %right ]
  ret i32 %x
}
