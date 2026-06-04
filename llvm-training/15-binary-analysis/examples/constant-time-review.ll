; Static side-channel review example.
; Verification: llvm-as -disable-output llvm-training/15-binary-analysis/examples/constant-time-review.ll

source_filename = "constant-time-review.ll"

define i32 @branch_on_secret(i32 %secret, i32 %a, i32 %b) {
entry:
  %bit = and i32 %secret, 1
  %cond = icmp ne i32 %bit, 0
  br i1 %cond, label %then, label %else

then:
  ret i32 %a

else:
  ret i32 %b
}

define i32 @select_on_secret(i32 %secret, i32 %a, i32 %b) {
entry:
  %bit = and i32 %secret, 1
  %cond = icmp ne i32 %bit, 0
  %v = select i1 %cond, i32 %a, i32 %b
  ret i32 %v
}
