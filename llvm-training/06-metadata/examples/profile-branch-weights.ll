; Branch weight metadata records expected edge frequencies.
source_filename = "profile-branch-weights.c"

define i32 @likely_positive(i32 %x) {
entry:
  %positive = icmp sgt i32 %x, 0
  br i1 %positive, label %hot, label %cold, !prof !0

hot:
  ret i32 1

cold:
  ret i32 0
}

!0 = !{!"branch_weights", i32 1000, i32 1}
