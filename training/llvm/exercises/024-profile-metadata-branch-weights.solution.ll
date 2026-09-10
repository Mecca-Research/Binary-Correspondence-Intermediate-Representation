source_filename = "024-profile-metadata-branch-weights.solution.ll"

define i32 @classify_hot_path(i32 %x) {
entry:
  %is_hot = icmp sgt i32 %x, 0
  br i1 %is_hot, label %hot, label %cold, !prof !0

hot:
  %hot_value = add nsw i32 %x, 1
  br label %merge

cold:
  %cold_value = sub nsw i32 0, %x
  br label %merge

merge:
  %result = phi i32 [ %hot_value, %hot ], [ %cold_value, %cold ]
  ret i32 %result
}

!0 = !{!"branch_weights", i32 9000, i32 100}
