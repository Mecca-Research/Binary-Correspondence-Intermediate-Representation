source_filename = "026-poison-freeze-repair.solution.ll"

define i32 @select_after_overflow(i32 %x) {
entry:
  %sum = add nsw i32 %x, 2147483647
  %frozen_sum = freeze i32 %sum
  %is_positive = icmp sgt i32 %frozen_sum, 0
  br i1 %is_positive, label %positive, label %nonpositive

positive:
  ret i32 1

nonpositive:
  ret i32 0
}
