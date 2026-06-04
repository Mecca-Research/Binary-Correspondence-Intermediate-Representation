; Exercise 011: Lower logical register IDs to bound storage pointers.
; Verification: llvm-as -disable-output llvm-training/exercises/011-register-binding-pattern.solution.ll

source_filename = "011-register-binding-pattern.solution.ll"

%bcir.claim.binding = type { i64, [4 x i32], [4 x i32] }

define i32 @bcir.exercise.copy_bound_register(ptr %claim, ptr %register_table) {
entry:
  %rd_array = getelementptr inbounds %bcir.claim.binding, ptr %claim, i32 0, i32 1
  %wr_array = getelementptr inbounds %bcir.claim.binding, ptr %claim, i32 0, i32 2
  %rd0_p = getelementptr inbounds [4 x i32], ptr %rd_array, i32 0, i32 0
  %wr0_p = getelementptr inbounds [4 x i32], ptr %wr_array, i32 0, i32 0
  %rd0 = load i32, ptr %rd0_p, align 4
  %wr0 = load i32, ptr %wr0_p, align 4

  %rd_idx = zext i32 %rd0 to i64
  %wr_idx = zext i32 %wr0 to i64
  %rd_slot_p = getelementptr ptr, ptr %register_table, i64 %rd_idx
  %wr_slot_p = getelementptr ptr, ptr %register_table, i64 %wr_idx
  %rd_ptr = load ptr, ptr %rd_slot_p, align 8
  %wr_ptr = load ptr, ptr %wr_slot_p, align 8

  %value = load i32, ptr %rd_ptr, align 4
  store i32 %value, ptr %wr_ptr, align 4
  ret i32 %value
}
