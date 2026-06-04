; Exercise 013: Lower mixed stride indexing to byte offsets.
; Verification: llvm-as -disable-output llvm-training/exercises/013-mixed-stride-indexing.solution.ll

source_filename = "013-mixed-stride-indexing.solution.ll"

define i32 @bcir.exercise.load_mixed_stride(ptr %base, i64 %row, i64 %col, i64 %row_stride_bytes, i64 %element_stride_bytes, i64 %bias_bytes) {
entry:
  %row_part = mul i64 %row, %row_stride_bytes
  %col_part = mul i64 %col, %element_stride_bytes
  %row_col = add i64 %row_part, %col_part
  %offset = add i64 %row_col, %bias_bytes
  %addr = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %addr, align 4
  ret i32 %value
}
