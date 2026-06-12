define i32 @bcir.exercise.load_mixed_stride(ptr %base, i64 %row, i64 %col, i64 %row_stride_bytes, i64 %element_stride_bytes, i64 %bias_bytes) {
entry:
  %value = load i32, ptr %base, align 4
  ret i32 %value
}
