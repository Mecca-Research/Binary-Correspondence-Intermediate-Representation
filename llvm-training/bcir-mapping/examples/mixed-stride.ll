; BCIR mixed-stride graph indexing sketch.
; Standalone example: row stride, element stride, and byte bias form an address.

source_filename = "mixed-stride.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define i64 @bcir.example.mixed_stride_offset(i64 %row, i64 %col, i64 %row_stride_bytes, i64 %element_stride_bytes, i64 %bias_bytes) {
entry:
  %row_offset = mul i64 %row, %row_stride_bytes
  %col_offset = mul i64 %col, %element_stride_bytes
  %base_offset = add i64 %row_offset, %col_offset
  %biased_offset = add i64 %base_offset, %bias_bytes
  ret i64 %biased_offset
}

define i32 @bcir.example.load_mixed_stride(ptr %base, i64 %row, i64 %col, i64 %row_stride_bytes, i64 %element_stride_bytes, i64 %bias_bytes) {
entry:
  %offset = call i64 @bcir.example.mixed_stride_offset(i64 %row, i64 %col, i64 %row_stride_bytes, i64 %element_stride_bytes, i64 %bias_bytes)
  %addr = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %addr, align 4
  ret i32 %value
}

define i32 @bcir.example.load_indexed_edge(ptr %base, ptr %indices, i64 %edge_index, i64 %element_stride_bytes) {
entry:
  %index_ptr = getelementptr inbounds i32, ptr %indices, i64 %edge_index
  %logical_index32 = load i32, ptr %index_ptr, align 4
  %logical_index64 = zext i32 %logical_index32 to i64
  %offset = mul i64 %logical_index64, %element_stride_bytes
  %addr = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %addr, align 4
  ret i32 %value
}
