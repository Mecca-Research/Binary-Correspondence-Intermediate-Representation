; Lowering of mixed-stride-graph.bcir.txt.
; Mixed row/column strides lower to byte offsets before the final typed load.

source_filename = "mixed-stride-graph.bcir.txt"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define i64 @bcir.map.mixed_stride.byte_offset(i64 %row, i64 %column, i64 %row_stride_bytes, i64 %column_stride_bytes, i64 %payload_bias_bytes) {
entry:
  %row_bytes = mul i64 %row, %row_stride_bytes
  %column_bytes = mul i64 %column, %column_stride_bytes
  %cell_bytes = add i64 %row_bytes, %column_bytes
  %payload_bytes = add i64 %cell_bytes, %payload_bias_bytes
  ret i64 %payload_bytes
}

define ptr @bcir.map.mixed_stride.payload_address(ptr %base, i64 %row, i64 %column, i64 %row_stride_bytes, i64 %column_stride_bytes, i64 %payload_bias_bytes) {
entry:
  %offset = call i64 @bcir.map.mixed_stride.byte_offset(i64 %row, i64 %column, i64 %row_stride_bytes, i64 %column_stride_bytes, i64 %payload_bias_bytes)
  %addr = getelementptr i8, ptr %base, i64 %offset
  ret ptr %addr
}

define i32 @bcir.map.mixed_stride.load_payload_i32(ptr %base, i64 %row, i64 %column, i64 %row_stride_bytes, i64 %column_stride_bytes, i64 %payload_bias_bytes) {
entry:
  %addr = call ptr @bcir.map.mixed_stride.payload_address(ptr %base, i64 %row, i64 %column, i64 %row_stride_bytes, i64 %column_stride_bytes, i64 %payload_bias_bytes)
  %value = load i32, ptr %addr, align 4
  ret i32 %value
}
