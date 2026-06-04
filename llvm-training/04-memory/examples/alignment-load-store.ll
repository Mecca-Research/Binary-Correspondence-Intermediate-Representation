; Alignment belongs to memory operations, not just pointer-typed values.
source_filename = "alignment-load-store.c"

define i64 @aligned_copy(ptr %dst, ptr %src) {
entry:
  %v = load i64, ptr %src, align 8
  store i64 %v, ptr %dst, align 8
  ret i64 %v
}
