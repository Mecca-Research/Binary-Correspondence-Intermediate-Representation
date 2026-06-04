source_filename = "bcir_prefetch_profiles.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

declare void @llvm.prefetch(ptr, i32, i32, i32)

define void @bcir.op.prefetch.linear(ptr %base, i64 %offset, i32 %locality, i32 %rw) {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)
  ret void
}

define void @bcir.op.prefetch.strided(ptr %base, i64 %offset, i64 %stride, i32 %distance, i32 %locality) {
entry:
  %d64 = zext i32 %distance to i64
  %ahead = mul i64 %d64, %stride
  %off2 = add i64 %offset, %ahead
  %p = getelementptr i8, ptr %base, i64 %off2
  call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)
  ret void
}
