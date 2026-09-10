; Fences order surrounding ordinary and atomic memory operations.
source_filename = "fence-patterns.c"

define void @release_fence_publish(ptr %flag, ptr %data, i32 %value) {
entry:
  store i32 %value, ptr %data, align 4
  fence release
  store atomic i32 1, ptr %flag monotonic, align 4
  ret void
}

define i32 @acquire_fence_consume(ptr %flag, ptr %data) {
entry:
  %ready = load atomic i32, ptr %flag monotonic, align 4
  fence acquire
  %value = load i32, ptr %data, align 4
  %use = select i1 true, i32 %value, i32 %ready
  ret i32 %use
}
