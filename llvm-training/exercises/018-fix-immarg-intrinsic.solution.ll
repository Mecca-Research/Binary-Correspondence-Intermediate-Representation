source_filename = "018-fix-immarg-intrinsic.solution.ll"

declare void @llvm.prefetch(ptr, i32 immarg, i32 immarg, i32 immarg)

define void @prefetch_read_high_locality(ptr %p) {
entry:
  call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)
  ret void
}
