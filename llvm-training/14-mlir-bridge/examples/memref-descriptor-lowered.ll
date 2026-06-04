; Simplified LLVM-level shape often produced from a ranked memref descriptor.
source_filename = "memref-descriptor-lowered.mlir"

%MemRef1D = type { ptr, ptr, i64, [1 x i64], [1 x i64] }

define i64 @memref_length(ptr %desc) {
entry:
  %sizes = getelementptr inbounds %MemRef1D, ptr %desc, i32 0, i32 3
  %size0p = getelementptr inbounds [1 x i64], ptr %sizes, i32 0, i32 0
  %n = load i64, ptr %size0p, align 8
  ret i64 %n
}
