; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @poisoned() {
  ret i32 poison
}

define i32 @contaminated(i32 %0) {
  ret i32 poison
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
