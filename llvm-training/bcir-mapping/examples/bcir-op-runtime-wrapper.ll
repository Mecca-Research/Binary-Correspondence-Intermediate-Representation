; Lowering of bcir-operation.prompt.md.
; A BCIR operation becomes a thin runtime-call wrapper with an explicit status.

source_filename = "bcir-operation.prompt.md"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i32 @bcir.rt.add_i32(ptr nocapture, i32, i32, i32)

define i32 @bcir.map.op.add_i32_wrapper(ptr %dst, i32 %lhs, i32 %rhs) {
entry:
  %flags = or i32 0, 1
  %status = call i32 @bcir.rt.add_i32(ptr %dst, i32 %lhs, i32 %rhs, i32 %flags)
  ret i32 %status
}

define i32 @bcir.map.op.add_i32_checked(ptr %dst, ptr %lhs_ptr, ptr %rhs_ptr) {
entry:
  %lhs = load i32, ptr %lhs_ptr, align 4
  %rhs = load i32, ptr %rhs_ptr, align 4
  %status = call i32 @bcir.map.op.add_i32_wrapper(ptr %dst, i32 %lhs, i32 %rhs)
  ret i32 %status
}
