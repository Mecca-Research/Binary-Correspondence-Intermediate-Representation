source_filename = "031-lower-runtime-call-boundary.solution.ll"

declare i32 @bcir_runtime_commit(ptr %ctx, ptr %claim, i64 %bytes)

define i32 @lower_runtime_call_boundary(ptr %ctx, ptr %claim, i64 %bytes) {
entry:
  %status = call i32 @bcir_runtime_commit(ptr %ctx, ptr %claim, i64 %bytes)
  %ok = icmp eq i32 %status, 0
  br i1 %ok, label %success, label %failure

success:
  ret i32 0

failure:
  ret i32 %status
}
