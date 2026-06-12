; Dragon Egg flow-execution boundary with explicit state, buffers, and status.

source_filename = "dragon-egg-flow-execution.ll"
target triple = "unknown-unknown-unknown"

declare i32 @llvm.bcir.dragon.egg.flow.execute(i64, ptr, ptr, i64, i32 immarg)
declare i32 @bcir.runtime.dragon.egg.flow.execute(i64, ptr, ptr, i64, i32)

define i32 @execute_selected_flow(i1 %has_target_extension, i64 %context,
                                  ptr %graph, ptr %payload, i64 %count) {
entry:
  br i1 %has_target_extension, label %target, label %fallback

target:
  %target_status = call i32 @llvm.bcir.dragon.egg.flow.execute(
      i64 %context, ptr %graph, ptr %payload, i64 %count, i32 0),
      !bcir.flow !0, !bcir.memory !1
  br label %done

fallback:
  %runtime_status = call i32 @bcir.runtime.dragon.egg.flow.execute(
      i64 %context, ptr %graph, ptr %payload, i64 %count, i32 0)
  br label %done

done:
  %status = phi i32 [ %target_status, %target ], [ %runtime_status, %fallback ]
  ret i32 %status
}

!0 = !{!"bcir.flow.v1", !"policy:bounded", !"queue:preferred-local"}
!1 = !{!"bcir.memory.v1", !"graph:read", !"payload:read-write", !"hierarchy:l1>l2>hbm"}
