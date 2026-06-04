source_filename = "bcir_lane_classifier.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

declare i8 @bcir.claim.lane(ptr)
declare i32 @bcir.claim.stride_code(ptr)
declare i64 @bcir.claim.imm(ptr, i32)

define i32 @bcir.classify.memory_lane(ptr %claim) {
; Classification rules:
; - Preserve explicit non-GGG lanes (U/UX/T/A/H).
; - Upgrade GGG (3) -> U (0) only for unit stride (stride_code==1).
; - Keep T lane (2) stable (no implicit upgrade in Phase 3 seed).
entry:
  %lane8 = call i8 @bcir.claim.lane(ptr %claim)
  %lane = zext i8 %lane8 to i32
  %stride = call i32 @bcir.claim.stride_code(ptr %claim)
  %is_g = icmp eq i32 %lane, 3
  br i1 %is_g, label %check_ggg_stride, label %explicit

check_ggg_stride:
  %unit = icmp eq i32 %stride, 1
  %ggg_lane = select i1 %unit, i32 0, i32 3
  ret i32 %ggg_lane

explicit:
  ret i32 %lane
}
