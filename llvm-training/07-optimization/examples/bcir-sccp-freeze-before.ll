; Before repair: the branch condition depends on an nsw add that can produce
; poison when %x is INT_MAX. SCCP and SimplifyCFG may reason about reachability,
; but a BCIR pass must not create control dependence on poison.
;
; Try:
;   opt -S -passes='sccp,loop-rotate,loop-vectorize,verify' \
;     bcir-sccp-freeze-before.ll -o -

source_filename = "bcir-sccp-freeze-before.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @bcir_hot_path(i32)
declare void @bcir_cold_path(i32)

define void @bcir_sccp_freeze_before(i32 %x) {
entry:
  %candidate = add nsw i32 %x, 1, !bcir.node !0
  %take.hot = icmp sgt i32 %candidate, 0
  br i1 %take.hot, label %hot, label %cold, !bcir.node !1

hot:
  call void @bcir_hot_path(i32 %candidate), !bcir.node !2
  br label %exit

cold:
  call void @bcir_cold_path(i32 %candidate), !bcir.node !3
  br label %exit

exit:
  ret void
}

!0 = !{!"bcir.node", i32 210, !"poison-capable folded value"}
!1 = !{!"bcir.node", i32 211, !"unsafe branch on poison-capable value"}
!2 = !{!"bcir.node", i32 212, !"hot diagnostic path"}
!3 = !{!"bcir.node", i32 213, !"cold diagnostic path"}
