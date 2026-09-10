; After repair: freeze stabilizes the poison-capable value before it controls
; reachability. SCCP can still simplify proven constants, but the BCIR lowering
; boundary no longer branches directly on poison.
;
; Try:
;   opt -S -passes='sccp,loop-rotate,loop-vectorize,verify' \
;     bcir-sccp-freeze-after.ll -o -

source_filename = "bcir-sccp-freeze-after.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @bcir_hot_path(i32)
declare void @bcir_cold_path(i32)

define void @bcir_sccp_freeze_after(i32 %x) {
entry:
  %candidate = add nsw i32 %x, 1, !bcir.node !0
  %stable = freeze i32 %candidate, !bcir.node !1
  %take.hot = icmp sgt i32 %stable, 0
  br i1 %take.hot, label %hot, label %cold, !bcir.node !2

hot:
  call void @bcir_hot_path(i32 %stable), !bcir.node !3
  br label %exit

cold:
  call void @bcir_cold_path(i32 %stable), !bcir.node !4
  br label %exit

exit:
  ret void
}

!0 = !{!"bcir.node", i32 220, !"poison-capable folded value"}
!1 = !{!"bcir.node", i32 221, !"freeze before reachability"}
!2 = !{!"bcir.node", i32 222, !"safe branch on frozen value"}
!3 = !{!"bcir.node", i32 223, !"hot diagnostic path"}
!4 = !{!"bcir.node", i32 224, !"cold diagnostic path"}
