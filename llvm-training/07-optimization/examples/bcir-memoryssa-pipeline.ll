; BCIR MemorySSA pipeline fixture.
;
; Try:
;   opt -passes='print<memoryssa>' bcir-memoryssa-pipeline.ll -disable-output
;   opt -S -passes='require<memoryssa>,function(bcir-memory-audit,verify)' \
;     bcir-memoryssa-pipeline.ll -o checked.ll
;
; The second command documents New Pass Manager intent for an out-of-tree
; function pass named bcir-memory-audit. MemorySSA provides the def/use shape;
; alias analysis still decides whether %candidate can clobber %node.slot.

source_filename = "bcir-memoryssa-pipeline.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @bcir_observe(ptr)

define i32 @bcir_memoryssa_pipeline(ptr noalias %graph, ptr %candidate, i1 %take.side) {
entry:
  %node.slot = getelementptr inbounds i32, ptr %graph, i64 0
  %edge.slot = getelementptr inbounds i32, ptr %graph, i64 1
  store i32 1, ptr %node.slot, align 4, !bcir.node !0
  br i1 %take.side, label %side, label %main

side:
  store i32 2, ptr %candidate, align 4, !bcir.node !1
  call void @bcir_observe(ptr %candidate), !bcir.node !2
  br label %join

main:
  store i32 3, ptr %edge.slot, align 4, !bcir.node !3
  br label %join

join:
  %result = load i32, ptr %node.slot, align 4, !bcir.node !4
  ret i32 %result
}

!0 = !{!"bcir.node", i32 200, !"seed node slot"}
!1 = !{!"bcir.node", i32 201, !"candidate alias write"}
!2 = !{!"bcir.node", i32 202, !"unknown observer call"}
!3 = !{!"bcir.node", i32 203, !"edge slot write"}
!4 = !{!"bcir.node", i32 204, !"reload node slot"}
