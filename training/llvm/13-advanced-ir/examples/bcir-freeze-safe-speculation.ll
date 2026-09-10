; BCIR safe speculation patterns with freeze.
; This module keeps unsafe "before" shapes in comments and verifier-valid
; "after" functions as executable examples.
source_filename = "bcir-freeze-safe-speculation.bcir"
target triple = "x86_64-unknown-linux-gnu"

; BEFORE: a BCIR compare derived from poison-capable arithmetic is speculated
; into control flow. For %x == 2147483647, %sum is poison, %pred is poison, and
; branching on %pred is undefined behavior.
;
; define i32 @before_speculative_branch(i32 %x) {
; entry:
;   %sum = add nsw i32 %x, 1
;   %pred = icmp sgt i32 %sum, 0
;   br i1 %pred, label %hot, label %cold
; hot:
;   ret i32 1
; cold:
;   ret i32 0
; }

; AFTER: freeze the branch condition at the speculation boundary. Non-poison
; executions keep their behavior; poison executions choose one stable direction
; rather than feeding poison to br.
define i32 @after_speculative_branch(i32 %x) {
entry:
  %sum = add nsw i32 %x, 1
  %pred = icmp sgt i32 %sum, 0
  %pred.safe = freeze i1 %pred
  br i1 %pred.safe, label %hot, label %cold

hot:
  ret i32 1

cold:
  ret i32 0
}

; BEFORE: if-conversion to select preserves profile/debug intent, but the
; select condition is still poison-capable.
;
; define i32 @before_speculative_select(i32 %x, i32 %old) {
; entry:
;   %sum = add nsw i32 %x, 1, !dbg !7
;   %take = icmp sgt i32 %sum, 42, !dbg !8
;   %merged = select i1 %take, i32 1, i32 %old, !dbg !9
;   ret i32 %merged, !dbg !10
; }

; AFTER: preserve the metadata on surviving instructions and freeze the
; poison-capable condition before it controls a speculative select.
define i32 @after_speculative_select(i32 %x, i32 %old) !dbg !5 {
entry:
  %sum = add nsw i32 %x, 1, !dbg !7
  %take = icmp sgt i32 %sum, 42, !dbg !8
  %take.safe = freeze i1 %take, !dbg !8
  %merged = select i1 %take.safe, i32 1, i32 %old, !dbg !9
  ret i32 %merged, !dbg !10
}

; BEFORE: a vector lane-validity mask is built from poison-capable arithmetic
; and then used to control lane activity.
;
; define <4 x i32> @before_vector_mask(<4 x i32> %idx, <4 x i32> %new,
;                                      <4 x i32> %old) {
; entry:
;   %next = add nsw <4 x i32> %idx,
;                   <i32 1, i32 1, i32 1, i32 1>
;   %mask = icmp slt <4 x i32> %next,
;                  <i32 16, i32 16, i32 16, i32 16>
;   %merged = select <4 x i1> %mask, <4 x i32> %new, <4 x i32> %old
;   ret <4 x i32> %merged
; }

; AFTER: freeze the vector mask before it controls vector predication. This does
; not prove inactive lane addresses or values valid; it only stabilizes the lane
; decisions used by this select.
define <4 x i32> @after_vector_mask(<4 x i32> %idx, <4 x i32> %new,
                                    <4 x i32> %old) {
entry:
  %next = add nsw <4 x i32> %idx, <i32 1, i32 1, i32 1, i32 1>
  %mask = icmp slt <4 x i32> %next, <i32 16, i32 16, i32 16, i32 16>
  %mask.safe = freeze <4 x i1> %mask
  %merged = select <4 x i1> %mask.safe, <4 x i32> %new, <4 x i32> %old
  ret <4 x i32> %merged
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "bcir-training", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "bcir-freeze-safe-speculation.c", directory: "/training")
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "after_speculative_select", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !12)
!6 = !{!11, !11, !11}
!7 = !DILocation(line: 2, column: 11, scope: !5)
!8 = !DILocation(line: 3, column: 11, scope: !5)
!9 = !DILocation(line: 4, column: 11, scope: !5)
!10 = !DILocation(line: 5, column: 3, scope: !5)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{}
