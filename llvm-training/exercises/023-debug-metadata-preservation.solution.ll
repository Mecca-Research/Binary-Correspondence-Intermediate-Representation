source_filename = "023-debug-metadata-preservation.solution.ll"
target triple = "x86_64-unknown-linux-gnu"

define i32 @accumulate_debug(i32 %a, i32 %b) !dbg !7 {
entry:
  %sum = add nsw i32 %a, %b, !dbg !12
  %twice = mul nsw i32 %sum, 2, !dbg !13
  ret i32 %twice, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm-training", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug-preserve.c", directory: "/llvm-training/exercises")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = !DISubroutineType(types: !6)
!6 = !{!2, !2, !2}
!7 = distinct !DISubprogram(name: "accumulate_debug", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!8 = !{}
!12 = !DILocation(line: 2, column: 11, scope: !7)
!13 = !DILocation(line: 3, column: 15, scope: !7)
!14 = !DILocation(line: 4, column: 3, scope: !7)
