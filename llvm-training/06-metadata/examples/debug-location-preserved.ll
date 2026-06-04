; Debug locations can be preserved while instruction names change.
source_filename = "debug-location-preserved.c"

define i32 @debug_location_preserved(i32 %x) !dbg !3 {
entry:
  %inc = add i32 %x, 1, !dbg !7
  %dbl = shl i32 %inc, 1, !dbg !8
  ret i32 %dbl, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "llvm-training", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "debug-location-preserved.c", directory: "/training")
!2 = !DISubroutineType(types: !{})
!3 = distinct !DISubprogram(name: "debug_location_preserved", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !DILocation(line: 2, column: 11, scope: !3)
!8 = !DILocation(line: 3, column: 11, scope: !3)
!9 = !DILocation(line: 4, column: 3, scope: !3)
