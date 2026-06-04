; Standalone example: minimal debug-info graph with instruction locations.
; Assemble: llvm-as llvm-training/06-metadata/examples/debug-location.ll -o /dev/null

source_filename = "debug-location.c"
target triple = "x86_64-unknown-linux-gnu"

define i32 @add_one(i32 %x) !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !10, metadata !DIExpression()), !dbg !12
  %y = add i32 %x, 1, !dbg !13
  ret i32 %y, !dbg !14
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "llvm-training", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug-location.c", directory: "/workspace/examples")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"llvm-training metadata example"}
!5 = distinct !DISubprogram(name: "add_one", linkageName: "add_one", scope: !1, file: !1, line: 3, type: !6, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 3, type: !8)
!12 = !DILocation(line: 3, column: 17, scope: !5)
!13 = !DILocation(line: 4, column: 12, scope: !5)
!14 = !DILocation(line: 5, column: 3, scope: !5)
