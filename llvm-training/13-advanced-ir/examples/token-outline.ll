target triple = "x86_64-unknown-linux-gnu"

; Token and metadata outline.
; Assemble/check with: llvm-as token-outline.ll -o /dev/null

source_filename = "token-outline.ll"

declare void @callee()
declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)
declare void @llvm.dbg.value(metadata, metadata, metadata)

define void @token_statepoint_outline(i32 %x) gc "statepoint-example" !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !7, metadata !DIExpression()), !dbg !10
  %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(
      i64 0, i32 0, ptr elementtype(void ()) @callee, i32 0, i32 0, i32 0, i32 0), !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm-training", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "token-outline.ll", directory: ".")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = distinct !DISubprogram(name: "token_statepoint_outline", scope: !1, file: !1, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 8, type: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 10, column: 3, scope: !4)
