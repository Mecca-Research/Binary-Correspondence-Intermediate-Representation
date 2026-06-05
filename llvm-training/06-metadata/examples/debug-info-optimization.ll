; Example: preserving, dropping, and salvaging debug info during optimization.
; Assemble: llvm-as llvm-training/06-metadata/examples/debug-info-optimization.ll -o /dev/null

source_filename = "debug-info-optimization.c"
target triple = "x86_64-unknown-linux-gnu"

define i32 @optimized_debug_values(i32 %x, ptr %slot) !dbg !5 {
entry:
  ; Before promotion, an address location may be described with dbg.declare.
  call void @llvm.dbg.declare(metadata ptr %slot, metadata !10, metadata !DIExpression()), !dbg !14

  ; After promotion, prefer value locations for the current SSA value.
  call void @llvm.dbg.value(metadata i32 %x, metadata !10, metadata !DIExpression()), !dbg !15

  ; Salvage "y = x + 4" even if the add instruction itself was folded away.
  call void @llvm.dbg.value(metadata i32 %x, metadata !11, metadata !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)), !dbg !16

  ; This synthesized mask has no source-level operation, so it intentionally
  ; has no !dbg attachment instead of copying a stale user line.
  %masked = and i32 %x, 255

  %result = add i32 %masked, 4, !dbg !17
  ret i32 %result, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "llvm-training", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug-info-optimization.c", directory: "/workspace/examples")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"llvm-training metadata example"}
!5 = distinct !DISubprogram(name: "optimized_debug_values", linkageName: "optimized_debug_values", scope: !1, file: !1, line: 3, type: !6, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !19}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11}
!10 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 3, type: !8)
!11 = !DILocalVariable(name: "y", scope: !5, file: !1, line: 4, type: !8)
!14 = !DILocation(line: 3, column: 36, scope: !5)
!15 = !DILocation(line: 3, column: 34, scope: !5)
!16 = !DILocation(line: 4, column: 11, scope: !5)
!17 = !DILocation(line: 7, column: 16, scope: !5)
!18 = !DILocation(line: 8, column: 3, scope: !5)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
