; Example: variable fragments describe a source variable split across IR values.
; Assemble: llvm-as llvm-training/06-metadata/examples/debug-variable-fragments.ll -o /dev/null
;
; Imagine a C-like source variable:
;   struct Pair { int lo; int hi; } p;
; After scalar replacement, p.lo and p.hi may be carried by separate i32 values.

source_filename = "debug-variable-fragments.c"
target triple = "x86_64-unknown-linux-gnu"

%struct.Pair = type { i32, i32 }

define i32 @sum_pair_fragments(i32 %lo, i32 %hi) !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata i32 %lo, metadata !10, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !14
  call void @llvm.dbg.value(metadata i32 %hi, metadata !10, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !15
  %sum = add i32 %lo, %hi, !dbg !16
  ret i32 %sum, !dbg !17
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "llvm-training", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug-variable-fragments.c", directory: "/workspace/examples")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"llvm-training metadata example"}
!5 = distinct !DISubprogram(name: "sum_pair_fragments", linkageName: "sum_pair_fragments", scope: !1, file: !1, line: 7, type: !6, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "p", scope: !5, file: !1, line: 8, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair", file: !1, line: 3, size: 64, elements: !12)
!12 = !{!13, !18}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "lo", scope: !11, file: !1, line: 4, baseType: !8, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "hi", scope: !11, file: !1, line: 5, baseType: !8, size: 32, offset: 32)
!14 = !DILocation(line: 8, column: 15, scope: !5)
!15 = !DILocation(line: 8, column: 19, scope: !5)
!16 = !DILocation(line: 9, column: 15, scope: !5)
!17 = !DILocation(line: 10, column: 3, scope: !5)
