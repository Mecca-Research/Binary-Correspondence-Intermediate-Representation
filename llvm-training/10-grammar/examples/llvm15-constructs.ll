; LLVM 15 syntax the grammar snapshot gained when it moved from llir/grammar 5a3820b to
; 05deced (six upstream commits): allockind/allocptr/allocalign attributes, global-variable
; sanitizer kinds, presplitcoroutine, fn_ret_thunk_extern, nosanitize_bounds, atomicrmw
; fmin/fmax, and DISubprogram targetFuncName. Assembles under llvm-as 18.1.3 and 23.1.0.
; The same delta REMOVED the fadd/fsub/fmul/udiv/sdiv/fdiv/urem/srem/frem constant
; expressions: `@x = global float fadd (float 1.0, float 2.0)` is refused by llvm-as 23
; ("fadd constexprs are no longer supported"), matching the grammar.
@g_nosan = global i32 0, no_sanitize_address
@g_dyninit = global i32 0, sanitize_address_dyninit
@g_nohw = global i32 0, no_sanitize_hwaddress

declare ptr @my_alloc(i64 allocalign, ptr allocptr) allockind("alloc,uninitialized") allocsize(0)
declare void @my_free(ptr allocptr) allockind("free")

define i32 @coro() presplitcoroutine {
  ret i32 0
}

define i32 @thunk() fn_ret_thunk_extern {
  ret i32 0
}

define void @bounds(ptr %p) nosanitize_bounds {
  store i32 1, ptr %p
  ret void
}

define float @atomic_fminmax(ptr %p, float %v) {
  %a = atomicrmw fmin ptr %p, float %v seq_cst
  %b = atomicrmw fmax ptr %p, float %a monotonic
  ret float %b
}

define void @dbg() !dbg !4 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "grammar witness", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "witness.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "dbg", linkageName: "dbg", scope: !1, file: !1, line: 1, type: !5, spFlags: DISPFlagDefinition, unit: !0, targetFuncName: "dbg_target")
!5 = !DISubroutineType(types: !6)
!6 = !{null}
