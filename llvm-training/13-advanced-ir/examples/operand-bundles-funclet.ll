target triple = "x86_64-pc-windows-msvc"

; Operand bundle outline: funclet token on a call inside a cleanup funclet.
; Assemble/check with: llvm-as operand-bundles-funclet.ll -o /dev/null

source_filename = "operand-bundles-funclet.ll"

declare void @may_unwind()
declare i32 @__CxxFrameHandler3(...)

define void @cleanup_funclet_bundle() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_unwind()
          to label %done unwind label %cleanup

cleanup:
  %pad = cleanuppad within none []
  call void @may_unwind() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller

done:
  ret void
}
