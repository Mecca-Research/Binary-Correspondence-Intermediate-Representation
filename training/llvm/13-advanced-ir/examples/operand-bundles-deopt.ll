target triple = "x86_64-unknown-linux-gnu"

; Operand bundle outline: deoptimization state on call and invoke.
; Assemble/check with: llvm-as operand-bundles-deopt.ll -o /dev/null

source_filename = "operand-bundles-deopt.ll"

declare void @callee(ptr)
declare i32 @may_throw(ptr)
declare i32 @personality(...)

define void @call_with_deopt(i32 %id, ptr %state) {
entry:
  call void @callee(ptr %state) [ "deopt"(i32 %id, ptr %state) ]
  ret void
}

define i32 @invoke_with_deopt(i32 %id, ptr %state) personality ptr @personality {
entry:
  %r = invoke i32 @may_throw(ptr %state) [ "deopt"(i32 %id, ptr %state) ]
          to label %ok unwind label %lpad

ok:
  ret i32 %r

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret i32 -1
}
