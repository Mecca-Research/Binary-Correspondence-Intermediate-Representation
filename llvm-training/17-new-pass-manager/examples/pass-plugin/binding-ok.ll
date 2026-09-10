; Known-good input for the out-of-tree BCIR binding checker: every `!bcir.reg`
; names a distinct source register, so 1:1 correspondence holds.
;
; This is valid LLVM IR and is part of the known-good manifest:
;   llvm-as   llvm-training/17-new-pass-manager/examples/pass-plugin/binding-ok.ll -o /dev/null
;   opt -passes=verify -S llvm-training/17-new-pass-manager/examples/pass-plugin/binding-ok.ll -o /dev/null
;
; With the plugin built (see ../../06-building-an-out-of-tree-pass.md):
;   opt -load-pass-plugin=<lib> -passes='bcir-verify-bindings' -disable-output \
;       llvm-training/17-new-pass-manager/examples/pass-plugin/binding-ok.ll
;   => bcir-verify-bindings: 4 binding(s), 0 violation(s)

define i32 @kernel(i32 %a, i32 %b) {
entry:
  %r1 = add i32 %a, %b, !bcir.reg !0
  %r2 = mul i32 %r1, 3, !bcir.reg !1
  %r3 = sub i32 %r2, %a, !bcir.reg !2
  ret i32 %r3
}

define i32 @other(i32 %x) {
entry:
  ; Register names are scoped per function: reusing "r1" here is not a
  ; collision, because correspondence is a per-function contract.
  %r1 = shl i32 %x, 2, !bcir.reg !0
  ret i32 %r1
}

!0 = !{!"r1"}
!1 = !{!"r2"}
!2 = !{!"r3"}
