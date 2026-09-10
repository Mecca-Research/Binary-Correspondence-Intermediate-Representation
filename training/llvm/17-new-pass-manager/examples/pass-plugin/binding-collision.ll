; Valid LLVM IR that the out-of-tree BCIR binding checker must reject.
;
; Nothing here is malformed: `llvm-as` accepts it and `opt -passes=verify`
; accepts it, which is the whole point. The 1:1 register-correspondence
; contract is a BCIR-level invariant that LLVM has no opinion about, so it
; needs a pass of its own. This module is what an optimizer leaves behind when
; it duplicates an instruction and copies its metadata along with it: two IR
; values now claim source register "r2".
;
; Known-good manifest entry (it must assemble and verify):
;   llvm-as   training/llvm/17-new-pass-manager/examples/pass-plugin/binding-collision.ll -o /dev/null
;   opt -passes=verify -S training/llvm/17-new-pass-manager/examples/pass-plugin/binding-collision.ll -o /dev/null
;
; With the plugin built (see ../../06-building-an-out-of-tree-pass.md):
;   opt -load-pass-plugin=<lib> -passes='bcir-verify-bindings' -disable-output \
;       training/llvm/17-new-pass-manager/examples/pass-plugin/binding-collision.ll
;   => bcir-verify-bindings: violation in function 'kernel': register 'r2' ...
;   => bcir-verify-bindings: 4 binding(s), 1 violation(s)
;
; The `<strict>` spelling turns the same report into a nonzero exit:
;   opt -load-pass-plugin=<lib> -passes='bcir-verify-bindings<strict>' ...

define i32 @kernel(i32 %a, i32 %b, i1 %c) {
entry:
  %r1 = add i32 %a, %b, !bcir.reg !0
  br i1 %c, label %hot, label %cold

hot:
  ; Speculated/duplicated copy: the metadata came along for the ride.
  %r2.hot = mul i32 %r1, 3, !bcir.reg !1
  br label %join

cold:
  %r2.cold = mul i32 %r1, 3, !bcir.reg !1
  br label %join

join:
  %r2 = phi i32 [ %r2.hot, %hot ], [ %r2.cold, %cold ]
  %r3 = sub i32 %r2, %a, !bcir.reg !2
  ret i32 %r3
}

!0 = !{!"r1"}
!1 = !{!"r2"}
!2 = !{!"r3"}
