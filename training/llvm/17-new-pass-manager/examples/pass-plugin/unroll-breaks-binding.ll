; A clean BCIR-annotated loop that a *stock* LLVM pass breaks.
;
; This module satisfies the 1:1 register-correspondence contract as written:
; three instructions, three distinct `!bcir.reg` bindings. Full unrolling then
; duplicates all three instructions four times, copying the metadata with each
; copy, and the contract is gone -- with no verifier error, no warning, and no
; change to any type.
;
; That is the entire argument for owning a custom pass: LLVM has no opinion
; about this invariant, so nothing in the stock pipeline can defend it.
;
; Known-good manifest entry (it must assemble and verify):
;   llvm-as   training/llvm/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll -o /dev/null
;   opt -passes=verify -S training/llvm/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll -o /dev/null
;
; With the plugin built (see ../../06-building-an-out-of-tree-pass.md):
;
;   # before: the contract holds
;   opt -load-pass-plugin=<lib> -passes='bcir-verify-bindings' -disable-output \
;       training/llvm/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll
;   => bcir-verify-bindings: 3 binding(s), 0 violation(s)
;
;   # after: 4x unrolled, and every binding is now claimed more than once
;   opt -load-pass-plugin=<lib> \
;       -passes='function(loop(loop-unroll-full)),bcir-verify-bindings' \
;       -disable-output \
;       training/llvm/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll
;   => bcir-verify-bindings: 10 binding(s), 7 violation(s)
;
; The exact counts depend on the LLVM version's unrolling decisions; the stable
; observation is "0 violations before, more than 0 after". The gate script
; training/llvm/tools/build-pass-plugin.sh asserts the stable form.

define i32 @sum(ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i32 %i, !bcir.reg !0
  %v = load i32, ptr %gep, align 4, !bcir.reg !1
  %acc.next = add i32 %acc, %v, !bcir.reg !2
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %acc.next
}

!0 = !{!"r1"}
!1 = !{!"r2"}
!2 = !{!"r3"}
