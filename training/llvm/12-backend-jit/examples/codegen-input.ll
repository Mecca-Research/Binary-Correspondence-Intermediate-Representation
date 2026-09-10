; codegen-input.ll — small module for llvm-as, llc, and JIT/codegen demos.
; Assemble/check with: llvm-as codegen-input.ll -o /dev/null
; Emit assembly with: llc codegen-input.ll -o -

source_filename = "codegen-input.ll"
target triple = "x86_64-unknown-linux-gnu"

@seed = global i64 3, align 8

define i32 @add_and_scale(i32 %a, i32 %b) {
entry:
  %sum = add nsw i32 %a, %b
  %scaled = shl nsw i32 %sum, 1
  ret i32 %scaled
}

define i64 @load_seed_plus(i64 %x) {
entry:
  %seed_value = load i64, ptr @seed, align 8
  %result = add i64 %seed_value, %x
  ret i64 %result
}

define i64 @sum_to_n(i64 %n) {
entry:
  %has_work = icmp sgt i64 %n, 0
  br i1 %has_work, label %loop, label %done

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %sum, %loop ]
  %sum = add i64 %acc, %i
  %next = add i64 %i, 1
  %keep_going = icmp slt i64 %next, %n
  br i1 %keep_going, label %loop, label %done

done:
  %result = phi i64 [ 0, %entry ], [ %sum, %loop ]
  ret i64 %result
}
