; Run:
;   opt -S -passes='instcombine,simplifycfg,adce' dead-code-before.ll -o -
;
; Several computations below are unused or algebraically trivial.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "dead-code-before.ll"

define i32 @dead_code(i32 %x) {
entry:
  %unused_mul = mul i32 %x, 42
  %plus_zero = add i32 %x, 0
  %always = icmp eq i32 1, 1
  br i1 %always, label %then, label %else

then:
  %unused_then = add i32 %unused_mul, 7
  ret i32 %plus_zero

else:
  ret i32 0
}
