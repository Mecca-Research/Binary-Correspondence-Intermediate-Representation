target triple = "x86_64-unknown-linux-gnu"

; Overflow-checked arithmetic intrinsics.
; Assemble/check with: llvm-as overflow-intrinsic.ll -o /dev/null

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)

define i64 @checked_unsigned_add_or_zero(i64 %a, i64 %b) {
entry:
  %pair = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %sum = extractvalue { i64, i1 } %pair, 0
  %overflow = extractvalue { i64, i1 } %pair, 1
  br i1 %overflow, label %bad, label %ok

ok:
  ret i64 %sum

bad:
  ret i64 0
}

define i1 @signed_add_overflowed(i64 %a, i64 %b) {
entry:
  %pair = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b)
  %overflow = extractvalue { i64, i1 } %pair, 1
  ret i1 %overflow
}
