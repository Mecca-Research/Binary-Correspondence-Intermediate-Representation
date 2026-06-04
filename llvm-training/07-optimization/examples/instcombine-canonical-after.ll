; Teaching snapshot for:
;   opt -S -passes=instcombine instcombine-canonical-before.ll -o instcombine-canonical-after.ll

target triple = "x86_64-unknown-linux-gnu"
source_filename = "instcombine-canonical-before.ll"

define i1 @canonical_predicate(i32 %x) {
entry:
  ret i1 true
}

define i32 @canonical_arithmetic(i32 %x) {
entry:
  %combined = shl i32 %x, 3
  ret i32 %combined
}
