; noalias parameters let optimizers reason that two pointer arguments do not overlap.
source_filename = "aliasing-noalias.c"

define i32 @store_then_load_noalias(ptr noalias %dst, ptr noalias %src) {
entry:
  store i32 42, ptr %dst, align 4
  %v = load i32, ptr %src, align 4
  ret i32 %v
}
