; After vectorization sketch: vector body plus scalar remainder.
source_filename = "predicated-load-after-vectorize.c"

define void @predicated_load_after(ptr %dst, ptr %src) {
entry:
  %v = load <4 x i32>, ptr %src, align 4
  store <4 x i32> %v, ptr %dst, align 4
  ret void
}
