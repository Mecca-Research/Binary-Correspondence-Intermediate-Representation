; Fixed and scalable vectors use different type spellings.
source_filename = "vector-types.c"

define <4 x i32> @fixed_vector_add(<4 x i32> %a, <4 x i32> %b) {
entry:
  %sum = add <4 x i32> %a, %b
  ret <4 x i32> %sum
}

define <vscale x 4 x i32> @scalable_vector_add(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
entry:
  %sum = add <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %sum
}
