source_filename = "028-lower-vertex-edge-fragment.solution.ll"

define void @lower_vertex_edge_fragment(ptr %vertex_values, ptr %edge_dst, ptr %edge_weight, ptr %out, i64 %vertex, i64 %edge) {
entry:
  %vertex_ptr = getelementptr float, ptr %vertex_values, i64 %vertex
  %v = load float, ptr %vertex_ptr, align 4
  %dst_ptr = getelementptr i32, ptr %edge_dst, i64 %edge
  %dst32 = load i32, ptr %dst_ptr, align 4
  %dst = zext i32 %dst32 to i64
  %weight_ptr = getelementptr float, ptr %edge_weight, i64 %edge
  %w = load float, ptr %weight_ptr, align 4
  %product = fmul float %v, %w
  %out_ptr = getelementptr float, ptr %out, i64 %dst
  %old = load float, ptr %out_ptr, align 4
  %new = fadd float %old, %product
  store float %new, ptr %out_ptr, align 4
  ret void
}
