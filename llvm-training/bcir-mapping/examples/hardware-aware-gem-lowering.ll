; BCIR hardware-aware GEM lowering sketch.
; Demonstrates custom intrinsic declaration, mixed-stride address formation,
; register-oriented vector operands, and hierarchical memory hint metadata.

source_filename = "hardware-aware-gem-lowering.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

; Custom backend intrinsic shape. A real BCIR-aware backend would register this
; name in TableGen/Intrinsic::ID handling; portable fallback pipelines can lower
; the same BCIR operation to ordinary fmul/fadd or a runtime call instead.
declare <4 x float> @llvm.bcir.gem.mixed.stride.v4f32(<4 x float>, <4 x float>, <4 x float>, i32 immarg)

define <4 x float> @bcir.example.gem_tile_lower(ptr %a_base, ptr %b_base, ptr %c_base, i64 %row, i64 %col, i64 %k, i64 %a_row_stride_bytes, i64 %a_k_stride_bytes, i64 %b_k_stride_bytes, i64 %b_col_stride_bytes, i64 %c_row_stride_bytes, i64 %c_col_stride_bytes) {
entry:
  ; Mixed-stride A[row, k]
  %a_row_offset = mul i64 %row, %a_row_stride_bytes
  %a_k_offset = mul i64 %k, %a_k_stride_bytes
  %a_offset = add i64 %a_row_offset, %a_k_offset
  %a_addr = getelementptr i8, ptr %a_base, i64 %a_offset, !bcir.memory !0

  ; Mixed-stride B[k, col]
  %b_k_offset = mul i64 %k, %b_k_stride_bytes
  %b_col_offset = mul i64 %col, %b_col_stride_bytes
  %b_offset = add i64 %b_k_offset, %b_col_offset
  %b_addr = getelementptr i8, ptr %b_base, i64 %b_offset, !bcir.memory !1

  ; Mixed-stride C[row, col]
  %c_row_offset = mul i64 %row, %c_row_stride_bytes
  %c_col_offset = mul i64 %col, %c_col_stride_bytes
  %c_offset = add i64 %c_row_offset, %c_col_offset
  %c_addr = getelementptr i8, ptr %c_base, i64 %c_offset, !bcir.memory !2

  ; Register-oriented operand layout: each logical tile fragment is already in
  ; one vector SSA value before the target intrinsic is selected.
  %a_vec = load <4 x float>, ptr %a_addr, align 16, !bcir.memory !0
  %b_vec = load <4 x float>, ptr %b_addr, align 16, !bcir.memory !1
  %acc_vec = load <4 x float>, ptr %c_addr, align 16, !bcir.memory !2
  %result_vec = call <4 x float> @llvm.bcir.gem.mixed.stride.v4f32(<4 x float> %a_vec, <4 x float> %b_vec, <4 x float> %acc_vec, i32 0), !bcir.memory !3
  store <4 x float> %result_vec, ptr %c_addr, align 16, !bcir.memory !2
  ret <4 x float> %result_vec
}

!0 = !{!"bcir.memory", !"level:l1", !"role:gem.a", !4, !5}
!1 = !{!"bcir.memory", !"level:l1", !"role:gem.b", !4, !6}
!2 = !{!"bcir.memory", !"level:l2", !"role:gem.c", !7, !8}
!3 = !{!"bcir.memory", !"operation:gem.mixed_stride", !"tile:m4n4k4", !0, !1, !2}
!4 = !{!"reuse", !"scope:k-tile", i32 3}
!5 = !{!"stride", !"row-major-a", !"row_stride:param", !"k_stride:param"}
!6 = !{!"stride", !"column-panel-b", !"k_stride:param", !"col_stride:param"}
!7 = !{!"reuse", !"scope:output-tile", i32 2}
!8 = !{!"writeback", !"after:k-reduction"}
