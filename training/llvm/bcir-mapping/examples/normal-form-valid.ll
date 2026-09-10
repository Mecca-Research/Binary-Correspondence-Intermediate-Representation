; A verifier-clean BCIR normal-form example. The custom metadata is advisory,
; while claim IDs, byte strides, address spaces, and the runtime boundary are
; represented by ordinary LLVM IR values and types.

target datalayout = "e-p:64:64-p1:64:64-i64:64-n8:16:32:64-S128"

declare void @bcir_runtime_store_i32(ptr addrspace(1), i32, i64)

define i32 @normal_form_update(ptr addrspace(1) %base, i64 %row, i64 %column, i32 %input) !bcir.stage !0 {
entry:
  %reg.r7 = add i32 %input, 0, !bcir.reg !1, !bcir.claim !2, !bcir.diag !3
  %row.bytes = mul i64 %row, 64
  %column.bytes = mul i64 %column, 4
  %byte.offset = add i64 %row.bytes, %column.bytes
  %element = getelementptr i8, ptr addrspace(1) %base, i64 %byte.offset, !bcir.stride !4, !bcir.claim !2, !bcir.diag !3
  %defined.r7 = freeze i32 %reg.r7, !bcir.reg !1, !bcir.claim !2
  call void @bcir_runtime_store_i32(ptr addrspace(1) %element, i32 %defined.r7, i64 42017), !bcir.runtime_boundary !5, !bcir.claim !2, !bcir.diag !3
  ret i32 %defined.r7
}

!bcir.normal_form = !{!6}

!0 = !{!"register-bound"}
!1 = !{!"r7", i32 1}
!2 = !{!"claim", i64 42017}
!3 = !{!"normal-form-valid.ll", i32 1, !"update graph cell"}
!4 = !{!"row_stride_bytes", i64 64, !"column_stride_bytes", i64 4}
!5 = !{!"bcir_runtime_store_i32", !"required"}
!6 = !{!"bcir-normal-form", i32 1}
