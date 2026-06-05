source_filename = "015-mlir-to-llvm-lowering-review.candidate.mlir"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

; Lowering for:
;   %src = bcir.bind.read %c[0]
;   %dst = bcir.bind.write %c[0]
;   %v = bcir.load %src[%row, %col] { row_stride_bytes = %rs, element_stride_bytes = 4 }
;   bcir.store %dst[%row, %col], %v { ham_hint = #bcir.ham_hint<HBM, 3> }
;
; %reg_table maps logical BCIR register IDs to concrete base pointers.
define void @bcir.exercise015.lower(ptr %claim, ptr %reg_table, i64 %row, i64 %col, i64 %row_stride_bytes) {
entry:
  ; Project rd[0] and wr[0] from the BCIR claim record.
  %rd_array = getelementptr inbounds %bcir.claim, ptr %claim, i32 0, i32 1
  %rd0_ptr = getelementptr inbounds [4 x i32], ptr %rd_array, i32 0, i32 0
  %src_rid = load i32, ptr %rd0_ptr, align 4
  %wr_array = getelementptr inbounds %bcir.claim, ptr %claim, i32 0, i32 2
  %wr0_ptr = getelementptr inbounds [4 x i32], ptr %wr_array, i32 0, i32 0
  %dst_rid = load i32, ptr %wr0_ptr, align 4

  ; Resolve logical BCIR register IDs through the runtime register table.
  %src_rid_zext = zext i32 %src_rid to i64
  %src_slot = getelementptr inbounds ptr, ptr %reg_table, i64 %src_rid_zext
  %src_base = load ptr, ptr %src_slot, align 8
  %dst_rid_zext = zext i32 %dst_rid to i64
  %dst_slot = getelementptr inbounds ptr, ptr %reg_table, i64 %dst_rid_zext
  %dst_base = load ptr, ptr %dst_slot, align 8

  ; Mixed-stride addressing: row_stride_bytes is dynamic, element stride is 4.
  %row_bytes = mul i64 %row, %row_stride_bytes
  %col_bytes = mul i64 %col, 4
  %byte_offset = add i64 %row_bytes, %col_bytes

  ; Final memory accesses are byte-addressed, then typed by the load/store.
  %src_addr = getelementptr i8, ptr %src_base, i64 %byte_offset
  %value = load i32, ptr %src_addr, align 4
  %dst_addr = getelementptr i8, ptr %dst_base, i64 %byte_offset
  store i32 %value, ptr %dst_addr, align 4, !bcir.ham.hint !0
  ret void
}

!bcir.ham.hints = !{!0}
!0 = !{!"domain", !"HBM", !"locality", i32 3}
