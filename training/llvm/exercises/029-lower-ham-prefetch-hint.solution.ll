source_filename = "029-lower-ham-prefetch-hint.solution.ll"

declare void @llvm.prefetch.p0(ptr nocapture readonly, i32 immarg, i32 immarg, i32 immarg)

define void @lower_ham_prefetch_hint(ptr %base, i64 %index, i64 %stride) {
entry:
  %offset = mul i64 %index, %stride
  %addr = getelementptr i8, ptr %base, i64 %offset
  call void @llvm.prefetch.p0(ptr %addr, i32 0, i32 3, i32 1), !bcir.ham.hint !1
  ret void
}

!bcir.ham.hints = !{!1}
!1 = !{!"domain", !"HBM", !"locality", i32 3, !"cache", !"data"}
