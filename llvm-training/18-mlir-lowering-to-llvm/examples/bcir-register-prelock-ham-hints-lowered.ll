; Standalone LLVM IR lowering of a BCIR register prelock plus HAM prefetch hint.
; The logical prelock is an explicit resource-table lookup. The HAM hint is both
; an llvm.prefetch call and diagnostic metadata.

target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.prefetch.p0(ptr nocapture readonly, i32 immarg, i32 immarg, i32 immarg)
declare void @bcir.gaadmsf.transfer(ptr, i64, i64)

define void @bcir_register_prelock_ham(ptr %resource_table) !bcir.graph !0 {
entry:
  %slot = getelementptr ptr, ptr %resource_table, i64 7
  %resource = load ptr, ptr %slot, align 8, !bcir.claim !1, !bcir.register !2
  call void @llvm.prefetch.p0(ptr %resource, i32 0, i32 3, i32 1), !bcir.claim !1, !bcir.ham !3
  call void @bcir.gaadmsf.transfer(ptr %resource, i64 64, i64 1204), !bcir.claim !1, !bcir.ham !3
  ret void
}

!0 = !{!"graph_id", i64 77, !"lowering", !"bcir-register-prelock-ham"}
!1 = !{!"claim_id", i64 1204, !"source", !"bcir.register_prelock"}
!2 = !{!"logical_register", !"gpr:r7", !"binding", !"resource_table_lookup"}
!3 = !{!"ham", !"prefetch", !"locality", i32 3, !"rw", !"read"}
