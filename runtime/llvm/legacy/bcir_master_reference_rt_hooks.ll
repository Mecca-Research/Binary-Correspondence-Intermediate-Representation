;===----------------------------------------------------------------------===;
; BCIR Master Reference Module (Pure LLVM IR Dialect)
; Canonical legal-LLVM substrate for BCIR surface/core/rop + GEM integration.
;===----------------------------------------------------------------------===;

source_filename = "bcir_master_reference.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.execctx = type {
  ptr, ; opaque runtime pointer
  i32, ; current phase
  i32, ; current epoch
  i32, ; last barrier scope
  i64, ; module id
  ptr  ; provenance sink
}

@bcir.rt.current_phase = internal global i32 -1, align 4
@bcir.rt.current_epoch = internal global i32 0, align 4
@bcir.rt.last_barrier_scope = internal global i32 0, align 4

; ---- LLVM intrinsics ----
declare void @llvm.trap() cold noreturn
declare void @llvm.assume(i1)

; ---- BCIR runtime hooks ----
declare void @bcir.rt.phase.enter(ptr, i32, i32)
declare void @bcir.rt.phase.leave(ptr, i32, i32)
declare void @bcir.rt.barrier(ptr, i32)
declare void @bcir.rt.prov.note(ptr, i64, i64, i64, i32, i32, i32, i32)

declare i32 @bcir.rt.load.i32(ptr, i64)
declare void @bcir.rt.store.i32(ptr, i64, i32)
declare <8 x i32> @bcir.rt.load.v8i32(ptr, i64)
declare void @bcir.rt.store.v8i32(ptr, i64, <8 x i32>)

declare i32 @bcir.rt.atomic.add.i32(ptr, i64, i32)
declare i32 @bcir.rt.atomic.sub.i32(ptr, i64, i32)
declare i32 @bcir.rt.atomic.xor.i32(ptr, i64, i32)

; ---- Lane helpers ----
declare <8 x i32> @bcir.rt.lane.shuffle.v8i32(<8 x i32>, <8 x i32>)
declare <8 x i32> @bcir.rt.lane.rotate.v8i32(<8 x i32>, i32)

; ---- Phase + epoch guard ----
define void @bcir.rt.assert_epoch_phase(i32 %prev_epoch, i32 %prev_phase,
                                        i32 %next_epoch, i32 %next_phase) {
entry:
  %same_epoch = icmp eq i32 %prev_epoch, %next_epoch
  br i1 %same_epoch, label %same, label %next

same:
  %phase_ok = icmp sge i32 %next_phase, %prev_phase
  br i1 %phase_ok, label %ok, label %trap

next:
  %epoch_ok = icmp sgt i32 %next_epoch, %prev_epoch
  br i1 %epoch_ok, label %ok, label %trap

trap:
  call void @llvm.trap()
  unreachable

ok:
  ret void
}

; ---- Canonical UX vector add ----
define void @bcir.example.ux_vec_add(ptr %ctx, ptr %a, ptr %b, ptr %c) {
entry:
  call void @bcir.rt.phase.enter(ptr %ctx, i32 0, i32 0)
  %va = call <8 x i32> @bcir.rt.load.v8i32(ptr %a, i64 0)
  %vb = call <8 x i32> @bcir.rt.load.v8i32(ptr %b, i64 0)

  call void @bcir.rt.phase.leave(ptr %ctx, i32 0, i32 0)
  call void @bcir.rt.phase.enter(ptr %ctx, i32 0, i32 1)
  %vc = add <8 x i32> %va, %vb

  call void @bcir.rt.phase.leave(ptr %ctx, i32 0, i32 1)
  call void @bcir.rt.phase.enter(ptr %ctx, i32 0, i32 2)
  call void @bcir.rt.store.v8i32(ptr %c, i64 0, <8 x i32> %vc)
  call void @bcir.rt.prov.note(ptr %ctx, i64 1001, i64 0, i64 0, i32 3, i32 1, i32 2, i32 0)
  call void @bcir.rt.barrier(ptr %ctx, i32 2)
  call void @bcir.rt.phase.leave(ptr %ctx, i32 0, i32 2)
  ret void
}

; ---- Canonical atomic path (A lane) ----
define i32 @bcir.example.atomic_add(ptr %ctx, ptr %base, i32 %delta) {
entry:
  call void @bcir.rt.phase.enter(ptr %ctx, i32 1, i32 0)
  %old = atomicrmw add ptr %base, i32 %delta seq_cst, align 4
  call void @bcir.rt.prov.note(ptr %ctx, i64 2001, i64 0, i64 0, i32 7, i32 4, i32 0, i32 1)
  call void @bcir.rt.barrier(ptr %ctx, i32 4)
  call void @bcir.rt.phase.leave(ptr %ctx, i32 1, i32 0)
  ret i32 %old
}

; ---- Canonical CAS ----
define { i32, i1 } @bcir.example.cas(ptr %p, i32 %expected, i32 %desired) {
entry:
  %pair = cmpxchg ptr %p, i32 %expected, i32 %desired seq_cst monotonic, align 4
  ret { i32, i1 } %pair
}

!bcir.module = !{!0}
!0 = !{!"bcir.master.reference", i32 1, !"legal_llvm_only", i1 true}
!bcir.pipeline = !{!1}
!1 = !{!"bcir.surface", !"bcir.core", !"bcir.rop", !"mlir.llvm", !"llvm.ir"}
!bcir.lanes = !{!2}
!2 = !{!"U", !"UX", !"T", !"GGG", !"A", !"H"}
