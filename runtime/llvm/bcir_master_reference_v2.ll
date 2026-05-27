source_filename = "bcir_master_reference_v2.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.execctx = type { ptr, i32, i32, i32, i64, ptr }
%bcir.batch = type { i32, i32, i32, i32, ptr, i64 }

declare void @llvm.trap() cold noreturn

define i64 @bcir.claim.opstride(ptr %c) alwaysinline {
entry:
  %p = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 0
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define i8 @bcir.claim.opcode(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.opstride(ptr %c)
  %x = trunc i64 %h to i8
  ret i8 %x
}

define i8 @bcir.claim.lane(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.opstride(ptr %c)
  %s = lshr i64 %h, 8
  %x = trunc i64 %s to i8
  ret i8 %x
}

define i32 @bcir.claim.phase(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.opstride(ptr %c)
  %s = lshr i64 %h, 16
  %m = and i64 %s, 4095
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.epoch(ptr %c) alwaysinline {
entry:
  %h = call i64 @bcir.claim.opstride(ptr %c)
  %s = lshr i64 %h, 28
  %m = and i64 %s, 4095
  %x = trunc i64 %m to i32
  ret i32 %x
}

define i32 @bcir.claim.rd(ptr %c, i32 %idx) alwaysinline {
entry:
  %arr = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 1
  %p = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 %idx
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @bcir.claim.wr(ptr %c, i32 %idx) alwaysinline {
entry:
  %arr = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 2
  %p = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 %idx
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i64 @bcir.claim.imm(ptr %c, i32 %idx) alwaysinline {
entry:
  %arr = getelementptr inbounds %bcir.claim, ptr %c, i32 0, i32 4
  %p = getelementptr inbounds [2 x i64], ptr %arr, i32 0, i32 %idx
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define void @bcir.op.phase.enter(ptr %ctx, i32 %epoch, i32 %phase) {
entry:
  %phase_p = getelementptr inbounds %bcir.execctx, ptr %ctx, i32 0, i32 1
  %epoch_p = getelementptr inbounds %bcir.execctx, ptr %ctx, i32 0, i32 2
  store i32 %phase, ptr %phase_p, align 4
  store i32 %epoch, ptr %epoch_p, align 4
  ret void
}

define void @bcir.op.phase.leave(ptr %ctx, i32 %epoch, i32 %phase) {
entry:
  ret void
}

define void @bcir.op.barrier(ptr %ctx, i32 %scope) {
entry:
  %scope_p = getelementptr inbounds %bcir.execctx, ptr %ctx, i32 0, i32 3
  store i32 %scope, ptr %scope_p, align 4
  fence seq_cst
  ret void
}

define i32 @bcir.op.load.i32(ptr %base, i64 %offset) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define void @bcir.op.store.i32(ptr %base, i64 %offset, i32 %value) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  store i32 %value, ptr %p, align 4
  ret void
}

define <8 x i32> @bcir.op.load.v8i32(ptr %base, i64 %offset) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %v = load <8 x i32>, ptr %p, align 32
  ret <8 x i32> %v
}

define void @bcir.op.store.v8i32(ptr %base, i64 %offset, <8 x i32> %value) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  store <8 x i32> %value, ptr %p, align 32
  ret void
}

define i32 @bcir.op.atomic.add.i32(ptr %base, i64 %offset, i32 %delta) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %old = atomicrmw add ptr %p, i32 %delta seq_cst, align 4
  ret i32 %old
}

define i32 @bcir.op.atomic.sub.i32(ptr %base, i64 %offset, i32 %delta) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %old = atomicrmw sub ptr %p, i32 %delta seq_cst, align 4
  ret i32 %old
}

define i32 @bcir.op.atomic.xor.i32(ptr %base, i64 %offset, i32 %delta) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %old = atomicrmw xor ptr %p, i32 %delta seq_cst, align 4
  ret i32 %old
}

define { i32, i1 } @bcir.op.cmpxchg.i32(ptr %base, i64 %offset, i32 %expected, i32 %desired) alwaysinline {
entry:
  %p = getelementptr i8, ptr %base, i64 %offset
  %pair = cmpxchg ptr %p, i32 %expected, i32 %desired seq_cst monotonic, align 4
  ret { i32, i1 } %pair
}

define ptr @bcir.registry.lookup(ptr %registry_table, i32 %rid) alwaysinline {
entry:
  %idx = and i32 %rid, 268435455
  %p = getelementptr ptr, ptr %registry_table, i32 %idx
  %base = load ptr, ptr %p, align 8
  ret ptr %base
}

define void @bcir.gem.execute_claim(ptr %ctx, ptr %claim, ptr %registry_table) {
entry:
  %op8 = call i8 @bcir.claim.opcode(ptr %claim)
  %op = zext i8 %op8 to i32
  switch i32 %op, label %unknown [
    i32 0, label %done
    i32 1, label %load
    i32 2, label %store
    i32 32, label %atomic_add
    i32 35, label %cas
    i32 48, label %barrier
    i32 49, label %phase_enter
    i32 50, label %phase_leave
  ]

phase_enter:
  %pe = call i32 @bcir.claim.epoch(ptr %claim)
  %pp = call i32 @bcir.claim.phase(ptr %claim)
  call void @bcir.op.phase.enter(ptr %ctx, i32 %pe, i32 %pp)
  br label %done

phase_leave:
  %le = call i32 @bcir.claim.epoch(ptr %claim)
  %lp = call i32 @bcir.claim.phase(ptr %claim)
  call void @bcir.op.phase.leave(ptr %ctx, i32 %le, i32 %lp)
  br label %done

load:
  %lrid = call i32 @bcir.claim.rd(ptr %claim, i32 0)
  %lbase = call ptr @bcir.registry.lookup(ptr %registry_table, i32 %lrid)
  %loff = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  %lane8 = call i8 @bcir.claim.lane(ptr %claim)
  %lane = zext i8 %lane8 to i32
  %is_ux = icmp eq i32 %lane, 1
  br i1 %is_ux, label %load_ux, label %load_u

load_u:
  %lv = call i32 @bcir.op.load.i32(ptr %lbase, i64 %loff)
  br label %done

load_ux:
  %lvx = call <8 x i32> @bcir.op.load.v8i32(ptr %lbase, i64 %loff)
  br label %done

store:
  %srid = call i32 @bcir.claim.wr(ptr %claim, i32 0)
  %sbase = call ptr @bcir.registry.lookup(ptr %registry_table, i32 %srid)
  %soff = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  call void @bcir.op.store.i32(ptr %sbase, i64 %soff, i32 0)
  br label %done

atomic_add:
  %arid = call i32 @bcir.claim.wr(ptr %claim, i32 0)
  %abase = call ptr @bcir.registry.lookup(ptr %registry_table, i32 %arid)
  %aoff = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  %delta64 = call i64 @bcir.claim.imm(ptr %claim, i32 1)
  %delta = trunc i64 %delta64 to i32
  %old = call i32 @bcir.op.atomic.add.i32(ptr %abase, i64 %aoff, i32 %delta)
  br label %done

cas:
  %crid = call i32 @bcir.claim.wr(ptr %claim, i32 0)
  %cbase = call ptr @bcir.registry.lookup(ptr %registry_table, i32 %crid)
  %coff = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  %expected64 = call i64 @bcir.claim.imm(ptr %claim, i32 1)
  %expected = trunc i64 %expected64 to i32
  %pair = call { i32, i1 } @bcir.op.cmpxchg.i32(ptr %cbase, i64 %coff, i32 %expected, i32 1)
  br label %done

barrier:
  call void @bcir.op.barrier(ptr %ctx, i32 2)
  br label %done

unknown:
  call void @llvm.trap()
  unreachable

done:
  ret void
}

define void @bcir.gem.execute_worklist(ptr %ctx, ptr %claims, i64 %count, ptr %registry_table) {
entry:
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %is_done = icmp uge i64 %i, %count
  br i1 %is_done, label %exit, label %body

body:
  %claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %i
  call void @bcir.gem.execute_claim(ptr %ctx, ptr %claim, ptr %registry_table)
  %next = add i64 %i, 1
  br label %loop

exit:
  ret void
}

define void @bcir.example.ux_vec_add(ptr %ctx, ptr %a, ptr %b, ptr %c) {
entry:
  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 0)
  %va = call <8 x i32> @bcir.op.load.v8i32(ptr %a, i64 0)
  %vb = call <8 x i32> @bcir.op.load.v8i32(ptr %b, i64 0)
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 0)

  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 1)
  %vc = add <8 x i32> %va, %vb
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 1)

  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 2)
  call void @bcir.op.store.v8i32(ptr %c, i64 0, <8 x i32> %vc)
  call void @bcir.op.barrier(ptr %ctx, i32 2)
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 2)
  ret void
}

!bcir.opcodes = !{!100}
!100 = !{!"NOP", i32 0, !"LOAD", i32 1, !"STORE", i32 2, !"ATOMIC_ADD", i32 32, !"CMPXCHG", i32 35, !"BARRIER", i32 48, !"PHASE_ENTER", i32 49, !"PHASE_LEAVE", i32 50}
!bcir.lanes = !{!101}
!101 = !{!"U", i32 0, !"UX", i32 1, !"T", i32 2, !"GGG", i32 3, !"A", i32 4, !"H", i32 5}
!bcir.domains = !{!102}
!102 = !{!"RAM", i32 0, !"VRAM", i32 1, !"NVM", i32 2, !"MMIO", i32 3, !"CXL", i32 4, !"HBM", i32 5}
!bcir.claim.layout = !{!103}
!103 = !{!"BCIR_ClaimV2", !"size_bytes", i32 64, !"control", i32 0, i32 8, !"rd_rids", i32 8, i32 16, !"wr_rids", i32 24, i32 16, !"hazard_domain", i32 40, i32 8, !"immediates", i32 48, i32 16}

!bcir.schedule = !{!300}
!300 = !{!"sort_order", !"epoch,phase,lane,opcode,type,hazard_domain", !"batch_key_bits", i32 64}

!bcir.master.phase3 = !{!400}
!400 = !{"phase3_seed", "stream-pack and batch execution path enabled"}
