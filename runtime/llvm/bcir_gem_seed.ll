source_filename = "bcir_gem_seed.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

declare i8 @bcir.claim.opcode(ptr)
declare i8 @bcir.claim.lane(ptr)
declare i32 @bcir.claim.phase(ptr)
declare i32 @bcir.claim.epoch(ptr)
declare i32 @bcir.claim.rd(ptr, i32)
declare i32 @bcir.claim.wr(ptr, i32)
declare i64 @bcir.claim.imm(ptr, i32)

declare void @bcir.op.phase.enter(ptr, i32, i32)
declare void @bcir.op.phase.leave(ptr, i32, i32)
declare void @bcir.op.barrier(ptr, i32)
declare i32 @bcir.op.load.i32(ptr, i64)
declare void @bcir.op.store.i32(ptr, i64, i32)
declare i32 @bcir.op.atomic.add.i32(ptr, i64, i32)
declare { i32, i1 } @bcir.op.cmpxchg.i32(ptr, i64, i32, i32)
declare i1 @bcir.verify.worklist(ptr, i64, ptr, i64)
declare void @bcir.gem.execute_worklist_phased(ptr, ptr, i64, ptr, i32)
declare i1 @bcir.verify.stream_pack(ptr)
declare void @bcir.gem.execute_stream_pack(ptr, ptr, ptr)
declare void @llvm.trap() cold noreturn

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
  %lv = call i32 @bcir.op.load.i32(ptr %lbase, i64 %loff)
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


define void @bcir.gem.verify_and_execute_worklist(ptr %ctx, ptr %claims, i64 %count, ptr %res_table, i64 %res_count, ptr %registry_table, i32 %max_phase) {
entry:
  %ok = call i1 @bcir.verify.worklist(ptr %claims, i64 %count, ptr %res_table, i64 %res_count)
  br i1 %ok, label %exec, label %bad
exec:
  call void @bcir.gem.execute_worklist_phased(ptr %ctx, ptr %claims, i64 %count, ptr %registry_table, i32 %max_phase)
  ret void
bad:
  call void @llvm.trap()
  unreachable
}


define void @bcir.gem.verify_and_execute_stream_pack(ptr %ctx, ptr %pack, ptr %registry_table) {
entry:
  %ok = call i1 @bcir.verify.stream_pack(ptr %pack)
  br i1 %ok, label %exec, label %bad
exec:
  call void @bcir.gem.execute_stream_pack(ptr %ctx, ptr %pack, ptr %registry_table)
  ret void
bad:
  call void @llvm.trap()
  unreachable
}
