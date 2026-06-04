source_filename = "bcir_phase_worklist.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

declare i32 @bcir.claim.phase(ptr)
declare void @bcir.gem.execute_claim(ptr, ptr, ptr)
declare void @bcir.op.phase.enter(ptr, i32, i32)
declare void @bcir.op.phase.leave(ptr, i32, i32)
declare void @bcir.op.barrier(ptr, i32)

define void @bcir.gem.execute_worklist_phase(ptr %ctx, ptr %claims, i64 %count, ptr %registry_table, i32 %phase) {
entry:
  call void @bcir.op.phase.enter(ptr %ctx, i32 0, i32 %phase)
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %cont]
  %done = icmp uge i64 %i, %count
  br i1 %done, label %exit, label %body
body:
  %claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %i
  %cp = call i32 @bcir.claim.phase(ptr %claim)
  %match = icmp eq i32 %cp, %phase
  br i1 %match, label %run, label %cont
run:
  call void @bcir.gem.execute_claim(ptr %ctx, ptr %claim, ptr %registry_table)
  br label %cont
cont:
  %next = add i64 %i, 1
  br label %loop
exit:
  call void @bcir.op.barrier(ptr %ctx, i32 2)
  call void @bcir.op.phase.leave(ptr %ctx, i32 0, i32 %phase)
  ret void
}

define void @bcir.gem.execute_worklist_phased(ptr %ctx, ptr %claims, i64 %count, ptr %registry_table, i32 %max_phase) {
entry:
  br label %ploop
ploop:
  %p = phi i32 [0, %entry], [%pnext, %pbody]
  %pd = icmp ugt i32 %p, %max_phase
  br i1 %pd, label %pexit, label %pbody
pbody:
  call void @bcir.gem.execute_worklist_phase(ptr %ctx, ptr %claims, i64 %count, ptr %registry_table, i32 %p)
  %pnext = add i32 %p, 1
  br label %ploop
pexit:
  ret void
}
