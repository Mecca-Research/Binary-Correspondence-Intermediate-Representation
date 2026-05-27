source_filename = "bcir_batch_executor.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }
%bcir.phase.range = type { i32, i32, i64, i64 }
%bcir.stream.pack = type { ptr, i64, ptr, i64, ptr, i64, ptr, i64 }

declare void @bcir.gem.execute_claim(ptr, ptr, ptr)
declare void @bcir.op.prefetch.linear(ptr, i64, i32, i32)
declare void @llvm.prefetch(ptr, i32, i32, i32)

define void @bcir.gem.execute_batch(ptr %ctx, ptr %claims, ptr %batch, ptr %registry_table) {
entry:
  call void @bcir.gem.execute_batch.generic(ptr %ctx, ptr %claims, ptr %batch, ptr %registry_table)
  ret void
}

define void @bcir.gem.execute_batch.generic(ptr %ctx, ptr %claims, ptr %batch, ptr %registry_table) {
entry:
  %first_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 6
  %count_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 7
  %first = load i64, ptr %first_p, align 8
  %count = load i64, ptr %count_p, align 8
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %done = icmp uge i64 %i, %count
  br i1 %done, label %exit, label %body

body:
  %idx = add i64 %first, %i
  %claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %idx
  call void @bcir.gem.execute_claim(ptr %ctx, ptr %claim, ptr %registry_table)
  %next = add i64 %i, 1
  %is_last = icmp uge i64 %next, %count
  br i1 %is_last, label %cont, label %do_prefetch

do_prefetch:
  %next_abs = add i64 %first, %next
  %next_claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %next_abs
  call void @llvm.prefetch(ptr %next_claim, i32 0, i32 3, i32 1)
  br label %cont

cont:
  br label %loop

exit:
  ret void
}

define void @bcir.gem.execute_stream_pack(ptr %ctx, ptr %pack, ptr %registry_table) {
entry:
  %claims_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 0
  %batches_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 2
  %phase_ranges_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 4
  %phase_count_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 5
  %claims = load ptr, ptr %claims_p, align 8
  %batches = load ptr, ptr %batches_p, align 8
  %phase_ranges = load ptr, ptr %phase_ranges_p, align 8
  %phase_count = load i64, ptr %phase_count_p, align 8
  br label %phase_loop

phase_loop:
  %pi = phi i64 [0, %entry], [%pi_next, %phase_done]
  %phase_done_cond = icmp uge i64 %pi, %phase_count
  br i1 %phase_done_cond, label %exit, label %phase_body

phase_body:
  %range = getelementptr inbounds %bcir.phase.range, ptr %phase_ranges, i64 %pi
  %b0_p = getelementptr inbounds %bcir.phase.range, ptr %range, i32 0, i32 2
  %b1_p = getelementptr inbounds %bcir.phase.range, ptr %range, i32 0, i32 3
  %b0 = load i64, ptr %b0_p, align 8
  %b1 = load i64, ptr %b1_p, align 8
  br label %batch_loop

batch_loop:
  %bi = phi i64 [%b0, %phase_body], [%bi_next, %batch_body]
  %batch_done = icmp uge i64 %bi, %b1
  br i1 %batch_done, label %phase_done, label %batch_body

batch_body:
  %batch = getelementptr inbounds %bcir.batch, ptr %batches, i64 %bi
  call void @bcir.gem.execute_batch(ptr %ctx, ptr %claims, ptr %batch, ptr %registry_table)
  %bi_next = add i64 %bi, 1
  br label %batch_loop

phase_done:
  %pi_next = add i64 %pi, 1
  br label %phase_loop

exit:
  ret void
}
