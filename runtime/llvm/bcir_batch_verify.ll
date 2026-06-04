source_filename = "bcir_batch_verify.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }
%bcir.phase.range = type { i32, i32, i64, i64 }
%bcir.stream.pack = type { ptr, i64, ptr, i64, ptr, i64, ptr, i64 }

declare i32 @bcir.claim.phase(ptr)
declare i8 @bcir.claim.opcode(ptr)
declare i32 @bcir.classify.memory_lane(ptr)

define i1 @bcir.verify.batch(ptr %claims, i64 %claim_count, ptr %batch) {
entry:
  %phase_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 1
  %lane_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 2
  %opcode_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 3
  %first_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 6
  %count_p = getelementptr inbounds %bcir.batch, ptr %batch, i32 0, i32 7
  %phase = load i32, ptr %phase_p, align 4
  %lane = load i32, ptr %lane_p, align 4
  %opcode = load i32, ptr %opcode_p, align 4
  %first = load i64, ptr %first_p, align 8
  %count = load i64, ptr %count_p, align 8
  %nonzero = icmp ne i64 %count, 0
  %end = add i64 %first, %count
  %in_bounds = icmp ule i64 %end, %claim_count
  %basic_ok = and i1 %nonzero, %in_bounds
  br i1 %basic_ok, label %loop, label %fail

loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %all_ok = phi i1 [true, %entry], [%acc, %body]
  %done = icmp uge i64 %i, %count
  br i1 %done, label %exit, label %body

body:
  %idx = add i64 %first, %i
  %claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %idx
  %cphase = call i32 @bcir.claim.phase(ptr %claim)
  %cop8 = call i8 @bcir.claim.opcode(ptr %claim)
  %cop = zext i8 %cop8 to i32
  %clane = call i32 @bcir.classify.memory_lane(ptr %claim)

  %phase_ok = icmp eq i32 %cphase, %phase
  %opcode_ok = icmp eq i32 %cop, %opcode
  %lane_ok = icmp eq i32 %clane, %lane

  %is_a_lane = icmp eq i32 %lane, 4
  %is_h_lane = icmp eq i32 %lane, 5
  %is_atomic_lo = icmp uge i32 %cop, 32
  %is_atomic_hi = icmp ule i32 %cop, 35
  %is_atomic = and i1 %is_atomic_lo, %is_atomic_hi
  %not_a_lane = xor i1 %is_a_lane, true
  %a_ok = or i1 %not_a_lane, %is_atomic
  %is_hazard_lo = icmp uge i32 %cop, 48
  %is_hazard_hi = icmp ule i32 %cop, 51
  %is_hazard_like = and i1 %is_hazard_lo, %is_hazard_hi
  %not_h_lane = xor i1 %is_h_lane, true
  %h_ok = or i1 %not_h_lane, %is_hazard_like

  %ok0 = and i1 %phase_ok, %opcode_ok
  %ok1 = and i1 %ok0, %lane_ok
  %ok2 = and i1 %ok1, %a_ok
  %ok = and i1 %ok2, %h_ok
  %acc = and i1 %all_ok, %ok
  %next = add i64 %i, 1
  br label %loop

exit:
  ret i1 %all_ok

fail:
  ret i1 false
}

define i1 @bcir.verify.stream_pack(ptr %pack) {
entry:
  %claims_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 0
  %claim_count_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 1
  %batches_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 2
  %batch_count_p = getelementptr inbounds %bcir.stream.pack, ptr %pack, i32 0, i32 3
  %claims = load ptr, ptr %claims_p, align 8
  %claim_count = load i64, ptr %claim_count_p, align 8
  %batches = load ptr, ptr %batches_p, align 8
  %batch_count = load i64, ptr %batch_count_p, align 8
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %all_ok = phi i1 [true, %entry], [%acc, %body]
  %done = icmp uge i64 %i, %batch_count
  br i1 %done, label %exit, label %body

body:
  %batch = getelementptr inbounds %bcir.batch, ptr %batches, i64 %i
  %ok = call i1 @bcir.verify.batch(ptr %claims, i64 %claim_count, ptr %batch)
  %acc = and i1 %all_ok, %ok
  %next = add i64 %i, 1
  br label %loop

exit:
  ret i1 %all_ok
}
