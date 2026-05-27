source_filename = "bcir_claim_verify.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.res = type { i32, i32, ptr, i64, i64, i64, i64 }

declare i8 @bcir.claim.opcode(ptr)
declare i8 @bcir.claim.lane(ptr)
declare i32 @bcir.claim.rd(ptr, i32)
declare i32 @bcir.claim.wr(ptr, i32)
declare i64 @bcir.claim.imm(ptr, i32)
declare i64 @bcir.claim.hazard_domain(ptr)
declare void @llvm.trap() cold noreturn

define i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %rid) {
entry:
  %idx = and i32 %rid, 268435455
  %idx64 = zext i32 %idx to i64
  %ok = icmp ult i64 %idx64, %res_count
  ret i1 %ok
}

define i1 @bcir.verify.bounds(ptr %claim, ptr %res_table, i64 %res_count) {
entry:
  %rid = call i32 @bcir.claim.rd(ptr %claim, i32 0)
  %rid_ok = call i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %rid)
  br i1 %rid_ok, label %check, label %fail
check:
  %idx = and i32 %rid, 268435455
  %res = getelementptr inbounds %bcir.res, ptr %res_table, i32 %idx
  %off = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  %size_p = getelementptr inbounds %bcir.res, ptr %res, i32 0, i32 3
  %size = load i64, ptr %size_p, align 8
  %width = add i64 %off, 4
  %ok = icmp ule i64 %width, %size
  ret i1 %ok
fail:
  ret i1 false
}

define i1 @bcir.verify.atomic_contract(ptr %claim, ptr %res_table, i64 %res_count) {
entry:
  %op8 = call i8 @bcir.claim.opcode(ptr %claim)
  %op = zext i8 %op8 to i32
  %lane8 = call i8 @bcir.claim.lane(ptr %claim)
  %lane = zext i8 %lane8 to i32
  %h = call i64 @bcir.claim.hazard_domain(ptr %claim)
  %mode = and i64 %h, 15
  %atomic_mode = icmp ne i64 %mode, 0
  %is_atomic_lo = icmp uge i32 %op, 32
  %is_atomic_hi = icmp ule i32 %op, 35
  %is_atomic = and i1 %is_atomic_lo, %is_atomic_hi
  %is_a_lane = icmp eq i32 %lane, 4
  %lane_or_hazard = or i1 %is_a_lane, %atomic_mode
  %not_atomic = xor i1 %is_atomic, true
  %ok = or i1 %not_atomic, %lane_or_hazard
  ret i1 %ok
}

define i1 @bcir.verify.claim(ptr %claim, ptr %res_table, i64 %res_count) {
entry:
  %op8 = call i8 @bcir.claim.opcode(ptr %claim)
  %op = zext i8 %op8 to i32
  %lane8 = call i8 @bcir.claim.lane(ptr %claim)
  %lane = zext i8 %lane8 to i32
  %op_ok = icmp ule i32 %op, 96
  %lane_ok = icmp ule i32 %lane, 5

  %rd = call i32 @bcir.claim.rd(ptr %claim, i32 0)
  %wr = call i32 @bcir.claim.wr(ptr %claim, i32 0)
  %rd_ok = call i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %rd)
  %wr_ok = call i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %wr)

  %is_load = icmp eq i32 %op, 1
  %is_store = icmp eq i32 %op, 2
  %is_atomic_lo = icmp uge i32 %op, 32
  %is_atomic_hi = icmp ule i32 %op, 35
  %is_atomic = and i1 %is_atomic_lo, %is_atomic_hi

  %load_ok = or i1 (xor i1 %is_load, true), %rd_ok
  %store_ok = or i1 (xor i1 %is_store, true), %wr_ok
  %atomic_rid_ok = or i1 (xor i1 %is_atomic, true), %wr_ok

  %bounds_ok = call i1 @bcir.verify.bounds(ptr %claim, ptr %res_table, i64 %res_count)
  %atomic_ok = call i1 @bcir.verify.atomic_contract(ptr %claim, ptr %res_table, i64 %res_count)

  %ok0 = and i1 %op_ok, %lane_ok
  %ok1 = and i1 %ok0, %load_ok
  %ok2 = and i1 %ok1, %store_ok
  %ok3 = and i1 %ok2, %atomic_rid_ok
  %ok4 = and i1 %ok3, %bounds_ok
  %ok = and i1 %ok4, %atomic_ok
  ret i1 %ok
}

define i1 @bcir.verify.worklist(ptr %claims, i64 %count, ptr %res_table, i64 %res_count) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %body]
  %all_ok = phi i1 [true, %entry], [%acc, %body]
  %done = icmp uge i64 %i, %count
  br i1 %done, label %exit, label %body
body:
  %claim = getelementptr inbounds %bcir.claim, ptr %claims, i64 %i
  %ok = call i1 @bcir.verify.claim(ptr %claim, ptr %res_table, i64 %res_count)
  %acc = and i1 %all_ok, %ok
  %next = add i64 %i, 1
  br label %loop
exit:
  ret i1 %all_ok
}
