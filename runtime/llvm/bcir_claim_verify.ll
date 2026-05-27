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
  %op8 = call i8 @bcir.claim.opcode(ptr %claim)
  %op = zext i8 %op8 to i32
  %rd = call i32 @bcir.claim.rd(ptr %claim, i32 0)
  %wr = call i32 @bcir.claim.wr(ptr %claim, i32 0)
  %is_store = icmp eq i32 %op, 2
  %is_atomic_lo = icmp uge i32 %op, 32
  %is_atomic_hi = icmp ule i32 %op, 35
  %is_atomic = and i1 %is_atomic_lo, %is_atomic_hi
  %write_like = or i1 %is_store, %is_atomic
  %rid = select i1 %write_like, i32 %wr, i32 %rd

  %rid_ok = call i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %rid)
  br i1 %rid_ok, label %check, label %fail
check:
  %idx = and i32 %rid, 268435455
  %res = getelementptr inbounds %bcir.res, ptr %res_table, i32 %idx
  %off = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  %off_nonneg = icmp sge i64 %off, 0
  %size_p = getelementptr inbounds %bcir.res, ptr %res, i32 0, i32 3
  %size = load i64, ptr %size_p, align 8
  %width = add i64 %off, 4
  %ok_w = icmp ule i64 %width, %size
  %ok = and i1 %off_nonneg, %ok_w
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
  %is_a_lane_non_atomic = and i1 %not_atomic, %is_a_lane
  %atomic_lane_ok = or i1 %not_atomic, %lane_or_hazard
  %not_a_lane_non_atomic = xor i1 %is_a_lane_non_atomic, true
  %ok = and i1 %atomic_lane_ok, %not_a_lane_non_atomic
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

  %not_load = xor i1 %is_load, true
  %load_ok = or i1 %not_load, %rd_ok
  %not_store = xor i1 %is_store, true
  %store_ok = or i1 %not_store, %wr_ok
  %not_atomic = xor i1 %is_atomic, true
  %atomic_rid_ok = or i1 %not_atomic, %wr_ok

  %needs_bounds = or i1 %is_load, (or i1 %is_store, %is_atomic)
  %bounds_eval = call i1 @bcir.verify.bounds(ptr %claim, ptr %res_table, i64 %res_count)
  %not_needs_bounds = xor i1 %needs_bounds, true
  %bounds_ok = or i1 %not_needs_bounds, %bounds_eval
  %atomic_eval = call i1 @bcir.verify.atomic_contract(ptr %claim, ptr %res_table, i64 %res_count)
  %atomic_ok = or i1 %not_atomic, %atomic_eval

  ; MMIO domain must be volatile or barriered (bit 0 or bit 1 in hazard domain)
  %h = call i64 @bcir.claim.hazard_domain(ptr %claim)
  %rid_sel = select i1 %is_load, i32 %rd, i32 %wr
  %rid_idx = and i32 %rid_sel, 268435455
  %rid_idx64 = zext i32 %rid_idx to i64
  %rid_in_range = icmp ult i64 %rid_idx64, %res_count
  br i1 %rid_in_range, label %mmio_check, label %mmio_skip

mmio_check:
  %res = getelementptr inbounds %bcir.res, ptr %res_table, i32 %rid_idx
  %domain_p = getelementptr inbounds %bcir.res, ptr %res, i32 0, i32 1
  %domain = load i32, ptr %domain_p, align 4
  %is_mmio = icmp eq i32 %domain, 2
  %vol_or_bar = and i64 %h, 3
  %has_vol_or_bar = icmp ne i64 %vol_or_bar, 0
  %not_mmio = xor i1 %is_mmio, true
  %mmio_ok = or i1 %not_mmio, %has_vol_or_bar
  br label %mmio_merge

mmio_skip:
  br label %mmio_merge

mmio_merge:
  %mmio_gate = phi i1 [ %mmio_ok, %mmio_check ], [ true, %mmio_skip ]

  %ok0 = and i1 %op_ok, %lane_ok
  %ok1 = and i1 %ok0, %load_ok
  %ok2 = and i1 %ok1, %store_ok
  %ok3 = and i1 %ok2, %atomic_rid_ok
  %ok4 = and i1 %ok3, %bounds_ok
  %ok5 = and i1 %ok4, %atomic_ok
  %ok = and i1 %ok5, %mmio_gate
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
