source_filename = "bcir_claim_verify.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.res = type { i32, i32, ptr, i64, i64, i64, i64 }
%bcir.exe = type { i32, i32, i64, i64, i64, i64 }
%bcir.wl = type { i32, i32, i32, ptr, i64, i64, i64, i64 }

declare i8 @bcir.claim.opcode(ptr)
declare i8 @bcir.claim.lane(ptr)
declare i32 @bcir.claim.rd(ptr, i32)
declare i32 @bcir.claim.wr(ptr, i32)
declare i64 @bcir.claim.imm(ptr, i32)
declare void @llvm.trap() cold noreturn

define i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %rid) {
entry:
  %idx = and i32 %rid, 268435455
  %idx64 = zext i32 %idx to i64
  %ok = icmp ult i64 %idx64, %res_count
  ret i1 %ok
}

define i1 @bcir.verify.bounds(ptr %claim, ptr %res) {
entry:
  %off = call i64 @bcir.claim.imm(ptr %claim, i32 0)
  %size_p = getelementptr inbounds %bcir.res, ptr %res, i32 0, i32 3
  %size = load i64, ptr %size_p, align 8
  %width = add i64 %off, 4
  %ok = icmp ule i64 %width, %size
  ret i1 %ok
}

define i1 @bcir.verify.atomic_contract(ptr %claim, ptr %res) {
entry:
  %op8 = call i8 @bcir.claim.opcode(ptr %claim)
  %op = zext i8 %op8 to i32
  %lane8 = call i8 @bcir.claim.lane(ptr %claim)
  %lane = zext i8 %lane8 to i32
  %is_atomic = icmp uge i32 %op, 32
  %is_a_lane = icmp eq i32 %lane, 4
  %not_atomic = xor i1 %is_atomic, true
  %ok = or i1 %not_atomic, %is_a_lane
  ret i1 %ok
}

define i1 @bcir.verify.generation(ptr %exe, ptr %wl) {
entry:
  %exe_map_p = getelementptr inbounds %bcir.exe, ptr %exe, i32 0, i32 4
  %exe_data_p = getelementptr inbounds %bcir.exe, ptr %exe, i32 0, i32 5
  %wl_map_p = getelementptr inbounds %bcir.wl, ptr %wl, i32 0, i32 6
  %wl_data_p = getelementptr inbounds %bcir.wl, ptr %wl, i32 0, i32 7
  %exe_map = load i64, ptr %exe_map_p, align 8
  %exe_data = load i64, ptr %exe_data_p, align 8
  %wl_map = load i64, ptr %wl_map_p, align 8
  %wl_data = load i64, ptr %wl_data_p, align 8
  %map_ok = icmp eq i64 %exe_map, %wl_map
  %data_ok = icmp eq i64 %exe_data, %wl_data
  %ok = and i1 %map_ok, %data_ok
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
  %rid = call i32 @bcir.claim.rd(ptr %claim, i32 0)
  %rid_ok = call i1 @bcir.verify.rid(ptr %res_table, i64 %res_count, i32 %rid)
  %idx = and i32 %rid, 268435455
  %res_p = getelementptr %bcir.res, ptr %res_table, i32 %idx
  %bounds_ok = call i1 @bcir.verify.bounds(ptr %claim, ptr %res_p)
  %atomic_ok = call i1 @bcir.verify.atomic_contract(ptr %claim, ptr %res_p)
  %ok0 = and i1 %op_ok, %lane_ok
  %ok1 = and i1 %ok0, %rid_ok
  %ok2 = and i1 %ok1, %bounds_ok
  %ok = and i1 %ok2, %atomic_ok
  br i1 %ok, label %pass, label %fail
fail:
  call void @llvm.trap()
  ret i1 false
pass:
  ret i1 true
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
