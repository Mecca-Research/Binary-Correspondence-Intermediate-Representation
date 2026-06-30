// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.msr_read / bcir.msr_write -> llvm.inline_asm (boot/CPU-state asm edge, D1.4): an x86-64
// model-specific-register access. Like bcir.creg_* these have NO cfront dual-rail twin, so the template is
// authored directly in LLVM-IR inline-asm syntax. Unlike creg, the register is a RUNTIME index (ECX), and
// the 64-bit value is split EDX:EAX -- so a `rdmsr` is a MULTI-OUTPUT inline-asm (`={ax},={dx}` ->
// !llvm.struct<(i32, i32)>) whose two halves are REASSEMBLED into the clean i64 op result, and a `wrmsr`
// SPLITS the i64 value into its low/high halves before the call. has_side_effects + a ~{memory} clobber are
// always set (an MSR is live CPU state: IA32_EFER toggles long mode, IA32_APIC_BASE relocates the APIC).

// (1) read MSR[index] -> "rdmsr", index in ECX ({cx}), result halves in EAX:EDX, reassembled to i64.
// CHECK-LABEL: func.func @rd_msr
func.func @rd_msr(%idx: i32) -> i64 {
  // CHECK: %[[S:.*]] = llvm.inline_asm has_side_effects "rdmsr", "={ax},={dx},{cx},~{memory}" %arg0 : (i32) -> !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[LO:.*]] = llvm.extractvalue %[[S]][0] : !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[HI:.*]] = llvm.extractvalue %[[S]][1] : !llvm.struct<(i32, i32)>
  // CHECK-DAG: %[[LO64:.*]] = llvm.zext %[[LO]] : i32 to i64
  // CHECK-DAG: %[[HI64:.*]] = llvm.zext %[[HI]] : i32 to i64
  // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[SH:.*]] = llvm.shl %[[HI64]], %[[C32]] : i64
  // CHECK: %[[V:.*]] = llvm.or %[[SH]], %[[LO64]] : i64
  %v = bcir.msr_read %idx : i32 -> i64
  // CHECK: return %[[V]] : i64
  return %v : i64
}

// (2) write MSR[index] = value -> the i64 is split (trunc / lshr+trunc) and "wrmsr" consumes
// (index, low, high); no result.
// CHECK-LABEL: func.func @wr_msr
func.func @wr_msr(%idx: i32, %val: i64) {
  // CHECK: %[[LO:.*]] = llvm.trunc %arg1 : i64 to i32
  // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[SH:.*]] = llvm.lshr %arg1, %[[C32]] : i64
  // CHECK: %[[HI:.*]] = llvm.trunc %[[SH]] : i64 to i32
  // CHECK: llvm.inline_asm has_side_effects "wrmsr", "{cx},{ax},{dx},~{memory}" %arg0, %[[LO]], %[[HI]] : (i32, i32, i32) -> ()
  // CHECK-NOT: = llvm.inline_asm
  bcir.msr_write %idx, %val : i32, i64
  return
}
