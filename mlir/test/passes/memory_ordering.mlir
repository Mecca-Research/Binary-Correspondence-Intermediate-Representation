// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// Phase 8 memory model: a BCIR barrier's `ordering` maps to the LLVM fence
// ordering (default seq_cst).

// CHECK-LABEL: func.func @fences
func.func @fences() {
  // CHECK: llvm.fence acquire
  bcir.barrier 0 {ordering = #bcir.mem_ordering<acquire>}
  // CHECK: llvm.fence release
  bcir.barrier 0 {ordering = #bcir.mem_ordering<release>}
  // CHECK: llvm.fence acq_rel
  bcir.barrier 0 {ordering = #bcir.mem_ordering<acq_rel>}
  // CHECK: llvm.fence seq_cst
  bcir.barrier 0
  return
}
