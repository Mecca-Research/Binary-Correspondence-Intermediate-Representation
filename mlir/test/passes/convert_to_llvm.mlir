// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// BCIR -> LLVM dialect (Phase 6): bcir.compute/barrier lower to LLVM ops via the
// TypeConverter + ConversionPatterns. bcir-opt now *lowers*, not just parses.

// CHECK-LABEL: func.func @kernel
func.func @kernel(%a: f32, %b: f32) -> f32 {
  // CHECK: %[[S:.*]] = llvm.fadd %arg0, %arg1 : f32
  %c = bcir.compute "fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: llvm.fence seq_cst
  bcir.barrier 4
  // CHECK: return %[[S]] : f32
  return %c : f32
}
