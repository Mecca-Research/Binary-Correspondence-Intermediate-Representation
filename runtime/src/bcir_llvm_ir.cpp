#include "bcir/bcir_llvm_ir.hpp"

namespace bcir {

std::vector<std::string> bcir_mlir_build_tasks() {
  return {
      "Stage 0: Build BCIR LLVM ABI substrate module before expanding core model",
      "Stage 1: Define MLIR BCIR dialect ops/types/attrs for surface->core",
      "Stage 2: Implement BCIR core verifier passes (lane/epoch/phase/hazard)",
      "Stage 3: Lower BCIR core to BCIR ROP schedule stream",
      "Stage 4: Convert BCIR ROP to MLIR LLVM dialect",
      "Stage 5: Emit legal LLVM IR only (load/store/atomicrmw/cmpxchg/fence/call)",
      "Stage 6: Add GEM schedule integration and deterministic execution order",
      "Stage 7: Add roundtrip and ABI-conformance tests",
  };
}

std::string bcir_reference_llvm_abi_module() {
  return R"LLVM(; BCIR LLVM ABI substrate (legal LLVM IR only)
source_filename = "bcir_backend_abi.ll"

declare void @llvm.trap() cold noreturn

%bcir.execctx = type { ptr, i32, i32, i64, i64, ptr }

declare void @bcir.rt.phase.enter(ptr, i32)
declare void @bcir.rt.phase.leave(ptr, i32)
declare void @bcir.rt.barrier(ptr, i32)
declare void @bcir.rt.prov.note(ptr, i64, i64, i64, i32, i32, i32, i32)

declare i32 @bcir.rt.load.i32(ptr, i64)
declare void @bcir.rt.store.i32(ptr, i64, i32)

declare i32 @bcir.rt.atomic.add.i32(ptr, i64, i32)
declare i32 @bcir.rt.atomic.sub.i32(ptr, i64, i32)
declare i32 @bcir.rt.atomic.xor.i32(ptr, i64, i32)

define i32 @bcir.example.atomic_add(ptr %ctx, ptr %base, i32 %delta) {
entry:
  call void @bcir.rt.phase.enter(ptr %ctx, i32 1)
  %old = atomicrmw add ptr %base, i32 %delta seq_cst, align 4
  call void @bcir.rt.barrier(ptr %ctx, i32 4)
  ret i32 %old
}

define i32 @bcir.example.cas(ptr %base, i32 %expected, i32 %desired) {
entry:
  %pair = cmpxchg ptr %base, i32 %expected, i32 %desired seq_cst monotonic, align 4
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}
)LLVM";
}

}  // namespace bcir
