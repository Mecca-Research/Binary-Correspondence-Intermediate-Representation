// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.atomic_rmw / bcir.atomic_cas -> llvm.inttoptr + llvm.atomicrmw / llvm.cmpxchg (§5.14 Phase 2:
// the first-class atomic RMW/CAS ops, lifted from the cfront `c.atomic.*` / `c.c11atom.*` /
// `c.cmpxchg.*` opcode-string claims so the ordering law sees them structurally). The optional
// #bcir.mem_ordering maps to the LLVM ordering: ABSENT = seq_cst (the C11/`__sync` default, the
// bcir.barrier precedent); a CAS derives its failure ordering per the LLVM strongest-failure rule
// (release -> monotonic, acq_rel -> acquire, else the success ordering).

// (1) fetch-add, default ordering: seq_cst.
// CHECK-LABEL: func.func @rmw_add
func.func @rmw_add(%addr: i64, %v: i32) -> i32 {
  // CHECK: %[[P:.*]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  // CHECK: %[[O:.*]] = llvm.atomicrmw add %[[P]], %arg1 seq_cst : !llvm.ptr, i32
  %old = bcir.atomic_rmw "add" %v, %addr : i32, i64 -> i32
  // CHECK: return %[[O]] : i32
  return %old : i32
}

// (2) exchange with an explicit acquire ordering (the #bcir.mem_ordering attr maps through).
// CHECK-LABEL: func.func @rmw_xchg_acquire
func.func @rmw_xchg_acquire(%addr: i64, %v: i64) -> i64 {
  // CHECK: llvm.atomicrmw xchg %{{.*}}, %arg1 acquire : !llvm.ptr, i64
  %old = bcir.atomic_rmw "exchange" %v, %addr {ordering = #bcir.mem_ordering<acquire>} : i64, i64 -> i64
  return %old : i64
}

// (3) strong CAS, default ordering: seq_cst/seq_cst; the result is the OLD value (extractvalue [0],
// the `c.cmpxchg.val` semantics).
// CHECK-LABEL: func.func @cas
func.func @cas(%addr: i64, %e: i32, %d: i32) -> i32 {
  // CHECK: %[[P:.*]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  // CHECK: %[[C:.*]] = llvm.cmpxchg %[[P]], %arg1, %arg2 seq_cst seq_cst : !llvm.ptr, i32
  // CHECK: %[[V:.*]] = llvm.extractvalue %[[C]][0] : !llvm.struct<(i32, i1)>
  %old = bcir.atomic_cas %e, %d, %addr : i32, i32, i64 -> i32
  // CHECK: return %[[V]] : i32
  return %old : i32
}

// (4) weak CAS with a release success ordering -- the failure ordering DERIVES to monotonic (never a
// release on the failure path), and the `weak` flag rides through.
// CHECK-LABEL: func.func @cas_weak_release
func.func @cas_weak_release(%addr: i64, %e: i32, %d: i32) -> i32 {
  // CHECK: llvm.cmpxchg weak %{{.*}}, %arg1, %arg2 release monotonic : !llvm.ptr, i32
  %old = bcir.atomic_cas weak %e, %d, %addr {ordering = #bcir.mem_ordering<release>} : i32, i32, i64 -> i32
  return %old : i32
}
